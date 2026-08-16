"""DuckDB-backed authoritative validation, proof, and cache evidence store.

DQP-025 / DatabaseEvidenceStore@1
=================================

:class:`DatabaseEvidenceStore` is the durable authority for validation and
proof receipts, attestations, analysis/proof cache keys, invalidations,
single-flight leases, and use outcomes. File or export freshness never
determines authority. Large external bodies remain digest-bound and are
verified on use.

Cache hits never promote assurance: every lookup re-evaluates admission
against the stored key, TTL, invalidations, and poison checks. Stale or
poisoned hits fail closed. Single-flight coordination deduplicates expensive
work without treating shared outcomes as assurance upgrades.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
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
EVIDENCE_INVALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-invalidation@1"
)
EVIDENCE_FLIGHT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-flight@1"
)
EVIDENCE_USE_OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-use-outcome@1"
)
EVIDENCE_ATTESTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-attestation@1"
)
PROJECTION_REBUILD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-projection-rebuild@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:database-evidence-store"
AUTHORITY_CLASS: Final[str] = "database_authority"
# Caches never promote assurance above the admitted level.
CACHE_ASSURANCE_POLICY: Final[str] = "never_promote_assurance"

DEFAULT_TTL_SECONDS: Final[int] = 24 * 60 * 60
DEFAULT_FLIGHT_LEASE_SECONDS: Final[int] = 5 * 60
DEFAULT_FLIGHT_WAIT_SECONDS: Final[int] = 10 * 60
DEFAULT_OUTCOME_TTL_SECONDS: Final[int] = 10 * 60
MAX_BODY_BYTES: Final[int] = 262_144
MAX_RECURSION_DEPTH: Final[int] = 8
MAX_ID_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 8_192

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS evidence_store_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    key_id VARCHAR NOT NULL,
    key_json VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    assurance_level VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    metadata_json VARCHAR NOT NULL DEFAULT '{}',
    redacted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    poisoned BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS evidence_receipts_key_idx
    ON evidence_receipts(key_id, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS evidence_receipts_kind_idx
    ON evidence_receipts(kind, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS evidence_invalidations (
    invalidation_id VARCHAR PRIMARY KEY,
    receipt_id VARCHAR NOT NULL,
    key_id VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL,
    invalidated_at_ms BIGINT NOT NULL,
    invalidated_by VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS evidence_invalidations_receipt_idx
    ON evidence_invalidations(receipt_id, invalidated_at_ms);
CREATE INDEX IF NOT EXISTS evidence_invalidations_key_idx
    ON evidence_invalidations(key_id, invalidated_at_ms);

CREATE TABLE IF NOT EXISTS evidence_attestations (
    attestation_id VARCHAR PRIMARY KEY,
    receipt_id VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    backend VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    snapshot_id VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_attestations_receipt_idx
    ON evidence_attestations(receipt_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS evidence_flights (
    key_id VARCHAR PRIMARY KEY,
    owner_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_flight_outcomes (
    key_id VARCHAR PRIMARY KEY,
    fencing_token BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    outcome_json VARCHAR NOT NULL,
    outcome_digest VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_use_outcomes (
    use_id VARCHAR PRIMARY KEY,
    receipt_id VARCHAR NOT NULL DEFAULT '',
    key_id VARCHAR NOT NULL DEFAULT '',
    status VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    used_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS evidence_use_outcomes_key_idx
    ON evidence_use_outcomes(key_id, used_at_ms DESC);

CREATE TABLE IF NOT EXISTS evidence_projections (
    projection_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    row_count BIGINT NOT NULL DEFAULT 0,
    rebuilt_from VARCHAR NOT NULL DEFAULT 'admitted_evidence',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseEvidenceStoreError(RuntimeError):
    """Base error for database evidence store failures."""


class DatabaseEvidenceStoreNotOpenError(DatabaseEvidenceStoreError):
    """Operation requires an open evidence store."""


class DatabaseEvidenceStoreIntegrityError(DatabaseEvidenceStoreError, ValueError):
    """Identity, digest, or payload integrity failure."""


class DatabaseEvidenceStoreBoundsError(DatabaseEvidenceStoreError, ValueError):
    """A resource or payload bound was exceeded."""


class DatabaseEvidenceStoreConflictError(DatabaseEvidenceStoreError):
    """Duplicate identity with a conflicting payload."""


class DatabaseEvidenceStorePoisonedError(DatabaseEvidenceStoreError):
    """Poisoned or corrupted cache entry rejected fail-closed."""


class DatabaseEvidenceStoreStaleError(DatabaseEvidenceStoreError):
    """Stale key or expired receipt rejected fail-closed."""


class SingleFlightError(DatabaseEvidenceStoreError):
    """Base error for single-flight coordination."""


class SingleFlightTimeout(SingleFlightError, TimeoutError):
    """No leader outcome was observed before the caller's deadline."""


class SingleFlightExecutionError(SingleFlightError):
    """The single-flight owner failed while producing the shared result."""


class DuckDBUnavailableError(DatabaseEvidenceStoreError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class EvidenceKind(str, Enum):
    VALIDATION = "validation"
    PROOF = "proof"
    ATTESTATION = "attestation"
    ANALYSIS = "analysis"
    CACHE = "cache"
    DATASET = "dataset"

    @classmethod
    def coerce(cls, value: Any) -> "EvidenceKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases = {
            "validation": cls.VALIDATION,
            "proof": cls.PROOF,
            "attestation": cls.ATTESTATION,
            "analysis": cls.ANALYSIS,
            "cache": cls.CACHE,
            "dataset": cls.DATASET,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseEvidenceStoreIntegrityError(
                f"unsupported evidence kind: {value!r}"
            ) from exc


class EvidenceVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    REJECTED = "rejected"

    @classmethod
    def coerce(cls, value: Any) -> "EvidenceVerdict":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases = {
            "pass": cls.PASS,
            "passed": cls.PASS,
            "ok": cls.PASS,
            "success": cls.PASS,
            "fail": cls.FAIL,
            "failed": cls.FAIL,
            "failure": cls.FAIL,
            "inconclusive": cls.INCONCLUSIVE,
            "error": cls.ERROR,
            "rejected": cls.REJECTED,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseEvidenceStoreIntegrityError(
                f"unsupported evidence verdict: {value!r}"
            ) from exc


class AssuranceLevel(str, Enum):
    """Ordered assurance levels. Cache hits never promote above stored level."""

    NONE = "none"
    HEURISTIC = "heuristic"
    VALIDATED = "validated"
    SOLVER_CHECKED = "solver_checked"
    KERNEL_VERIFIED = "kernel_verified"
    ATTESTED = "attested"

    @classmethod
    def coerce(cls, value: Any) -> "AssuranceLevel":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases = {
            "none": cls.NONE,
            "heuristic": cls.HEURISTIC,
            "validated": cls.VALIDATED,
            "solver_checked": cls.SOLVER_CHECKED,
            "solver-checked": cls.SOLVER_CHECKED,
            "kernel_verified": cls.KERNEL_VERIFIED,
            "kernel-verified": cls.KERNEL_VERIFIED,
            "attested": cls.ATTESTED,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseEvidenceStoreIntegrityError(
                f"unsupported assurance level: {value!r}"
            ) from exc

    @property
    def rank(self) -> int:
        order = (
            AssuranceLevel.NONE,
            AssuranceLevel.HEURISTIC,
            AssuranceLevel.VALIDATED,
            AssuranceLevel.SOLVER_CHECKED,
            AssuranceLevel.KERNEL_VERIFIED,
            AssuranceLevel.ATTESTED,
        )
        return order.index(self)


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
    KEY_MISMATCH = "key_mismatch"
    INSUFFICIENT_ASSURANCE = "required_assurance_not_satisfied"
    ASSURANCE_PROMOTION_FORBIDDEN = "assurance_promotion_forbidden"
    INCONCLUSIVE = "inconclusive_result"
    EXPIRED = "expired_receipt"


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
        raise DatabaseEvidenceStoreIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseEvidenceStoreIntegrityError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_ID_BYTES and name.endswith(("_id", "key_id")):
        raise DatabaseEvidenceStoreBoundsError(
            f"{name} exceeds {MAX_ID_BYTES} bytes"
        )
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
    if not _DIGEST_RE.fullmatch(text):
        raise DatabaseEvidenceStoreIntegrityError(
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
        raise DatabaseEvidenceStoreBoundsError(
            f"{name} exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise DatabaseEvidenceStoreIntegrityError(
            f"{name} must project to an object"
        )
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise DatabaseEvidenceStoreBoundsError(
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
        raise DatabaseEvidenceStorePoisonedError(
            "stored JSON is corrupted"
        ) from exc
    if not isinstance(value, dict):
        raise DatabaseEvidenceStorePoisonedError(
            "stored JSON must be an object"
        )
    return value


def _json_safe(value: Any) -> Any:
    """Project arbitrary row material into canonical JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def assurance_satisfies(
    actual: AssuranceLevel | str,
    required: AssuranceLevel | str,
) -> bool:
    """Return whether ``actual`` meets ``required`` without promotion tricks."""

    actual_level = AssuranceLevel.coerce(actual)
    required_level = AssuranceLevel.coerce(required)
    return actual_level.rank >= required_level.rank


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceKey:
    """Semantic cache / evidence key binding all applicability dimensions."""

    key_id: str
    kind: str
    subject_id: str
    semantic_roots: Mapping[str, Any] = field(default_factory=dict)
    policy_id: str = ""
    schema_id: str = EVIDENCE_KEY_SCHEMA
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind.coerce(self.kind).value)
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, "subject_id")
        )
        roots = dict(self.semantic_roots or {})
        extra = dict(self.extra or {})
        object.__setattr__(self, "semantic_roots", MappingProxyType(roots))
        object.__setattr__(self, "extra", MappingProxyType(extra))
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=False)
        )
        if not self.key_id:
            object.__setattr__(
                self,
                "key_id",
                _identity(
                    "evidence-key",
                    {
                        "kind": self.kind,
                        "subject_id": self.subject_id,
                        "semantic_roots": roots,
                        "policy_id": self.policy_id,
                        "extra": extra,
                    },
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_id,
            "key_id": self.key_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "semantic_roots": dict(self.semantic_roots),
            "policy_id": self.policy_id,
            "extra": dict(self.extra),
        }

    @classmethod
    def create(
        cls,
        *,
        kind: EvidenceKind | str,
        subject_id: str,
        semantic_roots: Mapping[str, Any] | None = None,
        policy_id: str = "",
        extra: Mapping[str, Any] | None = None,
        key_id: str = "",
    ) -> "EvidenceKey":
        roots = dict(semantic_roots or {})
        extras = dict(extra or {})
        selected_kind = EvidenceKind.coerce(kind).value
        selected_subject = _text(subject_id, "subject_id")
        selected_policy = _text(policy_id, "policy_id", required=False)
        computed = key_id or _identity(
            "evidence-key",
            {
                "kind": selected_kind,
                "subject_id": selected_subject,
                "semantic_roots": roots,
                "policy_id": selected_policy,
                "extra": extras,
            },
        )
        return cls(
            key_id=computed,
            kind=selected_kind,
            subject_id=selected_subject,
            semantic_roots=roots,
            policy_id=selected_policy,
            extra=extras,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceKey":
        return cls(
            key_id=str(value.get("key_id") or ""),
            kind=str(value.get("kind") or ""),
            subject_id=str(value.get("subject_id") or ""),
            semantic_roots=dict(value.get("semantic_roots") or {}),
            policy_id=str(value.get("policy_id") or ""),
            schema_id=str(value.get("schema") or EVIDENCE_KEY_SCHEMA),
            extra=dict(value.get("extra") or {}),
        )


@dataclass(frozen=True)
class EvidenceReceipt:
    """Admitted evidence receipt. Cache hits never promote its assurance."""

    receipt_id: str
    key: EvidenceKey
    kind: str
    verdict: str
    assurance_level: str
    content_digest: str
    body: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redacted: bool = False
    created_at_ms: int = 0
    expires_at_ms: int = 0
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    poisoned: bool = False
    schema: str = EVIDENCE_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "key": self.key.to_dict(),
            "kind": self.kind,
            "verdict": self.verdict,
            "assurance_level": self.assurance_level,
            "content_digest": self.content_digest,
            "body": dict(self.body),
            "metadata": dict(self.metadata),
            "redacted": self.redacted,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "snapshot_id": self.snapshot_id,
            "poisoned": self.poisoned,
            "authority": AUTHORITY_CLASS,
            "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
        }


@dataclass(frozen=True)
class EvidenceLookupResult:
    status: LookupStatus
    key: EvidenceKey
    receipt: EvidenceReceipt | None = None
    reason: RejectionReason | None = None
    use_id: str = ""

    @property
    def hit(self) -> bool:
        return self.status is LookupStatus.HIT and self.receipt is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "key": self.key.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "reason": None if self.reason is None else self.reason.value,
            "use_id": self.use_id,
            "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
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
class ProjectionRebuildReceipt:
    projection_id: str
    snapshot_id: str
    kind: str
    digest: str
    row_count: int
    rebuilt_from: str = "admitted_evidence"
    created_at: str = ""
    schema: str = PROJECTION_REBUILD_SCHEMA

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


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseEvidenceStore:
    """DuckDB authority for evidence, cache keys, invalidations, and flights."""

    INTERFACE: Final[str] = DATABASE_EVIDENCE_STORE_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        auto_redact: bool = True,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseEvidenceStore; install the "
                "optional duckdb dependency"
            )
        if (
            isinstance(default_ttl_seconds, bool)
            or not isinstance(default_ttl_seconds, int)
            or default_ttl_seconds <= 0
        ):
            raise DatabaseEvidenceStoreBoundsError(
                "default_ttl_seconds must be a positive integer"
            )
        self._path = Path(database_path)
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        self._default_ttl_seconds = int(default_ttl_seconds)
        self._clock = clock
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
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def open(self) -> "DatabaseEvidenceStore":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_EVIDENCE_STORE_INTERFACE),
                ("schema", DATABASE_EVIDENCE_STORE_SCHEMA),
                ("snapshot_id", self._snapshot_id),
                ("authority", AUTHORITY_CLASS),
                ("cache_assurance_policy", CACHE_ASSURANCE_POLICY),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO evidence_store_metadata(key, value)
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

    # -- put / admit ---------------------------------------------------------

    def put(
        self,
        key: EvidenceKey | Mapping[str, Any],
        *,
        verdict: EvidenceVerdict | str,
        assurance_level: AssuranceLevel | str = AssuranceLevel.VALIDATED,
        body: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        content_digest: str | None = None,
        ttl_seconds: int | None = None,
        kind: EvidenceKind | str | None = None,
        receipt_id: str | None = None,
        redact: bool | None = None,
    ) -> EvidenceReceipt:
        """Persist one evidence receipt. Negative results are admitted too."""

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        do_redact = self._auto_redact if redact is None else bool(redact)
        selected_verdict = EvidenceVerdict.coerce(verdict)
        selected_assurance = AssuranceLevel.coerce(assurance_level)
        selected_kind = EvidenceKind.coerce(kind or cache_key.kind)
        payload = _bounded_mapping(body, redact=do_redact, name="body")
        meta = _bounded_mapping(metadata, redact=do_redact, name="metadata")
        ttl = (
            self._default_ttl_seconds
            if ttl_seconds is None
            else _positive_int(ttl_seconds, "ttl_seconds")
        )
        now = self._now_ms()
        expires = now + ttl * 1000
        if content_digest is not None:
            digest = _require_digest(content_digest)
        else:
            digest = _sha256_digest(
                _canonical_json(
                    {
                        "key": cache_key.to_dict(),
                        "verdict": selected_verdict.value,
                        "assurance_level": selected_assurance.value,
                        "body": payload,
                        "metadata": meta,
                    }
                ).encode("utf-8")
            )
        material = {
            "key_id": cache_key.key_id,
            "kind": selected_kind.value,
            "verdict": selected_verdict.value,
            "assurance_level": selected_assurance.value,
            "content_digest": digest,
            "body": payload,
            "metadata": meta,
            "created_at_ms": now,
            "snapshot_id": self._snapshot_id,
        }
        selected_id = _text(
            receipt_id or _identity("evidence-receipt", material),
            "receipt_id",
        )

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM evidence_receipts WHERE receipt_id = ?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                current = self._receipt_from_row(_row_mapping(existing))
                if current.content_digest != digest:
                    raise DatabaseEvidenceStoreConflictError(
                        "receipt_id already exists with a different payload"
                    )
                return current

            connection.execute(
                """
                INSERT INTO evidence_receipts(
                    receipt_id, key_id, key_json, kind, verdict,
                    assurance_level, content_digest, body_json, metadata_json,
                    redacted, created_at_ms, expires_at_ms, snapshot_id,
                    poisoned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_id,
                    cache_key.key_id,
                    _canonical_json(cache_key.to_dict()),
                    selected_kind.value,
                    selected_verdict.value,
                    selected_assurance.value,
                    digest,
                    _canonical_json(payload),
                    _canonical_json(meta),
                    do_redact,
                    now,
                    expires,
                    self._snapshot_id,
                    False,
                ],
            )
            self._commit_if_idle(connection)
            return EvidenceReceipt(
                receipt_id=selected_id,
                key=cache_key,
                kind=selected_kind.value,
                verdict=selected_verdict.value,
                assurance_level=selected_assurance.value,
                content_digest=digest,
                body=MappingProxyType(payload),
                metadata=MappingProxyType(meta),
                redacted=do_redact,
                created_at_ms=now,
                expires_at_ms=expires,
                snapshot_id=self._snapshot_id,
                poisoned=False,
            )

    store = put
    admit = put

    def put_attestation(
        self,
        receipt_id: str,
        *,
        content_digest: str,
        backend: str = "",
        body: Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Attach an attestation record to an admitted receipt."""

        selected_receipt = _text(receipt_id, "receipt_id")
        digest = _require_digest(content_digest)
        payload = _bounded_mapping(body, redact=self._auto_redact, name="body")
        ttl = (
            self._default_ttl_seconds
            if ttl_seconds is None
            else _positive_int(ttl_seconds, "ttl_seconds")
        )
        now = self._now_ms()
        attestation_id = _identity(
            "attestation",
            {
                "receipt_id": selected_receipt,
                "content_digest": digest,
                "backend": backend,
                "body": payload,
                "created_at_ms": now,
            },
        )
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT receipt_id FROM evidence_receipts WHERE receipt_id = ?",
                [selected_receipt],
            ).fetchone()
            if row is None:
                raise DatabaseEvidenceStoreIntegrityError(
                    "attestation requires an admitted receipt"
                )
            connection.execute(
                """
                INSERT INTO evidence_attestations(
                    attestation_id, receipt_id, content_digest, backend,
                    body_json, created_at_ms, expires_at_ms, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attestation_id,
                    selected_receipt,
                    digest,
                    _text(backend, "backend", required=False),
                    _canonical_json(payload),
                    now,
                    now + ttl * 1000,
                    self._snapshot_id,
                ],
            )
            self._commit_if_idle(connection)
            return {
                "schema": EVIDENCE_ATTESTATION_SCHEMA,
                "attestation_id": attestation_id,
                "receipt_id": selected_receipt,
                "content_digest": digest,
                "backend": _text(backend, "backend", required=False),
                "created_at_ms": now,
                "expires_at_ms": now + ttl * 1000,
                "snapshot_id": self._snapshot_id,
            }

    # -- lookup / admission re-evaluation ------------------------------------

    def lookup(
        self,
        key: EvidenceKey | Mapping[str, Any],
        *,
        required_assurance: AssuranceLevel | str = AssuranceLevel.VALIDATED,
        allow_inconclusive: bool = False,
        record_use: bool = True,
    ) -> EvidenceLookupResult:
        """Lookup with full admission re-evaluation. Never promotes assurance."""

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        required = AssuranceLevel.coerce(required_assurance)
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT r.*,
                    EXISTS(
                        SELECT 1 FROM evidence_invalidations i
                        WHERE i.receipt_id = r.receipt_id
                    ) AS invalidated
                FROM evidence_receipts r
                WHERE r.key_id = ?
                ORDER BY r.created_at_ms DESC
                LIMIT 1
                """,
                [cache_key.key_id],
            ).fetchone()
            if row is None:
                result = EvidenceLookupResult(
                    status=LookupStatus.MISS,
                    key=cache_key,
                    reason=RejectionReason.CACHE_MISS,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.CACHE_MISS.value,
                    )
                return result

            mapping = _row_mapping(row)
            try:
                receipt = self._receipt_from_row(mapping)
            except DatabaseEvidenceStorePoisonedError:
                self._mark_poisoned(connection, str(mapping.get("receipt_id")))
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    reason=RejectionReason.POISONED,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.POISONED.value,
                    )
                return result

            # Key binding must match exactly — stale semantic roots fail closed.
            if receipt.key.to_dict() != cache_key.to_dict():
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    receipt=receipt,
                    reason=RejectionReason.KEY_MISMATCH,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.KEY_MISMATCH.value,
                    )
                return result

            if bool(mapping.get("poisoned")) or receipt.poisoned:
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    receipt=receipt,
                    reason=RejectionReason.POISONED,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.POISONED.value,
                    )
                return result

            if bool(mapping.get("invalidated")):
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    receipt=receipt,
                    reason=RejectionReason.INVALIDATED,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.INVALIDATED.value,
                    )
                return result

            if receipt.expires_at_ms <= now:
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    receipt=receipt,
                    reason=RejectionReason.STALE,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.STALE.value,
                    )
                return result

            if (
                receipt.verdict == EvidenceVerdict.INCONCLUSIVE.value
                and not allow_inconclusive
            ):
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    receipt=receipt,
                    reason=RejectionReason.INCONCLUSIVE,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=RejectionReason.INCONCLUSIVE.value,
                    )
                return result

            stored_assurance = AssuranceLevel.coerce(receipt.assurance_level)
            if not assurance_satisfies(stored_assurance, required):
                # Explicitly refuse promotion: a weaker stored level cannot
                # satisfy a stronger requirement by cache hit alone.
                reason = (
                    RejectionReason.ASSURANCE_PROMOTION_FORBIDDEN
                    if stored_assurance.rank < required.rank
                    else RejectionReason.INSUFFICIENT_ASSURANCE
                )
                result = EvidenceLookupResult(
                    status=LookupStatus.REJECTED,
                    key=cache_key,
                    receipt=receipt,
                    reason=reason,
                )
                if record_use:
                    result = self._record_use(
                        connection,
                        result,
                        reason=reason.value,
                    )
                return result

            result = EvidenceLookupResult(
                status=LookupStatus.HIT,
                key=cache_key,
                receipt=receipt,
            )
            if record_use:
                result = self._record_use(
                    connection,
                    result,
                    reason="hit",
                )
            return result

    def get_receipt(self, receipt_id: str) -> EvidenceReceipt | None:
        selected = _text(receipt_id, "receipt_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM evidence_receipts WHERE receipt_id = ?",
                [selected],
            ).fetchone()
            if row is None:
                return None
            return self._receipt_from_row(_row_mapping(row))

    def _receipt_from_row(self, row: Mapping[str, Any]) -> EvidenceReceipt:
        try:
            key = EvidenceKey.from_dict(_load_json_object(row.get("key_json")))
            body = _load_json_object(row.get("body_json"))
            metadata = _load_json_object(row.get("metadata_json"))
        except DatabaseEvidenceStorePoisonedError:
            raise
        except Exception as exc:
            raise DatabaseEvidenceStorePoisonedError(
                "poisoned durable receipt envelope"
            ) from exc

        receipt = EvidenceReceipt(
            receipt_id=str(row["receipt_id"]),
            key=key,
            kind=str(row["kind"]),
            verdict=str(row["verdict"]),
            assurance_level=str(row["assurance_level"]),
            content_digest=str(row["content_digest"]),
            body=MappingProxyType(body),
            metadata=MappingProxyType(metadata),
            redacted=bool(row.get("redacted")),
            created_at_ms=int(row["created_at_ms"]),
            expires_at_ms=int(row["expires_at_ms"]),
            snapshot_id=str(row.get("snapshot_id") or self._snapshot_id),
            poisoned=bool(row.get("poisoned")),
        )
        # Envelope integrity: key_id column must match reconstructed key.
        if str(row["key_id"]) != receipt.key.key_id:
            raise DatabaseEvidenceStorePoisonedError(
                "receipt key_id envelope mismatch"
            )
        if not _DIGEST_RE.fullmatch(receipt.content_digest):
            raise DatabaseEvidenceStorePoisonedError(
                "receipt content_digest is malformed"
            )
        return receipt

    def _mark_poisoned(self, connection: Any, receipt_id: str | None) -> None:
        if not receipt_id:
            return
        try:
            connection.execute(
                "UPDATE evidence_receipts SET poisoned = TRUE "
                "WHERE receipt_id = ?",
                [receipt_id],
            )
            self._commit_if_idle(connection)
        except Exception:
            pass

    def _record_use(
        self,
        connection: Any,
        result: EvidenceLookupResult,
        *,
        reason: str,
    ) -> EvidenceLookupResult:
        now = self._now_ms()
        use_id = _identity(
            "use",
            {
                "key_id": result.key.key_id,
                "status": result.status.value,
                "reason": reason,
                "used_at_ms": now,
                "nonce": str(uuid.uuid4()),
            },
        )
        connection.execute(
            """
            INSERT INTO evidence_use_outcomes(
                use_id, receipt_id, key_id, status, reason, used_at_ms,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                use_id,
                "" if result.receipt is None else result.receipt.receipt_id,
                result.key.key_id,
                result.status.value,
                reason,
                now,
                _canonical_json(
                    {
                        "status": result.status.value,
                        "reason": reason,
                        "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
                    }
                ),
            ],
        )
        self._commit_if_idle(connection)
        return EvidenceLookupResult(
            status=result.status,
            key=result.key,
            receipt=result.receipt,
            reason=result.reason,
            use_id=use_id,
        )

    # -- invalidation --------------------------------------------------------

    def invalidate(
        self,
        *,
        receipt_id: str = "",
        key: EvidenceKey | Mapping[str, Any] | None = None,
        reason: str = "manual",
        invalidated_by: str = "",
    ) -> int:
        """Invalidate one receipt or all receipts for a key."""

        now = self._now_ms()
        reason_text = _text(reason, "reason")
        by = _text(invalidated_by, "invalidated_by", required=False)
        changed = 0
        with self._lock:
            connection = self._require()
            targets: list[tuple[str, str]] = []
            if receipt_id:
                selected = _text(receipt_id, "receipt_id")
                row = connection.execute(
                    "SELECT receipt_id, key_id FROM evidence_receipts "
                    "WHERE receipt_id = ?",
                    [selected],
                ).fetchone()
                if row is not None:
                    mapping = _row_mapping(row)
                    targets.append(
                        (str(mapping["receipt_id"]), str(mapping["key_id"]))
                    )
            if key is not None:
                cache_key = (
                    key
                    if isinstance(key, EvidenceKey)
                    else EvidenceKey.from_dict(key)
                )
                rows = connection.execute(
                    "SELECT receipt_id, key_id FROM evidence_receipts "
                    "WHERE key_id = ?",
                    [cache_key.key_id],
                ).fetchall()
                for row in rows:
                    mapping = _row_mapping(row)
                    targets.append(
                        (str(mapping["receipt_id"]), str(mapping["key_id"]))
                    )
            for selected_receipt, selected_key in targets:
                inv_id = _identity(
                    "invalidation",
                    {
                        "receipt_id": selected_receipt,
                        "reason": reason_text,
                        "invalidated_at_ms": now,
                        "invalidated_by": by,
                    },
                )
                existing = connection.execute(
                    "SELECT 1 AS n FROM evidence_invalidations "
                    "WHERE receipt_id = ? AND reason = ?",
                    [selected_receipt, reason_text],
                ).fetchone()
                if existing is not None:
                    continue
                connection.execute(
                    """
                    INSERT INTO evidence_invalidations(
                        invalidation_id, receipt_id, key_id, reason,
                        invalidated_at_ms, invalidated_by, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        inv_id,
                        selected_receipt,
                        selected_key,
                        reason_text,
                        now,
                        by,
                        _canonical_json(
                            {
                                "reason": reason_text,
                                "invalidated_by": by,
                            }
                        ),
                    ],
                )
                changed += 1
            self._commit_if_idle(connection)
        return changed

    def mark_poisoned(self, receipt_id: str) -> None:
        """Explicitly quarantine a poisoned receipt."""

        selected = _text(receipt_id, "receipt_id")
        with self._lock:
            connection = self._require()
            connection.execute(
                "UPDATE evidence_receipts SET poisoned = TRUE "
                "WHERE receipt_id = ?",
                [selected],
            )
            self._commit_if_idle(connection)
            self.invalidate(
                receipt_id=selected,
                reason="poisoned",
                invalidated_by="database-evidence-store",
            )

    # -- single flight -------------------------------------------------------

    def single_flight(
        self,
        key: EvidenceKey | Mapping[str, Any] | str,
        producer: Callable[[], T],
        *,
        lease_seconds: int = DEFAULT_FLIGHT_LEASE_SECONDS,
        wait_seconds: int = DEFAULT_FLIGHT_WAIT_SECONDS,
        outcome_ttl_seconds: int = DEFAULT_OUTCOME_TTL_SECONDS,
        poll_seconds: float = 0.05,
    ) -> SingleFlightResult:
        """Deduplicate expensive work. Shared outcomes are not assurance."""

        if isinstance(key, EvidenceKey):
            key_id = key.key_id
        elif isinstance(key, Mapping):
            key_id = EvidenceKey.from_dict(key).key_id
        else:
            key_id = _text(key, "key")

        lease = _positive_int(lease_seconds, "lease_seconds")
        wait = _positive_int(wait_seconds, "wait_seconds")
        outcome_ttl = _positive_int(outcome_ttl_seconds, "outcome_ttl_seconds")
        owner_id = f"owner:{uuid.uuid4().hex}"
        deadline = self._clock() + wait

        while True:
            now_ms = self._now_ms()
            with self._lock:
                connection = self._require()
                # Prefer a fresh published outcome.
                outcome_row = connection.execute(
                    """
                    SELECT * FROM evidence_flight_outcomes
                    WHERE key_id = ? AND expires_at_ms > ?
                    """,
                    [key_id, now_ms],
                ).fetchone()
                if outcome_row is not None:
                    mapping = _row_mapping(outcome_row)
                    status = str(mapping["status"])
                    payload = _load_json_object(mapping.get("outcome_json"))
                    if status == "ok":
                        return SingleFlightResult(
                            value=payload.get("value"),
                            owner=False,
                            fencing_token=int(mapping["fencing_token"]),
                        )
                    raise SingleFlightExecutionError(
                        str(payload.get("error") or "shared producer failed")
                    )

                # Expire stale leases.
                connection.execute(
                    "DELETE FROM evidence_flights WHERE expires_at_ms <= ?",
                    [now_ms],
                )
                existing = connection.execute(
                    "SELECT * FROM evidence_flights WHERE key_id = ?",
                    [key_id],
                ).fetchone()
                if existing is None:
                    fencing = now_ms
                    lease_id = f"lease:{uuid.uuid4().hex}"
                    connection.execute(
                        """
                        INSERT INTO evidence_flights(
                            key_id, owner_id, lease_id, fencing_token,
                            acquired_at_ms, expires_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            key_id,
                            owner_id,
                            lease_id,
                            fencing,
                            now_ms,
                            now_ms + lease * 1000,
                        ],
                    )
                    self._commit_if_idle(connection)
                    # Produce outside the DB lock section? Keep lock released.
                    break
            if self._clock() >= deadline:
                raise SingleFlightTimeout(
                    f"single-flight wait exceeded for key {key_id}"
                )
            time.sleep(max(0.01, float(poll_seconds)))

        # Owner path.
        try:
            value = producer()
        except Exception as exc:
            self._publish_flight_outcome(
                key_id=key_id,
                owner_id=owner_id,
                status="error",
                payload={"error": f"{type(exc).__name__}: producer failed"},
                outcome_ttl_seconds=outcome_ttl,
            )
            raise SingleFlightExecutionError(
                f"single-flight producer failed: {type(exc).__name__}"
            ) from exc

        fencing_token = self._publish_flight_outcome(
            key_id=key_id,
            owner_id=owner_id,
            status="ok",
            payload={"value": value},
            outcome_ttl_seconds=outcome_ttl,
        )
        return SingleFlightResult(
            value=value,
            owner=True,
            fencing_token=fencing_token,
        )

    def _publish_flight_outcome(
        self,
        *,
        key_id: str,
        owner_id: str,
        status: str,
        payload: Mapping[str, Any],
        outcome_ttl_seconds: int,
    ) -> int:
        now_ms = self._now_ms()
        with self._lock:
            connection = self._require()
            lease_row = connection.execute(
                "SELECT * FROM evidence_flights WHERE key_id = ? AND owner_id = ?",
                [key_id, owner_id],
            ).fetchone()
            if lease_row is None:
                raise SingleFlightError(
                    "only the current single-flight owner may publish"
                )
            mapping = _row_mapping(lease_row)
            fencing = int(mapping["fencing_token"])
            body = dict(payload)
            encoded = _canonical_json(body)
            if len(encoded.encode("utf-8")) > MAX_BODY_BYTES:
                raise DatabaseEvidenceStoreBoundsError(
                    "single-flight outcome exceeds max body bytes"
                )
            digest = _sha256_digest(encoded.encode("utf-8"))
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_flight_outcomes(
                    key_id, fencing_token, status, outcome_json,
                    outcome_digest, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    key_id,
                    fencing,
                    status,
                    encoded,
                    digest,
                    now_ms,
                    now_ms + outcome_ttl_seconds * 1000,
                ],
            )
            connection.execute(
                "DELETE FROM evidence_flights WHERE key_id = ?",
                [key_id],
            )
            self._commit_if_idle(connection)
            return fencing

    # -- rebuild -------------------------------------------------------------

    def rebuild_projection(
        self,
        kind: str = "admitted_receipts",
    ) -> ProjectionRebuildReceipt:
        """Rebuild a projection strictly from admitted (non-poisoned) evidence."""

        selected_kind = _text(kind, "kind")
        with self._lock:
            connection = self._require()
            if selected_kind == "use_outcomes":
                rows = connection.execute(
                    "SELECT * FROM evidence_use_outcomes "
                    "ORDER BY used_at_ms, use_id"
                ).fetchall()
                material = [_row_mapping(row) for row in rows]
            elif selected_kind == "invalidations":
                rows = connection.execute(
                    "SELECT * FROM evidence_invalidations "
                    "ORDER BY invalidated_at_ms, invalidation_id"
                ).fetchall()
                material = [_row_mapping(row) for row in rows]
            else:
                rows = connection.execute(
                    """
                    SELECT r.* FROM evidence_receipts r
                    WHERE r.poisoned = FALSE
                      AND NOT EXISTS(
                          SELECT 1 FROM evidence_invalidations i
                          WHERE i.receipt_id = r.receipt_id
                      )
                    ORDER BY r.created_at_ms, r.receipt_id
                    """
                ).fetchall()
                material = [
                    self._receipt_from_row(_row_mapping(row)).to_dict()
                    for row in rows
                ]
            safe_material = [_json_safe(item) for item in material]
            digest = _sha256_digest(
                _canonical_json(safe_material).encode("utf-8")
            )
            stamp = _utc_iso()
            projection_id = _identity(
                "evidence-projection",
                {
                    "kind": selected_kind,
                    "snapshot_id": self._snapshot_id,
                    "digest": digest,
                },
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_projections(
                    projection_id, snapshot_id, kind, digest, row_count,
                    rebuilt_from, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    projection_id,
                    self._snapshot_id,
                    selected_kind,
                    digest,
                    len(safe_material),
                    "admitted_evidence",
                    stamp,
                    _canonical_json(
                        {
                            "kind": selected_kind,
                            "digest": digest,
                            "row_count": len(safe_material),
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
                row_count=len(safe_material),
                rebuilt_from="admitted_evidence",
                created_at=stamp,
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            connection = self._require()

            def count(table: str) -> int:
                row = connection.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()
                mapping = _row_mapping(row)
                return int(mapping.get("n") or 0)

            return {
                "interface": self.INTERFACE,
                "snapshot_id": self._snapshot_id,
                "authority": AUTHORITY_CLASS,
                "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
                "receipt_count": count("evidence_receipts"),
                "invalidation_count": count("evidence_invalidations"),
                "attestation_count": count("evidence_attestations"),
                "use_outcome_count": count("evidence_use_outcomes"),
                "flight_count": count("evidence_flights"),
            }


def open_database_evidence_store(
    database_path: Path | str,
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    auto_redact: bool = True,
    default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    clock: Callable[[], float] = time.time,
) -> DatabaseEvidenceStore:
    """Open and initialize a DatabaseEvidenceStore."""

    return DatabaseEvidenceStore(
        database_path,
        snapshot_id=snapshot_id,
        auto_redact=auto_redact,
        default_ttl_seconds=default_ttl_seconds,
        clock=clock,
    ).open()


__all__ = (
    "AUTHORITY_CLASS",
    "AssuranceLevel",
    "CACHE_ASSURANCE_POLICY",
    "DATABASE_EVIDENCE_STORE_INTERFACE",
    "DATABASE_EVIDENCE_STORE_SCHEMA",
    "DEFAULT_SNAPSHOT_ID",
    "DEFAULT_TTL_SECONDS",
    "DatabaseEvidenceStore",
    "DatabaseEvidenceStoreBoundsError",
    "DatabaseEvidenceStoreConflictError",
    "DatabaseEvidenceStoreError",
    "DatabaseEvidenceStoreIntegrityError",
    "DatabaseEvidenceStoreNotOpenError",
    "DatabaseEvidenceStorePoisonedError",
    "DatabaseEvidenceStoreStaleError",
    "DuckDBUnavailableError",
    "EVIDENCE_ATTESTATION_SCHEMA",
    "EVIDENCE_FLIGHT_SCHEMA",
    "EVIDENCE_INVALIDATION_SCHEMA",
    "EVIDENCE_KEY_SCHEMA",
    "EVIDENCE_RECEIPT_SCHEMA",
    "EVIDENCE_USE_OUTCOME_SCHEMA",
    "EvidenceKey",
    "EvidenceKind",
    "EvidenceLookupResult",
    "EvidenceReceipt",
    "EvidenceVerdict",
    "LookupStatus",
    "PROJECTION_REBUILD_SCHEMA",
    "ProjectionRebuildReceipt",
    "REDACTION_MARKER",
    "RejectionReason",
    "SingleFlightError",
    "SingleFlightExecutionError",
    "SingleFlightResult",
    "SingleFlightTimeout",
    "assurance_satisfies",
    "duckdb_available",
    "open_database_evidence_store",
)
