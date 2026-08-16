"""Incremental worktree scanning and AST invalidation (DatabaseRepositoryIndexer@1).

DQP-021 / Interfaces: ``DatabaseRepositoryIndexer@1``, ``ASTInvalidation@1``
===========================================================================

Polls Git/content identities, parses only added/changed/renamed files, retires
deleted path bindings, invalidates dependent symbol/impact/cache/proof rows,
and persists scan cursors plus coverage/frontier receipts.

Filesystem watchers and notifications are **hints only**.  Authoritative
advancement of a worktree snapshot head requires a complete scan (full,
incremental, or reconcile) that finishes without partial failure.  A partial
or crashed scan may record a frontier receipt but **never** advances the
authoritative snapshot head.

Acceptance properties
---------------------
* An incremental result equals a clean full scan for the same snapshot
  identity (repository, tree, overlay, parser, policy, scanner).
* Missed or coalesced notifications are recovered by content-identity
  reconciliation against the live worktree ledger.
* A partial scan never advances the authoritative snapshot head.
* Dependent facts cannot remain current after source, parser, or policy drift.

Cold import of this module performs no filesystem, database, network, provider,
or process action.  Opening an indexer is the first I/O boundary.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Final

from .duckdb_ast_index import (
    AUTHORITY_CLASS,
    DEFAULT_PARSER_ID,
    DEFAULT_SCANNER_VERSION,
    DuckDBASTIndex,
    DuckDBASTIndexError,
    DuckDBASTIndexIntegrityError,
    DuckDBASTIndexNotOpenError,
    ParseStatus,
    SnapshotIngestResult,
    SourceFileSpec,
    SourceSnapshot,
    duckdb_available,
    open_duckdb_ast_index,
)
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_REPOSITORY_INDEXER_INTERFACE: Final[str] = (
    "DatabaseRepositoryIndexer@1"
)
AST_INVALIDATION_INTERFACE: Final[str] = "ASTInvalidation@1"

DATABASE_REPOSITORY_INDEXER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-repository-indexer@1"
)
AST_INVALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ast-invalidation@1"
)
SCAN_CURSOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/scan-cursor@1"
)
COVERAGE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/scan-coverage-receipt@1"
)
DEPENDENT_FACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dependent-fact@1"
)

DEFAULT_INDEXER_SCANNER_VERSION: Final[str] = (
    "database-repository-indexer-scanner@1"
)
DEFAULT_POLICY_ID: Final[str] = "repository-scan-policy@1"
AUTHORITY_CLASS_INDEXER: Final[str] = "derived_evidence"

MAX_FILES_PER_SCAN: Final[int] = 100_000
MAX_REASON_BYTES: Final[int] = 1_024
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_PATH_BYTES: Final[int] = 4_096

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS indexer_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_cursors (
    worktree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    last_tree_id VARCHAR NOT NULL DEFAULT '',
    last_overlay_digest VARCHAR NOT NULL DEFAULT '',
    last_parser_id VARCHAR NOT NULL DEFAULT '',
    last_policy_id VARCHAR NOT NULL DEFAULT '',
    last_scanner_version VARCHAR NOT NULL DEFAULT '',
    last_notification_seq BIGINT NOT NULL DEFAULT 0,
    last_scan_run_id VARCHAR NOT NULL DEFAULT '',
    last_reconciled_at VARCHAR NOT NULL DEFAULT '',
    updated_at VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS authoritative_heads (
    worktree_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL DEFAULT '',
    parser_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL,
    scan_run_id VARCHAR NOT NULL,
    advanced_at VARCHAR NOT NULL,
    file_ledger_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    scan_run_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    target_tree_id VARCHAR NOT NULL,
    target_overlay_digest VARCHAR NOT NULL DEFAULT '',
    parser_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL DEFAULT '',
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS scan_runs_worktree_idx
    ON scan_runs(worktree_id, started_at);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    seq BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    change_kind VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL DEFAULT '',
    observed_at VARCHAR NOT NULL,
    applied INTEGER NOT NULL DEFAULT 0,
    coalesced_into VARCHAR NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS notifications_worktree_open_idx
    ON notifications(worktree_id, applied, seq);
CREATE UNIQUE INDEX IF NOT EXISTS notifications_worktree_seq_uidx
    ON notifications(worktree_id, seq);

CREATE TABLE IF NOT EXISTS ast_invalidations (
    invalidation_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL DEFAULT '',
    scan_run_id VARCHAR NOT NULL DEFAULT '',
    path VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    prior_content_digest VARCHAR NOT NULL DEFAULT '',
    new_content_digest VARCHAR NOT NULL DEFAULT '',
    prior_blob_identity VARCHAR NOT NULL DEFAULT '',
    replacement_blob_identity VARCHAR NOT NULL DEFAULT '',
    record_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS ast_invalidations_worktree_idx
    ON ast_invalidations(worktree_id, recorded_at);
CREATE INDEX IF NOT EXISTS ast_invalidations_path_idx
    ON ast_invalidations(worktree_id, path);

CREATE TABLE IF NOT EXISTS coverage_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    scan_run_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL DEFAULT '',
    path_count BIGINT NOT NULL,
    added_count BIGINT NOT NULL,
    changed_count BIGINT NOT NULL,
    deleted_count BIGINT NOT NULL,
    renamed_count BIGINT NOT NULL,
    reused_count BIGINT NOT NULL,
    parsed_count BIGINT NOT NULL,
    invalidated_count BIGINT NOT NULL,
    notification_applied_count BIGINT NOT NULL DEFAULT 0,
    notification_missed_count BIGINT NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS coverage_receipts_scan_idx
    ON coverage_receipts(scan_run_id);

CREATE TABLE IF NOT EXISTS dependent_facts (
    fact_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    fact_kind VARCHAR NOT NULL,
    subject_path VARCHAR NOT NULL DEFAULT '',
    subject_id VARCHAR NOT NULL DEFAULT '',
    currency VARCHAR NOT NULL,
    bound_snapshot_id VARCHAR NOT NULL DEFAULT '',
    bound_parser_id VARCHAR NOT NULL DEFAULT '',
    bound_policy_id VARCHAR NOT NULL DEFAULT '',
    invalidated_by VARCHAR NOT NULL DEFAULT '',
    updated_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS dependent_facts_worktree_currency_idx
    ON dependent_facts(worktree_id, currency);
CREATE INDEX IF NOT EXISTS dependent_facts_path_idx
    ON dependent_facts(worktree_id, subject_path);

CREATE TABLE IF NOT EXISTS scan_frontiers (
    frontier_id VARCHAR PRIMARY KEY,
    scan_run_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    paths_completed BIGINT NOT NULL DEFAULT 0,
    paths_total BIGINT NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS scan_frontiers_run_idx
    ON scan_frontiers(scan_run_id);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseRepositoryIndexerError(RuntimeError):
    """Base error for the database repository indexer."""


class DatabaseRepositoryIndexerNotOpenError(DatabaseRepositoryIndexerError):
    """Operation requires an open indexer."""


class DatabaseRepositoryIndexerIntegrityError(
    DatabaseRepositoryIndexerError, ValueError
):
    """Identity, path, or payload integrity failure."""


class DatabaseRepositoryIndexerBoundsError(
    DatabaseRepositoryIndexerError, ValueError
):
    """A resource or payload bound was exceeded."""


class DatabaseRepositoryIndexerConflictError(DatabaseRepositoryIndexerError):
    """Duplicate identity with a conflicting payload."""


class DuckDBUnavailableError(DatabaseRepositoryIndexerError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ScanMode(str, Enum):
    """How a scan was initiated."""

    FULL = "full"
    INCREMENTAL = "incremental"
    RECONCILE = "reconcile"


class ScanStatus(str, Enum):
    """Lifecycle of one scan run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"


class ChangeKind(str, Enum):
    """Watcher/notification change vocabulary (hints only)."""

    ADDED = "added"
    CHANGED = "changed"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNTRACKED = "untracked"
    SUBMODULE = "submodule"
    UNKNOWN = "unknown"


class InvalidationReason(str, Enum):
    """Why a path binding or dependent fact was invalidated."""

    SOURCE_CHANGED = "source_changed"
    PATH_DELETED = "path_deleted"
    PATH_RENAMED = "path_renamed"
    PARSER_DRIFT = "parser_drift"
    POLICY_DRIFT = "policy_drift"
    SUBMODULE_CHANGED = "submodule_changed"
    UNTRACKED_POLICY = "untracked_policy"


class FactKind(str, Enum):
    """Dependent derived-fact kinds that must not stay current after drift."""

    SYMBOL = "symbol"
    IMPACT = "impact"
    CACHE = "cache"
    PROOF = "proof"


class FactCurrency(str, Enum):
    """Currency of a dependent fact relative to the authoritative head."""

    CURRENT = "current"
    STALE = "stale"
    INVALIDATED = "invalidated"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(*, coarse: bool = False) -> str:
    now = datetime.now(timezone.utc)
    if coarse:
        return now.replace(microsecond=0).isoformat()
    # Microsecond precision keeps scan-run identities unique under rapid tests.
    return now.isoformat()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseRepositoryIndexerIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseRepositoryIndexerIntegrityError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseRepositoryIndexerBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatabaseRepositoryIndexerIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _normalize_digest(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("sha256:"):
        hexpart = text[len("sha256:") :]
        if len(hexpart) == 64 and all(
            ch in "0123456789abcdef" for ch in hexpart.casefold()
        ):
            return "sha256:" + hexpart.casefold()
        raise DatabaseRepositoryIndexerIntegrityError(
            "content_digest is not a sha256 digest"
        )
    if len(text) == 64 and all(
        ch in "0123456789abcdef" for ch in text.casefold()
    ):
        return "sha256:" + text.casefold()
    raise DatabaseRepositoryIndexerIntegrityError(
        "content_digest is not a sha256 digest"
    )


def _repo_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        raise DatabaseRepositoryIndexerIntegrityError("path is required")
    if "\x00" in text:
        raise DatabaseRepositoryIndexerIntegrityError("path contains NUL")
    if text.startswith("/") or text.startswith("~"):
        raise DatabaseRepositoryIndexerIntegrityError(
            f"path must be repository-relative: {text}"
        )
    pure = PurePosixPath(text)
    if ".." in pure.parts or pure.is_absolute():
        raise DatabaseRepositoryIndexerIntegrityError(
            f"path escapes repository root: {text}"
        )
    normalized = pure.as_posix().lstrip("./")
    if not normalized or normalized == ".":
        raise DatabaseRepositoryIndexerIntegrityError("path is required")
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise DatabaseRepositoryIndexerBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes"
        )
    return normalized


def _bounded_text(value: Any, maximum: int) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8", errors="surrogatepass")
    if len(encoded) <= maximum:
        return text
    # Truncate on byte boundary.
    clipped = encoded[:maximum]
    return clipped.decode("utf-8", errors="ignore")


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return {key: row[key] for key in row.keys()}  # type: ignore[attr-defined]
    except Exception:
        pass
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    return dict(row)


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in sql_text.split(";"):
        text = chunk.strip()
        if text:
            statements.append(text)
    return statements


def _coerce_file_spec(item: SourceFileSpec | Mapping[str, Any]) -> SourceFileSpec:
    if isinstance(item, SourceFileSpec):
        return item
    if not isinstance(item, Mapping):
        raise DatabaseRepositoryIndexerIntegrityError(
            "files entries must be SourceFileSpec or mappings"
        )
    return SourceFileSpec(
        path=str(item.get("path") or ""),
        content=item.get("content"),
        content_digest=str(item.get("content_digest") or ""),
        language=str(item.get("language") or ""),
        blob_id=str(item.get("blob_id") or ""),
        ignored=bool(item.get("ignored") or False),
        ast_record=item.get("ast_record"),
    )


def _file_ledger(
    files: Sequence[SourceFileSpec],
) -> dict[str, str]:
    return {item.path: item.content_digest for item in files}


def _detect_renames(
    deleted: Mapping[str, str],
    added: Mapping[str, str],
) -> tuple[dict[str, str], set[str], set[str]]:
    """Map new_path -> old_path for digest-matched renames.

    Returns ``(rename_map, consumed_deleted, consumed_added)``.
    """

    by_digest: dict[str, list[str]] = {}
    for path, digest in deleted.items():
        if digest:
            by_digest.setdefault(digest, []).append(path)
    for paths in by_digest.values():
        paths.sort()

    rename_map: dict[str, str] = {}
    consumed_deleted: set[str] = set()
    consumed_added: set[str] = set()
    for new_path in sorted(added):
        digest = added[new_path]
        candidates = by_digest.get(digest) or []
        while candidates:
            old_path = candidates.pop(0)
            if old_path in consumed_deleted:
                continue
            rename_map[new_path] = old_path
            consumed_deleted.add(old_path)
            consumed_added.add(new_path)
            break
    return rename_map, consumed_deleted, consumed_added


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASTInvalidation:
    """ASTInvalidation@1 — durable receipt that a path binding is no longer current."""

    INTERFACE: ClassVar[str] = AST_INVALIDATION_INTERFACE

    invalidation_id: str
    worktree_id: str
    path: str
    reason: InvalidationReason | str
    prior_content_digest: str = ""
    new_content_digest: str = ""
    prior_blob_identity: str = ""
    replacement_blob_identity: str = ""
    record_id: str = ""
    snapshot_id: str = ""
    scan_run_id: str = ""
    recorded_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        reason = self.reason
        if isinstance(reason, InvalidationReason):
            reason_value = reason
        else:
            reason_value = InvalidationReason(str(reason).strip())
        object.__setattr__(self, "reason", reason_value)
        for name in (
            "prior_content_digest",
            "new_content_digest",
            "prior_blob_identity",
            "replacement_blob_identity",
            "record_id",
            "snapshot_id",
            "scan_run_id",
        ):
            raw = str(getattr(self, name) or "").strip()
            if name.endswith("digest") and raw:
                raw = _normalize_digest(raw)
            object.__setattr__(self, name, raw)
        stamp = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", stamp)
        computed = _identity(
            "ast-invalidation",
            {
                "schema": AST_INVALIDATION_SCHEMA,
                "worktree_id": self.worktree_id,
                "path": self.path,
                "reason": self.reason.value
                if isinstance(self.reason, InvalidationReason)
                else str(self.reason),
                "prior_content_digest": self.prior_content_digest,
                "new_content_digest": self.new_content_digest,
                "snapshot_id": self.snapshot_id,
                "scan_run_id": self.scan_run_id,
                "recorded_at": self.recorded_at,
            },
        )
        claimed = str(self.invalidation_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseRepositoryIndexerIntegrityError(
                "AST invalidation identity does not match payload"
            )
        object.__setattr__(self, "invalidation_id", claimed or computed)

    @property
    def interface(self) -> str:
        return AST_INVALIDATION_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": AST_INVALIDATION_INTERFACE,
            "schema": AST_INVALIDATION_SCHEMA,
            "invalidation_id": self.invalidation_id,
            "worktree_id": self.worktree_id,
            "path": self.path,
            "reason": self.reason.value
            if isinstance(self.reason, InvalidationReason)
            else str(self.reason),
            "prior_content_digest": self.prior_content_digest,
            "new_content_digest": self.new_content_digest,
            "prior_blob_identity": self.prior_blob_identity,
            "replacement_blob_identity": self.replacement_blob_identity,
            "record_id": self.record_id,
            "snapshot_id": self.snapshot_id,
            "scan_run_id": self.scan_run_id,
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS_INDEXER,
        }


@dataclass(frozen=True)
class ScanCursor:
    """Persisted scan progress for one worktree (not authoritative head)."""

    worktree_id: str
    repository_id: str
    last_tree_id: str = ""
    last_overlay_digest: str = ""
    last_parser_id: str = ""
    last_policy_id: str = ""
    last_scanner_version: str = ""
    last_notification_seq: int = 0
    last_scan_run_id: str = ""
    last_reconciled_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "last_notification_seq",
            _nonneg_int(int(self.last_notification_seq), "last_notification_seq"),
        )
        stamp = str(self.updated_at or "").strip() or _utc_iso()
        object.__setattr__(self, "updated_at", stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCAN_CURSOR_SCHEMA,
            "worktree_id": self.worktree_id,
            "repository_id": self.repository_id,
            "last_tree_id": self.last_tree_id,
            "last_overlay_digest": self.last_overlay_digest,
            "last_parser_id": self.last_parser_id,
            "last_policy_id": self.last_policy_id,
            "last_scanner_version": self.last_scanner_version,
            "last_notification_seq": self.last_notification_seq,
            "last_scan_run_id": self.last_scan_run_id,
            "last_reconciled_at": self.last_reconciled_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class AuthoritativeHead:
    """The only snapshot head that may be treated as current for a worktree."""

    worktree_id: str
    snapshot_id: str
    repository_id: str
    tree_id: str
    overlay_digest: str
    parser_id: str
    policy_id: str
    scanner_version: str
    scan_run_id: str
    advanced_at: str
    file_ledger: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "parser_id", _text(self.parser_id, "parser_id")
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "scanner_version",
            _text(self.scanner_version, "scanner_version"),
        )
        object.__setattr__(
            self, "scan_run_id", _text(self.scan_run_id, "scan_run_id")
        )
        overlay = str(self.overlay_digest or "").strip()
        if overlay:
            overlay = _normalize_digest(overlay)
        object.__setattr__(self, "overlay_digest", overlay)
        ledger = {
            _repo_path(path): _normalize_digest(digest)
            for path, digest in dict(self.file_ledger or {}).items()
        }
        object.__setattr__(self, "file_ledger", MappingProxyType(ledger))
        stamp = str(self.advanced_at or "").strip() or _utc_iso()
        object.__setattr__(self, "advanced_at", stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worktree_id": self.worktree_id,
            "snapshot_id": self.snapshot_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "overlay_digest": self.overlay_digest,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "scanner_version": self.scanner_version,
            "scan_run_id": self.scan_run_id,
            "advanced_at": self.advanced_at,
            "file_ledger": dict(self.file_ledger),
            "authority": AUTHORITY_CLASS_INDEXER,
        }


@dataclass(frozen=True)
class CoverageReceipt:
    """Bounded coverage/frontier receipt for one scan run."""

    receipt_id: str
    scan_run_id: str
    worktree_id: str
    snapshot_id: str
    path_count: int
    added_count: int
    changed_count: int
    deleted_count: int
    renamed_count: int
    reused_count: int
    parsed_count: int
    invalidated_count: int
    notification_applied_count: int = 0
    notification_missed_count: int = 0
    complete: bool = False
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "path_count",
            "added_count",
            "changed_count",
            "deleted_count",
            "renamed_count",
            "reused_count",
            "parsed_count",
            "invalidated_count",
            "notification_applied_count",
            "notification_missed_count",
        ):
            object.__setattr__(
                self, name, _nonneg_int(int(getattr(self, name)), name)
            )
        object.__setattr__(self, "complete", bool(self.complete))
        stamp = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", stamp)
        body = dict(self.body or {})
        object.__setattr__(self, "body", MappingProxyType(body))
        computed = _identity(
            "coverage-receipt",
            {
                "schema": COVERAGE_RECEIPT_SCHEMA,
                "scan_run_id": self.scan_run_id,
                "worktree_id": self.worktree_id,
                "snapshot_id": self.snapshot_id,
                "path_count": self.path_count,
                "added_count": self.added_count,
                "changed_count": self.changed_count,
                "deleted_count": self.deleted_count,
                "renamed_count": self.renamed_count,
                "reused_count": self.reused_count,
                "parsed_count": self.parsed_count,
                "invalidated_count": self.invalidated_count,
                "complete": self.complete,
                "recorded_at": self.recorded_at,
            },
        )
        claimed = str(self.receipt_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseRepositoryIndexerIntegrityError(
                "coverage receipt identity does not match payload"
            )
        object.__setattr__(self, "receipt_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COVERAGE_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "scan_run_id": self.scan_run_id,
            "worktree_id": self.worktree_id,
            "snapshot_id": self.snapshot_id,
            "path_count": self.path_count,
            "added_count": self.added_count,
            "changed_count": self.changed_count,
            "deleted_count": self.deleted_count,
            "renamed_count": self.renamed_count,
            "reused_count": self.reused_count,
            "parsed_count": self.parsed_count,
            "invalidated_count": self.invalidated_count,
            "notification_applied_count": self.notification_applied_count,
            "notification_missed_count": self.notification_missed_count,
            "complete": self.complete,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
            "authority": AUTHORITY_CLASS_INDEXER,
        }


@dataclass(frozen=True)
class DependentFact:
    """A symbol/impact/cache/proof fact bound to a scan identity."""

    fact_id: str
    worktree_id: str
    fact_kind: FactKind | str
    subject_path: str = ""
    subject_id: str = ""
    currency: FactCurrency | str = FactCurrency.CURRENT
    bound_snapshot_id: str = ""
    bound_parser_id: str = ""
    bound_policy_id: str = ""
    invalidated_by: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        kind = self.fact_kind
        if not isinstance(kind, FactKind):
            kind = FactKind(str(kind).strip())
        object.__setattr__(self, "fact_kind", kind)
        currency = self.currency
        if not isinstance(currency, FactCurrency):
            currency = FactCurrency(str(currency).strip())
        object.__setattr__(self, "currency", currency)
        path = str(self.subject_path or "").strip()
        if path:
            path = _repo_path(path)
        object.__setattr__(self, "subject_path", path)
        object.__setattr__(
            self, "subject_id", str(self.subject_id or "").strip()
        )
        stamp = str(self.updated_at or "").strip() or _utc_iso()
        object.__setattr__(self, "updated_at", stamp)
        computed = _identity(
            "dependent-fact",
            {
                "schema": DEPENDENT_FACT_SCHEMA,
                "worktree_id": self.worktree_id,
                "fact_kind": self.fact_kind.value
                if isinstance(self.fact_kind, FactKind)
                else str(self.fact_kind),
                "subject_path": self.subject_path,
                "subject_id": self.subject_id,
                "bound_snapshot_id": self.bound_snapshot_id,
            },
        )
        claimed = str(self.fact_id or "").strip()
        if claimed and claimed != computed:
            # Allow explicit stable IDs that still round-trip.
            object.__setattr__(self, "fact_id", claimed)
        else:
            object.__setattr__(self, "fact_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEPENDENT_FACT_SCHEMA,
            "fact_id": self.fact_id,
            "worktree_id": self.worktree_id,
            "fact_kind": self.fact_kind.value
            if isinstance(self.fact_kind, FactKind)
            else str(self.fact_kind),
            "subject_path": self.subject_path,
            "subject_id": self.subject_id,
            "currency": self.currency.value
            if isinstance(self.currency, FactCurrency)
            else str(self.currency),
            "bound_snapshot_id": self.bound_snapshot_id,
            "bound_parser_id": self.bound_parser_id,
            "bound_policy_id": self.bound_policy_id,
            "invalidated_by": self.invalidated_by,
            "updated_at": self.updated_at,
            "authority": AUTHORITY_CLASS_INDEXER,
        }


@dataclass(frozen=True)
class NotificationRecord:
    """One watcher/notification hint (never authority)."""

    notification_id: str
    worktree_id: str
    seq: int
    path: str
    change_kind: ChangeKind | str
    content_digest: str = ""
    observed_at: str = ""
    applied: bool = False
    coalesced_into: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "seq", _nonneg_int(int(self.seq), "seq"))
        kind = self.change_kind
        if not isinstance(kind, ChangeKind):
            kind = ChangeKind(str(kind).strip() or ChangeKind.UNKNOWN.value)
        object.__setattr__(self, "change_kind", kind)
        digest = str(self.content_digest or "").strip()
        if digest:
            digest = _normalize_digest(digest)
        object.__setattr__(self, "content_digest", digest)
        stamp = str(self.observed_at or "").strip() or _utc_iso()
        object.__setattr__(self, "observed_at", stamp)
        object.__setattr__(self, "applied", bool(self.applied))
        object.__setattr__(
            self, "coalesced_into", str(self.coalesced_into or "").strip()
        )
        computed = _identity(
            "scan-notification",
            {
                "worktree_id": self.worktree_id,
                "seq": self.seq,
                "path": self.path,
                "change_kind": self.change_kind.value
                if isinstance(self.change_kind, ChangeKind)
                else str(self.change_kind),
                "content_digest": self.content_digest,
                "observed_at": self.observed_at,
            },
        )
        claimed = str(self.notification_id or "").strip()
        object.__setattr__(self, "notification_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "worktree_id": self.worktree_id,
            "seq": self.seq,
            "path": self.path,
            "change_kind": self.change_kind.value
            if isinstance(self.change_kind, ChangeKind)
            else str(self.change_kind),
            "content_digest": self.content_digest,
            "observed_at": self.observed_at,
            "applied": self.applied,
            "coalesced_into": self.coalesced_into,
            "authority": "hint_only",
        }


@dataclass(frozen=True)
class ScanDelta:
    """Path-level diff between the previous head ledger and the current tree."""

    added: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    renamed: Mapping[str, str] = field(default_factory=dict)
    unchanged: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "changed": list(self.changed),
            "deleted": list(self.deleted),
            "renamed": dict(self.renamed),
            "unchanged": list(self.unchanged),
        }


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one scan attempt (complete or partial)."""

    scan_run_id: str
    worktree_id: str
    mode: ScanMode | str
    status: ScanStatus | str
    repository_id: str
    tree_id: str
    overlay_digest: str
    parser_id: str
    policy_id: str
    scanner_version: str
    snapshot: SourceSnapshot | None
    ingest: SnapshotIngestResult | None
    delta: ScanDelta
    invalidations: tuple[ASTInvalidation, ...]
    coverage: CoverageReceipt
    head_advanced: bool
    authoritative_head: AuthoritativeHead | None
    started_at: str
    finished_at: str

    @property
    def complete(self) -> bool:
        status = (
            self.status
            if isinstance(self.status, ScanStatus)
            else ScanStatus(str(self.status))
        )
        return status is ScanStatus.SUCCEEDED and self.head_advanced

    @property
    def snapshot_id(self) -> str:
        if self.snapshot is not None:
            return self.snapshot.snapshot_id
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": DATABASE_REPOSITORY_INDEXER_INTERFACE,
            "scan_run_id": self.scan_run_id,
            "worktree_id": self.worktree_id,
            "mode": self.mode.value
            if isinstance(self.mode, ScanMode)
            else str(self.mode),
            "status": self.status.value
            if isinstance(self.status, ScanStatus)
            else str(self.status),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "overlay_digest": self.overlay_digest,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "scanner_version": self.scanner_version,
            "snapshot_id": self.snapshot_id,
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "ingest": self.ingest.to_dict() if self.ingest else None,
            "delta": self.delta.to_dict(),
            "invalidations": [item.to_dict() for item in self.invalidations],
            "coverage": self.coverage.to_dict(),
            "head_advanced": self.head_advanced,
            "authoritative_head": (
                self.authoritative_head.to_dict()
                if self.authoritative_head
                else None
            ),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "authority": AUTHORITY_CLASS_INDEXER,
        }


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class DatabaseRepositoryIndexer:
    """Incremental worktree scanner with fenced authoritative heads.

    Interface: ``DatabaseRepositoryIndexer@1``.
    """

    INTERFACE: Final[str] = DATABASE_REPOSITORY_INDEXER_INTERFACE
    SCHEMA: Final[str] = DATABASE_REPOSITORY_INDEXER_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        ast_database_path: Path | str | None = None,
        parser_id: str = DEFAULT_PARSER_ID,
        scanner_version: str = DEFAULT_INDEXER_SCANNER_VERSION,
        policy_id: str = DEFAULT_POLICY_ID,
        ast_index: DuckDBASTIndex | None = None,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseRepositoryIndexer; install "
                "the optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._ast_path = (
            Path(ast_database_path)
            if ast_database_path is not None
            else self._path.with_name(self._path.stem + ".ast.duckdb")
        )
        self._parser_id = _text(parser_id or DEFAULT_PARSER_ID, "parser_id")
        self._scanner_version = _text(
            scanner_version or DEFAULT_INDEXER_SCANNER_VERSION,
            "scanner_version",
        )
        self._policy_id = _text(policy_id or DEFAULT_POLICY_ID, "policy_id")
        self._external_ast = ast_index
        self._ast: DuckDBASTIndex | None = None
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def ast_database_path(self) -> Path:
        return self._ast_path

    @property
    def parser_id(self) -> str:
        return self._parser_id

    @property
    def scanner_version(self) -> str:
        return self._scanner_version

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    @property
    def ast_index(self) -> DuckDBASTIndex:
        if self._ast is None or not self._ast.is_open:
            raise DatabaseRepositoryIndexerNotOpenError(
                "AST index is not open"
            )
        return self._ast

    def open(self) -> "DatabaseRepositoryIndexer":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_REPOSITORY_INDEXER_INTERFACE),
                ("schema", DATABASE_REPOSITORY_INDEXER_SCHEMA),
                ("parser_id", self._parser_id),
                ("scanner_version", self._scanner_version),
                ("policy_id", self._policy_id),
                ("authority", AUTHORITY_CLASS_INDEXER),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO indexer_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            if self._external_ast is not None:
                if not self._external_ast.is_open:
                    self._external_ast.open()
                self._ast = self._external_ast
            else:
                self._ast = open_duckdb_ast_index(
                    self._ast_path,
                    parser_id=self._parser_id,
                    scanner_version=DEFAULT_SCANNER_VERSION,
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
            if self._external_ast is None and self._ast is not None:
                try:
                    self._ast.close()
                except Exception:
                    pass
            if self._external_ast is None:
                self._ast = None

    def __enter__(self) -> "DatabaseRepositoryIndexer":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseRepositoryIndexerNotOpenError(
                "DatabaseRepositoryIndexer is not open"
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

    # -- public metadata -----------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                "SELECT key, value FROM indexer_metadata ORDER BY key ASC"
            ).fetchall()
            result = {
                str(_row_mapping(row)["key"]): str(_row_mapping(row)["value"])
                for row in rows
            }
            result["ast_interface"] = self.ast_index.INTERFACE
            return result

    # -- notifications (hints only) ------------------------------------------

    def notify_change(
        self,
        *,
        worktree_id: str,
        path: str,
        change_kind: ChangeKind | str = ChangeKind.CHANGED,
        content_digest: str = "",
        observed_at: str | None = None,
        coalesce: bool = True,
    ) -> NotificationRecord:
        """Record a watcher hint.  Never advances the authoritative head."""

        wt = _text(worktree_id, "worktree_id")
        repo_path = _repo_path(path)
        kind = (
            change_kind
            if isinstance(change_kind, ChangeKind)
            else ChangeKind(str(change_kind).strip() or "unknown")
        )
        stamp = _text(observed_at or _utc_iso(), "observed_at")
        digest = str(content_digest or "").strip()
        if digest:
            digest = _normalize_digest(digest)

        with self._lock:
            connection = self._require()
            try:
                connection.execute("BEGIN TRANSACTION")
                seq = self._next_notification_seq(connection, wt)
                if coalesce:
                    open_rows = connection.execute(
                        """
                        SELECT notification_id, path, change_kind, content_digest
                        FROM notifications
                        WHERE worktree_id = ? AND applied = 0
                          AND coalesced_into = '' AND path = ?
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        [wt, repo_path],
                    ).fetchall()
                    if open_rows:
                        prior = _row_mapping(open_rows[0])
                        # Coalesce into the newest open notification for path.
                        notification = NotificationRecord(
                            notification_id="",
                            worktree_id=wt,
                            seq=seq,
                            path=repo_path,
                            change_kind=kind,
                            content_digest=digest or str(
                                prior.get("content_digest") or ""
                            ),
                            observed_at=stamp,
                            applied=False,
                            coalesced_into=str(prior["notification_id"]),
                        )
                        # Promote the prior row to the latest kind/digest.
                        connection.execute(
                            """
                            UPDATE notifications
                            SET change_kind = ?, content_digest = ?,
                                observed_at = ?
                            WHERE notification_id = ?
                            """,
                            [
                                kind.value,
                                digest or str(prior.get("content_digest") or ""),
                                stamp,
                                str(prior["notification_id"]),
                            ],
                        )
                        self._insert_notification(connection, notification)
                        self._bump_cursor_seq(connection, wt, seq, stamp)
                        connection.execute("COMMIT")
                        self._commit_if_idle(connection)
                        return notification

                notification = NotificationRecord(
                    notification_id="",
                    worktree_id=wt,
                    seq=seq,
                    path=repo_path,
                    change_kind=kind,
                    content_digest=digest,
                    observed_at=stamp,
                    applied=False,
                )
                self._insert_notification(connection, notification)
                self._bump_cursor_seq(connection, wt, seq, stamp)
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return notification

    def list_open_notifications(
        self, worktree_id: str
    ) -> tuple[NotificationRecord, ...]:
        wt = _text(worktree_id, "worktree_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT notification_id, worktree_id, seq, path, change_kind,
                       content_digest, observed_at, applied, coalesced_into
                FROM notifications
                WHERE worktree_id = ? AND applied = 0 AND coalesced_into = ''
                ORDER BY seq ASC
                """,
                [wt],
            ).fetchall()
            return tuple(self._notification_from_row(row) for row in rows)

    # -- dependent facts -----------------------------------------------------

    def register_dependent_fact(
        self,
        *,
        worktree_id: str,
        fact_kind: FactKind | str,
        subject_path: str = "",
        subject_id: str = "",
        bound_snapshot_id: str = "",
        bound_parser_id: str = "",
        bound_policy_id: str = "",
        fact_id: str = "",
    ) -> DependentFact:
        """Register a derived fact as current against the given binding."""

        head = self.get_authoritative_head(worktree_id)
        snapshot_id = bound_snapshot_id or (
            head.snapshot_id if head is not None else ""
        )
        parser = bound_parser_id or (
            head.parser_id if head is not None else self._parser_id
        )
        policy = bound_policy_id or (
            head.policy_id if head is not None else self._policy_id
        )
        stamp = _utc_iso()
        fact = DependentFact(
            fact_id=fact_id,
            worktree_id=worktree_id,
            fact_kind=fact_kind,
            subject_path=subject_path,
            subject_id=subject_id,
            currency=FactCurrency.CURRENT,
            bound_snapshot_id=snapshot_id,
            bound_parser_id=parser,
            bound_policy_id=policy,
            updated_at=stamp,
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                INSERT OR REPLACE INTO dependent_facts (
                    fact_id, worktree_id, fact_kind, subject_path, subject_id,
                    currency, bound_snapshot_id, bound_parser_id,
                    bound_policy_id, invalidated_by, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
                """,
                [
                    fact.fact_id,
                    fact.worktree_id,
                    fact.fact_kind.value
                    if isinstance(fact.fact_kind, FactKind)
                    else str(fact.fact_kind),
                    fact.subject_path,
                    fact.subject_id,
                    FactCurrency.CURRENT.value,
                    fact.bound_snapshot_id,
                    fact.bound_parser_id,
                    fact.bound_policy_id,
                    stamp,
                ],
            )
            self._commit_if_idle(connection)
        return fact

    def list_dependent_facts(
        self,
        worktree_id: str,
        *,
        currency: FactCurrency | str | None = None,
        subject_path: str | None = None,
    ) -> tuple[DependentFact, ...]:
        wt = _text(worktree_id, "worktree_id")
        clauses = ["worktree_id = ?"]
        params: list[Any] = [wt]
        if currency is not None:
            cur = (
                currency
                if isinstance(currency, FactCurrency)
                else FactCurrency(str(currency))
            )
            clauses.append("currency = ?")
            params.append(cur.value)
        if subject_path is not None:
            clauses.append("subject_path = ?")
            params.append(_repo_path(subject_path))
        sql = f"""
            SELECT fact_id, worktree_id, fact_kind, subject_path, subject_id,
                   currency, bound_snapshot_id, bound_parser_id,
                   bound_policy_id, invalidated_by, updated_at
            FROM dependent_facts
            WHERE {' AND '.join(clauses)}
            ORDER BY fact_kind ASC, subject_path ASC, fact_id ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            return tuple(self._fact_from_row(row) for row in rows)

    # -- heads / cursors / invalidations -------------------------------------

    def get_authoritative_head(
        self, worktree_id: str
    ) -> AuthoritativeHead | None:
        wt = _text(worktree_id, "worktree_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT worktree_id, snapshot_id, repository_id, tree_id,
                       overlay_digest, parser_id, policy_id, scanner_version,
                       scan_run_id, advanced_at, file_ledger_json
                FROM authoritative_heads
                WHERE worktree_id = ?
                LIMIT 1
                """,
                [wt],
            ).fetchone()
            if row is None:
                return None
            return self._head_from_row(row)

    def get_scan_cursor(self, worktree_id: str) -> ScanCursor | None:
        wt = _text(worktree_id, "worktree_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT worktree_id, repository_id, last_tree_id,
                       last_overlay_digest, last_parser_id, last_policy_id,
                       last_scanner_version, last_notification_seq,
                       last_scan_run_id, last_reconciled_at, updated_at
                FROM scan_cursors
                WHERE worktree_id = ?
                LIMIT 1
                """,
                [wt],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            return ScanCursor(
                worktree_id=str(mapping["worktree_id"]),
                repository_id=str(mapping["repository_id"]),
                last_tree_id=str(mapping.get("last_tree_id") or ""),
                last_overlay_digest=str(
                    mapping.get("last_overlay_digest") or ""
                ),
                last_parser_id=str(mapping.get("last_parser_id") or ""),
                last_policy_id=str(mapping.get("last_policy_id") or ""),
                last_scanner_version=str(
                    mapping.get("last_scanner_version") or ""
                ),
                last_notification_seq=int(
                    mapping.get("last_notification_seq") or 0
                ),
                last_scan_run_id=str(mapping.get("last_scan_run_id") or ""),
                last_reconciled_at=str(
                    mapping.get("last_reconciled_at") or ""
                ),
                updated_at=str(mapping.get("updated_at") or ""),
            )

    def list_invalidations(
        self,
        worktree_id: str,
        *,
        snapshot_id: str | None = None,
        path: str | None = None,
    ) -> tuple[ASTInvalidation, ...]:
        wt = _text(worktree_id, "worktree_id")
        clauses = ["worktree_id = ?"]
        params: list[Any] = [wt]
        if snapshot_id is not None:
            clauses.append("snapshot_id = ?")
            params.append(_text(snapshot_id, "snapshot_id"))
        if path is not None:
            clauses.append("path = ?")
            params.append(_repo_path(path))
        sql = f"""
            SELECT invalidation_id, worktree_id, snapshot_id, scan_run_id,
                   path, reason, prior_content_digest, new_content_digest,
                   prior_blob_identity, replacement_blob_identity, record_id,
                   recorded_at
            FROM ast_invalidations
            WHERE {' AND '.join(clauses)}
            ORDER BY recorded_at ASC, path ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            return tuple(self._invalidation_from_row(row) for row in rows)

    def get_coverage_receipt(
        self, scan_run_id: str
    ) -> CoverageReceipt | None:
        selected = _text(scan_run_id, "scan_run_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT receipt_id, scan_run_id, worktree_id, snapshot_id,
                       path_count, added_count, changed_count, deleted_count,
                       renamed_count, reused_count, parsed_count,
                       invalidated_count, notification_applied_count,
                       notification_missed_count, complete, recorded_at,
                       body_json
                FROM coverage_receipts
                WHERE scan_run_id = ?
                LIMIT 1
                """,
                [selected],
            ).fetchone()
            if row is None:
                return None
            return self._coverage_from_row(row)

    # -- scanning ------------------------------------------------------------

    def full_scan(
        self,
        *,
        worktree_id: str,
        repository_id: str,
        tree_id: str,
        files: Sequence[SourceFileSpec | Mapping[str, Any]],
        overlay_digest: str = "",
        parser_id: str | None = None,
        policy_id: str | None = None,
        scanner_version: str | None = None,
        crash_after_paths: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ScanResult:
        """Clean full scan of the current worktree identity."""

        return self._scan(
            worktree_id=worktree_id,
            repository_id=repository_id,
            tree_id=tree_id,
            files=files,
            overlay_digest=overlay_digest,
            parser_id=parser_id,
            policy_id=policy_id,
            scanner_version=scanner_version,
            mode=ScanMode.FULL,
            force_full=True,
            crash_after_paths=crash_after_paths,
            on_progress=on_progress,
        )

    def incremental_scan(
        self,
        *,
        worktree_id: str,
        repository_id: str,
        tree_id: str,
        files: Sequence[SourceFileSpec | Mapping[str, Any]],
        overlay_digest: str = "",
        parser_id: str | None = None,
        policy_id: str | None = None,
        scanner_version: str | None = None,
        crash_after_paths: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        apply_notifications: bool = True,
    ) -> ScanResult:
        """Incremental scan using content identities (notifications are hints)."""

        return self._scan(
            worktree_id=worktree_id,
            repository_id=repository_id,
            tree_id=tree_id,
            files=files,
            overlay_digest=overlay_digest,
            parser_id=parser_id,
            policy_id=policy_id,
            scanner_version=scanner_version,
            mode=ScanMode.INCREMENTAL,
            force_full=False,
            crash_after_paths=crash_after_paths,
            on_progress=on_progress,
            apply_notifications=apply_notifications,
        )

    def reconcile(
        self,
        *,
        worktree_id: str,
        repository_id: str,
        tree_id: str,
        files: Sequence[SourceFileSpec | Mapping[str, Any]],
        overlay_digest: str = "",
        parser_id: str | None = None,
        policy_id: str | None = None,
        scanner_version: str | None = None,
        crash_after_paths: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ScanResult:
        """Recover missed notifications by reconciling content identities."""

        return self._scan(
            worktree_id=worktree_id,
            repository_id=repository_id,
            tree_id=tree_id,
            files=files,
            overlay_digest=overlay_digest,
            parser_id=parser_id,
            policy_id=policy_id,
            scanner_version=scanner_version,
            mode=ScanMode.RECONCILE,
            force_full=False,
            crash_after_paths=crash_after_paths,
            on_progress=on_progress,
            apply_notifications=True,
            mark_reconciled=True,
        )

    def snapshot_equivalence(
        self,
        left_snapshot_id: str,
        right_snapshot_id: str,
    ) -> dict[str, Any]:
        """Compare two AST snapshots for clean-rebuild equivalence."""

        left = _text(left_snapshot_id, "left_snapshot_id")
        right = _text(right_snapshot_id, "right_snapshot_id")
        ast = self.ast_index
        left_files = {
            str(item["path"]): str(item["content_digest"])
            for item in ast.list_files(left)
        }
        right_files = {
            str(item["path"]): str(item["content_digest"])
            for item in ast.list_files(right)
        }
        left_symbols = {
            (item.path, item.qualified_name, item.fingerprint)
            for item in ast.list_symbols(left)
        }
        right_symbols = {
            (item.path, item.qualified_name, item.fingerprint)
            for item in ast.list_symbols(right)
        }
        left_frontiers = {
            (item.path, item.status.value if isinstance(item.status, ParseStatus) else str(item.status), item.reason)
            for item in ast.list_frontiers(left)
        }
        right_frontiers = {
            (item.path, item.status.value if isinstance(item.status, ParseStatus) else str(item.status), item.reason)
            for item in ast.list_frontiers(right)
        }
        equal = (
            left_files == right_files
            and left_symbols == right_symbols
            and left_frontiers == right_frontiers
        )
        return {
            "equal": equal,
            "left_snapshot_id": left,
            "right_snapshot_id": right,
            "file_diff": {
                "only_left": sorted(set(left_files) - set(right_files)),
                "only_right": sorted(set(right_files) - set(left_files)),
                "digest_mismatch": sorted(
                    path
                    for path in set(left_files) & set(right_files)
                    if left_files[path] != right_files[path]
                ),
            },
            "symbol_count_left": len(left_symbols),
            "symbol_count_right": len(right_symbols),
            "frontier_count_left": len(left_frontiers),
            "frontier_count_right": len(right_frontiers),
        }

    # -- internal scan implementation ----------------------------------------

    def _scan(
        self,
        *,
        worktree_id: str,
        repository_id: str,
        tree_id: str,
        files: Sequence[SourceFileSpec | Mapping[str, Any]],
        overlay_digest: str,
        parser_id: str | None,
        policy_id: str | None,
        scanner_version: str | None,
        mode: ScanMode,
        force_full: bool,
        crash_after_paths: int | None,
        on_progress: Callable[[int, int], None] | None,
        apply_notifications: bool = False,
        mark_reconciled: bool = False,
    ) -> ScanResult:
        wt = _text(worktree_id, "worktree_id")
        repo = _text(repository_id, "repository_id")
        tree = _text(tree_id, "tree_id")
        overlay = str(overlay_digest or "").strip()
        if overlay:
            overlay = _normalize_digest(overlay)
        selected_parser = _text(parser_id or self._parser_id, "parser_id")
        selected_policy = _text(policy_id or self._policy_id, "policy_id")
        selected_scanner = _text(
            scanner_version or self._scanner_version, "scanner_version"
        )
        specs = [_coerce_file_spec(item) for item in files]
        if len(specs) > MAX_FILES_PER_SCAN:
            raise DatabaseRepositoryIndexerBoundsError(
                f"scan exceeds {MAX_FILES_PER_SCAN} files"
            )
        if len({item.path for item in specs}) != len(specs):
            raise DatabaseRepositoryIndexerIntegrityError(
                "scan paths must be unique"
            )
        specs = sorted(specs, key=lambda item: item.path)
        current_ledger = _file_ledger(specs)
        started = _utc_iso()
        scan_run_id = _identity(
            "scan-run",
            {
                "worktree_id": wt,
                "repository_id": repo,
                "tree_id": tree,
                "overlay_digest": overlay,
                "parser_id": selected_parser,
                "policy_id": selected_policy,
                "scanner_version": selected_scanner,
                "mode": mode.value,
                "started_at": started,
                "path_count": len(current_ledger),
                "ledger_digest": _sha256_bytes(
                    _canonical_json(current_ledger).encode("utf-8")
                ),
                # Bind to a unique token so rapid identical requests never
                # collide on scan_runs primary key within the same timestamp.
                "nonce": _sha256_bytes(
                    f"{wt}:{mode.value}:{started}:{id(specs)}".encode("utf-8")
                ),
            },
        )

        prior_head = self.get_authoritative_head(wt)
        prior_ledger: dict[str, str] = (
            dict(prior_head.file_ledger) if prior_head is not None else {}
        )

        parser_drift = bool(
            prior_head is not None and prior_head.parser_id != selected_parser
        )
        policy_drift = bool(
            prior_head is not None and prior_head.policy_id != selected_policy
        )
        # Full mode or any drift forces a clean rebuild path selection, but
        # still records precise invalidations against the prior head.
        effective_force_full = force_full or parser_drift or policy_drift

        # Identical tree/overlay/parser/policy/ledger re-scan is idempotent:
        # reaffirm the existing head without re-inserting AST snapshot rows
        # (source snapshot identity is tree-scoped and not re-ingestable).
        if (
            prior_head is not None
            and not parser_drift
            and not policy_drift
            and prior_head.tree_id == tree
            and prior_head.overlay_digest == overlay
            and prior_head.repository_id == repo
            and prior_ledger == current_ledger
            and crash_after_paths is None
        ):
            return self._reaffirm_head_scan(
                scan_run_id=scan_run_id,
                worktree_id=wt,
                repository_id=repo,
                tree_id=tree,
                overlay_digest=overlay,
                parser_id=selected_parser,
                policy_id=selected_policy,
                scanner_version=selected_scanner,
                mode=mode,
                prior_head=prior_head,
                current_ledger=current_ledger,
                started_at=started,
                mark_reconciled=mark_reconciled,
                apply_notifications=apply_notifications,
            )

        delta = self._compute_delta(
            prior_ledger=prior_ledger if not effective_force_full else {},
            current_ledger=current_ledger,
        )
        if effective_force_full and prior_ledger:
            # Under full/drift rebuild, every prior path that is gone is a
            # delete; every current path is treated as re-parsed/reused.
            deleted = tuple(
                sorted(path for path in prior_ledger if path not in current_ledger)
            )
            added = tuple(
                sorted(path for path in current_ledger if path not in prior_ledger)
            )
            changed = tuple(
                sorted(
                    path
                    for path in current_ledger
                    if path in prior_ledger
                    and prior_ledger[path] != current_ledger[path]
                )
            )
            unchanged = tuple(
                sorted(
                    path
                    for path in current_ledger
                    if path in prior_ledger
                    and prior_ledger[path] == current_ledger[path]
                    and not parser_drift
                    and not policy_drift
                )
            )
            rename_map, _, _ = _detect_renames(
                {path: prior_ledger[path] for path in deleted},
                {path: current_ledger[path] for path in added},
            )
            deleted = tuple(
                path for path in deleted if path not in set(rename_map.values())
            )
            added = tuple(path for path in added if path not in rename_map)
            delta = ScanDelta(
                added=added,
                changed=changed,
                deleted=deleted,
                renamed=MappingProxyType(dict(rename_map)),
                unchanged=unchanged,
            )

        # Build ingest file set.  Unchanged digests may omit source bodies so
        # the AST index reuses the parse cache — equivalence still holds.
        ingest_files = self._build_ingest_files(
            specs=specs,
            delta=delta,
            force_full=effective_force_full,
            parser_drift=parser_drift,
            policy_drift=policy_drift,
        )

        # Simulate partial crash before AST ingest commits a new head.
        if crash_after_paths is not None:
            limit = _nonneg_int(crash_after_paths, "crash_after_paths")
            if on_progress is not None:
                on_progress(min(limit, len(ingest_files)), len(ingest_files))
            return self._record_partial_scan(
                scan_run_id=scan_run_id,
                worktree_id=wt,
                repository_id=repo,
                tree_id=tree,
                overlay_digest=overlay,
                parser_id=selected_parser,
                policy_id=selected_policy,
                scanner_version=selected_scanner,
                mode=mode,
                delta=delta,
                paths_completed=min(limit, len(ingest_files)),
                paths_total=len(ingest_files),
                started_at=started,
                reason="partial_scan_crash",
                prior_head=prior_head,
            )

        if on_progress is not None:
            on_progress(0, len(ingest_files))

        try:
            ingest = self.ast_index.ingest_snapshot(
                repository_id=repo,
                tree_id=tree,
                files=ingest_files,
                overlay_digest=overlay,
                worktree_id=wt,
                scanner_version=selected_scanner,
                parser_id=selected_parser,
                created_at=started,
            )
        except DuckDBASTIndexError as exc:
            return self._record_partial_scan(
                scan_run_id=scan_run_id,
                worktree_id=wt,
                repository_id=repo,
                tree_id=tree,
                overlay_digest=overlay,
                parser_id=selected_parser,
                policy_id=selected_policy,
                scanner_version=selected_scanner,
                mode=mode,
                delta=delta,
                paths_completed=0,
                paths_total=len(ingest_files),
                started_at=started,
                reason=f"ast_ingest_failed:{exc}",
                prior_head=prior_head,
            )

        if on_progress is not None:
            on_progress(len(ingest_files), len(ingest_files))

        finished = _utc_iso()
        snapshot = ingest.snapshot
        invalidations = self._build_invalidations(
            worktree_id=wt,
            snapshot_id=snapshot.snapshot_id,
            scan_run_id=scan_run_id,
            prior_ledger=prior_ledger,
            current_ledger=current_ledger,
            delta=delta,
            parser_drift=parser_drift,
            policy_drift=policy_drift,
            recorded_at=finished,
        )

        notification_applied = 0
        notification_missed = 0
        if apply_notifications:
            notification_applied, notification_missed = (
                self._apply_notifications_for_scan(
                    worktree_id=wt,
                    current_ledger=current_ledger,
                    prior_ledger=prior_ledger,
                    delta=delta,
                )
            )

        reused = int(ingest.reused_unit_count)
        parsed = int(ingest.new_unit_count)
        coverage = CoverageReceipt(
            receipt_id="",
            scan_run_id=scan_run_id,
            worktree_id=wt,
            snapshot_id=snapshot.snapshot_id,
            path_count=len(specs),
            added_count=len(delta.added),
            changed_count=len(delta.changed),
            deleted_count=len(delta.deleted),
            renamed_count=len(delta.renamed),
            reused_count=reused,
            parsed_count=parsed,
            invalidated_count=len(invalidations),
            notification_applied_count=notification_applied,
            notification_missed_count=notification_missed,
            complete=True,
            recorded_at=finished,
            body={
                "mode": mode.value,
                "parser_drift": parser_drift,
                "policy_drift": policy_drift,
                "force_full": effective_force_full,
                "indexed_file_count": ingest.indexed_file_count,
                "excluded_file_count": ingest.excluded_file_count,
            },
        )

        head = AuthoritativeHead(
            worktree_id=wt,
            snapshot_id=snapshot.snapshot_id,
            repository_id=repo,
            tree_id=tree,
            overlay_digest=overlay,
            parser_id=selected_parser,
            policy_id=selected_policy,
            scanner_version=selected_scanner,
            scan_run_id=scan_run_id,
            advanced_at=finished,
            file_ledger=current_ledger,
        )

        with self._lock:
            connection = self._require()
            try:
                connection.execute("BEGIN TRANSACTION")
                self._insert_scan_run(
                    connection,
                    scan_run_id=scan_run_id,
                    worktree_id=wt,
                    repository_id=repo,
                    mode=mode,
                    status=ScanStatus.SUCCEEDED,
                    target_tree_id=tree,
                    target_overlay_digest=overlay,
                    parser_id=selected_parser,
                    policy_id=selected_policy,
                    scanner_version=selected_scanner,
                    snapshot_id=snapshot.snapshot_id,
                    started_at=started,
                    finished_at=finished,
                    body={
                        "delta": delta.to_dict(),
                        "head_advanced": True,
                        "parser_drift": parser_drift,
                        "policy_drift": policy_drift,
                    },
                )
                self._insert_coverage(connection, coverage)
                for item in invalidations:
                    self._insert_invalidation(connection, item)
                self._advance_head(connection, head)
                self._upsert_cursor(
                    connection,
                    ScanCursor(
                        worktree_id=wt,
                        repository_id=repo,
                        last_tree_id=tree,
                        last_overlay_digest=overlay,
                        last_parser_id=selected_parser,
                        last_policy_id=selected_policy,
                        last_scanner_version=selected_scanner,
                        last_notification_seq=self._current_notification_seq(
                            connection, wt
                        ),
                        last_scan_run_id=scan_run_id,
                        last_reconciled_at=finished if mark_reconciled else (
                            prior_head.advanced_at if prior_head else ""
                        ),
                        updated_at=finished,
                    ),
                    preserve_reconciled=not mark_reconciled,
                )
                self._invalidate_dependent_facts(
                    connection,
                    worktree_id=wt,
                    invalidations=invalidations,
                    parser_drift=parser_drift,
                    policy_drift=policy_drift,
                    new_parser_id=selected_parser,
                    new_policy_id=selected_policy,
                    new_snapshot_id=snapshot.snapshot_id,
                    recorded_at=finished,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)

        return ScanResult(
            scan_run_id=scan_run_id,
            worktree_id=wt,
            mode=mode,
            status=ScanStatus.SUCCEEDED,
            repository_id=repo,
            tree_id=tree,
            overlay_digest=overlay,
            parser_id=selected_parser,
            policy_id=selected_policy,
            scanner_version=selected_scanner,
            snapshot=snapshot,
            ingest=ingest,
            delta=delta,
            invalidations=invalidations,
            coverage=coverage,
            head_advanced=True,
            authoritative_head=head,
            started_at=started,
            finished_at=finished,
        )

    def _reaffirm_head_scan(
        self,
        *,
        scan_run_id: str,
        worktree_id: str,
        repository_id: str,
        tree_id: str,
        overlay_digest: str,
        parser_id: str,
        policy_id: str,
        scanner_version: str,
        mode: ScanMode,
        prior_head: AuthoritativeHead,
        current_ledger: Mapping[str, str],
        started_at: str,
        mark_reconciled: bool,
        apply_notifications: bool,
    ) -> ScanResult:
        """Idempotent complete scan when the authoritative identity is unchanged."""

        finished = _utc_iso()
        delta = ScanDelta(
            unchanged=tuple(sorted(current_ledger)),
        )
        notification_applied = 0
        notification_missed = 0
        if apply_notifications:
            notification_applied, notification_missed = (
                self._apply_notifications_for_scan(
                    worktree_id=worktree_id,
                    current_ledger=current_ledger,
                    prior_ledger=dict(prior_head.file_ledger),
                    delta=delta,
                )
            )
        snapshot = self.ast_index.get_snapshot(prior_head.snapshot_id)
        coverage = CoverageReceipt(
            receipt_id="",
            scan_run_id=scan_run_id,
            worktree_id=worktree_id,
            snapshot_id=prior_head.snapshot_id,
            path_count=len(current_ledger),
            added_count=0,
            changed_count=0,
            deleted_count=0,
            renamed_count=0,
            reused_count=len(current_ledger),
            parsed_count=0,
            invalidated_count=0,
            notification_applied_count=notification_applied,
            notification_missed_count=notification_missed,
            complete=True,
            recorded_at=finished,
            body={
                "mode": mode.value,
                "reaffirmed": True,
                "prior_snapshot_id": prior_head.snapshot_id,
            },
        )
        head = AuthoritativeHead(
            worktree_id=worktree_id,
            snapshot_id=prior_head.snapshot_id,
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
            parser_id=parser_id,
            policy_id=policy_id,
            scanner_version=scanner_version,
            scan_run_id=scan_run_id,
            advanced_at=finished,
            file_ledger=current_ledger,
        )
        with self._lock:
            connection = self._require()
            try:
                connection.execute("BEGIN TRANSACTION")
                self._insert_scan_run(
                    connection,
                    scan_run_id=scan_run_id,
                    worktree_id=worktree_id,
                    repository_id=repository_id,
                    mode=mode,
                    status=ScanStatus.SUCCEEDED,
                    target_tree_id=tree_id,
                    target_overlay_digest=overlay_digest,
                    parser_id=parser_id,
                    policy_id=policy_id,
                    scanner_version=scanner_version,
                    snapshot_id=prior_head.snapshot_id,
                    started_at=started_at,
                    finished_at=finished,
                    body={
                        "delta": delta.to_dict(),
                        "head_advanced": True,
                        "reaffirmed": True,
                    },
                )
                self._insert_coverage(connection, coverage)
                self._advance_head(connection, head)
                self._upsert_cursor(
                    connection,
                    ScanCursor(
                        worktree_id=worktree_id,
                        repository_id=repository_id,
                        last_tree_id=tree_id,
                        last_overlay_digest=overlay_digest,
                        last_parser_id=parser_id,
                        last_policy_id=policy_id,
                        last_scanner_version=scanner_version,
                        last_notification_seq=self._current_notification_seq(
                            connection, worktree_id
                        ),
                        last_scan_run_id=scan_run_id,
                        last_reconciled_at=finished if mark_reconciled else "",
                        updated_at=finished,
                    ),
                    preserve_reconciled=not mark_reconciled,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)

        return ScanResult(
            scan_run_id=scan_run_id,
            worktree_id=worktree_id,
            mode=mode,
            status=ScanStatus.SUCCEEDED,
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
            parser_id=parser_id,
            policy_id=policy_id,
            scanner_version=scanner_version,
            snapshot=snapshot,
            ingest=None,
            delta=delta,
            invalidations=(),
            coverage=coverage,
            head_advanced=True,
            authoritative_head=head,
            started_at=started_at,
            finished_at=finished,
        )

    def _record_partial_scan(
        self,
        *,
        scan_run_id: str,
        worktree_id: str,
        repository_id: str,
        tree_id: str,
        overlay_digest: str,
        parser_id: str,
        policy_id: str,
        scanner_version: str,
        mode: ScanMode,
        delta: ScanDelta,
        paths_completed: int,
        paths_total: int,
        started_at: str,
        reason: str,
        prior_head: AuthoritativeHead | None,
    ) -> ScanResult:
        finished = _utc_iso()
        coverage = CoverageReceipt(
            receipt_id="",
            scan_run_id=scan_run_id,
            worktree_id=worktree_id,
            snapshot_id="",
            path_count=paths_total,
            added_count=len(delta.added),
            changed_count=len(delta.changed),
            deleted_count=len(delta.deleted),
            renamed_count=len(delta.renamed),
            reused_count=0,
            parsed_count=0,
            invalidated_count=0,
            complete=False,
            recorded_at=finished,
            body={
                "mode": mode.value,
                "partial": True,
                "reason": _bounded_text(reason, MAX_REASON_BYTES),
                "paths_completed": paths_completed,
                "paths_total": paths_total,
            },
        )
        with self._lock:
            connection = self._require()
            try:
                connection.execute("BEGIN TRANSACTION")
                self._insert_scan_run(
                    connection,
                    scan_run_id=scan_run_id,
                    worktree_id=worktree_id,
                    repository_id=repository_id,
                    mode=mode,
                    status=ScanStatus.PARTIAL,
                    target_tree_id=tree_id,
                    target_overlay_digest=overlay_digest,
                    parser_id=parser_id,
                    policy_id=policy_id,
                    scanner_version=scanner_version,
                    snapshot_id="",
                    started_at=started_at,
                    finished_at=finished,
                    body={
                        "delta": delta.to_dict(),
                        "head_advanced": False,
                        "partial": True,
                        "reason": _bounded_text(reason, MAX_REASON_BYTES),
                    },
                )
                self._insert_coverage(connection, coverage)
                frontier_id = _identity(
                    "scan-frontier",
                    {
                        "scan_run_id": scan_run_id,
                        "worktree_id": worktree_id,
                        "status": ScanStatus.PARTIAL.value,
                        "reason": reason,
                    },
                )
                connection.execute(
                    """
                    INSERT INTO scan_frontiers (
                        frontier_id, scan_run_id, worktree_id, status, reason,
                        paths_completed, paths_total, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        frontier_id,
                        scan_run_id,
                        worktree_id,
                        ScanStatus.PARTIAL.value,
                        _bounded_text(reason, MAX_REASON_BYTES),
                        paths_completed,
                        paths_total,
                        finished,
                    ],
                )
                # Cursor may record the attempt, but the authoritative head
                # is intentionally left unchanged.
                self._upsert_cursor(
                    connection,
                    ScanCursor(
                        worktree_id=worktree_id,
                        repository_id=repository_id,
                        last_tree_id=(
                            prior_head.tree_id if prior_head else ""
                        ),
                        last_overlay_digest=(
                            prior_head.overlay_digest if prior_head else ""
                        ),
                        last_parser_id=(
                            prior_head.parser_id if prior_head else parser_id
                        ),
                        last_policy_id=(
                            prior_head.policy_id if prior_head else policy_id
                        ),
                        last_scanner_version=(
                            prior_head.scanner_version
                            if prior_head
                            else scanner_version
                        ),
                        last_notification_seq=self._current_notification_seq(
                            connection, worktree_id
                        ),
                        last_scan_run_id=scan_run_id,
                        last_reconciled_at="",
                        updated_at=finished,
                    ),
                    preserve_reconciled=True,
                    preserve_tree=True,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)

        return ScanResult(
            scan_run_id=scan_run_id,
            worktree_id=worktree_id,
            mode=mode,
            status=ScanStatus.PARTIAL,
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
            parser_id=parser_id,
            policy_id=policy_id,
            scanner_version=scanner_version,
            snapshot=None,
            ingest=None,
            delta=delta,
            invalidations=(),
            coverage=coverage,
            head_advanced=False,
            authoritative_head=prior_head,
            started_at=started_at,
            finished_at=finished,
        )

    def _compute_delta(
        self,
        *,
        prior_ledger: Mapping[str, str],
        current_ledger: Mapping[str, str],
    ) -> ScanDelta:
        prior_paths = set(prior_ledger)
        current_paths = set(current_ledger)
        deleted_paths = {
            path: prior_ledger[path]
            for path in prior_paths - current_paths
        }
        added_paths = {
            path: current_ledger[path]
            for path in current_paths - prior_paths
        }
        rename_map, consumed_deleted, consumed_added = _detect_renames(
            deleted_paths, added_paths
        )
        deleted = tuple(
            sorted(
                path
                for path in deleted_paths
                if path not in consumed_deleted
            )
        )
        added = tuple(
            sorted(
                path for path in added_paths if path not in consumed_added
            )
        )
        changed = tuple(
            sorted(
                path
                for path in current_paths & prior_paths
                if prior_ledger[path] != current_ledger[path]
            )
        )
        unchanged = tuple(
            sorted(
                path
                for path in current_paths & prior_paths
                if prior_ledger[path] == current_ledger[path]
            )
        )
        return ScanDelta(
            added=added,
            changed=changed,
            deleted=deleted,
            renamed=MappingProxyType(dict(rename_map)),
            unchanged=unchanged,
        )

    def _build_ingest_files(
        self,
        *,
        specs: Sequence[SourceFileSpec],
        delta: ScanDelta,
        force_full: bool,
        parser_drift: bool,
        policy_drift: bool,
    ) -> list[SourceFileSpec]:
        """Select bodies for ingest.

        Unchanged paths may be digest-only (parse-cache reuse).  Changed,
        added, renamed targets, and any drift-forced path require content or
        an attached AST record so the AST index can reparse or reuse units.
        """

        need_body = set(delta.added) | set(delta.changed) | set(delta.renamed)
        result: list[SourceFileSpec] = []
        for spec in specs:
            if force_full or parser_drift or policy_drift or spec.path in need_body:
                # Added/changed/renamed (or forced rebuild): keep caller body.
                result.append(spec)
                continue
            # Unchanged path: digest-only so the AST parse cache is authority
            # and we never re-invoke a parser for stable content identities.
            result.append(
                SourceFileSpec(
                    path=spec.path,
                    content_digest=spec.content_digest,
                    language=spec.language,
                    blob_id=spec.blob_id,
                    ignored=spec.ignored,
                )
            )
        return result

    def _build_invalidations(
        self,
        *,
        worktree_id: str,
        snapshot_id: str,
        scan_run_id: str,
        prior_ledger: Mapping[str, str],
        current_ledger: Mapping[str, str],
        delta: ScanDelta,
        parser_drift: bool,
        policy_drift: bool,
        recorded_at: str,
    ) -> tuple[ASTInvalidation, ...]:
        items: list[ASTInvalidation] = []
        for path in delta.deleted:
            items.append(
                ASTInvalidation(
                    invalidation_id="",
                    worktree_id=worktree_id,
                    path=path,
                    reason=InvalidationReason.PATH_DELETED,
                    prior_content_digest=prior_ledger.get(path, ""),
                    prior_blob_identity=prior_ledger.get(path, ""),
                    snapshot_id=snapshot_id,
                    scan_run_id=scan_run_id,
                    recorded_at=recorded_at,
                )
            )
        for new_path, old_path in sorted(delta.renamed.items()):
            items.append(
                ASTInvalidation(
                    invalidation_id="",
                    worktree_id=worktree_id,
                    path=old_path,
                    reason=InvalidationReason.PATH_RENAMED,
                    prior_content_digest=prior_ledger.get(old_path, ""),
                    new_content_digest=current_ledger.get(new_path, ""),
                    prior_blob_identity=prior_ledger.get(old_path, ""),
                    replacement_blob_identity=current_ledger.get(new_path, ""),
                    snapshot_id=snapshot_id,
                    scan_run_id=scan_run_id,
                    recorded_at=recorded_at,
                )
            )
        for path in delta.changed:
            items.append(
                ASTInvalidation(
                    invalidation_id="",
                    worktree_id=worktree_id,
                    path=path,
                    reason=InvalidationReason.SOURCE_CHANGED,
                    prior_content_digest=prior_ledger.get(path, ""),
                    new_content_digest=current_ledger.get(path, ""),
                    prior_blob_identity=prior_ledger.get(path, ""),
                    replacement_blob_identity=current_ledger.get(path, ""),
                    snapshot_id=snapshot_id,
                    scan_run_id=scan_run_id,
                    recorded_at=recorded_at,
                )
            )
        if parser_drift:
            for path in sorted(set(prior_ledger) | set(current_ledger)):
                items.append(
                    ASTInvalidation(
                        invalidation_id="",
                        worktree_id=worktree_id,
                        path=path,
                        reason=InvalidationReason.PARSER_DRIFT,
                        prior_content_digest=prior_ledger.get(path, ""),
                        new_content_digest=current_ledger.get(path, ""),
                        snapshot_id=snapshot_id,
                        scan_run_id=scan_run_id,
                        recorded_at=recorded_at,
                    )
                )
        if policy_drift:
            for path in sorted(set(prior_ledger) | set(current_ledger)):
                items.append(
                    ASTInvalidation(
                        invalidation_id="",
                        worktree_id=worktree_id,
                        path=path,
                        reason=InvalidationReason.POLICY_DRIFT,
                        prior_content_digest=prior_ledger.get(path, ""),
                        new_content_digest=current_ledger.get(path, ""),
                        snapshot_id=snapshot_id,
                        scan_run_id=scan_run_id,
                        recorded_at=recorded_at,
                    )
                )
        # Stable order for determinism.
        items.sort(key=lambda item: (item.path, item.reason.value, item.invalidation_id))
        return tuple(items)

    def _apply_notifications_for_scan(
        self,
        *,
        worktree_id: str,
        current_ledger: Mapping[str, str],
        prior_ledger: Mapping[str, str],
        delta: ScanDelta,
    ) -> tuple[int, int]:
        """Mark open notifications applied; count those that missed the delta."""

        open_notes = self.list_open_notifications(worktree_id)
        if not open_notes:
            return 0, 0
        covered_paths = (
            set(delta.added)
            | set(delta.changed)
            | set(delta.deleted)
            | set(delta.renamed)
            | set(delta.renamed.values())
        )
        # Also cover paths whose digest matches (noop notifications).
        for note in open_notes:
            if note.path in current_ledger and note.path in prior_ledger:
                if current_ledger[note.path] == prior_ledger[note.path]:
                    covered_paths.add(note.path)

        applied = 0
        missed = 0
        with self._lock:
            connection = self._require()
            for note in open_notes:
                if note.path in covered_paths or note.path not in prior_ledger:
                    connection.execute(
                        """
                        UPDATE notifications SET applied = 1
                        WHERE notification_id = ?
                        """,
                        [note.notification_id],
                    )
                    applied += 1
                else:
                    # Notification claimed a change that identity reconciliation
                    # did not observe as a mutation — still mark applied so
                    # reconcile can clear stale hints, but count as missed.
                    connection.execute(
                        """
                        UPDATE notifications SET applied = 1
                        WHERE notification_id = ?
                        """,
                        [note.notification_id],
                    )
                    applied += 1
                    missed += 1
            # Coalesced children are closed with their parents.
            connection.execute(
                """
                UPDATE notifications SET applied = 1
                WHERE worktree_id = ? AND applied = 0 AND coalesced_into != ''
                """,
                [worktree_id],
            )
            self._commit_if_idle(connection)
        return applied, missed

    def _invalidate_dependent_facts(
        self,
        connection: Any,
        *,
        worktree_id: str,
        invalidations: Sequence[ASTInvalidation],
        parser_drift: bool,
        policy_drift: bool,
        new_parser_id: str,
        new_policy_id: str,
        new_snapshot_id: str,
        recorded_at: str,
    ) -> None:
        path_reasons: dict[str, str] = {}
        for item in invalidations:
            path_reasons[item.path] = (
                item.reason.value
                if isinstance(item.reason, InvalidationReason)
                else str(item.reason)
            )

        rows = connection.execute(
            """
            SELECT fact_id, subject_path, bound_parser_id, bound_policy_id,
                   currency
            FROM dependent_facts
            WHERE worktree_id = ? AND currency = ?
            """,
            [worktree_id, FactCurrency.CURRENT.value],
        ).fetchall()
        for row in rows:
            mapping = _row_mapping(row)
            fact_id = str(mapping["fact_id"])
            subject_path = str(mapping.get("subject_path") or "")
            bound_parser = str(mapping.get("bound_parser_id") or "")
            bound_policy = str(mapping.get("bound_policy_id") or "")
            reason = ""
            if subject_path and subject_path in path_reasons:
                reason = path_reasons[subject_path]
            elif parser_drift and bound_parser and bound_parser != new_parser_id:
                reason = InvalidationReason.PARSER_DRIFT.value
            elif policy_drift and bound_policy and bound_policy != new_policy_id:
                reason = InvalidationReason.POLICY_DRIFT.value
            elif parser_drift:
                reason = InvalidationReason.PARSER_DRIFT.value
            elif policy_drift:
                reason = InvalidationReason.POLICY_DRIFT.value
            if not reason:
                # Path still current and no drift — rebind to new snapshot.
                connection.execute(
                    """
                    UPDATE dependent_facts
                    SET bound_snapshot_id = ?, updated_at = ?
                    WHERE fact_id = ?
                    """,
                    [new_snapshot_id, recorded_at, fact_id],
                )
                continue
            connection.execute(
                """
                UPDATE dependent_facts
                SET currency = ?, invalidated_by = ?, updated_at = ?
                WHERE fact_id = ?
                """,
                [
                    FactCurrency.INVALIDATED.value,
                    reason,
                    recorded_at,
                    fact_id,
                ],
            )

    # -- row mappers / persistence -------------------------------------------

    def _next_notification_seq(self, connection: Any, worktree_id: str) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(seq), 0) AS n
            FROM notifications WHERE worktree_id = ?
            """,
            [worktree_id],
        ).fetchone()
        return int(_row_mapping(row).get("n") or 0) + 1

    def _current_notification_seq(
        self, connection: Any, worktree_id: str
    ) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(seq), 0) AS n
            FROM notifications WHERE worktree_id = ?
            """,
            [worktree_id],
        ).fetchone()
        return int(_row_mapping(row).get("n") or 0)

    def _bump_cursor_seq(
        self,
        connection: Any,
        worktree_id: str,
        seq: int,
        stamp: str,
    ) -> None:
        row = connection.execute(
            "SELECT worktree_id FROM scan_cursors WHERE worktree_id = ?",
            [worktree_id],
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO scan_cursors (
                    worktree_id, repository_id, last_notification_seq,
                    updated_at
                ) VALUES (?, '', ?, ?)
                """,
                [worktree_id, seq, stamp],
            )
        else:
            connection.execute(
                """
                UPDATE scan_cursors
                SET last_notification_seq = ?, updated_at = ?
                WHERE worktree_id = ?
                """,
                [seq, stamp, worktree_id],
            )

    def _insert_notification(
        self, connection: Any, notification: NotificationRecord
    ) -> None:
        connection.execute(
            """
            INSERT INTO notifications (
                notification_id, worktree_id, seq, path, change_kind,
                content_digest, observed_at, applied, coalesced_into
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                notification.notification_id,
                notification.worktree_id,
                int(notification.seq),
                notification.path,
                notification.change_kind.value
                if isinstance(notification.change_kind, ChangeKind)
                else str(notification.change_kind),
                notification.content_digest,
                notification.observed_at,
                1 if notification.applied else 0,
                notification.coalesced_into,
            ],
        )

    def _insert_scan_run(
        self,
        connection: Any,
        *,
        scan_run_id: str,
        worktree_id: str,
        repository_id: str,
        mode: ScanMode,
        status: ScanStatus,
        target_tree_id: str,
        target_overlay_digest: str,
        parser_id: str,
        policy_id: str,
        scanner_version: str,
        snapshot_id: str,
        started_at: str,
        finished_at: str,
        body: Mapping[str, Any],
    ) -> None:
        body_json = _canonical_json(dict(body))
        if len(body_json.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            raise DatabaseRepositoryIndexerBoundsError(
                "scan run body exceeds bound"
            )
        connection.execute(
            """
            INSERT INTO scan_runs (
                scan_run_id, worktree_id, repository_id, mode, status,
                target_tree_id, target_overlay_digest, parser_id, policy_id,
                scanner_version, snapshot_id, started_at, finished_at,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                scan_run_id,
                worktree_id,
                repository_id,
                mode.value,
                status.value,
                target_tree_id,
                target_overlay_digest,
                parser_id,
                policy_id,
                scanner_version,
                snapshot_id,
                started_at,
                finished_at,
                body_json,
            ],
        )

    def _insert_coverage(
        self, connection: Any, coverage: CoverageReceipt
    ) -> None:
        connection.execute(
            """
            INSERT INTO coverage_receipts (
                receipt_id, scan_run_id, worktree_id, snapshot_id, path_count,
                added_count, changed_count, deleted_count, renamed_count,
                reused_count, parsed_count, invalidated_count,
                notification_applied_count, notification_missed_count,
                complete, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                coverage.receipt_id,
                coverage.scan_run_id,
                coverage.worktree_id,
                coverage.snapshot_id,
                coverage.path_count,
                coverage.added_count,
                coverage.changed_count,
                coverage.deleted_count,
                coverage.renamed_count,
                coverage.reused_count,
                coverage.parsed_count,
                coverage.invalidated_count,
                coverage.notification_applied_count,
                coverage.notification_missed_count,
                1 if coverage.complete else 0,
                coverage.recorded_at,
                _canonical_json(dict(coverage.body)),
            ],
        )

    def _insert_invalidation(
        self, connection: Any, item: ASTInvalidation
    ) -> None:
        connection.execute(
            """
            INSERT INTO ast_invalidations (
                invalidation_id, worktree_id, snapshot_id, scan_run_id, path,
                reason, prior_content_digest, new_content_digest,
                prior_blob_identity, replacement_blob_identity, record_id,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                item.invalidation_id,
                item.worktree_id,
                item.snapshot_id,
                item.scan_run_id,
                item.path,
                item.reason.value
                if isinstance(item.reason, InvalidationReason)
                else str(item.reason),
                item.prior_content_digest,
                item.new_content_digest,
                item.prior_blob_identity,
                item.replacement_blob_identity,
                item.record_id,
                item.recorded_at,
            ],
        )

    def _advance_head(
        self, connection: Any, head: AuthoritativeHead
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO authoritative_heads (
                worktree_id, snapshot_id, repository_id, tree_id,
                overlay_digest, parser_id, policy_id, scanner_version,
                scan_run_id, advanced_at, file_ledger_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                head.worktree_id,
                head.snapshot_id,
                head.repository_id,
                head.tree_id,
                head.overlay_digest,
                head.parser_id,
                head.policy_id,
                head.scanner_version,
                head.scan_run_id,
                head.advanced_at,
                _canonical_json(dict(head.file_ledger)),
            ],
        )

    def _upsert_cursor(
        self,
        connection: Any,
        cursor: ScanCursor,
        *,
        preserve_reconciled: bool = False,
        preserve_tree: bool = False,
    ) -> None:
        existing = connection.execute(
            """
            SELECT last_tree_id, last_overlay_digest, last_parser_id,
                   last_policy_id, last_scanner_version, last_reconciled_at,
                   last_notification_seq
            FROM scan_cursors WHERE worktree_id = ?
            """,
            [cursor.worktree_id],
        ).fetchone()
        last_tree = cursor.last_tree_id
        last_overlay = cursor.last_overlay_digest
        last_parser = cursor.last_parser_id
        last_policy = cursor.last_policy_id
        last_scanner = cursor.last_scanner_version
        last_reconciled = cursor.last_reconciled_at
        if existing is not None:
            mapping = _row_mapping(existing)
            if preserve_tree:
                last_tree = str(mapping.get("last_tree_id") or last_tree)
                last_overlay = str(
                    mapping.get("last_overlay_digest") or last_overlay
                )
                last_parser = str(
                    mapping.get("last_parser_id") or last_parser
                )
                last_policy = str(
                    mapping.get("last_policy_id") or last_policy
                )
                last_scanner = str(
                    mapping.get("last_scanner_version") or last_scanner
                )
            if preserve_reconciled:
                last_reconciled = str(
                    mapping.get("last_reconciled_at") or last_reconciled
                )
        connection.execute(
            """
            INSERT OR REPLACE INTO scan_cursors (
                worktree_id, repository_id, last_tree_id, last_overlay_digest,
                last_parser_id, last_policy_id, last_scanner_version,
                last_notification_seq, last_scan_run_id, last_reconciled_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                cursor.worktree_id,
                cursor.repository_id,
                last_tree,
                last_overlay,
                last_parser,
                last_policy,
                last_scanner,
                int(cursor.last_notification_seq),
                cursor.last_scan_run_id,
                last_reconciled,
                cursor.updated_at,
            ],
        )

    def _notification_from_row(self, row: Any) -> NotificationRecord:
        mapping = _row_mapping(row)
        return NotificationRecord(
            notification_id=str(mapping["notification_id"]),
            worktree_id=str(mapping["worktree_id"]),
            seq=int(mapping["seq"]),
            path=str(mapping["path"]),
            change_kind=str(mapping["change_kind"]),
            content_digest=str(mapping.get("content_digest") or ""),
            observed_at=str(mapping.get("observed_at") or ""),
            applied=bool(int(mapping.get("applied") or 0)),
            coalesced_into=str(mapping.get("coalesced_into") or ""),
        )

    def _invalidation_from_row(self, row: Any) -> ASTInvalidation:
        mapping = _row_mapping(row)
        return ASTInvalidation(
            invalidation_id=str(mapping["invalidation_id"]),
            worktree_id=str(mapping["worktree_id"]),
            path=str(mapping["path"]),
            reason=str(mapping["reason"]),
            prior_content_digest=str(
                mapping.get("prior_content_digest") or ""
            ),
            new_content_digest=str(mapping.get("new_content_digest") or ""),
            prior_blob_identity=str(
                mapping.get("prior_blob_identity") or ""
            ),
            replacement_blob_identity=str(
                mapping.get("replacement_blob_identity") or ""
            ),
            record_id=str(mapping.get("record_id") or ""),
            snapshot_id=str(mapping.get("snapshot_id") or ""),
            scan_run_id=str(mapping.get("scan_run_id") or ""),
            recorded_at=str(mapping.get("recorded_at") or ""),
        )

    def _coverage_from_row(self, row: Any) -> CoverageReceipt:
        mapping = _row_mapping(row)
        body_raw = str(mapping.get("body_json") or "{}")
        try:
            body = json.loads(body_raw)
        except json.JSONDecodeError:
            body = {}
        return CoverageReceipt(
            receipt_id=str(mapping["receipt_id"]),
            scan_run_id=str(mapping["scan_run_id"]),
            worktree_id=str(mapping["worktree_id"]),
            snapshot_id=str(mapping.get("snapshot_id") or ""),
            path_count=int(mapping.get("path_count") or 0),
            added_count=int(mapping.get("added_count") or 0),
            changed_count=int(mapping.get("changed_count") or 0),
            deleted_count=int(mapping.get("deleted_count") or 0),
            renamed_count=int(mapping.get("renamed_count") or 0),
            reused_count=int(mapping.get("reused_count") or 0),
            parsed_count=int(mapping.get("parsed_count") or 0),
            invalidated_count=int(mapping.get("invalidated_count") or 0),
            notification_applied_count=int(
                mapping.get("notification_applied_count") or 0
            ),
            notification_missed_count=int(
                mapping.get("notification_missed_count") or 0
            ),
            complete=bool(int(mapping.get("complete") or 0)),
            recorded_at=str(mapping.get("recorded_at") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def _fact_from_row(self, row: Any) -> DependentFact:
        mapping = _row_mapping(row)
        return DependentFact(
            fact_id=str(mapping["fact_id"]),
            worktree_id=str(mapping["worktree_id"]),
            fact_kind=str(mapping["fact_kind"]),
            subject_path=str(mapping.get("subject_path") or ""),
            subject_id=str(mapping.get("subject_id") or ""),
            currency=str(mapping.get("currency") or FactCurrency.CURRENT.value),
            bound_snapshot_id=str(mapping.get("bound_snapshot_id") or ""),
            bound_parser_id=str(mapping.get("bound_parser_id") or ""),
            bound_policy_id=str(mapping.get("bound_policy_id") or ""),
            invalidated_by=str(mapping.get("invalidated_by") or ""),
            updated_at=str(mapping.get("updated_at") or ""),
        )

    def _head_from_row(self, row: Any) -> AuthoritativeHead:
        mapping = _row_mapping(row)
        ledger_raw = str(mapping.get("file_ledger_json") or "{}")
        try:
            ledger = json.loads(ledger_raw)
        except json.JSONDecodeError:
            ledger = {}
        if not isinstance(ledger, Mapping):
            ledger = {}
        return AuthoritativeHead(
            worktree_id=str(mapping["worktree_id"]),
            snapshot_id=str(mapping["snapshot_id"]),
            repository_id=str(mapping["repository_id"]),
            tree_id=str(mapping["tree_id"]),
            overlay_digest=str(mapping.get("overlay_digest") or ""),
            parser_id=str(mapping["parser_id"]),
            policy_id=str(mapping["policy_id"]),
            scanner_version=str(mapping["scanner_version"]),
            scan_run_id=str(mapping["scan_run_id"]),
            advanced_at=str(mapping["advanced_at"]),
            file_ledger={
                str(path): str(digest) for path, digest in ledger.items()
            },
        )


def open_database_repository_indexer(
    database_path: Path | str,
    *,
    ast_database_path: Path | str | None = None,
    parser_id: str = DEFAULT_PARSER_ID,
    scanner_version: str = DEFAULT_INDEXER_SCANNER_VERSION,
    policy_id: str = DEFAULT_POLICY_ID,
    ast_index: DuckDBASTIndex | None = None,
) -> DatabaseRepositoryIndexer:
    """Open (or create) a database repository indexer."""

    return DatabaseRepositoryIndexer(
        database_path,
        ast_database_path=ast_database_path,
        parser_id=parser_id,
        scanner_version=scanner_version,
        policy_id=policy_id,
        ast_index=ast_index,
    ).open()


__all__ = [
    "AST_INVALIDATION_INTERFACE",
    "AST_INVALIDATION_SCHEMA",
    "ASTInvalidation",
    "AUTHORITY_CLASS_INDEXER",
    "AuthoritativeHead",
    "ChangeKind",
    "CoverageReceipt",
    "DATABASE_REPOSITORY_INDEXER_INTERFACE",
    "DATABASE_REPOSITORY_INDEXER_SCHEMA",
    "DEFAULT_INDEXER_SCANNER_VERSION",
    "DEFAULT_POLICY_ID",
    "DatabaseRepositoryIndexer",
    "DatabaseRepositoryIndexerBoundsError",
    "DatabaseRepositoryIndexerConflictError",
    "DatabaseRepositoryIndexerError",
    "DatabaseRepositoryIndexerIntegrityError",
    "DatabaseRepositoryIndexerNotOpenError",
    "DependentFact",
    "DuckDBUnavailableError",
    "FactCurrency",
    "FactKind",
    "InvalidationReason",
    "NotificationRecord",
    "ScanCursor",
    "ScanDelta",
    "ScanMode",
    "ScanResult",
    "ScanStatus",
    "duckdb_available",
    "open_database_repository_indexer",
]
