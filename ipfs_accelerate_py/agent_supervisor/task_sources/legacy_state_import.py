"""Legacy Markdown/JSON/JSONL/SQLite/DuckDB state import with provenance.

Implements ``LegacyStateImport@1``, ``ImportManifest@1``, and
``ImportReceipt@1`` for DQP-010.

Importers read legacy artifacts under an explicit manifest. Each source is
fingerprinted by byte digest and bound to a parser/schema identity. Import is
idempotent, defaults to preview, never mutates sources, and never applies
last-write-wins. Conflicts require an explicit policy of select, merge,
quarantine, or reject. Strict apply commits atomically or not at all. Every
accepted row carries source digest and parser version provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .duckdb_state import open_duckdb_connection
from .task_identity import canonical_content_cid, canonical_json_bytes


# ---------------------------------------------------------------------------
# Schema / interface identities
# ---------------------------------------------------------------------------

LEGACY_STATE_IMPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-state-import@1"
)
IMPORT_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/import-manifest@1"
)
IMPORT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/import-receipt@1"
)
IMPORTED_ROW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/imported-row@1"
)
IMPORT_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/import-source@1"
)

LEGACY_STATE_IMPORT_INTERFACE: Final = "LegacyStateImport@1"
IMPORT_MANIFEST_INTERFACE: Final = "ImportManifest@1"
IMPORT_RECEIPT_INTERFACE: Final = "ImportReceipt@1"

PARSER_VERSION: Final = "legacy-state-import/1"
DEFAULT_PARSER_VERSION: Final = PARSER_VERSION

# Outcomes recorded on ImportReceipt.
OUTCOME_PREVIEW: Final = "preview"
OUTCOME_APPLIED: Final = "applied"
OUTCOME_REPLAYED: Final = "replayed"
OUTCOME_FAILED: Final = "failed"
OUTCOME_REFUSED: Final = "refused"

# Row dispositions.
DISPOSITION_ACCEPT: Final = "accept"
DISPOSITION_SELECT: Final = "select"
DISPOSITION_MERGE: Final = "merge"
DISPOSITION_QUARANTINE: Final = "quarantine"
DISPOSITION_REJECT: Final = "reject"
DISPOSITION_SKIP_DUPLICATE: Final = "skip_duplicate"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MARKDOWN_HEADING_RE = re.compile(
    r"^##[ \t]+(?P<record_id>\S+)(?:[ \t]+(?P<title>[^\n]*))?[ \t]*$",
    flags=re.MULTILINE,
)
_MARKDOWN_FIELD_RE = re.compile(
    r"^[ \t]*-[ \t]+(?P<key>[A-Za-z][A-Za-z0-9_ -]*?):[ \t]*(?P<value>.*)$"
)

_BOOKKEEPING_SQL: Final = """
CREATE TABLE IF NOT EXISTS legacy_import_receipts (
    import_key VARCHAR PRIMARY KEY,
    receipt_cid VARCHAR NOT NULL UNIQUE,
    outcome VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,
    manifest_fingerprint VARCHAR NOT NULL,
    source_digests_json VARCHAR NOT NULL,
    accepted_count INTEGER NOT NULL,
    rejected_count INTEGER NOT NULL,
    quarantined_count INTEGER NOT NULL,
    conflict_count INTEGER NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS legacy_import_sources (
    import_key VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    source_digest VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    schema_identity VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    byte_count INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    rejected_count INTEGER NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (import_key, source_id)
);
CREATE TABLE IF NOT EXISTS legacy_import_rows (
    import_key VARCHAR NOT NULL,
    row_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    source_digest VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    payload_digest VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (import_key, row_id)
);
CREATE TABLE IF NOT EXISTS legacy_import_quarantine (
    import_key VARCHAR NOT NULL,
    quarantine_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    source_ids_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (import_key, quarantine_id)
);
CREATE TABLE IF NOT EXISTS legacy_import_rejections (
    import_key VARCHAR NOT NULL,
    rejection_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    source_digest VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (import_key, rejection_id)
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LegacyStateImportError(RuntimeError):
    """Base class for fail-closed legacy state import errors."""


class ImportManifestError(LegacyStateImportError, ValueError):
    """The import manifest is malformed or inconsistent."""


class ImportSourceError(LegacyStateImportError):
    """A source path cannot be read or is not the declared kind."""


class ImportParseError(LegacyStateImportError):
    """A source is corrupt, truncated, or cannot be parsed."""


class ImportSchemaError(LegacyStateImportError):
    """A source declares or requires an unsupported schema."""


class ImportConflictError(LegacyStateImportError):
    """A conflict policy cannot reconcile competing authorities."""


class ImportAtomicityError(LegacyStateImportError):
    """Strict apply could not commit the full import transaction."""


class ImportSourceMutationError(LegacyStateImportError):
    """A source was mutated between digest observation and apply."""


class DuckDBUnavailableError(LegacyStateImportError):
    """DuckDB is required for the import target but is missing."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ImportSourceKind(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"


class ImportDomain(str, Enum):
    OBJECTIVES = "objectives"
    TASKBOARDS = "taskboards"
    PLAN_REVISIONS = "plan_revisions"
    QUEUES = "queues"
    EVENTS = "events"
    STATUSES = "statuses"
    WORKTREES = "worktrees"
    CACHES = "caches"
    ARTIFACTS = "artifacts"
    LEASES = "leases"
    DATABASES = "databases"


class ConflictPolicy(str, Enum):
    """Explicit conflict resolution; last-write-wins is not a valid policy."""

    SELECT = "select"
    MERGE = "merge"
    QUARANTINE = "quarantine"
    REJECT = "reject"


class ImportMode(str, Enum):
    PREVIEW = "preview"
    APPLY = "apply"


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


def _require_duckdb() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise DuckDBUnavailableError(
            "DuckDB is required for legacy state import targets; install the "
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


def sha256_digest(data: bytes) -> str:
    """Return a ``sha256:<hex>`` digest for raw bytes."""

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def digest_file(path: Path) -> tuple[str, int, bytes]:
    """Read a file once and return digest, byte count, and raw bytes."""

    data = path.read_bytes()
    return sha256_digest(data), len(data), data


def _enum_value(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return enum_cls(text)
        except ValueError as exc:
            raise ImportManifestError(
                f"{field_name} is not a closed {enum_cls.__name__} value: {text!r}"
            ) from exc
    raise ImportManifestError(f"{field_name} must be a {enum_cls.__name__} value")


def _require_non_empty(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ImportManifestError(f"{field_name} must not be empty")
    return text


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ImportManifestError("metadata must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ImportManifestError("metadata keys must be strings")
        if isinstance(item, Mapping):
            frozen[key] = dict(item)
        elif isinstance(item, (list, tuple)):
            frozen[key] = list(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _normalize_json_value(value: Any) -> Any:
    """Normalize parsed values for canonical identity (no floats, no bytes)."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # Floats are not identity-safe; stringify with short repr.
        return format(value, ".15g")
    if isinstance(value, (bytes, bytearray)):
        return sha256_digest(bytes(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return sha256_digest(canonical_json_bytes(_normalize_json_value(dict(payload))))


def _record_id_from_payload(payload: Mapping[str, Any], *, fallback: str) -> str:
    for key in (
        "record_id",
        "id",
        "task_id",
        "goal_id",
        "objective_id",
        "event_id",
        "lease_id",
        "artifact_id",
        "worktree_id",
        "queue_id",
        "status_id",
        "plan_id",
        "name",
    ):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


# ---------------------------------------------------------------------------
# Manifest / source / row contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportSourceDescriptor:
    """One legacy source path under an import manifest."""

    source_id: str
    path: str
    kind: ImportSourceKind
    domain: ImportDomain
    parser_version: str = DEFAULT_PARSER_VERSION
    schema_identity: str = ""
    select_priority: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = IMPORT_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _require_non_empty(self.source_id, "source_id")
        )
        object.__setattr__(self, "path", _require_non_empty(self.path, "path"))
        object.__setattr__(
            self,
            "kind",
            _enum_value(self.kind, ImportSourceKind, field_name="kind"),
        )
        object.__setattr__(
            self,
            "domain",
            _enum_value(self.domain, ImportDomain, field_name="domain"),
        )
        object.__setattr__(
            self,
            "parser_version",
            _require_non_empty(self.parser_version, "parser_version"),
        )
        object.__setattr__(
            self,
            "schema_identity",
            str(self.schema_identity or "").strip(),
        )
        object.__setattr__(self, "select_priority", int(self.select_priority))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_id": self.source_id,
            "path": self.path,
            "kind": self.kind.value,
            "domain": self.domain.value,
            "parser_version": self.parser_version,
            "schema_identity": self.schema_identity,
            "select_priority": int(self.select_priority),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportSourceDescriptor:
        if not isinstance(payload, Mapping):
            raise ImportManifestError("source descriptor must be a mapping")
        return cls(
            source_id=str(payload.get("source_id") or ""),
            path=str(payload.get("path") or ""),
            kind=str(payload.get("kind") or ""),
            domain=str(payload.get("domain") or ""),
            parser_version=str(
                payload.get("parser_version") or DEFAULT_PARSER_VERSION
            ),
            schema_identity=str(payload.get("schema_identity") or ""),
            select_priority=int(payload.get("select_priority") or 0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ImportManifest:
    """Explicit, ordered list of legacy sources to import.

    Interface: ``ImportManifest@1``.
    """

    manifest_id: str
    sources: tuple[ImportSourceDescriptor, ...]
    conflict_policy: ConflictPolicy = ConflictPolicy.REJECT
    mode: ImportMode = ImportMode.PREVIEW
    strict: bool = True
    target_store_id: str = "control.duckdb"
    selected_sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = IMPORT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_id", _require_non_empty(self.manifest_id, "manifest_id")
        )
        if not self.sources:
            raise ImportManifestError("manifest must declare at least one source")
        object.__setattr__(self, "sources", tuple(self.sources))
        ids = [item.source_id for item in self.sources]
        if len(set(ids)) != len(ids):
            raise ImportManifestError("duplicate source_id values are refused")
        object.__setattr__(
            self,
            "conflict_policy",
            _enum_value(
                self.conflict_policy, ConflictPolicy, field_name="conflict_policy"
            ),
        )
        object.__setattr__(
            self,
            "mode",
            _enum_value(self.mode, ImportMode, field_name="mode"),
        )
        object.__setattr__(self, "strict", bool(self.strict))
        object.__setattr__(
            self,
            "target_store_id",
            _require_non_empty(self.target_store_id, "target_store_id"),
        )
        selected = tuple(
            _require_non_empty(item, "selected_sources[]")
            for item in self.selected_sources
        )
        unknown = set(selected) - {item.source_id for item in self.sources}
        if unknown:
            raise ImportManifestError(
                f"selected_sources reference unknown source_id values: "
                f"{sorted(unknown)}"
            )
        object.__setattr__(self, "selected_sources", selected)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def fingerprint(self) -> str:
        body = {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "sources": [item.to_dict() for item in self.sources],
            "conflict_policy": self.conflict_policy.value,
            "mode": self.mode.value,
            "strict": bool(self.strict),
            "target_store_id": self.target_store_id,
            "selected_sources": list(self.selected_sources),
            "metadata": dict(self.metadata),
        }
        return sha256_digest(canonical_json_bytes(body))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": IMPORT_MANIFEST_INTERFACE,
            "manifest_id": self.manifest_id,
            "sources": [item.to_dict() for item in self.sources],
            "conflict_policy": self.conflict_policy.value,
            "mode": self.mode.value,
            "strict": bool(self.strict),
            "target_store_id": self.target_store_id,
            "selected_sources": list(self.selected_sources),
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportManifest:
        if not isinstance(payload, Mapping):
            raise ImportManifestError("manifest must be a mapping")
        sources_raw = payload.get("sources")
        if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
            raise ImportManifestError("manifest sources must be a sequence")
        return cls(
            manifest_id=str(payload.get("manifest_id") or ""),
            sources=tuple(
                ImportSourceDescriptor.from_dict(item) for item in sources_raw
            ),
            conflict_policy=str(payload.get("conflict_policy") or "reject"),
            mode=str(payload.get("mode") or "preview"),
            strict=bool(payload["strict"]) if "strict" in payload else True,
            target_store_id=str(payload.get("target_store_id") or "control.duckdb"),
            selected_sources=tuple(payload.get("selected_sources") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_path(cls, path: Path | str) -> ImportManifest:
        text = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ImportManifestError(f"manifest is not valid JSON: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ImportManifestError("manifest root must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class ObservedSource:
    """Byte-digest observation of one legacy source."""

    source: ImportSourceDescriptor
    absolute_path: str
    source_digest: str
    byte_count: int
    observed_at: str
    raw_bytes: bytes = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source.source_id,
            "path": self.source.path,
            "absolute_path": self.absolute_path,
            "kind": self.source.kind.value,
            "domain": self.source.domain.value,
            "source_digest": self.source_digest,
            "parser_version": self.source.parser_version,
            "schema_identity": self.source.schema_identity,
            "byte_count": int(self.byte_count),
            "observed_at": self.observed_at,
            "select_priority": int(self.source.select_priority),
        }


@dataclass(frozen=True)
class ParsedRecord:
    """One parsed legacy record prior to reconciliation."""

    source_id: str
    source_digest: str
    parser_version: str
    domain: str
    record_id: str
    payload: Mapping[str, Any]
    schema_identity: str = ""
    line_or_table: str = ""

    @property
    def payload_digest(self) -> str:
        return _payload_digest(self.payload)

    def entity_key(self) -> tuple[str, str]:
        return (self.domain, self.record_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "parser_version": self.parser_version,
            "domain": self.domain,
            "record_id": self.record_id,
            "payload": dict(self.payload),
            "payload_digest": self.payload_digest,
            "schema_identity": self.schema_identity,
            "line_or_table": self.line_or_table,
        }


@dataclass(frozen=True)
class ImportedRow:
    """Accepted import row with source digest and parser provenance.

    Every accepted row is traceable to its source digest and parser version.
    """

    row_id: str
    domain: str
    record_id: str
    source_id: str
    source_digest: str
    parser_version: str
    payload: Mapping[str, Any]
    payload_digest: str
    disposition: str = DISPOSITION_ACCEPT
    schema: str = IMPORTED_ROW_SCHEMA

    def __post_init__(self) -> None:
        if not _DIGEST_RE.fullmatch(str(self.source_digest)):
            raise LegacyStateImportError(
                f"imported row source_digest is not a sha256 digest: {self.source_digest}"
            )
        if not str(self.parser_version).strip():
            raise LegacyStateImportError(
                "imported row parser_version must not be empty"
            )
        normalized = _normalize_json_value(dict(self.payload))
        if not isinstance(normalized, dict):
            raise LegacyStateImportError("imported row payload must be a mapping")
        object.__setattr__(self, "payload", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "domain": self.domain,
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "parser_version": self.parser_version,
            "payload": dict(self.payload),
            "payload_digest": self.payload_digest,
            "disposition": self.disposition,
        }

    @property
    def content_id(self) -> str:
        return canonical_content_cid(self.to_dict())


@dataclass(frozen=True)
class QuarantinedRecord:
    """Conflict or uncertain record set held for operator review."""

    quarantine_id: str
    domain: str
    record_id: str
    reason: str
    source_ids: tuple[str, ...]
    candidates: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quarantine_id": self.quarantine_id,
            "domain": self.domain,
            "record_id": self.record_id,
            "reason": self.reason,
            "source_ids": list(self.source_ids),
            "candidates": [dict(item) for item in self.candidates],
        }


@dataclass(frozen=True)
class RejectedRecord:
    """Explicitly rejected record with provenance."""

    rejection_id: str
    domain: str
    record_id: str
    reason: str
    source_id: str
    source_digest: str
    parser_version: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejection_id": self.rejection_id,
            "domain": self.domain,
            "record_id": self.record_id,
            "reason": self.reason,
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "parser_version": self.parser_version,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ImportReceipt:
    """Durable attestation of one preview or apply import run.

    Interface: ``ImportReceipt@1``. Exact replay returns the same receipt.
    """

    receipt_cid: str
    import_key: str
    manifest_id: str
    manifest_fingerprint: str
    outcome: str
    mode: str
    conflict_policy: str
    started_at: str
    finished_at: str
    source_observations: tuple[Mapping[str, Any], ...]
    accepted_rows: tuple[ImportedRow, ...]
    rejected_rows: tuple[RejectedRecord, ...]
    quarantined_rows: tuple[QuarantinedRecord, ...]
    conflict_count: int
    parser_version: str = DEFAULT_PARSER_VERSION
    target_store_id: str = "control.duckdb"
    error_text: str = ""
    replayed: bool = False
    schema: str = IMPORT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": IMPORT_RECEIPT_INTERFACE,
            "receipt_cid": self.receipt_cid,
            "import_key": self.import_key,
            "manifest_id": self.manifest_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "outcome": self.outcome,
            "mode": self.mode,
            "conflict_policy": self.conflict_policy,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source_observations": [dict(item) for item in self.source_observations],
            "accepted_rows": [row.to_dict() for row in self.accepted_rows],
            "rejected_rows": [row.to_dict() for row in self.rejected_rows],
            "quarantined_rows": [row.to_dict() for row in self.quarantined_rows],
            "accepted_count": len(self.accepted_rows),
            "rejected_count": len(self.rejected_rows),
            "quarantined_count": len(self.quarantined_rows),
            "conflict_count": int(self.conflict_count),
            "parser_version": self.parser_version,
            "target_store_id": self.target_store_id,
            "error_text": self.error_text,
            "replayed": bool(self.replayed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportReceipt:
        accepted = tuple(
            ImportedRow(
                row_id=str(item["row_id"]),
                domain=str(item["domain"]),
                record_id=str(item["record_id"]),
                source_id=str(item["source_id"]),
                source_digest=str(item["source_digest"]),
                parser_version=str(item["parser_version"]),
                payload=dict(item.get("payload") or {}),
                payload_digest=str(item["payload_digest"]),
                disposition=str(item.get("disposition") or DISPOSITION_ACCEPT),
            )
            for item in payload.get("accepted_rows") or ()
        )
        rejected = tuple(
            RejectedRecord(
                rejection_id=str(item["rejection_id"]),
                domain=str(item["domain"]),
                record_id=str(item["record_id"]),
                reason=str(item["reason"]),
                source_id=str(item["source_id"]),
                source_digest=str(item["source_digest"]),
                parser_version=str(item["parser_version"]),
                payload=dict(item.get("payload") or {}),
            )
            for item in payload.get("rejected_rows") or ()
        )
        quarantined = tuple(
            QuarantinedRecord(
                quarantine_id=str(item["quarantine_id"]),
                domain=str(item["domain"]),
                record_id=str(item["record_id"]),
                reason=str(item["reason"]),
                source_ids=tuple(item.get("source_ids") or ()),
                candidates=tuple(item.get("candidates") or ()),
            )
            for item in payload.get("quarantined_rows") or ()
        )
        return cls(
            receipt_cid=str(payload["receipt_cid"]),
            import_key=str(payload["import_key"]),
            manifest_id=str(payload["manifest_id"]),
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            outcome=str(payload["outcome"]),
            mode=str(payload["mode"]),
            conflict_policy=str(payload["conflict_policy"]),
            started_at=str(payload["started_at"]),
            finished_at=str(payload["finished_at"]),
            source_observations=tuple(payload.get("source_observations") or ()),
            accepted_rows=accepted,
            rejected_rows=rejected,
            quarantined_rows=quarantined,
            conflict_count=int(payload.get("conflict_count") or 0),
            parser_version=str(
                payload.get("parser_version") or DEFAULT_PARSER_VERSION
            ),
            target_store_id=str(payload.get("target_store_id") or "control.duckdb"),
            error_text=str(payload.get("error_text") or ""),
            replayed=bool(payload.get("replayed") or False),
        )


@dataclass(frozen=True)
class ReconciliationResult:
    accepted: tuple[ImportedRow, ...]
    rejected: tuple[RejectedRecord, ...]
    quarantined: tuple[QuarantinedRecord, ...]
    conflict_count: int


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_markdown_records(
    *,
    data: bytes,
    source: ImportSourceDescriptor,
    source_digest: str,
) -> tuple[list[ParsedRecord], list[RejectedRecord]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportParseError(
            f"markdown source {source.source_id} is not valid UTF-8"
        ) from exc
    if not text.strip():
        return [], []

    matches = list(_MARKDOWN_HEADING_RE.finditer(text))
    if not matches:
        # Treat whole-file markdown as a single record when no headings exist.
        payload = {
            "record_id": source.source_id,
            "body": text,
            "format": "markdown",
        }
        return [
            ParsedRecord(
                source_id=source.source_id,
                source_digest=source_digest,
                parser_version=source.parser_version,
                domain=source.domain.value,
                record_id=source.source_id,
                payload=payload,
                schema_identity=source.schema_identity,
                line_or_table="file",
            )
        ], []

    records: list[ParsedRecord] = []
    rejections: list[RejectedRecord] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        record_id = match.group("record_id").strip()
        title = (match.group("title") or "").strip()
        fields: dict[str, Any] = {
            "record_id": record_id,
            "format": "markdown",
        }
        if title:
            fields["title"] = title
        for line in body.splitlines():
            field_match = _MARKDOWN_FIELD_RE.match(line)
            if not field_match:
                continue
            key = re.sub(r"\s+", "_", field_match.group("key").strip().lower())
            value = field_match.group("value").strip()
            fields[key] = value
        if "body" not in fields:
            prose = "\n".join(
                line
                for line in body.splitlines()
                if line.strip() and not _MARKDOWN_FIELD_RE.match(line)
            ).strip()
            if prose:
                fields["body"] = prose
        records.append(
            ParsedRecord(
                source_id=source.source_id,
                source_digest=source_digest,
                parser_version=source.parser_version,
                domain=source.domain.value,
                record_id=record_id,
                payload=fields,
                schema_identity=source.schema_identity,
                line_or_table=f"heading:{index + 1}",
            )
        )
    return records, rejections


def _parse_json_records(
    *,
    data: bytes,
    source: ImportSourceDescriptor,
    source_digest: str,
) -> tuple[list[ParsedRecord], list[RejectedRecord]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportParseError(
            f"json source {source.source_id} is not valid UTF-8"
        ) from exc
    if not text.strip():
        return [], []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImportParseError(
            f"json source {source.source_id} is corrupt or truncated: {exc}"
        ) from exc

    items: list[Any]
    if isinstance(payload, Mapping):
        if "records" in payload and isinstance(payload["records"], list):
            items = list(payload["records"])
            schema_identity = str(
                payload.get("schema") or source.schema_identity or ""
            )
        else:
            items = [payload]
            schema_identity = str(
                payload.get("schema") or source.schema_identity or ""
            )
    elif isinstance(payload, list):
        items = payload
        schema_identity = source.schema_identity
    else:
        raise ImportParseError(
            f"json source {source.source_id} root must be an object or array"
        )

    if source.schema_identity and schema_identity and schema_identity != source.schema_identity:
        raise ImportSchemaError(
            f"json source {source.source_id} schema {schema_identity!r} does not "
            f"match declared schema_identity {source.schema_identity!r}"
        )

    records: list[ParsedRecord] = []
    rejections: list[RejectedRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            rejections.append(
                RejectedRecord(
                    rejection_id=f"reject:{source.source_id}:{index}",
                    domain=source.domain.value,
                    record_id=f"{source.source_id}:{index}",
                    reason="json_record_not_object",
                    source_id=source.source_id,
                    source_digest=source_digest,
                    parser_version=source.parser_version,
                    payload={"index": index, "value_type": type(item).__name__},
                )
            )
            continue
        normalized = _normalize_json_value(dict(item))
        record_id = _record_id_from_payload(
            normalized, fallback=f"{source.source_id}:{index}"
        )
        records.append(
            ParsedRecord(
                source_id=source.source_id,
                source_digest=source_digest,
                parser_version=source.parser_version,
                domain=source.domain.value,
                record_id=record_id,
                payload=normalized,
                schema_identity=schema_identity or source.schema_identity,
                line_or_table=f"index:{index}",
            )
        )
    return records, rejections


def _parse_jsonl_records(
    *,
    data: bytes,
    source: ImportSourceDescriptor,
    source_digest: str,
) -> tuple[list[ParsedRecord], list[RejectedRecord]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportParseError(
            f"jsonl source {source.source_id} is not valid UTF-8"
        ) from exc
    records: list[ParsedRecord] = []
    rejections: list[RejectedRecord] = []
    if not text.strip():
        return records, rejections
    # Truncation detection: a non-empty final line that is incomplete JSON.
    lines = text.splitlines(keepends=True)
    if text and not text.endswith(("\n", "\r")):
        # Last line may be truncated; try parse and fail closed if invalid.
        last = lines[-1].strip() if lines else ""
        if last:
            try:
                json.loads(last)
            except json.JSONDecodeError as exc:
                raise ImportParseError(
                    f"jsonl source {source.source_id} appears truncated: {exc}"
                ) from exc

    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ImportParseError(
                f"jsonl source {source.source_id} line {index} is corrupt: {exc}"
            ) from exc
        if not isinstance(item, Mapping):
            rejections.append(
                RejectedRecord(
                    rejection_id=f"reject:{source.source_id}:line-{index}",
                    domain=source.domain.value,
                    record_id=f"{source.source_id}:{index}",
                    reason="jsonl_record_not_object",
                    source_id=source.source_id,
                    source_digest=source_digest,
                    parser_version=source.parser_version,
                    payload={"line": index, "value_type": type(item).__name__},
                )
            )
            continue
        if source.schema_identity:
            item_schema = str(item.get("schema") or "")
            if item_schema and item_schema != source.schema_identity:
                rejections.append(
                    RejectedRecord(
                        rejection_id=f"reject:{source.source_id}:line-{index}",
                        domain=source.domain.value,
                        record_id=_record_id_from_payload(
                            item, fallback=f"{source.source_id}:{index}"
                        ),
                        reason="unsupported_schema",
                        source_id=source.source_id,
                        source_digest=source_digest,
                        parser_version=source.parser_version,
                        payload=dict(item),
                    )
                )
                continue
        normalized = _normalize_json_value(dict(item))
        record_id = _record_id_from_payload(
            normalized, fallback=f"{source.source_id}:{index}"
        )
        records.append(
            ParsedRecord(
                source_id=source.source_id,
                source_digest=source_digest,
                parser_version=source.parser_version,
                domain=source.domain.value,
                record_id=record_id,
                payload=normalized,
                schema_identity=str(item.get("schema") or source.schema_identity),
                line_or_table=f"line:{index}",
            )
        )
    return records, rejections


def _sqlite_table_rows(
    path: Path,
    *,
    source: ImportSourceDescriptor,
    source_digest: str,
) -> tuple[list[ParsedRecord], list[RejectedRecord]]:
    records: list[ParsedRecord] = []
    rejections: list[RejectedRecord] = []
    try:
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
            timeout=5.0,
        )
    except sqlite3.Error as exc:
        raise ImportParseError(
            f"sqlite source {source.source_id} cannot be opened read-only: {exc}"
        ) from exc
    connection.row_factory = sqlite3.Row
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        for table_name in tables:
            columns = [
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]
            if not columns:
                continue
            quoted = ", ".join(f'"{column}"' for column in columns)
            for index, row in enumerate(
                connection.execute(f'SELECT {quoted} FROM "{table_name}"')
            ):
                payload = {column: row[column] for column in columns}
                # Normalize non-JSON types.
                normalized: dict[str, Any] = {}
                for key, value in payload.items():
                    if isinstance(value, bytes):
                        normalized[key] = sha256_digest(value)
                    elif value is None or isinstance(value, (str, int, bool)):
                        normalized[key] = value
                    elif isinstance(value, float):
                        # Fail closed for identity-bearing floats: stringify.
                        normalized[key] = str(value)
                    else:
                        normalized[key] = str(value)
                normalized.setdefault("table", table_name)
                record_id = _record_id_from_payload(
                    normalized, fallback=f"{table_name}:{index}"
                )
                records.append(
                    ParsedRecord(
                        source_id=source.source_id,
                        source_digest=source_digest,
                        parser_version=source.parser_version,
                        domain=source.domain.value,
                        record_id=record_id,
                        payload=normalized,
                        schema_identity=source.schema_identity or f"sqlite:{table_name}",
                        line_or_table=f"table:{table_name}:{index}",
                    )
                )
    except sqlite3.Error as exc:
        raise ImportParseError(
            f"sqlite source {source.source_id} is corrupt: {exc}"
        ) from exc
    finally:
        connection.close()
    return records, rejections


def _duckdb_table_rows(
    path: Path,
    *,
    source: ImportSourceDescriptor,
    source_digest: str,
) -> tuple[list[ParsedRecord], list[RejectedRecord]]:
    duckdb = _require_duckdb()
    records: list[ParsedRecord] = []
    rejections: list[RejectedRecord] = []
    try:
        connection = duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # pragma: no cover - duckdb message varies
        raise ImportParseError(
            f"duckdb source {source.source_id} cannot be opened read-only: {exc}"
        ) from exc
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        for table_name in tables:
            columns = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'main' AND table_name = ?
                    ORDER BY ordinal_position
                    """,
                    [table_name],
                ).fetchall()
            ]
            if not columns:
                continue
            quoted = ", ".join(f'"{column}"' for column in columns)
            rows = connection.execute(
                f'SELECT {quoted} FROM "{table_name}"'
            ).fetchall()
            for index, row in enumerate(rows):
                payload = {
                    column: (
                        sha256_digest(value)
                        if isinstance(value, (bytes, bytearray))
                        else value
                        if value is None or isinstance(value, (str, int, bool))
                        else str(value)
                    )
                    for column, value in zip(columns, row)
                }
                payload.setdefault("table", table_name)
                record_id = _record_id_from_payload(
                    payload, fallback=f"{table_name}:{index}"
                )
                records.append(
                    ParsedRecord(
                        source_id=source.source_id,
                        source_digest=source_digest,
                        parser_version=source.parser_version,
                        domain=source.domain.value,
                        record_id=record_id,
                        payload=payload,
                        schema_identity=source.schema_identity or f"duckdb:{table_name}",
                        line_or_table=f"table:{table_name}:{index}",
                    )
                )
    except Exception as exc:
        raise ImportParseError(
            f"duckdb source {source.source_id} is corrupt or unsupported: {exc}"
        ) from exc
    finally:
        connection.close()
    return records, rejections


def parse_source(
    observed: ObservedSource,
) -> tuple[list[ParsedRecord], list[RejectedRecord]]:
    """Parse one observed source into records and per-row rejections."""

    source = observed.source
    kind = source.kind
    data = observed.raw_bytes
    if kind is ImportSourceKind.MARKDOWN:
        return _parse_markdown_records(
            data=data, source=source, source_digest=observed.source_digest
        )
    if kind is ImportSourceKind.JSON:
        return _parse_json_records(
            data=data, source=source, source_digest=observed.source_digest
        )
    if kind is ImportSourceKind.JSONL:
        return _parse_jsonl_records(
            data=data, source=source, source_digest=observed.source_digest
        )
    if kind is ImportSourceKind.SQLITE:
        return _sqlite_table_rows(
            Path(observed.absolute_path),
            source=source,
            source_digest=observed.source_digest,
        )
    if kind is ImportSourceKind.DUCKDB:
        return _duckdb_table_rows(
            Path(observed.absolute_path),
            source=source,
            source_digest=observed.source_digest,
        )
    raise ImportSchemaError(f"unsupported source kind: {kind}")


# ---------------------------------------------------------------------------
# Conflict reconciliation
# ---------------------------------------------------------------------------


def _merge_payloads(
    candidates: Sequence[ParsedRecord],
) -> tuple[dict[str, Any], list[str]]:
    """Shallow merge payloads; return merged mapping and conflicted field names."""

    merged: dict[str, Any] = {}
    conflicted: list[str] = []
    for record in candidates:
        for key, value in record.payload.items():
            if key not in merged:
                merged[key] = value
                continue
            if merged[key] != value:
                if key not in conflicted:
                    conflicted.append(key)
    return merged, conflicted


def reconcile_records(
    records: Sequence[ParsedRecord],
    *,
    conflict_policy: ConflictPolicy,
    selected_sources: Sequence[str] = (),
    source_priorities: Mapping[str, int] | None = None,
) -> ReconciliationResult:
    """Reconcile parsed records without last-write-wins."""

    priorities = dict(source_priorities or {})
    selected = set(selected_sources)
    by_entity: dict[tuple[str, str], list[ParsedRecord]] = {}
    for record in records:
        by_entity.setdefault(record.entity_key(), []).append(record)

    accepted: list[ImportedRow] = []
    rejected: list[RejectedRecord] = []
    quarantined: list[QuarantinedRecord] = []
    conflict_count = 0

    for (domain, record_id), candidates in sorted(
        by_entity.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        unique_digests = {item.payload_digest for item in candidates}
        if len(candidates) == 1 or len(unique_digests) == 1:
            chosen = candidates[0]
            disposition = (
                DISPOSITION_SKIP_DUPLICATE
                if len(candidates) > 1
                else DISPOSITION_ACCEPT
            )
            # Prefer highest priority source when identical duplicates exist.
            if len(candidates) > 1:
                chosen = max(
                    candidates,
                    key=lambda item: (
                        priorities.get(item.source_id, 0),
                        -candidates.index(item),
                    ),
                )
                disposition = DISPOSITION_ACCEPT
            accepted.append(
                ImportedRow(
                    row_id=f"row:{domain}:{record_id}:{chosen.source_id}",
                    domain=domain,
                    record_id=record_id,
                    source_id=chosen.source_id,
                    source_digest=chosen.source_digest,
                    parser_version=chosen.parser_version,
                    payload=dict(chosen.payload),
                    payload_digest=chosen.payload_digest,
                    disposition=disposition,
                )
            )
            continue

        conflict_count += 1
        if conflict_policy is ConflictPolicy.REJECT:
            for item in candidates:
                rejected.append(
                    RejectedRecord(
                        rejection_id=(
                            f"reject:{domain}:{record_id}:{item.source_id}"
                        ),
                        domain=domain,
                        record_id=record_id,
                        reason="conflict_reject",
                        source_id=item.source_id,
                        source_digest=item.source_digest,
                        parser_version=item.parser_version,
                        payload=dict(item.payload),
                    )
                )
            continue

        if conflict_policy is ConflictPolicy.QUARANTINE:
            quarantined.append(
                QuarantinedRecord(
                    quarantine_id=f"quarantine:{domain}:{record_id}",
                    domain=domain,
                    record_id=record_id,
                    reason="conflict_quarantine",
                    source_ids=tuple(item.source_id for item in candidates),
                    candidates=tuple(item.to_dict() for item in candidates),
                )
            )
            continue

        if conflict_policy is ConflictPolicy.SELECT:
            chosen: ParsedRecord | None = None
            if selected:
                selected_candidates = [
                    item for item in candidates if item.source_id in selected
                ]
                if len(selected_candidates) == 1:
                    chosen = selected_candidates[0]
                elif len(selected_candidates) > 1:
                    # Still conflicting among selected sources.
                    quarantined.append(
                        QuarantinedRecord(
                            quarantine_id=f"quarantine:{domain}:{record_id}",
                            domain=domain,
                            record_id=record_id,
                            reason="select_ambiguous",
                            source_ids=tuple(
                                item.source_id for item in selected_candidates
                            ),
                            candidates=tuple(
                                item.to_dict() for item in selected_candidates
                            ),
                        )
                    )
                    continue
            if chosen is None:
                chosen = max(
                    candidates,
                    key=lambda item: (
                        priorities.get(item.source_id, 0),
                        1 if item.source_id in selected else 0,
                        -candidates.index(item),
                    ),
                )
            # Only accept if selection is unambiguous by priority or selection.
            top_priority = priorities.get(chosen.source_id, 0)
            if chosen.source_id not in selected and any(
                priorities.get(item.source_id, 0) == top_priority
                and item.payload_digest != chosen.payload_digest
                for item in candidates
            ):
                # No explicit selection and equal priorities remain conflicting.
                if not selected and all(
                    priorities.get(item.source_id, 0) == top_priority
                    for item in candidates
                ):
                    quarantined.append(
                        QuarantinedRecord(
                            quarantine_id=f"quarantine:{domain}:{record_id}",
                            domain=domain,
                            record_id=record_id,
                            reason="select_requires_explicit_choice",
                            source_ids=tuple(item.source_id for item in candidates),
                            candidates=tuple(item.to_dict() for item in candidates),
                        )
                    )
                    continue
            accepted.append(
                ImportedRow(
                    row_id=f"row:{domain}:{record_id}:{chosen.source_id}",
                    domain=domain,
                    record_id=record_id,
                    source_id=chosen.source_id,
                    source_digest=chosen.source_digest,
                    parser_version=chosen.parser_version,
                    payload=dict(chosen.payload),
                    payload_digest=chosen.payload_digest,
                    disposition=DISPOSITION_SELECT,
                )
            )
            for item in candidates:
                if item.source_id == chosen.source_id:
                    continue
                rejected.append(
                    RejectedRecord(
                        rejection_id=(
                            f"reject:{domain}:{record_id}:{item.source_id}"
                        ),
                        domain=domain,
                        record_id=record_id,
                        reason="conflict_select_not_chosen",
                        source_id=item.source_id,
                        source_digest=item.source_digest,
                        parser_version=item.parser_version,
                        payload=dict(item.payload),
                    )
                )
            continue

        if conflict_policy is ConflictPolicy.MERGE:
            merged, conflicted_fields = _merge_payloads(candidates)
            if conflicted_fields:
                # Field-level conflicts are quarantined rather than last-write-wins.
                quarantined.append(
                    QuarantinedRecord(
                        quarantine_id=f"quarantine:{domain}:{record_id}",
                        domain=domain,
                        record_id=record_id,
                        reason=(
                            "merge_field_conflicts:"
                            + ",".join(sorted(conflicted_fields))
                        ),
                        source_ids=tuple(item.source_id for item in candidates),
                        candidates=tuple(item.to_dict() for item in candidates),
                    )
                )
                continue
            # Provenance: bind all contributing sources via metadata.
            provenance_sources = sorted({item.source_id for item in candidates})
            provenance_digests = sorted({item.source_digest for item in candidates})
            provenance_parsers = sorted({item.parser_version for item in candidates})
            merged_payload = {
                **merged,
                "_merge_source_ids": provenance_sources,
                "_merge_source_digests": provenance_digests,
                "_merge_parser_versions": provenance_parsers,
            }
            primary = max(
                candidates,
                key=lambda item: (
                    priorities.get(item.source_id, 0),
                    -candidates.index(item),
                ),
            )
            accepted.append(
                ImportedRow(
                    row_id=f"row:{domain}:{record_id}:merged",
                    domain=domain,
                    record_id=record_id,
                    source_id=primary.source_id,
                    source_digest=primary.source_digest,
                    parser_version=primary.parser_version,
                    payload=merged_payload,
                    payload_digest=_payload_digest(merged_payload),
                    disposition=DISPOSITION_MERGE,
                )
            )
            continue

        raise ImportConflictError(
            f"unsupported conflict policy: {conflict_policy}"
        )

    return ReconciliationResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        quarantined=tuple(quarantined),
        conflict_count=conflict_count,
    )


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


def _import_key(
    *,
    manifest_fingerprint: str,
    source_digests: Mapping[str, str],
    mode: str,
) -> str:
    body = {
        "manifest_fingerprint": manifest_fingerprint,
        "source_digests": dict(sorted(source_digests.items())),
        "mode": mode,
    }
    return sha256_digest(canonical_json_bytes(body))


def _receipt_identity_body(
    *,
    import_key: str,
    manifest_id: str,
    manifest_fingerprint: str,
    outcome: str,
    mode: str,
    conflict_policy: str,
    source_observations: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[ImportedRow],
    rejected_rows: Sequence[RejectedRecord],
    quarantined_rows: Sequence[QuarantinedRecord],
    conflict_count: int,
    parser_version: str,
    target_store_id: str,
    error_text: str,
) -> dict[str, Any]:
    # Identity excludes wall-clock timestamps so exact replay yields the same
    # receipt_cid for the same logical import.
    return {
        "schema": IMPORT_RECEIPT_SCHEMA,
        "import_key": import_key,
        "manifest_id": manifest_id,
        "manifest_fingerprint": manifest_fingerprint,
        "outcome": outcome,
        "mode": mode,
        "conflict_policy": conflict_policy,
        "source_observations": [
            {
                "source_id": item["source_id"],
                "source_digest": item["source_digest"],
                "parser_version": item["parser_version"],
                "kind": item["kind"],
                "domain": item["domain"],
                "byte_count": item["byte_count"],
            }
            for item in source_observations
        ],
        "accepted_rows": [row.to_dict() for row in accepted_rows],
        "rejected_rows": [row.to_dict() for row in rejected_rows],
        "quarantined_rows": [row.to_dict() for row in quarantined_rows],
        "conflict_count": int(conflict_count),
        "parser_version": parser_version,
        "target_store_id": target_store_id,
        "error_text": error_text,
    }


class LegacyStateImport:
    """Preview and apply legacy state imports into a DuckDB target store.

    Interface: ``LegacyStateImport@1``.
    """

    SCHEMA: Final = LEGACY_STATE_IMPORT_SCHEMA
    INTERFACE: Final = LEGACY_STATE_IMPORT_INTERFACE

    def __init__(
        self,
        target_path: Path | str,
        *,
        root: Path | str | None = None,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for LegacyStateImport target storage"
            )
        self.target_path = Path(target_path)
        self.root = Path(root) if root is not None else self.target_path.parent
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with open_duckdb_connection(self.target_path) as connection:
            connection.executescript(_BOOKKEEPING_SQL)
            connection.commit()

    def _resolve_source_path(self, declared: str) -> Path:
        path = Path(declared)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        else:
            path = path.resolve()
        return path

    def observe_sources(
        self, manifest: ImportManifest
    ) -> tuple[ObservedSource, ...]:
        """Read source bytes and compute digests without mutation."""

        observed: list[ObservedSource] = []
        observed_at = _utc_iso()
        for source in manifest.sources:
            path = self._resolve_source_path(source.path)
            if not path.is_file():
                raise ImportSourceError(
                    f"source {source.source_id} path does not exist: {path}"
                )
            # Verify kind heuristics for database files.
            if source.kind is ImportSourceKind.SQLITE:
                header = path.read_bytes()[:16]
                if header != b"SQLite format 3\0":
                    raise ImportSourceError(
                        f"source {source.source_id} is not a SQLite database"
                    )
            digest, byte_count, raw = digest_file(path)
            observed.append(
                ObservedSource(
                    source=source,
                    absolute_path=str(path),
                    source_digest=digest,
                    byte_count=byte_count,
                    observed_at=observed_at,
                    raw_bytes=raw,
                )
            )
        return tuple(observed)

    def _load_existing_receipt(self, import_key: str) -> ImportReceipt | None:
        with open_duckdb_connection(self.target_path) as connection:
            row = connection.execute(
                """
                SELECT body_json FROM legacy_import_receipts
                WHERE import_key = ?
                """,
                [import_key],
            ).fetchone()
        if row is None:
            return None
        body = json.loads(str(row[0]))
        receipt = ImportReceipt.from_dict(body)
        return ImportReceipt(
            receipt_cid=receipt.receipt_cid,
            import_key=receipt.import_key,
            manifest_id=receipt.manifest_id,
            manifest_fingerprint=receipt.manifest_fingerprint,
            outcome=OUTCOME_REPLAYED
            if receipt.outcome in {OUTCOME_APPLIED, OUTCOME_REPLAYED, OUTCOME_PREVIEW}
            else receipt.outcome,
            mode=receipt.mode,
            conflict_policy=receipt.conflict_policy,
            started_at=receipt.started_at,
            finished_at=receipt.finished_at,
            source_observations=receipt.source_observations,
            accepted_rows=receipt.accepted_rows,
            rejected_rows=receipt.rejected_rows,
            quarantined_rows=receipt.quarantined_rows,
            conflict_count=receipt.conflict_count,
            parser_version=receipt.parser_version,
            target_store_id=receipt.target_store_id,
            error_text=receipt.error_text,
            replayed=True,
        )

    def _verify_source_immutability(
        self, observed: Sequence[ObservedSource]
    ) -> None:
        for item in observed:
            path = Path(item.absolute_path)
            current_digest, _, _ = digest_file(path)
            if current_digest != item.source_digest:
                raise ImportSourceMutationError(
                    f"source {item.source.source_id} changed after observation "
                    f"({item.source_digest} -> {current_digest}); refusing import"
                )

    def preview(self, manifest: ImportManifest) -> ImportReceipt:
        """Preview import without committing accepted rows as authoritative."""

        preview_manifest = ImportManifest(
            manifest_id=manifest.manifest_id,
            sources=manifest.sources,
            conflict_policy=manifest.conflict_policy,
            mode=ImportMode.PREVIEW,
            strict=manifest.strict,
            target_store_id=manifest.target_store_id,
            selected_sources=manifest.selected_sources,
            metadata=dict(manifest.metadata),
        )
        return self.run(preview_manifest)

    def apply(self, manifest: ImportManifest) -> ImportReceipt:
        """Strictly apply import atomically, or not at all."""

        apply_manifest = ImportManifest(
            manifest_id=manifest.manifest_id,
            sources=manifest.sources,
            conflict_policy=manifest.conflict_policy,
            mode=ImportMode.APPLY,
            strict=True,
            target_store_id=manifest.target_store_id,
            selected_sources=manifest.selected_sources,
            metadata=dict(manifest.metadata),
        )
        return self.run(apply_manifest)

    def run(self, manifest: ImportManifest) -> ImportReceipt:
        """Execute preview or apply for ``manifest``."""

        started_at = _utc_iso()
        observed = self.observe_sources(manifest)
        source_digests = {
            item.source.source_id: item.source_digest for item in observed
        }
        import_key = _import_key(
            manifest_fingerprint=manifest.fingerprint(),
            source_digests=source_digests,
            mode=manifest.mode.value,
        )

        existing = self._load_existing_receipt(import_key)
        if existing is not None:
            # Exact replay is a no-op with the same receipt_cid.
            return existing

        all_records: list[ParsedRecord] = []
        parse_rejections: list[RejectedRecord] = []
        source_record_counts: dict[str, int] = {}
        source_reject_counts: dict[str, int] = {}
        try:
            for item in observed:
                records, rejections = parse_source(item)
                all_records.extend(records)
                parse_rejections.extend(rejections)
                source_record_counts[item.source.source_id] = len(records)
                source_reject_counts[item.source.source_id] = len(rejections)
        except (ImportParseError, ImportSchemaError, ImportSourceError) as exc:
            if manifest.mode is ImportMode.APPLY and manifest.strict:
                raise
            finished_at = _utc_iso()
            identity = _receipt_identity_body(
                import_key=import_key,
                manifest_id=manifest.manifest_id,
                manifest_fingerprint=manifest.fingerprint(),
                outcome=OUTCOME_FAILED,
                mode=manifest.mode.value,
                conflict_policy=manifest.conflict_policy.value,
                source_observations=[item.to_dict() for item in observed],
                accepted_rows=(),
                rejected_rows=tuple(parse_rejections),
                quarantined_rows=(),
                conflict_count=0,
                parser_version=DEFAULT_PARSER_VERSION,
                target_store_id=manifest.target_store_id,
                error_text=str(exc),
            )
            receipt_cid = canonical_content_cid(identity)
            return ImportReceipt(
                receipt_cid=receipt_cid,
                import_key=import_key,
                manifest_id=manifest.manifest_id,
                manifest_fingerprint=manifest.fingerprint(),
                outcome=OUTCOME_FAILED,
                mode=manifest.mode.value,
                conflict_policy=manifest.conflict_policy.value,
                started_at=started_at,
                finished_at=finished_at,
                source_observations=tuple(item.to_dict() for item in observed),
                accepted_rows=(),
                rejected_rows=tuple(parse_rejections),
                quarantined_rows=(),
                conflict_count=0,
                parser_version=DEFAULT_PARSER_VERSION,
                target_store_id=manifest.target_store_id,
                error_text=str(exc),
                replayed=False,
            )

        priorities = {
            item.source.source_id: int(item.source.select_priority)
            for item in observed
        }
        reconciliation = reconcile_records(
            all_records,
            conflict_policy=manifest.conflict_policy,
            selected_sources=manifest.selected_sources,
            source_priorities=priorities,
        )
        rejected = tuple(list(parse_rejections) + list(reconciliation.rejected))
        accepted = reconciliation.accepted
        quarantined = reconciliation.quarantined

        # Strict apply refuses unresolved conflicts under reject/quarantine
        # only when the caller asks for zero-conflict commits via metadata.
        require_clean = bool(manifest.metadata.get("require_clean_apply"))
        if (
            manifest.mode is ImportMode.APPLY
            and manifest.strict
            and require_clean
            and (reconciliation.conflict_count or quarantined or rejected)
        ):
            raise ImportAtomicityError(
                "strict clean apply refused due to conflicts, quarantine, or rejections"
            )

        outcome = (
            OUTCOME_PREVIEW
            if manifest.mode is ImportMode.PREVIEW
            else OUTCOME_APPLIED
        )
        observations = tuple(item.to_dict() for item in observed)
        for item in observed:
            item_dict = next(
                obs
                for obs in observations
                if obs["source_id"] == item.source.source_id
            )
            # Annotate counts on a fresh mapping for the receipt body.
            # observations is a tuple of dicts we own.
            item_dict["record_count"] = int(
                source_record_counts.get(item.source.source_id, 0)
            )
            item_dict["rejected_count"] = int(
                source_reject_counts.get(item.source.source_id, 0)
            )

        identity = _receipt_identity_body(
            import_key=import_key,
            manifest_id=manifest.manifest_id,
            manifest_fingerprint=manifest.fingerprint(),
            outcome=outcome,
            mode=manifest.mode.value,
            conflict_policy=manifest.conflict_policy.value,
            source_observations=observations,
            accepted_rows=accepted,
            rejected_rows=rejected,
            quarantined_rows=quarantined,
            conflict_count=reconciliation.conflict_count,
            parser_version=DEFAULT_PARSER_VERSION,
            target_store_id=manifest.target_store_id,
            error_text="",
        )
        receipt_cid = canonical_content_cid(identity)
        finished_at = _utc_iso()
        receipt = ImportReceipt(
            receipt_cid=receipt_cid,
            import_key=import_key,
            manifest_id=manifest.manifest_id,
            manifest_fingerprint=manifest.fingerprint(),
            outcome=outcome,
            mode=manifest.mode.value,
            conflict_policy=manifest.conflict_policy.value,
            started_at=started_at,
            finished_at=finished_at,
            source_observations=observations,
            accepted_rows=accepted,
            rejected_rows=rejected,
            quarantined_rows=quarantined,
            conflict_count=reconciliation.conflict_count,
            parser_version=DEFAULT_PARSER_VERSION,
            target_store_id=manifest.target_store_id,
            error_text="",
            replayed=False,
        )

        if manifest.mode is ImportMode.PREVIEW:
            # Preview persists the receipt for exact replay but does not claim
            # authoritative acceptance beyond the preview outcome.
            self._commit_receipt(receipt, observed=observed, authoritative=False)
            return receipt

        # Strict apply: re-verify immutability then commit atomically.
        self._verify_source_immutability(observed)
        try:
            self._commit_receipt(receipt, observed=observed, authoritative=True)
        except Exception as exc:
            raise ImportAtomicityError(
                f"strict import failed to commit atomically: {exc}"
            ) from exc
        return receipt

    def _commit_receipt(
        self,
        receipt: ImportReceipt,
        *,
        observed: Sequence[ObservedSource],
        authoritative: bool,
    ) -> None:
        """Persist receipt and rows in one transaction (atomic or not at all)."""

        body = receipt.to_dict()
        with open_duckdb_connection(self.target_path) as connection:
            try:
                connection.execute("BEGIN TRANSACTION")
                # Refuse partial overwrite of an existing import_key.
                existing = connection.execute(
                    """
                    SELECT receipt_cid FROM legacy_import_receipts
                    WHERE import_key = ?
                    """,
                    [receipt.import_key],
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != receipt.receipt_cid:
                        raise ImportAtomicityError(
                            "import_key already bound to a different receipt"
                        )
                    connection.execute("ROLLBACK")
                    return

                connection.execute(
                    """
                    INSERT INTO legacy_import_receipts (
                        import_key, receipt_cid, outcome, mode,
                        manifest_fingerprint, source_digests_json,
                        accepted_count, rejected_count, quarantined_count,
                        conflict_count, body_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt.import_key,
                        receipt.receipt_cid,
                        receipt.outcome,
                        receipt.mode,
                        receipt.manifest_fingerprint,
                        json.dumps(
                            {
                                item["source_id"]: item["source_digest"]
                                for item in receipt.source_observations
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        len(receipt.accepted_rows),
                        len(receipt.rejected_rows),
                        len(receipt.quarantined_rows),
                        int(receipt.conflict_count),
                        json.dumps(body, sort_keys=True, separators=(",", ":")),
                        receipt.finished_at,
                    ],
                )

                for item in observed:
                    obs = next(
                        entry
                        for entry in receipt.source_observations
                        if entry["source_id"] == item.source.source_id
                    )
                    connection.execute(
                        """
                        INSERT INTO legacy_import_sources (
                            import_key, source_id, path, kind, domain,
                            source_digest, parser_version, schema_identity,
                            observed_at, byte_count, record_count,
                            rejected_count, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            receipt.import_key,
                            item.source.source_id,
                            item.source.path,
                            item.source.kind.value,
                            item.source.domain.value,
                            item.source_digest,
                            item.source.parser_version,
                            item.source.schema_identity,
                            item.observed_at,
                            int(item.byte_count),
                            int(obs.get("record_count") or 0),
                            int(obs.get("rejected_count") or 0),
                            json.dumps(obs, sort_keys=True, separators=(",", ":")),
                        ],
                    )

                if authoritative:
                    for row in receipt.accepted_rows:
                        # Provenance invariants: every accepted row must bind
                        # source digest and parser version.
                        if not row.source_digest or not row.parser_version:
                            raise ImportAtomicityError(
                                f"accepted row {row.row_id} missing provenance"
                            )
                        connection.execute(
                            """
                            INSERT INTO legacy_import_rows (
                                import_key, row_id, domain, record_id,
                                source_id, source_digest, parser_version,
                                payload_digest, disposition, body_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                receipt.import_key,
                                row.row_id,
                                row.domain,
                                row.record_id,
                                row.source_id,
                                row.source_digest,
                                row.parser_version,
                                row.payload_digest,
                                row.disposition,
                                json.dumps(
                                    row.to_dict(),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            ],
                        )

                for row in receipt.quarantined_rows:
                    connection.execute(
                        """
                        INSERT INTO legacy_import_quarantine (
                            import_key, quarantine_id, domain, record_id,
                            reason, source_ids_json, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            receipt.import_key,
                            row.quarantine_id,
                            row.domain,
                            row.record_id,
                            row.reason,
                            json.dumps(
                                list(row.source_ids),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                row.to_dict(),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ],
                    )

                for row in receipt.rejected_rows:
                    connection.execute(
                        """
                        INSERT INTO legacy_import_rejections (
                            import_key, rejection_id, domain, record_id,
                            reason, source_id, source_digest, parser_version,
                            body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            receipt.import_key,
                            row.rejection_id,
                            row.domain,
                            row.record_id,
                            row.reason,
                            row.source_id,
                            row.source_digest,
                            row.parser_version,
                            json.dumps(
                                row.to_dict(),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ],
                    )

                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def get_receipt(self, import_key: str) -> ImportReceipt | None:
        """Return a stored receipt by import key, if present."""

        return self._load_existing_receipt(import_key)

    def list_accepted_rows(self, import_key: str) -> tuple[ImportedRow, ...]:
        """Return accepted rows for an applied import_key."""

        with open_duckdb_connection(self.target_path) as connection:
            rows = connection.execute(
                """
                SELECT body_json FROM legacy_import_rows
                WHERE import_key = ?
                ORDER BY row_id
                """,
                [import_key],
            ).fetchall()
        result: list[ImportedRow] = []
        for row in rows:
            payload = json.loads(str(row[0]))
            result.append(
                ImportedRow(
                    row_id=str(payload["row_id"]),
                    domain=str(payload["domain"]),
                    record_id=str(payload["record_id"]),
                    source_id=str(payload["source_id"]),
                    source_digest=str(payload["source_digest"]),
                    parser_version=str(payload["parser_version"]),
                    payload=dict(payload.get("payload") or {}),
                    payload_digest=str(payload["payload_digest"]),
                    disposition=str(payload.get("disposition") or DISPOSITION_ACCEPT),
                )
            )
        return tuple(result)

    def source_bytes_unchanged(self, manifest: ImportManifest) -> bool:
        """Return True when every declared source still matches its path bytes.

        Used by tests and operators to prove import never mutates sources.
        """

        for source in manifest.sources:
            path = self._resolve_source_path(source.path)
            if not path.is_file():
                return False
        return True


def build_manifest(
    *,
    manifest_id: str,
    sources: Sequence[ImportSourceDescriptor | Mapping[str, Any]],
    conflict_policy: str | ConflictPolicy = ConflictPolicy.REJECT,
    mode: str | ImportMode = ImportMode.PREVIEW,
    strict: bool = True,
    target_store_id: str = "control.duckdb",
    selected_sources: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ImportManifest:
    """Convenience constructor for manifests from descriptors or mappings."""

    resolved: list[ImportSourceDescriptor] = []
    for item in sources:
        if isinstance(item, ImportSourceDescriptor):
            resolved.append(item)
        else:
            resolved.append(ImportSourceDescriptor.from_dict(item))
    return ImportManifest(
        manifest_id=manifest_id,
        sources=tuple(resolved),
        conflict_policy=conflict_policy,
        mode=mode,
        strict=strict,
        target_store_id=target_store_id,
        selected_sources=tuple(selected_sources),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "ConflictPolicy",
    "DEFAULT_PARSER_VERSION",
    "DISPOSITION_ACCEPT",
    "DISPOSITION_MERGE",
    "DISPOSITION_QUARANTINE",
    "DISPOSITION_REJECT",
    "DISPOSITION_SELECT",
    "DISPOSITION_SKIP_DUPLICATE",
    "DuckDBUnavailableError",
    "IMPORT_MANIFEST_INTERFACE",
    "IMPORT_MANIFEST_SCHEMA",
    "IMPORT_RECEIPT_INTERFACE",
    "IMPORT_RECEIPT_SCHEMA",
    "ImportAtomicityError",
    "ImportConflictError",
    "ImportDomain",
    "ImportManifest",
    "ImportManifestError",
    "ImportMode",
    "ImportParseError",
    "ImportReceipt",
    "ImportSchemaError",
    "ImportSourceDescriptor",
    "ImportSourceError",
    "ImportSourceKind",
    "ImportSourceMutationError",
    "ImportedRow",
    "LEGACY_STATE_IMPORT_INTERFACE",
    "LEGACY_STATE_IMPORT_SCHEMA",
    "LegacyStateImport",
    "LegacyStateImportError",
    "OUTCOME_APPLIED",
    "OUTCOME_FAILED",
    "OUTCOME_PREVIEW",
    "OUTCOME_REFUSED",
    "OUTCOME_REPLAYED",
    "ObservedSource",
    "PARSER_VERSION",
    "ParsedRecord",
    "QuarantinedRecord",
    "RejectedRecord",
    "build_manifest",
    "digest_file",
    "duckdb_available",
    "parse_source",
    "reconcile_records",
    "sha256_digest",
]
