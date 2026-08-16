"""Legacy Markdown/JSON/JSONL/SQLite/DuckDB state import with provenance.

DQP-010 / LegacyStateImport@1
=============================

Importers read legacy artifacts under an explicit :class:`ImportManifest`.
Each source is observed with a byte digest, parser/schema identity, path,
timestamp observation, record counts, rejected rows, and a reconciliation
decision. Import is idempotent and defaults to preview. Conflicting
authorities are never silently last-write-wins; an operator or deterministic
policy must ``select``, ``merge``, ``quarantine``, or ``reject`` them.

Strict apply commits all accepted rows atomically or not at all. Exact replay
of a previously applied import is a no-op that returns the same receipt.
Every accepted row remains traceable to its source digest and parser version.
Imported sources are never modified or deleted.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .task_identity import canonical_content_cid, canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

LEGACY_STATE_IMPORT_INTERFACE: Final[str] = "LegacyStateImport@1"
IMPORT_MANIFEST_INTERFACE: Final[str] = "ImportManifest@1"
IMPORT_RECEIPT_INTERFACE: Final[str] = "ImportReceipt@1"

LEGACY_STATE_IMPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legacy-state-import@1"
)
IMPORT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/import-manifest@1"
)
IMPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/import-receipt@1"
)
IMPORT_SOURCE_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/import-source-observation@1"
)
IMPORTED_ROW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/imported-row@1"
)
IMPORT_CONFLICT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/import-conflict@1"
)

PARSER_VERSION: Final[str] = "legacy-state-import/1"
MANIFEST_VERSION: Final[int] = 1

# Outcomes recorded on ImportReceipt.
OUTCOME_PREVIEWED: Final[str] = "previewed"
OUTCOME_APPLIED: Final[str] = "applied"
OUTCOME_REPLAYED: Final[str] = "replayed"
OUTCOME_REJECTED: Final[str] = "rejected"
OUTCOME_FAILED: Final[str] = "failed"

MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_SOURCES: Final[int] = 256
MAX_RECORDS_PER_SOURCE: Final[int] = 50_000
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_PATH_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512

_HEADING_RE = re.compile(
    r"^##[ \t]+(?P<record_id>\S+)(?:[ \t]+(?P<title>[^\n]*))?[ \t]*$",
    flags=re.MULTILINE,
)
_FIELD_RE = re.compile(
    r"^[ \t]*[-*][ \t]+(?P<key>[A-Za-z0-9_./:-]+)[ \t]*:[ \t]*(?P<value>.*)$"
)
_TASK_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
# SQL table identifiers used in quoted SELECT projections only.
_SAFE_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS import_receipts (
    receipt_cid VARCHAR PRIMARY KEY,
    import_id VARCHAR NOT NULL UNIQUE,
    manifest_cid VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS imported_rows (
    row_cid VARCHAR PRIMARY KEY,
    import_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    source_digest VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (domain, record_id)
);
CREATE TABLE IF NOT EXISTS quarantined_rows (
    quarantine_cid VARCHAR PRIMARY KEY,
    import_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    source_digest VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS import_source_observations (
    source_id VARCHAR NOT NULL,
    import_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    media_type VARCHAR NOT NULL,
    source_digest VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    record_count INTEGER NOT NULL,
    accepted_count INTEGER NOT NULL,
    rejected_count INTEGER NOT NULL,
    quarantined_count INTEGER NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (import_id, source_id)
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LegacyStateImportError(RuntimeError):
    """Base class for fail-closed legacy import errors."""


class ImportManifestError(LegacyStateImportError, ValueError):
    """The import manifest is malformed or inconsistent."""


class ImportSourceError(LegacyStateImportError):
    """A declared source cannot be observed or parsed safely."""


class ImportConflictError(LegacyStateImportError):
    """Conflicting authorities require an explicit reconciliation decision."""


class ImportStrictError(LegacyStateImportError):
    """Strict import refused because one or more rows were not accepted."""


class ImportAtomicityError(LegacyStateImportError):
    """A strict apply could not commit atomically and was rolled back."""


class ImportSourceMutationError(LegacyStateImportError):
    """An import path attempted to modify or delete a declared source."""


class DuckDBUnavailableError(LegacyStateImportError):
    """DuckDB is required for durable apply but is missing."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ImportMediaType(str, Enum):
    """Closed set of legacy media types the importer understands."""

    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"


class ImportMode(str, Enum):
    """Import execution mode. Preview is the safe default."""

    PREVIEW = "preview"
    APPLY = "apply"


class ConflictPolicy(str, Enum):
    """Explicit reconciliation; last-write-wins is intentionally absent."""

    SELECT = "select"
    MERGE = "merge"
    QUARANTINE = "quarantine"
    REJECT = "reject"


class ImportDomain(str, Enum):
    """Canonical destination domains for imported records."""

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
    GENERIC = "generic"


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
            "DuckDB is required for durable legacy-state import apply; "
            "install the optional duckdb dependency"
        ) from exc
    return duckdb


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_iso(value: datetime | None = None) -> str:
    moment = value or _utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _text(value: Any, field_name: str, *, required: bool = True, limit: int = MAX_ID_BYTES) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ImportManifestError(f"{field_name} must be a string")
    else:
        text = value
    if text != text.strip():
        raise ImportManifestError(f"{field_name} has leading or trailing whitespace")
    if required and not text:
        raise ImportManifestError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise ImportManifestError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise ImportManifestError(f"{field_name} exceeds its byte bound")
    return text


def _enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise ImportManifestError(
                f"{field_name} is not a closed {enum_cls.__name__} value"
            ) from exc
    raise ImportManifestError(f"{field_name} must be a {enum_cls.__name__} value")


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ImportManifestError("expected a mapping")
    return {str(key): member for key, member in value.items()}


def _stable_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _infer_media_type(path: Path) -> ImportMediaType:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return ImportMediaType.MARKDOWN
    if suffix == ".json":
        return ImportMediaType.JSON
    if suffix == ".jsonl":
        return ImportMediaType.JSONL
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return ImportMediaType.SQLITE
    if suffix == ".duckdb":
        return ImportMediaType.DUCKDB
    raise ImportManifestError(
        f"cannot infer media type for {path.name!r}; declare media_type explicitly"
    )


def _normalize_record_id(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    if len(text.encode("utf-8")) > MAX_ID_BYTES:
        raise ImportSourceError(f"record_id exceeds byte bound: {text[:64]}...")
    return text


def _require_safe_table_name(table_name: str) -> str:
    """Refuse operator-declared table names that are not simple identifiers."""

    name = str(table_name or "").strip()
    if not name:
        raise ImportSourceError("table_name must not be empty when declared")
    if not _SAFE_TABLE_RE.fullmatch(name):
        raise ImportSourceError(
            f"unsupported table_name {name!r}; expected [A-Za-z_][A-Za-z0-9_]*"
        )
    if len(name.encode("utf-8")) > MAX_ID_BYTES:
        raise ImportSourceError("table_name exceeds byte bound")
    return name


def _quote_ident(identifier: str) -> str:
    """Quote a SQL identifier; double embedded quotes, refuse NUL/newlines."""

    name = str(identifier or "")
    if not name or "\x00" in name or "\n" in name or "\r" in name:
        raise ImportSourceError(f"unsupported table identifier: {name!r}")
    if len(name.encode("utf-8")) > MAX_ID_BYTES:
        raise ImportSourceError("table identifier exceeds byte bound")
    return '"' + name.replace('"', '""') + '"'


def _record_payload_bytes(payload: Mapping[str, Any]) -> int:
    return len(canonical_json_bytes(dict(payload)))


# ---------------------------------------------------------------------------
# Manifest / observation / row contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportConflictResolution:
    """Operator or policy decision for one conflicting record identity."""

    domain: str
    record_id: str
    policy: ConflictPolicy
    selected_source_id: str = ""
    schema: str = IMPORT_CONFLICT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _text(self.domain, "domain"))
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(
            self,
            "policy",
            _enum(self.policy, ConflictPolicy, field_name="policy"),
        )
        selected = _text(
            self.selected_source_id,
            "selected_source_id",
            required=False,
        )
        object.__setattr__(self, "selected_source_id", selected)
        if self.policy is ConflictPolicy.SELECT and not selected:
            raise ImportManifestError(
                "select conflict resolution requires selected_source_id"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domain": self.domain,
            "record_id": self.record_id,
            "policy": self.policy.value,
            "selected_source_id": self.selected_source_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportConflictResolution:
        data = _mapping(payload)
        return cls(
            domain=str(data.get("domain") or ""),
            record_id=str(data.get("record_id") or ""),
            policy=str(data.get("policy") or ConflictPolicy.REJECT.value),
            selected_source_id=str(data.get("selected_source_id") or ""),
        )


@dataclass(frozen=True)
class ImportSourceSpec:
    """One legacy source declared in an import manifest."""

    source_id: str
    path: str
    media_type: ImportMediaType
    domain: ImportDomain = ImportDomain.GENERIC
    parser_version: str = PARSER_VERSION
    table_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        path = _text(self.path, "path", limit=MAX_PATH_BYTES)
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "media_type",
            _enum(self.media_type, ImportMediaType, field_name="media_type"),
        )
        object.__setattr__(
            self,
            "domain",
            _enum(self.domain, ImportDomain, field_name="domain"),
        )
        object.__setattr__(
            self,
            "parser_version",
            _text(self.parser_version, "parser_version", limit=MAX_ID_BYTES),
        )
        table = _text(self.table_name, "table_name", required=False)
        if table and not _SAFE_TABLE_RE.fullmatch(table):
            raise ImportManifestError(
                f"unsupported table_name {table!r}; expected [A-Za-z_][A-Za-z0-9_]*"
            )
        object.__setattr__(self, "table_name", table)
        meta = dict(self.metadata or {})
        for key in meta:
            if not isinstance(key, str):
                raise ImportManifestError("metadata keys must be strings")
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "media_type": self.media_type.value,
            "domain": self.domain.value,
            "parser_version": self.parser_version,
            "table_name": self.table_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportSourceSpec:
        data = _mapping(payload)
        path = str(data.get("path") or "")
        media = data.get("media_type")
        if media is None and path:
            media = _infer_media_type(Path(path)).value
        return cls(
            source_id=str(data.get("source_id") or Path(path).name or "source"),
            path=path,
            media_type=str(media or ""),
            domain=str(data.get("domain") or ImportDomain.GENERIC.value),
            parser_version=str(data.get("parser_version") or PARSER_VERSION),
            table_name=str(data.get("table_name") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ImportManifest:
    """Explicit, content-addressed description of a legacy import batch.

    Interface: ImportManifest@1
    """

    import_id: str
    sources: tuple[ImportSourceSpec, ...]
    mode: ImportMode = ImportMode.PREVIEW
    strict: bool = True
    default_conflict_policy: ConflictPolicy = ConflictPolicy.REJECT
    conflict_resolutions: tuple[ImportConflictResolution, ...] = ()
    target_database: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = IMPORT_MANIFEST_SCHEMA
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "import_id", _text(self.import_id, "import_id"))
        sources = tuple(self.sources or ())
        object.__setattr__(self, "sources", sources)
        if not sources:
            raise ImportManifestError("import manifest requires at least one source")
        if len(sources) > MAX_SOURCES:
            raise ImportManifestError(f"import manifest exceeds {MAX_SOURCES} sources")
        ids = [source.source_id for source in sources]
        if len(set(ids)) != len(ids):
            raise ImportManifestError("duplicate source_id values are refused")
        for source in sources:
            if not isinstance(source, ImportSourceSpec):
                raise ImportManifestError("sources must be ImportSourceSpec values")
        object.__setattr__(
            self,
            "mode",
            _enum(self.mode, ImportMode, field_name="mode"),
        )
        object.__setattr__(
            self,
            "default_conflict_policy",
            _enum(
                self.default_conflict_policy,
                ConflictPolicy,
                field_name="default_conflict_policy",
            ),
        )
        resolutions = tuple(self.conflict_resolutions or ())
        for item in resolutions:
            if not isinstance(item, ImportConflictResolution):
                raise ImportManifestError(
                    "conflict_resolutions must be ImportConflictResolution values"
                )
        object.__setattr__(self, "conflict_resolutions", resolutions)
        object.__setattr__(
            self,
            "target_database",
            _text(self.target_database, "target_database", required=False, limit=MAX_PATH_BYTES),
        )
        meta = dict(self.metadata or {})
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        if int(self.version) != MANIFEST_VERSION:
            raise ImportManifestError(
                f"unsupported import manifest version {self.version}"
            )

    @property
    def manifest_cid(self) -> str:
        return canonical_content_cid(self.to_dict())

    def resolution_for(
        self, domain: str, record_id: str
    ) -> ImportConflictResolution | None:
        for item in self.conflict_resolutions:
            if item.domain == domain and item.record_id == record_id:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": int(self.version),
            "interface": IMPORT_MANIFEST_INTERFACE,
            "import_id": self.import_id,
            "mode": self.mode.value,
            "strict": bool(self.strict),
            "default_conflict_policy": self.default_conflict_policy.value,
            "sources": [source.to_dict() for source in self.sources],
            "conflict_resolutions": [
                item.to_dict() for item in self.conflict_resolutions
            ],
            "target_database": self.target_database,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportManifest:
        data = _mapping(payload)
        sources = tuple(
            ImportSourceSpec.from_dict(item)
            for item in (data.get("sources") or ())
        )
        resolutions = tuple(
            ImportConflictResolution.from_dict(item)
            for item in (data.get("conflict_resolutions") or ())
        )
        return cls(
            import_id=str(data.get("import_id") or ""),
            sources=sources,
            mode=str(data.get("mode") or ImportMode.PREVIEW.value),
            strict=bool(data.get("strict", True)),
            default_conflict_policy=str(
                data.get("default_conflict_policy") or ConflictPolicy.REJECT.value
            ),
            conflict_resolutions=resolutions,
            target_database=str(data.get("target_database") or ""),
            metadata=dict(data.get("metadata") or {}),
            version=int(data.get("version") or MANIFEST_VERSION),
        )

    @classmethod
    def from_paths(
        cls,
        import_id: str,
        paths: Sequence[str | Path],
        *,
        mode: ImportMode | str = ImportMode.PREVIEW,
        strict: bool = True,
        default_conflict_policy: ConflictPolicy | str = ConflictPolicy.REJECT,
        domain: ImportDomain | str = ImportDomain.GENERIC,
        target_database: str | Path = "",
        conflict_resolutions: Sequence[ImportConflictResolution] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> ImportManifest:
        sources: list[ImportSourceSpec] = []
        for index, raw in enumerate(paths):
            path = Path(raw)
            media = _infer_media_type(path)
            sources.append(
                ImportSourceSpec(
                    source_id=f"src-{index + 1}-{path.name}",
                    path=str(path),
                    media_type=media,
                    domain=domain,  # type: ignore[arg-type]
                )
            )
        return cls(
            import_id=import_id,
            sources=tuple(sources),
            mode=mode,  # type: ignore[arg-type]
            strict=strict,
            default_conflict_policy=default_conflict_policy,  # type: ignore[arg-type]
            conflict_resolutions=tuple(conflict_resolutions),
            target_database=str(target_database or ""),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class ParsedRecord:
    """One parsed legacy record prior to reconciliation."""

    domain: str
    record_id: str
    payload: Mapping[str, Any]
    source_id: str
    source_digest: str
    parser_version: str
    media_type: str
    line_or_index: int = 0
    rejected: bool = False
    reject_reason: str = ""

    def semantic_key(self) -> tuple[str, str]:
        return (self.domain, self.record_id)

    def content_cid(self) -> str:
        return canonical_content_cid(
            {
                "domain": self.domain,
                "record_id": self.record_id,
                "payload": dict(self.payload),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPORTED_ROW_SCHEMA,
            "domain": self.domain,
            "record_id": self.record_id,
            "payload": dict(self.payload),
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "parser_version": self.parser_version,
            "media_type": self.media_type,
            "line_or_index": int(self.line_or_index),
            "rejected": bool(self.rejected),
            "reject_reason": self.reject_reason,
            "content_cid": self.content_cid() if not self.rejected else "",
        }


@dataclass(frozen=True)
class SourceObservation:
    """Byte/schema/parser provenance for one declared source."""

    source_id: str
    path: str
    media_type: str
    source_digest: str
    parser_version: str
    observed_at: str
    byte_size: int
    record_count: int
    accepted_count: int = 0
    rejected_count: int = 0
    quarantined_count: int = 0
    schema: str = IMPORT_SOURCE_OBSERVATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_id": self.source_id,
            "path": self.path,
            "media_type": self.media_type,
            "source_digest": self.source_digest,
            "parser_version": self.parser_version,
            "observed_at": self.observed_at,
            "byte_size": int(self.byte_size),
            "record_count": int(self.record_count),
            "accepted_count": int(self.accepted_count),
            "rejected_count": int(self.rejected_count),
            "quarantined_count": int(self.quarantined_count),
        }


@dataclass(frozen=True)
class ConflictReport:
    """One multi-source identity conflict and its reconciliation decision."""

    domain: str
    record_id: str
    policy: str
    decision: str
    source_ids: tuple[str, ...]
    selected_source_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPORT_CONFLICT_SCHEMA,
            "domain": self.domain,
            "record_id": self.record_id,
            "policy": self.policy,
            "decision": self.decision,
            "source_ids": list(self.source_ids),
            "selected_source_id": self.selected_source_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ImportReceipt:
    """Durable attestation for one preview/apply/replay import attempt.

    Interface: ImportReceipt@1
    """

    receipt_cid: str
    import_id: str
    manifest_cid: str
    mode: str
    outcome: str
    parser_version: str
    started_at: str
    finished_at: str
    source_observations: tuple[SourceObservation, ...]
    accepted_rows: tuple[Mapping[str, Any], ...]
    rejected_rows: tuple[Mapping[str, Any], ...]
    quarantined_rows: tuple[Mapping[str, Any], ...]
    conflicts: tuple[ConflictReport, ...]
    strict: bool
    applied: bool
    replayed: bool
    error_text: str = ""
    schema: str = IMPORT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": IMPORT_RECEIPT_INTERFACE,
            "receipt_cid": self.receipt_cid,
            "import_id": self.import_id,
            "manifest_cid": self.manifest_cid,
            "mode": self.mode,
            "outcome": self.outcome,
            "parser_version": self.parser_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source_observations": [
                item.to_dict() for item in self.source_observations
            ],
            "accepted_rows": [dict(item) for item in self.accepted_rows],
            "rejected_rows": [dict(item) for item in self.rejected_rows],
            "quarantined_rows": [dict(item) for item in self.quarantined_rows],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "strict": bool(self.strict),
            "applied": bool(self.applied),
            "replayed": bool(self.replayed),
            "error_text": self.error_text,
            "accepted_count": len(self.accepted_rows),
            "rejected_count": len(self.rejected_rows),
            "quarantined_count": len(self.quarantined_rows),
            "conflict_count": len(self.conflicts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImportReceipt:
        data = _mapping(payload)
        observations = tuple(
            SourceObservation(
                source_id=str(item["source_id"]),
                path=str(item["path"]),
                media_type=str(item["media_type"]),
                source_digest=str(item["source_digest"]),
                parser_version=str(item["parser_version"]),
                observed_at=str(item["observed_at"]),
                byte_size=int(item.get("byte_size") or 0),
                record_count=int(item.get("record_count") or 0),
                accepted_count=int(item.get("accepted_count") or 0),
                rejected_count=int(item.get("rejected_count") or 0),
                quarantined_count=int(item.get("quarantined_count") or 0),
            )
            for item in (data.get("source_observations") or ())
        )
        conflicts = tuple(
            ConflictReport(
                domain=str(item["domain"]),
                record_id=str(item["record_id"]),
                policy=str(item["policy"]),
                decision=str(item["decision"]),
                source_ids=tuple(str(x) for x in (item.get("source_ids") or ())),
                selected_source_id=str(item.get("selected_source_id") or ""),
                reason=str(item.get("reason") or ""),
            )
            for item in (data.get("conflicts") or ())
        )
        return cls(
            receipt_cid=str(data["receipt_cid"]),
            import_id=str(data["import_id"]),
            manifest_cid=str(data["manifest_cid"]),
            mode=str(data["mode"]),
            outcome=str(data["outcome"]),
            parser_version=str(data["parser_version"]),
            started_at=str(data["started_at"]),
            finished_at=str(data["finished_at"]),
            source_observations=observations,
            accepted_rows=tuple(
                MappingProxyType(dict(item))
                for item in (data.get("accepted_rows") or ())
            ),
            rejected_rows=tuple(
                MappingProxyType(dict(item))
                for item in (data.get("rejected_rows") or ())
            ),
            quarantined_rows=tuple(
                MappingProxyType(dict(item))
                for item in (data.get("quarantined_rows") or ())
            ),
            conflicts=conflicts,
            strict=bool(data.get("strict", True)),
            applied=bool(data.get("applied", False)),
            replayed=bool(data.get("replayed", False)),
            error_text=str(data.get("error_text") or ""),
        )


def _receipt_cid_from_body(body: Mapping[str, Any]) -> str:
    # Exclude receipt_cid itself and volatile timestamps when identity is
    # recomputed for exact-replay comparison of semantic outcomes.
    material: dict[str, Any] = {}
    for key, value in body.items():
        if key in {
            "receipt_cid",
            "started_at",
            "finished_at",
            "outcome",
            "replayed",
            "applied",
            "mode",
            "error_text",
        }:
            continue
        if key == "source_observations" and isinstance(value, list):
            material[key] = [
                {
                    inner_key: inner_value
                    for inner_key, inner_value in dict(item).items()
                    if inner_key != "observed_at"
                }
                for item in value
            ]
            continue
        material[key] = value
    return canonical_content_cid(material)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _reject(
    *,
    domain: str,
    record_id: str,
    source_id: str,
    source_digest: str,
    parser_version: str,
    media_type: str,
    reason: str,
    line_or_index: int = 0,
    payload: Mapping[str, Any] | None = None,
) -> ParsedRecord:
    return ParsedRecord(
        domain=domain,
        record_id=record_id or f"rejected:{line_or_index}",
        payload=MappingProxyType(dict(payload or {})),
        source_id=source_id,
        source_digest=source_digest,
        parser_version=parser_version,
        media_type=media_type,
        line_or_index=line_or_index,
        rejected=True,
        reject_reason=reason,
    )


def _accept(
    *,
    domain: str,
    record_id: str,
    payload: Mapping[str, Any],
    source_id: str,
    source_digest: str,
    parser_version: str,
    media_type: str,
    line_or_index: int = 0,
) -> ParsedRecord:
    body = dict(payload)
    if _record_payload_bytes(body) > MAX_RECORD_BYTES:
        return _reject(
            domain=domain,
            record_id=record_id,
            source_id=source_id,
            source_digest=source_digest,
            parser_version=parser_version,
            media_type=media_type,
            reason="record exceeds byte bound",
            line_or_index=line_or_index,
            payload={"record_id": record_id},
        )
    return ParsedRecord(
        domain=domain,
        record_id=record_id,
        payload=MappingProxyType(body),
        source_id=source_id,
        source_digest=source_digest,
        parser_version=parser_version,
        media_type=media_type,
        line_or_index=line_or_index,
    )


def parse_json_records(
    text: str,
    *,
    source_id: str,
    source_digest: str,
    parser_version: str,
    domain: str,
) -> list[ParsedRecord]:
    media = ImportMediaType.JSON.value
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [
            _reject(
                domain=domain,
                record_id="document",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"corrupt json: {exc}",
            )
        ]

    items: list[Any]
    if isinstance(payload, list):
        items = list(payload)
    elif isinstance(payload, Mapping):
        if isinstance(payload.get("records"), list):
            items = list(payload["records"])
        elif isinstance(payload.get("tasks"), list):
            items = list(payload["tasks"])
        elif isinstance(payload.get("objectives"), list):
            items = list(payload["objectives"])
        else:
            items = [payload]
    else:
        return [
            _reject(
                domain=domain,
                record_id="document",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason="unsupported json root type",
            )
        ]

    if len(items) > MAX_RECORDS_PER_SOURCE:
        return [
            _reject(
                domain=domain,
                record_id="document",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"source exceeds {MAX_RECORDS_PER_SOURCE} records",
            )
        ]

    records: list[ParsedRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            records.append(
                _reject(
                    domain=domain,
                    record_id=f"index:{index}",
                    source_id=source_id,
                    source_digest=source_digest,
                    parser_version=parser_version,
                    media_type=media,
                    reason="record is not an object",
                    line_or_index=index,
                )
            )
            continue
        body = {str(key): value for key, value in item.items()}
        record_id = _normalize_record_id(
            body.get("record_id")
            or body.get("id")
            or body.get("task_id")
            or body.get("objective_id")
            or body.get("name"),
            fallback=f"index:{index}",
        )
        records.append(
            _accept(
                domain=domain,
                record_id=record_id,
                payload=body,
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                line_or_index=index,
            )
        )
    return records


def parse_jsonl_records(
    text: str,
    *,
    source_id: str,
    source_digest: str,
    parser_version: str,
    domain: str,
) -> list[ParsedRecord]:
    media = ImportMediaType.JSONL.value
    records: list[ParsedRecord] = []
    # Preserve blank lines as structural positions; reject truncated tails.
    lines = text.splitlines()
    if text and not text.endswith("\n") and lines:
        # Truncated final line is still parseable if complete JSON; leave to
        # JSON decoder. Empty file is valid (zero records).
        pass
    if len(lines) > MAX_RECORDS_PER_SOURCE:
        return [
            _reject(
                domain=domain,
                record_id="document",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"source exceeds {MAX_RECORDS_PER_SOURCE} records",
            )
        ]
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            records.append(
                _reject(
                    domain=domain,
                    record_id=f"line:{index}",
                    source_id=source_id,
                    source_digest=source_digest,
                    parser_version=parser_version,
                    media_type=media,
                    reason=f"corrupt/truncated jsonl line: {exc}",
                    line_or_index=index,
                )
            )
            continue
        if not isinstance(item, Mapping):
            records.append(
                _reject(
                    domain=domain,
                    record_id=f"line:{index}",
                    source_id=source_id,
                    source_digest=source_digest,
                    parser_version=parser_version,
                    media_type=media,
                    reason="jsonl record is not an object",
                    line_or_index=index,
                )
            )
            continue
        body = {str(key): value for key, value in item.items()}
        record_id = _normalize_record_id(
            body.get("record_id")
            or body.get("id")
            or body.get("event_id")
            or body.get("task_id")
            or body.get("name"),
            fallback=f"line:{index}",
        )
        records.append(
            _accept(
                domain=domain,
                record_id=record_id,
                payload=body,
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                line_or_index=index,
            )
        )
    return records


def parse_markdown_records(
    text: str,
    *,
    source_id: str,
    source_digest: str,
    parser_version: str,
    domain: str,
) -> list[ParsedRecord]:
    media = ImportMediaType.MARKDOWN.value
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        # Treat whole document as one generic record when no task headings.
        if not text.strip():
            return []
        return [
            _accept(
                domain=domain,
                record_id="document",
                payload={"title": "", "body": text, "format": "markdown"},
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                line_or_index=0,
            )
        ]

    if len(matches) > MAX_RECORDS_PER_SOURCE:
        return [
            _reject(
                domain=domain,
                record_id="document",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"source exceeds {MAX_RECORDS_PER_SOURCE} records",
            )
        ]

    records: list[ParsedRecord] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body_text = text[start:end].strip("\n")
        record_id = match.group("record_id").strip()
        title = (match.group("title") or "").strip()
        if not _TASK_ID_RE.match(record_id):
            records.append(
                _reject(
                    domain=domain,
                    record_id=record_id or f"heading:{index}",
                    source_id=source_id,
                    source_digest=source_digest,
                    parser_version=parser_version,
                    media_type=media,
                    reason="unsupported markdown record id",
                    line_or_index=index,
                )
            )
            continue
        fields: dict[str, Any] = {
            "record_id": record_id,
            "title": title,
            "format": "markdown",
        }
        body_lines: list[str] = []
        for line in body_text.splitlines():
            field_match = _FIELD_RE.match(line)
            if field_match:
                key = field_match.group("key").strip().casefold().replace(" ", "_")
                fields[key] = field_match.group("value").strip()
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if body:
            fields["body"] = body
        records.append(
            _accept(
                domain=domain,
                record_id=record_id,
                payload=fields,
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                line_or_index=index,
            )
        )
    return records


def parse_sqlite_records(
    path: Path,
    *,
    source_id: str,
    source_digest: str,
    parser_version: str,
    domain: str,
    table_name: str = "",
) -> list[ParsedRecord]:
    media = ImportMediaType.SQLITE.value
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return [
            _reject(
                domain=domain,
                record_id="database",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"corrupt/unreadable sqlite: {exc}",
            )
        ]
    try:
        connection.row_factory = sqlite3.Row
        tables: list[str]
        if table_name:
            try:
                tables = [_require_safe_table_name(table_name)]
            except ImportSourceError as exc:
                return [
                    _reject(
                        domain=domain,
                        record_id="database",
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        reason=str(exc),
                    )
                ]
        else:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            tables = [str(row[0]) for row in rows]
        if not tables:
            return []
        records: list[ParsedRecord] = []
        index = 0
        for table in tables:
            try:
                quoted = _quote_ident(table)
            except ImportSourceError as exc:
                records.append(
                    _reject(
                        domain=domain,
                        record_id=table or "table",
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        reason=str(exc),
                        line_or_index=index,
                    )
                )
                continue
            try:
                cursor = connection.execute(f"SELECT * FROM {quoted}")
            except sqlite3.Error as exc:
                records.append(
                    _reject(
                        domain=domain,
                        record_id=table,
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        reason=f"unsupported/unreadable table {table}: {exc}",
                        line_or_index=index,
                    )
                )
                continue
            columns = [str(item[0]) for item in cursor.description or ()]
            for row in cursor.fetchall():
                if index >= MAX_RECORDS_PER_SOURCE:
                    records.append(
                        _reject(
                            domain=domain,
                            record_id="database",
                            source_id=source_id,
                            source_digest=source_digest,
                            parser_version=parser_version,
                            media_type=media,
                            reason=f"source exceeds {MAX_RECORDS_PER_SOURCE} records",
                            line_or_index=index,
                        )
                    )
                    return records
                body = {
                    columns[i]: row[i]
                    for i in range(len(columns))
                }
                # Normalize non-JSON-native values.
                normalized: dict[str, Any] = {}
                for key, value in body.items():
                    if isinstance(value, (bytes, bytearray, memoryview)):
                        normalized[key] = bytes(value).hex()
                    elif value is None or isinstance(value, (str, bool, int)):
                        normalized[key] = value
                    elif isinstance(value, float):
                        # Contracts reject floats in canonical identity; stringify.
                        normalized[key] = format(value, ".15g")
                    else:
                        normalized[key] = str(value)
                normalized["table"] = table
                record_id = _normalize_record_id(
                    normalized.get("record_id")
                    or normalized.get("id")
                    or normalized.get("task_id")
                    or f"{table}:{index}",
                    fallback=f"{table}:{index}",
                )
                records.append(
                    _accept(
                        domain=domain,
                        record_id=record_id,
                        payload=normalized,
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        line_or_index=index,
                    )
                )
                index += 1
        return records
    finally:
        connection.close()


def parse_duckdb_records(
    path: Path,
    *,
    source_id: str,
    source_digest: str,
    parser_version: str,
    domain: str,
    table_name: str = "",
) -> list[ParsedRecord]:
    media = ImportMediaType.DUCKDB.value
    if not duckdb_available():
        return [
            _reject(
                domain=domain,
                record_id="database",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason="duckdb package unavailable",
            )
        ]
    duckdb = _require_duckdb()
    try:
        # Fail closed: never open a legacy DuckDB source for write.
        connection = duckdb.connect(str(path), read_only=True)
    except TypeError as exc:
        return [
            _reject(
                domain=domain,
                record_id="database",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"duckdb read_only connect unsupported: {exc}",
            )
        ]
    except Exception as exc:  # noqa: BLE001 - surface as rejected source
        return [
            _reject(
                domain=domain,
                record_id="database",
                source_id=source_id,
                source_digest=source_digest,
                parser_version=parser_version,
                media_type=media,
                reason=f"corrupt/unreadable duckdb: {exc}",
            )
        ]
    try:
        if table_name:
            try:
                tables = [_require_safe_table_name(table_name)]
            except ImportSourceError as exc:
                return [
                    _reject(
                        domain=domain,
                        record_id="database",
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        reason=str(exc),
                    )
                ]
        else:
            rows = connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            tables = [str(row[0]) for row in rows]
        records: list[ParsedRecord] = []
        index = 0
        for table in tables:
            try:
                quoted = _quote_ident(table)
            except ImportSourceError as exc:
                records.append(
                    _reject(
                        domain=domain,
                        record_id=table or "table",
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        reason=str(exc),
                        line_or_index=index,
                    )
                )
                continue
            try:
                result = connection.execute(f"SELECT * FROM {quoted}")
            except Exception as exc:  # noqa: BLE001
                records.append(
                    _reject(
                        domain=domain,
                        record_id=table,
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        reason=f"unsupported/unreadable table {table}: {exc}",
                        line_or_index=index,
                    )
                )
                continue
            columns = [str(item[0]) for item in (result.description or ())]
            for row in result.fetchall():
                if index >= MAX_RECORDS_PER_SOURCE:
                    records.append(
                        _reject(
                            domain=domain,
                            record_id="database",
                            source_id=source_id,
                            source_digest=source_digest,
                            parser_version=parser_version,
                            media_type=media,
                            reason=f"source exceeds {MAX_RECORDS_PER_SOURCE} records",
                            line_or_index=index,
                        )
                    )
                    return records
                body = {
                    columns[i]: row[i]
                    for i in range(len(columns))
                }
                normalized: dict[str, Any] = {}
                for key, value in body.items():
                    if isinstance(value, (bytes, bytearray, memoryview)):
                        normalized[key] = bytes(value).hex()
                    elif value is None or isinstance(value, (str, bool, int)):
                        normalized[key] = value
                    elif isinstance(value, float):
                        normalized[key] = format(value, ".15g")
                    else:
                        normalized[key] = str(value)
                normalized["table"] = table
                record_id = _normalize_record_id(
                    normalized.get("record_id")
                    or normalized.get("id")
                    or normalized.get("task_id")
                    or f"{table}:{index}",
                    fallback=f"{table}:{index}",
                )
                records.append(
                    _accept(
                        domain=domain,
                        record_id=record_id,
                        payload=normalized,
                        source_id=source_id,
                        source_digest=source_digest,
                        parser_version=parser_version,
                        media_type=media,
                        line_or_index=index,
                    )
                )
                index += 1
        return records
    finally:
        connection.close()


def parse_source(
    spec: ImportSourceSpec,
    *,
    source_root: Path | None = None,
) -> tuple[SourceObservation, list[ParsedRecord]]:
    """Observe and parse one declared source without mutating it."""

    path = Path(spec.path).expanduser()
    if source_root is not None and not path.is_absolute():
        candidate = (Path(source_root) / path).resolve()
        root = Path(source_root).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ImportSourceError(
                f"source path escapes source_root: {spec.path}"
            ) from exc
        path = candidate
    elif not path.is_absolute():
        if ".." in path.parts:
            raise ImportSourceError(
                f"relative source path must not contain '..': {spec.path}"
            )
        path = path.resolve()
    if not path.exists():
        raise ImportSourceError(f"source path does not exist: {path}")
    if not path.is_file():
        raise ImportSourceError(f"source path is not a file: {path}")
    byte_size = path.stat().st_size
    if byte_size > MAX_SOURCE_BYTES:
        raise ImportSourceError(
            f"source {spec.source_id} exceeds {MAX_SOURCE_BYTES} bytes"
        )
    source_digest = _sha256_file(path)
    observed_at = _utc_iso()
    domain = spec.domain.value
    parser_version = spec.parser_version
    media = spec.media_type

    if media is ImportMediaType.MARKDOWN:
        text = path.read_text(encoding="utf-8")
        records = parse_markdown_records(
            text,
            source_id=spec.source_id,
            source_digest=source_digest,
            parser_version=parser_version,
            domain=domain,
        )
    elif media is ImportMediaType.JSON:
        text = path.read_text(encoding="utf-8")
        records = parse_json_records(
            text,
            source_id=spec.source_id,
            source_digest=source_digest,
            parser_version=parser_version,
            domain=domain,
        )
    elif media is ImportMediaType.JSONL:
        text = path.read_text(encoding="utf-8")
        records = parse_jsonl_records(
            text,
            source_id=spec.source_id,
            source_digest=source_digest,
            parser_version=parser_version,
            domain=domain,
        )
    elif media is ImportMediaType.SQLITE:
        records = parse_sqlite_records(
            path,
            source_id=spec.source_id,
            source_digest=source_digest,
            parser_version=parser_version,
            domain=domain,
            table_name=spec.table_name,
        )
    elif media is ImportMediaType.DUCKDB:
        records = parse_duckdb_records(
            path,
            source_id=spec.source_id,
            source_digest=source_digest,
            parser_version=parser_version,
            domain=domain,
            table_name=spec.table_name,
        )
    else:
        raise ImportSourceError(f"unsupported media type: {media}")

    observation = SourceObservation(
        source_id=spec.source_id,
        path=str(path),
        media_type=media.value,
        source_digest=source_digest,
        parser_version=parser_version,
        observed_at=observed_at,
        byte_size=int(byte_size),
        record_count=len(records),
    )
    return observation, records


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _merge_payloads(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Deterministic field union; fail when the same field differs."""

    merged = dict(left)
    for key, value in right.items():
        if key not in merged:
            merged[key] = value
            continue
        if merged[key] == value:
            continue
        # Identical semantic content after canonicalization?
        try:
            if _stable_json(merged[key]) == _stable_json(value):
                continue
        except Exception:  # noqa: BLE001
            return None
        return None
    return merged


def reconcile_records(
    records: Sequence[ParsedRecord],
    *,
    default_policy: ConflictPolicy,
    resolutions: Sequence[ImportConflictResolution],
) -> tuple[
    list[ParsedRecord],
    list[ParsedRecord],
    list[ParsedRecord],
    list[ConflictReport],
]:
    """Resolve multi-source identity conflicts without last-write-wins."""

    resolution_map = {
        (item.domain, item.record_id): item for item in resolutions
    }
    rejected = [item for item in records if item.rejected]
    live = [item for item in records if not item.rejected]

    by_key: dict[tuple[str, str], list[ParsedRecord]] = {}
    for item in live:
        by_key.setdefault(item.semantic_key(), []).append(item)

    accepted: list[ParsedRecord] = []
    quarantined: list[ParsedRecord] = []
    conflicts: list[ConflictReport] = []

    for key, group in sorted(by_key.items(), key=lambda pair: pair[0]):
        domain, record_id = key
        if len(group) == 1:
            accepted.append(group[0])
            continue

        # Identical content from multiple sources is a duplicate, not a conflict.
        content_ids = {item.content_cid() for item in group}
        if len(content_ids) == 1:
            # Prefer the first source in stable source_id order for provenance.
            winner = sorted(group, key=lambda item: item.source_id)[0]
            accepted.append(winner)
            conflicts.append(
                ConflictReport(
                    domain=domain,
                    record_id=record_id,
                    policy="duplicate",
                    decision="deduplicated",
                    source_ids=tuple(sorted(item.source_id for item in group)),
                    selected_source_id=winner.source_id,
                    reason="exact duplicate content across sources",
                )
            )
            continue

        resolution = resolution_map.get(key)
        policy = resolution.policy if resolution is not None else default_policy
        source_ids = tuple(sorted(item.source_id for item in group))

        if policy is ConflictPolicy.REJECT:
            for item in group:
                rejected.append(
                    ParsedRecord(
                        domain=item.domain,
                        record_id=item.record_id,
                        payload=item.payload,
                        source_id=item.source_id,
                        source_digest=item.source_digest,
                        parser_version=item.parser_version,
                        media_type=item.media_type,
                        line_or_index=item.line_or_index,
                        rejected=True,
                        reject_reason="conflict:reject",
                    )
                )
            conflicts.append(
                ConflictReport(
                    domain=domain,
                    record_id=record_id,
                    policy=policy.value,
                    decision="rejected",
                    source_ids=source_ids,
                    reason="conflicting authorities rejected",
                )
            )
            continue

        if policy is ConflictPolicy.QUARANTINE:
            for item in group:
                quarantined.append(item)
            conflicts.append(
                ConflictReport(
                    domain=domain,
                    record_id=record_id,
                    policy=policy.value,
                    decision="quarantined",
                    source_ids=source_ids,
                    reason="conflicting authorities quarantined",
                )
            )
            continue

        if policy is ConflictPolicy.SELECT:
            if resolution is None or not resolution.selected_source_id:
                for item in group:
                    rejected.append(
                        ParsedRecord(
                            domain=item.domain,
                            record_id=item.record_id,
                            payload=item.payload,
                            source_id=item.source_id,
                            source_digest=item.source_digest,
                            parser_version=item.parser_version,
                            media_type=item.media_type,
                            line_or_index=item.line_or_index,
                            rejected=True,
                            reject_reason="conflict:select requires selected_source_id",
                        )
                    )
                conflicts.append(
                    ConflictReport(
                        domain=domain,
                        record_id=record_id,
                        policy=policy.value,
                        decision="rejected",
                        source_ids=source_ids,
                        reason="select policy missing selected_source_id",
                    )
                )
                continue
            selected = [
                item
                for item in group
                if item.source_id == resolution.selected_source_id
            ]
            if not selected:
                for item in group:
                    rejected.append(
                        ParsedRecord(
                            domain=item.domain,
                            record_id=item.record_id,
                            payload=item.payload,
                            source_id=item.source_id,
                            source_digest=item.source_digest,
                            parser_version=item.parser_version,
                            media_type=item.media_type,
                            line_or_index=item.line_or_index,
                            rejected=True,
                            reject_reason=(
                                "conflict:select source not present in conflict set"
                            ),
                        )
                    )
                conflicts.append(
                    ConflictReport(
                        domain=domain,
                        record_id=record_id,
                        policy=policy.value,
                        decision="rejected",
                        source_ids=source_ids,
                        selected_source_id=resolution.selected_source_id,
                        reason="selected source_id not among conflict candidates",
                    )
                )
                continue
            accepted.append(selected[0])
            for item in group:
                if item.source_id != selected[0].source_id:
                    quarantined.append(item)
            conflicts.append(
                ConflictReport(
                    domain=domain,
                    record_id=record_id,
                    policy=policy.value,
                    decision="selected",
                    source_ids=source_ids,
                    selected_source_id=selected[0].source_id,
                    reason="operator/policy selected one authority",
                )
            )
            continue

        if policy is ConflictPolicy.MERGE:
            ordered = sorted(group, key=lambda item: item.source_id)
            merged_payload: dict[str, Any] | None = dict(ordered[0].payload)
            merge_failed = False
            for item in ordered[1:]:
                assert merged_payload is not None
                candidate = _merge_payloads(merged_payload, item.payload)
                if candidate is None:
                    merge_failed = True
                    break
                merged_payload = candidate
            if merge_failed or merged_payload is None:
                for item in group:
                    quarantined.append(item)
                conflicts.append(
                    ConflictReport(
                        domain=domain,
                        record_id=record_id,
                        policy=policy.value,
                        decision="quarantined",
                        source_ids=source_ids,
                        reason="merge failed on contradictory fields",
                    )
                )
                continue
            # Provenance binds every contributing source digest.
            provenance = {
                "merged_from_source_ids": list(source_ids),
                "merged_from_source_digests": sorted(
                    {item.source_digest for item in group}
                ),
                "merged_from_parser_versions": sorted(
                    {item.parser_version for item in group}
                ),
            }
            payload = dict(merged_payload)
            payload["_import_merge"] = provenance
            winner = ordered[0]
            accepted.append(
                ParsedRecord(
                    domain=domain,
                    record_id=record_id,
                    payload=MappingProxyType(payload),
                    source_id=winner.source_id,
                    source_digest=winner.source_digest,
                    parser_version=winner.parser_version,
                    media_type=winner.media_type,
                    line_or_index=winner.line_or_index,
                )
            )
            conflicts.append(
                ConflictReport(
                    domain=domain,
                    record_id=record_id,
                    policy=policy.value,
                    decision="merged",
                    source_ids=source_ids,
                    selected_source_id=winner.source_id,
                    reason="complementary fields merged deterministically",
                )
            )
            continue

        raise ImportConflictError(f"unknown conflict policy: {policy}")

    return accepted, rejected, quarantined, conflicts


# ---------------------------------------------------------------------------
# Durable store
# ---------------------------------------------------------------------------


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split simple semicolon-delimited DDL for DuckDB execute()."""

    statements: list[str] = []
    for chunk in sql_text.split(";"):
        statement = chunk.strip()
        if statement:
            statements.append(statement)
    return statements


class ImportStore:
    """DuckDB-backed durable store for import receipts and accepted rows."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> Any:
        duckdb = _require_duckdb()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.database_path))

    def _ensure_schema(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                # DuckDB execute() accepts one statement at a time.
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
            finally:
                connection.close()

    def get_receipt(self, import_id: str) -> ImportReceipt | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT body_json FROM import_receipts WHERE import_id = ?",
                    [import_id],
                ).fetchone()
            finally:
                connection.close()
        if row is None:
            return None
        return ImportReceipt.from_dict(json.loads(row[0]))

    def list_accepted(self, import_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            connection = self._connect()
            try:
                if import_id is None:
                    rows = connection.execute(
                        "SELECT body_json FROM imported_rows ORDER BY domain, record_id"
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT body_json FROM imported_rows "
                        "WHERE import_id = ? ORDER BY domain, record_id",
                        [import_id],
                    ).fetchall()
            finally:
                connection.close()
        return [json.loads(row[0]) for row in rows]

    def apply_atomic(
        self,
        receipt: ImportReceipt,
        *,
        source_digests_before: Mapping[str, str],
        source_paths: Mapping[str, Path],
    ) -> ImportReceipt:
        """Commit receipt + rows atomically, or roll back entirely.

        Also re-checks source digests immediately before commit to enforce
        source immutability and exact-replay provenance.
        """

        for source_id, path in source_paths.items():
            if not path.is_file():
                raise ImportSourceMutationError(
                    f"source disappeared before commit: {source_id}"
                )
            current = _sha256_file(path)
            expected = source_digests_before[source_id]
            if current != expected:
                raise ImportSourceMutationError(
                    f"source mutated during import: {source_id} "
                    f"(was {expected}, now {current})"
                )

        body = receipt.to_dict()
        body_json = _stable_json(body)

        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN TRANSACTION")
                existing = connection.execute(
                    "SELECT body_json FROM import_receipts WHERE import_id = ?",
                    [receipt.import_id],
                ).fetchone()
                if existing is not None:
                    prior = ImportReceipt.from_dict(json.loads(existing[0]))
                    connection.execute("ROLLBACK")
                    # Exact replay: same semantic receipt identity.
                    if prior.manifest_cid != receipt.manifest_cid:
                        raise ImportConflictError(
                            f"import_id {receipt.import_id} already applied "
                            "with a different manifest"
                        )
                    return ImportReceipt(
                        receipt_cid=prior.receipt_cid,
                        import_id=prior.import_id,
                        manifest_cid=prior.manifest_cid,
                        mode=receipt.mode,
                        outcome=OUTCOME_REPLAYED,
                        parser_version=prior.parser_version,
                        started_at=prior.started_at,
                        finished_at=prior.finished_at,
                        source_observations=prior.source_observations,
                        accepted_rows=prior.accepted_rows,
                        rejected_rows=prior.rejected_rows,
                        quarantined_rows=prior.quarantined_rows,
                        conflicts=prior.conflicts,
                        strict=prior.strict,
                        applied=True,
                        replayed=True,
                        error_text="",
                    )

                connection.execute(
                    """
                    INSERT INTO import_receipts (
                        receipt_cid, import_id, manifest_cid, mode, outcome,
                        parser_version, started_at, finished_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt.receipt_cid,
                        receipt.import_id,
                        receipt.manifest_cid,
                        receipt.mode,
                        receipt.outcome,
                        receipt.parser_version,
                        receipt.started_at,
                        receipt.finished_at,
                        body_json,
                    ],
                )
                for observation in receipt.source_observations:
                    connection.execute(
                        """
                        INSERT INTO import_source_observations (
                            source_id, import_id, path, media_type, source_digest,
                            parser_version, observed_at, record_count,
                            accepted_count, rejected_count, quarantined_count,
                            body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            observation.source_id,
                            receipt.import_id,
                            observation.path,
                            observation.media_type,
                            observation.source_digest,
                            observation.parser_version,
                            observation.observed_at,
                            observation.record_count,
                            observation.accepted_count,
                            observation.rejected_count,
                            observation.quarantined_count,
                            _stable_json(observation.to_dict()),
                        ],
                    )
                for row in receipt.accepted_rows:
                    row_cid = str(row.get("content_cid") or canonical_content_cid(dict(row)))
                    connection.execute(
                        """
                        INSERT INTO imported_rows (
                            row_cid, import_id, domain, record_id, source_id,
                            source_digest, parser_version, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            row_cid,
                            receipt.import_id,
                            str(row["domain"]),
                            str(row["record_id"]),
                            str(row["source_id"]),
                            str(row["source_digest"]),
                            str(row["parser_version"]),
                            _stable_json(dict(row)),
                        ],
                    )
                for row in receipt.quarantined_rows:
                    quarantine_cid = canonical_content_cid(
                        {
                            "import_id": receipt.import_id,
                            "row": dict(row),
                        }
                    )
                    connection.execute(
                        """
                        INSERT INTO quarantined_rows (
                            quarantine_cid, import_id, domain, record_id,
                            source_id, source_digest, parser_version, reason,
                            body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            quarantine_cid,
                            receipt.import_id,
                            str(row.get("domain") or ""),
                            str(row.get("record_id") or ""),
                            str(row.get("source_id") or ""),
                            str(row.get("source_digest") or ""),
                            str(row.get("parser_version") or ""),
                            str(row.get("reject_reason") or "quarantined"),
                            _stable_json(dict(row)),
                        ],
                    )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                connection.close()
        return receipt


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class LegacyStateImport:
    """Preview/apply legacy state imports with provenance and reconciliation.

    Interface: LegacyStateImport@1
    """

    def __init__(
        self,
        *,
        target_database: Path | str | None = None,
        source_root: Path | str | None = None,
        store: ImportStore | None = None,
    ) -> None:
        self.target_database = (
            Path(target_database) if target_database is not None else None
        )
        self.source_root = Path(source_root) if source_root is not None else None
        self._store = store
        self._memory_receipts: dict[str, ImportReceipt] = {}
        self._memory_rows: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @property
    def interface(self) -> str:
        return LEGACY_STATE_IMPORT_INTERFACE

    def _resolve_store(self, manifest: ImportManifest) -> ImportStore | None:
        if self._store is not None:
            return self._store
        target = manifest.target_database or (
            str(self.target_database) if self.target_database is not None else ""
        )
        if not target:
            return None
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required when target_database is set for apply"
            )
        return ImportStore(target)

    def _source_paths(self, manifest: ImportManifest) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for source in manifest.sources:
            path = Path(source.path)
            if self.source_root is not None and not path.is_absolute():
                candidate = (self.source_root / path).resolve()
                root = self.source_root.resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise ImportSourceError(
                        f"source path escapes source_root: {source.path}"
                    ) from exc
                path = candidate
            paths[source.source_id] = path
        return paths

    def _assert_sources_immutable(
        self,
        paths: Mapping[str, Path],
        digests: Mapping[str, str],
    ) -> None:
        for source_id, path in paths.items():
            if not path.is_file():
                raise ImportSourceMutationError(
                    f"source missing or deleted: {source_id}"
                )
            # Refuse write-opens by checking digest stability only; we never
            # open sources for write.
            current = _sha256_file(path)
            if current != digests[source_id]:
                raise ImportSourceMutationError(
                    f"source mutated during import: {source_id}"
                )

    def preview(self, manifest: ImportManifest) -> ImportReceipt:
        """Parse and reconcile without writing to the target store."""

        working = ImportManifest(
            import_id=manifest.import_id,
            sources=manifest.sources,
            mode=ImportMode.PREVIEW,
            strict=manifest.strict,
            default_conflict_policy=manifest.default_conflict_policy,
            conflict_resolutions=manifest.conflict_resolutions,
            target_database=manifest.target_database,
            metadata=dict(manifest.metadata),
        )
        return self._run(working, apply=False)

    def apply(self, manifest: ImportManifest) -> ImportReceipt:
        """Strict-or-lax apply; exact replay is a no-op with the same receipt."""

        working = ImportManifest(
            import_id=manifest.import_id,
            sources=manifest.sources,
            mode=ImportMode.APPLY,
            strict=manifest.strict,
            default_conflict_policy=manifest.default_conflict_policy,
            conflict_resolutions=manifest.conflict_resolutions,
            target_database=manifest.target_database,
            metadata=dict(manifest.metadata),
        )
        return self._run(working, apply=True)

    def run(self, manifest: ImportManifest) -> ImportReceipt:
        """Dispatch by manifest.mode (defaults to preview)."""

        if manifest.mode is ImportMode.APPLY:
            return self.apply(manifest)
        return self.preview(manifest)

    def get_receipt(self, import_id: str) -> ImportReceipt | None:
        store = self._store
        if store is None and self.target_database is not None and duckdb_available():
            store = ImportStore(self.target_database)
        if store is not None:
            found = store.get_receipt(import_id)
            if found is not None:
                return found
        return self._memory_receipts.get(import_id)

    def list_accepted_rows(self, import_id: str | None = None) -> list[dict[str, Any]]:
        store = self._store
        if store is None and self.target_database is not None and duckdb_available():
            store = ImportStore(self.target_database)
        if store is not None:
            return store.list_accepted(import_id)
        if import_id is None:
            rows: list[dict[str, Any]] = []
            for batch in self._memory_rows.values():
                rows.extend(batch)
            return rows
        return list(self._memory_rows.get(import_id, []))

    def _run(self, manifest: ImportManifest, *, apply: bool) -> ImportReceipt:
        started_at = _utc_iso()
        store = self._resolve_store(manifest) if apply else None

        # Exact replay short-circuit for apply.
        if apply:
            prior = None
            if store is not None:
                prior = store.get_receipt(manifest.import_id)
            if prior is None:
                prior = self._memory_receipts.get(manifest.import_id)
            if prior is not None:
                if prior.manifest_cid != manifest.manifest_cid:
                    raise ImportConflictError(
                        f"import_id {manifest.import_id} already applied "
                        "with a different manifest"
                    )
                # Re-observe sources; digests must still match prior receipt.
                paths = self._source_paths(manifest)
                for observation in prior.source_observations:
                    path = paths.get(observation.source_id)
                    if path is None or not path.is_file():
                        raise ImportSourceMutationError(
                            f"source missing on replay: {observation.source_id}"
                        )
                    current = _sha256_file(path)
                    if current != observation.source_digest:
                        raise ImportSourceMutationError(
                            f"source mutated since prior import: "
                            f"{observation.source_id}"
                        )
                return ImportReceipt(
                    receipt_cid=prior.receipt_cid,
                    import_id=prior.import_id,
                    manifest_cid=prior.manifest_cid,
                    mode=ImportMode.APPLY.value,
                    outcome=OUTCOME_REPLAYED,
                    parser_version=prior.parser_version,
                    started_at=prior.started_at,
                    finished_at=prior.finished_at,
                    source_observations=prior.source_observations,
                    accepted_rows=prior.accepted_rows,
                    rejected_rows=prior.rejected_rows,
                    quarantined_rows=prior.quarantined_rows,
                    conflicts=prior.conflicts,
                    strict=prior.strict,
                    applied=True,
                    replayed=True,
                    error_text="",
                )

        observations: list[SourceObservation] = []
        all_records: list[ParsedRecord] = []
        source_digests: dict[str, str] = {}
        source_paths = self._source_paths(manifest)

        for spec in manifest.sources:
            observation, records = parse_source(spec, source_root=self.source_root)
            observations.append(observation)
            all_records.extend(records)
            source_digests[spec.source_id] = observation.source_digest

        accepted, rejected, quarantined, conflicts = reconcile_records(
            all_records,
            default_policy=manifest.default_conflict_policy,
            resolutions=manifest.conflict_resolutions,
        )

        # Annotate observations with disposition counts.
        accepted_by_source: dict[str, int] = {}
        rejected_by_source: dict[str, int] = {}
        quarantined_by_source: dict[str, int] = {}
        for item in accepted:
            accepted_by_source[item.source_id] = (
                accepted_by_source.get(item.source_id, 0) + 1
            )
        for item in rejected:
            rejected_by_source[item.source_id] = (
                rejected_by_source.get(item.source_id, 0) + 1
            )
        for item in quarantined:
            quarantined_by_source[item.source_id] = (
                quarantined_by_source.get(item.source_id, 0) + 1
            )
        annotated_observations = tuple(
            SourceObservation(
                source_id=item.source_id,
                path=item.path,
                media_type=item.media_type,
                source_digest=item.source_digest,
                parser_version=item.parser_version,
                observed_at=item.observed_at,
                byte_size=item.byte_size,
                record_count=item.record_count,
                accepted_count=accepted_by_source.get(item.source_id, 0),
                rejected_count=rejected_by_source.get(item.source_id, 0),
                quarantined_count=quarantined_by_source.get(item.source_id, 0),
            )
            for item in observations
        )

        accepted_rows = tuple(
            MappingProxyType(item.to_dict()) for item in accepted
        )
        rejected_rows = tuple(
            MappingProxyType(item.to_dict()) for item in rejected
        )
        quarantined_rows = tuple(
            MappingProxyType(item.to_dict()) for item in quarantined
        )

        error_text = ""
        outcome = OUTCOME_PREVIEWED if not apply else OUTCOME_APPLIED
        if manifest.strict and (rejected or any(
            report.decision in {"rejected"} for report in conflicts
        )):
            # Strict mode: any rejected row fails apply (preview still reports).
            if apply:
                outcome = OUTCOME_REJECTED
                error_text = (
                    f"strict import refused: {len(rejected)} rejected row(s)"
                )
            elif rejected:
                error_text = (
                    f"strict preview reports {len(rejected)} rejected row(s)"
                )

        finished_at = _utc_iso()
        provisional = {
            "schema": IMPORT_RECEIPT_SCHEMA,
            "interface": IMPORT_RECEIPT_INTERFACE,
            "import_id": manifest.import_id,
            "manifest_cid": manifest.manifest_cid,
            "parser_version": PARSER_VERSION,
            "source_observations": [
                item.to_dict() for item in annotated_observations
            ],
            "accepted_rows": [dict(item) for item in accepted_rows],
            "rejected_rows": [dict(item) for item in rejected_rows],
            "quarantined_rows": [dict(item) for item in quarantined_rows],
            "conflicts": [item.to_dict() for item in conflicts],
            "strict": bool(manifest.strict),
        }
        receipt_cid = _receipt_cid_from_body(provisional)

        receipt = ImportReceipt(
            receipt_cid=receipt_cid,
            import_id=manifest.import_id,
            manifest_cid=manifest.manifest_cid,
            mode=manifest.mode.value,
            outcome=outcome,
            parser_version=PARSER_VERSION,
            started_at=started_at,
            finished_at=finished_at,
            source_observations=annotated_observations,
            accepted_rows=accepted_rows,
            rejected_rows=rejected_rows,
            quarantined_rows=quarantined_rows,
            conflicts=tuple(conflicts),
            strict=bool(manifest.strict),
            applied=False,
            replayed=False,
            error_text=error_text,
        )

        if not apply:
            return receipt

        if outcome == OUTCOME_REJECTED:
            raise ImportStrictError(error_text or "strict import refused")

        # Re-check source immutability before commit.
        self._assert_sources_immutable(source_paths, source_digests)

        applied_receipt = ImportReceipt(
            receipt_cid=receipt.receipt_cid,
            import_id=receipt.import_id,
            manifest_cid=receipt.manifest_cid,
            mode=ImportMode.APPLY.value,
            outcome=OUTCOME_APPLIED,
            parser_version=receipt.parser_version,
            started_at=receipt.started_at,
            finished_at=_utc_iso(),
            source_observations=receipt.source_observations,
            accepted_rows=receipt.accepted_rows,
            rejected_rows=receipt.rejected_rows,
            quarantined_rows=receipt.quarantined_rows,
            conflicts=receipt.conflicts,
            strict=receipt.strict,
            applied=True,
            replayed=False,
            error_text="",
        )

        if store is not None:
            try:
                committed = store.apply_atomic(
                    applied_receipt,
                    source_digests_before=source_digests,
                    source_paths=source_paths,
                )
            except Exception as exc:
                raise ImportAtomicityError(
                    f"strict import rolled back: {exc}"
                ) from exc
            with self._lock:
                self._memory_receipts[committed.import_id] = committed
                self._memory_rows[committed.import_id] = [
                    dict(row) for row in committed.accepted_rows
                ]
            return committed

        # In-memory apply for hermetic tests without a DuckDB target.
        with self._lock:
            self._memory_receipts[applied_receipt.import_id] = applied_receipt
            self._memory_rows[applied_receipt.import_id] = [
                dict(row) for row in applied_receipt.accepted_rows
            ]
        return applied_receipt


def build_import_manifest(
    import_id: str,
    sources: Sequence[Mapping[str, Any] | ImportSourceSpec],
    **kwargs: Any,
) -> ImportManifest:
    """Convenience constructor for tests and operators."""

    resolved: list[ImportSourceSpec] = []
    for item in sources:
        if isinstance(item, ImportSourceSpec):
            resolved.append(item)
        else:
            resolved.append(ImportSourceSpec.from_dict(item))
    return ImportManifest(
        import_id=import_id,
        sources=tuple(resolved),
        **kwargs,
    )


__all__ = [
    "LEGACY_STATE_IMPORT_INTERFACE",
    "IMPORT_MANIFEST_INTERFACE",
    "IMPORT_RECEIPT_INTERFACE",
    "LEGACY_STATE_IMPORT_SCHEMA",
    "IMPORT_MANIFEST_SCHEMA",
    "IMPORT_RECEIPT_SCHEMA",
    "PARSER_VERSION",
    "OUTCOME_PREVIEWED",
    "OUTCOME_APPLIED",
    "OUTCOME_REPLAYED",
    "OUTCOME_REJECTED",
    "OUTCOME_FAILED",
    "LegacyStateImportError",
    "ImportManifestError",
    "ImportSourceError",
    "ImportConflictError",
    "ImportStrictError",
    "ImportAtomicityError",
    "ImportSourceMutationError",
    "DuckDBUnavailableError",
    "ImportMediaType",
    "ImportMode",
    "ConflictPolicy",
    "ImportDomain",
    "ImportConflictResolution",
    "ImportSourceSpec",
    "ImportManifest",
    "ParsedRecord",
    "SourceObservation",
    "ConflictReport",
    "ImportReceipt",
    "ImportStore",
    "LegacyStateImport",
    "build_import_manifest",
    "duckdb_available",
    "parse_source",
    "parse_json_records",
    "parse_jsonl_records",
    "parse_markdown_records",
    "parse_sqlite_records",
    "parse_duckdb_records",
    "reconcile_records",
]
