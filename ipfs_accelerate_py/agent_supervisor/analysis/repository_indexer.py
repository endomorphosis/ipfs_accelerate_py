"""Complete, incremental repository index for SwissKnife contract assurance.

The indexer composes the exact :mod:`repository_snapshot` path ledger with the
path-independent polyglot AST producer, :class:`AnalysisASTIndex`, the compact
analysis cache, and the supervisor's bounded artifact CAS.  A repository path
row never contains source text or an AST body.  Those immutable bodies are
stored once in CAS and rows retain only integrity-checkable shallow references.

Every snapshot disposition produces exactly one row.  Semantic and supported
structured documents additionally produce an ``ASTBlobRecord`` (including a
typed parse-error record on failure).  Unchanged content can therefore be
reused at the same path or after a rename without invoking a parser.

There is one mutable ``current.json`` per index root.  Writers serialize builds
with a process-safe lock and publish by ``os.replace``; readers only open
immutable CAS objects and an atomically published manifest, so they never need
to block a scan.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..core.conflict_graph import ASTBlobRecord
from ..runtime.artifact_store import (
    ArtifactBlobIntegrityError,
    ArtifactQuotaPolicy,
    BlobReference,
    BoundedArtifactStore,
    RetentionClass,
)
from .analysis_ast_index import (
    ASTBlobInvalidation,
    AnalysisASTIndex,
    AnalysisASTIndexStats,
    IndexedASTPath,
    build_analysis_ast_index,
)
from .analysis_cache import (
    AnalysisCache,
    AnalysisOutcome,
    build_analysis_cache_key,
)
from .analyzer_health import (
    AnalyzerHealthReport,
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
    classify_analyzer_health,
)
from .polyglot_ast_provider import (
    POLYGLOT_AST_PROVIDER_SCHEMA,
    PolyglotASTProvider,
    PolyglotASTProviderError,
    language_for_path,
)
from .repository_snapshot import (
    CoverageDisposition,
    CoverageKind,
    DependencyIdentity,
    DependencyIdentityKind,
    EntryKind,
    GitStatus,
    GitlinkRecord,
    RepositorySnapshot,
    RepositorySnapshotStats,
    ScopePolicy,
    build_repository_snapshot,
)


REPOSITORY_INDEXER_VERSION: Final = "repository-indexer@1"
REPOSITORY_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-repository-index@1"
)
REPOSITORY_INDEX_ROW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-repository-index-row@1"
)
REPOSITORY_INDEX_STATS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-repository-index-stats@1"
)
REPOSITORY_INDEX_CACHE_OBJECTIVE: Final = "sca-g021-whole-tree-index@1"
REPOSITORY_INDEX_CACHE_SCHEMA: Final = "ast-blob-record@1"

DEFAULT_MAX_COMPACT_ROW_BYTES: Final = 4_096
HARD_MAX_COMPACT_ROW_BYTES: Final = 65_536
DEFAULT_MAX_PARSE_ERROR_BYTES: Final = 1_024
DEFAULT_MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
DEFAULT_MAX_INDEX_PATHS: Final = 100_000
DEFAULT_CAS_MAX_BYTES: Final = 2 * 1024 * 1024 * 1024
DEFAULT_CAS_MAX_BLOBS: Final = 250_000

_DELETED_STATUSES = frozenset(
    {GitStatus.DELETED.value, GitStatus.STAGED_DELETION.value}
)
_PARSER_KINDS = frozenset(
    {CoverageKind.SEMANTIC_AST, CoverageKind.STRUCTURED_DATA}
)
_SUPPORTED_STRUCTURED_SUFFIXES = frozenset({".json"})
_FORBIDDEN_BODY_KEYS = frozenset(
    {
        "body",
        "bytes",
        "contents",
        "file_contents",
        "source",
        "source_body",
        "source_code",
        "source_contents",
        "source_text",
        "ast",
        "ast_body",
    }
)


class RepositoryIndexerError(RuntimeError):
    """Base exception for an incomplete or corrupt repository index."""


class RepositoryIndexIntegrityError(RepositoryIndexerError, ValueError):
    """A persisted row, manifest, or CAS object failed exact verification."""


class RepositoryIndexSourceChanged(RepositoryIndexerError):
    """A source changed after its snapshot digest was established."""


class RepositoryIndexBoundsExceeded(RepositoryIndexerError, ValueError):
    """A resource or compact-row bound was exceeded."""


class RepositoryIndexUnavailable(RepositoryIndexerError):
    """No current repository index is available."""


class ParserStatus(str, Enum):
    """Complete per-path parser outcome vocabulary."""

    INDEXED = "indexed"
    CACHE_HIT = "cache_hit"
    PARSE_FAILURE = "parse_failure"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"
    DELETED = "deleted"


def canonical_repository_index_bytes(value: Any) -> bytes:
    """Encode deterministic JSON and reject non-portable values."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not (float("-inf") < item < float("inf")):
                raise RepositoryIndexIntegrityError(
                    "canonical JSON cannot contain NaN or infinity"
                )
            return item
        if isinstance(item, Enum):
            return normalize(item.value)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise RepositoryIndexIntegrityError(
                    "canonical JSON keys must be strings"
                )
            return {
                key: normalize(item[key])
                for key in sorted(item)
            }
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            return normalize(converter())
        raise RepositoryIndexIntegrityError(
            f"unsupported canonical value: {type(item).__name__}"
        )

    try:
        return json.dumps(
            normalize(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, RepositoryIndexIntegrityError):
            raise
        raise RepositoryIndexIntegrityError(
            "repository index must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    return (
        f"{prefix}:sha256:"
        + hashlib.sha256(canonical_repository_index_bytes(value)).hexdigest()
    )


def _normalize_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in raw
    ):
        raise RepositoryIndexIntegrityError(
            f"invalid repository index path: {value!r}"
        )
    normalized = path.as_posix()
    if normalized != raw.rstrip("/"):
        raise RepositoryIndexIntegrityError(
            f"non-canonical repository index path: {value!r}"
        )
    return normalized


def _bounded_text(value: Any, maximum: int) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= maximum:
        return text
    marker = "…[truncated]"
    budget = max(0, maximum - len(marker.encode("utf-8")))
    return encoded[:budget].decode("utf-8", "ignore") + marker


def _contains_body(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_BODY_KEYS:
                return True
            if _contains_body(child):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_body(item) for item in value)
    return False


def _reference_dict(
    value: BlobReference | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    reference = (
        value if isinstance(value, BlobReference) else BlobReference.from_dict(value)
    )
    return reference.to_dict()


@dataclass(frozen=True)
class RepositoryIndexRow:
    """One bounded, body-free path-ledger row."""

    path: str
    disposition_kind: CoverageKind | str
    declared_kind: CoverageKind | str
    reason_code: str
    policy_rule: str
    git_status: GitStatus | str
    content_digest: str = ""
    source_ref: Mapping[str, Any] | None = None
    ast_ref: Mapping[str, Any] | None = None
    ast_record_id: str = ""
    language: str = ""
    parser_status: ParserStatus | str = ParserStatus.NOT_APPLICABLE
    parser_reason: str = ""
    parser_identity: str = ""
    reused_from_path: str = ""
    tracked: bool = True
    overlay: bool = False
    max_row_bytes: int = field(
        default=DEFAULT_MAX_COMPACT_ROW_BYTES, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(
            self, "disposition_kind", CoverageKind(self.disposition_kind)
        )
        object.__setattr__(self, "declared_kind", CoverageKind(self.declared_kind))
        object.__setattr__(self, "git_status", GitStatus(self.git_status))
        object.__setattr__(self, "parser_status", ParserStatus(self.parser_status))
        for name in (
            "reason_code",
            "policy_rule",
            "content_digest",
            "ast_record_id",
            "language",
            "parser_identity",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        object.__setattr__(
            self,
            "parser_reason",
            _bounded_text(self.parser_reason, DEFAULT_MAX_PARSE_ERROR_BYTES),
        )
        if self.reused_from_path:
            object.__setattr__(
                self, "reused_from_path", _normalize_path(self.reused_from_path)
            )
        object.__setattr__(self, "source_ref", _reference_dict(self.source_ref))
        object.__setattr__(self, "ast_ref", _reference_dict(self.ast_ref))
        maximum = int(self.max_row_bytes)
        if not 256 <= maximum <= HARD_MAX_COMPACT_ROW_BYTES:
            raise RepositoryIndexBoundsExceeded(
                "max_row_bytes must be between 256 and "
                f"{HARD_MAX_COMPACT_ROW_BYTES}"
            )
        if not self.reason_code or not self.policy_rule:
            raise RepositoryIndexIntegrityError(
                f"path row lacks an explicit disposition reason: {self.path}"
            )
        if self.parser_status in {
            ParserStatus.INDEXED,
            ParserStatus.CACHE_HIT,
            ParserStatus.PARSE_FAILURE,
        }:
            if self.source_ref is None or self.ast_ref is None:
                raise RepositoryIndexIntegrityError(
                    f"parsed row lacks CAS references: {self.path}"
                )
            if not self.ast_record_id or not self.parser_identity:
                raise RepositoryIndexIntegrityError(
                    f"parsed row lacks AST/parser identity: {self.path}"
                )
        payload = self.to_dict()
        if _contains_body(payload):
            raise RepositoryIndexIntegrityError(
                f"compact row embeds a source or AST body: {self.path}"
            )
        size = len(canonical_repository_index_bytes(payload))
        if size > maximum:
            raise RepositoryIndexBoundsExceeded(
                f"compact row for {self.path} is {size} bytes; limit is {maximum}"
            )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": REPOSITORY_INDEX_ROW_SCHEMA,
            "path": self.path,
            "disposition_kind": self.disposition_kind.value,
            "declared_kind": self.declared_kind.value,
            "reason_code": self.reason_code,
            "policy_rule": self.policy_rule,
            "git_status": self.git_status.value,
            "content_digest": self.content_digest,
            "source_ref": dict(self.source_ref) if self.source_ref else None,
            "ast_ref": dict(self.ast_ref) if self.ast_ref else None,
            "ast_record_id": self.ast_record_id,
            "language": self.language,
            "parser_status": self.parser_status.value,
            "parser_reason": self.parser_reason,
            "parser_identity": self.parser_identity,
            "reused_from_path": self.reused_from_path,
            "tracked": bool(self.tracked),
            "overlay": bool(self.overlay),
        }

    @property
    def row_id(self) -> str:
        return _identity("sca-repository-index-row", self._content_dict())

    @property
    def serialized_size(self) -> int:
        return len(canonical_repository_index_bytes(self.to_dict()))

    @property
    def source_blob_ref(self) -> Mapping[str, Any] | None:
        return self.source_ref

    @property
    def ast_record_ref(self) -> Mapping[str, Any] | None:
        return self.ast_ref

    def to_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, **self._content_dict()}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        max_row_bytes: int = DEFAULT_MAX_COMPACT_ROW_BYTES,
    ) -> "RepositoryIndexRow":
        if not isinstance(value, Mapping):
            raise RepositoryIndexIntegrityError("path row must be an object")
        result = cls(
            path=value.get("path", ""),
            disposition_kind=value.get("disposition_kind", ""),
            declared_kind=value.get(
                "declared_kind", value.get("disposition_kind", "")
            ),
            reason_code=value.get("reason_code", ""),
            policy_rule=value.get("policy_rule", ""),
            git_status=value.get("git_status", ""),
            content_digest=value.get("content_digest", ""),
            source_ref=value.get("source_ref"),
            ast_ref=value.get("ast_ref"),
            ast_record_id=value.get("ast_record_id", ""),
            language=value.get("language", ""),
            parser_status=value.get(
                "parser_status", ParserStatus.NOT_APPLICABLE.value
            ),
            parser_reason=value.get("parser_reason", ""),
            parser_identity=value.get("parser_identity", ""),
            reused_from_path=value.get("reused_from_path", ""),
            tracked=bool(value.get("tracked", True)),
            overlay=bool(value.get("overlay", False)),
            max_row_bytes=max_row_bytes,
        )
        claimed = str(value.get("row_id") or "")
        if claimed and claimed != result.row_id:
            raise RepositoryIndexIntegrityError(
                f"path row identity mismatch: {result.path}"
            )
        return result


@dataclass(frozen=True)
class RepositoryIndexStats:
    """Operational accounting for one build (not part of index identity)."""

    snapshot_path_count: int
    tracked_path_count: int
    row_count: int
    eligible_parser_path_count: int
    indexed_path_count: int
    reused_path_count: int
    renamed_reuse_count: int
    parsed_path_count: int
    parse_failure_count: int
    unsupported_parser_count: int
    deleted_path_count: int
    invalidated_path_count: int
    corruption_recovery_count: int
    source_blob_write_count: int
    ast_blob_write_count: int
    max_row_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = int(getattr(self, name))
            if value < 0:
                raise RepositoryIndexIntegrityError(
                    f"negative repository index statistic: {name}"
                )
            object.__setattr__(self, name, value)
        if self.row_count != self.snapshot_path_count:
            raise RepositoryIndexIntegrityError(
                "repository index silently skipped one or more snapshot paths"
            )

    @property
    def cache_hit_ratio(self) -> float:
        return (
            self.reused_path_count / self.eligible_parser_path_count
            if self.eligible_parser_path_count
            else 0.0
        )

    @property
    def cache_hit_count(self) -> int:
        return self.reused_path_count

    @property
    def reused_blob_count(self) -> int:
        return self.reused_path_count

    @property
    def new_blob_count(self) -> int:
        return self.parsed_path_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPOSITORY_INDEX_STATS_SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            "cache_hit_ratio": self.cache_hit_ratio,
        }

    @classmethod
    def empty_for_rows(
        cls, rows: Sequence[RepositoryIndexRow]
    ) -> "RepositoryIndexStats":
        tracked = sum(1 for row in rows if row.tracked)
        eligible = sum(
            1
            for row in rows
            if row.parser_status
            in {
                ParserStatus.INDEXED,
                ParserStatus.CACHE_HIT,
                ParserStatus.PARSE_FAILURE,
            }
        )
        return cls(
            snapshot_path_count=len(rows),
            tracked_path_count=tracked,
            row_count=len(rows),
            eligible_parser_path_count=eligible,
            indexed_path_count=eligible,
            reused_path_count=0,
            renamed_reuse_count=0,
            parsed_path_count=0,
            parse_failure_count=sum(
                row.parser_status is ParserStatus.PARSE_FAILURE for row in rows
            ),
            unsupported_parser_count=sum(
                row.parser_status is ParserStatus.UNSUPPORTED for row in rows
            ),
            deleted_path_count=sum(
                row.parser_status is ParserStatus.DELETED for row in rows
            ),
            invalidated_path_count=0,
            corruption_recovery_count=0,
            source_blob_write_count=0,
            ast_blob_write_count=0,
            max_row_bytes=max(
                (row.serialized_size for row in rows), default=0
            ),
        )


@dataclass(frozen=True)
class RepositoryIndex:
    """Current semantic repository index plus build-local audit evidence."""

    snapshot: RepositorySnapshot
    rows: tuple[RepositoryIndexRow, ...]
    ast_index: AnalysisASTIndex
    health: AnalyzerHealthReport
    build_stats: RepositoryIndexStats
    invalidations: tuple[ASTBlobInvalidation, ...] = ()

    def __post_init__(self) -> None:
        rows = tuple(sorted(self.rows, key=lambda item: item.path))
        if len(rows) != len({item.path for item in rows}):
            raise RepositoryIndexIntegrityError(
                "repository index paths must be unique"
            )
        if len(rows) != len(self.snapshot.dispositions):
            raise RepositoryIndexIntegrityError(
                "repository index does not account for every snapshot path"
            )
        if tuple(row.path for row in rows) != tuple(
            item.path for item in self.snapshot.dispositions
        ):
            raise RepositoryIndexIntegrityError(
                "repository index row ledger differs from snapshot ledger"
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "invalidations",
            tuple(
                sorted(
                    {
                        item.invalidation_id: item
                        for item in self.invalidations
                    }.values(),
                    key=lambda item: item.invalidation_id,
                )
            ),
        )

    def _content_dict(self) -> dict[str, Any]:
        # Build-local cache outcomes and invalidation transition history are
        # deliberately excluded.  Cold and warm builds of one exact snapshot
        # serialize to the same bytes.
        return {
            "schema": REPOSITORY_INDEX_SCHEMA,
            "indexer_version": REPOSITORY_INDEXER_VERSION,
            "snapshot": self.snapshot.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "ast_index_id": self.ast_index.index_id,
            "health": self.health.to_dict(),
        }

    @property
    def index_id(self) -> str:
        return _identity("sca-repository-index", self._content_dict())

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return bool(
            self.health.status is AnalyzerHealthStatus.HEALTHY
            and self.build_stats.row_count
            == self.snapshot.stats.disposition_count
        )

    @property
    def path_count(self) -> int:
        return len(self.rows)

    @property
    def path_rows(self) -> tuple[RepositoryIndexRow, ...]:
        return self.rows

    @property
    def coverage_rows(self) -> tuple[RepositoryIndexRow, ...]:
        return self.rows

    @property
    def stats(self) -> RepositoryIndexStats:
        return self.build_stats

    @property
    def analyzer_health(self) -> AnalyzerHealthReport:
        return self.health

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id

    @property
    def ast_index_id(self) -> str:
        return self.ast_index.index_id

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_repository_index_bytes(self.to_dict())

    def row_for_path(self, path: str) -> RepositoryIndexRow | None:
        normalized = _normalize_path(path)
        return next((row for row in self.rows if row.path == normalized), None)

    def to_dict(self) -> dict[str, Any]:
        return {"index_id": self.index_id, **self._content_dict()}

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return self.canonical_bytes.decode("utf-8")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )


class RepositoryCAS:
    """Small recovery adapter over the supervisor bounded artifact CAS."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_bytes: int = DEFAULT_CAS_MAX_BYTES,
        max_blobs: int = DEFAULT_CAS_MAX_BLOBS,
        max_blob_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    ) -> None:
        self.root = Path(root)
        self.store = BoundedArtifactStore(
            self.root,
            quotas=ArtifactQuotaPolicy(
                max_bytes=max_bytes,
                max_blobs=max_blobs,
                max_projections=4_096,
                max_blob_bytes=max_blob_bytes,
            ),
        )
        self.corruption_recoveries = 0

    def put(
        self,
        payload: bytes,
        *,
        kind: str,
        media_type: str,
    ) -> BlobReference:
        try:
            return self.store.put_blob(
                payload,
                kind=kind,
                retention_class=RetentionClass.AUTHORITATIVE,
                media_type=media_type,
            )
        except ArtifactBlobIntegrityError:
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            artifact_id = f"blob:{digest}"
            # The bounded store intentionally refuses to overwrite immutable
            # content.  A verified corrupt object is not immutable content, so
            # remove only that exact digest under the store lock and republish.
            with self.store._locked():
                metadata = self.store._manifest["blobs"].pop(
                    artifact_id, None
                )
                if metadata is not None:
                    try:
                        reference = BlobReference.from_dict(metadata)
                        self.store._blob_path(reference).unlink()
                    except (OSError, TypeError, ValueError):
                        pass
                    self.store._write_manifest(self.store._manifest)
            self.corruption_recoveries += 1
            return self.store.put_blob(
                payload,
                kind=kind,
                retention_class=RetentionClass.AUTHORITATIVE,
                media_type=media_type,
            )

    def verify(self, reference: Mapping[str, Any] | BlobReference) -> bool:
        return self.store.verify_blob(reference)

    def read(
        self, reference: Mapping[str, Any] | BlobReference
    ) -> bytes:
        return self.store.read_blob(reference)

    def put_json(self, payload: Mapping[str, Any], *, kind: str) -> BlobReference:
        return self.put(
            canonical_repository_index_bytes(payload),
            kind=kind,
            media_type="application/json",
        )

    def read_json(
        self, reference: Mapping[str, Any] | BlobReference
    ) -> Mapping[str, Any]:
        try:
            value = json.loads(self.read(reference))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RepositoryIndexIntegrityError(
                "CAS JSON object is corrupt"
            ) from exc
        if not isinstance(value, Mapping):
            raise RepositoryIndexIntegrityError(
                "CAS JSON object must contain a mapping"
            )
        return value

    def close(self) -> None:
        self.store.close()


SourceLoader = Callable[[CoverageDisposition], bytes]


class RepositoryIndexer:
    """Build and atomically publish one complete current repository index."""

    def __init__(
        self,
        index_root: Path | str,
        *,
        provider: PolyglotASTProvider | None = None,
        cache: AnalysisCache | None = None,
        cas: RepositoryCAS | None = None,
        health_thresholds: AnalyzerHealthThresholds
        | Mapping[str, Any]
        | None = None,
        max_compact_row_bytes: int = DEFAULT_MAX_COMPACT_ROW_BYTES,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
        max_paths: int = DEFAULT_MAX_INDEX_PATHS,
    ) -> None:
        self.index_root = Path(index_root)
        self.index_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.current_path = self.index_root / "current.json"
        self.snapshots_path = self.index_root / "snapshots"
        self.snapshots_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path = self.index_root / ".repository-index.lock"
        self.provider = provider or PolyglotASTProvider()
        self.cache = cache or AnalysisCache(
            self.index_root / "analysis-cache",
            max_entries=max(512, min(max_paths, 100_000)),
            max_bytes=256 * 1024 * 1024,
        )
        self.cas = cas or RepositoryCAS(
            self.index_root / "cas",
            max_blob_bytes=max_source_bytes,
        )
        self.health_thresholds = AnalyzerHealthThresholds.from_value(
            health_thresholds
        )
        self.max_compact_row_bytes = int(max_compact_row_bytes)
        self.max_source_bytes = int(max_source_bytes)
        self.max_paths = int(max_paths)
        if not 256 <= self.max_compact_row_bytes <= HARD_MAX_COMPACT_ROW_BYTES:
            raise RepositoryIndexBoundsExceeded(
                "max_compact_row_bytes is outside the hard bound"
            )
        if self.max_source_bytes < 1 or self.max_paths < 1:
            raise RepositoryIndexBoundsExceeded(
                "max_source_bytes and max_paths must be positive"
            )
        self._thread_lock = threading.RLock()
        self.parser_identity = self._parser_identity()

    def _parser_identity(self) -> str:
        extractor_digest = ""
        extractor_path = getattr(self.provider, "extractor_path", None)
        if extractor_path:
            try:
                extractor_digest = "sha256:" + hashlib.sha256(
                    Path(extractor_path).read_bytes()
                ).hexdigest()
            except OSError:
                extractor_digest = "unavailable"
        limits = getattr(self.provider, "limits", None)
        return _identity(
            "sca-repository-parser",
            {
                "indexer_version": REPOSITORY_INDEXER_VERSION,
                "provider_schema": POLYGLOT_AST_PROVIDER_SCHEMA,
                "provider_class": (
                    f"{type(self.provider).__module__}."
                    f"{type(self.provider).__qualname__}"
                ),
                "limits": limits.to_dict() if limits is not None else {},
                "node_executable": getattr(
                    self.provider, "node_executable", ""
                ),
                "typescript_path": getattr(
                    self.provider, "typescript_path", ""
                ),
                "expected_typescript_version": getattr(
                    self.provider, "expected_typescript_version", ""
                ),
                "extractor_digest": extractor_digest,
            },
        )

    def _exclusive_build(self):
        class _Lock:
            def __init__(inner, outer: "RepositoryIndexer") -> None:
                inner.outer = outer
                inner.handle: Any = None

            def __enter__(inner) -> None:
                inner.outer._thread_lock.acquire()
                inner.handle = inner.outer.lock_path.open("a+b")
                fcntl.flock(inner.handle.fileno(), fcntl.LOCK_EX)

            def __exit__(inner, *_args: Any) -> None:
                try:
                    fcntl.flock(inner.handle.fileno(), fcntl.LOCK_UN)
                    inner.handle.close()
                finally:
                    inner.outer._thread_lock.release()

        return _Lock(self)

    def _default_source_loader(
        self, snapshot: RepositorySnapshot
    ) -> SourceLoader:
        root = Path(snapshot.repository_root)

        def load(disposition: CoverageDisposition) -> bytes:
            path = root.joinpath(*PurePosixPath(disposition.path).parts)
            try:
                before = path.stat()
                payload = path.read_bytes()
                after = path.stat()
            except OSError as exc:
                raise RepositoryIndexSourceChanged(
                    f"source is unavailable after snapshot: {disposition.path}"
                ) from exc
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RepositoryIndexSourceChanged(
                    f"source changed while being read: {disposition.path}"
                )
            return payload

        return load

    def _validate_source(
        self, disposition: CoverageDisposition, payload: bytes
    ) -> None:
        if len(payload) > self.max_source_bytes:
            raise RepositoryIndexBoundsExceeded(
                f"source exceeds {self.max_source_bytes} bytes: "
                f"{disposition.path}"
            )
        actual = "sha256:" + hashlib.sha256(payload).hexdigest()
        if disposition.content_digest and actual != disposition.content_digest:
            raise RepositoryIndexSourceChanged(
                f"snapshot/source digest mismatch at {disposition.path}"
            )

    @staticmethod
    def _language(disposition: CoverageDisposition) -> str:
        try:
            return language_for_path(disposition.path)
        except PolyglotASTProviderError:
            # The reviewed scope includes the TypeScript module spellings even
            # when an older producer's path helper only lists .ts/.tsx.
            return {
                ".cts": "typescript",
                ".mts": "typescript",
            }.get(PurePosixPath(disposition.path).suffix.casefold(), "")

    @staticmethod
    def _requires_parser(disposition: CoverageDisposition) -> bool:
        if disposition.kind not in _PARSER_KINDS:
            return False
        if disposition.git_status.value in _DELETED_STATUSES:
            return False
        if disposition.kind is CoverageKind.STRUCTURED_DATA:
            return (
                PurePosixPath(disposition.path).suffix.casefold()
                in _SUPPORTED_STRUCTURED_SUFFIXES
            )
        return True

    def _cache_key(
        self,
        disposition: CoverageDisposition,
        *,
        language: str,
        snapshot: RepositorySnapshot,
    ):
        return build_analysis_cache_key(
            repository_tree_identity=disposition.content_digest,
            objective_revision=REPOSITORY_INDEX_CACHE_OBJECTIVE,
            analyzer_version=self.parser_identity,
            schema_version=REPOSITORY_INDEX_CACHE_SCHEMA,
            configuration_digest={
                "language": language,
                "max_source_bytes": self.max_source_bytes,
            },
            query_digest=disposition.content_digest,
            policy_digest=snapshot.scope_policy_id,
        )

    def _load_ast_reference(
        self, reference: Mapping[str, Any] | None
    ) -> ASTBlobRecord | None:
        if not reference or not self.cas.verify(reference):
            return None
        try:
            return ASTBlobRecord.from_dict(self.cas.read_json(reference))
        except (TypeError, ValueError, RepositoryIndexerError):
            return None

    def _reuse_row(
        self,
        disposition: CoverageDisposition,
        candidates: Sequence[RepositoryIndexRow],
    ) -> tuple[RepositoryIndexRow, ASTBlobRecord] | None:
        for old in candidates:
            if (
                old.content_digest != disposition.content_digest
                or old.parser_identity != self.parser_identity
                or old.source_ref is None
                or old.ast_ref is None
                or not self.cas.verify(old.source_ref)
            ):
                continue
            record = self._load_ast_reference(old.ast_ref)
            if record is None or record.record_id != old.ast_record_id:
                continue
            status = (
                ParserStatus.PARSE_FAILURE
                if record.parse_error
                else ParserStatus.INDEXED
            )
            row = RepositoryIndexRow(
                path=disposition.path,
                disposition_kind=(
                    CoverageKind.PARSE_FAILURE
                    if record.parse_error
                    else disposition.kind
                ),
                declared_kind=disposition.kind,
                reason_code=(
                    "parser_reported_failure"
                    if record.parse_error
                    else disposition.reason_code
                ),
                policy_rule=disposition.policy_rule,
                git_status=disposition.git_status,
                content_digest=disposition.content_digest,
                source_ref=old.source_ref,
                ast_ref=old.ast_ref,
                ast_record_id=record.record_id,
                language=record.language,
                parser_status=status,
                parser_reason=record.parse_error,
                parser_identity=self.parser_identity,
                tracked=disposition.tracked,
                overlay=disposition.overlay,
                max_row_bytes=self.max_compact_row_bytes,
            )
            return row, record
        return None

    def _cached_record(
        self,
        disposition: CoverageDisposition,
        *,
        language: str,
        snapshot: RepositorySnapshot,
    ) -> tuple[ASTBlobRecord, Mapping[str, Any]] | None:
        lookup = self.cache.lookup(
            self._cache_key(
                disposition, language=language, snapshot=snapshot
            )
        )
        receipt = lookup.receipt
        if not lookup.hit or not isinstance(receipt, Mapping):
            return None
        refs = receipt.get("artifact_refs")
        if (
            not isinstance(refs, Sequence)
            or isinstance(refs, (str, bytes))
            or not refs
            or not isinstance(refs[0], Mapping)
        ):
            return None
        reference = refs[0]
        record = self._load_ast_reference(reference)
        if record is None:
            return None
        if (
            record.source_sha256 != disposition.content_digest
            or record.record_id != receipt.get("receipt_id")
        ):
            return None
        return record, reference

    def _store_cache(
        self,
        disposition: CoverageDisposition,
        *,
        language: str,
        snapshot: RepositorySnapshot,
        record: ASTBlobRecord,
        ast_ref: BlobReference,
    ) -> None:
        outcome = (
            AnalysisOutcome.PARTIAL
            if record.parse_error
            else AnalysisOutcome.SUCCESSFUL
        )
        stored = self.cache.put(
            self._cache_key(
                disposition, language=language, snapshot=snapshot
            ),
            {
                "status": outcome.value,
                "receipt_id": record.record_id,
                "summary": {
                    "language": record.language,
                    "parse_error": bool(record.parse_error),
                },
                "artifact_refs": [ast_ref.to_dict()],
            },
        )
        if not stored.stored:
            raise RepositoryIndexerError(
                "analysis cache rejected compact AST receipt: "
                + ",".join(stored.reason_codes)
            )

    def _parse(
        self,
        payload: bytes,
        disposition: CoverageDisposition,
        *,
        language: str,
    ) -> ASTBlobRecord:
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        blob_identity = f"blob:{digest}"
        try:
            source = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ASTBlobRecord(
                blob_identity=blob_identity,
                source_sha256=digest,
                parse_error=(
                    f"UnicodeDecodeError at byte {exc.start}: {exc.reason}"
                ),
                language=language or "unknown",
            )
        try:
            return self.provider.extract(
                source,
                language,
                blob_identity=blob_identity,
                source_sha256=digest,
            )
        except PolyglotASTProviderError as exc:
            return ASTBlobRecord(
                blob_identity=blob_identity,
                source_sha256=digest,
                parse_error=f"{exc.reason_code}: {exc}",
                language=language or "unknown",
            )
        except Exception as exc:
            # A parser boundary exception is evidence of partial analysis, not
            # permission to silently drop the path.
            return ASTBlobRecord(
                blob_identity=blob_identity,
                source_sha256=digest,
                parse_error=(
                    f"parser_exception:{type(exc).__name__}: {exc}"
                ),
                language=language or "unknown",
            )

    def _non_parser_row(
        self, disposition: CoverageDisposition
    ) -> RepositoryIndexRow:
        if disposition.git_status.value in _DELETED_STATUSES:
            status = ParserStatus.DELETED
            reason = "path_deleted"
        elif disposition.kind is CoverageKind.STRUCTURED_DATA:
            status = ParserStatus.UNSUPPORTED
            reason = "structured_parser_unsupported"
        else:
            status = ParserStatus.NOT_APPLICABLE
            reason = disposition.reason_code
        return RepositoryIndexRow(
            path=disposition.path,
            disposition_kind=disposition.kind,
            declared_kind=disposition.kind,
            reason_code=reason,
            policy_rule=disposition.policy_rule,
            git_status=disposition.git_status,
            content_digest=disposition.content_digest,
            parser_status=status,
            parser_reason=(
                "no deterministic structured parser is registered for this suffix"
                if status is ParserStatus.UNSUPPORTED
                else ""
            ),
            tracked=disposition.tracked,
            overlay=disposition.overlay,
            max_row_bytes=self.max_compact_row_bytes,
        )

    def _health(
        self,
        snapshot: RepositorySnapshot,
        rows: Sequence[RepositoryIndexRow],
    ) -> AnalyzerHealthReport:
        eligible = [
            row
            for row in rows
            if row.parser_status
            in {
                ParserStatus.INDEXED,
                ParserStatus.CACHE_HIT,
                ParserStatus.PARSE_FAILURE,
            }
        ]
        failures = sum(
            row.parser_status is ParserStatus.PARSE_FAILURE for row in eligible
        )
        reused = sum(
            row.parser_status is ParserStatus.CACHE_HIT for row in eligible
        )
        parsed = len(eligible) - failures - reused
        tracked = sum(row.tracked for row in rows)
        inventory = {
            "git_root_count": 1 if snapshot.head_commit_id else 0,
            "expected_git_root_count": 1,
            "tracked_file_count": tracked,
            "eligible_file_count": len(eligible),
            "excluded_file_count": tracked - len(eligible),
            "parsed_file_count": parsed,
            "cache_hit_count": reused,
            "parser_failure_count": failures,
            "raw_candidate_count": 0,
            "seen_candidate_count": 0,
            "deduplicated_candidate_count": 0,
            "rejected_candidate_count": 0,
            "appended_task_count": 0,
            "coverage_complete": (
                len(rows) == len(snapshot.dispositions)
                and tracked == snapshot.stats.tracked_path_count
            ),
            "scan_complete": len(rows) == len(snapshot.dispositions),
        }
        # RepositoryIndexer has a closed parser registry and every invocation
        # is verified by exact source/record identities.  Parser failures enter
        # the inventory above and can never be hidden by this registry canary.
        canaries = {
            "registry_present": True,
            "registry_errors": [],
            "fixture_count": 1,
            "passed": True,
        }
        return classify_analyzer_health(
            inventory,
            canaries=canaries,
            thresholds=self.health_thresholds,
        )

    def build(
        self,
        snapshot: RepositorySnapshot,
        *,
        source_loader: SourceLoader | None = None,
        publish: bool = True,
    ) -> RepositoryIndex:
        """Build a complete index for one already-established snapshot."""

        if not isinstance(snapshot, RepositorySnapshot):
            raise TypeError("snapshot must be a RepositorySnapshot")
        snapshot.assert_exhaustive_tracked_coverage()
        if len(snapshot.dispositions) > self.max_paths:
            raise RepositoryIndexBoundsExceeded(
                f"snapshot exceeds {self.max_paths} paths"
            )
        loader = source_loader or self._default_source_loader(snapshot)

        with self._exclusive_build():
            old_rows, previous_ast = self._load_previous_state_unlocked()
            old_by_digest: dict[str, list[RepositoryIndexRow]] = {}
            for row in old_rows:
                if row.content_digest:
                    old_by_digest.setdefault(row.content_digest, []).append(row)
            for candidates in old_by_digest.values():
                candidates.sort(key=lambda item: item.path)

            rows: list[RepositoryIndexRow] = []
            records: dict[str, ASTBlobRecord] = {}
            reused = renamed = parsed = failures = unsupported = 0
            source_writes = ast_writes = 0
            recovery_start = self.cas.corruption_recoveries

            for disposition in snapshot.dispositions:
                if not self._requires_parser(disposition):
                    row = self._non_parser_row(disposition)
                    unsupported += row.parser_status is ParserStatus.UNSUPPORTED
                    rows.append(row)
                    continue

                candidates = old_by_digest.get(
                    disposition.content_digest, ()
                )
                reused_pair = self._reuse_row(disposition, candidates)
                if reused_pair is not None:
                    row, record = reused_pair
                    rows.append(row)
                    records[disposition.path] = record
                    reused += 1
                    renamed += not any(
                        old.path == disposition.path for old in candidates
                    )
                    failures += bool(record.parse_error)
                    continue

                language = self._language(disposition)
                if not language:
                    rows.append(
                        RepositoryIndexRow(
                            path=disposition.path,
                            disposition_kind=CoverageKind.UNSUPPORTED,
                            declared_kind=disposition.kind,
                            reason_code="parser_language_unsupported",
                            policy_rule=disposition.policy_rule,
                            git_status=disposition.git_status,
                            content_digest=disposition.content_digest,
                            parser_status=ParserStatus.UNSUPPORTED,
                            parser_reason="no parser language mapping for path",
                            tracked=disposition.tracked,
                            overlay=disposition.overlay,
                            max_row_bytes=self.max_compact_row_bytes,
                        )
                    )
                    unsupported += 1
                    continue

                cached = self._cached_record(
                    disposition, language=language, snapshot=snapshot
                )
                payload: bytes | None = None
                if cached is not None:
                    record, ast_mapping = cached
                    try:
                        payload = loader(disposition)
                        self._validate_source(disposition, payload)
                    except RepositoryIndexerError:
                        raise
                    source_ref = self.cas.put(
                        payload,
                        kind="repository-source",
                        media_type="text/plain; charset=utf-8",
                    )
                    source_writes += 1
                    row_status = (
                        ParserStatus.PARSE_FAILURE
                        if record.parse_error
                        else ParserStatus.INDEXED
                    )
                    row = RepositoryIndexRow(
                        path=disposition.path,
                        disposition_kind=(
                            CoverageKind.PARSE_FAILURE
                            if record.parse_error
                            else disposition.kind
                        ),
                        declared_kind=disposition.kind,
                        reason_code=(
                            "parser_reported_failure"
                            if record.parse_error
                            else disposition.reason_code
                        ),
                        policy_rule=disposition.policy_rule,
                        git_status=disposition.git_status,
                        content_digest=disposition.content_digest,
                        source_ref=source_ref.to_dict(),
                        ast_ref=ast_mapping,
                        ast_record_id=record.record_id,
                        language=record.language,
                        parser_status=row_status,
                        parser_reason=record.parse_error,
                        parser_identity=self.parser_identity,
                        tracked=disposition.tracked,
                        overlay=disposition.overlay,
                        max_row_bytes=self.max_compact_row_bytes,
                    )
                    rows.append(row)
                    records[disposition.path] = record
                    reused += 1
                    failures += bool(record.parse_error)
                    continue

                payload = loader(disposition)
                self._validate_source(disposition, payload)
                source_ref = self.cas.put(
                    payload,
                    kind="repository-source",
                    media_type="text/plain; charset=utf-8",
                )
                source_writes += 1
                record = self._parse(
                    payload, disposition, language=language
                )
                ast_ref = self.cas.put_json(
                    record.to_dict(), kind="repository-ast-record"
                )
                ast_writes += 1
                self._store_cache(
                    disposition,
                    language=language,
                    snapshot=snapshot,
                    record=record,
                    ast_ref=ast_ref,
                )
                status = (
                    ParserStatus.PARSE_FAILURE
                    if record.parse_error
                    else ParserStatus.INDEXED
                )
                row = RepositoryIndexRow(
                    path=disposition.path,
                    disposition_kind=(
                        CoverageKind.PARSE_FAILURE
                        if record.parse_error
                        else disposition.kind
                    ),
                    declared_kind=disposition.kind,
                    reason_code=(
                        "parser_reported_failure"
                        if record.parse_error
                        else disposition.reason_code
                    ),
                    policy_rule=disposition.policy_rule,
                    git_status=disposition.git_status,
                    content_digest=disposition.content_digest,
                    source_ref=source_ref.to_dict(),
                    ast_ref=ast_ref.to_dict(),
                    ast_record_id=record.record_id,
                    language=record.language,
                    parser_status=status,
                    parser_reason=record.parse_error,
                    parser_identity=self.parser_identity,
                    tracked=disposition.tracked,
                    overlay=disposition.overlay,
                    max_row_bytes=self.max_compact_row_bytes,
                )
                rows.append(row)
                records[disposition.path] = record
                parsed += 1
                failures += bool(record.parse_error)

            ast_index = build_analysis_ast_index(
                records,
                previous=previous_ast,
            )
            health = self._health(snapshot, rows)
            max_row_size = max(
                (row.serialized_size for row in rows), default=0
            )
            stats = RepositoryIndexStats(
                snapshot_path_count=len(snapshot.dispositions),
                tracked_path_count=sum(row.tracked for row in rows),
                row_count=len(rows),
                eligible_parser_path_count=len(records),
                indexed_path_count=len(records),
                reused_path_count=reused,
                renamed_reuse_count=renamed,
                parsed_path_count=parsed,
                parse_failure_count=failures,
                unsupported_parser_count=unsupported,
                deleted_path_count=sum(
                    row.parser_status is ParserStatus.DELETED for row in rows
                ),
                invalidated_path_count=ast_index.stats.invalidated_blob_count,
                corruption_recovery_count=(
                    self.cas.corruption_recoveries - recovery_start
                ),
                source_blob_write_count=source_writes,
                ast_blob_write_count=ast_writes,
                max_row_bytes=max_row_size,
            )
            transition_invalidations = tuple(
                item
                for item in ast_index.invalidations
                if previous_ast is None
                or item.invalidation_id
                not in {
                    prior.invalidation_id
                    for prior in previous_ast.invalidations
                }
            )
            result = RepositoryIndex(
                snapshot=snapshot,
                rows=tuple(rows),
                ast_index=ast_index,
                health=health,
                build_stats=stats,
                invalidations=transition_invalidations,
            )
            if publish:
                self._publish_unlocked(result)
            return result

    index_snapshot = build
    build_index = build

    def index_repository(
        self,
        repository_root: Path | str,
        *,
        scope_policy: ScopePolicy | Mapping[str, Any] | None = None,
        scope_config_path: Path | str | None = None,
        allow_dirty_analysis: bool | None = None,
        snapshot_max_paths: int | None = None,
        snapshot_max_file_bytes: int | None = None,
        snapshot_max_total_bytes: int | None = None,
    ) -> RepositoryIndex:
        """Snapshot, build, and publish an exact repository index."""

        kwargs: dict[str, Any] = {
            "scope_policy": scope_policy,
            "scope_config_path": scope_config_path,
            "allow_dirty_analysis": allow_dirty_analysis,
            "max_paths": snapshot_max_paths or self.max_paths,
        }
        if snapshot_max_file_bytes is not None:
            kwargs["max_file_bytes"] = snapshot_max_file_bytes
        if snapshot_max_total_bytes is not None:
            kwargs["max_total_bytes"] = snapshot_max_total_bytes
        snapshot = build_repository_snapshot(repository_root, **kwargs)
        return self.build(snapshot)

    index = index_repository
    scan = index_repository

    def _publish_unlocked(self, index: RepositoryIndex) -> None:
        encoded = index.canonical_bytes + b"\n"
        digest = index.index_id.rsplit(":", 1)[-1]
        immutable_path = self.snapshots_path / f"{digest}.json"
        if immutable_path.exists():
            try:
                existing = immutable_path.read_bytes()
            except OSError as exc:
                raise RepositoryIndexIntegrityError(
                    "immutable index snapshot is unreadable"
                ) from exc
            if existing != encoded:
                raise RepositoryIndexIntegrityError(
                    "immutable index snapshot identity collision"
                )
        else:
            _atomic_write(immutable_path, encoded, replace=False)
        _atomic_write(self.current_path, encoded, replace=True)

    def _load_current_unlocked(
        self, *, required: bool
    ) -> RepositoryIndex | None:
        try:
            encoded = self.current_path.read_bytes()
        except FileNotFoundError:
            if required:
                raise RepositoryIndexUnavailable(
                    f"no current index at {self.current_path}"
                )
            return None
        except OSError as exc:
            raise RepositoryIndexUnavailable(
                f"current index is unreadable: {self.current_path}"
            ) from exc
        try:
            value = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RepositoryIndexIntegrityError(
                "current repository index is corrupt"
            ) from exc
        if not isinstance(value, Mapping):
            raise RepositoryIndexIntegrityError(
                "current repository index must be an object"
            )
        return self._decode_index(value)

    def _load_previous_state_unlocked(
        self,
    ) -> tuple[tuple[RepositoryIndexRow, ...], AnalysisASTIndex | None]:
        """Load verified reusable pieces without trusting corrupt CAS bodies.

        A public reader must reject any corrupt referenced object.  A writer,
        however, must be able to recover by re-reading source.  This recovery
        path first verifies the immutable manifest identity and every compact
        row, then admits only AST objects that independently pass CAS and
        record-identity verification.
        """

        try:
            value = json.loads(self.current_path.read_bytes())
            if not isinstance(value, Mapping):
                return (), None
            claimed = str(value.get("index_id") or "")
            content = {
                key: value[key]
                for key in value
                if key != "index_id"
            }
            if (
                value.get("schema") != REPOSITORY_INDEX_SCHEMA
                or claimed != _identity("sca-repository-index", content)
            ):
                return (), None
            _snapshot_from_dict(value.get("snapshot"))
            rows = tuple(
                RepositoryIndexRow.from_dict(
                    item, max_row_bytes=self.max_compact_row_bytes
                )
                for item in value.get("rows", ())
            )
        except (
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            RepositoryIndexerError,
        ):
            return (), None
        records: list[IndexedASTPath] = []
        for row in rows:
            if row.ast_ref is None:
                continue
            record = self._load_ast_reference(row.ast_ref)
            if record is not None and record.record_id == row.ast_record_id:
                records.append(IndexedASTPath(row.path, record))
        return rows, AnalysisASTIndex(path_records=tuple(records))

    def load_current(self) -> RepositoryIndex:
        """Read and verify the atomically published current index."""

        result = self._load_current_unlocked(required=True)
        assert result is not None
        return result

    read_current = load_current
    load = load_current

    def _decode_index(self, value: Mapping[str, Any]) -> RepositoryIndex:
        if value.get("schema") != REPOSITORY_INDEX_SCHEMA:
            raise RepositoryIndexIntegrityError(
                "unsupported repository index schema"
            )
        snapshot = _snapshot_from_dict(value.get("snapshot"))
        rows = tuple(
            RepositoryIndexRow.from_dict(
                item, max_row_bytes=self.max_compact_row_bytes
            )
            for item in value.get("rows", ())
        )
        path_records: list[IndexedASTPath] = []
        for row in rows:
            if row.ast_ref is None:
                continue
            record = self._load_ast_reference(row.ast_ref)
            if record is None or record.record_id != row.ast_record_id:
                raise RepositoryIndexIntegrityError(
                    f"current AST CAS reference is corrupt: {row.path}"
                )
            path_records.append(IndexedASTPath(row.path, record))
        ast_index = AnalysisASTIndex(path_records=tuple(path_records))
        claimed_ast = str(value.get("ast_index_id") or "")
        if claimed_ast != ast_index.index_id:
            raise RepositoryIndexIntegrityError(
                "current AST index identity mismatch"
            )
        health_value = value.get("health")
        if not isinstance(health_value, Mapping):
            raise RepositoryIndexIntegrityError(
                "current index lacks analyzer health"
            )
        health = AnalyzerHealthReport(
            status=AnalyzerHealthStatus(str(health_value.get("status") or "")),
            reasons=tuple(health_value.get("reasons") or ()),
            thresholds=AnalyzerHealthThresholds.from_value(
                health_value.get("thresholds")
            ),
            metrics=dict(health_value.get("metrics") or {}),
        )
        result = RepositoryIndex(
            snapshot=snapshot,
            rows=rows,
            ast_index=ast_index,
            health=health,
            build_stats=RepositoryIndexStats.empty_for_rows(rows),
        )
        claimed = str(value.get("index_id") or "")
        if claimed != result.index_id:
            raise RepositoryIndexIntegrityError(
                "current repository index identity mismatch"
            )
        return result

    def close(self) -> None:
        self.cas.close()


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            return
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _snapshot_from_dict(value: Any) -> RepositorySnapshot:
    if not isinstance(value, Mapping):
        raise RepositoryIndexIntegrityError(
            "repository index snapshot must be an object"
        )
    dispositions: list[CoverageDisposition] = []
    for item in value.get("dispositions", ()):
        if not isinstance(item, Mapping):
            raise RepositoryIndexIntegrityError(
                "snapshot disposition must be an object"
            )
        disposition = CoverageDisposition(
            path=item.get("path", ""),
            kind=item.get("kind", ""),
            git_status=item.get("git_status", ""),
            entry_kind=item.get("entry_kind", ""),
            reason_code=item.get("reason_code", ""),
            policy_rule=item.get("policy_rule", ""),
            content_digest=item.get("content_digest", ""),
            git_mode=item.get("git_mode", ""),
            git_object_id=item.get("git_object_id", ""),
            rename_from=item.get("rename_from", ""),
            tracked=bool(item.get("tracked", True)),
            overlay=bool(item.get("overlay", False)),
            dependency_identity_id=item.get("dependency_identity_id", ""),
            schema_version=int(item.get("schema_version", 1)),
        )
        claimed = str(item.get("disposition_id") or "")
        if claimed and claimed != disposition.disposition_id:
            raise RepositoryIndexIntegrityError(
                f"snapshot disposition identity mismatch: {disposition.path}"
            )
        dispositions.append(disposition)
    dependencies = tuple(
        DependencyIdentity(
            kind=item.get("kind", DependencyIdentityKind.MANIFEST.value),
            path=item.get("path", ""),
            digest=item.get("digest", ""),
            tool_name=item.get("tool_name", ""),
            tool_version=item.get("tool_version", ""),
            git_object_id=item.get("git_object_id", ""),
            reason_code=item.get("reason_code", ""),
        )
        for item in value.get("dependency_identities", ())
    )
    gitlinks = tuple(
        GitlinkRecord(
            path=item.get("path", ""),
            commit_id=item.get("commit_id", ""),
            mode=item.get("mode", "160000"),
            head_object_id=item.get("head_object_id", ""),
            index_object_id=item.get("index_object_id", ""),
        )
        for item in value.get("gitlinks", ())
    )
    stats_value = value.get("stats") or {}
    if not isinstance(stats_value, Mapping):
        raise RepositoryIndexIntegrityError("snapshot stats must be an object")
    stats = RepositorySnapshotStats(
        **{
            name: int(stats_value.get(name, 0))
            for name in RepositorySnapshotStats.__dataclass_fields__
        }
    )
    snapshot = RepositorySnapshot(
        primary_root=value.get("primary_root", "."),
        head_commit_id=value.get("head_commit_id", ""),
        head_tree_id=value.get("head_tree_id", ""),
        index_tree_id=value.get("index_tree_id", ""),
        scope_policy_id=value.get("scope_policy_id", ""),
        scope_id=value.get("scope_id", ""),
        dispositions=tuple(dispositions),
        dependency_identities=dependencies,
        gitlinks=gitlinks,
        stats=stats,
        schema_version=int(value.get("schema_version", 1)),
        repository_root=value.get("repository_root", ""),
        git_directory=value.get("git_directory", ""),
        allow_dirty_analysis=bool(value.get("allow_dirty_analysis", True)),
    )
    claimed = str(value.get("snapshot_id") or "")
    if claimed and claimed != snapshot.snapshot_id:
        raise RepositoryIndexIntegrityError(
            "repository snapshot identity mismatch"
        )
    snapshot.assert_exhaustive_tracked_coverage()
    return snapshot


def build_repository_index(
    snapshot: RepositorySnapshot,
    *,
    index_root: Path | str,
    provider: PolyglotASTProvider | None = None,
    source_loader: SourceLoader | None = None,
    health_thresholds: AnalyzerHealthThresholds
    | Mapping[str, Any]
    | None = None,
) -> RepositoryIndex:
    """Convenience entry point for one complete snapshot build."""

    indexer = RepositoryIndexer(
        index_root,
        provider=provider,
        health_thresholds=health_thresholds,
    )
    return indexer.build(snapshot, source_loader=source_loader)


__all__ = [
    "DEFAULT_MAX_COMPACT_ROW_BYTES",
    "DEFAULT_MAX_INDEX_PATHS",
    "DEFAULT_MAX_SOURCE_BYTES",
    "HARD_MAX_COMPACT_ROW_BYTES",
    "ParserStatus",
    "REPOSITORY_INDEXER_VERSION",
    "REPOSITORY_INDEX_CACHE_SCHEMA",
    "REPOSITORY_INDEX_ROW_SCHEMA",
    "REPOSITORY_INDEX_SCHEMA",
    "REPOSITORY_INDEX_STATS_SCHEMA",
    "RepositoryCAS",
    "RepositoryIndex",
    "RepositoryIndexBoundsExceeded",
    "RepositoryIndexIntegrityError",
    "RepositoryIndexRow",
    "RepositoryIndexSourceChanged",
    "RepositoryIndexStats",
    "RepositoryIndexUnavailable",
    "RepositoryIndexer",
    "RepositoryIndexerError",
    "SourceLoader",
    "build_repository_index",
    "canonical_repository_index_bytes",
]
