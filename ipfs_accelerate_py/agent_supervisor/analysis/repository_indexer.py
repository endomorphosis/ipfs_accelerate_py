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
    MULTI_ROOT_PROVIDER_INDEX_EVIDENCE,
    MULTI_ROOT_REPOSITORY_SNAPSHOT_INTERFACE,
    CoverageDisposition,
    CoverageKind,
    DependencyIdentity,
    DependencyIdentityKind,
    EntryKind,
    GitStatus,
    GitlinkRecord,
    MultiRootRepositorySnapshot,
    ProviderPackageSpec,
    ProviderRootContradiction,
    ProviderRootContradictionKind,
    ProviderRootObservation,
    ProviderRootStatus,
    RepositorySnapshot,
    RepositorySnapshotStats,
    ScopePolicy,
    build_multi_root_repository_snapshot,
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
DEFAULT_MAX_PARSER_SOURCE_BYTES: Final = 16 * 1024 * 1024
DEFAULT_MAX_SOURCE_BYTES: Final = 64 * 1024 * 1024
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
                if disposition.entry_kind is EntryKind.SYMLINK:
                    before = path.lstat()
                    payload = os.fsencode(os.readlink(path))
                    after = path.lstat()
                else:
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
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
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
        *,
        language: str,
    ) -> tuple[RepositoryIndexRow, ASTBlobRecord] | None:
        for old in candidates:
            if (
                old.content_digest != disposition.content_digest
                or old.parser_identity != self.parser_identity
                or old.language != language
                or old.source_ref is None
                or old.ast_ref is None
                or not self.cas.verify(old.source_ref)
            ):
                continue
            record = self._load_ast_reference(old.ast_ref)
            if (
                record is None
                or record.record_id != old.ast_record_id
                or record.language != language
            ):
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

                candidates = old_by_digest.get(
                    disposition.content_digest, ()
                )
                reused_pair = self._reuse_row(
                    disposition,
                    candidates,
                    language=language,
                )
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


# ---------------------------------------------------------------------------
# Multi-root provider package index (SCA-G043)
# ---------------------------------------------------------------------------

MULTI_ROOT_REPOSITORY_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-multi-root-repository-index@1"
)
PROVIDER_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-provider-index@1"
)
CROSS_ROOT_SYMBOL_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-cross-root-symbol-identity@1"
)
PROVIDER_INDEX_BASELINE_RELATIVE: Final = (
    "data/agent_supervisor/swissknife_contract_assurance/baseline/provider-index.json"
)


class CrossRootSymbolJoinError(RepositoryIndexerError, ValueError):
    """Cross-root symbol join failed closed (inexact or ambiguous identity)."""


@dataclass(frozen=True)
class CrossRootSymbolIdentity:
    """Exact package/module/function identity for cross-root joins.

    Joins never use path equality alone: two symbols may only be considered
    the same when package, module, and function names match exactly.
    """

    package: str
    module: str
    function: str
    path: str = ""
    root_id: str = ""

    def __post_init__(self) -> None:
        package = str(self.package or "").strip()
        module = str(self.module or "").strip()
        function = str(self.function or "").strip()
        if not package or not module or not function:
            raise CrossRootSymbolJoinError(
                "cross-root symbol identity requires package, module, and function"
            )
        if any(
            part != part.strip() or not part
            for part in (package, module, function)
        ):
            raise CrossRootSymbolJoinError(
                "cross-root symbol fields must be non-empty stripped strings"
            )
        if "/" in package or "\\" in package or ".." in package:
            raise CrossRootSymbolJoinError(
                f"invalid package identity: {package!r}"
            )
        object.__setattr__(self, "package", package)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "function", function)
        if self.path:
            object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(self, "root_id", str(self.root_id or "").strip())

    @property
    def qualified_name(self) -> str:
        return f"{self.package}:{self.module}.{self.function}"

    @property
    def identity_id(self) -> str:
        return _identity(
            "sca-cross-root-symbol",
            {
                "schema": CROSS_ROOT_SYMBOL_IDENTITY_SCHEMA,
                "package": self.package,
                "module": self.module,
                "function": self.function,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CROSS_ROOT_SYMBOL_IDENTITY_SCHEMA,
            "identity_id": self.identity_id,
            "package": self.package,
            "module": self.module,
            "function": self.function,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "root_id": self.root_id,
        }


def make_cross_root_symbol(
    *,
    package: str,
    module: str,
    function: str,
    path: str = "",
    root_id: str = "",
) -> CrossRootSymbolIdentity:
    """Construct one exact package/module/function identity."""

    return CrossRootSymbolIdentity(
        package=package,
        module=module,
        function=function,
        path=path,
        root_id=root_id,
    )


def join_cross_root_symbols(
    *identities: CrossRootSymbolIdentity,
) -> CrossRootSymbolIdentity:
    """Join symbols only when package, module, and function match exactly.

    Path and root metadata may differ (they name distinct roots).  Partial or
    ambiguous identities fail closed.
    """

    if not identities:
        raise CrossRootSymbolJoinError("cross-root join requires at least one identity")
    for item in identities:
        if not isinstance(item, CrossRootSymbolIdentity):
            raise CrossRootSymbolJoinError(
                "cross-root join accepts only CrossRootSymbolIdentity values"
            )
    head = identities[0]
    for item in identities[1:]:
        if (
            item.package != head.package
            or item.module != head.module
            or item.function != head.function
        ):
            raise CrossRootSymbolJoinError(
                "cross-root join requires exact package/module/function equality; "
                f"got {head.qualified_name!r} vs {item.qualified_name!r}"
            )
    # Preserve the first path/root as the canonical join witness; others may
    # differ by root without affecting the exact identity.
    return head


def module_name_for_package_path(package: str, path: str) -> str:
    """Derive a Python module name from a package-relative source path."""

    normalized = _normalize_path(path)
    pure = PurePosixPath(normalized)
    if pure.suffix != ".py":
        raise CrossRootSymbolJoinError(
            f"module identity requires a .py path, got {path!r}"
        )
    parts = list(pure.with_suffix("").parts)
    if not parts:
        raise CrossRootSymbolJoinError(f"empty module path: {path!r}")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return package
    return package + "." + ".".join(parts)


def extract_package_function_symbols(
    package: str,
    path: str,
    source: bytes | str,
    *,
    root_id: str = "",
) -> tuple[CrossRootSymbolIdentity, ...]:
    """Extract top-level and nested function identities from one Python source."""

    import ast

    text = source.decode("utf-8") if isinstance(source, (bytes, bytearray)) else str(source)
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise CrossRootSymbolJoinError(
            f"cannot extract symbols from {path}: {exc}"
        ) from exc
    module = module_name_for_package_path(package, path)
    found: list[CrossRootSymbolIdentity] = []

    class _Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            symbol = ".".join((*self.scope, node.name))
            found.append(
                CrossRootSymbolIdentity(
                    package=package,
                    module=module,
                    function=symbol,
                    path=path,
                    root_id=root_id,
                )
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    _Collector().visit(tree)
    return tuple(
        sorted(
            found,
            key=lambda item: (item.module, item.function, item.path),
        )
    )


@dataclass(frozen=True)
class ProviderRootIndex:
    """One independently indexed provider package root."""

    observation: ProviderRootObservation
    index: RepositoryIndex | None
    symbols: tuple[CrossRootSymbolIdentity, ...] = ()
    health: AnalyzerHealthReport | None = None
    symbol_eligible_file_count: int = 0
    symbol_extracted_file_count: int = 0
    symbol_failed_file_count: int = 0
    symbol_extraction_enabled: bool = True
    symbol_extraction_complete: bool = False
    symbol_extraction_reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observation.indexed and self.index is None:
            raise RepositoryIndexIntegrityError(
                f"indexed provider {self.observation.package} lacks a repository index"
            )
        if self.index is not None and self.observation.snapshot is not None:
            if self.index.snapshot.snapshot_id != self.observation.snapshot.snapshot_id:
                raise RepositoryIndexIntegrityError(
                    f"provider index snapshot mismatch for {self.observation.package}"
                )
        health = self.health
        if health is None and self.index is not None:
            health = self.index.health
        object.__setattr__(self, "health", health)
        object.__setattr__(
            self,
            "symbols",
            tuple(
                sorted(
                    self.symbols,
                    key=lambda item: (item.module, item.function, item.path),
                )
            ),
        )
        for name in (
            "symbol_eligible_file_count",
            "symbol_extracted_file_count",
            "symbol_failed_file_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise RepositoryIndexIntegrityError(
                    f"{name} must be non-negative"
                )
            object.__setattr__(self, name, value)
        accounted = (
            self.symbol_extracted_file_count + self.symbol_failed_file_count
        )
        if accounted > self.symbol_eligible_file_count:
            raise RepositoryIndexIntegrityError(
                "symbol extraction counts exceed eligible Python files"
            )
        reasons = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in self.symbol_extraction_reason_codes
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "symbol_extraction_reason_codes", reasons)
        if self.symbol_extraction_complete and (
            not self.symbol_extraction_enabled
            or self.symbol_failed_file_count
            or self.symbol_extracted_file_count
            != self.symbol_eligible_file_count
            or reasons
        ):
            raise RepositoryIndexIntegrityError(
                "complete symbol extraction has incomplete accounting"
            )

    @property
    def package(self) -> str:
        return self.observation.package

    @property
    def indexed(self) -> bool:
        return bool(self.observation.indexed and self.index is not None)

    @property
    def opaque_gitlink(self) -> bool:
        return bool(self.observation.opaque_gitlink)

    @property
    def healthy(self) -> bool:
        return bool(
            self.health is not None
            and self.health.status is AnalyzerHealthStatus.HEALTHY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "observation": self.observation.compact_dict(),
            "index_id": self.index.index_id if self.index is not None else "",
            "snapshot_id": (
                self.observation.snapshot.snapshot_id
                if self.observation.snapshot is not None
                else ""
            ),
            "health": self.health.to_dict() if self.health is not None else None,
            "symbol_count": len(self.symbols),
            "symbols": [item.to_dict() for item in self.symbols],
            "symbol_extraction": {
                "enabled": bool(self.symbol_extraction_enabled),
                "complete": bool(self.symbol_extraction_complete),
                "eligible_file_count": self.symbol_eligible_file_count,
                "extracted_file_count": self.symbol_extracted_file_count,
                "failed_file_count": self.symbol_failed_file_count,
                "skipped_file_count": max(
                    0,
                    self.symbol_eligible_file_count
                    - self.symbol_extracted_file_count
                    - self.symbol_failed_file_count,
                ),
                "reason_codes": list(self.symbol_extraction_reason_codes),
            },
            "indexed": self.indexed,
            "opaque_gitlink": self.opaque_gitlink,
            "healthy": self.healthy,
            "build_stats": (
                self.index.build_stats.to_dict() if self.index is not None else {}
            ),
        }

    def compact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("symbols", None)
        return payload


@dataclass(frozen=True)
class MultiRootRepositoryIndex:
    """Independent provider package indexes with exact cross-root join policy."""

    multi_root_snapshot: MultiRootRepositorySnapshot
    providers: tuple[ProviderRootIndex, ...]
    contradictions: tuple[ProviderRootContradiction, ...]

    def __post_init__(self) -> None:
        packages = [item.package for item in self.providers]
        if len(packages) != len(set(packages)):
            raise RepositoryIndexIntegrityError(
                "multi-root provider indexes must be unique per package"
            )
        object.__setattr__(
            self,
            "providers",
            tuple(sorted(self.providers, key=lambda item: item.package)),
        )
        object.__setattr__(
            self,
            "contradictions",
            tuple(
                sorted(
                    self.contradictions,
                    key=lambda item: (
                        item.package,
                        item.kind.value,
                        item.detail,
                    ),
                )
            ),
        )

    @property
    def multi_root_id(self) -> str:
        return _identity(
            "sca-multi-root-repository-index",
            self._content_dict(),
        )

    @property
    def all_providers_indexed(self) -> bool:
        return bool(self.providers) and all(item.indexed for item in self.providers)

    @property
    def all_providers_healthy(self) -> bool:
        return bool(self.providers) and all(item.healthy for item in self.providers)

    @property
    def all_symbol_extractions_complete(self) -> bool:
        return bool(self.providers) and all(
            item.symbol_extraction_complete for item in self.providers
        )

    @property
    def any_opaque_gitlink(self) -> bool:
        return any(item.opaque_gitlink for item in self.providers)

    @property
    def exhaustive_parity_allowed(self) -> bool:
        """Exhaustive multi-root parity requires every provider healthy and indexed.

        Partial provider health, opaque gitlinks, or root contradictions block
        exhaustive parity claims fail-closed.
        """

        if not self.providers:
            return False
        if self.contradictions:
            return False
        if self.any_opaque_gitlink:
            return False
        if not self.all_providers_indexed:
            return False
        if not self.all_providers_healthy:
            return False
        if not self.all_symbol_extractions_complete:
            return False
        if self.multi_root_snapshot.has_blocking_contradictions:
            return False
        return True

    def provider_for_package(self, package: str) -> ProviderRootIndex | None:
        name = str(package or "").strip()
        for item in self.providers:
            if item.package == name:
                return item
        return None

    def symbols_for_package(self, package: str) -> tuple[CrossRootSymbolIdentity, ...]:
        root = self.provider_for_package(package)
        return root.symbols if root is not None else ()

    def join_symbols(
        self, *identities: CrossRootSymbolIdentity
    ) -> CrossRootSymbolIdentity:
        """Exact package/module/function join across provider roots."""

        return join_cross_root_symbols(*identities)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": MULTI_ROOT_REPOSITORY_INDEX_SCHEMA,
            "indexer_version": REPOSITORY_INDEXER_VERSION,
            "evidence_id": MULTI_ROOT_PROVIDER_INDEX_EVIDENCE,
            "multi_root_snapshot_id": self.multi_root_snapshot.multi_root_id,
            "providers": [item.compact_dict() for item in self.providers],
            "contradictions": [item.to_dict() for item in self.contradictions],
            "cross_root_join_policy": "package_module_function_exact",
            "bodies_in_cas": True,
            "exhaustive_parity_allowed": self.exhaustive_parity_allowed,
            "all_providers_indexed": self.all_providers_indexed,
            "all_providers_healthy": self.all_providers_healthy,
            "all_symbol_extractions_complete": (
                self.all_symbol_extractions_complete
            ),
            "any_opaque_gitlink": self.any_opaque_gitlink,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "multi_root_id": self.multi_root_id,
            "multi_root_snapshot": self.multi_root_snapshot.compact_dict(),
            "providers": [item.to_dict() for item in self.providers],
        }

    def to_provider_index_baseline(self) -> dict[str, Any]:
        """Compact baseline document for ``provider-index.json``."""

        return {
            "schema": PROVIDER_INDEX_SCHEMA,
            "schema_version": 1,
            "interface": MULTI_ROOT_REPOSITORY_SNAPSHOT_INTERFACE,
            "evidence_id": MULTI_ROOT_PROVIDER_INDEX_EVIDENCE,
            "indexer_version": REPOSITORY_INDEXER_VERSION,
            "multi_root_id": self.multi_root_id,
            "multi_root_snapshot_id": self.multi_root_snapshot.multi_root_id,
            "scope_id": self.multi_root_snapshot.scope_id,
            "scope_policy_id": self.multi_root_snapshot.scope_policy_id,
            "cross_root_join_policy": "package_module_function_exact",
            "bodies_in_cas": True,
            "primary_snapshot_distinct": True,
            "exhaustive_parity_allowed": self.exhaustive_parity_allowed,
            "all_providers_indexed": self.all_providers_indexed,
            "all_providers_healthy": self.all_providers_healthy,
            "all_symbol_extractions_complete": (
                self.all_symbol_extractions_complete
            ),
            "any_opaque_gitlink": self.any_opaque_gitlink,
            "has_blocking_contradictions": (
                self.multi_root_snapshot.has_blocking_contradictions
                or bool(self.contradictions)
            ),
            "providers": [
                {
                    "package": item.package,
                    "scope_path": item.observation.scope_path,
                    "status": item.observation.status.value,
                    "indexed": item.indexed,
                    "opaque_gitlink": item.opaque_gitlink,
                    "origin_url": item.observation.origin_url,
                    "gitlink_commit_id": item.observation.gitlink_commit_id,
                    "head_commit_id": item.observation.head_commit_id,
                    "head_tree_id": item.observation.head_tree_id,
                    "index_tree_id": item.observation.index_tree_id,
                    "dirty": item.observation.dirty,
                    "version_divergent": item.observation.version_divergent,
                    "moved": item.observation.moved,
                    "snapshot_id": (
                        item.observation.snapshot.snapshot_id
                        if item.observation.snapshot is not None
                        else ""
                    ),
                    "index_id": item.index.index_id if item.index is not None else "",
                    "health_status": (
                        item.health.status.value if item.health is not None else ""
                    ),
                    "tracked_path_count": (
                        item.observation.snapshot.stats.tracked_path_count
                        if item.observation.snapshot is not None
                        else 0
                    ),
                    "semantic_path_count": (
                        item.observation.snapshot.stats.semantic_path_count
                        if item.observation.snapshot is not None
                        else 0
                    ),
                    "symbol_count": len(item.symbols),
                    "symbol_extraction": item.to_dict()["symbol_extraction"],
                    "reason_code": item.observation.reason_code,
                    "contradictions": [
                        c.to_dict() for c in item.observation.contradictions
                    ],
                }
                for item in self.providers
            ],
            "contradictions": [item.to_dict() for item in self.contradictions],
        }


def build_multi_root_repository_index(
    superproject_root: Path | str,
    *,
    index_root: Path | str,
    scope_policy: ScopePolicy | Mapping[str, Any] | None = None,
    scope_config_path: Path | str | None = None,
    provider_packages: Sequence[ProviderPackageSpec | Mapping[str, Any]]
    | None = None,
    provider: PolyglotASTProvider | None = None,
    health_thresholds: AnalyzerHealthThresholds
    | Mapping[str, Any]
    | None = None,
    include_primary_snapshot: bool = False,
    allow_dirty_analysis: bool | None = None,
    max_paths: int | None = None,
    max_symbol_files_per_package: int = DEFAULT_MAX_INDEX_PATHS,
    extract_symbols: bool = True,
    multi_root_snapshot: MultiRootRepositorySnapshot | None = None,
) -> MultiRootRepositoryIndex:
    """Index configured provider package roots as independent CAS-backed trees.

    Source and AST bodies remain in each root's CAS.  Cross-root joins use only
    exact package/module/function identities.  Partial provider health or
    opaque gitlink roots block exhaustive parity.
    """

    root = Path(index_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    symbol_file_limit = int(max_symbol_files_per_package)
    if symbol_file_limit < 0:
        raise RepositoryIndexBoundsExceeded(
            "max_symbol_files_per_package must be non-negative"
        )

    snapshot = multi_root_snapshot or build_multi_root_repository_snapshot(
        superproject_root,
        scope_policy=scope_policy,
        scope_config_path=scope_config_path,
        provider_packages=provider_packages,
        include_primary_snapshot=include_primary_snapshot,
        allow_dirty_analysis=allow_dirty_analysis,
        max_paths=max_paths or DEFAULT_MAX_INDEX_PATHS,
        inventory_providers=True,
    )

    provider_indexes: list[ProviderRootIndex] = []
    contradictions: list[ProviderRootContradiction] = list(snapshot.contradictions)

    for observation in snapshot.providers:
        package_index_root = root / "providers" / observation.package
        package_index_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not observation.indexed or observation.snapshot is None:
            if observation.opaque_gitlink:
                contradictions.append(
                    ProviderRootContradiction(
                        kind=ProviderRootContradictionKind.OPAQUE_GITLINK,
                        package=observation.package,
                        scope_path=observation.scope_path,
                        detail="provider source was not indexed; gitlink remains opaque",
                        gitlink_commit_id=observation.gitlink_commit_id,
                        head_commit_id=observation.head_commit_id,
                    )
                )
            provider_indexes.append(
                ProviderRootIndex(
                    observation=observation,
                    index=None,
                    symbols=(),
                    health=None,
                    symbol_extraction_enabled=False,
                    symbol_extraction_complete=False,
                    symbol_extraction_reason_codes=(
                        "provider_root_not_indexed",
                    ),
                )
            )
            continue

        indexer = RepositoryIndexer(
            package_index_root,
            provider=provider,
            health_thresholds=health_thresholds,
            max_paths=max_paths or DEFAULT_MAX_INDEX_PATHS,
        )
        try:
            index = indexer.build(observation.snapshot, publish=True)
        finally:
            indexer.close()

        symbols: list[CrossRootSymbolIdentity] = []
        python_rows = [
            row
            for row in index.rows
            if row.path.endswith(".py")
            and row.disposition_kind is CoverageKind.SEMANTIC_AST
        ]
        symbol_eligible_file_count = len(python_rows)
        symbol_extracted_file_count = 0
        symbol_failed_file_count = 0
        symbol_reason_codes: set[str] = set()
        if extract_symbols:
            # Re-open CAS via a reader indexer to pull source bodies only for
            # symbol extraction; bodies are never embedded in rows or baseline.
            reader = RepositoryIndexer(
                package_index_root,
                provider=provider,
                health_thresholds=health_thresholds,
            )
            try:
                for row in python_rows[:symbol_file_limit]:
                    if row.source_ref is None:
                        symbol_failed_file_count += 1
                        symbol_reason_codes.add("symbol_source_ref_missing")
                        continue
                    try:
                        source = reader.cas.read(row.source_ref)
                    except Exception:
                        symbol_failed_file_count += 1
                        symbol_reason_codes.add("symbol_source_read_failed")
                        continue
                    try:
                        symbols.extend(
                            extract_package_function_symbols(
                                observation.package,
                                row.path,
                                source,
                                root_id=observation.observation_id,
                            )
                        )
                    except CrossRootSymbolJoinError:
                        symbol_failed_file_count += 1
                        symbol_reason_codes.add("symbol_parse_failed")
                        continue
                    symbol_extracted_file_count += 1
            finally:
                reader.close()
            if symbol_eligible_file_count > symbol_file_limit:
                symbol_reason_codes.add("symbol_extraction_truncated")
        else:
            symbol_reason_codes.add("symbol_extraction_disabled")

        symbol_extraction_complete = bool(
            extract_symbols
            and symbol_extracted_file_count == symbol_eligible_file_count
            and symbol_failed_file_count == 0
            and not symbol_reason_codes
        )
        if not symbol_extraction_complete:
            skipped = max(
                0,
                symbol_eligible_file_count
                - symbol_extracted_file_count
                - symbol_failed_file_count,
            )
            contradictions.append(
                ProviderRootContradiction(
                    kind=ProviderRootContradictionKind.PARTIAL_HEALTH,
                    package=observation.package,
                    scope_path=observation.scope_path,
                    detail=(
                        "symbol extraction incomplete: "
                        f"eligible={symbol_eligible_file_count},"
                        f"extracted={symbol_extracted_file_count},"
                        f"failed={symbol_failed_file_count},"
                        f"skipped={skipped},"
                        "reasons="
                        + ",".join(sorted(symbol_reason_codes))
                    ),
                    gitlink_commit_id=observation.gitlink_commit_id,
                    head_commit_id=observation.head_commit_id,
                )
            )

        health = index.health
        if health.status is not AnalyzerHealthStatus.HEALTHY:
            contradictions.append(
                ProviderRootContradiction(
                    kind=ProviderRootContradictionKind.PARTIAL_HEALTH,
                    package=observation.package,
                    scope_path=observation.scope_path,
                    detail=(
                        f"provider analyzer health is {health.status.value}: "
                        + ",".join(health.reasons[:5])
                    ),
                    gitlink_commit_id=observation.gitlink_commit_id,
                    head_commit_id=observation.head_commit_id,
                )
            )

        provider_indexes.append(
            ProviderRootIndex(
                observation=observation,
                index=index,
                symbols=tuple(symbols),
                health=health,
                symbol_eligible_file_count=symbol_eligible_file_count,
                symbol_extracted_file_count=symbol_extracted_file_count,
                symbol_failed_file_count=symbol_failed_file_count,
                symbol_extraction_enabled=bool(extract_symbols),
                symbol_extraction_complete=symbol_extraction_complete,
                symbol_extraction_reason_codes=tuple(symbol_reason_codes),
            )
        )

    # Deduplicate contradictions by identity.
    unique: dict[str, ProviderRootContradiction] = {}
    for item in contradictions:
        unique[item.contradiction_id] = item

    return MultiRootRepositoryIndex(
        multi_root_snapshot=snapshot,
        providers=tuple(provider_indexes),
        contradictions=tuple(unique.values()),
    )


def write_provider_index_baseline(
    multi_root_index: MultiRootRepositoryIndex,
    destination: Path | str,
) -> Path:
    """Atomically publish the compact provider-index baseline document."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = multi_root_index.to_provider_index_baseline()
    encoded = canonical_repository_index_bytes(payload) + b"\n"
    _atomic_write(path, encoded, replace=True)
    return path


# ---------------------------------------------------------------------------
# Planning analysis inventory helpers (PDR-011)
# ---------------------------------------------------------------------------

PLANNING_PATH_CATEGORIES: Final[tuple[str, ...]] = (
    "tests",
    "config",
    "build",
    "schema",
    "docs",
    "policies",
    "source",
    "generated",
    "other",
)

# Frontiers that remain open until an optional provider certifies support.
PLANNING_OPEN_FRONTIER_KINDS: Final[tuple[str, ...]] = (
    "cfg",
    "dataflow",
    "native",
    "generated",
    "concurrency",
)

_PLANNING_BUILD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "cargo.lock",
        "cargo.toml",
        "cmakelists.txt",
        "composer.json",
        "dockerfile",
        "gemfile",
        "go.mod",
        "go.sum",
        "makefile",
        "meson.build",
        "package-lock.json",
        "package.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pom.xml",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "yarn.lock",
    }
)
_PLANNING_POLICY_NAMES: Final[frozenset[str]] = frozenset(
    {
        "agents.md",
        "code_of_conduct.md",
        "codeowners",
        "contributing.md",
        "license",
        "license.md",
        "notice",
        "security.md",
    }
)
_PLANNING_CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".toml", ".ini", ".cfg", ".conf", ".config"}
)
_PLANNING_CONFIG_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".editorconfig",
        ".flake8",
        ".gitattributes",
        ".gitignore",
        ".pre-commit-config.yaml",
        "pytest.ini",
        "ruff.toml",
        "setup.cfg",
        "tox.ini",
    }
)
_PLANNING_SCHEMA_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".json", ".jsonschema", ".schema", ".proto", ".avsc", ".graphql", ".xsd"}
)
_PLANNING_DOC_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".md", ".mdx", ".rst", ".adoc", ".txt"}
)
_PLANNING_GENERATED_PARTS: Final[frozenset[str]] = frozenset(
    {"generated", "dist", "build", "out", "target", "__pycache__"}
)
_PLANNING_NATIVE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".s",
        ".S",
        ".so",
        ".dylib",
        ".dll",
        ".a",
        ".o",
    }
)


def planning_path_category(path: str) -> str:
    """Classify one repository-relative path for the planning inventory."""

    pure = PurePosixPath(_normalize_path(path))
    name = pure.name.casefold()
    parts = {part.casefold() for part in pure.parts[:-1]}
    suffix = pure.suffix.casefold()
    suffixes = {item.casefold() for item in pure.suffixes}

    if parts & _PLANNING_GENERATED_PARTS or any(
        part.endswith("_generated") or part.endswith(".generated")
        for part in pure.parts
    ):
        return "generated"
    if (
        "test" in parts
        or "tests" in parts
        or "testing" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".spec.js")
        or name.endswith(".test.ts")
        or name.endswith(".test.js")
    ):
        return "tests"
    if (
        name in _PLANNING_POLICY_NAMES
        or "policy" in name
        or name.endswith(".todo.md")
        or name.endswith(".objectives.md")
        or "policies" in parts
        or "policy" in parts
    ):
        return "policies"
    if (
        name in _PLANNING_BUILD_NAMES
        or (name.startswith("requirements") and suffix == ".txt")
        or name == "dockerfile"
    ):
        return "build"
    if (
        "schema" in parts
        or "schemas" in parts
        or name.endswith(".schema.json")
        or ".schema" in suffixes
        or (
            suffix in _PLANNING_SCHEMA_SUFFIXES
            and ("schema" in name or "schemas" in parts or pure.stem.endswith("_schema"))
        )
        or suffix in {".proto", ".avsc", ".graphql", ".xsd"}
    ):
        return "schema"
    if (
        "config" in parts
        or "configs" in parts
        or name in _PLANNING_CONFIG_NAMES
        or suffix in _PLANNING_CONFIG_SUFFIXES
        or (suffix in {".yaml", ".yml", ".json"} and "config" in name)
    ):
        return "config"
    if (
        "doc" in parts
        or "docs" in parts
        or "documentation" in parts
        or suffix in _PLANNING_DOC_SUFFIXES
    ):
        return "docs"
    if suffix in {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".java",
        ".kt",
        ".go",
        ".rs",
        ".rb",
        ".php",
    } or suffix in _PLANNING_NATIVE_SUFFIXES:
        return "source"
    return "other"


def planning_category_inventory(
    paths: Sequence[str],
    *,
    max_paths_per_category: int = 256,
) -> dict[str, Any]:
    """Build a body-free category inventory for planning and Doctor use."""

    buckets: dict[str, list[str]] = {name: [] for name in PLANNING_PATH_CATEGORIES}
    for path in paths:
        category = planning_path_category(path)
        buckets.setdefault(category, []).append(_normalize_path(path))
    inventory: dict[str, Any] = {}
    for category in PLANNING_PATH_CATEGORIES:
        ordered = tuple(sorted(set(buckets.get(category, ()))))
        inventory[category] = {
            "count": len(ordered),
            "paths": list(ordered[: max(0, int(max_paths_per_category))]),
            "truncated": len(ordered) > max_paths_per_category,
        }
    inventory["totals"] = {
        category: int(inventory[category]["count"])
        for category in PLANNING_PATH_CATEGORIES
    }
    return inventory


def open_frontiers_from_repository_index(
    index: RepositoryIndex | None,
    *,
    optional_provider_status: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Record CFG/dataflow/native/generated/concurrency open frontiers.

    Exact local indexing never claims these analyses are closed.  Optional
    providers may mark a frontier degraded or abstained; they cannot silently
    claim completeness.
    """

    status_map = {
        str(key): str(value)
        for key, value in dict(optional_provider_status or {}).items()
    }
    native_paths: list[str] = []
    generated_paths: list[str] = []
    if index is not None:
        for row in index.rows:
            category = planning_path_category(row.path)
            if (
                category == "generated"
                or row.disposition_kind is CoverageKind.BINARY_OR_GENERATED
            ):
                generated_paths.append(row.path)
            pure = PurePosixPath(row.path)
            if pure.suffix.casefold() in _PLANNING_NATIVE_SUFFIXES:
                native_paths.append(row.path)

    frontiers: list[dict[str, Any]] = []
    for kind in PLANNING_OPEN_FRONTIER_KINDS:
        provider_status = status_map.get(kind, status_map.get(f"frontier:{kind}", ""))
        if provider_status in {"available", "supported", "closed"}:
            # Optional providers may only degrade; they never close these
            # frontiers without a certified capability receipt (out of scope).
            frontier_status = "degraded"
            reason = "optional_provider_uncertified"
        elif provider_status in {"missing", "unavailable", "abstain", "abstained"}:
            frontier_status = "abstained"
            reason = "optional_provider_unavailable"
        elif provider_status in {"error", "failed", "degraded"}:
            frontier_status = "degraded"
            reason = "optional_provider_degraded"
        else:
            frontier_status = "open"
            reason = "local_index_does_not_close_frontier"

        evidence_paths: list[str] = []
        if kind == "native":
            evidence_paths = sorted(set(native_paths))[:64]
            if evidence_paths and frontier_status == "open":
                reason = "native_or_ffi_paths_present"
        elif kind == "generated":
            evidence_paths = sorted(set(generated_paths))[:64]
            if evidence_paths and frontier_status == "open":
                reason = "generated_paths_present"

        frontiers.append(
            {
                "kind": kind,
                "frontier_id": f"frontier:{kind}",
                "status": frontier_status,
                "reason_code": reason,
                "path_count": len(evidence_paths),
                "sample_paths": evidence_paths[:16],
            }
        )
    return tuple(frontiers)


__all__ = [
    "CROSS_ROOT_SYMBOL_IDENTITY_SCHEMA",
    "CrossRootSymbolIdentity",
    "CrossRootSymbolJoinError",
    "DEFAULT_MAX_COMPACT_ROW_BYTES",
    "DEFAULT_MAX_INDEX_PATHS",
    "DEFAULT_MAX_PARSER_SOURCE_BYTES",
    "DEFAULT_MAX_SOURCE_BYTES",
    "HARD_MAX_COMPACT_ROW_BYTES",
    "MULTI_ROOT_PROVIDER_INDEX_EVIDENCE",
    "MULTI_ROOT_REPOSITORY_INDEX_SCHEMA",
    "MultiRootRepositoryIndex",
    "PLANNING_OPEN_FRONTIER_KINDS",
    "PLANNING_PATH_CATEGORIES",
    "PROVIDER_INDEX_BASELINE_RELATIVE",
    "PROVIDER_INDEX_SCHEMA",
    "ParserStatus",
    "ProviderRootIndex",
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
    "build_multi_root_repository_index",
    "build_repository_index",
    "canonical_repository_index_bytes",
    "extract_package_function_symbols",
    "join_cross_root_symbols",
    "make_cross_root_symbol",
    "module_name_for_package_path",
    "open_frontiers_from_repository_index",
    "planning_category_inventory",
    "planning_path_category",
    "write_provider_index_baseline",
    "write_provider_index_baseline_from_snapshot",
]
