"""Snapshot-bound, body-free vector nomination for code symbols.

This is deliberately a small adapter over :mod:`analysis_ast_index`.  It does
not parse source, retain source bodies, or decide that a similar symbol is a
valid repair target.  Its only job is to make a *complete*, exact-snapshot
collection of symbol vectors available for recall.  Every result is marked
``semantic_authority=False`` and carries the roots needed to reject stale or
mixed-snapshot use.

The public records are immutable and content addressed.  ``previous`` is a
cache/transition input only: a clean build supplied with the same current
rows, reviewed lineage, and tombstones has the same index identity as an
incremental build.  A blob relocation is not a semantic rename; it is exposed
as lineage only when a caller supplies a reviewed lineage receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .analysis_ast_index import AnalysisASTIndex, IndexedASTPath


CODE_SYMBOL_VECTOR_INDEX_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-symbol-vector-index@1"
)
CODE_SYMBOL_VECTOR_ROW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-symbol-vector-row@1"
)
CODE_SYMBOL_VECTOR_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-vector-query@1"
)
CODE_SYMBOL_VECTOR_HIT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-vector-hit@1"
)
CODE_SYMBOL_VECTOR_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-vector-result@1"
)
CODE_SYMBOL_VECTOR_TOMBSTONE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-symbol-vector-tombstone@1"
)
CODE_SYMBOL_VECTOR_LINEAGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-symbol-vector-lineage@1"
)
CODE_SYMBOL_VECTOR_CONFIG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-symbol-vector-config@1"
)

DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_ROW_BYTES = 8_192
HARD_MAX_ROW_BYTES = 65_536
DEFAULT_MAX_METADATA_ITEMS = 32
DEFAULT_MAX_REFERENCE_BYTES = 320
DEFAULT_MAX_RESULTS = 50
HARD_MAX_RESULTS = 200
_BODY_KEYS = frozenset({
    "body", "source", "source_body", "source_text", "source_code",
    "contents", "content", "bytes", "text", "raw", "ast", "ast_body",
    "embedding_body", "prompt", "completion", "model_output",
})
_METRICS = frozenset({"cosine", "dot_product"})
_NORMALIZATIONS = frozenset({"l2", "none"})
_TOMBSTONE_REASONS = frozenset({"path_deleted", "blob_changed", "symbol_removed"})
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*")


class CodeSymbolVectorIndexError(ValueError):
    """The index is malformed, incomplete, or unsafe to use."""


class CodeSymbolVectorIndexIntegrityError(CodeSymbolVectorIndexError):
    """A claimed content identity or exact binding did not verify."""


class CodeSymbolVectorIndexStaleError(CodeSymbolVectorIndexError):
    """A query or hit is bound to a different snapshot/configuration."""


class CodeSymbolVectorIndexBoundsError(CodeSymbolVectorIndexError):
    """A compact row or query exceeds the admitted bounds."""


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodeSymbolVectorIndexIntegrityError("canonical JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CodeSymbolVectorIndexIntegrityError("canonical JSON keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    raise CodeSymbolVectorIndexIntegrityError(
        f"unsupported canonical value: {type(value).__name__}"
    )


def canonical_code_symbol_vector_index_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _canonical(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CodeSymbolVectorIndexError):
            raise
        raise CodeSymbolVectorIndexIntegrityError("index value is not canonical JSON") from exc


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:" + hashlib.sha256(
        canonical_code_symbol_vector_index_bytes(value)
    ).hexdigest()


def _text(value: Any, name: str, *, required: bool = True, maximum: int = 512) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise CodeSymbolVectorIndexError(f"{name} is required")
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise CodeSymbolVectorIndexBoundsError(f"{name} is invalid or exceeds its bound")
    return result


def _path(value: Any) -> str:
    raw = _text(value, "path").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    parsed = PurePosixPath(raw)
    if not raw or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != raw.rstrip("/"):
        raise CodeSymbolVectorIndexError(f"invalid repository path: {value!r}")
    return parsed.as_posix()


def _safe_value(value: Any, name: str = "reference") -> str:
    result = _text(value, name, maximum=DEFAULT_MAX_REFERENCE_BYTES)
    # References can name symbols and immutable receipts, but never carry a
    # source/body field encoded in a mapping or an unbounded multiline body.
    if "\n" in result or "\r" in result:
        raise CodeSymbolVectorIndexError(f"{name} must be a compact reference")
    return result


def _references(value: Any, name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        raise CodeSymbolVectorIndexError(f"{name} must contain references, not mappings")
    else:
        try:
            values = iter(value)
        except TypeError:
            values = (value,)
    result = tuple(sorted({_safe_value(item, name) for item in values}))
    if len(result) > DEFAULT_MAX_METADATA_ITEMS:
        raise CodeSymbolVectorIndexBoundsError(f"{name} exceeds its item bound")
    return result


def _reject_bodies(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _BODY_KEYS:
                raise CodeSymbolVectorIndexError("code-symbol index rows must not contain bodies")
            _reject_bodies(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_bodies(child)


def _vector(value: Any, dimensions: int, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise CodeSymbolVectorIndexError(f"{name} must be a numeric vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CodeSymbolVectorIndexError(f"{name} must be a numeric vector") from exc
    if len(result) != dimensions:
        raise CodeSymbolVectorIndexError(
            f"{name} dimension mismatch: expected {dimensions}, got {len(result)}"
        )
    if not all(math.isfinite(item) for item in result):
        raise CodeSymbolVectorIndexError(f"{name} contains non-finite values")
    return result


def _is_l2_normalized(vector: Sequence[float]) -> bool:
    return abs(math.sqrt(sum(item * item for item in vector)) - 1.0) <= 1e-6


def _module_for_path(path: str) -> str:
    name = path[:-3] if path.endswith(".py") else path
    parts = list(PurePosixPath(name).parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@dataclass(frozen=True)
class CodeVectorIndexConfig:
    """Versioned vector/chunking parameters that must never be mixed."""

    producer_id: str
    chunker_id: str
    normalization: str
    model_id: str
    model_revision: str
    dimensions: int
    metric: str
    configuration_id: str

    def __post_init__(self) -> None:
        for name in ("producer_id", "chunker_id", "model_id", "model_revision", "configuration_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        normalization = _text(self.normalization, "normalization").casefold()
        metric = _text(self.metric, "metric").casefold()
        if normalization not in _NORMALIZATIONS:
            raise CodeSymbolVectorIndexError("normalization must be l2 or none")
        if metric not in _METRICS:
            raise CodeSymbolVectorIndexError("metric must be cosine or dot_product")
        if isinstance(self.dimensions, bool) or int(self.dimensions) < 1 or int(self.dimensions) > 65_536:
            raise CodeSymbolVectorIndexError("dimensions must be an integer from 1 through 65536")
        if metric == "cosine" and normalization != "l2":
            raise CodeSymbolVectorIndexError("cosine metric requires l2 normalization")
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "dimensions", int(self.dimensions))

    @property
    def config_id(self) -> str:
        return _identity("code-vector-config", self.to_dict(include_config_id=False))

    def to_dict(self, *, include_config_id: bool = True) -> dict[str, Any]:
        result = {"schema": CODE_SYMBOL_VECTOR_CONFIG_SCHEMA, **asdict(self)}
        if include_config_id:
            result["config_id"] = self.config_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeVectorIndexConfig":
        _reject_bodies(value)
        allowed = {"schema", "config_id", *cls.__dataclass_fields__}
        unknown = set(value).difference(allowed)
        if unknown or value.get("schema", CODE_SYMBOL_VECTOR_CONFIG_SCHEMA) != CODE_SYMBOL_VECTOR_CONFIG_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported code vector config payload")
        result = cls(**{name: value.get(name, "") for name in cls.__dataclass_fields__})
        claimed = str(value.get("config_id") or "")
        if claimed and claimed != result.config_id:
            raise CodeSymbolVectorIndexIntegrityError("code vector config identity mismatch")
        return result


@dataclass(frozen=True)
class CodeSymbolASTSidecarRef:
    """Bounded references to rich AST facts; no AST/source body is copied."""

    ast_record_id: str
    blob_identity: str
    source_sha256: str
    symbol_hash: str = ""
    signature_refs: tuple[str, ...] = ()
    call_refs: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    error_refs: tuple[str, ...] = ()
    documentation_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()
    ownership_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("ast_record_id", "blob_identity", "source_sha256"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "symbol_hash", _text(self.symbol_hash, "symbol_hash", required=False))
        for name in (
            "signature_refs", "call_refs", "effect_refs", "error_refs",
            "documentation_refs", "test_refs", "ownership_refs",
        ):
            object.__setattr__(self, name, _references(getattr(self, name), name))

    @property
    def sidecar_id(self) -> str:
        return _identity("code-symbol-ast-sidecar", self.to_dict(include_sidecar_id=False))

    def to_dict(self, *, include_sidecar_id: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["schema"] = "ipfs_accelerate_py/agent-supervisor/code-symbol-ast-sidecar@1"
        result = {key: result[key] for key in sorted(result)}
        if include_sidecar_id:
            result["sidecar_id"] = self.sidecar_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeSymbolASTSidecarRef":
        _reject_bodies(value)
        allowed = {"schema", "sidecar_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed):
            raise CodeSymbolVectorIndexIntegrityError("unknown AST sidecar fields")
        if value.get("schema", "ipfs_accelerate_py/agent-supervisor/code-symbol-ast-sidecar@1") != "ipfs_accelerate_py/agent-supervisor/code-symbol-ast-sidecar@1":
            raise CodeSymbolVectorIndexIntegrityError("unsupported AST sidecar schema")
        result = cls(**{name: value.get(name, ()) for name in cls.__dataclass_fields__})
        claimed = str(value.get("sidecar_id") or "")
        if claimed and claimed != result.sidecar_id:
            raise CodeSymbolVectorIndexIntegrityError("AST sidecar identity mismatch")
        return result


@dataclass(frozen=True)
class CodeSymbolLineage:
    """An explicitly reviewed, blob-preserving relocation fact.

    ``old_symbol`` and ``new_symbol`` are intentionally not asserted to be
    equivalent.  The receipt says only that the immutable blob moved and a
    reviewer supplied a lineage reference.
    """

    old_path: str
    new_path: str
    blob_identity: str
    review_ref: str
    old_symbol: str = ""
    new_symbol: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "old_path", _path(self.old_path))
        object.__setattr__(self, "new_path", _path(self.new_path))
        if self.old_path == self.new_path:
            raise CodeSymbolVectorIndexError("lineage must describe a relocation")
        object.__setattr__(self, "blob_identity", _text(self.blob_identity, "blob_identity"))
        object.__setattr__(self, "review_ref", _safe_value(self.review_ref, "review_ref"))
        object.__setattr__(self, "old_symbol", _text(self.old_symbol, "old_symbol", required=False))
        object.__setattr__(self, "new_symbol", _text(self.new_symbol, "new_symbol", required=False))

    @property
    def lineage_id(self) -> str:
        return _identity("code-symbol-lineage", self.to_dict(include_lineage_id=False))

    def to_dict(self, *, include_lineage_id: bool = True) -> dict[str, Any]:
        result = {"schema": CODE_SYMBOL_VECTOR_LINEAGE_SCHEMA, **asdict(self)}
        if include_lineage_id:
            result["lineage_id"] = self.lineage_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeSymbolLineage":
        _reject_bodies(value)
        allowed = {"schema", "lineage_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_LINEAGE_SCHEMA) != CODE_SYMBOL_VECTOR_LINEAGE_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported lineage payload")
        result = cls(**{name: value.get(name, "") for name in cls.__dataclass_fields__})
        claimed = str(value.get("lineage_id") or "")
        if claimed and claimed != result.lineage_id:
            raise CodeSymbolVectorIndexIntegrityError("lineage identity mismatch")
        return result


@dataclass(frozen=True)
class CodeSymbolIndexRow:
    """One bounded vector and body-free symbol provenance row."""

    path: str
    symbol: str
    qualified_symbol: str
    line_start: int
    line_end: int
    sidecar: CodeSymbolASTSidecarRef
    embedding: tuple[float, ...]
    metadata_refs: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        for name in ("symbol", "qualified_symbol"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("line_start", "line_end"):
            value = int(getattr(self, name))
            if value < 1:
                raise CodeSymbolVectorIndexError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.line_end < self.line_start:
            raise CodeSymbolVectorIndexError("line_end must not precede line_start")
        sidecar = self.sidecar
        if not isinstance(sidecar, CodeSymbolASTSidecarRef):
            if not isinstance(sidecar, Mapping):
                raise CodeSymbolVectorIndexError("row sidecar must be an AST sidecar reference")
            sidecar = CodeSymbolASTSidecarRef.from_dict(sidecar)
        object.__setattr__(self, "sidecar", sidecar)
        if isinstance(self.embedding, (str, bytes, bytearray, Mapping)):
            raise CodeSymbolVectorIndexError("row embedding must be numeric")
        vector = tuple(float(item) for item in self.embedding)
        if not vector or not all(math.isfinite(item) for item in vector):
            raise CodeSymbolVectorIndexError("row embedding must be finite and non-empty")
        object.__setattr__(self, "embedding", vector)
        object.__setattr__(self, "metadata_refs", _references(self.metadata_refs, "metadata_refs"))
        object.__setattr__(self, "lineage_ids", _references(self.lineage_ids, "lineage_ids"))
        if len(canonical_code_symbol_vector_index_bytes(self.to_dict(include_row_id=False))) > HARD_MAX_ROW_BYTES:
            raise CodeSymbolVectorIndexBoundsError("code symbol row exceeds hard bound")

    @property
    def row_id(self) -> str:
        return _identity("code-symbol-vector-row", self.to_dict(include_row_id=False))

    @property
    def vector(self) -> tuple[float, ...]:
        return self.embedding

    def to_dict(self, *, include_row_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CODE_SYMBOL_VECTOR_ROW_SCHEMA,
            "path": self.path,
            "symbol": self.symbol,
            "qualified_symbol": self.qualified_symbol,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "sidecar": self.sidecar.to_dict(),
            "embedding": list(self.embedding),
            "metadata_refs": list(self.metadata_refs),
            "lineage_ids": list(self.lineage_ids),
        }
        if include_row_id:
            result["row_id"] = self.row_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeSymbolIndexRow":
        _reject_bodies(value)
        allowed = {"schema", "row_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_ROW_SCHEMA) != CODE_SYMBOL_VECTOR_ROW_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported code symbol row payload")
        result = cls(**{name: value.get(name, ()) for name in cls.__dataclass_fields__})
        claimed = str(value.get("row_id") or "")
        if claimed and claimed != result.row_id:
            raise CodeSymbolVectorIndexIntegrityError("code symbol row identity mismatch")
        return result


@dataclass(frozen=True)
class CodeSymbolIndexTombstone:
    path: str
    symbol: str
    row_id: str
    blob_identity: str
    source_sha256: str
    ast_record_id: str
    reason: str
    replacement_row_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        for name in ("symbol", "row_id", "blob_identity", "source_sha256", "ast_record_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        reason = _text(self.reason, "reason")
        if reason not in _TOMBSTONE_REASONS:
            raise CodeSymbolVectorIndexError("unsupported code symbol tombstone reason")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "replacement_row_id", _text(self.replacement_row_id, "replacement_row_id", required=False))

    @property
    def tombstone_id(self) -> str:
        return _identity("code-symbol-vector-tombstone", self.to_dict(include_tombstone_id=False))

    def to_dict(self, *, include_tombstone_id: bool = True) -> dict[str, Any]:
        result = {"schema": CODE_SYMBOL_VECTOR_TOMBSTONE_SCHEMA, **asdict(self)}
        if include_tombstone_id:
            result["tombstone_id"] = self.tombstone_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeSymbolIndexTombstone":
        _reject_bodies(value)
        allowed = {"schema", "tombstone_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_TOMBSTONE_SCHEMA) != CODE_SYMBOL_VECTOR_TOMBSTONE_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported tombstone payload")
        result = cls(**{name: value.get(name, "") for name in cls.__dataclass_fields__})
        claimed = str(value.get("tombstone_id") or "")
        if claimed and claimed != result.tombstone_id:
            raise CodeSymbolVectorIndexIntegrityError("tombstone identity mismatch")
        return result


@dataclass(frozen=True)
class CodeVectorIndexSnapshot:
    """The complete immutable vector index for one exact repository tree."""

    forest_id: str
    tree_id: str
    coverage_id: str
    coverage_complete: bool
    included_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    ast_index_id: str
    config: CodeVectorIndexConfig
    rows: tuple[CodeSymbolIndexRow, ...]
    tombstones: tuple[CodeSymbolIndexTombstone, ...] = ()
    lineage: tuple[CodeSymbolLineage, ...] = ()
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES

    def __post_init__(self) -> None:
        for name in ("forest_id", "tree_id", "coverage_id", "ast_index_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.coverage_complete is not True:
            raise CodeSymbolVectorIndexError("incomplete coverage cannot produce a code vector index")
        if not 512 <= int(self.max_row_bytes) <= HARD_MAX_ROW_BYTES:
            raise CodeSymbolVectorIndexBoundsError("max_row_bytes is outside the hard bound")
        object.__setattr__(self, "max_row_bytes", int(self.max_row_bytes))
        included = tuple(sorted({_path(item) for item in self.included_paths}))
        excluded = tuple(sorted({_path(item) for item in self.excluded_paths}))
        if set(included).intersection(excluded):
            raise CodeSymbolVectorIndexError("included and excluded paths overlap")
        object.__setattr__(self, "included_paths", included)
        object.__setattr__(self, "excluded_paths", excluded)
        config = self.config if isinstance(self.config, CodeVectorIndexConfig) else CodeVectorIndexConfig.from_dict(self.config)
        object.__setattr__(self, "config", config)
        rows = tuple(self.rows)
        if len(rows) > DEFAULT_MAX_ROWS:
            raise CodeSymbolVectorIndexBoundsError("code vector index exceeds its row bound")
        if any(not isinstance(row, CodeSymbolIndexRow) for row in rows):
            raise CodeSymbolVectorIndexError("snapshot rows must be CodeSymbolIndexRow values")
        rows = tuple(sorted(rows, key=lambda row: (row.path, row.qualified_symbol, row.row_id)))
        if len({row.row_id for row in rows}) != len(rows):
            raise CodeSymbolVectorIndexIntegrityError("duplicate code symbol rows")
        for row in rows:
            if row.path not in included:
                raise CodeSymbolVectorIndexIntegrityError("row path is absent from complete coverage")
            _vector(row.embedding, config.dimensions, name="row embedding")
            if config.normalization == "l2" and not _is_l2_normalized(row.embedding):
                raise CodeSymbolVectorIndexIntegrityError("row embedding violates l2 normalization")
            if len(canonical_code_symbol_vector_index_bytes(row.to_dict())) > self.max_row_bytes:
                raise CodeSymbolVectorIndexBoundsError("code symbol row exceeds configured bound")
        object.__setattr__(self, "rows", rows)
        tombstones = tuple(self.tombstones)
        if any(not isinstance(item, CodeSymbolIndexTombstone) for item in tombstones):
            raise CodeSymbolVectorIndexError("snapshot tombstones must be canonical values")
        by_tombstone = {item.tombstone_id: item for item in tombstones}
        object.__setattr__(self, "tombstones", tuple(sorted(by_tombstone.values(), key=lambda item: item.tombstone_id)))
        lineage = tuple(self.lineage)
        if any(not isinstance(item, CodeSymbolLineage) for item in lineage):
            raise CodeSymbolVectorIndexError("snapshot lineage must be canonical values")
        by_lineage = {item.lineage_id: item for item in lineage}
        lineage = tuple(sorted(by_lineage.values(), key=lambda item: item.lineage_id))
        current_blobs = {(row.path, row.sidecar.blob_identity) for row in rows}
        for item in lineage:
            if (item.new_path, item.blob_identity) not in current_blobs:
                raise CodeSymbolVectorIndexIntegrityError("lineage does not bind a current blob relocation")
        known_lineage = {item.lineage_id for item in lineage}
        if any(set(row.lineage_ids).difference(known_lineage) for row in rows):
            raise CodeSymbolVectorIndexIntegrityError("row references unreviewed or forged lineage")
        object.__setattr__(self, "lineage", lineage)

    @property
    def index_id(self) -> str:
        return _identity("code-symbol-vector-index", self._content_dict())

    @property
    def index_root_id(self) -> str:
        return self.index_id

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_SYMBOL_VECTOR_INDEX_SCHEMA,
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "coverage_id": self.coverage_id,
            "coverage_complete": True,
            "included_paths": list(self.included_paths),
            "excluded_paths": list(self.excluded_paths),
            "ast_index_id": self.ast_index_id,
            "config": self.config.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "tombstones": [item.to_dict() for item in self.tombstones],
            "lineage": [item.to_dict() for item in self.lineage],
            "max_row_bytes": self.max_row_bytes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"index_id": self.index_id, **self._content_dict()}

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_code_symbol_vector_index_bytes(self.to_dict()).decode("utf-8")
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeVectorIndexSnapshot":
        _reject_bodies(value)
        allowed = {"index_id", "schema", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_INDEX_SCHEMA) != CODE_SYMBOL_VECTOR_INDEX_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported vector snapshot payload")
        result = cls(
            forest_id=value.get("forest_id", ""), tree_id=value.get("tree_id", ""),
            coverage_id=value.get("coverage_id", ""), coverage_complete=value.get("coverage_complete", False),
            included_paths=tuple(value.get("included_paths") or ()), excluded_paths=tuple(value.get("excluded_paths") or ()),
            ast_index_id=value.get("ast_index_id", ""), config=CodeVectorIndexConfig.from_dict(value.get("config") or {}),
            rows=tuple(CodeSymbolIndexRow.from_dict(item) for item in value.get("rows") or ()),
            tombstones=tuple(CodeSymbolIndexTombstone.from_dict(item) for item in value.get("tombstones") or ()),
            lineage=tuple(CodeSymbolLineage.from_dict(item) for item in value.get("lineage") or ()),
            max_row_bytes=value.get("max_row_bytes", DEFAULT_MAX_ROW_BYTES),
        )
        claimed = str(value.get("index_id") or "")
        if claimed and claimed != result.index_id:
            raise CodeSymbolVectorIndexIntegrityError("code vector snapshot identity mismatch")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "CodeVectorIndexSnapshot":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise CodeSymbolVectorIndexIntegrityError("vector snapshot JSON must be an object")
        return cls.from_dict(payload)

    def search(self, query: "CodeVectorQuery | Sequence[float]", *, max_results: int | None = None) -> "CodeVectorSearchResult":
        if not isinstance(query, CodeVectorQuery):
            query = CodeVectorQuery.for_snapshot(self, query_vector=query, max_results=max_results or DEFAULT_MAX_RESULTS)
        return search_code_symbol_vector_index(self, query)

    query = search


@dataclass(frozen=True)
class CodeVectorQuery:
    forest_id: str
    tree_id: str
    index_id: str
    config_id: str
    dimensions: int
    metric: str
    query_vector: tuple[float, ...]
    max_results: int = DEFAULT_MAX_RESULTS
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("forest_id", "tree_id", "index_id", "config_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if isinstance(self.dimensions, bool) or int(self.dimensions) < 1:
            raise CodeSymbolVectorIndexError("query dimensions must be positive")
        object.__setattr__(self, "dimensions", int(self.dimensions))
        metric = _text(self.metric, "metric").casefold()
        if metric not in _METRICS:
            raise CodeSymbolVectorIndexError("unsupported query metric")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "query_vector", _vector(self.query_vector, self.dimensions, name="query vector"))
        if isinstance(self.max_results, bool) or not 1 <= int(self.max_results) <= HARD_MAX_RESULTS:
            raise CodeSymbolVectorIndexBoundsError("max_results is outside the hard bound")
        object.__setattr__(self, "max_results", int(self.max_results))
        if self.semantic_authority is not False:
            raise CodeSymbolVectorIndexIntegrityError("vector queries cannot claim semantic authority")
        object.__setattr__(self, "semantic_authority", False)

    @property
    def query_id(self) -> str:
        return _identity("code-vector-query", self.to_dict(include_query_id=False))

    def to_dict(self, *, include_query_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": CODE_SYMBOL_VECTOR_QUERY_SCHEMA, **asdict(self), "semantic_authority": False}
        result["query_vector"] = list(self.query_vector)
        if include_query_id:
            result["query_id"] = self.query_id
        return result

    @classmethod
    def for_snapshot(cls, snapshot: CodeVectorIndexSnapshot, *, query_vector: Sequence[float], max_results: int = DEFAULT_MAX_RESULTS) -> "CodeVectorQuery":
        return cls(snapshot.forest_id, snapshot.tree_id, snapshot.index_id, snapshot.config.config_id, snapshot.config.dimensions, snapshot.config.metric, tuple(query_vector), max_results)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeVectorQuery":
        _reject_bodies(value)
        allowed = {"schema", "query_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_QUERY_SCHEMA) != CODE_SYMBOL_VECTOR_QUERY_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported code vector query payload")
        result = cls(**{name: value.get(name, ()) for name in cls.__dataclass_fields__})
        claimed = str(value.get("query_id") or "")
        if claimed and claimed != result.query_id:
            raise CodeSymbolVectorIndexIntegrityError("code vector query identity mismatch")
        return result


@dataclass(frozen=True)
class CodeVectorHit:
    row: CodeSymbolIndexRow
    index_id: str
    query_id: str
    score: float
    rank: int
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.row, CodeSymbolIndexRow):
            if not isinstance(self.row, Mapping):
                raise CodeSymbolVectorIndexError("hit row must be canonical")
            object.__setattr__(self, "row", CodeSymbolIndexRow.from_dict(self.row))
        object.__setattr__(self, "index_id", _text(self.index_id, "index_id"))
        object.__setattr__(self, "query_id", _text(self.query_id, "query_id"))
        if not math.isfinite(float(self.score)):
            raise CodeSymbolVectorIndexError("vector hit score must be finite")
        object.__setattr__(self, "score", float(self.score))
        if isinstance(self.rank, bool) or int(self.rank) < 1:
            raise CodeSymbolVectorIndexError("hit rank must be positive")
        object.__setattr__(self, "rank", int(self.rank))
        if self.semantic_authority is not False:
            raise CodeSymbolVectorIndexIntegrityError("vector hits cannot claim semantic authority")
        object.__setattr__(self, "semantic_authority", False)

    @property
    def hit_id(self) -> str:
        return _identity("code-vector-hit", self.to_dict(include_hit_id=False))

    @property
    def row_id(self) -> str:
        return self.row.row_id

    def to_dict(self, *, include_hit_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": CODE_SYMBOL_VECTOR_HIT_SCHEMA, "row": self.row.to_dict(), "index_id": self.index_id, "query_id": self.query_id, "score": self.score, "rank": self.rank, "semantic_authority": False}
        if include_hit_id:
            result["hit_id"] = self.hit_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeVectorHit":
        _reject_bodies(value)
        allowed = {"schema", "hit_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_HIT_SCHEMA) != CODE_SYMBOL_VECTOR_HIT_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported code vector hit payload")
        result = cls(**{name: value.get(name, ()) for name in cls.__dataclass_fields__})
        claimed = str(value.get("hit_id") or "")
        if claimed and claimed != result.hit_id:
            raise CodeSymbolVectorIndexIntegrityError("code vector hit identity mismatch")
        return result


@dataclass(frozen=True)
class CodeVectorSearchResult:
    query: CodeVectorQuery
    index_id: str
    hits: tuple[CodeVectorHit, ...]
    complete: bool = True
    searched_row_count: int = 0
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        query = self.query if isinstance(self.query, CodeVectorQuery) else CodeVectorQuery.from_dict(self.query)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "index_id", _text(self.index_id, "index_id"))
        hits = tuple(self.hits)
        if any(not isinstance(hit, CodeVectorHit) for hit in hits):
            raise CodeSymbolVectorIndexError("search result hits must be canonical")
        if self.complete is not True:
            raise CodeSymbolVectorIndexIntegrityError("incomplete vector results are not admissible")
        if self.searched_row_count < 0:
            raise CodeSymbolVectorIndexError("searched_row_count must not be negative")
        if self.semantic_authority is not False:
            raise CodeSymbolVectorIndexIntegrityError("vector results cannot claim semantic authority")
        if query.index_id != self.index_id:
            raise CodeSymbolVectorIndexIntegrityError("search result query is bound to a different index")
        if any(hit.index_id != self.index_id or hit.query_id != query.query_id for hit in hits):
            raise CodeSymbolVectorIndexIntegrityError("search result mixes stale or forged hits")
        if tuple(hit.rank for hit in hits) != tuple(range(1, len(hits) + 1)):
            raise CodeSymbolVectorIndexIntegrityError("search result ranks are not complete and deterministic")
        if int(self.searched_row_count) < len(hits):
            raise CodeSymbolVectorIndexIntegrityError("search result cannot be complete when fewer rows were searched than returned")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "semantic_authority", False)

    @property
    def result_id(self) -> str:
        return _identity("code-vector-result", self.to_dict(include_result_id=False))

    @property
    def results(self) -> tuple[CodeVectorHit, ...]:
        return self.hits

    def to_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": CODE_SYMBOL_VECTOR_RESULT_SCHEMA, "query": self.query.to_dict(), "index_id": self.index_id, "hits": [item.to_dict() for item in self.hits], "complete": True, "searched_row_count": self.searched_row_count, "semantic_authority": False}
        if include_result_id:
            result["result_id"] = self.result_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeVectorSearchResult":
        _reject_bodies(value)
        allowed = {"schema", "result_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed) or value.get("schema", CODE_SYMBOL_VECTOR_RESULT_SCHEMA) != CODE_SYMBOL_VECTOR_RESULT_SCHEMA:
            raise CodeSymbolVectorIndexIntegrityError("unsupported vector result payload")
        result = cls(
            query=CodeVectorQuery.from_dict(value.get("query") or {}), index_id=value.get("index_id", ""),
            hits=tuple(CodeVectorHit.from_dict(item) for item in value.get("hits") or ()), complete=value.get("complete", False),
            searched_row_count=int(value.get("searched_row_count", 0)), semantic_authority=value.get("semantic_authority", False),
        )
        claimed = str(value.get("result_id") or "")
        if claimed and claimed != result.result_id:
            raise CodeSymbolVectorIndexIntegrityError("vector result identity mismatch")
        return result


class VectorSearchProvider(Protocol):
    """Optional backend protocol.  It may nominate only complete local rows."""

    def embed(self, row: CodeSymbolIndexRow) -> Sequence[float]: ...


def _coerce_ast_index(value: Any) -> tuple[AnalysisASTIndex, Any | None]:
    if isinstance(value, AnalysisASTIndex):
        return value, None
    ast = getattr(value, "ast_index", None)
    if isinstance(ast, AnalysisASTIndex):
        return ast, value
    if isinstance(value, Mapping):
        if "ast_index" in value:
            ast_value = value["ast_index"]
            return (ast_value if isinstance(ast_value, AnalysisASTIndex) else AnalysisASTIndex.from_dict(ast_value)), value
        return AnalysisASTIndex.from_dict(value), None
    raise CodeSymbolVectorIndexError("an AnalysisASTIndex or RepositoryIndex is required")


def _feature_refs(features: Mapping[str, Any] | None, keys: Sequence[str]) -> dict[str, tuple[str, ...]]:
    if not features:
        return {}
    _reject_bodies(features)
    result: dict[str, tuple[str, ...]] = {}
    for key in keys:
        raw = features.get(key)
        if raw is not None:
            result[key] = _references(raw, f"feature refs for {key}")
    return result


def _sidecar(indexed: IndexedASTPath, symbol: str, features: Mapping[str, Any] | None) -> CodeSymbolASTSidecarRef:
    record = indexed.ast_record
    # These are compact AST facts, not source slices.  Extra docs/tests/etc.
    # must be immutable external references supplied by the caller.
    signature = tuple(item for item in record.interfaces if item.startswith(symbol + ":") or item.startswith(symbol + "("))
    calls = tuple(item for item in record.calls if item.startswith(symbol + "->"))
    effects = tuple(item for item in record.state_transitions if item.startswith(symbol + ":"))
    supplied = _feature_refs(features, ("error_refs", "documentation_refs", "test_refs", "ownership_refs", "effect_refs"))
    return CodeSymbolASTSidecarRef(
        ast_record_id=indexed.record_id, blob_identity=indexed.blob_identity,
        source_sha256=indexed.source_sha256, symbol_hash=record.symbol_hashes.get(symbol, ""),
        signature_refs=signature, call_refs=calls, effect_refs=tuple((*effects, *supplied.get("effect_refs", ()))),
        error_refs=supplied.get("error_refs", ()), documentation_refs=supplied.get("documentation_refs", ()),
        test_refs=supplied.get("test_refs", ()), ownership_refs=supplied.get("ownership_refs", ()),
    )


def _lookup_vector(vectors: Mapping[str, Sequence[float]] | Callable[[CodeSymbolIndexRow], Sequence[float]] | VectorSearchProvider, row: CodeSymbolIndexRow) -> Sequence[float]:
    if callable(vectors):
        return vectors(row)
    embed = getattr(vectors, "embed", None)
    if callable(embed):
        return embed(row)
    if not isinstance(vectors, Mapping):
        raise CodeSymbolVectorIndexError("vectors must be a mapping or admitted local embedding provider")
    for key in (row.qualified_symbol, f"{row.path}:{row.symbol}", row.symbol, row.row_id):
        if key in vectors:
            return vectors[key]
    raise CodeSymbolVectorIndexError(f"missing deterministic vector for {row.path}:{row.symbol}")


def _coerce_lineage(values: Any) -> tuple[CodeSymbolLineage, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, Mapping):
        values = (values,)
    return tuple(item if isinstance(item, CodeSymbolLineage) else CodeSymbolLineage.from_dict(item) for item in values)


def _prior_tombstones(previous: CodeVectorIndexSnapshot | None, current_rows: Sequence[CodeSymbolIndexRow]) -> tuple[CodeSymbolIndexTombstone, ...]:
    if previous is None:
        return ()
    current_by_key = {(item.path, item.symbol): item for item in current_rows}
    additions: list[CodeSymbolIndexTombstone] = []
    for old in previous.rows:
        replacement = current_by_key.get((old.path, old.symbol))
        if replacement is not None and replacement.row_id == old.row_id:
            continue
        reason = "blob_changed" if replacement is not None else "path_deleted"
        additions.append(CodeSymbolIndexTombstone(
            path=old.path, symbol=old.symbol, row_id=old.row_id,
            blob_identity=old.sidecar.blob_identity, source_sha256=old.sidecar.source_sha256,
            ast_record_id=old.sidecar.ast_record_id, reason=reason,
            replacement_row_id=replacement.row_id if replacement is not None else "",
        ))
    return tuple((*previous.tombstones, *additions))


def build_code_symbol_vector_index(
    ast_index: AnalysisASTIndex | Any,
    *,
    forest_id: str = "",
    tree_id: str = "",
    coverage_id: str = "",
    included_paths: Iterable[str] | None = None,
    excluded_paths: Iterable[str] = (),
    coverage_complete: bool = True,
    producer_id: str = "code-symbol-vector-indexer@1",
    chunker_id: str = "ast-symbol@1",
    normalization: str = "l2",
    model_id: str = "deterministic-fixture",
    model_revision: str = "1",
    dimensions: int | None = None,
    metric: str = "cosine",
    configuration_id: str = "code-symbol-vector-default@1",
    vectors: Mapping[str, Sequence[float]] | Callable[[CodeSymbolIndexRow], Sequence[float]] | VectorSearchProvider | None = None,
    feature_references: Mapping[str, Mapping[str, Any]] | None = None,
    metadata_references: Mapping[str, Iterable[str]] | None = None,
    reviewed_lineage: Iterable[CodeSymbolLineage | Mapping[str, Any]] = (),
    tombstones: Iterable[CodeSymbolIndexTombstone | Mapping[str, Any]] | None = None,
    previous: CodeVectorIndexSnapshot | Mapping[str, Any] | None = None,
    previous_index: CodeVectorIndexSnapshot | Mapping[str, Any] | None = None,
    exhaustive: bool = False,
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES,
    **aliases: Any,
) -> CodeVectorIndexSnapshot:
    """Build one exact, complete snapshot from body-free AST records.

    ``vectors`` is intentionally required.  A caller must provide an admitted
    deterministic fixture mapping or a local provider; this adapter never
    sends source-derived text to an implicit remote embedding service.
    """
    if previous is not None and previous_index is not None:
        raise CodeSymbolVectorIndexError("provide only one of previous or previous_index")
    if previous is None:
        previous = previous_index
    if previous is not None and not isinstance(previous, CodeVectorIndexSnapshot):
        previous = CodeVectorIndexSnapshot.from_dict(previous)
    if aliases:
        # A few explicit spellings make producer integrations readable without
        # accepting unknown configuration that could silently change a root.
        alias_names = {"repository_forest_id": "forest_id", "repository_tree_id": "tree_id", "model": "model_id", "model_config_id": "configuration_id", "embedding_vectors": "vectors", "embeddings": "vectors", "embedding_provider": "vectors", "vector_provider": "vectors"}
        unknown = set(aliases).difference(alias_names)
        if unknown:
            raise CodeSymbolVectorIndexError("unknown code vector index options: " + ", ".join(sorted(unknown)))
        for source, target in alias_names.items():
            if source in aliases:
                if target == "forest_id" and not forest_id: forest_id = aliases[source]
                elif target == "tree_id" and not tree_id: tree_id = aliases[source]
                elif target == "model_id" and model_id == "deterministic-fixture": model_id = aliases[source]
                elif target == "configuration_id" and configuration_id == "code-symbol-vector-default@1": configuration_id = aliases[source]
                elif target == "vectors" and vectors is None: vectors = aliases[source]
    index, repository = _coerce_ast_index(ast_index)
    if repository is not None:
        snapshot = getattr(repository, "snapshot", None)
        if snapshot is not None:
            forest_id = forest_id or getattr(snapshot, "snapshot_id", "")
            tree_id = tree_id or getattr(snapshot, "head_tree_id", "")
        coverage_id = coverage_id or getattr(repository, "index_id", "")
        if included_paths is None:
            included_paths = index.paths
        if not excluded_paths:
            excluded_paths = tuple(sorted(set(getattr(repository, "path_rows", ()) and [row.path for row in repository.path_rows] or ()).difference(index.paths)))
        coverage_complete = coverage_complete and bool(getattr(repository, "safe_for_completion_reasoning", False))
    if not forest_id or not tree_id:
        raise CodeSymbolVectorIndexError("forest_id and tree_id are required; AST evidence alone cannot invent repository roots")
    if not coverage_id:
        coverage_id = index.index_id
    if included_paths is None:
        included_paths = index.paths
    if vectors is None:
        raise CodeSymbolVectorIndexError("vectors are required; no implicit embedding backend is admitted")
    paths = tuple(sorted({_path(item) for item in included_paths}))
    if set(index.paths).difference(paths):
        raise CodeSymbolVectorIndexError("complete coverage omits AST-indexed paths")
    prototypes: list[CodeSymbolIndexRow] = []
    for indexed in index.path_records:
        for symbol in indexed.ast_record.qualified_symbols:
            line_start, line_end = indexed.ast_record.symbol_lines.get(symbol, (1, 1))
            if line_start < 1: line_start = 1
            if line_end < line_start: line_end = line_start
            qualified = f"{_module_for_path(indexed.path)}.{symbol}".strip(".")
            features = (feature_references or {}).get(qualified) or (feature_references or {}).get(f"{indexed.path}:{symbol}") or (feature_references or {}).get(symbol)
            metadata = (metadata_references or {}).get(qualified) or (metadata_references or {}).get(f"{indexed.path}:{symbol}") or ()
            prototypes.append(CodeSymbolIndexRow(indexed.path, symbol, qualified, line_start, line_end, _sidecar(indexed, symbol, features), (0.0,), metadata_refs=tuple(metadata)))
    if not prototypes:
        raise CodeSymbolVectorIndexError("complete code vector index has no symbols")
    if dimensions is None:
        first = _lookup_vector(vectors, prototypes[0])
        try: dimensions = len(first)
        except TypeError as exc: raise CodeSymbolVectorIndexError("vector dimensions are unavailable") from exc
    config = CodeVectorIndexConfig(producer_id, chunker_id, normalization, model_id, model_revision, dimensions, metric, configuration_id)
    lineage = _coerce_lineage(reviewed_lineage)
    rows: list[CodeSymbolIndexRow] = []
    for prototype in prototypes:
        vector = _vector(_lookup_vector(vectors, prototype), config.dimensions, name="row embedding")
        if config.normalization == "l2" and not _is_l2_normalized(vector):
            raise CodeSymbolVectorIndexError("row embedding violates configured l2 normalization")
        # An empty new_symbol is deliberately a blob-relocation receipt for
        # every symbol projected from that blob.  It does not claim a rename;
        # a populated name narrows the reviewed provenance to one symbol.
        row_lineage = tuple(
            item.lineage_id for item in lineage
            if item.new_path == prototype.path
            and item.blob_identity == prototype.sidecar.blob_identity
            and (not item.new_symbol or item.new_symbol == prototype.symbol)
        )
        rows.append(CodeSymbolIndexRow(prototype.path, prototype.symbol, prototype.qualified_symbol, prototype.line_start, prototype.line_end, prototype.sidecar, vector, prototype.metadata_refs, row_lineage))
    # A reviewed relocation must bind actual old/current blobs.  It cannot be
    # used to assert that symbol names or semantics are equivalent.
    if lineage:
        if previous is None:
            raise CodeSymbolVectorIndexError("reviewed lineage requires the previous exact snapshot")
        old_blobs = {(row.path, row.sidecar.blob_identity) for row in previous.rows}
        new_blobs = {(row.path, row.sidecar.blob_identity) for row in rows}
        for item in lineage:
            if (item.old_path, item.blob_identity) not in old_blobs or (item.new_path, item.blob_identity) not in new_blobs:
                raise CodeSymbolVectorIndexIntegrityError("reviewed lineage does not preserve the exact moved blob")
    if tombstones is None:
        effective_tombstones = () if exhaustive else _prior_tombstones(previous, rows)
    else:
        effective_tombstones = tuple(item if isinstance(item, CodeSymbolIndexTombstone) else CodeSymbolIndexTombstone.from_dict(item) for item in tombstones)
    return CodeVectorIndexSnapshot(
        forest_id=forest_id, tree_id=tree_id, coverage_id=coverage_id, coverage_complete=coverage_complete,
        included_paths=paths, excluded_paths=tuple(excluded_paths), ast_index_id=index.index_id,
        config=config, rows=tuple(rows), tombstones=effective_tombstones, lineage=lineage,
        max_row_bytes=max_row_bytes,
    )


def _score(metric: str, query: Sequence[float], vector: Sequence[float]) -> float:
    dot = sum(left * right for left, right in zip(query, vector))
    if metric == "cosine":
        # Config validation guarantees l2-normalization for snapshot rows. A
        # query is checked here so direct construction cannot sneak one in.
        if not _is_l2_normalized(query):
            raise CodeSymbolVectorIndexError("cosine query violates l2 normalization")
    return dot


def search_code_symbol_vector_index(snapshot: CodeVectorIndexSnapshot | Mapping[str, Any], query: CodeVectorQuery | Mapping[str, Any] | Sequence[float], *, max_results: int | None = None) -> CodeVectorSearchResult:
    if not isinstance(snapshot, CodeVectorIndexSnapshot):
        snapshot = CodeVectorIndexSnapshot.from_dict(snapshot)
    if not isinstance(query, CodeVectorQuery):
        if isinstance(query, Mapping):
            query = CodeVectorQuery.from_dict(query)
        else:
            query = CodeVectorQuery.for_snapshot(snapshot, query_vector=query, max_results=max_results or DEFAULT_MAX_RESULTS)
    if max_results is not None and max_results != query.max_results:
        query = CodeVectorQuery(query.forest_id, query.tree_id, query.index_id, query.config_id, query.dimensions, query.metric, query.query_vector, max_results)
    if (query.forest_id, query.tree_id, query.index_id, query.config_id, query.dimensions, query.metric) != (snapshot.forest_id, snapshot.tree_id, snapshot.index_id, snapshot.config.config_id, snapshot.config.dimensions, snapshot.config.metric):
        raise CodeSymbolVectorIndexStaleError("code vector query roots/configuration do not match the current snapshot")
    if snapshot.config.normalization == "l2" and not _is_l2_normalized(query.query_vector):
        raise CodeSymbolVectorIndexError("query vector violates configured l2 normalization")
    ranked = sorted(((_score(snapshot.config.metric, query.query_vector, row.embedding), row) for row in snapshot.rows), key=lambda item: (-item[0], item[1].path, item[1].qualified_symbol, item[1].row_id))
    hits = tuple(CodeVectorHit(row, snapshot.index_id, query.query_id, score, rank + 1) for rank, (score, row) in enumerate(ranked[:query.max_results]))
    result = CodeVectorSearchResult(query, snapshot.index_id, hits, complete=True, searched_row_count=len(snapshot.rows))
    return validate_code_vector_search_result(snapshot, result)


def validate_code_vector_search_result(
    snapshot: CodeVectorIndexSnapshot | Mapping[str, Any],
    result: CodeVectorSearchResult | Mapping[str, Any],
) -> CodeVectorSearchResult:
    """Fail closed unless a response covers the exact current local row set.

    This is the boundary consumers should use when deserializing a persisted
    response.  A result envelope alone cannot prove that its claimed row count
    covers an index it does not contain.
    """
    if not isinstance(snapshot, CodeVectorIndexSnapshot):
        snapshot = CodeVectorIndexSnapshot.from_dict(snapshot)
    if not isinstance(result, CodeVectorSearchResult):
        result = CodeVectorSearchResult.from_dict(result)
    query = result.query
    if (query.forest_id, query.tree_id, query.index_id, query.config_id, query.dimensions, query.metric) != (snapshot.forest_id, snapshot.tree_id, snapshot.index_id, snapshot.config.config_id, snapshot.config.dimensions, snapshot.config.metric):
        raise CodeSymbolVectorIndexStaleError("code vector result roots/configuration do not match the current snapshot")
    if result.index_id != snapshot.index_id or result.complete is not True or result.searched_row_count != len(snapshot.rows):
        raise CodeSymbolVectorIndexIntegrityError("incomplete or stale code vector result")
    current = {row.row_id for row in snapshot.rows}
    if len({hit.row_id for hit in result.hits}) != len(result.hits) or any(hit.row_id not in current for hit in result.hits):
        raise CodeSymbolVectorIndexIntegrityError("code vector result contains a row absent from the exact snapshot")
    return result


# Readable aliases for adjacent retrieval adapters.
build_code_vector_index = build_code_symbol_vector_index
search_code_vector_index = search_code_symbol_vector_index
CodeSymbolVectorIndex = CodeVectorIndexSnapshot
CodeSymbolVectorIndexSnapshot = CodeVectorIndexSnapshot
CodeVectorIndexRow = CodeSymbolIndexRow
CodeVectorTombstone = CodeSymbolIndexTombstone


__all__ = [
    "CODE_SYMBOL_VECTOR_INDEX_SCHEMA", "CODE_SYMBOL_VECTOR_ROW_SCHEMA", "CODE_SYMBOL_VECTOR_QUERY_SCHEMA", "CODE_SYMBOL_VECTOR_HIT_SCHEMA", "CodeSymbolVectorIndexError", "CodeSymbolVectorIndexIntegrityError", "CodeSymbolVectorIndexStaleError", "CodeSymbolVectorIndexBoundsError", "CodeVectorIndexConfig", "CodeSymbolASTSidecarRef", "CodeSymbolLineage", "CodeSymbolIndexRow", "CodeSymbolIndexTombstone", "CodeVectorIndexSnapshot", "CodeVectorQuery", "CodeVectorHit", "CodeVectorSearchResult", "VectorSearchProvider", "canonical_code_symbol_vector_index_bytes", "build_code_symbol_vector_index", "build_code_vector_index", "search_code_symbol_vector_index", "search_code_vector_index", "validate_code_vector_search_result", "CodeSymbolVectorIndex", "CodeSymbolVectorIndexSnapshot", "CodeVectorIndexRow", "CodeVectorTombstone",
]
