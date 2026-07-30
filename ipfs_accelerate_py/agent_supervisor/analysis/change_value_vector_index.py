"""Snapshot-bound, body-free vector nomination for values and behaviors.

This adapter indexes *candidates* that may later satisfy a missing input or
required behavior.  It deliberately does **not**:

* retain source/AST bodies;
* claim that a nearest or same-typed value is compatible;
* grant ``semantic_authority`` to any hit or result.

Its job is high-recall, exact-snapshot nomination.  Identity binds forest/tree,
coverage, chunking/normalization, embedding model/revision/dimensions, distance
metric, configuration, included/excluded paths, and tombstones.  Queries must
bind the exact missing contract and consumer context.  Incremental update of
the same current rows, reviewed lineage, and tombstones equals a clean rebuild.

Poisoned, stale, cross-tree, forged, dimension-mismatched, or incomplete
results fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .analysis_ast_index import AnalysisASTIndex, IndexedASTPath
from .code_symbol_vector_index import (
    CodeSymbolIndexRow,
    CodeVectorIndexSnapshot,
)


CHANGE_VALUE_VECTOR_INDEX_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-index@1"
)
CHANGE_VALUE_VECTOR_ROW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-row@1"
)
CHANGE_VALUE_VECTOR_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-query@1"
)
CHANGE_VALUE_VECTOR_HIT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-hit@1"
)
CHANGE_VALUE_VECTOR_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-result@1"
)
CHANGE_VALUE_VECTOR_TOMBSTONE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-tombstone@1"
)
CHANGE_VALUE_VECTOR_LINEAGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-lineage@1"
)
CHANGE_VALUE_VECTOR_CONFIG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-vector-config@1"
)
CHANGE_VALUE_SIDECAR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-value-ast-graph-sidecar@1"
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
_TOMBSTONE_REASONS = frozenset({
    "path_deleted", "blob_changed", "value_removed", "symbol_removed",
})
_COMPATIBILITY_CLAIM_KEYS = frozenset({
    "compatible", "compatibility", "compatibility_claim",
    "type_compatible", "semantically_compatible", "is_compatible",
    "admits_compatibility", "proved_compatible",
})
_FACTORY_MARKERS = frozenset({
    "create", "make", "build", "factory", "from_dict", "from_json",
    "from_config", "get_instance", "instance", "builder",
})
_CONSTRUCTOR_MARKERS = frozenset({"__init__", "__new__", "constructor", "new"})
_SCHEMA_MARKERS = frozenset({
    "schema", "model", "dto", "payload", "message", "record",
})
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|test_fixtures?|fixtures?|mocks?|conftest)(/|$)",
    re.IGNORECASE,
)
_DOC_PATH_RE = re.compile(
    r"(^|/)(docs?|documentation|examples?|readme)(/|$)",
    re.IGNORECASE,
)
_CONFIG_MARKERS = frozenset({
    "config", "settings", "options", "feature_flag", "env", "getenv",
})


class ChangeValueVectorIndexError(ValueError):
    """The value index is malformed, incomplete, or unsafe to use."""


class ChangeValueVectorIndexIntegrityError(ChangeValueVectorIndexError):
    """A claimed content identity or exact binding did not verify."""


class ChangeValueVectorIndexStaleError(ChangeValueVectorIndexError):
    """A query or hit is bound to a different snapshot/configuration."""


class ChangeValueVectorIndexBoundsError(ChangeValueVectorIndexError):
    """A compact row or query exceeds the admitted bounds."""


class ChangeValueKind(str, Enum):
    """Closed kinds of value/behavior nomination rows."""

    VARIABLE = "variable"
    PARAMETER = "parameter"
    FIELD = "field"
    RETURN = "return"
    CONSTRUCTOR = "constructor"
    FACTORY = "factory"
    BUILDER = "builder"
    METHOD = "method"
    CLASS = "class"
    SCHEMA = "schema"
    TYPE = "type"
    CONFIG_PROVIDER = "config_provider"
    REQUEST_CONTEXT = "request_context"
    DI_BINDING = "di_binding"
    TEST = "test"
    FIXTURE = "fixture"
    DOCUMENTATION = "documentation"
    HISTORY = "history"
    BEHAVIOR = "behavior"
    EXPRESSION = "expression"
    SYMBOL = "symbol"


class ChangeValueSignal(str, Enum):
    """Signal families that may contribute to a nomination hit."""

    VECTOR = "vector"
    GRAPH = "graph"
    AST = "ast"
    TYPE = "type"
    LINEAGE = "lineage"
    HISTORY = "history"
    LEXICAL = "lexical"
    OWNERSHIP = "ownership"
    SCHEMA = "schema"
    TEST = "test"
    DOCUMENTATION = "documentation"


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChangeValueVectorIndexIntegrityError(
                "canonical JSON cannot contain NaN or infinity"
            )
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ChangeValueVectorIndexIntegrityError(
                "canonical JSON keys must be strings"
            )
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    raise ChangeValueVectorIndexIntegrityError(
        f"unsupported canonical value: {type(value).__name__}"
    )


def canonical_change_value_vector_index_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ChangeValueVectorIndexError):
            raise
        raise ChangeValueVectorIndexIntegrityError(
            "index value is not canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:" + hashlib.sha256(
        canonical_change_value_vector_index_bytes(value)
    ).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = 512,
) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ChangeValueVectorIndexError(f"{name} is required")
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise ChangeValueVectorIndexBoundsError(
            f"{name} is invalid or exceeds its bound"
        )
    return result


def _path(value: Any) -> str:
    raw = _text(value, "path").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    parsed = PurePosixPath(raw)
    if (
        not raw
        or parsed.is_absolute()
        or ".." in parsed.parts
        or parsed.as_posix() != raw.rstrip("/")
    ):
        raise ChangeValueVectorIndexError(f"invalid repository path: {value!r}")
    return parsed.as_posix()


def _safe_value(value: Any, name: str = "reference") -> str:
    result = _text(value, name, maximum=DEFAULT_MAX_REFERENCE_BYTES)
    if "\n" in result or "\r" in result:
        raise ChangeValueVectorIndexError(f"{name} must be a compact reference")
    return result


def _references(value: Any, name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        raise ChangeValueVectorIndexError(
            f"{name} must contain references, not mappings"
        )
    else:
        try:
            values = iter(value)
        except TypeError:
            values = (value,)
    result = tuple(sorted({_safe_value(item, name) for item in values}))
    if len(result) > DEFAULT_MAX_METADATA_ITEMS:
        raise ChangeValueVectorIndexBoundsError(
            f"{name} exceeds its item bound"
        )
    return result


def _reject_bodies(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            if folded in _BODY_KEYS:
                raise ChangeValueVectorIndexError(
                    "change-value index rows must not contain bodies"
                )
            if folded in _COMPATIBILITY_CLAIM_KEYS and bool(child):
                raise ChangeValueVectorIndexIntegrityError(
                    "same-typed or similar values receive no compatibility claim"
                )
            _reject_bodies(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_bodies(child)


def _vector(value: Any, dimensions: int, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ChangeValueVectorIndexError(f"{name} must be a numeric vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ChangeValueVectorIndexError(
            f"{name} must be a numeric vector"
        ) from exc
    if len(result) != dimensions:
        raise ChangeValueVectorIndexError(
            f"{name} dimension mismatch: expected {dimensions}, got {len(result)}"
        )
    if not all(math.isfinite(item) for item in result):
        raise ChangeValueVectorIndexError(f"{name} contains non-finite values")
    return result


def _is_l2_normalized(vector: Sequence[float]) -> bool:
    return abs(math.sqrt(sum(item * item for item in vector)) - 1.0) <= 1e-6


def _module_for_path(path: str) -> str:
    name = path[:-3] if path.endswith(".py") else path
    parts = list(PurePosixPath(name).parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _enum_value(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise ChangeValueVectorIndexError(
            f"unsupported {name}: {text}"
        ) from exc


def _infer_kind(path: str, symbol: str) -> ChangeValueKind:
    simple = symbol.rsplit(".", 1)[-1]
    lower = simple.casefold()
    if _TEST_PATH_RE.search(path) or lower.startswith("test_"):
        return ChangeValueKind.TEST
    if _DOC_PATH_RE.search(path):
        return ChangeValueKind.DOCUMENTATION
    if lower in _CONSTRUCTOR_MARKERS or simple == "__init__":
        return ChangeValueKind.CONSTRUCTOR
    if any(marker in lower for marker in _FACTORY_MARKERS):
        return ChangeValueKind.FACTORY
    if any(marker in lower for marker in _SCHEMA_MARKERS):
        return ChangeValueKind.SCHEMA
    if any(marker in lower for marker in _CONFIG_MARKERS):
        return ChangeValueKind.CONFIG_PROVIDER
    if simple[:1].isupper() and "_" not in simple:
        return ChangeValueKind.CLASS
    if "." in symbol and simple[:1].islower():
        return ChangeValueKind.METHOD
    return ChangeValueKind.SYMBOL


@dataclass(frozen=True)
class ChangeValueIndexConfig:
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
        for name in (
            "producer_id",
            "chunker_id",
            "model_id",
            "model_revision",
            "configuration_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        normalization = _text(self.normalization, "normalization").casefold()
        metric = _text(self.metric, "metric").casefold()
        if normalization not in _NORMALIZATIONS:
            raise ChangeValueVectorIndexError(
                "normalization must be l2 or none"
            )
        if metric not in _METRICS:
            raise ChangeValueVectorIndexError(
                "metric must be cosine or dot_product"
            )
        if (
            isinstance(self.dimensions, bool)
            or int(self.dimensions) < 1
            or int(self.dimensions) > 65_536
        ):
            raise ChangeValueVectorIndexError(
                "dimensions must be an integer from 1 through 65536"
            )
        if metric == "cosine" and normalization != "l2":
            raise ChangeValueVectorIndexError(
                "cosine metric requires l2 normalization"
            )
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "dimensions", int(self.dimensions))

    @property
    def config_id(self) -> str:
        return _identity(
            "change-value-vector-config",
            self.to_dict(include_config_id=False),
        )

    def to_dict(self, *, include_config_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": CHANGE_VALUE_VECTOR_CONFIG_SCHEMA,
            **asdict(self),
        }
        if include_config_id:
            result["config_id"] = self.config_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangeValueIndexConfig":
        _reject_bodies(value)
        allowed = {"schema", "config_id", *cls.__dataclass_fields__}
        unknown = set(value).difference(allowed)
        if (
            unknown
            or value.get("schema", CHANGE_VALUE_VECTOR_CONFIG_SCHEMA)
            != CHANGE_VALUE_VECTOR_CONFIG_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change value vector config payload"
            )
        result = cls(
            **{name: value.get(name, "") for name in cls.__dataclass_fields__}
        )
        claimed = str(value.get("config_id") or "")
        if claimed and claimed != result.config_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change value vector config identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueSidecarRef:
    """Bounded source/AST/graph references; no bodies are copied."""

    ast_record_id: str
    blob_identity: str
    source_sha256: str
    graph_node_refs: tuple[str, ...] = ()
    type_refs: tuple[str, ...] = ()
    constructor_refs: tuple[str, ...] = ()
    factory_refs: tuple[str, ...] = ()
    schema_refs: tuple[str, ...] = ()
    definition_use_refs: tuple[str, ...] = ()
    scope_refs: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    ownership_refs: tuple[str, ...] = ()
    documentation_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()
    history_refs: tuple[str, ...] = ()
    symbol_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("ast_record_id", "blob_identity", "source_sha256"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "symbol_hash",
            _text(self.symbol_hash, "symbol_hash", required=False),
        )
        for name in (
            "graph_node_refs",
            "type_refs",
            "constructor_refs",
            "factory_refs",
            "schema_refs",
            "definition_use_refs",
            "scope_refs",
            "effect_refs",
            "ownership_refs",
            "documentation_refs",
            "test_refs",
            "history_refs",
        ):
            object.__setattr__(
                self, name, _references(getattr(self, name), name)
            )

    @property
    def sidecar_id(self) -> str:
        return _identity(
            "change-value-sidecar",
            self.to_dict(include_sidecar_id=False),
        )

    def to_dict(self, *, include_sidecar_id: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["schema"] = CHANGE_VALUE_SIDECAR_SCHEMA
        result = {key: result[key] for key in sorted(result)}
        if include_sidecar_id:
            result["sidecar_id"] = self.sidecar_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangeValueSidecarRef":
        _reject_bodies(value)
        allowed = {"schema", "sidecar_id", *cls.__dataclass_fields__}
        if set(value).difference(allowed):
            raise ChangeValueVectorIndexIntegrityError(
                "unknown change-value sidecar fields"
            )
        if (
            value.get("schema", CHANGE_VALUE_SIDECAR_SCHEMA)
            != CHANGE_VALUE_SIDECAR_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change-value sidecar schema"
            )
        result = cls(
            **{
                name: value.get(name, ())
                for name in cls.__dataclass_fields__
            }
        )
        claimed = str(value.get("sidecar_id") or "")
        if claimed and claimed != result.sidecar_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change-value sidecar identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueLineage:
    """An explicitly reviewed, blob-preserving relocation fact.

    Relocation is not a semantic rename or compatibility claim.
    """

    old_path: str
    new_path: str
    blob_identity: str
    review_ref: str
    old_value: str = ""
    new_value: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "old_path", _path(self.old_path))
        object.__setattr__(self, "new_path", _path(self.new_path))
        if self.old_path == self.new_path:
            raise ChangeValueVectorIndexError(
                "lineage must describe a relocation"
            )
        object.__setattr__(
            self, "blob_identity", _text(self.blob_identity, "blob_identity")
        )
        object.__setattr__(
            self, "review_ref", _safe_value(self.review_ref, "review_ref")
        )
        object.__setattr__(
            self,
            "old_value",
            _text(self.old_value, "old_value", required=False),
        )
        object.__setattr__(
            self,
            "new_value",
            _text(self.new_value, "new_value", required=False),
        )

    @property
    def lineage_id(self) -> str:
        return _identity(
            "change-value-lineage",
            self.to_dict(include_lineage_id=False),
        )

    def to_dict(self, *, include_lineage_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": CHANGE_VALUE_VECTOR_LINEAGE_SCHEMA,
            **asdict(self),
        }
        if include_lineage_id:
            result["lineage_id"] = self.lineage_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangeValueLineage":
        _reject_bodies(value)
        allowed = {"schema", "lineage_id", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_LINEAGE_SCHEMA)
            != CHANGE_VALUE_VECTOR_LINEAGE_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change-value lineage payload"
            )
        result = cls(
            **{name: value.get(name, "") for name in cls.__dataclass_fields__}
        )
        claimed = str(value.get("lineage_id") or "")
        if claimed and claimed != result.lineage_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change-value lineage identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueIndexRow:
    """One bounded vector and body-free value/behavior provenance row."""

    path: str
    name: str
    qualified_name: str
    kind: ChangeValueKind
    line_start: int
    line_end: int
    sidecar: ChangeValueSidecarRef
    embedding: tuple[float, ...]
    type_ref: str = ""
    scope_ref: str = ""
    signal_provenance: tuple[str, ...] = ()
    metadata_refs: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()
    semantic_authority: bool = False
    compatibility_claim: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        for name in ("name", "qualified_name"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "kind", _enum_value(self.kind, ChangeValueKind, "kind")
        )
        for name in ("line_start", "line_end"):
            value = int(getattr(self, name))
            if value < 1:
                raise ChangeValueVectorIndexError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.line_end < self.line_start:
            raise ChangeValueVectorIndexError(
                "line_end must not precede line_start"
            )
        sidecar = self.sidecar
        if not isinstance(sidecar, ChangeValueSidecarRef):
            if not isinstance(sidecar, Mapping):
                raise ChangeValueVectorIndexError(
                    "row sidecar must be a change-value sidecar reference"
                )
            sidecar = ChangeValueSidecarRef.from_dict(sidecar)
        object.__setattr__(self, "sidecar", sidecar)
        if isinstance(self.embedding, (str, bytes, bytearray, Mapping)):
            raise ChangeValueVectorIndexError(
                "row embedding must be numeric"
            )
        vector = tuple(float(item) for item in self.embedding)
        if not vector or not all(math.isfinite(item) for item in vector):
            raise ChangeValueVectorIndexError(
                "row embedding must be finite and non-empty"
            )
        object.__setattr__(self, "embedding", vector)
        object.__setattr__(
            self, "type_ref", _text(self.type_ref, "type_ref", required=False)
        )
        object.__setattr__(
            self,
            "scope_ref",
            _text(self.scope_ref, "scope_ref", required=False),
        )
        signals = _references(self.signal_provenance, "signal_provenance")
        for signal in signals:
            try:
                ChangeValueSignal(signal)
            except ValueError as exc:
                raise ChangeValueVectorIndexError(
                    f"unsupported signal provenance: {signal}"
                ) from exc
        object.__setattr__(self, "signal_provenance", signals)
        object.__setattr__(
            self, "metadata_refs", _references(self.metadata_refs, "metadata_refs")
        )
        object.__setattr__(
            self, "lineage_ids", _references(self.lineage_ids, "lineage_ids")
        )
        if self.semantic_authority is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "value index rows cannot claim semantic authority"
            )
        if self.compatibility_claim is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "same-typed or similar values receive no compatibility claim"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "compatibility_claim", False)
        encoded = canonical_change_value_vector_index_bytes(
            self.to_dict(include_row_id=False)
        )
        if len(encoded) > HARD_MAX_ROW_BYTES:
            raise ChangeValueVectorIndexBoundsError(
                "change value row exceeds hard bound"
            )

    @property
    def row_id(self) -> str:
        return _identity(
            "change-value-vector-row",
            self.to_dict(include_row_id=False),
        )

    @property
    def vector(self) -> tuple[float, ...]:
        return self.embedding

    def to_dict(self, *, include_row_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CHANGE_VALUE_VECTOR_ROW_SCHEMA,
            "path": self.path,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind.value
            if isinstance(self.kind, ChangeValueKind)
            else str(self.kind),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "sidecar": self.sidecar.to_dict(),
            "embedding": list(self.embedding),
            "type_ref": self.type_ref,
            "scope_ref": self.scope_ref,
            "signal_provenance": list(self.signal_provenance),
            "metadata_refs": list(self.metadata_refs),
            "lineage_ids": list(self.lineage_ids),
            "semantic_authority": False,
            "compatibility_claim": False,
        }
        if include_row_id:
            result["row_id"] = self.row_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangeValueIndexRow":
        _reject_bodies(value)
        allowed = {"schema", "row_id", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_ROW_SCHEMA)
            != CHANGE_VALUE_VECTOR_ROW_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change value row payload"
            )
        result = cls(
            **{
                name: value.get(
                    name,
                    ()
                    if name
                    in {
                        "signal_provenance",
                        "metadata_refs",
                        "lineage_ids",
                        "embedding",
                    }
                    else (
                        False
                        if name
                        in {"semantic_authority", "compatibility_claim"}
                        else ""
                    ),
                )
                for name in cls.__dataclass_fields__
            }
        )
        claimed = str(value.get("row_id") or "")
        if claimed and claimed != result.row_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change value row identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueIndexTombstone:
    path: str
    name: str
    row_id: str
    blob_identity: str
    source_sha256: str
    ast_record_id: str
    reason: str
    replacement_row_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        for name in (
            "name",
            "row_id",
            "blob_identity",
            "source_sha256",
            "ast_record_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        reason = _text(self.reason, "reason")
        if reason not in _TOMBSTONE_REASONS:
            raise ChangeValueVectorIndexError(
                "unsupported change value tombstone reason"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "replacement_row_id",
            _text(
                self.replacement_row_id,
                "replacement_row_id",
                required=False,
            ),
        )

    @property
    def tombstone_id(self) -> str:
        return _identity(
            "change-value-vector-tombstone",
            self.to_dict(include_tombstone_id=False),
        )

    def to_dict(self, *, include_tombstone_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": CHANGE_VALUE_VECTOR_TOMBSTONE_SCHEMA,
            **asdict(self),
        }
        if include_tombstone_id:
            result["tombstone_id"] = self.tombstone_id
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ChangeValueIndexTombstone":
        _reject_bodies(value)
        allowed = {"schema", "tombstone_id", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_TOMBSTONE_SCHEMA)
            != CHANGE_VALUE_VECTOR_TOMBSTONE_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change-value tombstone payload"
            )
        result = cls(
            **{name: value.get(name, "") for name in cls.__dataclass_fields__}
        )
        claimed = str(value.get("tombstone_id") or "")
        if claimed and claimed != result.tombstone_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change-value tombstone identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueIndexSnapshot:
    """The complete immutable value/behavior vector index for one tree."""

    forest_id: str
    tree_id: str
    coverage_id: str
    coverage_complete: bool
    included_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    ast_index_id: str
    config: ChangeValueIndexConfig
    rows: tuple[ChangeValueIndexRow, ...]
    tombstones: tuple[ChangeValueIndexTombstone, ...] = ()
    lineage: tuple[ChangeValueLineage, ...] = ()
    graph_index_id: str = ""
    code_symbol_index_id: str = ""
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES

    def __post_init__(self) -> None:
        for name in ("forest_id", "tree_id", "coverage_id", "ast_index_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "graph_index_id",
            _text(self.graph_index_id, "graph_index_id", required=False),
        )
        object.__setattr__(
            self,
            "code_symbol_index_id",
            _text(
                self.code_symbol_index_id,
                "code_symbol_index_id",
                required=False,
            ),
        )
        if self.coverage_complete is not True:
            raise ChangeValueVectorIndexError(
                "incomplete coverage cannot produce a change value vector index"
            )
        if not 512 <= int(self.max_row_bytes) <= HARD_MAX_ROW_BYTES:
            raise ChangeValueVectorIndexBoundsError(
                "max_row_bytes is outside the hard bound"
            )
        object.__setattr__(self, "max_row_bytes", int(self.max_row_bytes))
        included = tuple(sorted({_path(item) for item in self.included_paths}))
        excluded = tuple(sorted({_path(item) for item in self.excluded_paths}))
        if set(included).intersection(excluded):
            raise ChangeValueVectorIndexError(
                "included and excluded paths overlap"
            )
        object.__setattr__(self, "included_paths", included)
        object.__setattr__(self, "excluded_paths", excluded)
        config = (
            self.config
            if isinstance(self.config, ChangeValueIndexConfig)
            else ChangeValueIndexConfig.from_dict(self.config)
        )
        object.__setattr__(self, "config", config)
        rows = tuple(self.rows)
        if len(rows) > DEFAULT_MAX_ROWS:
            raise ChangeValueVectorIndexBoundsError(
                "change value vector index exceeds its row bound"
            )
        if any(not isinstance(row, ChangeValueIndexRow) for row in rows):
            raise ChangeValueVectorIndexError(
                "snapshot rows must be ChangeValueIndexRow values"
            )
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (row.path, row.qualified_name, row.row_id),
            )
        )
        if len({row.row_id for row in rows}) != len(rows):
            raise ChangeValueVectorIndexIntegrityError(
                "duplicate change value rows"
            )
        for row in rows:
            if row.path not in included:
                raise ChangeValueVectorIndexIntegrityError(
                    "row path is absent from complete coverage"
                )
            _vector(row.embedding, config.dimensions, name="row embedding")
            if config.normalization == "l2" and not _is_l2_normalized(
                row.embedding
            ):
                raise ChangeValueVectorIndexIntegrityError(
                    "row embedding violates l2 normalization"
                )
            if (
                len(canonical_change_value_vector_index_bytes(row.to_dict()))
                > self.max_row_bytes
            ):
                raise ChangeValueVectorIndexBoundsError(
                    "change value row exceeds configured bound"
                )
            if row.semantic_authority is not False:
                raise ChangeValueVectorIndexIntegrityError(
                    "value index rows cannot claim semantic authority"
                )
            if row.compatibility_claim is not False:
                raise ChangeValueVectorIndexIntegrityError(
                    "same-typed or similar values receive no compatibility claim"
                )
        object.__setattr__(self, "rows", rows)
        tombstones = tuple(self.tombstones)
        if any(
            not isinstance(item, ChangeValueIndexTombstone)
            for item in tombstones
        ):
            raise ChangeValueVectorIndexError(
                "snapshot tombstones must be canonical values"
            )
        by_tombstone = {item.tombstone_id: item for item in tombstones}
        object.__setattr__(
            self,
            "tombstones",
            tuple(
                sorted(
                    by_tombstone.values(),
                    key=lambda item: item.tombstone_id,
                )
            ),
        )
        lineage = tuple(self.lineage)
        if any(not isinstance(item, ChangeValueLineage) for item in lineage):
            raise ChangeValueVectorIndexError(
                "snapshot lineage must be canonical values"
            )
        by_lineage = {item.lineage_id: item for item in lineage}
        lineage = tuple(
            sorted(by_lineage.values(), key=lambda item: item.lineage_id)
        )
        current_blobs = {
            (row.path, row.sidecar.blob_identity) for row in rows
        }
        for item in lineage:
            if (item.new_path, item.blob_identity) not in current_blobs:
                raise ChangeValueVectorIndexIntegrityError(
                    "lineage does not bind a current blob relocation"
                )
        known_lineage = {item.lineage_id for item in lineage}
        if any(
            set(row.lineage_ids).difference(known_lineage) for row in rows
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "row references unreviewed or forged lineage"
            )
        object.__setattr__(self, "lineage", lineage)

    @property
    def index_id(self) -> str:
        return _identity(
            "change-value-vector-index", self._content_dict()
        )

    @property
    def index_root_id(self) -> str:
        return self.index_id

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": CHANGE_VALUE_VECTOR_INDEX_SCHEMA,
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "coverage_id": self.coverage_id,
            "coverage_complete": True,
            "included_paths": list(self.included_paths),
            "excluded_paths": list(self.excluded_paths),
            "ast_index_id": self.ast_index_id,
            "graph_index_id": self.graph_index_id,
            "code_symbol_index_id": self.code_symbol_index_id,
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
            return canonical_change_value_vector_index_bytes(
                self.to_dict()
            ).decode("utf-8")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ChangeValueIndexSnapshot":
        _reject_bodies(value)
        allowed = {"index_id", "schema", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_INDEX_SCHEMA)
            != CHANGE_VALUE_VECTOR_INDEX_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change value vector snapshot payload"
            )
        result = cls(
            forest_id=value.get("forest_id", ""),
            tree_id=value.get("tree_id", ""),
            coverage_id=value.get("coverage_id", ""),
            coverage_complete=value.get("coverage_complete", False),
            included_paths=tuple(value.get("included_paths") or ()),
            excluded_paths=tuple(value.get("excluded_paths") or ()),
            ast_index_id=value.get("ast_index_id", ""),
            graph_index_id=value.get("graph_index_id", ""),
            code_symbol_index_id=value.get("code_symbol_index_id", ""),
            config=ChangeValueIndexConfig.from_dict(value.get("config") or {}),
            rows=tuple(
                ChangeValueIndexRow.from_dict(item)
                for item in value.get("rows") or ()
            ),
            tombstones=tuple(
                ChangeValueIndexTombstone.from_dict(item)
                for item in value.get("tombstones") or ()
            ),
            lineage=tuple(
                ChangeValueLineage.from_dict(item)
                for item in value.get("lineage") or ()
            ),
            max_row_bytes=value.get("max_row_bytes", DEFAULT_MAX_ROW_BYTES),
        )
        claimed = str(value.get("index_id") or "")
        if claimed and claimed != result.index_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change value vector snapshot identity mismatch"
            )
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes
    ) -> "ChangeValueIndexSnapshot":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ChangeValueVectorIndexIntegrityError(
                "change value vector snapshot JSON must be an object"
            )
        return cls.from_dict(payload)

    def search(
        self,
        query: "ChangeValueQuery | Sequence[float]",
        *,
        missing_requirement_id: str = "",
        missing_contract_refs: Iterable[str] = (),
        consumer_context_refs: Iterable[str] = (),
        consumer_path: str = "",
        obligation_id: str = "",
        max_results: int | None = None,
    ) -> "ChangeValueSearchResult":
        if not isinstance(query, ChangeValueQuery):
            query = ChangeValueQuery.for_snapshot(
                self,
                query_vector=query,
                missing_requirement_id=missing_requirement_id,
                missing_contract_refs=missing_contract_refs,
                consumer_context_refs=consumer_context_refs,
                consumer_path=consumer_path,
                obligation_id=obligation_id,
                max_results=max_results or DEFAULT_MAX_RESULTS,
            )
        return search_change_value_vector_index(self, query)

    query = search


@dataclass(frozen=True)
class ChangeValueQuery:
    """Exact snapshot-bound query with missing-contract and consumer context."""

    forest_id: str
    tree_id: str
    index_id: str
    config_id: str
    dimensions: int
    metric: str
    query_vector: tuple[float, ...]
    missing_requirement_id: str
    missing_contract_refs: tuple[str, ...] = ()
    consumer_context_refs: tuple[str, ...] = ()
    consumer_path: str = ""
    obligation_id: str = ""
    max_results: int = DEFAULT_MAX_RESULTS
    semantic_authority: bool = False
    compatibility_claim: bool = False

    def __post_init__(self) -> None:
        for name in ("forest_id", "tree_id", "index_id", "config_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "missing_requirement_id",
            _text(self.missing_requirement_id, "missing_requirement_id"),
        )
        if isinstance(self.dimensions, bool) or int(self.dimensions) < 1:
            raise ChangeValueVectorIndexError(
                "query dimensions must be positive"
            )
        object.__setattr__(self, "dimensions", int(self.dimensions))
        metric = _text(self.metric, "metric").casefold()
        if metric not in _METRICS:
            raise ChangeValueVectorIndexError("unsupported query metric")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(
            self,
            "query_vector",
            _vector(self.query_vector, self.dimensions, name="query vector"),
        )
        object.__setattr__(
            self,
            "missing_contract_refs",
            _references(self.missing_contract_refs, "missing_contract_refs"),
        )
        object.__setattr__(
            self,
            "consumer_context_refs",
            _references(self.consumer_context_refs, "consumer_context_refs"),
        )
        # Consumer context is required: either an explicit path/obligation or
        # compact consumer context references must bind the query.
        object.__setattr__(
            self,
            "consumer_path",
            _text(self.consumer_path, "consumer_path", required=False),
        )
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", required=False),
        )
        if (
            not self.consumer_context_refs
            and not self.consumer_path
            and not self.obligation_id
        ):
            raise ChangeValueVectorIndexError(
                "query requires consumer context (path, obligation, or refs)"
            )
        if not self.missing_contract_refs and not self.missing_requirement_id:
            raise ChangeValueVectorIndexError(
                "query requires exact missing contract binding"
            )
        if (
            isinstance(self.max_results, bool)
            or not 1 <= int(self.max_results) <= HARD_MAX_RESULTS
        ):
            raise ChangeValueVectorIndexBoundsError(
                "max_results is outside the hard bound"
            )
        object.__setattr__(self, "max_results", int(self.max_results))
        if self.semantic_authority is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "value vector queries cannot claim semantic authority"
            )
        if self.compatibility_claim is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "same-typed or similar values receive no compatibility claim"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "compatibility_claim", False)

    @property
    def query_id(self) -> str:
        return _identity(
            "change-value-vector-query",
            self.to_dict(include_query_id=False),
        )

    def to_dict(self, *, include_query_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CHANGE_VALUE_VECTOR_QUERY_SCHEMA,
            **asdict(self),
            "semantic_authority": False,
            "compatibility_claim": False,
        }
        result["query_vector"] = list(self.query_vector)
        result["missing_contract_refs"] = list(self.missing_contract_refs)
        result["consumer_context_refs"] = list(self.consumer_context_refs)
        if include_query_id:
            result["query_id"] = self.query_id
        return result

    @classmethod
    def for_snapshot(
        cls,
        snapshot: ChangeValueIndexSnapshot,
        *,
        query_vector: Sequence[float],
        missing_requirement_id: str,
        missing_contract_refs: Iterable[str] = (),
        consumer_context_refs: Iterable[str] = (),
        consumer_path: str = "",
        obligation_id: str = "",
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> "ChangeValueQuery":
        return cls(
            snapshot.forest_id,
            snapshot.tree_id,
            snapshot.index_id,
            snapshot.config.config_id,
            snapshot.config.dimensions,
            snapshot.config.metric,
            tuple(query_vector),
            missing_requirement_id,
            tuple(missing_contract_refs),
            tuple(consumer_context_refs),
            consumer_path,
            obligation_id,
            max_results,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangeValueQuery":
        _reject_bodies(value)
        allowed = {"schema", "query_id", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_QUERY_SCHEMA)
            != CHANGE_VALUE_VECTOR_QUERY_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change value vector query payload"
            )
        result = cls(
            **{
                name: value.get(
                    name,
                    ()
                    if name
                    in {
                        "query_vector",
                        "missing_contract_refs",
                        "consumer_context_refs",
                    }
                    else (
                        False
                        if name
                        in {"semantic_authority", "compatibility_claim"}
                        else (
                            DEFAULT_MAX_RESULTS
                            if name == "max_results"
                            else ""
                        )
                    ),
                )
                for name in cls.__dataclass_fields__
            }
        )
        claimed = str(value.get("query_id") or "")
        if claimed and claimed != result.query_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change value vector query identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueHit:
    """One non-authoritative nomination hit with signal provenance."""

    row: ChangeValueIndexRow
    index_id: str
    query_id: str
    score: float
    rank: int
    signal_provenance: tuple[str, ...] = ()
    semantic_authority: bool = False
    compatibility_claim: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.row, ChangeValueIndexRow):
            if not isinstance(self.row, Mapping):
                raise ChangeValueVectorIndexError("hit row must be canonical")
            object.__setattr__(
                self, "row", ChangeValueIndexRow.from_dict(self.row)
            )
        object.__setattr__(self, "index_id", _text(self.index_id, "index_id"))
        object.__setattr__(self, "query_id", _text(self.query_id, "query_id"))
        if not math.isfinite(float(self.score)):
            raise ChangeValueVectorIndexError(
                "value vector hit score must be finite"
            )
        object.__setattr__(self, "score", float(self.score))
        if isinstance(self.rank, bool) or int(self.rank) < 1:
            raise ChangeValueVectorIndexError("hit rank must be positive")
        object.__setattr__(self, "rank", int(self.rank))
        signals = _references(self.signal_provenance, "signal_provenance")
        if not signals:
            # Hits always retain at least the vector signal that produced them,
            # plus any row-level provenance already bound to the candidate.
            signals = tuple(
                sorted(
                    {
                        ChangeValueSignal.VECTOR.value,
                        *self.row.signal_provenance,
                    }
                )
            )
        for signal in signals:
            try:
                ChangeValueSignal(signal)
            except ValueError as exc:
                raise ChangeValueVectorIndexError(
                    f"unsupported hit signal provenance: {signal}"
                ) from exc
        object.__setattr__(self, "signal_provenance", signals)
        if self.semantic_authority is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "value vector hits cannot claim semantic authority"
            )
        if self.compatibility_claim is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "same-typed or similar values receive no compatibility claim"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "compatibility_claim", False)

    @property
    def hit_id(self) -> str:
        return _identity(
            "change-value-vector-hit",
            self.to_dict(include_hit_id=False),
        )

    @property
    def row_id(self) -> str:
        return self.row.row_id

    def to_dict(self, *, include_hit_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CHANGE_VALUE_VECTOR_HIT_SCHEMA,
            "row": self.row.to_dict(),
            "index_id": self.index_id,
            "query_id": self.query_id,
            "score": self.score,
            "rank": self.rank,
            "signal_provenance": list(self.signal_provenance),
            "semantic_authority": False,
            "compatibility_claim": False,
        }
        if include_hit_id:
            result["hit_id"] = self.hit_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangeValueHit":
        _reject_bodies(value)
        allowed = {"schema", "hit_id", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_HIT_SCHEMA)
            != CHANGE_VALUE_VECTOR_HIT_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change value vector hit payload"
            )
        result = cls(
            **{
                name: value.get(
                    name,
                    ()
                    if name in {"signal_provenance"}
                    else (
                        False
                        if name
                        in {"semantic_authority", "compatibility_claim"}
                        else ""
                    ),
                )
                for name in cls.__dataclass_fields__
            }
        )
        claimed = str(value.get("hit_id") or "")
        if claimed and claimed != result.hit_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change value vector hit identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ChangeValueSearchResult:
    query: ChangeValueQuery
    index_id: str
    hits: tuple[ChangeValueHit, ...]
    complete: bool = True
    searched_row_count: int = 0
    semantic_authority: bool = False
    compatibility_claim: bool = False

    def __post_init__(self) -> None:
        query = (
            self.query
            if isinstance(self.query, ChangeValueQuery)
            else ChangeValueQuery.from_dict(self.query)
        )
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "index_id", _text(self.index_id, "index_id"))
        hits = tuple(self.hits)
        if any(not isinstance(hit, ChangeValueHit) for hit in hits):
            raise ChangeValueVectorIndexError(
                "search result hits must be canonical"
            )
        if self.complete is not True:
            raise ChangeValueVectorIndexIntegrityError(
                "incomplete value vector results are not admissible"
            )
        if self.searched_row_count < 0:
            raise ChangeValueVectorIndexError(
                "searched_row_count must not be negative"
            )
        if self.semantic_authority is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "value vector results cannot claim semantic authority"
            )
        if self.compatibility_claim is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "same-typed or similar values receive no compatibility claim"
            )
        if query.index_id != self.index_id:
            raise ChangeValueVectorIndexIntegrityError(
                "search result query is bound to a different index"
            )
        if any(
            hit.index_id != self.index_id or hit.query_id != query.query_id
            for hit in hits
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "search result mixes stale or forged hits"
            )
        if tuple(hit.rank for hit in hits) != tuple(
            range(1, len(hits) + 1)
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "search result ranks are not complete and deterministic"
            )
        if int(self.searched_row_count) < len(hits):
            raise ChangeValueVectorIndexIntegrityError(
                "search result cannot be complete when fewer rows were "
                "searched than returned"
            )
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "compatibility_claim", False)

    @property
    def result_id(self) -> str:
        return _identity(
            "change-value-vector-result",
            self.to_dict(include_result_id=False),
        )

    @property
    def results(self) -> tuple[ChangeValueHit, ...]:
        return self.hits

    def to_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CHANGE_VALUE_VECTOR_RESULT_SCHEMA,
            "query": self.query.to_dict(),
            "index_id": self.index_id,
            "hits": [item.to_dict() for item in self.hits],
            "complete": True,
            "searched_row_count": self.searched_row_count,
            "semantic_authority": False,
            "compatibility_claim": False,
        }
        if include_result_id:
            result["result_id"] = self.result_id
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ChangeValueSearchResult":
        _reject_bodies(value)
        allowed = {"schema", "result_id", *cls.__dataclass_fields__}
        if (
            set(value).difference(allowed)
            or value.get("schema", CHANGE_VALUE_VECTOR_RESULT_SCHEMA)
            != CHANGE_VALUE_VECTOR_RESULT_SCHEMA
        ):
            raise ChangeValueVectorIndexIntegrityError(
                "unsupported change value vector result payload"
            )
        result = cls(
            query=ChangeValueQuery.from_dict(value.get("query") or {}),
            index_id=value.get("index_id", ""),
            hits=tuple(
                ChangeValueHit.from_dict(item)
                for item in value.get("hits") or ()
            ),
            complete=value.get("complete", False),
            searched_row_count=int(value.get("searched_row_count", 0)),
            semantic_authority=value.get("semantic_authority", False),
            compatibility_claim=value.get("compatibility_claim", False),
        )
        claimed = str(value.get("result_id") or "")
        if claimed and claimed != result.result_id:
            raise ChangeValueVectorIndexIntegrityError(
                "change value vector result identity mismatch"
            )
        return result


class ChangeValueVectorSearchProvider(Protocol):
    """Optional backend protocol.  It may nominate only complete local rows."""

    def embed(self, row: ChangeValueIndexRow) -> Sequence[float]: ...


def _coerce_ast_index(value: Any) -> tuple[AnalysisASTIndex, Any | None]:
    if isinstance(value, AnalysisASTIndex):
        return value, None
    ast = getattr(value, "ast_index", None)
    if isinstance(ast, AnalysisASTIndex):
        return ast, value
    if isinstance(value, Mapping):
        if "ast_index" in value:
            ast_value = value["ast_index"]
            return (
                (
                    ast_value
                    if isinstance(ast_value, AnalysisASTIndex)
                    else AnalysisASTIndex.from_dict(ast_value)
                ),
                value,
            )
        return AnalysisASTIndex.from_dict(value), None
    raise ChangeValueVectorIndexError(
        "an AnalysisASTIndex or RepositoryIndex is required"
    )


def _feature_refs(
    features: Mapping[str, Any] | None, keys: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    if not features:
        return {}
    _reject_bodies(features)
    result: dict[str, tuple[str, ...]] = {}
    for key in keys:
        raw = features.get(key)
        if raw is not None:
            result[key] = _references(raw, f"feature refs for {key}")
    return result


def _sidecar(
    indexed: IndexedASTPath,
    symbol: str,
    features: Mapping[str, Any] | None,
    *,
    graph_node_refs: Iterable[str] = (),
) -> ChangeValueSidecarRef:
    record = indexed.ast_record
    signatures = tuple(
        item
        for item in record.interfaces
        if item.startswith(symbol + ":") or item.startswith(symbol + "(")
    )
    calls = tuple(
        item for item in record.calls if item.startswith(symbol + "->")
    )
    effects = tuple(
        item
        for item in record.state_transitions
        if item.startswith(symbol + ":")
    )
    supplied = _feature_refs(
        features,
        (
            "graph_node_refs",
            "type_refs",
            "constructor_refs",
            "factory_refs",
            "schema_refs",
            "definition_use_refs",
            "scope_refs",
            "effect_refs",
            "ownership_refs",
            "documentation_refs",
            "test_refs",
            "history_refs",
        ),
    )
    return ChangeValueSidecarRef(
        ast_record_id=indexed.record_id,
        blob_identity=indexed.blob_identity,
        source_sha256=indexed.source_sha256,
        symbol_hash=record.symbol_hashes.get(symbol, ""),
        graph_node_refs=tuple(
            {
                *graph_node_refs,
                *supplied.get("graph_node_refs", ()),
            }
        ),
        type_refs=tuple(
            {
                *signatures,
                *supplied.get("type_refs", ()),
            }
        ),
        constructor_refs=supplied.get("constructor_refs", ()),
        factory_refs=supplied.get("factory_refs", ()),
        schema_refs=supplied.get("schema_refs", ()),
        definition_use_refs=tuple(
            {
                *calls,
                *supplied.get("definition_use_refs", ()),
            }
        ),
        scope_refs=supplied.get("scope_refs", ()),
        effect_refs=tuple(
            {
                *effects,
                *supplied.get("effect_refs", ()),
            }
        ),
        ownership_refs=supplied.get("ownership_refs", ()),
        documentation_refs=supplied.get("documentation_refs", ()),
        test_refs=supplied.get("test_refs", ()),
        history_refs=supplied.get("history_refs", ()),
    )


def _default_signals(kind: ChangeValueKind) -> tuple[str, ...]:
    base = {ChangeValueSignal.AST.value, ChangeValueSignal.VECTOR.value}
    if kind is ChangeValueKind.SCHEMA:
        base.add(ChangeValueSignal.SCHEMA.value)
    if kind is ChangeValueKind.TEST or kind is ChangeValueKind.FIXTURE:
        base.add(ChangeValueSignal.TEST.value)
    if kind is ChangeValueKind.DOCUMENTATION:
        base.add(ChangeValueSignal.DOCUMENTATION.value)
    if kind is ChangeValueKind.HISTORY:
        base.add(ChangeValueSignal.HISTORY.value)
    if kind in {
        ChangeValueKind.FACTORY,
        ChangeValueKind.CONSTRUCTOR,
        ChangeValueKind.BUILDER,
        ChangeValueKind.DI_BINDING,
    }:
        base.add(ChangeValueSignal.GRAPH.value)
    return tuple(sorted(base))


def _lookup_vector(
    vectors: Mapping[str, Sequence[float]]
    | Callable[[ChangeValueIndexRow], Sequence[float]]
    | ChangeValueVectorSearchProvider,
    row: ChangeValueIndexRow,
) -> Sequence[float]:
    if callable(vectors):
        return vectors(row)
    embed = getattr(vectors, "embed", None)
    if callable(embed):
        return embed(row)
    if not isinstance(vectors, Mapping):
        raise ChangeValueVectorIndexError(
            "vectors must be a mapping or admitted local embedding provider"
        )
    for key in (
        row.qualified_name,
        f"{row.path}:{row.name}",
        row.name,
        row.row_id,
        row.kind.value if isinstance(row.kind, ChangeValueKind) else str(row.kind),
    ):
        if key in vectors:
            return vectors[key]
    raise ChangeValueVectorIndexError(
        f"missing deterministic vector for {row.path}:{row.name}"
    )


def _coerce_lineage(values: Any) -> tuple[ChangeValueLineage, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, Mapping):
        values = (values,)
    return tuple(
        item
        if isinstance(item, ChangeValueLineage)
        else ChangeValueLineage.from_dict(item)
        for item in values
    )


def _prior_tombstones(
    previous: ChangeValueIndexSnapshot | None,
    current_rows: Sequence[ChangeValueIndexRow],
) -> tuple[ChangeValueIndexTombstone, ...]:
    if previous is None:
        return ()
    current_by_key = {
        (item.path, item.name): item for item in current_rows
    }
    additions: list[ChangeValueIndexTombstone] = []
    for old in previous.rows:
        replacement = current_by_key.get((old.path, old.name))
        if replacement is not None and replacement.row_id == old.row_id:
            continue
        reason = "blob_changed" if replacement is not None else "path_deleted"
        additions.append(
            ChangeValueIndexTombstone(
                path=old.path,
                name=old.name,
                row_id=old.row_id,
                blob_identity=old.sidecar.blob_identity,
                source_sha256=old.sidecar.source_sha256,
                ast_record_id=old.sidecar.ast_record_id,
                reason=reason,
                replacement_row_id=(
                    replacement.row_id if replacement is not None else ""
                ),
            )
        )
    return tuple((*previous.tombstones, *additions))


def _row_from_code_symbol(
    row: CodeSymbolIndexRow,
    *,
    kind: ChangeValueKind | None = None,
    vectors: Mapping[str, Sequence[float]]
    | Callable[[ChangeValueIndexRow], Sequence[float]]
    | ChangeValueVectorSearchProvider
    | None = None,
) -> ChangeValueIndexRow:
    inferred = kind or _infer_kind(row.path, row.symbol)
    prototype = ChangeValueIndexRow(
        path=row.path,
        name=row.symbol,
        qualified_name=row.qualified_symbol,
        kind=inferred,
        line_start=row.line_start,
        line_end=row.line_end,
        sidecar=ChangeValueSidecarRef(
            ast_record_id=row.sidecar.ast_record_id,
            blob_identity=row.sidecar.blob_identity,
            source_sha256=row.sidecar.source_sha256,
            symbol_hash=row.sidecar.symbol_hash,
            type_refs=row.sidecar.signature_refs,
            definition_use_refs=row.sidecar.call_refs,
            effect_refs=row.sidecar.effect_refs,
            ownership_refs=row.sidecar.ownership_refs,
            documentation_refs=row.sidecar.documentation_refs,
            test_refs=row.sidecar.test_refs,
        ),
        embedding=row.embedding if vectors is None else (0.0,),
        type_ref=row.sidecar.signature_refs[0]
        if row.sidecar.signature_refs
        else "",
        scope_ref=row.path,
        signal_provenance=_default_signals(inferred),
        metadata_refs=row.metadata_refs,
        lineage_ids=(),
    )
    if vectors is not None:
        vector = _vector(
            _lookup_vector(vectors, prototype),
            len(row.embedding) if row.embedding else len(_lookup_vector(vectors, prototype)),
            name="row embedding",
        )
        return ChangeValueIndexRow(
            path=prototype.path,
            name=prototype.name,
            qualified_name=prototype.qualified_name,
            kind=prototype.kind,
            line_start=prototype.line_start,
            line_end=prototype.line_end,
            sidecar=prototype.sidecar,
            embedding=vector,
            type_ref=prototype.type_ref,
            scope_ref=prototype.scope_ref,
            signal_provenance=prototype.signal_provenance,
            metadata_refs=prototype.metadata_refs,
            lineage_ids=prototype.lineage_ids,
        )
    return prototype


def build_change_value_vector_index(
    ast_index: AnalysisASTIndex | Any | None = None,
    *,
    forest_id: str = "",
    tree_id: str = "",
    coverage_id: str = "",
    included_paths: Iterable[str] | None = None,
    excluded_paths: Iterable[str] = (),
    coverage_complete: bool = True,
    producer_id: str = "change-value-vector-indexer@1",
    chunker_id: str = "value-behavior@1",
    normalization: str = "l2",
    model_id: str = "deterministic-fixture",
    model_revision: str = "1",
    dimensions: int | None = None,
    metric: str = "cosine",
    configuration_id: str = "change-value-vector-default@1",
    vectors: Mapping[str, Sequence[float]]
    | Callable[[ChangeValueIndexRow], Sequence[float]]
    | ChangeValueVectorSearchProvider
    | None = None,
    feature_references: Mapping[str, Mapping[str, Any]] | None = None,
    metadata_references: Mapping[str, Iterable[str]] | None = None,
    graph_node_references: Mapping[str, Iterable[str]] | None = None,
    explicit_rows: Iterable[ChangeValueIndexRow | Mapping[str, Any]] = (),
    code_symbol_index: CodeVectorIndexSnapshot | Mapping[str, Any] | None = None,
    graph_index_id: str = "",
    reviewed_lineage: Iterable[ChangeValueLineage | Mapping[str, Any]] = (),
    tombstones: Iterable[ChangeValueIndexTombstone | Mapping[str, Any]]
    | None = None,
    previous: ChangeValueIndexSnapshot | Mapping[str, Any] | None = None,
    previous_index: ChangeValueIndexSnapshot | Mapping[str, Any] | None = None,
    exhaustive: bool = False,
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES,
    **aliases: Any,
) -> ChangeValueIndexSnapshot:
    """Build one exact, complete value/behavior nomination snapshot.

    ``vectors`` is required unless every row already carries an embedding via
    ``explicit_rows`` or ``code_symbol_index``.  No implicit remote embedding
    backend is admitted.  Same-typed or similar candidates never receive a
    compatibility claim.
    """
    if previous is not None and previous_index is not None:
        raise ChangeValueVectorIndexError(
            "provide only one of previous or previous_index"
        )
    if previous is None:
        previous = previous_index
    if previous is not None and not isinstance(
        previous, ChangeValueIndexSnapshot
    ):
        previous = ChangeValueIndexSnapshot.from_dict(previous)
    if code_symbol_index is not None and not isinstance(
        code_symbol_index, CodeVectorIndexSnapshot
    ):
        code_symbol_index = CodeVectorIndexSnapshot.from_dict(
            code_symbol_index
        )
    if aliases:
        alias_names = {
            "repository_forest_id": "forest_id",
            "repository_tree_id": "tree_id",
            "model": "model_id",
            "model_config_id": "configuration_id",
            "embedding_vectors": "vectors",
            "embeddings": "vectors",
            "embedding_provider": "vectors",
            "vector_provider": "vectors",
            "symbol_index": "code_symbol_index",
            "program_graph_id": "graph_index_id",
        }
        unknown = set(aliases).difference(alias_names)
        if unknown:
            raise ChangeValueVectorIndexError(
                "unknown change value vector index options: "
                + ", ".join(sorted(unknown))
            )
        for source, target in alias_names.items():
            if source not in aliases:
                continue
            if target == "forest_id" and not forest_id:
                forest_id = aliases[source]
            elif target == "tree_id" and not tree_id:
                tree_id = aliases[source]
            elif target == "model_id" and model_id == "deterministic-fixture":
                model_id = aliases[source]
            elif (
                target == "configuration_id"
                and configuration_id == "change-value-vector-default@1"
            ):
                configuration_id = aliases[source]
            elif target == "vectors" and vectors is None:
                vectors = aliases[source]
            elif target == "code_symbol_index" and code_symbol_index is None:
                raw = aliases[source]
                code_symbol_index = (
                    raw
                    if isinstance(raw, CodeVectorIndexSnapshot)
                    else CodeVectorIndexSnapshot.from_dict(raw)
                )
            elif target == "graph_index_id" and not graph_index_id:
                graph_index_id = aliases[source]

    index: AnalysisASTIndex | None = None
    repository: Any | None = None
    if ast_index is not None:
        index, repository = _coerce_ast_index(ast_index)
    if repository is not None:
        snapshot = getattr(repository, "snapshot", None)
        if snapshot is not None:
            forest_id = forest_id or getattr(snapshot, "snapshot_id", "")
            tree_id = tree_id or getattr(snapshot, "head_tree_id", "")
        coverage_id = coverage_id or getattr(repository, "index_id", "")
        if included_paths is None and index is not None:
            included_paths = index.paths
        if not excluded_paths and repository is not None:
            path_rows = getattr(repository, "path_rows", ()) or ()
            repo_paths = {row.path for row in path_rows}
            if index is not None:
                excluded_paths = tuple(
                    sorted(repo_paths.difference(index.paths))
                )
        coverage_complete = coverage_complete and bool(
            getattr(repository, "safe_for_completion_reasoning", False)
        )
    if not forest_id or not tree_id:
        raise ChangeValueVectorIndexError(
            "forest_id and tree_id are required; AST evidence alone cannot "
            "invent repository roots"
        )
    if not coverage_id:
        if index is not None:
            coverage_id = index.index_id
        elif code_symbol_index is not None:
            coverage_id = code_symbol_index.coverage_id
        else:
            raise ChangeValueVectorIndexError("coverage_id is required")

    prototypes: list[ChangeValueIndexRow] = []
    for item in explicit_rows:
        row = (
            item
            if isinstance(item, ChangeValueIndexRow)
            else ChangeValueIndexRow.from_dict(item)
        )
        prototypes.append(row)

    if code_symbol_index is not None:
        for symbol_row in code_symbol_index.rows:
            prototypes.append(
                _row_from_code_symbol(symbol_row, vectors=None)
            )
        if included_paths is None:
            included_paths = code_symbol_index.included_paths
        if not graph_index_id:
            graph_index_id = graph_index_id
        if not forest_id:
            forest_id = code_symbol_index.forest_id
        if not tree_id:
            tree_id = code_symbol_index.tree_id

    if index is not None:
        if included_paths is None:
            included_paths = index.paths
        for indexed in index.path_records:
            for symbol in indexed.ast_record.qualified_symbols:
                line_start, line_end = indexed.ast_record.symbol_lines.get(
                    symbol, (1, 1)
                )
                if line_start < 1:
                    line_start = 1
                if line_end < line_start:
                    line_end = line_start
                qualified = (
                    f"{_module_for_path(indexed.path)}.{symbol}".strip(".")
                )
                features = (
                    (feature_references or {}).get(qualified)
                    or (feature_references or {}).get(
                        f"{indexed.path}:{symbol}"
                    )
                    or (feature_references or {}).get(symbol)
                )
                metadata = (
                    (metadata_references or {}).get(qualified)
                    or (metadata_references or {}).get(
                        f"{indexed.path}:{symbol}"
                    )
                    or ()
                )
                graph_refs = (
                    (graph_node_references or {}).get(qualified)
                    or (graph_node_references or {}).get(
                        f"{indexed.path}:{symbol}"
                    )
                    or ()
                )
                kind = _infer_kind(indexed.path, symbol)
                prototypes.append(
                    ChangeValueIndexRow(
                        path=indexed.path,
                        name=symbol,
                        qualified_name=qualified,
                        kind=kind,
                        line_start=line_start,
                        line_end=line_end,
                        sidecar=_sidecar(
                            indexed,
                            symbol,
                            features,
                            graph_node_refs=graph_refs,
                        ),
                        embedding=(0.0,),
                        type_ref="",
                        scope_ref=indexed.path,
                        signal_provenance=_default_signals(kind),
                        metadata_refs=tuple(metadata),
                    )
                )

    if not prototypes:
        raise ChangeValueVectorIndexError(
            "complete change value vector index has no value/behavior rows"
        )
    if included_paths is None:
        included_paths = tuple(sorted({row.path for row in prototypes}))
    paths = tuple(sorted({_path(item) for item in included_paths}))
    if index is not None and set(index.paths).difference(paths):
        raise ChangeValueVectorIndexError(
            "complete coverage omits AST-indexed paths"
        )

    # Deduplicate by (path, name) keeping the first complete prototype.
    deduped: dict[tuple[str, str], ChangeValueIndexRow] = {}
    for prototype in prototypes:
        key = (prototype.path, prototype.name)
        if key not in deduped:
            deduped[key] = prototype
    prototypes = list(deduped.values())

    needs_vectors = any(
        len(row.embedding) == 1 and row.embedding == (0.0,)
        for row in prototypes
    ) or any(not row.embedding for row in prototypes)
    if needs_vectors and vectors is None:
        # Rows carried from a code-symbol index already have embeddings.
        if all(
            len(row.embedding) > 1
            or (len(row.embedding) == 1 and row.embedding != (0.0,))
            for row in prototypes
        ):
            needs_vectors = False
        else:
            raise ChangeValueVectorIndexError(
                "vectors are required; no implicit embedding backend is admitted"
            )

    if dimensions is None:
        sample = next(
            (
                row.embedding
                for row in prototypes
                if len(row.embedding) > 1
                or (len(row.embedding) == 1 and row.embedding != (0.0,))
            ),
            None,
        )
        if sample is not None:
            dimensions = len(sample)
        elif vectors is not None:
            first = _lookup_vector(vectors, prototypes[0])
            try:
                dimensions = len(first)
            except TypeError as exc:
                raise ChangeValueVectorIndexError(
                    "vector dimensions are unavailable"
                ) from exc
        else:
            raise ChangeValueVectorIndexError(
                "vector dimensions are unavailable"
            )

    config = ChangeValueIndexConfig(
        producer_id,
        chunker_id,
        normalization,
        model_id,
        model_revision,
        dimensions,
        metric,
        configuration_id,
    )
    lineage = _coerce_lineage(reviewed_lineage)
    rows: list[ChangeValueIndexRow] = []
    for prototype in prototypes:
        if (
            len(prototype.embedding) == config.dimensions
            and not (
                len(prototype.embedding) == 1
                and prototype.embedding == (0.0,)
            )
        ):
            vector = _vector(
                prototype.embedding,
                config.dimensions,
                name="row embedding",
            )
        else:
            if vectors is None:
                raise ChangeValueVectorIndexError(
                    f"missing deterministic vector for "
                    f"{prototype.path}:{prototype.name}"
                )
            vector = _vector(
                _lookup_vector(vectors, prototype),
                config.dimensions,
                name="row embedding",
            )
        if config.normalization == "l2" and not _is_l2_normalized(vector):
            raise ChangeValueVectorIndexError(
                "row embedding violates configured l2 normalization"
            )
        row_lineage = tuple(
            item.lineage_id
            for item in lineage
            if item.new_path == prototype.path
            and item.blob_identity == prototype.sidecar.blob_identity
            and (not item.new_value or item.new_value == prototype.name)
        )
        # Preserve any reviewed lineage already on an explicit row.
        combined_lineage = tuple(
            sorted({*prototype.lineage_ids, *row_lineage})
        )
        rows.append(
            ChangeValueIndexRow(
                path=prototype.path,
                name=prototype.name,
                qualified_name=prototype.qualified_name,
                kind=prototype.kind,
                line_start=prototype.line_start,
                line_end=prototype.line_end,
                sidecar=prototype.sidecar,
                embedding=vector,
                type_ref=prototype.type_ref,
                scope_ref=prototype.scope_ref,
                signal_provenance=prototype.signal_provenance
                or _default_signals(prototype.kind),
                metadata_refs=prototype.metadata_refs,
                lineage_ids=combined_lineage,
            )
        )

    if lineage:
        if previous is None:
            raise ChangeValueVectorIndexError(
                "reviewed lineage requires the previous exact snapshot"
            )
        old_blobs = {
            (row.path, row.sidecar.blob_identity) for row in previous.rows
        }
        new_blobs = {
            (row.path, row.sidecar.blob_identity) for row in rows
        }
        for item in lineage:
            if (
                (item.old_path, item.blob_identity) not in old_blobs
                or (item.new_path, item.blob_identity) not in new_blobs
            ):
                raise ChangeValueVectorIndexIntegrityError(
                    "reviewed lineage does not preserve the exact moved blob"
                )

    if tombstones is None:
        effective_tombstones = (
            () if exhaustive else _prior_tombstones(previous, rows)
        )
    else:
        effective_tombstones = tuple(
            item
            if isinstance(item, ChangeValueIndexTombstone)
            else ChangeValueIndexTombstone.from_dict(item)
            for item in tombstones
        )

    symbol_index_id = ""
    if code_symbol_index is not None:
        symbol_index_id = code_symbol_index.index_id
        if code_symbol_index.forest_id != forest_id or (
            code_symbol_index.tree_id != tree_id
        ):
            raise ChangeValueVectorIndexStaleError(
                "code symbol index roots do not match the value index snapshot"
            )

    return ChangeValueIndexSnapshot(
        forest_id=forest_id,
        tree_id=tree_id,
        coverage_id=coverage_id,
        coverage_complete=coverage_complete,
        included_paths=paths,
        excluded_paths=tuple(excluded_paths),
        ast_index_id=index.index_id if index is not None else (
            code_symbol_index.ast_index_id
            if code_symbol_index is not None
            else "ast-index:none"
        ),
        graph_index_id=graph_index_id,
        code_symbol_index_id=symbol_index_id,
        config=config,
        rows=tuple(rows),
        tombstones=effective_tombstones,
        lineage=lineage,
        max_row_bytes=max_row_bytes,
    )


def _score(
    metric: str, query: Sequence[float], vector: Sequence[float]
) -> float:
    dot = sum(left * right for left, right in zip(query, vector))
    if metric == "cosine":
        if not _is_l2_normalized(query):
            raise ChangeValueVectorIndexError(
                "cosine query violates l2 normalization"
            )
    return dot


def search_change_value_vector_index(
    snapshot: ChangeValueIndexSnapshot | Mapping[str, Any],
    query: ChangeValueQuery | Mapping[str, Any],
    *,
    max_results: int | None = None,
) -> ChangeValueSearchResult:
    if not isinstance(snapshot, ChangeValueIndexSnapshot):
        snapshot = ChangeValueIndexSnapshot.from_dict(snapshot)
    if not isinstance(query, ChangeValueQuery):
        if isinstance(query, Mapping):
            query = ChangeValueQuery.from_dict(query)
        else:
            raise ChangeValueVectorIndexError(
                "change value query must bind missing contract and consumer context"
            )
    if max_results is not None and max_results != query.max_results:
        query = ChangeValueQuery(
            query.forest_id,
            query.tree_id,
            query.index_id,
            query.config_id,
            query.dimensions,
            query.metric,
            query.query_vector,
            query.missing_requirement_id,
            query.missing_contract_refs,
            query.consumer_context_refs,
            query.consumer_path,
            query.obligation_id,
            max_results,
        )
    if (
        query.forest_id,
        query.tree_id,
        query.index_id,
        query.config_id,
        query.dimensions,
        query.metric,
    ) != (
        snapshot.forest_id,
        snapshot.tree_id,
        snapshot.index_id,
        snapshot.config.config_id,
        snapshot.config.dimensions,
        snapshot.config.metric,
    ):
        raise ChangeValueVectorIndexStaleError(
            "change value query roots/configuration do not match the current snapshot"
        )
    if snapshot.config.normalization == "l2" and not _is_l2_normalized(
        query.query_vector
    ):
        raise ChangeValueVectorIndexError(
            "query vector violates configured l2 normalization"
        )
    ranked = sorted(
        (
            (
                _score(
                    snapshot.config.metric,
                    query.query_vector,
                    row.embedding,
                ),
                row,
            )
            for row in snapshot.rows
        ),
        key=lambda item: (
            -item[0],
            item[1].path,
            item[1].qualified_name,
            item[1].row_id,
        ),
    )
    hits = tuple(
        ChangeValueHit(
            row,
            snapshot.index_id,
            query.query_id,
            score,
            rank + 1,
            signal_provenance=tuple(
                sorted(
                    {
                        ChangeValueSignal.VECTOR.value,
                        *row.signal_provenance,
                    }
                )
            ),
        )
        for rank, (score, row) in enumerate(ranked[: query.max_results])
    )
    result = ChangeValueSearchResult(
        query,
        snapshot.index_id,
        hits,
        complete=True,
        searched_row_count=len(snapshot.rows),
    )
    return validate_change_value_search_result(snapshot, result)


def validate_change_value_search_result(
    snapshot: ChangeValueIndexSnapshot | Mapping[str, Any],
    result: ChangeValueSearchResult | Mapping[str, Any],
) -> ChangeValueSearchResult:
    """Fail closed unless a response covers the exact current local row set."""

    if not isinstance(snapshot, ChangeValueIndexSnapshot):
        snapshot = ChangeValueIndexSnapshot.from_dict(snapshot)
    if not isinstance(result, ChangeValueSearchResult):
        result = ChangeValueSearchResult.from_dict(result)
    query = result.query
    if (
        query.forest_id,
        query.tree_id,
        query.index_id,
        query.config_id,
        query.dimensions,
        query.metric,
    ) != (
        snapshot.forest_id,
        snapshot.tree_id,
        snapshot.index_id,
        snapshot.config.config_id,
        snapshot.config.dimensions,
        snapshot.config.metric,
    ):
        raise ChangeValueVectorIndexStaleError(
            "change value result roots/configuration do not match the current snapshot"
        )
    if (
        result.index_id != snapshot.index_id
        or result.complete is not True
        or result.searched_row_count != len(snapshot.rows)
    ):
        raise ChangeValueVectorIndexIntegrityError(
            "incomplete or stale change value vector result"
        )
    if result.semantic_authority is not False:
        raise ChangeValueVectorIndexIntegrityError(
            "value vector results cannot claim semantic authority"
        )
    if result.compatibility_claim is not False:
        raise ChangeValueVectorIndexIntegrityError(
            "same-typed or similar values receive no compatibility claim"
        )
    current = {row.row_id for row in snapshot.rows}
    if len({hit.row_id for hit in result.hits}) != len(result.hits) or any(
        hit.row_id not in current for hit in result.hits
    ):
        raise ChangeValueVectorIndexIntegrityError(
            "change value result contains a row absent from the exact snapshot"
        )
    for hit in result.hits:
        if hit.semantic_authority is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "value vector hits cannot claim semantic authority"
            )
        if hit.compatibility_claim is not False:
            raise ChangeValueVectorIndexIntegrityError(
                "same-typed or similar values receive no compatibility claim"
            )
        if not hit.signal_provenance:
            raise ChangeValueVectorIndexIntegrityError(
                "value vector hits must retain signal provenance"
            )
    return result


# Readable aliases for adjacent retrieval / propagation adapters.
build_change_value_index = build_change_value_vector_index
search_change_value_index = search_change_value_vector_index
ChangeValueVectorIndex = ChangeValueIndexSnapshot
ChangeValueVectorIndexSnapshot = ChangeValueIndexSnapshot
ChangeValueVectorRow = ChangeValueIndexRow
ChangeValueTombstone = ChangeValueIndexTombstone
ChangeValueVectorQuery = ChangeValueQuery
ChangeValueVectorHit = ChangeValueHit
ChangeValueVectorSearchResult = ChangeValueSearchResult


__all__ = [
    "CHANGE_VALUE_VECTOR_INDEX_SCHEMA",
    "CHANGE_VALUE_VECTOR_ROW_SCHEMA",
    "CHANGE_VALUE_VECTOR_QUERY_SCHEMA",
    "CHANGE_VALUE_VECTOR_HIT_SCHEMA",
    "CHANGE_VALUE_VECTOR_RESULT_SCHEMA",
    "CHANGE_VALUE_VECTOR_TOMBSTONE_SCHEMA",
    "CHANGE_VALUE_VECTOR_LINEAGE_SCHEMA",
    "CHANGE_VALUE_VECTOR_CONFIG_SCHEMA",
    "CHANGE_VALUE_SIDECAR_SCHEMA",
    "ChangeValueVectorIndexError",
    "ChangeValueVectorIndexIntegrityError",
    "ChangeValueVectorIndexStaleError",
    "ChangeValueVectorIndexBoundsError",
    "ChangeValueKind",
    "ChangeValueSignal",
    "ChangeValueIndexConfig",
    "ChangeValueSidecarRef",
    "ChangeValueLineage",
    "ChangeValueIndexRow",
    "ChangeValueIndexTombstone",
    "ChangeValueIndexSnapshot",
    "ChangeValueQuery",
    "ChangeValueHit",
    "ChangeValueSearchResult",
    "ChangeValueVectorSearchProvider",
    "canonical_change_value_vector_index_bytes",
    "build_change_value_vector_index",
    "build_change_value_index",
    "search_change_value_vector_index",
    "search_change_value_index",
    "validate_change_value_search_result",
    "ChangeValueVectorIndex",
    "ChangeValueVectorIndexSnapshot",
    "ChangeValueVectorRow",
    "ChangeValueTombstone",
    "ChangeValueVectorQuery",
    "ChangeValueVectorHit",
    "ChangeValueVectorSearchResult",
]
