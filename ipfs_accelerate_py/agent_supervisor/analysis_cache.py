"""Bounded, content-addressed cache for analysis-stage receipts.

The cache deliberately stores only compact receipt projections.  Source text,
decoded model responses, AST bodies, and embedded artifact graphs belong in an
artifact store; cache entries contain references to those artifacts instead.

Each entry is an integrity-checked canonical JSON document addressed by every
input that can change an analysis result.  Writes are serialized across
threads and processes and published with ``os.replace`` so readers observe
either the old complete entry or the new complete entry, never a partial file.
The module has no dependency on analysis contracts so it can be used by
parallel analysis implementations and integrated with shared contracts later.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterator


ANALYSIS_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/analysis-cache-key@1"
)
ANALYSIS_CACHE_ENTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/analysis-cache-entry@1"
)
ANALYSIS_CACHE_SCHEMA: Final = ANALYSIS_CACHE_ENTRY_SCHEMA

DEFAULT_MAX_ENTRIES: Final = 512
DEFAULT_MAX_BYTES: Final = 32 * 1024 * 1024
DEFAULT_MAX_ENTRY_BYTES: Final = 128 * 1024
DEFAULT_MAX_RECEIPT_BYTES: Final = 96 * 1024
DEFAULT_NEGATIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_MAX_NEGATIVE_TTL_SECONDS: Final = 60 * 60
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_VALUE_DEPTH: Final = 6
DEFAULT_MAX_STRING_BYTES: Final = 16 * 1024

_ENTRY_SUFFIX = ".json"
_DIGEST_PREFIX = "analysis-cache-key:sha256:"
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()

# These payloads are intentionally excluded even when they happen to fit under
# the byte limit.  Their digests and artifact references are safe to cache.
_FORBIDDEN_RECEIPT_FIELDS = frozenset(
    {
        "source",
        "source_text",
        "source_code",
        "source_contents",
        "file_contents",
        "decoded_output",
        "decoded_model_output",
        "raw_model_output",
        "model_output",
        "prompt",
        "completion",
        "ast",
        "ast_body",
        "ast_bodies",
        "artifact_graph",
        "artifact_graphs",
        "graph",
    }
)
_ARTIFACT_REFERENCE_FIELDS = frozenset(
    {"artifact_id", "cid", "digest", "uri", "path", "ref"}
)


class AnalysisCacheError(RuntimeError):
    """Base class for analysis-cache failures."""


class ReceiptValidationError(AnalysisCacheError, ValueError):
    """A receipt contains non-compact or non-canonical data."""


class AnalysisOutcome(str, Enum):
    """Stable analysis outcomes persisted in cache entries."""

    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def coerce(cls, value: Any) -> "AnalysisOutcome":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "success": cls.SUCCESSFUL,
            "succeeded": cls.SUCCESSFUL,
            "complete": cls.SUCCESSFUL,
            "completed": cls.SUCCESSFUL,
            "ok": cls.SUCCESSFUL,
            "timeout": cls.TIMED_OUT,
            "timedout": cls.TIMED_OUT,
            "error": cls.FAILED,
            "failure": cls.FAILED,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ReceiptValidationError(
                f"analysis status must be one of: {choices}"
            ) from exc

    @property
    def is_completion_evidence(self) -> bool:
        return self is AnalysisOutcome.SUCCESSFUL


# Descriptive and short spellings for embedding callers.
AnalysisStatus = AnalysisOutcome
AnalysisCacheEntryStatus = AnalysisOutcome


class AnalysisCacheLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"


class AnalysisCacheReason(str, Enum):
    """Stable reason codes for metrics, audit records, and scheduling."""

    EXACT_KEY_HIT = "exact_key_hit"
    CACHE_MISS = "cache_miss"
    REPOSITORY_TREE_IDENTITY_CHANGED = "repository_tree_identity_changed"
    OBJECTIVE_REVISION_CHANGED = "objective_revision_changed"
    ANALYZER_VERSION_CHANGED = "analyzer_version_changed"
    SCHEMA_VERSION_CHANGED = "schema_version_changed"
    CONFIGURATION_DIGEST_CHANGED = "configuration_digest_changed"
    QUERY_DIGEST_CHANGED = "query_digest_changed"
    POLICY_DIGEST_CHANGED = "policy_digest_changed"
    STALE_ENTRY = "stale_entry"
    STALE_NEGATIVE_ENTRY = "stale_negative_entry"
    CORRUPT_ENTRY = "corrupt_entry"
    NOT_COMPLETION_EVIDENCE = "not_completion_evidence"
    MALFORMED_RECEIPT = "malformed_receipt"
    ENTRY_TOO_LARGE = "entry_too_large"

    # Compatibility aliases using shorter dimension names.
    REPOSITORY_TREE_CHANGED = "repository_tree_identity_changed"
    CONFIGURATION_CHANGED = "configuration_digest_changed"
    QUERY_CHANGED = "query_digest_changed"
    POLICY_CHANGED = "policy_digest_changed"


CacheLookupStatus = AnalysisCacheLookupStatus
CacheInvalidationReason = AnalysisCacheReason
CacheReason = AnalysisCacheReason


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode deterministic JSON, rejecting non-portable values."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("canonical JSON cannot contain NaN or infinity")
            return item
        if isinstance(item, Enum):
            return normalize(item.value)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("canonical JSON object keys must be strings")
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            return normalize(converter())
        raise ValueError(
            f"unsupported canonical JSON value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_analysis_json(value: Any) -> str:
    """Return the canonical JSON representation used by this cache."""

    return _canonical_json_bytes(value).decode("utf-8")


def digest_analysis_input(value: Any) -> str:
    """Return a lowercase SHA-256 digest for canonical JSON input."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identity_component(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{name} must not be empty")
    try:
        # Round-trip so mutable caller-owned mappings cannot mutate a key.
        return json.loads(canonical_analysis_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc


@dataclass(frozen=True, init=False)
class AnalysisCacheKey:
    """Content address binding every input capable of changing analysis."""

    repository_tree_identity: Any
    objective_revision: Any
    analyzer_version: Any
    schema_version: Any
    configuration_digest: Any
    query_digest: Any
    policy_digest: Any

    def __init__(
        self,
        repository_tree_identity: Any = None,
        objective_revision: Any = None,
        analyzer_version: Any = None,
        schema_version: Any = None,
        configuration_digest: Any = None,
        query_digest: Any = None,
        policy_digest: Any = None,
        *,
        repository_tree: Any = None,
        tree_identity: Any = None,
        config_digest: Any = None,
    ) -> None:
        tree_values = [
            item
            for item in (
                repository_tree_identity,
                repository_tree,
                tree_identity,
            )
            if item is not None
        ]
        if not tree_values:
            tree = None
        else:
            tree = tree_values[0]
            canonical_tree = canonical_analysis_json(tree)
            if any(
                canonical_analysis_json(item) != canonical_tree
                for item in tree_values[1:]
            ):
                raise ValueError("repository tree identity aliases disagree")
        if configuration_digest is not None and config_digest is not None:
            if canonical_analysis_json(
                configuration_digest
            ) != canonical_analysis_json(config_digest):
                raise ValueError("configuration digest aliases disagree")
        configuration = (
            configuration_digest
            if configuration_digest is not None
            else config_digest
        )
        values = {
            "repository_tree_identity": tree,
            "objective_revision": objective_revision,
            "analyzer_version": analyzer_version,
            "schema_version": schema_version,
            "configuration_digest": configuration,
            "query_digest": query_digest,
            "policy_digest": policy_digest,
        }
        for name, value in values.items():
            object.__setattr__(self, name, _identity_component(value, name))

    @property
    def repository_tree(self) -> Any:
        return self.repository_tree_identity

    @property
    def config_digest(self) -> Any:
        return self.configuration_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CACHE_KEY_SCHEMA,
            "repository_tree_identity": self.repository_tree_identity,
            "objective_revision": self.objective_revision,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "policy_digest": self.policy_digest,
        }

    @property
    def digest(self) -> str:
        return digest_analysis_input(self.to_dict())

    @property
    def key_id(self) -> str:
        return f"{_DIGEST_PREFIX}{self.digest}"

    @property
    def cache_key(self) -> str:
        return self.key_id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisCacheKey":
        if not isinstance(value, Mapping):
            raise ValueError("analysis cache key must be an object")
        schema = value.get("schema", value.get("schema_version_id"))
        if schema not in (None, ANALYSIS_CACHE_KEY_SCHEMA):
            raise ValueError("unsupported analysis cache-key schema")
        return cls(
            repository_tree_identity=value.get(
                "repository_tree_identity", value.get("repository_tree")
            ),
            objective_revision=value.get("objective_revision"),
            analyzer_version=value.get("analyzer_version"),
            schema_version=value.get("schema_version"),
            configuration_digest=value.get(
                "configuration_digest", value.get("config_digest")
            ),
            query_digest=value.get("query_digest"),
            policy_digest=value.get("policy_digest"),
        )


def build_analysis_cache_key(
    *,
    repository_tree_identity: Any = None,
    repository_tree: Any = None,
    tree_identity: Any = None,
    objective_revision: Any,
    analyzer_version: Any,
    schema_version: Any,
    configuration_digest: Any = None,
    config_digest: Any = None,
    query_digest: Any,
    policy_digest: Any,
) -> AnalysisCacheKey:
    """Build an analysis key while accepting common tree/config aliases."""

    return AnalysisCacheKey(
        repository_tree_identity=repository_tree_identity,
        repository_tree=repository_tree,
        tree_identity=tree_identity,
        objective_revision=objective_revision,
        analyzer_version=analyzer_version,
        schema_version=schema_version,
        configuration_digest=configuration_digest,
        config_digest=config_digest,
        query_digest=query_digest,
        policy_digest=policy_digest,
    )


make_analysis_cache_key = build_analysis_cache_key


def _normalized_field_name(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _compact_value(
    value: Any,
    *,
    path: str,
    depth: int,
    max_depth: int,
    max_string_bytes: int,
) -> Any:
    if depth > max_depth:
        raise ReceiptValidationError(
            f"{path} exceeds maximum receipt nesting depth {max_depth}"
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReceiptValidationError(f"{path} cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return _compact_value(
            value.value,
            path=path,
            depth=depth,
            max_depth=max_depth,
            max_string_bytes=max_string_bytes,
        )
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max_string_bytes:
            raise ReceiptValidationError(
                f"{path} contains an oversized inline string; store it as an artifact"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ReceiptValidationError(f"{path} keys must be strings")
            normalized_key = _normalized_field_name(raw_key)
            if (
                normalized_key in _FORBIDDEN_RECEIPT_FIELDS
                or normalized_key.endswith("_body")
                or normalized_key.endswith("_bodies")
                or normalized_key.endswith("_graph")
                or normalized_key.endswith("_ast")
            ):
                raise ReceiptValidationError(
                    f"{path}.{raw_key} must be stored as an artifact reference"
                )
            result[raw_key] = _compact_value(
                child,
                path=f"{path}.{raw_key}",
                depth=depth + 1,
                max_depth=max_depth,
                max_string_bytes=max_string_bytes,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        return [
            _compact_value(
                child,
                path=f"{path}[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_string_bytes=max_string_bytes,
            )
            for index, child in enumerate(value)
        ]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _compact_value(
            converter(),
            path=path,
            depth=depth,
            max_depth=max_depth,
            max_string_bytes=max_string_bytes,
        )
    raise ReceiptValidationError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def _normalize_artifact_reference(
    value: Any,
    *,
    index: int,
    max_string_bytes: int,
) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        value = {"ref": str(value)}
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, Mapping) or not value:
        raise ReceiptValidationError(
            f"artifact_refs[{index}] must be a nonempty object or reference string"
        )
    result: dict[str, Any] = {}
    for raw_key, child in value.items():
        if not isinstance(raw_key, str):
            raise ReceiptValidationError(
                f"artifact_refs[{index}] keys must be strings"
            )
        normalized_key = _normalized_field_name(raw_key)
        if (
            normalized_key in _FORBIDDEN_RECEIPT_FIELDS
            or normalized_key.endswith("_body")
            or normalized_key.endswith("_bodies")
            or normalized_key.endswith("_graph")
            or normalized_key.endswith("_ast")
        ):
            raise ReceiptValidationError(
                f"artifact_refs[{index}].{raw_key} cannot embed artifact content"
            )
        if isinstance(child, Path):
            child = str(child)
        if child is not None and not isinstance(child, (str, bool, int, float)):
            raise ReceiptValidationError(
                f"artifact_refs[{index}].{raw_key} must be a scalar"
            )
        result[raw_key] = _compact_value(
            child,
            path=f"artifact_refs[{index}].{raw_key}",
            depth=0,
            max_depth=0,
            max_string_bytes=max_string_bytes,
        )
    if not any(
        _normalized_field_name(key) in _ARTIFACT_REFERENCE_FIELDS
        and value not in (None, "")
        for key, value in result.items()
    ):
        raise ReceiptValidationError(
            f"artifact_refs[{index}] requires an id, CID, digest, URI, path, or ref"
        )
    return result


def compact_analysis_receipt(
    receipt: Mapping[str, Any] | Any,
    *,
    status: AnalysisOutcome | str | None = None,
    max_receipt_bytes: int = DEFAULT_MAX_RECEIPT_BYTES,
    max_depth: int = DEFAULT_MAX_VALUE_DEPTH,
    max_string_bytes: int = DEFAULT_MAX_STRING_BYTES,
) -> dict[str, Any]:
    """Validate and normalize a compact analysis receipt.

    Unknown compact metadata is retained for forward compatibility.  Artifact
    references are the sole exception: they are normalized to a shallow list
    so an artifact graph cannot be recursively embedded in the cache.
    """

    converter = getattr(receipt, "to_dict", None)
    if callable(converter):
        receipt = converter()
    if not isinstance(receipt, Mapping):
        raise ReceiptValidationError("analysis receipt must be an object")
    raw = dict(receipt)
    raw_status = status
    if raw_status is None:
        raw_status = raw.get("status", raw.get("outcome"))
    typed_status = AnalysisOutcome.coerce(raw_status)
    if "status" in raw and status is not None:
        if AnalysisOutcome.coerce(raw["status"]) is not typed_status:
            raise ReceiptValidationError("receipt status and status argument disagree")
    if "outcome" in raw:
        if AnalysisOutcome.coerce(raw["outcome"]) is not typed_status:
            raise ReceiptValidationError("receipt outcome and status disagree")
        raw.pop("outcome")
    # Completion authority is derived from the typed outcome and lookup
    # freshness.  Never persist caller-supplied booleans that could be mistaken
    # for an independent completion decision.
    raw.pop("completion_evidence", None)
    raw.pop("is_completion_evidence", None)

    artifacts = raw.pop("artifacts", None)
    artifact_refs = raw.pop("artifact_refs", None)
    if artifacts is not None and artifact_refs is not None:
        raise ReceiptValidationError(
            "use artifact_refs only; artifacts and artifact_refs cannot both be set"
        )
    references = artifact_refs if artifact_refs is not None else artifacts
    if references is None:
        references = ()
    if isinstance(references, (str, bytes, bytearray)) or not isinstance(
        references, Sequence
    ):
        raise ReceiptValidationError("artifact_refs must be an array")

    raw["status"] = typed_status.value
    normalized = _compact_value(
        raw,
        path="receipt",
        depth=0,
        max_depth=max_depth,
        max_string_bytes=max_string_bytes,
    )
    normalized["artifact_refs"] = [
        _normalize_artifact_reference(
            item, index=index, max_string_bytes=max_string_bytes
        )
        for index, item in enumerate(references)
    ]
    encoded = _canonical_json_bytes(normalized)
    if len(encoded) > max_receipt_bytes:
        raise ReceiptValidationError(
            "analysis receipt exceeds max_receipt_bytes; persist bulky data as artifacts"
        )
    return normalized


@dataclass(frozen=True)
class AnalysisReceipt:
    """Typed convenience wrapper around a compact receipt projection."""

    status: AnalysisOutcome
    receipt_id: str = ""
    summary: Any = field(default_factory=dict)
    artifact_refs: tuple[Any, ...] = ()
    diagnostics: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AnalysisOutcome.coerce(self.status))
        normalized = compact_analysis_receipt(self.to_dict_unchecked())
        object.__setattr__(self, "receipt_id", str(normalized.get("receipt_id") or ""))
        object.__setattr__(self, "summary", normalized.get("summary", {}))
        object.__setattr__(
            self, "artifact_refs", tuple(normalized.get("artifact_refs", ()))
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(str(item) for item in normalized.get("diagnostics", ())),
        )
        object.__setattr__(self, "counts", normalized.get("counts", {}))
        object.__setattr__(self, "metadata", normalized.get("metadata", {}))

    def to_dict_unchecked(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "receipt_id": self.receipt_id,
            "summary": self.summary,
            "artifact_refs": list(self.artifact_refs),
            "diagnostics": list(self.diagnostics),
            "counts": dict(self.counts),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return compact_analysis_receipt(self.to_dict_unchecked())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisReceipt":
        normalized = compact_analysis_receipt(value)
        return cls(
            status=AnalysisOutcome.coerce(normalized["status"]),
            receipt_id=str(normalized.get("receipt_id") or ""),
            summary=normalized.get("summary", {}),
            artifact_refs=tuple(normalized.get("artifact_refs", ())),
            diagnostics=tuple(normalized.get("diagnostics", ())),
            counts=normalized.get("counts", {}),
            metadata=normalized.get("metadata", {}),
        )


@dataclass(frozen=True)
class AnalysisCacheEntry:
    key: AnalysisCacheKey
    receipt: Mapping[str, Any]
    status: AnalysisOutcome
    created_at_ms: int
    expires_at_ms: int | None = None
    entry_digest: str = ""

    @property
    def is_completion_evidence(self) -> bool:
        return self.status.is_completion_evidence

    @property
    def completion_evidence(self) -> bool:
        return self.is_completion_evidence

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CACHE_ENTRY_SCHEMA,
            "key_id": self.key.key_id,
            "key": self.key.to_dict(),
            "status": self.status.value,
            "receipt": dict(self.receipt),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @property
    def computed_digest(self) -> str:
        return f"sha256:{digest_analysis_input(self._unsigned_dict())}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_dict(),
            "entry_digest": self.entry_digest or self.computed_digest,
        }

    def serialize(self) -> bytes:
        return _canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def create(
        cls,
        key: AnalysisCacheKey,
        receipt: Mapping[str, Any],
        *,
        created_at_ms: int,
        expires_at_ms: int | None,
    ) -> "AnalysisCacheEntry":
        status = AnalysisOutcome.coerce(receipt.get("status"))
        entry = cls(
            key=key,
            receipt=dict(receipt),
            status=status,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
        )
        return cls(
            key=entry.key,
            receipt=entry.receipt,
            status=entry.status,
            created_at_ms=entry.created_at_ms,
            expires_at_ms=entry.expires_at_ms,
            entry_digest=entry.computed_digest,
        )


@dataclass(frozen=True)
class AnalysisCacheLookupResult:
    status: AnalysisCacheLookupStatus
    key: AnalysisCacheKey
    entry: AnalysisCacheEntry | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.status is AnalysisCacheLookupStatus.HIT

    @property
    def miss(self) -> bool:
        return self.status is AnalysisCacheLookupStatus.MISS

    @property
    def invalidated(self) -> bool:
        return self.status is AnalysisCacheLookupStatus.INVALIDATED

    @property
    def receipt(self) -> Mapping[str, Any] | None:
        return self.entry.receipt if self.hit and self.entry is not None else None

    @property
    def outcome(self) -> AnalysisOutcome | None:
        return self.entry.status if self.entry is not None else None

    @property
    def is_completion_evidence(self) -> bool:
        return bool(
            self.hit
            and self.entry is not None
            and self.entry.is_completion_evidence
        )

    @property
    def completion_evidence(self) -> bool:
        return self.is_completion_evidence

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def reason(self) -> str:
        return self.reason_code


@dataclass(frozen=True)
class AnalysisCacheStoreResult:
    stored: bool
    key: AnalysisCacheKey
    entry: AnalysisCacheEntry | None = None
    reason_codes: tuple[str, ...] = ()
    evicted_count: int = 0

    def __bool__(self) -> bool:
        return self.stored

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True)
class AnalysisCacheStats:
    entry_count: int
    total_bytes: int
    successful_count: int
    partial_count: int
    failed_count: int
    timed_out_count: int
    inconclusive_count: int
    corrupt_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "entry_count": self.entry_count,
            "total_bytes": self.total_bytes,
            "successful_count": self.successful_count,
            "partial_count": self.partial_count,
            "failed_count": self.failed_count,
            "timed_out_count": self.timed_out_count,
            "inconclusive_count": self.inconclusive_count,
            "corrupt_count": self.corrupt_count,
        }


def _strict_json_loads(value: bytes) -> Any:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = child
        return result

    return json.loads(
        value.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {constant}")
        ),
    )


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_cache_lock(
    path: Path, *, timeout_seconds: float
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _thread_lock(path)
    deadline = time.monotonic() + timeout_seconds
    if not lock.acquire(timeout=max(0.0, timeout_seconds)):
        raise TimeoutError(f"timed out acquiring analysis cache thread lock: {path}")
    handle = path.open("a+b")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring analysis cache process lock: {path}"
                    )
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        lock.release()


_KEY_DIMENSIONS: tuple[tuple[str, AnalysisCacheReason], ...] = (
    (
        "repository_tree_identity",
        AnalysisCacheReason.REPOSITORY_TREE_IDENTITY_CHANGED,
    ),
    ("objective_revision", AnalysisCacheReason.OBJECTIVE_REVISION_CHANGED),
    ("analyzer_version", AnalysisCacheReason.ANALYZER_VERSION_CHANGED),
    ("schema_version", AnalysisCacheReason.SCHEMA_VERSION_CHANGED),
    (
        "configuration_digest",
        AnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED,
    ),
    ("query_digest", AnalysisCacheReason.QUERY_DIGEST_CHANGED),
    ("policy_digest", AnalysisCacheReason.POLICY_DIGEST_CHANGED),
)


class AnalysisCache:
    """A durable bounded cache of compact analysis receipts."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
        max_receipt_bytes: int = DEFAULT_MAX_RECEIPT_BYTES,
        default_negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS,
        max_negative_ttl_seconds: int = DEFAULT_MAX_NEGATIVE_TTL_SECONDS,
        default_success_ttl_seconds: int | None = None,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        for name, value in (
            ("max_entries", max_entries),
            ("max_bytes", max_bytes),
            ("max_entry_bytes", max_entry_bytes),
            ("max_receipt_bytes", max_receipt_bytes),
            ("default_negative_ttl_seconds", default_negative_ttl_seconds),
            ("max_negative_ttl_seconds", max_negative_ttl_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_entry_bytes > max_bytes:
            raise ValueError("max_entry_bytes cannot exceed max_bytes")
        if max_receipt_bytes > max_entry_bytes:
            raise ValueError("max_receipt_bytes cannot exceed max_entry_bytes")
        if default_negative_ttl_seconds > max_negative_ttl_seconds:
            raise ValueError(
                "default_negative_ttl_seconds cannot exceed max_negative_ttl_seconds"
            )
        if default_success_ttl_seconds is not None and (
            isinstance(default_success_ttl_seconds, bool)
            or not isinstance(default_success_ttl_seconds, int)
            or default_success_ttl_seconds <= 0
        ):
            raise ValueError(
                "default_success_ttl_seconds must be a positive integer or None"
            )
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        if path is None:
            path = tempfile.mkdtemp(prefix="analysis-cache-")
        self.path = Path(path)
        self.cache_dir = self.path
        self.entries_dir = self.path / "entries"
        self.lock_path = self.path / ".analysis-cache.lock"
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self.max_receipt_bytes = max_receipt_bytes
        self.default_negative_ttl_seconds = default_negative_ttl_seconds
        self.max_negative_ttl_seconds = max_negative_ttl_seconds
        self.default_success_ttl_seconds = default_success_ttl_seconds
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._clock = clock
        self.entries_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path, 0o700)
        except OSError:
            pass

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _coerce_key(
        self, key: AnalysisCacheKey | Mapping[str, Any]
    ) -> AnalysisCacheKey:
        return (
            key
            if isinstance(key, AnalysisCacheKey)
            else AnalysisCacheKey.from_dict(key)
        )

    def entry_path(
        self, key: AnalysisCacheKey | Mapping[str, Any] | str
    ) -> Path:
        if isinstance(key, str):
            digest = (
                key[len(_DIGEST_PREFIX) :]
                if key.startswith(_DIGEST_PREFIX)
                else key
            )
            if not (
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("analysis cache key digest must be 64 lowercase hex chars")
        else:
            digest = self._coerce_key(key).digest
        return self.entries_dir / digest[:2] / f"{digest}{_ENTRY_SUFFIX}"

    def _entry_paths(self) -> list[Path]:
        if not self.entries_dir.exists():
            return []
        return sorted(self.entries_dir.glob("*/*" + _ENTRY_SUFFIX))

    def _decode_entry(
        self, raw: bytes, *, expected_key: AnalysisCacheKey | None = None
    ) -> AnalysisCacheEntry:
        if len(raw) > self.max_entry_bytes:
            raise ValueError("entry exceeds configured max_entry_bytes")
        payload = _strict_json_loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("cache entry must be an object")
        if payload.get("schema") != ANALYSIS_CACHE_ENTRY_SCHEMA:
            raise ValueError("unsupported analysis cache entry schema")
        key = AnalysisCacheKey.from_dict(payload.get("key"))
        if payload.get("key_id") != key.key_id:
            raise ValueError("cache entry key binding mismatch")
        if expected_key is not None and key != expected_key:
            raise ValueError("cache entry path/key binding mismatch")
        receipt_value = payload.get("receipt")
        receipt = compact_analysis_receipt(
            receipt_value,
            max_receipt_bytes=self.max_receipt_bytes,
        )
        status = AnalysisOutcome.coerce(payload.get("status"))
        if AnalysisOutcome.coerce(receipt.get("status")) is not status:
            raise ValueError("cache entry receipt status mismatch")
        created_at_ms = payload.get("created_at_ms")
        expires_at_ms = payload.get("expires_at_ms")
        if (
            isinstance(created_at_ms, bool)
            or not isinstance(created_at_ms, int)
            or created_at_ms < 0
        ):
            raise ValueError("cache entry has invalid created_at_ms")
        if expires_at_ms is not None and (
            isinstance(expires_at_ms, bool)
            or not isinstance(expires_at_ms, int)
            or expires_at_ms <= created_at_ms
        ):
            raise ValueError("cache entry has invalid expires_at_ms")
        if not status.is_completion_evidence and expires_at_ms is None:
            raise ValueError("non-success cache entry must have bounded expiry")
        entry = AnalysisCacheEntry(
            key=key,
            receipt=receipt,
            status=status,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
            entry_digest=str(payload.get("entry_digest") or ""),
        )
        if entry.entry_digest != entry.computed_digest:
            raise ValueError("cache entry integrity digest mismatch")
        return entry

    def _read_path(
        self, path: Path, *, expected_key: AnalysisCacheKey | None = None
    ) -> AnalysisCacheEntry:
        entry = self._decode_entry(path.read_bytes(), expected_key=expected_key)
        if path.stem != entry.key.digest or path.parent.name != entry.key.digest[:2]:
            raise ValueError("cache entry content address/path mismatch")
        return entry

    def _is_stale(self, entry: AnalysisCacheEntry, now_ms: int) -> bool:
        return (
            entry.expires_at_ms is not None
            and now_ms >= entry.expires_at_ms
        )

    def lookup(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        *,
        require_completion_evidence: bool = False,
    ) -> AnalysisCacheLookupResult:
        cache_key = self._coerce_key(key)
        path = self.entry_path(cache_key)
        now_ms = self._now_ms()
        try:
            entry = self._read_path(path, expected_key=cache_key)
        except FileNotFoundError:
            entry = None
        except (OSError, TypeError, ValueError, ReceiptValidationError):
            return AnalysisCacheLookupResult(
                AnalysisCacheLookupStatus.INVALIDATED,
                cache_key,
                reason_codes=(AnalysisCacheReason.CORRUPT_ENTRY.value,),
            )

        if entry is not None:
            if self._is_stale(entry, now_ms):
                reason = (
                    AnalysisCacheReason.STALE_ENTRY
                    if entry.is_completion_evidence
                    else AnalysisCacheReason.STALE_NEGATIVE_ENTRY
                )
                return AnalysisCacheLookupResult(
                    AnalysisCacheLookupStatus.INVALIDATED,
                    cache_key,
                    entry=entry,
                    reason_codes=(reason.value,),
                )
            if require_completion_evidence and not entry.is_completion_evidence:
                return AnalysisCacheLookupResult(
                    AnalysisCacheLookupStatus.INVALIDATED,
                    cache_key,
                    entry=entry,
                    reason_codes=(
                        AnalysisCacheReason.NOT_COMPLETION_EVIDENCE.value,
                    ),
                )
            return AnalysisCacheLookupResult(
                AnalysisCacheLookupStatus.HIT,
                cache_key,
                entry=entry,
                reason_codes=(AnalysisCacheReason.EXACT_KEY_HIT.value,),
            )

        candidate = self._closest_candidate(cache_key, now_ms=now_ms)
        if candidate is None:
            return AnalysisCacheLookupResult(
                AnalysisCacheLookupStatus.MISS,
                cache_key,
                reason_codes=(AnalysisCacheReason.CACHE_MISS.value,),
            )
        differences = tuple(
            reason.value
            for name, reason in _KEY_DIMENSIONS
            if getattr(candidate.key, name) != getattr(cache_key, name)
        )
        return AnalysisCacheLookupResult(
            AnalysisCacheLookupStatus.INVALIDATED,
            cache_key,
            reason_codes=differences
            or (AnalysisCacheReason.CACHE_MISS.value,),
        )

    def get(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        *,
        require_completion_evidence: bool = False,
    ) -> AnalysisCacheLookupResult:
        return self.lookup(
            key, require_completion_evidence=require_completion_evidence
        )

    def _closest_candidate(
        self, key: AnalysisCacheKey, *, now_ms: int
    ) -> AnalysisCacheEntry | None:
        candidates: list[tuple[int, int, str, AnalysisCacheEntry]] = []
        for path in self._entry_paths():
            try:
                entry = self._read_path(path)
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                continue
            if self._is_stale(entry, now_ms):
                continue
            distance = sum(
                getattr(entry.key, name) != getattr(key, name)
                for name, _reason in _KEY_DIMENSIONS
            )
            # A completely unrelated entry is a miss, not evidence that this
            # specific analysis was invalidated in every dimension.
            if distance == len(_KEY_DIMENSIONS):
                continue
            candidates.append(
                (distance, -entry.created_at_ms, entry.key.key_id, entry)
            )
        return min(candidates)[-1] if candidates else None

    def put(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        receipt: Mapping[str, Any] | AnalysisReceipt,
        *,
        status: AnalysisOutcome | str | None = None,
        ttl_seconds: int | None = None,
    ) -> AnalysisCacheStoreResult:
        cache_key = self._coerce_key(key)
        try:
            normalized = compact_analysis_receipt(
                receipt,
                status=status,
                max_receipt_bytes=self.max_receipt_bytes,
            )
        except (TypeError, ValueError, ReceiptValidationError):
            return AnalysisCacheStoreResult(
                False,
                cache_key,
                reason_codes=(AnalysisCacheReason.MALFORMED_RECEIPT.value,),
            )
        outcome = AnalysisOutcome.coerce(normalized["status"])
        ttl = self._effective_ttl(outcome, ttl_seconds)
        now_ms = self._now_ms()
        expires_at_ms = None if ttl is None else now_ms + ttl * 1000
        entry = AnalysisCacheEntry.create(
            cache_key,
            normalized,
            created_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
        )
        encoded = entry.serialize()
        if len(encoded) > self.max_entry_bytes or len(encoded) > self.max_bytes:
            return AnalysisCacheStoreResult(
                False,
                cache_key,
                reason_codes=(AnalysisCacheReason.ENTRY_TOO_LARGE.value,),
            )

        path = self.entry_path(cache_key)
        with _exclusive_cache_lock(
            self.lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._atomic_write(path, encoded)
            evicted = self._prune_locked(
                now_ms=now_ms, keep_path=path
            )
            # Verify after replacement while still holding the writer lock.
            try:
                persisted = self._read_path(path, expected_key=cache_key)
            except (OSError, TypeError, ValueError, ReceiptValidationError) as exc:
                raise AnalysisCacheError(
                    f"persisted analysis cache entry failed verification: {path}"
                ) from exc
        return AnalysisCacheStoreResult(
            True,
            cache_key,
            entry=persisted,
            evicted_count=evicted,
        )

    store = put

    def _effective_ttl(
        self, outcome: AnalysisOutcome, ttl_seconds: int | None
    ) -> int | None:
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a positive integer or None")
        if outcome.is_completion_evidence:
            return (
                self.default_success_ttl_seconds
                if ttl_seconds is None
                else ttl_seconds
            )
        requested = (
            self.default_negative_ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )
        return min(requested, self.max_negative_ttl_seconds)

    def _atomic_write(self, path: Path, encoded: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, path)
            # Make the directory entry durable when the platform supports it.
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

    def prune(self) -> int:
        """Remove corrupt, expired, and oldest excess entries."""

        with _exclusive_cache_lock(
            self.lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            return self._prune_locked(now_ms=self._now_ms())

    def _prune_locked(
        self, *, now_ms: int, keep_path: Path | None = None
    ) -> int:
        removed = 0
        records: list[tuple[Path, AnalysisCacheEntry, int]] = []
        for path in self._entry_paths():
            try:
                size = path.stat().st_size
                entry = self._read_path(path)
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
                continue
            if self._is_stale(entry, now_ms):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    records.append((path, entry, size))
                continue
            records.append((path, entry, size))

        total = sum(item[2] for item in records)
        excess_count = max(0, len(records) - self.max_entries)
        # Oldest entries go first.  The entry just written is retained whenever
        # possible so a successful put truthfully reports a durable store.
        eviction_order = sorted(
            records,
            key=lambda item: (
                item[0] == keep_path,
                item[1].created_at_ms,
                item[1].key.key_id,
            ),
        )
        for path, _entry, size in eviction_order:
            if excess_count <= 0 and total <= self.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
            total -= size
            excess_count = max(0, excess_count - 1)
        self._remove_empty_shards()
        return removed

    def _remove_empty_shards(self) -> None:
        if not self.entries_dir.exists():
            return
        for path in self.entries_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                path.rmdir()
            except OSError:
                pass

    def stats(self) -> AnalysisCacheStats:
        counts = {status: 0 for status in AnalysisOutcome}
        total = 0
        corrupt = 0
        entry_count = 0
        now_ms = self._now_ms()
        for path in self._entry_paths():
            try:
                size = path.stat().st_size
                total += size
                entry = self._read_path(path)
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                corrupt += 1
                continue
            if self._is_stale(entry, now_ms):
                continue
            entry_count += 1
            counts[entry.status] += 1
        return AnalysisCacheStats(
            entry_count=entry_count,
            total_bytes=total,
            successful_count=counts[AnalysisOutcome.SUCCESSFUL],
            partial_count=counts[AnalysisOutcome.PARTIAL],
            failed_count=counts[AnalysisOutcome.FAILED],
            timed_out_count=counts[AnalysisOutcome.TIMED_OUT],
            inconclusive_count=counts[AnalysisOutcome.INCONCLUSIVE],
            corrupt_count=corrupt,
        )

    def clear(self) -> int:
        """Remove all cache entries, preserving the cache directory and lock."""

        removed = 0
        with _exclusive_cache_lock(
            self.lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            for path in self._entry_paths():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
            self._remove_empty_shards()
        return removed


# Names consistent with the formal-verification cache's public vocabulary.
ContentAddressedAnalysisCache = AnalysisCache
AnalysisLookupResult = AnalysisCacheLookupResult
AnalysisStoreResult = AnalysisCacheStoreResult


__all__ = [
    "ANALYSIS_CACHE_ENTRY_SCHEMA",
    "ANALYSIS_CACHE_KEY_SCHEMA",
    "ANALYSIS_CACHE_SCHEMA",
    "AnalysisCache",
    "AnalysisCacheEntry",
    "AnalysisCacheEntryStatus",
    "AnalysisCacheError",
    "AnalysisCacheKey",
    "AnalysisCacheLookupResult",
    "AnalysisCacheLookupStatus",
    "AnalysisCacheReason",
    "AnalysisCacheStats",
    "AnalysisCacheStoreResult",
    "AnalysisLookupResult",
    "AnalysisOutcome",
    "AnalysisReceipt",
    "AnalysisStatus",
    "AnalysisStoreResult",
    "CacheInvalidationReason",
    "CacheLookupStatus",
    "CacheReason",
    "ContentAddressedAnalysisCache",
    "ReceiptValidationError",
    "build_analysis_cache_key",
    "canonical_analysis_json",
    "compact_analysis_receipt",
    "digest_analysis_input",
    "make_analysis_cache_key",
]
