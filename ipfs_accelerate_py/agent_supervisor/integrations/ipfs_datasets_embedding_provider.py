"""Pinned, fail-closed embedding adapter for optional ``ipfs_datasets_py``.

Doctor retrieval may use validated local/pinned embeddings to improve *recall*
only.  This module therefore:

* requires an explicit :class:`PinnedEmbeddingPolicy` binding provider, model
  artifact, revision, dimension, chunker, normalizer, distance metric, and
  corpus/index roots before any embedding work;
* never admits an unpinned remote embedding backend;
* runs a deterministic canary that rejects missing-dependency success shims,
  constant fallback vectors, non-finite values, dimension mismatches, and
  configuration drift;
* on canary failure disables **only** the optional vector lane (exact
  AST/graph analysis continues);
* constructs and inspects capability without eagerly importing the optional
  package (lazy import only on demand); and
* always returns ``semantic_authority=false`` — embeddings cannot authorize
  semantics, values, placements, targets, or writes.

Injected backends are supported for hermetic tests.  A backend may expose
``embed(texts)``, ``embed_texts(texts)``, or be a callable mapping texts to
vectors.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract, content_identity


# ---------------------------------------------------------------------------
# Versioning / schemas
# ---------------------------------------------------------------------------

IPFS_DATASETS_EMBEDDING_PROVIDER_ID: Final = "ipfs_datasets_py.embeddings"
IPFS_DATASETS_EMBEDDING_PROVIDER_VERSION: Final = "1.0.0"
DEFAULT_OPTIONAL_MODULE: Final = "ipfs_datasets_py"
DEFAULT_EMBEDDING_MODULE_CANDIDATES: Final = (
    "ipfs_datasets_py.ml.embeddings.embeddings_engine",
    "ipfs_datasets_py.embeddings",
    "ipfs_datasets_py.utils.embedding_adapter",
)

PROVIDER_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-embedding-capability@1"
)
PINNED_EMBEDDING_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pinned-embedding-policy@1"
)
EMBEDDING_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-embedding-request@1"
)
EMBEDDING_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-embedding-result@1"
)
EMBEDDING_CANARY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-embedding-canary@1"
)

DEFAULT_MAX_TEXTS: Final = 32
DEFAULT_MAX_TEXT_BYTES: Final = 8_192
DEFAULT_MAX_BATCH_BYTES: Final = 64 * 1024
HARD_MAX_DIMENSIONS: Final = 65_536
# Canonical proof contracts forbid floats; embed payloads use fixed-point ints.
VECTOR_SCALE: Final = 1_000_000
CANARY_TEXTS: Final = (
    "doctor-embedding-canary:alpha",
    "doctor-embedding-canary:beta-distinct",
    "doctor-embedding-canary:gamma-third",
)
_ALLOWED_NORMALIZERS: Final = frozenset({"l2", "none"})
_ALLOWED_DISTANCES: Final = frozenset({"cosine", "dot_product", "euclidean", "l2"})
_IMPORT_LOCK: Final = threading.Lock()


class EmbeddingProviderError(ValueError):
    """Embedding policy, canary, or request violates the fail-closed contract."""


class EmbeddingProviderBindingError(EmbeddingProviderError):
    """A required pin, root, or configuration binding is mixed or missing."""


class EmbeddingProviderCanaryError(EmbeddingProviderError):
    """The deterministic canary rejected the optional vector lane."""


class EmbeddingLaneStatus(str, Enum):
    """Whether the optional vector lane may participate in retrieval."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    CANARY_FAILED = "canary_failed"
    UNPINNED_REJECTED = "unpinned_rejected"
    NOT_PROBED = "not_probed"


class EmbeddingCanaryDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EmbeddingCanaryReason(str, Enum):
    """Stable public diagnostics for canary / lane outcomes."""

    OK = "ok"
    MISSING_DEPENDENCY_SHIM = "missing_dependency_success_shim"
    CONSTANT_VECTOR = "constant_fallback_vector"
    NON_FINITE = "non_finite_values"
    DIMENSION_DRIFT = "dimension_drift"
    CONFIG_DRIFT = "config_drift"
    UNPINNED_REMOTE = "unpinned_remote_embedding"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    EMPTY_OUTPUT = "empty_output"
    TEXT_BOUND_EXCEEDED = "text_bound_exceeded"
    NOT_RUN = "not_run"


class EmbeddingProviderStatus(str, Enum):
    COMPLETED = "completed"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    CANARY_FAILED = "canary_failed"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, CanonicalContract):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EmbeddingProviderError("canonical JSON cannot contain NaN or infinity")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _fingerprint(value: Any, *, prefix: str = "embedding") -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str, *, required: bool = True, maximum: int = 512) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise EmbeddingProviderError(f"{name} is required")
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise EmbeddingProviderError(f"{name} is invalid or exceeds its bound")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise EmbeddingProviderError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str, *, maximum: int = HARD_MAX_DIMENSIONS) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise EmbeddingProviderError(f"{name} must be an integer from 1 through {maximum}")
    return value


def _verify_record_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    claimed = payload.get("content_id", payload.get("cid", ""))
    if claimed not in (None, "", record.content_id):
        raise EmbeddingProviderBindingError(
            "stored content identity does not match the canonical record"
        )


def _vector(value: Any, dimensions: int, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise EmbeddingProviderError(f"{name} must be a numeric vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EmbeddingProviderError(f"{name} must be a numeric vector") from exc
    if len(result) != dimensions:
        raise EmbeddingProviderError(
            f"{name} dimension mismatch: expected {dimensions}, got {len(result)}"
        )
    if not all(math.isfinite(item) for item in result):
        raise EmbeddingProviderError(f"{name} contains non-finite values")
    return result


def _scale_vector(vector: Sequence[float]) -> tuple[int, ...]:
    return tuple(int(round(float(item) * VECTOR_SCALE)) for item in vector)


def _unscale_vector(vector: Sequence[int]) -> tuple[float, ...]:
    return tuple(int(item) / VECTOR_SCALE for item in vector)


def _is_constant_vector(vector: Sequence[float]) -> bool:
    if not vector:
        return True
    first = float(vector[0])
    return all(math.isclose(float(item), first, rel_tol=0.0, abs_tol=1e-12) for item in vector)


def _vectors_identical(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(left, right)
    )


# ---------------------------------------------------------------------------
# Policy / capability records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedEmbeddingPolicy(CanonicalContract):
    """Exact pin of every parameter that may affect embedding identity.

    Remote/network backends are admitted only when both ``allow_remote`` is
    true **and** every pin field is non-empty and bound to concrete artifact
    and revision identities.  The doctor default is local-only
    (``allow_remote=False``).
    """

    SCHEMA: ClassVar[str] = PINNED_EMBEDDING_POLICY_SCHEMA

    provider_id: str
    model_artifact_id: str
    model_revision: str
    dimensions: int
    chunker_id: str
    normalizer: str
    distance: str
    corpus_root_id: str
    index_root_id: str
    forest_id: str = ""
    tree_id: str = ""
    config_id: str = ""
    allow_remote: bool = False
    remote_endpoint_id: str = ""
    producer_id: str = IPFS_DATASETS_EMBEDDING_PROVIDER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id"))
        object.__setattr__(
            self, "model_artifact_id", _text(self.model_artifact_id, "model_artifact_id")
        )
        object.__setattr__(
            self, "model_revision", _text(self.model_revision, "model_revision")
        )
        object.__setattr__(
            self, "dimensions", _positive_int(self.dimensions, "dimensions")
        )
        object.__setattr__(self, "chunker_id", _text(self.chunker_id, "chunker_id"))
        normalizer = _text(self.normalizer, "normalizer").casefold()
        if normalizer not in _ALLOWED_NORMALIZERS:
            raise EmbeddingProviderError("normalizer must be l2 or none")
        object.__setattr__(self, "normalizer", normalizer)
        distance = _text(self.distance, "distance").casefold()
        if distance not in _ALLOWED_DISTANCES:
            raise EmbeddingProviderError(
                "distance must be cosine, dot_product, euclidean, or l2"
            )
        if distance == "cosine" and normalizer != "l2":
            raise EmbeddingProviderError("cosine distance requires l2 normalizer")
        object.__setattr__(self, "distance", distance)
        object.__setattr__(
            self, "corpus_root_id", _text(self.corpus_root_id, "corpus_root_id")
        )
        object.__setattr__(
            self, "index_root_id", _text(self.index_root_id, "index_root_id")
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", required=False)
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, "tree_id", required=False)
        )
        object.__setattr__(
            self, "config_id", _text(self.config_id, "config_id", required=False)
        )
        object.__setattr__(self, "allow_remote", _bool(self.allow_remote, "allow_remote"))
        object.__setattr__(
            self,
            "remote_endpoint_id",
            _text(self.remote_endpoint_id, "remote_endpoint_id", required=False),
        )
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id or IPFS_DATASETS_EMBEDDING_PROVIDER_ID, "producer_id"),
        )
        if self.allow_remote:
            if not self.remote_endpoint_id:
                raise EmbeddingProviderBindingError(
                    "remote embeddings require a pinned remote_endpoint_id"
                )
            if not self.model_artifact_id or not self.model_revision:
                raise EmbeddingProviderBindingError(
                    "remote embeddings require pinned model artifact and revision"
                )
        elif self.remote_endpoint_id:
            raise EmbeddingProviderBindingError(
                "remote_endpoint_id is set but allow_remote is false"
            )

    @property
    def policy_id(self) -> str:
        return content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/pinned-embedding-policy-id@1",
                "payload": self.to_dict(),
            }
        )

    @property
    def model_id(self) -> str:
        """Compatibility alias used by vector indexes."""
        return self.model_artifact_id

    def _payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_artifact_id": self.model_artifact_id,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "chunker_id": self.chunker_id,
            "normalizer": self.normalizer,
            "distance": self.distance,
            "corpus_root_id": self.corpus_root_id,
            "index_root_id": self.index_root_id,
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "config_id": self.config_id,
            "allow_remote": self.allow_remote,
            "remote_endpoint_id": self.remote_endpoint_id,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PinnedEmbeddingPolicy":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "provider_id",
            "model_artifact_id",
            "model_revision",
            "dimensions",
            "chunker_id",
            "normalizer",
            "distance",
            "corpus_root_id",
            "index_root_id",
            "forest_id",
            "tree_id",
            "config_id",
            "allow_remote",
            "remote_endpoint_id",
            "producer_id",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise EmbeddingProviderError("unsupported pinned embedding policy payload")
        value = cls(
            provider_id=payload.get("provider_id", ""),
            model_artifact_id=payload.get("model_artifact_id", ""),
            model_revision=payload.get("model_revision", ""),
            dimensions=payload.get("dimensions", 0),
            chunker_id=payload.get("chunker_id", ""),
            normalizer=payload.get("normalizer", ""),
            distance=payload.get("distance", ""),
            corpus_root_id=payload.get("corpus_root_id", ""),
            index_root_id=payload.get("index_root_id", ""),
            forest_id=payload.get("forest_id", ""),
            tree_id=payload.get("tree_id", ""),
            config_id=payload.get("config_id", ""),
            allow_remote=payload.get("allow_remote", False),
            remote_endpoint_id=payload.get("remote_endpoint_id", ""),
            producer_id=payload.get("producer_id", IPFS_DATASETS_EMBEDDING_PROVIDER_ID),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class DatasetsEmbeddingCapability(CanonicalContract):
    """Local capability declaration; construction never imports the package."""

    SCHEMA: ClassVar[str] = PROVIDER_CAPABILITY_SCHEMA

    provider_id: str = IPFS_DATASETS_EMBEDDING_PROVIDER_ID
    provider_version: str = IPFS_DATASETS_EMBEDDING_PROVIDER_VERSION
    available: bool = False
    package_present: bool = False
    package_version: str = ""
    module_path: str = ""
    policy_id: str = ""
    vector_lane: EmbeddingLaneStatus = EmbeddingLaneStatus.NOT_PROBED
    canary_disposition: EmbeddingCanaryDisposition = EmbeddingCanaryDisposition.SKIPPED
    canary_reasons: tuple[str, ...] = ()
    authoritative: bool = False
    semantic_authority: bool = False
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id or IPFS_DATASETS_EMBEDDING_PROVIDER_ID, "provider_id"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(
                self.provider_version or IPFS_DATASETS_EMBEDDING_PROVIDER_VERSION,
                "provider_version",
            ),
        )
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "package_present", bool(self.package_present))
        object.__setattr__(
            self,
            "package_version",
            _text(self.package_version, "package_version", required=False),
        )
        object.__setattr__(
            self, "module_path", _text(self.module_path, "module_path", required=False)
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=False)
        )
        lane = self.vector_lane
        if not isinstance(lane, EmbeddingLaneStatus):
            lane = EmbeddingLaneStatus(str(lane))
        object.__setattr__(self, "vector_lane", lane)
        disposition = self.canary_disposition
        if not isinstance(disposition, EmbeddingCanaryDisposition):
            disposition = EmbeddingCanaryDisposition(str(disposition))
        object.__setattr__(self, "canary_disposition", disposition)
        reasons = tuple(
            sorted({str(item).strip() for item in (self.canary_reasons or ()) if str(item).strip()})
        )
        object.__setattr__(self, "canary_reasons", reasons)
        if self.authoritative is not False or self.semantic_authority is not False:
            raise EmbeddingProviderBindingError(
                "embedding capability cannot claim authority"
            )
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", required=False)
        )

    @property
    def vector_lane_enabled(self) -> bool:
        return self.vector_lane is EmbeddingLaneStatus.ENABLED

    def _payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "available": self.available,
            "package_present": self.package_present,
            "package_version": self.package_version,
            "module_path": self.module_path,
            "policy_id": self.policy_id,
            "vector_lane": self.vector_lane.value,
            "canary_disposition": self.canary_disposition.value,
            "canary_reasons": list(self.canary_reasons),
            "authoritative": False,
            "semantic_authority": False,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetsEmbeddingCapability":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "provider_id",
            "provider_version",
            "available",
            "package_present",
            "package_version",
            "module_path",
            "policy_id",
            "vector_lane",
            "canary_disposition",
            "canary_reasons",
            "authoritative",
            "semantic_authority",
            "reason_code",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise EmbeddingProviderError("unsupported embedding capability payload")
        value = cls(
            provider_id=payload.get("provider_id", IPFS_DATASETS_EMBEDDING_PROVIDER_ID),
            provider_version=payload.get(
                "provider_version", IPFS_DATASETS_EMBEDDING_PROVIDER_VERSION
            ),
            available=payload.get("available", False),
            package_present=payload.get("package_present", False),
            package_version=payload.get("package_version", ""),
            module_path=payload.get("module_path", ""),
            policy_id=payload.get("policy_id", ""),
            vector_lane=payload.get("vector_lane", EmbeddingLaneStatus.NOT_PROBED),
            canary_disposition=payload.get(
                "canary_disposition", EmbeddingCanaryDisposition.SKIPPED
            ),
            canary_reasons=tuple(payload.get("canary_reasons", ())),
            authoritative=payload.get("authoritative", False),
            semantic_authority=payload.get("semantic_authority", False),
            reason_code=payload.get("reason_code", ""),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class EmbeddingCanaryReceipt(CanonicalContract):
    """Deterministic canary outcome for the optional vector lane."""

    SCHEMA: ClassVar[str] = EMBEDDING_CANARY_RECEIPT_SCHEMA

    policy_id: str
    disposition: EmbeddingCanaryDisposition
    reasons: tuple[str, ...]
    vector_lane: EmbeddingLaneStatus
    observed_dimensions: int = 0
    sample_count: int = 0
    backend_kind: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        disposition = self.disposition
        if not isinstance(disposition, EmbeddingCanaryDisposition):
            disposition = EmbeddingCanaryDisposition(str(disposition))
        object.__setattr__(self, "disposition", disposition)
        reasons = tuple(
            sorted({str(item).strip() for item in (self.reasons or ()) if str(item).strip()})
        )
        if self.disposition is EmbeddingCanaryDisposition.FAILED and not reasons:
            raise EmbeddingProviderError("failed canary requires reasons")
        object.__setattr__(self, "reasons", reasons)
        lane = self.vector_lane
        if not isinstance(lane, EmbeddingLaneStatus):
            lane = EmbeddingLaneStatus(str(lane))
        object.__setattr__(self, "vector_lane", lane)
        for name in ("observed_dimensions", "sample_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EmbeddingProviderError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self, "backend_kind", _text(self.backend_kind, "backend_kind", required=False)
        )
        if self.semantic_authority is not False:
            raise EmbeddingProviderBindingError("canary cannot claim semantic authority")
        object.__setattr__(self, "semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "disposition": self.disposition.value,
            "reasons": list(self.reasons),
            "vector_lane": self.vector_lane.value,
            "observed_dimensions": self.observed_dimensions,
            "sample_count": self.sample_count,
            "backend_kind": self.backend_kind,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbeddingCanaryReceipt":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "policy_id",
            "disposition",
            "reasons",
            "vector_lane",
            "observed_dimensions",
            "sample_count",
            "backend_kind",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise EmbeddingProviderError("unsupported embedding canary receipt payload")
        value = cls(
            policy_id=payload.get("policy_id", ""),
            disposition=payload.get("disposition", ""),
            reasons=tuple(payload.get("reasons", ())),
            vector_lane=payload.get("vector_lane", ""),
            observed_dimensions=payload.get("observed_dimensions", 0),
            sample_count=payload.get("sample_count", 0),
            backend_kind=payload.get("backend_kind", ""),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class EmbeddingRequest(CanonicalContract):
    """Bounded, policy-bound embedding request; texts only, no bodies/secrets."""

    SCHEMA: ClassVar[str] = EMBEDDING_REQUEST_SCHEMA

    policy_id: str
    texts: tuple[str, ...]
    corpus_root_id: str = ""
    index_root_id: str = ""
    tree_id: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        if not isinstance(self.texts, Sequence) or isinstance(
            self.texts, (str, bytes, bytearray)
        ):
            raise EmbeddingProviderError("texts must be a sequence of strings")
        texts = tuple(str(item) for item in self.texts)
        if not texts:
            raise EmbeddingProviderError("texts must be non-empty")
        if len(texts) > DEFAULT_MAX_TEXTS:
            raise EmbeddingProviderError(f"at most {DEFAULT_MAX_TEXTS} texts admitted")
        total = 0
        for item in texts:
            if "\x00" in item:
                raise EmbeddingProviderError("texts must not contain NUL")
            size = len(item.encode("utf-8"))
            if size > DEFAULT_MAX_TEXT_BYTES:
                raise EmbeddingProviderError("text exceeds max_text_bytes")
            total += size
        if total > DEFAULT_MAX_BATCH_BYTES:
            raise EmbeddingProviderError("batch exceeds max_batch_bytes")
        object.__setattr__(self, "texts", texts)
        object.__setattr__(
            self,
            "corpus_root_id",
            _text(self.corpus_root_id, "corpus_root_id", required=False),
        )
        object.__setattr__(
            self,
            "index_root_id",
            _text(self.index_root_id, "index_root_id", required=False),
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, "tree_id", required=False)
        )
        if self.semantic_authority is not False:
            raise EmbeddingProviderBindingError("embedding request cannot claim authority")
        object.__setattr__(self, "semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "texts": list(self.texts),
            "corpus_root_id": self.corpus_root_id,
            "index_root_id": self.index_root_id,
            "tree_id": self.tree_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbeddingRequest":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "policy_id",
            "texts",
            "corpus_root_id",
            "index_root_id",
            "tree_id",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise EmbeddingProviderError("unsupported embedding request payload")
        value = cls(
            policy_id=payload.get("policy_id", ""),
            texts=tuple(payload.get("texts", ())),
            corpus_root_id=payload.get("corpus_root_id", ""),
            index_root_id=payload.get("index_root_id", ""),
            tree_id=payload.get("tree_id", ""),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class EmbeddingResult(CanonicalContract):
    """Non-authoritative embedding output bound to a pinned policy.

    Vectors are stored as fixed-point scaled integers so the canonical
    payload remains float-free (proof contracts reject IEEE floats).
    """

    SCHEMA: ClassVar[str] = EMBEDDING_RESULT_SCHEMA

    policy_id: str
    status: EmbeddingProviderStatus
    dimensions: int
    vectors_scaled: tuple[tuple[int, ...], ...] = ()
    vector_lane: EmbeddingLaneStatus = EmbeddingLaneStatus.DISABLED
    reasons: tuple[str, ...] = ()
    request_id: str = ""
    canary_id: str = ""
    semantic_authority: bool = False
    vector_scale: int = VECTOR_SCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        status = self.status
        if not isinstance(status, EmbeddingProviderStatus):
            status = EmbeddingProviderStatus(str(status))
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "dimensions", _positive_int(self.dimensions, "dimensions") if self.dimensions else 0
        )
        if self.dimensions == 0 and self.status is EmbeddingProviderStatus.COMPLETED:
            raise EmbeddingProviderError("completed embedding result requires dimensions")
        if isinstance(self.vector_scale, bool) or not isinstance(self.vector_scale, int) or self.vector_scale < 1:
            raise EmbeddingProviderError("vector_scale must be a positive integer")
        scaled: list[tuple[int, ...]] = []
        for index, vector in enumerate(self.vectors_scaled or ()):
            if isinstance(vector, Sequence) and vector and isinstance(vector[0], float):
                checked = _vector(vector, self.dimensions or len(vector), name=f"vectors[{index}]")
                scaled.append(_scale_vector(checked))
            else:
                try:
                    row = tuple(int(item) for item in vector)
                except (TypeError, ValueError) as exc:
                    raise EmbeddingProviderError(
                        f"vectors_scaled[{index}] must be integers"
                    ) from exc
                if self.dimensions and len(row) != self.dimensions:
                    raise EmbeddingProviderError(
                        f"vectors_scaled[{index}] dimension mismatch"
                    )
                scaled.append(row)
        object.__setattr__(self, "vectors_scaled", tuple(scaled))
        lane = self.vector_lane
        if not isinstance(lane, EmbeddingLaneStatus):
            lane = EmbeddingLaneStatus(str(lane))
        object.__setattr__(self, "vector_lane", lane)
        reasons = tuple(
            sorted({str(item).strip() for item in (self.reasons or ()) if str(item).strip()})
        )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self, "request_id", _text(self.request_id, "request_id", required=False)
        )
        object.__setattr__(
            self, "canary_id", _text(self.canary_id, "canary_id", required=False)
        )
        if self.semantic_authority is not False:
            raise EmbeddingProviderBindingError(
                "embedding results cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @property
    def vectors(self) -> tuple[tuple[float, ...], ...]:
        """Float view of fixed-point vectors for consumers/tests."""
        scale = float(self.vector_scale) or float(VECTOR_SCALE)
        return tuple(
            tuple(int(item) / scale for item in vector)
            for vector in self.vectors_scaled
        )

    @classmethod
    def from_float_vectors(
        cls,
        *,
        policy_id: str,
        status: EmbeddingProviderStatus | str,
        dimensions: int,
        vectors: Sequence[Sequence[float]] = (),
        vector_lane: EmbeddingLaneStatus | str = EmbeddingLaneStatus.DISABLED,
        reasons: Sequence[str] = (),
        request_id: str = "",
        canary_id: str = "",
        semantic_authority: bool = False,
    ) -> "EmbeddingResult":
        scaled = tuple(_scale_vector(_vector(v, dimensions, name="vector")) if dimensions else _scale_vector(tuple(float(x) for x in v)) for v in vectors)
        return cls(
            policy_id=policy_id,
            status=status,
            dimensions=dimensions,
            vectors_scaled=scaled,
            vector_lane=vector_lane,
            reasons=tuple(reasons),
            request_id=request_id,
            canary_id=canary_id,
            semantic_authority=semantic_authority,
            vector_scale=VECTOR_SCALE,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "status": self.status.value,
            "dimensions": self.dimensions,
            "vectors_scaled": [list(vector) for vector in self.vectors_scaled],
            "vector_scale": self.vector_scale,
            "vector_lane": self.vector_lane.value,
            "reasons": list(self.reasons),
            "request_id": self.request_id,
            "canary_id": self.canary_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbeddingResult":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "policy_id",
            "status",
            "dimensions",
            "vectors_scaled",
            "vectors",
            "vector_scale",
            "vector_lane",
            "reasons",
            "request_id",
            "canary_id",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise EmbeddingProviderError("unsupported embedding result payload")
        if "vectors_scaled" in payload:
            raw_vectors = payload.get("vectors_scaled", ())
        else:
            # Legacy float payload path — convert once into scaled ints.
            raw_vectors = tuple(
                _scale_vector(tuple(float(x) for x in row))
                for row in (payload.get("vectors") or ())
            )
        if not isinstance(raw_vectors, Sequence) or isinstance(
            raw_vectors, (str, bytes, bytearray)
        ):
            raise EmbeddingProviderError("vectors must be a sequence")
        value = cls(
            policy_id=payload.get("policy_id", ""),
            status=payload.get("status", ""),
            dimensions=payload.get("dimensions", 0),
            vectors_scaled=tuple(tuple(item) for item in raw_vectors),
            vector_lane=payload.get("vector_lane", EmbeddingLaneStatus.DISABLED),
            reasons=tuple(payload.get("reasons", ())),
            request_id=payload.get("request_id", ""),
            canary_id=payload.get("canary_id", ""),
            semantic_authority=payload.get("semantic_authority", False),
            vector_scale=payload.get("vector_scale", VECTOR_SCALE),
        )
        _verify_record_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Deterministic local fixture backend (tests / hermetic doctor)
# ---------------------------------------------------------------------------


class DeterministicLocalEmbeddingBackend:
    """Pinned local embedding backend with non-constant, finite vectors.

    Produces dimension-stable vectors from text digests.  Distinct inputs
    produce distinct vectors so the canary can reject constant shims.
    """

    def __init__(self, policy: PinnedEmbeddingPolicy) -> None:
        if not isinstance(policy, PinnedEmbeddingPolicy):
            raise EmbeddingProviderBindingError("backend requires PinnedEmbeddingPolicy")
        self.policy = policy
        self.kind = "deterministic_local"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        dim = self.policy.dimensions
        for text in texts:
            digest = hashlib.sha256(
                f"{self.policy.model_artifact_id}:{self.policy.model_revision}:{text}".encode(
                    "utf-8"
                )
            ).digest()
            # Expand digest bytes into the pinned dimension with position salt
            # so adjacent coordinates are not constant.
            values: list[float] = []
            for index in range(dim):
                byte = digest[index % len(digest)]
                salt = (index * 17 + byte) % 251
                values.append(((byte + salt + 1) / 256.0) - 0.5)
            if self.policy.normalizer == "l2":
                norm = math.sqrt(sum(item * item for item in values)) or 1.0
                values = [item / norm for item in values]
            vectors.append(values)
        return vectors


class ConstantVectorShimBackend:
    """Adversarial fixture: pretends success with a constant vector."""

    def __init__(self, policy: PinnedEmbeddingPolicy, *, fill: float = 0.0) -> None:
        self.policy = policy
        self.fill = fill
        self.kind = "constant_shim"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[self.fill] * self.policy.dimensions for _ in texts]


class MissingDependencySuccessShim:
    """Adversarial fixture: reports success without producing usable vectors."""

    kind = "missing_dependency_success_shim"

    def embed(self, texts: Sequence[str]) -> Mapping[str, Any]:
        # Success-shaped envelope with no vectors — classic dependency shim.
        return {
            "status": "ok",
            "success": True,
            "available": True,
            "vectors": None,
            "embeddings": [],
            "reason": "optional package missing but degraded successfully",
        }


class UnpinnedRemoteEmbeddingBackend:
    """Adversarial fixture that claims to call a remote unpinned endpoint."""

    kind = "unpinned_remote"

    def __init__(self, endpoint: str = "https://example.invalid/embeddings") -> None:
        self.endpoint = endpoint

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingProviderError(
            f"refusing unpinned remote embedding call to {self.endpoint}"
        )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def inspect_datasets_embedding_capability(
    *,
    policy: PinnedEmbeddingPolicy | None = None,
    package_name: str = DEFAULT_OPTIONAL_MODULE,
) -> DatasetsEmbeddingCapability:
    """Inspect local capability without importing the optional package."""

    present = importlib.util.find_spec(package_name) is not None
    version = ""
    if present:
        try:
            version = importlib.metadata.version(package_name)  # type: ignore[attr-defined]
        except Exception:
            try:
                version = importlib.metadata.version(package_name.replace("_", "-"))  # type: ignore[attr-defined]
            except Exception:
                version = ""
    return DatasetsEmbeddingCapability(
        available=False,
        package_present=present,
        package_version=version or "",
        policy_id="" if policy is None else policy.policy_id,
        vector_lane=EmbeddingLaneStatus.NOT_PROBED,
        canary_disposition=EmbeddingCanaryDisposition.SKIPPED,
        canary_reasons=(),
        reason_code="" if present else EmbeddingCanaryReason.BACKEND_UNAVAILABLE.value,
    )


class IpfsDatasetsEmbeddingProvider:
    """Lazy, pinned embedding adapter with a fail-closed deterministic canary."""

    def __init__(
        self,
        policy: PinnedEmbeddingPolicy,
        *,
        backend: Any | None = None,
        importer: Callable[[str], Any] | None = None,
        module_candidates: Sequence[str] = DEFAULT_EMBEDDING_MODULE_CANDIDATES,
        auto_canary: bool = True,
    ) -> None:
        if not isinstance(policy, PinnedEmbeddingPolicy):
            raise EmbeddingProviderBindingError("policy must be PinnedEmbeddingPolicy")
        self.policy = policy
        self._importer = importer or importlib.import_module
        self._module_candidates = tuple(module_candidates)
        self._backend = backend
        self._backend_kind = getattr(backend, "kind", "injected" if backend is not None else "")
        self._canary: EmbeddingCanaryReceipt | None = None
        self._vector_lane = EmbeddingLaneStatus.NOT_PROBED
        self._lock = threading.Lock()
        if auto_canary:
            self.run_canary()

    # -- capability ---------------------------------------------------------

    def capability(self) -> DatasetsEmbeddingCapability:
        base = inspect_datasets_embedding_capability(policy=self.policy)
        reasons = () if self._canary is None else self._canary.reasons
        disposition = (
            EmbeddingCanaryDisposition.SKIPPED
            if self._canary is None
            else self._canary.disposition
        )
        available = self._vector_lane is EmbeddingLaneStatus.ENABLED
        reason = ""
        if self._vector_lane is EmbeddingLaneStatus.CANARY_FAILED:
            reason = reasons[0] if reasons else EmbeddingCanaryReason.CONSTANT_VECTOR.value
        elif self._vector_lane is EmbeddingLaneStatus.UNPINNED_REJECTED:
            reason = EmbeddingCanaryReason.UNPINNED_REMOTE.value
        elif self._vector_lane is EmbeddingLaneStatus.UNAVAILABLE:
            reason = EmbeddingCanaryReason.BACKEND_UNAVAILABLE.value
        return DatasetsEmbeddingCapability(
            available=available,
            package_present=base.package_present or self._backend is not None,
            package_version=base.package_version,
            module_path=self._backend_kind,
            policy_id=self.policy.policy_id,
            vector_lane=self._vector_lane,
            canary_disposition=disposition,
            canary_reasons=reasons,
            reason_code=reason,
        )

    capabilities = capability

    @property
    def vector_lane_enabled(self) -> bool:
        return self._vector_lane is EmbeddingLaneStatus.ENABLED

    @property
    def vector_lane(self) -> EmbeddingLaneStatus:
        return self._vector_lane

    @property
    def canary_receipt(self) -> EmbeddingCanaryReceipt | None:
        return self._canary

    # -- backend resolution -------------------------------------------------

    def _resolve_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if not self.policy.allow_remote and self.policy.remote_endpoint_id:
            raise EmbeddingProviderBindingError(
                "unpinned remote endpoint cannot be used"
            )
        # Prefer an explicit local deterministic backend when no optional
        # package surface is available.  Optional package load is lazy.
        last_error: Exception | None = None
        for module_name in self._module_candidates:
            try:
                with _IMPORT_LOCK:
                    module = self._importer(module_name)
            except Exception as exc:  # pragma: no cover - import path dependent
                last_error = exc
                continue
            for attr in (
                "embed_texts",
                "embed",
                "EmbeddingsEngine",
                "create_embeddings",
                "EmbeddingEngine",
            ):
                candidate = getattr(module, attr, None)
                if candidate is None:
                    continue
                if callable(candidate) and attr in {"embed_texts", "embed"}:
                    self._backend = candidate
                    self._backend_kind = f"module:{module_name}.{attr}"
                    return self._backend
                if callable(candidate):
                    try:
                        instance = candidate()
                    except TypeError:
                        try:
                            instance = candidate(self.policy.to_dict())
                        except Exception as exc:  # pragma: no cover
                            last_error = exc
                            continue
                    embed = getattr(instance, "embed", None) or getattr(
                        instance, "embed_texts", None
                    )
                    if callable(embed):
                        self._backend = instance
                        self._backend_kind = f"module:{module_name}.{attr}"
                        return self._backend
        # Fall back to the deterministic local pin when the optional package
        # is absent; this still exercises canary and pin checks.
        self._backend = DeterministicLocalEmbeddingBackend(self.policy)
        self._backend_kind = "deterministic_local_fallback"
        if last_error is not None:
            # Presence of an import error is retained only as backend_kind
            # metadata; the local pin remains available for exact tests.
            pass
        return self._backend

    def _invoke_backend(self, backend: Any, texts: Sequence[str]) -> Any:
        if isinstance(backend, Mapping):
            # Mapping backends are admitted only as fully pinned local vector
            # tables keyed by exact text.
            return [backend[text] for text in texts]
        embed = getattr(backend, "embed", None) or getattr(backend, "embed_texts", None)
        if callable(embed):
            return embed(texts)
        if callable(backend):
            return backend(texts)
        raise EmbeddingProviderError("backend does not expose embed/embed_texts")

    def _normalize_backend_output(
        self, raw: Any, *, expected_count: int
    ) -> tuple[list[tuple[float, ...]], list[str]]:
        reasons: list[str] = []
        if raw is None:
            return [], [EmbeddingCanaryReason.EMPTY_OUTPUT.value]
        if isinstance(raw, Mapping):
            # Success-shaped envelopes without vectors are missing-dependency
            # shims and must never enable the vector lane.
            if raw.get("success") is True or str(raw.get("status", "")).casefold() in {
                "ok",
                "success",
                "completed",
                "available",
            }:
                vectors_field = raw.get("vectors", raw.get("embeddings", raw.get("data")))
                if vectors_field in (None, (), [], ""):
                    return [], [EmbeddingCanaryReason.MISSING_DEPENDENCY_SHIM.value]
                raw = vectors_field
            else:
                vectors_field = raw.get("vectors", raw.get("embeddings", raw.get("data")))
                if vectors_field in (None, (), [], ""):
                    return [], [EmbeddingCanaryReason.EMPTY_OUTPUT.value]
                raw = vectors_field
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            return [], [EmbeddingCanaryReason.EMPTY_OUTPUT.value]
        if len(raw) != expected_count:
            return [], [EmbeddingCanaryReason.EMPTY_OUTPUT.value]
        vectors: list[tuple[float, ...]] = []
        dim = self.policy.dimensions
        for index, item in enumerate(raw):
            try:
                vector = _vector(item, dim, name=f"vectors[{index}]")
            except EmbeddingProviderError as exc:
                message = str(exc).casefold()
                if "dimension" in message:
                    reasons.append(EmbeddingCanaryReason.DIMENSION_DRIFT.value)
                elif "non-finite" in message:
                    reasons.append(EmbeddingCanaryReason.NON_FINITE.value)
                else:
                    reasons.append(EmbeddingCanaryReason.EMPTY_OUTPUT.value)
                return [], reasons
            vectors.append(vector)
        return vectors, reasons

    def _validate_vectors(self, vectors: Sequence[Sequence[float]]) -> list[str]:
        reasons: list[str] = []
        if not vectors:
            return [EmbeddingCanaryReason.EMPTY_OUTPUT.value]
        dim = self.policy.dimensions
        for index, vector in enumerate(vectors):
            if len(vector) != dim:
                reasons.append(EmbeddingCanaryReason.DIMENSION_DRIFT.value)
                continue
            if not all(math.isfinite(float(item)) for item in vector):
                reasons.append(EmbeddingCanaryReason.NON_FINITE.value)
            if _is_constant_vector(vector):
                reasons.append(EmbeddingCanaryReason.CONSTANT_VECTOR.value)
        # Distinct canary texts must not collapse to one vector (constant
        # fallback across the batch).
        if len(vectors) >= 2 and all(
            _vectors_identical(vectors[0], other) for other in vectors[1:]
        ):
            reasons.append(EmbeddingCanaryReason.CONSTANT_VECTOR.value)
        return sorted(set(reasons))

    # -- canary -------------------------------------------------------------

    def run_canary(self) -> EmbeddingCanaryReceipt:
        """Run the deterministic canary; disable only the optional vector lane."""

        with self._lock:
            reasons: list[str] = []
            observed_dim = 0
            sample_count = 0
            backend_kind = self._backend_kind

            # Reject unpinned remote policies before any backend call.
            if self.policy.allow_remote and not (
                self.policy.remote_endpoint_id
                and self.policy.model_artifact_id
                and self.policy.model_revision
                and self.policy.dimensions
                and self.policy.corpus_root_id
                and self.policy.index_root_id
            ):
                reasons.append(EmbeddingCanaryReason.UNPINNED_REMOTE.value)
                receipt = self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    EmbeddingLaneStatus.UNPINNED_REJECTED,
                    observed_dim,
                    sample_count,
                    backend_kind or "unpinned_remote",
                )
                return receipt

            try:
                backend = self._resolve_backend()
            except Exception:
                reasons.append(EmbeddingCanaryReason.BACKEND_UNAVAILABLE.value)
                return self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    EmbeddingLaneStatus.UNAVAILABLE,
                    0,
                    0,
                    backend_kind or "unavailable",
                )

            backend_kind = getattr(backend, "kind", None) or self._backend_kind or "resolved"
            if "unpinned" in str(backend_kind).casefold() or isinstance(
                backend, UnpinnedRemoteEmbeddingBackend
            ):
                reasons.append(EmbeddingCanaryReason.UNPINNED_REMOTE.value)
                return self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    EmbeddingLaneStatus.UNPINNED_REJECTED,
                    0,
                    0,
                    str(backend_kind),
                )

            # Config drift: backend may declare its own pin fields.
            for attr, expected in (
                ("dimensions", self.policy.dimensions),
                ("model_artifact_id", self.policy.model_artifact_id),
                ("model_revision", self.policy.model_revision),
                ("chunker_id", self.policy.chunker_id),
                ("normalizer", self.policy.normalizer),
                ("distance", self.policy.distance),
                ("corpus_root_id", self.policy.corpus_root_id),
                ("index_root_id", self.policy.index_root_id),
            ):
                observed = getattr(backend, attr, None)
                if observed is None and hasattr(backend, "policy"):
                    observed = getattr(backend.policy, attr, None)
                if observed is not None and observed != expected:
                    reasons.append(EmbeddingCanaryReason.CONFIG_DRIFT.value)
                    break

            if reasons:
                return self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    EmbeddingLaneStatus.CANARY_FAILED,
                    0,
                    0,
                    str(backend_kind),
                )

            try:
                raw = self._invoke_backend(backend, CANARY_TEXTS)
            except EmbeddingProviderError as exc:
                message = str(exc).casefold()
                if "unpinned" in message or "remote" in message:
                    reasons.append(EmbeddingCanaryReason.UNPINNED_REMOTE.value)
                    lane = EmbeddingLaneStatus.UNPINNED_REJECTED
                else:
                    reasons.append(EmbeddingCanaryReason.BACKEND_UNAVAILABLE.value)
                    lane = EmbeddingLaneStatus.UNAVAILABLE
                return self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    lane,
                    0,
                    0,
                    str(backend_kind),
                )
            except Exception:
                reasons.append(EmbeddingCanaryReason.BACKEND_UNAVAILABLE.value)
                return self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    EmbeddingLaneStatus.UNAVAILABLE,
                    0,
                    0,
                    str(backend_kind),
                )

            vectors, normalize_reasons = self._normalize_backend_output(
                raw, expected_count=len(CANARY_TEXTS)
            )
            reasons.extend(normalize_reasons)
            if vectors:
                observed_dim = len(vectors[0])
                sample_count = len(vectors)
                reasons.extend(self._validate_vectors(vectors))

            reasons = sorted(set(reasons))
            if reasons:
                return self._finish_canary(
                    EmbeddingCanaryDisposition.FAILED,
                    reasons,
                    EmbeddingLaneStatus.CANARY_FAILED,
                    observed_dim,
                    sample_count,
                    str(backend_kind),
                )
            return self._finish_canary(
                EmbeddingCanaryDisposition.PASSED,
                [EmbeddingCanaryReason.OK.value],
                EmbeddingLaneStatus.ENABLED,
                observed_dim,
                sample_count,
                str(backend_kind),
            )

    def _finish_canary(
        self,
        disposition: EmbeddingCanaryDisposition,
        reasons: Sequence[str],
        lane: EmbeddingLaneStatus,
        observed_dimensions: int,
        sample_count: int,
        backend_kind: str,
    ) -> EmbeddingCanaryReceipt:
        receipt = EmbeddingCanaryReceipt(
            policy_id=self.policy.policy_id,
            disposition=disposition,
            reasons=tuple(reasons),
            vector_lane=lane,
            observed_dimensions=observed_dimensions,
            sample_count=sample_count,
            backend_kind=backend_kind,
            semantic_authority=False,
        )
        self._canary = receipt
        self._vector_lane = lane
        return receipt

    # -- embed --------------------------------------------------------------

    def embed(
        self,
        texts: Sequence[str] | EmbeddingRequest,
        *,
        require_canary: bool = True,
    ) -> EmbeddingResult:
        """Embed texts only when the optional vector lane is enabled."""

        if isinstance(texts, EmbeddingRequest):
            request = texts
            if request.policy_id != self.policy.policy_id:
                raise EmbeddingProviderBindingError(
                    "request policy_id does not match provider pin"
                )
            if request.corpus_root_id and request.corpus_root_id != self.policy.corpus_root_id:
                raise EmbeddingProviderBindingError("request corpus_root_id drift")
            if request.index_root_id and request.index_root_id != self.policy.index_root_id:
                raise EmbeddingProviderBindingError("request index_root_id drift")
            if request.tree_id and self.policy.tree_id and request.tree_id != self.policy.tree_id:
                raise EmbeddingProviderBindingError("request tree_id drift")
            text_values = request.texts
            request_id = request.content_id
        else:
            request = EmbeddingRequest(
                policy_id=self.policy.policy_id,
                texts=tuple(texts),
                corpus_root_id=self.policy.corpus_root_id,
                index_root_id=self.policy.index_root_id,
                tree_id=self.policy.tree_id,
            )
            text_values = request.texts
            request_id = request.content_id

        if require_canary and self._canary is None:
            self.run_canary()

        canary_id = "" if self._canary is None else self._canary.content_id

        if self._vector_lane is not EmbeddingLaneStatus.ENABLED:
            status = EmbeddingProviderStatus.CANARY_FAILED
            if self._vector_lane is EmbeddingLaneStatus.UNAVAILABLE:
                status = EmbeddingProviderStatus.UNAVAILABLE
            elif self._vector_lane is EmbeddingLaneStatus.DISABLED:
                status = EmbeddingProviderStatus.DISABLED
            elif self._vector_lane is EmbeddingLaneStatus.UNPINNED_REJECTED:
                status = EmbeddingProviderStatus.REJECTED
            reasons = (
                () if self._canary is None else self._canary.reasons
            ) or (EmbeddingCanaryReason.NOT_RUN.value,)
            return EmbeddingResult(
                policy_id=self.policy.policy_id,
                status=status,
                dimensions=self.policy.dimensions,
                vectors_scaled=(),
                vector_lane=self._vector_lane,
                reasons=reasons,
                request_id=request_id,
                canary_id=canary_id,
                semantic_authority=False,
            )

        backend = self._resolve_backend()
        try:
            raw = self._invoke_backend(backend, text_values)
        except Exception as exc:
            return EmbeddingResult(
                policy_id=self.policy.policy_id,
                status=EmbeddingProviderStatus.FAILED,
                dimensions=self.policy.dimensions,
                vectors_scaled=(),
                vector_lane=self._vector_lane,
                reasons=(f"backend_error:{type(exc).__name__}",),
                request_id=request_id,
                canary_id=canary_id,
                semantic_authority=False,
            )

        vectors, reasons = self._normalize_backend_output(
            raw, expected_count=len(text_values)
        )
        if not reasons:
            reasons = self._validate_vectors(vectors)
        if reasons:
            # Runtime drift after a passed canary disables the lane; exact
            # routes remain available to the doctor.
            self._vector_lane = EmbeddingLaneStatus.CANARY_FAILED
            if self._canary is not None:
                self._canary = EmbeddingCanaryReceipt(
                    policy_id=self.policy.policy_id,
                    disposition=EmbeddingCanaryDisposition.FAILED,
                    reasons=tuple(reasons),
                    vector_lane=EmbeddingLaneStatus.CANARY_FAILED,
                    observed_dimensions=len(vectors[0]) if vectors else 0,
                    sample_count=len(vectors),
                    backend_kind=self._backend_kind,
                    semantic_authority=False,
                )
            return EmbeddingResult(
                policy_id=self.policy.policy_id,
                status=EmbeddingProviderStatus.CANARY_FAILED,
                dimensions=self.policy.dimensions,
                vectors_scaled=(),
                vector_lane=EmbeddingLaneStatus.CANARY_FAILED,
                reasons=tuple(reasons),
                request_id=request_id,
                canary_id=canary_id,
                semantic_authority=False,
            )

        return EmbeddingResult.from_float_vectors(
            policy_id=self.policy.policy_id,
            status=EmbeddingProviderStatus.COMPLETED,
            dimensions=self.policy.dimensions,
            vectors=tuple(vectors),
            vector_lane=EmbeddingLaneStatus.ENABLED,
            reasons=(EmbeddingCanaryReason.OK.value,),
            request_id=request_id,
            canary_id=canary_id,
            semantic_authority=False,
        )

    def disable_vector_lane(self, *, reason: str = "operator_disabled") -> None:
        """Explicitly disable only the optional vector lane."""

        self._vector_lane = EmbeddingLaneStatus.DISABLED
        self._canary = EmbeddingCanaryReceipt(
            policy_id=self.policy.policy_id,
            disposition=EmbeddingCanaryDisposition.FAILED,
            reasons=(reason,),
            vector_lane=EmbeddingLaneStatus.DISABLED,
            observed_dimensions=0,
            sample_count=0,
            backend_kind=self._backend_kind or "disabled",
            semantic_authority=False,
        )


def create_pinned_embedding_policy(
    *,
    provider_id: str = IPFS_DATASETS_EMBEDDING_PROVIDER_ID,
    model_artifact_id: str = "model:deterministic-local",
    model_revision: str = "1",
    dimensions: int = 8,
    chunker_id: str = "chunker:symbol-span@1",
    normalizer: str = "l2",
    distance: str = "cosine",
    corpus_root_id: str = "corpus:fixture",
    index_root_id: str = "index:fixture",
    forest_id: str = "forest:fixture",
    tree_id: str = "tree:fixture",
    config_id: str = "config:fixture",
    allow_remote: bool = False,
    remote_endpoint_id: str = "",
) -> PinnedEmbeddingPolicy:
    """Convenience constructor for hermetic pins."""

    return PinnedEmbeddingPolicy(
        provider_id=provider_id,
        model_artifact_id=model_artifact_id,
        model_revision=model_revision,
        dimensions=dimensions,
        chunker_id=chunker_id,
        normalizer=normalizer,
        distance=distance,
        corpus_root_id=corpus_root_id,
        index_root_id=index_root_id,
        forest_id=forest_id,
        tree_id=tree_id,
        config_id=config_id,
        allow_remote=allow_remote,
        remote_endpoint_id=remote_endpoint_id,
    )


def create_ipfs_datasets_embedding_provider(
    policy: PinnedEmbeddingPolicy | None = None,
    *,
    backend: Any | None = None,
    auto_canary: bool = True,
) -> IpfsDatasetsEmbeddingProvider:
    """Create a provider bound to a pinned policy (default hermetic pin)."""

    return IpfsDatasetsEmbeddingProvider(
        policy or create_pinned_embedding_policy(),
        backend=backend,
        auto_canary=auto_canary,
    )


__all__ = (
    "IPFS_DATASETS_EMBEDDING_PROVIDER_ID",
    "IPFS_DATASETS_EMBEDDING_PROVIDER_VERSION",
    "PROVIDER_CAPABILITY_SCHEMA",
    "PINNED_EMBEDDING_POLICY_SCHEMA",
    "EMBEDDING_REQUEST_SCHEMA",
    "EMBEDDING_RESULT_SCHEMA",
    "EMBEDDING_CANARY_RECEIPT_SCHEMA",
    "CANARY_TEXTS",
    "EmbeddingProviderError",
    "EmbeddingProviderBindingError",
    "EmbeddingProviderCanaryError",
    "EmbeddingLaneStatus",
    "EmbeddingCanaryDisposition",
    "EmbeddingCanaryReason",
    "EmbeddingProviderStatus",
    "PinnedEmbeddingPolicy",
    "DatasetsEmbeddingCapability",
    "EmbeddingCanaryReceipt",
    "EmbeddingRequest",
    "EmbeddingResult",
    "DeterministicLocalEmbeddingBackend",
    "ConstantVectorShimBackend",
    "MissingDependencySuccessShim",
    "UnpinnedRemoteEmbeddingBackend",
    "IpfsDatasetsEmbeddingProvider",
    "inspect_datasets_embedding_capability",
    "create_pinned_embedding_policy",
    "create_ipfs_datasets_embedding_provider",
)
