"""Bounded ``ipfs_datasets_py`` GraphRAG/IPLD projection over program graphs.

This module is an **optional lazy projection and query adapter** over the
canonical program evidence graph owned by :mod:`program_graph` (VFS-008).
It never synthesizes calls, contracts, findings, proofs, completion, or
mutation authority.  GraphRAG may only rank neighborhoods that already exist
in the admitted canonical evidence set.

Properties:

* constructing the provider and inspecting its local capability declaration
  never imports ``ipfs_datasets_py``;
* projection emits deterministic chunk CIDs and provenance links, never a
  recursive unbounded IPLD graph dump;
* item, depth, byte, and time costs are hard-bounded;
* query responses return compact references, scores, and ranking reasons;
* missing, incompatible, partial, or poisoned optional surfaces produce an
  explicit local-fallback or inconclusive/poisoned result; and
* every result carries permanent non-authority claims.

Conflict policy: the canonical graph remains owned by VFS-008; this module is
only a ranking/indexing projection.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .program_graph import (
    DEFAULT_MAX_CHUNK_EDGES,
    DEFAULT_MAX_CHUNK_NODES,
    GraphChunk,
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphError,
    ProgramGraphNode,
    canonical_program_json,
)
from .proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Public schema / identity constants
# ---------------------------------------------------------------------------

IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID: Final = (
    "ipfs_datasets_py.program_graph_projection"
)
IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_VERSION: Final = "1.0.0"
IPFS_DATASETS_PROGRAM_GRAPH_PROTOCOL_VERSION: Final = 1

PROVIDER_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-capability@1"
)
PROVIDER_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-projection@1"
)
PROVIDER_CHUNK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-chunk-projection@1"
)
PROVIDER_QUERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-query@1"
)
PROVIDER_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-query-result@1"
)
PROVIDER_REFERENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-reference@1"
)
PROVIDER_PROVENANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-program-graph-provenance-link@1"
)

DEFAULT_OPTIONAL_ROOT: Final = "ipfs_datasets_py"
DEFAULT_MAX_ITEMS: Final = 256
DEFAULT_MAX_RESULTS: Final = 32
DEFAULT_MAX_DEPTH: Final = 3
DEFAULT_MAX_HOPS: Final = 2
DEFAULT_MAX_BYTES: Final = 128 * 1024
DEFAULT_MAX_QUERY_BYTES: Final = 16 * 1024
DEFAULT_MAX_REASON_BYTES: Final = 1024
DEFAULT_MAX_LABEL_BYTES: Final = 4096
DEFAULT_TIMEOUT_MS: Final = 5_000
DEFAULT_PROBE_TIMEOUT_MS: Final = 2_000
DEFAULT_MAX_PROVENANCE_LINKS: Final = 1_024
DEFAULT_MAX_CHUNK_COUNT: Final = 512

# Surfaces probed on the optional package.  Order is diagnostic only.
_GRAPH_API_CANDIDATES: Final = (
    "search.graphrag_integration.graphrag_integration",
    "search.graph_query",
    "knowledge_graphs",
    "utils.cid_utils",
)

_QUERY_CLASS_NAMES: Final = (
    "GraphRAGQueryEngine",
    "GraphRAGIntegration",
    "GraphQueryEngine",
    "HybridVectorGraphSearch",
)

_QUERY_BOUND_PARAMS: Final = frozenset(
    {
        "top_k",
        "max_results",
        "limit",
        "max_nodes_visited",
        "max_edges_traversed",
        "max_graph_hops",
        "max_hops",
        "timeout_ms",
        "timeout",
        "budgets",
    }
)

# Authority claims that are permanently false for every projection/query.
_AUTHORITY_FALSE: Final = MappingProxyType(
    {
        "creates_calls": False,
        "creates_contracts": False,
        "creates_findings": False,
        "creates_proofs": False,
        "completion_authority": False,
        "mutation_authority": False,
        "ranking_only": True,
        "canonical_evidence_only": True,
        "non_authoritative": True,
    }
)

_FORBIDDEN_RESULT_FIELDS: Final = frozenset(
    {
        "source",
        "source_body",
        "source_code",
        "source_text",
        "file_contents",
        "content",
        "raw",
        "raw_output",
        "decoded_output",
        "model_output",
        "model_response",
        "prompt",
        "completion",
        "transcript",
        "ast",
        "ast_body",
        "nested_graph",
        "embedding",
        "proof",
        "finding",
        "contract_body",
        "call_body",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+", re.IGNORECASE)

ModuleImporter = Callable[[str], Any]
Clock = Callable[[], float]


# ---------------------------------------------------------------------------
# Errors and closed vocabularies
# ---------------------------------------------------------------------------


class ProgramGraphProviderError(ValueError):
    """A graph-projection request, policy, or result violates the contract."""


class ProgramGraphProviderBoundsError(ProgramGraphProviderError):
    """A hard item/depth/byte/time bound was exceeded."""


class ProgramGraphProviderPoisonError(ProgramGraphProviderError):
    """Optional backend output attempted to invent or re-author evidence."""


class ProjectionMode(str, Enum):
    """How the projection was produced."""

    LOCAL_FALLBACK = "local_fallback"
    GRAPHRAG = "graphrag"
    IPLD = "ipld"
    MIXED = "mixed"


class ProjectionStatus(str, Enum):
    """Closed status vocabulary for projections and queries."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    LOCAL_FALLBACK = "local_fallback"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    INCONCLUSIVE = "inconclusive"
    POISONED = "poisoned"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DISABLED = "disabled"


class CapabilityHealth(str, Enum):
    LAZY = "lazy"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    PARTIAL = "partial"


class ReasonCode(str, Enum):
    LAZY_NOT_PROBED = "lazy_not_probed"
    PROVIDER_DISABLED = "provider_disabled"
    OPTIONAL_MODULE_UNAVAILABLE = "optional_module_unavailable"
    OPTIONAL_API_INCOMPATIBLE = "optional_api_incompatible"
    OPTIONAL_API_PARTIAL = "optional_api_partial"
    LOCAL_FALLBACK_PROJECTION = "local_fallback_projection"
    LOCAL_FALLBACK_QUERY = "local_fallback_query"
    BOUNDED_PROJECTION = "bounded_projection"
    BOUNDED_QUERY = "bounded_query"
    PARTIAL_TRUNCATION = "partial_truncation"
    POISONED_BACKEND_RESULT = "poisoned_backend_result"
    INCONCLUSIVE_BACKEND = "inconclusive_backend"
    TIMEOUT = "timeout"
    FAILED = "failed"
    DETERMINISTIC_RANKING = "deterministic_ranking"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _canonical_value(value: Any, *, name: str = "value", depth: int = 0) -> Any:
    if depth > 12:
        raise ProgramGraphProviderError(f"{name} exceeds maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProgramGraphProviderError(f"{name} must be finite")
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value, name=name, depth=depth + 1)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ProgramGraphProviderError(f"{name} object keys must be strings")
        return {
            key: _canonical_value(item, name=name, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, name=name, depth=depth + 1) for item in value
        ]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical_value(converter(), name=name, depth=depth + 1)
    raise ProgramGraphProviderError(
        f"{name} contains unsupported {type(value).__name__}"
    )


def _json_bytes(value: Any, *, name: str = "value") -> bytes:
    try:
        return json.dumps(
            _canonical_value(value, name=name),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ProgramGraphProviderError):
            raise
        raise ProgramGraphProviderError(f"{name} must be canonical JSON") from exc


def _content_id(value: Any, *, name: str) -> str:
    """Stable content address for adapter envelopes (not necessarily a CID)."""

    return f"{name}:sha256:" + hashlib.sha256(_json_bytes(value, name=name)).hexdigest()


def _chunk_cid(value: Any) -> str:
    """Deterministic CIDv1-style identity for an IPLD chunk body."""

    return content_identity(_canonical_value(value, name="chunk"))


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = DEFAULT_MAX_LABEL_BYTES,
) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ProgramGraphProviderError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ProgramGraphProviderError(f"{name} is required")
    if "\x00" in result:
        raise ProgramGraphProviderError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > max_bytes:
        raise ProgramGraphProviderError(f"{name} exceeds {max_bytes} bytes")
    return result


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ProgramGraphProviderError(
            f"{name} must be an integer between 1 and {maximum}"
        )
    return value


def _non_negative_int(value: Any, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ProgramGraphProviderError(
            f"{name} must be an integer between 0 and {maximum}"
        )
    return value


def _tokens(value: Any) -> frozenset[str]:
    text = str(value or "").casefold()
    if not text:
        return frozenset()
    return frozenset(_TOKEN_RE.findall(text))


def _authority_payload() -> dict[str, Any]:
    return dict(_AUTHORITY_FALSE)


def _find_forbidden(value: Any, *, depth: int = 0) -> str | None:
    if depth > 12:
        return "depth"
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return "non_string_key"
            if key in _FORBIDDEN_RESULT_FIELDS:
                return key
            if key in {
                "creates_calls",
                "creates_contracts",
                "creates_findings",
                "creates_proofs",
                "completion_authority",
                "mutation_authority",
            } and item is True:
                return f"authority:{key}"
            found = _find_forbidden(item, depth=depth + 1)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_forbidden(item, depth=depth + 1)
            if found:
                return found
    return None


def _signature_params(func: Any) -> frozenset[str]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise ProgramGraphProviderError(
            f"cannot inspect signature: {exc}"
        ) from exc
    return frozenset(signature.parameters)


def _has_bounds(params: frozenset[str]) -> bool:
    return bool(params & _QUERY_BOUND_PARAMS)


# ---------------------------------------------------------------------------
# Bounds, policy, and capability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphProjectionBounds:
    """Hard resource bounds for projection and query."""

    max_items: int = DEFAULT_MAX_ITEMS
    max_results: int = DEFAULT_MAX_RESULTS
    max_depth: int = DEFAULT_MAX_DEPTH
    max_hops: int = DEFAULT_MAX_HOPS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES
    max_chunk_nodes: int = DEFAULT_MAX_CHUNK_NODES
    max_chunk_edges: int = DEFAULT_MAX_CHUNK_EDGES
    max_chunk_count: int = DEFAULT_MAX_CHUNK_COUNT
    max_provenance_links: int = DEFAULT_MAX_PROVENANCE_LINKS
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def __post_init__(self) -> None:
        limits = {
            "max_items": 10_000,
            "max_results": 1_000,
            "max_depth": 16,
            "max_hops": 16,
            "max_bytes": 16 * 1024 * 1024,
            "max_query_bytes": 1024 * 1024,
            "max_chunk_nodes": DEFAULT_MAX_CHUNK_NODES,
            "max_chunk_edges": DEFAULT_MAX_CHUNK_EDGES,
            "max_chunk_count": 10_000,
            "max_provenance_links": 100_000,
            "timeout_ms": 10 * 60 * 1000,
        }
        for name, maximum in limits.items():
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=maximum),
            )
        if self.max_query_bytes > self.max_bytes:
            raise ProgramGraphProviderError(
                "max_query_bytes cannot exceed max_bytes"
            )
        if self.max_results > self.max_items:
            raise ProgramGraphProviderError(
                "max_results cannot exceed max_items"
            )
        if self.max_hops > self.max_depth:
            raise ProgramGraphProviderError(
                "max_hops cannot exceed max_depth"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_items": self.max_items,
            "max_results": self.max_results,
            "max_depth": self.max_depth,
            "max_hops": self.max_hops,
            "max_bytes": self.max_bytes,
            "max_query_bytes": self.max_query_bytes,
            "max_chunk_nodes": self.max_chunk_nodes,
            "max_chunk_edges": self.max_chunk_edges,
            "max_chunk_count": self.max_chunk_count,
            "max_provenance_links": self.max_provenance_links,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_value(
        cls, value: "GraphProjectionBounds | Mapping[str, Any] | None"
    ) -> "GraphProjectionBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProgramGraphProviderError("bounds must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ProgramGraphProviderError(
                "unknown bounds: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))

    def intersect(self, other: "GraphProjectionBounds") -> "GraphProjectionBounds":
        return GraphProjectionBounds(
            max_items=min(self.max_items, other.max_items),
            max_results=min(self.max_results, other.max_results),
            max_depth=min(self.max_depth, other.max_depth),
            max_hops=min(self.max_hops, other.max_hops),
            max_bytes=min(self.max_bytes, other.max_bytes),
            max_query_bytes=min(self.max_query_bytes, other.max_query_bytes),
            max_chunk_nodes=min(self.max_chunk_nodes, other.max_chunk_nodes),
            max_chunk_edges=min(self.max_chunk_edges, other.max_chunk_edges),
            max_chunk_count=min(self.max_chunk_count, other.max_chunk_count),
            max_provenance_links=min(
                self.max_provenance_links, other.max_provenance_links
            ),
            timeout_ms=min(self.timeout_ms, other.timeout_ms),
        )


@dataclass(frozen=True)
class GraphProjectionPolicy:
    """Supervisor policy for the optional GraphRAG/IPLD projection adapter."""

    enabled: bool = True
    module_root: str = DEFAULT_OPTIONAL_ROOT
    prefer_backend: bool = True
    allow_local_fallback: bool = True
    bounds: GraphProjectionBounds = field(default_factory=GraphProjectionBounds)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ProgramGraphProviderError("enabled must be a boolean")
        if not isinstance(self.prefer_backend, bool):
            raise ProgramGraphProviderError("prefer_backend must be a boolean")
        if not isinstance(self.allow_local_fallback, bool):
            raise ProgramGraphProviderError(
                "allow_local_fallback must be a boolean"
            )
        object.__setattr__(
            self,
            "module_root",
            _text(self.module_root, "module_root", max_bytes=255),
        )
        object.__setattr__(
            self, "bounds", GraphProjectionBounds.from_value(self.bounds)
        )

    @property
    def policy_id(self) -> str:
        return _content_id(self.to_dict(), name="program-graph-provider-policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "module_root": self.module_root,
            "prefer_backend": self.prefer_backend,
            "allow_local_fallback": self.allow_local_fallback,
            "bounds": self.bounds.to_dict(),
        }

    @classmethod
    def from_value(
        cls, value: "GraphProjectionPolicy | Mapping[str, Any] | None"
    ) -> "GraphProjectionPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProgramGraphProviderError("policy must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ProgramGraphProviderError(
                "unknown policy fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            enabled=value.get("enabled", True),
            module_root=value.get("module_root", DEFAULT_OPTIONAL_ROOT),
            prefer_backend=value.get("prefer_backend", True),
            allow_local_fallback=value.get("allow_local_fallback", True),
            bounds=GraphProjectionBounds.from_value(value.get("bounds")),
        )


@dataclass(frozen=True)
class GraphProjectionCapability:
    """Local capability declaration for the projection provider."""

    health: CapabilityHealth = CapabilityHealth.LAZY
    mode: ProjectionMode = ProjectionMode.LOCAL_FALLBACK
    imported: bool = False
    reason_code: ReasonCode = ReasonCode.LAZY_NOT_PROBED
    reason: str = "capability not probed"
    surfaces: tuple[str, ...] = ()
    bounds: GraphProjectionBounds = field(default_factory=GraphProjectionBounds)
    provider_version: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.health, CapabilityHealth):
            object.__setattr__(
                self, "health", CapabilityHealth(str(self.health))
            )
        if not isinstance(self.mode, ProjectionMode):
            object.__setattr__(self, "mode", ProjectionMode(str(self.mode)))
        if not isinstance(self.reason_code, ReasonCode):
            object.__setattr__(
                self, "reason_code", ReasonCode(str(self.reason_code))
            )
        if not isinstance(self.imported, bool):
            raise ProgramGraphProviderError("imported must be a boolean")
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, "reason", required=False, max_bytes=DEFAULT_MAX_REASON_BYTES),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(
                self.provider_version,
                "provider_version",
                required=False,
                max_bytes=128,
            ),
        )
        surfaces = tuple(
            sorted(
                {
                    _text(item, "surface", max_bytes=255)
                    for item in (self.surfaces or ())
                    if str(item or "").strip()
                }
            )
        )
        if len(surfaces) > 64:
            raise ProgramGraphProviderError("too many capability surfaces")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(
            self, "bounds", GraphProjectionBounds.from_value(self.bounds)
        )

    @property
    def available(self) -> bool:
        return self.health in {
            CapabilityHealth.HEALTHY,
            CapabilityHealth.DEGRADED,
            CapabilityHealth.PARTIAL,
        }

    @property
    def capability_id(self) -> str:
        return _content_id(self._payload(), name="program-graph-capability")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CAPABILITY_SCHEMA,
            "protocol_version": IPFS_DATASETS_PROGRAM_GRAPH_PROTOCOL_VERSION,
            "provider_id": IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID,
            "adapter_version": IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_VERSION,
            "provider_version": self.provider_version,
            "health": self.health.value,
            "available": self.available,
            "imported": self.imported,
            "mode": self.mode.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "surfaces": list(self.surfaces),
            "bounds": self.bounds.to_dict(),
            "lazy_import": True,
            **_authority_payload(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, **self._payload()}


def inspect_program_graph_provider_capability(
    policy: GraphProjectionPolicy | Mapping[str, Any] | None = None,
) -> GraphProjectionCapability:
    """Return the deterministic metadata-only capability declaration.

    Never imports the optional package.
    """

    selected = GraphProjectionPolicy.from_value(policy)
    if not selected.enabled:
        return GraphProjectionCapability(
            health=CapabilityHealth.UNAVAILABLE,
            mode=ProjectionMode.LOCAL_FALLBACK,
            reason_code=ReasonCode.PROVIDER_DISABLED,
            reason="program graph provider is disabled",
            bounds=selected.bounds,
        )
    return GraphProjectionCapability(
        health=CapabilityHealth.LAZY,
        mode=ProjectionMode.LOCAL_FALLBACK,
        reason_code=ReasonCode.LAZY_NOT_PROBED,
        reason="local fallback always available; optional GraphRAG not probed",
        bounds=selected.bounds,
    )


# ---------------------------------------------------------------------------
# Projection records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceLink:
    """Compact provenance edge between content-addressed artifacts."""

    source_cid: str
    target_cid: str
    kind: str
    producer: str = ""
    component_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_cid", _text(self.source_cid, "source_cid", max_bytes=256)
        )
        object.__setattr__(
            self, "target_cid", _text(self.target_cid, "target_cid", max_bytes=256)
        )
        object.__setattr__(
            self, "kind", _text(self.kind, "kind", max_bytes=128)
        )
        object.__setattr__(
            self,
            "producer",
            _text(self.producer, "producer", required=False, max_bytes=256),
        )
        object.__setattr__(
            self,
            "component_id",
            _text(
                self.component_id, "component_id", required=False, max_bytes=256
            ),
        )

    @property
    def link_id(self) -> str:
        return _content_id(self._payload(), name="provenance-link")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_PROVENANCE_SCHEMA,
            "source_cid": self.source_cid,
            "target_cid": self.target_cid,
            "kind": self.kind,
            "producer": self.producer,
            "component_id": self.component_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"link_id": self.link_id, **self._payload()}


@dataclass(frozen=True)
class ProjectedChunk:
    """One bounded IPLD-shaped projection of a program-graph chunk."""

    chunk_key: str
    chunk_id: str
    chunk_cid: str
    forest_id: str
    component_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    node_count: int
    edge_count: int
    provenance_links: tuple[ProvenanceLink, ...] = ()
    blob_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "chunk_key", _text(self.chunk_key, "chunk_key", max_bytes=512)
        )
        object.__setattr__(
            self, "chunk_id", _text(self.chunk_id, "chunk_id", max_bytes=256)
        )
        object.__setattr__(
            self, "chunk_cid", _text(self.chunk_cid, "chunk_cid", max_bytes=256)
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", max_bytes=512)
        )
        object.__setattr__(
            self,
            "component_ids",
            tuple(
                sorted(
                    {
                        _text(item, "component_id", max_bytes=256)
                        for item in (self.component_ids or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "node_ids",
            tuple(
                sorted(
                    {
                        _text(item, "node_id", max_bytes=512)
                        for item in (self.node_ids or ())
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "edge_ids",
            tuple(
                sorted(
                    {
                        _text(item, "edge_id", max_bytes=512)
                        for item in (self.edge_ids or ())
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "node_count",
            _non_negative_int(self.node_count, "node_count", maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "edge_count",
            _non_negative_int(self.edge_count, "edge_count", maximum=1_000_000),
        )
        if self.node_count != len(self.node_ids):
            raise ProgramGraphProviderError("node_count does not match node_ids")
        if self.edge_count != len(self.edge_ids):
            raise ProgramGraphProviderError("edge_count does not match edge_ids")
        links = tuple(self.provenance_links or ())
        for link in links:
            if not isinstance(link, ProvenanceLink):
                raise ProgramGraphProviderError(
                    "provenance_links must contain ProvenanceLink"
                )
        object.__setattr__(self, "provenance_links", links)
        object.__setattr__(
            self,
            "blob_cids",
            tuple(
                sorted(
                    {
                        _text(item, "blob_cid", max_bytes=256)
                        for item in (self.blob_cids or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CHUNK_SCHEMA,
            "chunk_key": self.chunk_key,
            "chunk_id": self.chunk_id,
            "chunk_cid": self.chunk_cid,
            "forest_id": self.forest_id,
            "component_ids": list(self.component_ids),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "blob_cids": list(self.blob_cids),
            "provenance_links": [link.to_dict() for link in self.provenance_links],
        }


@dataclass(frozen=True)
class GraphProjection:
    """Bounded IPLD/GraphRAG projection of a canonical program graph."""

    forest_id: str
    graph_id: str
    chunks: tuple[ProjectedChunk, ...]
    mode: ProjectionMode
    status: ProjectionStatus
    reason_code: ReasonCode
    reason: str
    bounds: GraphProjectionBounds
    index_cid: str = ""
    truncated: bool = False
    truncation_reason: str = ""
    capability: GraphProjectionCapability | None = None
    provenance_links: tuple[ProvenanceLink, ...] = ()
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", max_bytes=512)
        )
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id", max_bytes=256)
        )
        if not isinstance(self.mode, ProjectionMode):
            object.__setattr__(self, "mode", ProjectionMode(str(self.mode)))
        if not isinstance(self.status, ProjectionStatus):
            object.__setattr__(
                self, "status", ProjectionStatus(str(self.status))
            )
        if not isinstance(self.reason_code, ReasonCode):
            object.__setattr__(
                self, "reason_code", ReasonCode(str(self.reason_code))
            )
        object.__setattr__(
            self,
            "reason",
            _text(
                self.reason,
                "reason",
                required=False,
                max_bytes=DEFAULT_MAX_REASON_BYTES,
            ),
        )
        object.__setattr__(
            self, "bounds", GraphProjectionBounds.from_value(self.bounds)
        )
        object.__setattr__(
            self,
            "index_cid",
            _text(self.index_cid, "index_cid", required=False, max_bytes=256),
        )
        if not isinstance(self.truncated, bool):
            raise ProgramGraphProviderError("truncated must be a boolean")
        object.__setattr__(
            self,
            "truncation_reason",
            _text(
                self.truncation_reason,
                "truncation_reason",
                required=False,
                max_bytes=DEFAULT_MAX_REASON_BYTES,
            ),
        )
        chunks = tuple(self.chunks or ())
        for chunk in chunks:
            if not isinstance(chunk, ProjectedChunk):
                raise ProgramGraphProviderError(
                    "chunks must contain ProjectedChunk values"
                )
            if chunk.forest_id != self.forest_id:
                raise ProgramGraphProviderError(
                    "chunk forest_id must match projection forest_id"
                )
        # Deterministic order by chunk_key.
        object.__setattr__(
            self,
            "chunks",
            tuple(sorted(chunks, key=lambda item: item.chunk_key)),
        )
        links = tuple(self.provenance_links or ())
        for link in links:
            if not isinstance(link, ProvenanceLink):
                raise ProgramGraphProviderError(
                    "provenance_links must contain ProvenanceLink"
                )
        object.__setattr__(self, "provenance_links", links)
        object.__setattr__(
            self,
            "elapsed_ms",
            _non_negative_int(self.elapsed_ms, "elapsed_ms", maximum=10**9),
        )
        if self.capability is not None and not isinstance(
            self.capability, GraphProjectionCapability
        ):
            raise ProgramGraphProviderError(
                "capability must be GraphProjectionCapability"
            )

    @property
    def projection_id(self) -> str:
        return _content_id(self._identity_payload(), name="program-graph-projection")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_PROJECTION_SCHEMA,
            "protocol_version": IPFS_DATASETS_PROGRAM_GRAPH_PROTOCOL_VERSION,
            "provider_id": IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID,
            "adapter_version": IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_VERSION,
            "forest_id": self.forest_id,
            "graph_id": self.graph_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "bounds": self.bounds.to_dict(),
            "index_cid": self.index_cid,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "provenance_links": [link.to_dict() for link in self.provenance_links],
            **_authority_payload(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "projection_id": self.projection_id,
            "elapsed_ms": self.elapsed_ms,
            **self._identity_payload(),
        }
        if self.capability is not None:
            payload["capability"] = self.capability.to_dict()
        encoded = _json_bytes(payload, name="projection")
        if len(encoded) > self.bounds.max_bytes * 4:
            # Soft diagnostic only; construction already applies hard chunk bounds.
            pass
        return payload

    def allowed_node_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for chunk in self.chunks:
            ids.update(chunk.node_ids)
        return frozenset(ids)

    def allowed_edge_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for chunk in self.chunks:
            ids.update(chunk.edge_ids)
        return frozenset(ids)

    def chunk_cid_for_node(self, node_id: str) -> str | None:
        for chunk in self.chunks:
            if node_id in chunk.node_ids:
                return chunk.chunk_cid
        return None


@dataclass(frozen=True)
class GraphProjectionQuery:
    """Bounded ranking query over a projection."""

    text: str
    forest_id: str = ""
    seed_node_ids: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    max_results: int | None = None
    max_hops: int | None = None
    max_depth: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _text(self.text, "query text", required=False, max_bytes=DEFAULT_MAX_QUERY_BYTES),
        )
        object.__setattr__(
            self,
            "forest_id",
            _text(self.forest_id, "forest_id", required=False, max_bytes=512),
        )
        seeds = tuple(
            sorted(
                {
                    _text(item, "seed_node_id", max_bytes=512)
                    for item in (self.seed_node_ids or ())
                    if str(item or "").strip()
                }
            )
        )
        if len(seeds) > DEFAULT_MAX_ITEMS:
            raise ProgramGraphProviderError("too many seed_node_ids")
        object.__setattr__(self, "seed_node_ids", seeds)
        kinds = tuple(
            sorted(
                {
                    _text(item, "kind", max_bytes=128).casefold()
                    for item in (self.kinds or ())
                    if str(item or "").strip()
                }
            )
        )
        object.__setattr__(self, "kinds", kinds)
        if self.max_results is not None:
            object.__setattr__(
                self,
                "max_results",
                _positive_int(self.max_results, "max_results", maximum=1_000),
            )
        if self.max_hops is not None:
            object.__setattr__(
                self,
                "max_hops",
                _positive_int(self.max_hops, "max_hops", maximum=16),
            )
        if self.max_depth is not None:
            object.__setattr__(
                self,
                "max_depth",
                _positive_int(self.max_depth, "max_depth", maximum=16),
            )
        if not isinstance(self.payload, Mapping):
            raise ProgramGraphProviderError("payload must be an object")
        payload = _canonical_value(dict(self.payload), name="payload")
        if _find_forbidden(payload):
            raise ProgramGraphProviderError(
                "query payload contains forbidden fields"
            )
        object.__setattr__(self, "payload", MappingProxyType(payload))
        if not self.text and not self.seed_node_ids:
            raise ProgramGraphProviderError(
                "query requires text and/or seed_node_ids"
            )

    @property
    def query_id(self) -> str:
        return _content_id(self.to_dict(), name="program-graph-query")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_QUERY_SCHEMA,
            "text": self.text,
            "forest_id": self.forest_id,
            "seed_node_ids": list(self.seed_node_ids),
            "kinds": list(self.kinds),
            "max_results": self.max_results,
            "max_hops": self.max_hops,
            "max_depth": self.max_depth,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_value(
        cls, value: "GraphProjectionQuery | str | Mapping[str, Any]"
    ) -> "GraphProjectionQuery":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if not isinstance(value, Mapping):
            raise ProgramGraphProviderError("query must be a string or object")
        return cls(
            text=str(value.get("text") or value.get("query") or ""),
            forest_id=str(value.get("forest_id") or ""),
            seed_node_ids=tuple(value.get("seed_node_ids") or ()),
            kinds=tuple(value.get("kinds") or ()),
            max_results=value.get("max_results"),
            max_hops=value.get("max_hops"),
            max_depth=value.get("max_depth"),
            payload=value.get("payload") or {},
        )


@dataclass(frozen=True)
class RankedGraphReference:
    """One ranked compact reference into the projection."""

    node_id: str
    chunk_cid: str
    score: float
    ranking_reason: str
    kind: str = ""
    qualified_name: str = ""
    path: str = ""
    blob_cid: str = ""
    hop_distance: int = 0
    rank: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _text(self.node_id, "node_id", max_bytes=512)
        )
        object.__setattr__(
            self, "chunk_cid", _text(self.chunk_cid, "chunk_cid", max_bytes=256)
        )
        try:
            score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise ProgramGraphProviderError("score must be numeric") from exc
        if not math.isfinite(score) or score < 0.0:
            raise ProgramGraphProviderError(
                "score must be a finite non-negative number"
            )
        object.__setattr__(self, "score", round(score, 6))
        object.__setattr__(
            self,
            "ranking_reason",
            _text(
                self.ranking_reason,
                "ranking_reason",
                max_bytes=DEFAULT_MAX_REASON_BYTES,
            ),
        )
        for name in ("kind", "qualified_name", "path", "blob_cid"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=False,
                    max_bytes=DEFAULT_MAX_LABEL_BYTES,
                ),
            )
        object.__setattr__(
            self,
            "hop_distance",
            _non_negative_int(self.hop_distance, "hop_distance", maximum=1_000),
        )
        object.__setattr__(
            self,
            "rank",
            _non_negative_int(self.rank, "rank", maximum=1_000_000),
        )

    @property
    def reference_id(self) -> str:
        return _content_id(
            {
                "schema": PROVIDER_REFERENCE_SCHEMA,
                "node_id": self.node_id,
                "chunk_cid": self.chunk_cid,
                "score": self.score,
                "ranking_reason": self.ranking_reason,
                "kind": self.kind,
                "qualified_name": self.qualified_name,
                "path": self.path,
                "blob_cid": self.blob_cid,
                "hop_distance": self.hop_distance,
            },
            name="graph-reference",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_REFERENCE_SCHEMA,
            "reference_id": self.reference_id,
            "node_id": self.node_id,
            "chunk_cid": self.chunk_cid,
            "score": self.score,
            "score_millionths": int(round(self.score * 1_000_000)),
            "ranking_reason": self.ranking_reason,
            "kind": self.kind,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "blob_cid": self.blob_cid,
            "hop_distance": self.hop_distance,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class GraphQueryResult:
    """Bounded, non-authoritative ranking result over a projection."""

    query_id: str
    projection_id: str
    forest_id: str
    status: ProjectionStatus
    reason_code: ReasonCode
    reason: str
    mode: ProjectionMode
    references: tuple[RankedGraphReference, ...]
    bounds: GraphProjectionBounds
    ranking_method: str = "deterministic_lexical_hop"
    truncated: bool = False
    truncation_reason: str = ""
    considered_count: int = 0
    elapsed_ms: int = 0
    capability: GraphProjectionCapability | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_id", _text(self.query_id, "query_id", max_bytes=256)
        )
        object.__setattr__(
            self,
            "projection_id",
            _text(self.projection_id, "projection_id", max_bytes=256),
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", max_bytes=512)
        )
        if not isinstance(self.status, ProjectionStatus):
            object.__setattr__(
                self, "status", ProjectionStatus(str(self.status))
            )
        if not isinstance(self.reason_code, ReasonCode):
            object.__setattr__(
                self, "reason_code", ReasonCode(str(self.reason_code))
            )
        if not isinstance(self.mode, ProjectionMode):
            object.__setattr__(self, "mode", ProjectionMode(str(self.mode)))
        object.__setattr__(
            self,
            "reason",
            _text(
                self.reason,
                "reason",
                required=False,
                max_bytes=DEFAULT_MAX_REASON_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "ranking_method",
            _text(self.ranking_method, "ranking_method", max_bytes=128),
        )
        refs = tuple(self.references or ())
        for ref in refs:
            if not isinstance(ref, RankedGraphReference):
                raise ProgramGraphProviderError(
                    "references must contain RankedGraphReference"
                )
        object.__setattr__(self, "references", refs)
        object.__setattr__(
            self, "bounds", GraphProjectionBounds.from_value(self.bounds)
        )
        if not isinstance(self.truncated, bool):
            raise ProgramGraphProviderError("truncated must be a boolean")
        object.__setattr__(
            self,
            "truncation_reason",
            _text(
                self.truncation_reason,
                "truncation_reason",
                required=False,
                max_bytes=DEFAULT_MAX_REASON_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "considered_count",
            _non_negative_int(
                self.considered_count, "considered_count", maximum=10**9
            ),
        )
        object.__setattr__(
            self,
            "elapsed_ms",
            _non_negative_int(self.elapsed_ms, "elapsed_ms", maximum=10**9),
        )

    @property
    def result_id(self) -> str:
        return _content_id(self._identity_payload(), name="program-graph-query-result")

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_RESULT_SCHEMA,
            "protocol_version": IPFS_DATASETS_PROGRAM_GRAPH_PROTOCOL_VERSION,
            "provider_id": IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID,
            "adapter_version": IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_VERSION,
            "query_id": self.query_id,
            "projection_id": self.projection_id,
            "forest_id": self.forest_id,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "mode": self.mode.value,
            "ranking_method": self.ranking_method,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "considered_count": self.considered_count,
            "bounds": self.bounds.to_dict(),
            "references": [item.to_dict() for item in self.references],
            **_authority_payload(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "result_id": self.result_id,
            "elapsed_ms": self.elapsed_ms,
            "safe_for_completion_reasoning": False,
            **self._identity_payload(),
        }
        if self.capability is not None:
            payload["capability"] = self.capability.to_dict()
        return payload


# ---------------------------------------------------------------------------
# Local projection and ranking
# ---------------------------------------------------------------------------


def _node_blob_cid(node: ProgramGraphNode) -> str:
    return str(node.binding.blob_cid or "")


def _node_reference_body(node: ProgramGraphNode) -> dict[str, Any]:
    """Compact IPLD-shaped node reference — never the full AST/source body."""

    return {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "record_key": node.record_key,
        "component_id": node.component_id,
        "qualified_name": node.qualified_name,
        "path": node.path,
        "language": node.language,
        "blob_cid": _node_blob_cid(node),
        "producer": node.binding.producer,
        "resolver_status": node.binding.resolver_status.value,
        "forest_id": node.binding.forest_id,
    }


def _edge_reference_body(edge: ProgramGraphEdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "kind": edge.kind.value,
        "source": edge.source,
        "target": edge.target,
        "component_id": edge.component_id,
        "blob_cid": edge.binding.blob_cid,
        "producer": edge.binding.producer,
        "resolver_status": edge.binding.resolver_status.value,
        "forest_id": edge.binding.forest_id,
    }


def _project_chunk(
    chunk: GraphChunk,
    *,
    bounds: GraphProjectionBounds,
) -> ProjectedChunk:
    if len(chunk.nodes) > bounds.max_chunk_nodes:
        raise ProgramGraphProviderBoundsError(
            f"chunk {chunk.chunk_key!r} exceeds max_chunk_nodes"
        )
    if len(chunk.edges) > bounds.max_chunk_edges:
        raise ProgramGraphProviderBoundsError(
            f"chunk {chunk.chunk_key!r} exceeds max_chunk_edges"
        )

    node_bodies = [_node_reference_body(node) for node in chunk.nodes]
    edge_bodies = [_edge_reference_body(edge) for edge in chunk.edges]
    ipld_body = {
        "schema": PROVIDER_CHUNK_SCHEMA,
        "chunk_key": chunk.chunk_key,
        "chunk_id": chunk.chunk_id,
        "forest_id": chunk.forest_id,
        "component_ids": list(chunk.component_ids),
        "nodes": node_bodies,
        "edges": edge_bodies,
    }
    chunk_cid = _chunk_cid(ipld_body)

    links: list[ProvenanceLink] = []
    blob_cids: set[str] = set()
    for node in chunk.nodes:
        blob = _node_blob_cid(node)
        if blob:
            blob_cids.add(blob)
            links.append(
                ProvenanceLink(
                    source_cid=chunk_cid,
                    target_cid=blob,
                    kind="projects_blob",
                    producer=node.binding.producer,
                    component_id=node.component_id,
                )
            )
        # Link chunk to the canonical program-graph node identity (not a CID,
        # but a stable content-bound node_id treated as target reference).
        links.append(
            ProvenanceLink(
                source_cid=chunk_cid,
                target_cid=content_identity(
                    {"schema": "program-graph-node-ref@1", "node_id": node.node_id}
                ),
                kind="projects_node",
                producer=node.binding.producer,
                component_id=node.component_id,
            )
        )
    if len(links) > bounds.max_provenance_links:
        links = links[: bounds.max_provenance_links]

    return ProjectedChunk(
        chunk_key=chunk.chunk_key,
        chunk_id=chunk.chunk_id,
        chunk_cid=chunk_cid,
        forest_id=chunk.forest_id,
        component_ids=chunk.component_ids,
        node_ids=tuple(node.node_id for node in chunk.nodes),
        edge_ids=tuple(edge.edge_id for edge in chunk.edges),
        node_count=len(chunk.nodes),
        edge_count=len(chunk.edges),
        provenance_links=tuple(links),
        blob_cids=tuple(sorted(blob_cids)),
    )


def project_program_graph_local(
    graph: ProgramGraph,
    *,
    bounds: GraphProjectionBounds | Mapping[str, Any] | None = None,
) -> GraphProjection:
    """Project a canonical program graph into deterministic IPLD-shaped chunks.

    This path never imports ``ipfs_datasets_py``.  It is the always-available
    local fallback and the ground truth for what may be ranked later.
    """

    if not isinstance(graph, ProgramGraph):
        if isinstance(graph, Mapping):
            graph = ProgramGraph.from_dict(graph)
        else:
            raise ProgramGraphProviderError(
                "graph must be a ProgramGraph or mapping"
            )

    selected = GraphProjectionBounds.from_value(bounds)
    started = time.monotonic()
    truncated = False
    truncation_reason = ""

    try:
        raw_chunks = graph.chunk_all_components(
            max_nodes=selected.max_chunk_nodes,
            max_edges=selected.max_chunk_edges,
        )
    except ProgramGraphError as exc:
        raise ProgramGraphProviderError(str(exc)) from exc

    if len(raw_chunks) > selected.max_chunk_count:
        raw_chunks = raw_chunks[: selected.max_chunk_count]
        truncated = True
        truncation_reason = "max_chunk_count"

    projected: list[ProjectedChunk] = []
    total_items = 0
    for chunk in raw_chunks:
        item_cost = len(chunk.nodes) + len(chunk.edges)
        if total_items + item_cost > selected.max_items and projected:
            truncated = True
            truncation_reason = truncation_reason or "max_items"
            break
        projected_chunk = _project_chunk(chunk, bounds=selected)
        # Byte bound on the compact projection envelope of this chunk.
        encoded = _json_bytes(projected_chunk.to_dict(), name="chunk")
        if len(encoded) > selected.max_bytes:
            raise ProgramGraphProviderBoundsError(
                f"projected chunk {chunk.chunk_key!r} exceeds max_bytes"
            )
        projected.append(projected_chunk)
        total_items += item_cost

    provenance: list[ProvenanceLink] = []
    for chunk in projected:
        provenance.extend(chunk.provenance_links)
        provenance.append(
            ProvenanceLink(
                source_cid=chunk.chunk_cid,
                target_cid=content_identity(
                    {
                        "schema": "program-graph-ref@1",
                        "graph_id": graph.graph_id,
                    }
                ),
                kind="derived_from_graph",
                producer=graph.producer,
            )
        )
    if len(provenance) > selected.max_provenance_links:
        provenance = provenance[: selected.max_provenance_links]
        truncated = True
        truncation_reason = truncation_reason or "max_provenance_links"

    index_body = {
        "schema": "program-graph-projection-index@1",
        "forest_id": graph.forest_id,
        "graph_id": graph.graph_id,
        "chunk_cids": [chunk.chunk_cid for chunk in projected],
        "chunk_keys": [chunk.chunk_key for chunk in projected],
    }
    index_cid = _chunk_cid(index_body)

    status = (
        ProjectionStatus.PARTIAL
        if truncated
        else ProjectionStatus.LOCAL_FALLBACK
    )
    reason_code = (
        ReasonCode.PARTIAL_TRUNCATION
        if truncated
        else ReasonCode.LOCAL_FALLBACK_PROJECTION
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return GraphProjection(
        forest_id=graph.forest_id,
        graph_id=graph.graph_id,
        chunks=tuple(projected),
        mode=ProjectionMode.LOCAL_FALLBACK,
        status=status,
        reason_code=reason_code,
        reason=(
            f"local deterministic projection"
            + (f"; truncated: {truncation_reason}" if truncated else "")
        ),
        bounds=selected,
        index_cid=index_cid,
        truncated=truncated,
        truncation_reason=truncation_reason,
        provenance_links=tuple(provenance),
        elapsed_ms=elapsed_ms,
    )


def _score_node(
    node: ProgramGraphNode,
    *,
    query_tokens: frozenset[str],
    seed_ids: frozenset[str],
    hop_distance: int,
    kinds: frozenset[str],
) -> tuple[float, str] | None:
    if kinds and node.kind.value.casefold() not in kinds:
        return None

    reasons: list[str] = []
    score = 0.0

    if node.node_id in seed_ids:
        score += 1.0
        reasons.append("seed_node")

    if query_tokens:
        fields = {
            "qualified_name": node.qualified_name,
            "path": node.path,
            "record_key": node.record_key,
            "kind": node.kind.value,
            "component_id": node.component_id,
        }
        matched_fields: list[str] = []
        overlap_total = 0
        for field_name, field_value in fields.items():
            field_tokens = _tokens(field_value)
            if not field_tokens:
                continue
            overlap = query_tokens & field_tokens
            if overlap:
                overlap_total += len(overlap)
                matched_fields.append(field_name)
        if overlap_total:
            # Normalize by query size so scores stay in [0, 1] for lexical part.
            lexical = min(1.0, overlap_total / max(1, len(query_tokens)))
            score += lexical
            reasons.append(
                "lexical_match:" + ",".join(sorted(matched_fields))
            )
        elif node.node_id not in seed_ids:
            return None

    if hop_distance > 0:
        # Mild decay for multi-hop neighbors of seeds.
        score *= max(0.1, 1.0 - 0.15 * hop_distance)
        reasons.append(f"hop_distance={hop_distance}")

    if score <= 0.0:
        return None
    if not reasons:
        reasons.append("unspecified")
    return score, ";".join(reasons)


def rank_projection_local(
    graph: ProgramGraph,
    projection: GraphProjection,
    query: GraphProjectionQuery,
    *,
    bounds: GraphProjectionBounds | None = None,
) -> GraphQueryResult:
    """Rank only nodes present in ``projection`` (canonical evidence)."""

    selected = (bounds or projection.bounds)
    if query.max_results is not None:
        selected = GraphProjectionBounds(
            **{
                **selected.to_dict(),
                "max_results": min(selected.max_results, query.max_results),
            }
        )
    max_hops = selected.max_hops
    if query.max_hops is not None:
        max_hops = min(max_hops, query.max_hops)
    max_depth = selected.max_depth
    if query.max_depth is not None:
        max_depth = min(max_depth, query.max_depth)
    # max_hops is the traversal budget; max_depth is the structural depth cap.
    effective_hops = min(max_hops, max_depth)

    started = time.monotonic()
    allowed_nodes = projection.allowed_node_ids()
    allowed_edges = projection.allowed_edge_ids()
    query_tokens = _tokens(query.text)
    seed_ids = frozenset(query.seed_node_ids) & allowed_nodes
    kinds = frozenset(query.kinds)

    # Build adjacency restricted to projected edges only.
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in allowed_nodes}
    for edge in graph.edges:
        if edge.edge_id not in allowed_edges:
            continue
        if edge.source in allowed_nodes and edge.target in allowed_nodes:
            adjacency.setdefault(edge.source, set()).add(edge.target)
            adjacency.setdefault(edge.target, set()).add(edge.source)

    # BFS from seeds (or all nodes when query is pure text).
    distances: dict[str, int] = {}
    if seed_ids:
        queue = list(sorted(seed_ids))
        for seed in queue:
            distances[seed] = 0
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            dist = distances[current]
            if dist >= effective_hops:
                continue
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
        candidate_ids = set(distances)
    else:
        candidate_ids = set(allowed_nodes)
        distances = {node_id: 0 for node_id in candidate_ids}

    node_map = {node.node_id: node for node in graph.nodes if node.node_id in allowed_nodes}
    scored: list[RankedGraphReference] = []
    considered = 0
    for node_id in sorted(candidate_ids):
        node = node_map.get(node_id)
        if node is None:
            continue
        considered += 1
        hop = distances.get(node_id, 0)
        scored_pair = _score_node(
            node,
            query_tokens=query_tokens,
            seed_ids=seed_ids,
            hop_distance=hop,
            kinds=kinds,
        )
        if scored_pair is None:
            continue
        score, reason = scored_pair
        chunk_cid = projection.chunk_cid_for_node(node_id) or ""
        if not chunk_cid:
            # Node not in projection is never rankable.
            continue
        scored.append(
            RankedGraphReference(
                node_id=node_id,
                chunk_cid=chunk_cid,
                score=score,
                ranking_reason=reason,
                kind=node.kind.value,
                qualified_name=node.qualified_name,
                path=node.path,
                blob_cid=_node_blob_cid(node),
                hop_distance=hop,
            )
        )

    # Stable total order: score desc, node_id asc.
    scored.sort(key=lambda item: (-item.score, item.node_id))
    truncated = False
    truncation_reason = ""
    if len(scored) > selected.max_results:
        scored = scored[: selected.max_results]
        truncated = True
        truncation_reason = "max_results"

    ranked = tuple(
        RankedGraphReference(
            node_id=item.node_id,
            chunk_cid=item.chunk_cid,
            score=item.score,
            ranking_reason=item.ranking_reason,
            kind=item.kind,
            qualified_name=item.qualified_name,
            path=item.path,
            blob_cid=item.blob_cid,
            hop_distance=item.hop_distance,
            rank=index,
        )
        for index, item in enumerate(scored)
    )

    # Enforce response byte bound by dropping lowest ranks if needed.
    def _encode(refs: tuple[RankedGraphReference, ...]) -> bytes:
        trial = GraphQueryResult(
            query_id=query.query_id,
            projection_id=projection.projection_id,
            forest_id=projection.forest_id,
            status=ProjectionStatus.LOCAL_FALLBACK,
            reason_code=ReasonCode.DETERMINISTIC_RANKING,
            reason="size_probe",
            mode=ProjectionMode.LOCAL_FALLBACK,
            references=refs,
            bounds=selected,
            truncated=truncated,
            truncation_reason=truncation_reason,
            considered_count=considered,
        )
        return _json_bytes(trial.to_dict(), name="query-result")

    while ranked and len(_encode(ranked)) > selected.max_bytes:
        ranked = ranked[:-1]
        truncated = True
        truncation_reason = truncation_reason or "max_bytes"

    status = (
        ProjectionStatus.PARTIAL
        if truncated
        else ProjectionStatus.LOCAL_FALLBACK
    )
    reason_code = (
        ReasonCode.PARTIAL_TRUNCATION
        if truncated
        else ReasonCode.DETERMINISTIC_RANKING
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return GraphQueryResult(
        query_id=query.query_id,
        projection_id=projection.projection_id,
        forest_id=projection.forest_id,
        status=status,
        reason_code=reason_code,
        reason=(
            "deterministic lexical+hop ranking over canonical projection"
            + (f"; truncated: {truncation_reason}" if truncated else "")
        ),
        mode=ProjectionMode.LOCAL_FALLBACK,
        references=ranked,
        bounds=selected,
        ranking_method="deterministic_lexical_hop",
        truncated=truncated,
        truncation_reason=truncation_reason,
        considered_count=considered,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Optional backend probe and sanitization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BackendSurface:
    module: str
    symbol: str
    method: str
    bounded: bool
    parameters: tuple[str, ...]


def _probe_module_surface(
    module: Any, module_name: str
) -> _BackendSurface | None:
    for attr_name in _QUERY_CLASS_NAMES:
        target = getattr(module, attr_name, None)
        if target is None:
            continue
        if inspect.isclass(target):
            method = getattr(target, "query", None)
            method_name = "query"
            if method is None:
                method = getattr(target, "search", None)
                method_name = "search"
            if method is None:
                continue
            try:
                params = _signature_params(method)
            except ProgramGraphProviderError:
                continue
            return _BackendSurface(
                module=module_name,
                symbol=attr_name,
                method=method_name,
                bounded=_has_bounds(params) or method_name != "query",
                parameters=tuple(sorted(params))[:32],
            )
    query_fn = getattr(module, "query", None)
    if callable(query_fn):
        try:
            params = _signature_params(query_fn)
        except ProgramGraphProviderError:
            params = frozenset()
        return _BackendSurface(
            module=module_name,
            symbol="query",
            method="query",
            bounded=_has_bounds(params),
            parameters=tuple(sorted(params))[:32],
        )
    return None


def _sanitize_backend_hits(
    raw: Any,
    *,
    projection: GraphProjection,
    graph: ProgramGraph,
    bounds: GraphProjectionBounds,
) -> tuple[tuple[RankedGraphReference, ...], str | None]:
    """Filter optional backend hits to canonical evidence only.

    Returns ``(references, poison_reason)``.  A non-``None`` poison reason
    means the backend invented evidence or claimed forbidden authority.
    """

    if raw is None:
        return (), None
    if _find_forbidden(raw):
        return (), f"forbidden_field:{_find_forbidden(raw)}"

    # Normalize common GraphRAG response envelopes.
    items: Sequence[Any]
    if isinstance(raw, Mapping):
        if raw.get("completion_authority") is True:
            return (), "authority:completion_authority"
        if raw.get("mutation_authority") is True:
            return (), "authority:mutation_authority"
        for key in (
            "results",
            "hits",
            "references",
            "hybrid_results",
            "graph_results",
            "vector_results",
        ):
            if key in raw and isinstance(raw[key], Sequence) and not isinstance(
                raw[key], (str, bytes)
            ):
                items = raw[key]
                break
        else:
            # Single mapping treated as one hit.
            items = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = raw
    else:
        return (), "unsupported_backend_payload"

    allowed_nodes = projection.allowed_node_ids()
    node_map = {
        node.node_id: node for node in graph.nodes if node.node_id in allowed_nodes
    }
    refs: list[RankedGraphReference] = []
    for item in items:
        if not isinstance(item, Mapping):
            converter = getattr(item, "to_dict", None)
            item = converter() if callable(converter) else None
        if not isinstance(item, Mapping):
            return (), "non_object_hit"
        if _find_forbidden(item):
            return (), f"forbidden_field:{_find_forbidden(item)}"

        node_id = str(
            item.get("node_id")
            or item.get("entity_id")
            or item.get("id")
            or item.get("symbol")
            or ""
        ).strip()
        # If backend returns a qualified name, map it into the projection.
        if node_id not in allowed_nodes:
            qname = str(item.get("qualified_name") or item.get("name") or "").strip()
            path = str(item.get("path") or "").strip()
            matched = None
            for candidate in node_map.values():
                if qname and candidate.qualified_name == qname:
                    matched = candidate.node_id
                    break
                if path and candidate.path == path and path:
                    matched = candidate.node_id
                    break
            if matched is None:
                # Unknown identity is poison: GraphRAG must not invent nodes.
                return (), f"unknown_node:{node_id or qname or path or '<empty>'}"
            node_id = matched

        node = node_map[node_id]
        chunk_cid = projection.chunk_cid_for_node(node_id) or ""
        if not chunk_cid:
            return (), f"node_missing_chunk:{node_id}"

        # Backend may not override chunk CID or invent a different blob.
        claimed_chunk = str(item.get("chunk_cid") or "").strip()
        if claimed_chunk and claimed_chunk != chunk_cid:
            return (), f"forged_chunk_cid:{claimed_chunk}"
        claimed_blob = str(item.get("blob_cid") or "").strip()
        if claimed_blob and claimed_blob != _node_blob_cid(node):
            return (), f"forged_blob_cid:{claimed_blob}"

        try:
            score = float(item.get("score", item.get("relevance", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return (), "non_numeric_score"
        if not math.isfinite(score) or score < 0.0:
            return (), "invalid_score"

        reason = str(
            item.get("ranking_reason")
            or item.get("reason")
            or item.get("explanation")
            or "backend_rank"
        ).strip()
        if len(reason.encode("utf-8")) > DEFAULT_MAX_REASON_BYTES:
            reason = reason.encode("utf-8")[: DEFAULT_MAX_REASON_BYTES - 3].decode(
                "utf-8", errors="ignore"
            ) + "..."

        refs.append(
            RankedGraphReference(
                node_id=node_id,
                chunk_cid=chunk_cid,
                score=score,
                ranking_reason=f"backend:{reason}",
                kind=node.kind.value,
                qualified_name=node.qualified_name,
                path=node.path,
                blob_cid=_node_blob_cid(node),
                hop_distance=int(item.get("hop_distance") or 0)
                if not isinstance(item.get("hop_distance"), bool)
                else 0,
            )
        )
        if len(refs) > bounds.max_items:
            return (), "backend_exceeded_max_items"

    # Stable order before applying max_results.
    refs.sort(key=lambda item: (-item.score, item.node_id))
    if len(refs) > bounds.max_results:
        refs = refs[: bounds.max_results]
    ranked = tuple(
        RankedGraphReference(
            node_id=item.node_id,
            chunk_cid=item.chunk_cid,
            score=item.score,
            ranking_reason=item.ranking_reason,
            kind=item.kind,
            qualified_name=item.qualified_name,
            path=item.path,
            blob_cid=item.blob_cid,
            hop_distance=item.hop_distance,
            rank=index,
        )
        for index, item in enumerate(refs)
    )
    return ranked, None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class IpfsDatasetsProgramGraphProvider:
    """Lazy, bounded GraphRAG/IPLD projection provider for program graphs.

    Canonical graph construction remains owned by :mod:`program_graph`.  This
    adapter only projects, indexes, and ranks admitted evidence.
    """

    provider_id = IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID
    provider_version = IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_VERSION
    protocol_version = IPFS_DATASETS_PROGRAM_GRAPH_PROTOCOL_VERSION

    def __init__(
        self,
        policy: GraphProjectionPolicy | Mapping[str, Any] | None = None,
        *,
        importer: ModuleImporter | None = None,
        backend: Any = None,
        clock: Clock | None = None,
        enabled: bool | None = None,
        module_root: str | None = None,
        bounds: GraphProjectionBounds | Mapping[str, Any] | None = None,
    ) -> None:
        selected = GraphProjectionPolicy.from_value(policy)
        overrides = any(
            item is not None for item in (enabled, module_root, bounds)
        )
        if policy is not None and overrides:
            raise ProgramGraphProviderError(
                "policy cannot be combined with policy field overrides"
            )
        if overrides:
            selected = GraphProjectionPolicy(
                enabled=True if enabled is None else enabled,
                module_root=module_root or DEFAULT_OPTIONAL_ROOT,
                bounds=GraphProjectionBounds.from_value(bounds),
            )
        if importer is not None and not callable(importer):
            raise ProgramGraphProviderError("importer must be callable")
        self.policy = selected
        self._importer = importer or importlib.import_module
        self._backend = backend
        self._clock = clock or time.monotonic
        self._probe_lock = threading.Lock()
        self._cached_capability: GraphProjectionCapability | None = None
        self._cached_surface: _BackendSurface | None = None
        self._cached_backend: Any = None

    # -- capability --------------------------------------------------------

    def capabilities(self) -> GraphProjectionCapability:
        """Metadata-only declaration; never imports the optional package."""

        return inspect_program_graph_provider_capability(self.policy)

    capability = capabilities

    def clear_probe_cache(self) -> None:
        with self._probe_lock:
            self._cached_capability = None
            self._cached_surface = None
            # Keep explicit injected backend; only drop import-derived cache.
            if self._backend is None:
                self._cached_backend = None

    def probe(
        self, *, force: bool = False
    ) -> GraphProjectionCapability:
        """Probe optional graph/IPLD/index/query APIs under a time budget."""

        if not self.policy.enabled:
            return GraphProjectionCapability(
                health=CapabilityHealth.UNAVAILABLE,
                mode=ProjectionMode.LOCAL_FALLBACK,
                reason_code=ReasonCode.PROVIDER_DISABLED,
                reason="program graph provider is disabled",
                bounds=self.policy.bounds,
            )

        with self._probe_lock:
            if self._cached_capability is not None and not force:
                return self._cached_capability
            capability = self._probe_unlocked()
            self._cached_capability = capability
            return capability

    def _probe_unlocked(self) -> GraphProjectionCapability:
        # Injected backend short-circuits import discovery.
        if self._backend is not None:
            surface = self._surface_from_backend(self._backend)
            self._cached_backend = self._backend
            self._cached_surface = surface
            if surface is None:
                return GraphProjectionCapability(
                    health=CapabilityHealth.INCOMPATIBLE,
                    mode=ProjectionMode.LOCAL_FALLBACK,
                    imported=True,
                    reason_code=ReasonCode.OPTIONAL_API_INCOMPATIBLE,
                    reason="injected backend has no compatible query surface",
                    surfaces=("backend:injected",),
                    bounds=self.policy.bounds,
                    provider_version="injected",
                )
            if not surface.bounded:
                return GraphProjectionCapability(
                    health=CapabilityHealth.INCOMPATIBLE,
                    mode=ProjectionMode.LOCAL_FALLBACK,
                    imported=True,
                    reason_code=ReasonCode.OPTIONAL_API_INCOMPATIBLE,
                    reason=(
                        f"{surface.symbol}.{surface.method} lacks bound parameters"
                    ),
                    surfaces=(f"{surface.module}:{surface.symbol}",),
                    bounds=self.policy.bounds,
                    provider_version="injected",
                )
            return GraphProjectionCapability(
                health=CapabilityHealth.HEALTHY,
                mode=ProjectionMode.GRAPHRAG,
                imported=True,
                reason_code=ReasonCode.BOUNDED_QUERY,
                reason=(
                    f"injected backend exposes bounded "
                    f"{surface.symbol}.{surface.method}"
                ),
                surfaces=(f"{surface.module}:{surface.symbol}",),
                bounds=self.policy.bounds,
                provider_version="injected",
            )

        root = self.policy.module_root
        surfaces_found: list[str] = []
        imported_any = False
        best: _BackendSurface | None = None
        deadline = self._clock() + (self.policy.bounds.timeout_ms / 1000.0)
        # Use a tighter probe budget than query budget.
        probe_timeout = min(
            self.policy.bounds.timeout_ms, DEFAULT_PROBE_TIMEOUT_MS
        ) / 1000.0

        for relative in _GRAPH_API_CANDIDATES:
            if self._clock() > deadline:
                break
            module_name = f"{root}.{relative}" if relative else root
            # cid_utils is not a query surface but proves strict CID helpers.
            try:
                module = self._timed_import(module_name, timeout=probe_timeout)
            except TimeoutError:
                return GraphProjectionCapability(
                    health=CapabilityHealth.DEGRADED,
                    mode=ProjectionMode.LOCAL_FALLBACK,
                    imported=imported_any,
                    reason_code=ReasonCode.TIMEOUT,
                    reason=f"probe timed out importing {module_name}",
                    surfaces=tuple(surfaces_found),
                    bounds=self.policy.bounds,
                )
            except Exception:
                continue
            imported_any = True
            surfaces_found.append(f"import:{module_name}")
            if relative.endswith("cid_utils"):
                # Require at least one CID helper for IPLD identity support.
                helpers = [
                    name
                    for name in (
                        "cid_for_dag_json",
                        "cid_for_obj",
                        "validate_cid",
                        "canonical_dag_json_bytes",
                    )
                    if callable(getattr(module, name, None))
                ]
                if helpers:
                    surfaces_found.append("cid_utils:" + ",".join(helpers))
                continue
            surface = _probe_module_surface(module, module_name)
            if surface is None:
                continue
            surfaces_found.append(
                f"{surface.module}:{surface.symbol}.{surface.method}"
            )
            if surface.bounded and best is None:
                best = surface
                self._cached_backend = module
                self._cached_surface = surface

        if best is not None:
            return GraphProjectionCapability(
                health=CapabilityHealth.HEALTHY,
                mode=ProjectionMode.GRAPHRAG,
                imported=True,
                reason_code=ReasonCode.BOUNDED_QUERY,
                reason=(
                    f"compatible bounded query surface "
                    f"{best.symbol}.{best.method}"
                ),
                surfaces=tuple(surfaces_found),
                bounds=self.policy.bounds,
            )

        if imported_any:
            return GraphProjectionCapability(
                health=CapabilityHealth.PARTIAL,
                mode=ProjectionMode.LOCAL_FALLBACK,
                imported=True,
                reason_code=ReasonCode.OPTIONAL_API_PARTIAL,
                reason=(
                    "optional modules imported but no compatible bounded "
                    "query surface was found"
                ),
                surfaces=tuple(surfaces_found),
                bounds=self.policy.bounds,
            )

        return GraphProjectionCapability(
            health=CapabilityHealth.UNAVAILABLE,
            mode=ProjectionMode.LOCAL_FALLBACK,
            imported=False,
            reason_code=ReasonCode.OPTIONAL_MODULE_UNAVAILABLE,
            reason="no GraphRAG/IPLD/index modules are importable",
            surfaces=(),
            bounds=self.policy.bounds,
        )

    def _timed_import(self, module_name: str, *, timeout: float) -> Any:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._importer, module_name)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                raise TimeoutError(
                    f"import of {module_name} exceeded {timeout}s"
                ) from exc

    def _surface_from_backend(self, backend: Any) -> _BackendSurface | None:
        if inspect.isclass(type(backend)) and not inspect.isclass(backend):
            # Instance: look for query/search methods.
            for method_name in ("query", "search", "retrieve", "rank"):
                method = getattr(backend, method_name, None)
                if not callable(method):
                    continue
                try:
                    params = _signature_params(method)
                except ProgramGraphProviderError:
                    params = frozenset()
                return _BackendSurface(
                    module="injected",
                    symbol=type(backend).__name__,
                    method=method_name,
                    bounded=_has_bounds(params) or method_name != "query",
                    parameters=tuple(sorted(params))[:32],
                )
        if callable(backend):
            try:
                params = _signature_params(backend)
            except ProgramGraphProviderError:
                params = frozenset()
            return _BackendSurface(
                module="injected",
                symbol="callable",
                method="__call__",
                bounded=_has_bounds(params),
                parameters=tuple(sorted(params))[:32],
            )
        return _probe_module_surface(backend, "injected")

    # -- project -----------------------------------------------------------

    def project(
        self,
        graph: ProgramGraph | Mapping[str, Any],
        *,
        bounds: GraphProjectionBounds | Mapping[str, Any] | None = None,
        probe: bool = True,
    ) -> GraphProjection:
        """Project a canonical program graph into deterministic chunk CIDs.

        Always produces a local projection first (the admitted evidence set).
        Optional backend enrichment may annotate mode when healthy; it may
        never expand the node/edge identity set.
        """

        if not self.policy.enabled:
            local = project_program_graph_local(
                graph if isinstance(graph, ProgramGraph) else ProgramGraph.from_dict(graph),  # type: ignore[arg-type]
                bounds=self.policy.bounds,
            )
            return GraphProjection(
                forest_id=local.forest_id,
                graph_id=local.graph_id,
                chunks=local.chunks,
                mode=ProjectionMode.LOCAL_FALLBACK,
                status=ProjectionStatus.DISABLED,
                reason_code=ReasonCode.PROVIDER_DISABLED,
                reason="provider disabled; local projection only",
                bounds=local.bounds,
                index_cid=local.index_cid,
                truncated=local.truncated,
                truncation_reason=local.truncation_reason,
                provenance_links=local.provenance_links,
                elapsed_ms=local.elapsed_ms,
                capability=self.capabilities(),
            )

        selected = self.policy.bounds
        if bounds is not None:
            selected = selected.intersect(
                GraphProjectionBounds.from_value(bounds)
            )

        if isinstance(graph, Mapping):
            graph = ProgramGraph.from_dict(graph)
        if not isinstance(graph, ProgramGraph):
            raise ProgramGraphProviderError(
                "graph must be a ProgramGraph or mapping"
            )

        local = project_program_graph_local(graph, bounds=selected)
        capability = (
            self.probe()
            if probe
            else inspect_program_graph_provider_capability(self.policy)
        )

        # Local projection is always the evidence set.  Backend mode is
        # informational once a compatible surface exists.
        mode = local.mode
        status = local.status
        reason_code = local.reason_code
        reason = local.reason
        if capability.health is CapabilityHealth.HEALTHY:
            mode = ProjectionMode.GRAPHRAG
            if not local.truncated:
                status = ProjectionStatus.COMPLETED
                reason_code = ReasonCode.BOUNDED_PROJECTION
                reason = (
                    "local deterministic projection with compatible GraphRAG surface"
                )
        elif capability.health is CapabilityHealth.PARTIAL:
            mode = ProjectionMode.MIXED
            status = ProjectionStatus.PARTIAL
            reason_code = ReasonCode.OPTIONAL_API_PARTIAL
            reason = capability.reason
        elif capability.health is CapabilityHealth.INCOMPATIBLE:
            if not self.policy.allow_local_fallback:
                status = ProjectionStatus.INCOMPATIBLE
                reason_code = ReasonCode.OPTIONAL_API_INCOMPATIBLE
                reason = capability.reason
            else:
                status = ProjectionStatus.LOCAL_FALLBACK
                reason_code = ReasonCode.LOCAL_FALLBACK_PROJECTION
                reason = (
                    f"incompatible optional API; local fallback: {capability.reason}"
                )
        elif capability.health is CapabilityHealth.UNAVAILABLE:
            if not self.policy.allow_local_fallback:
                status = ProjectionStatus.UNAVAILABLE
                reason_code = ReasonCode.OPTIONAL_MODULE_UNAVAILABLE
                reason = capability.reason
            else:
                status = ProjectionStatus.LOCAL_FALLBACK
                reason_code = ReasonCode.LOCAL_FALLBACK_PROJECTION
                reason = (
                    f"optional module unavailable; local fallback: {capability.reason}"
                )

        return GraphProjection(
            forest_id=local.forest_id,
            graph_id=local.graph_id,
            chunks=local.chunks,
            mode=mode,
            status=status,
            reason_code=reason_code,
            reason=reason,
            bounds=local.bounds,
            index_cid=local.index_cid,
            truncated=local.truncated,
            truncation_reason=local.truncation_reason,
            provenance_links=local.provenance_links,
            elapsed_ms=local.elapsed_ms,
            capability=capability,
        )

    # -- query -------------------------------------------------------------

    def query(
        self,
        graph: ProgramGraph | Mapping[str, Any],
        query: GraphProjectionQuery | str | Mapping[str, Any],
        *,
        projection: GraphProjection | None = None,
        bounds: GraphProjectionBounds | Mapping[str, Any] | None = None,
        use_backend: bool | None = None,
    ) -> GraphQueryResult:
        """Rank canonical evidence under hard item/depth/byte/time bounds."""

        if isinstance(graph, Mapping):
            graph = ProgramGraph.from_dict(graph)
        if not isinstance(graph, ProgramGraph):
            raise ProgramGraphProviderError(
                "graph must be a ProgramGraph or mapping"
            )
        query_obj = GraphProjectionQuery.from_value(query)
        if len(_json_bytes(query_obj.to_dict(), name="query")) > (
            self.policy.bounds.max_query_bytes
        ):
            raise ProgramGraphProviderBoundsError("query exceeds max_query_bytes")

        selected = self.policy.bounds
        if bounds is not None:
            selected = selected.intersect(
                GraphProjectionBounds.from_value(bounds)
            )
        if projection is None:
            projection = self.project(graph, bounds=selected, probe=True)
        elif projection.forest_id != graph.forest_id:
            raise ProgramGraphProviderError(
                "projection forest_id does not match graph forest_id"
            )

        prefer_backend = (
            self.policy.prefer_backend if use_backend is None else use_backend
        )
        capability = projection.capability or self.probe()

        if prefer_backend and capability.health is CapabilityHealth.HEALTHY:
            backend_result = self._query_backend(
                graph=graph,
                projection=projection,
                query=query_obj,
                bounds=selected,
                capability=capability,
            )
            if backend_result is not None:
                return backend_result

        # Local deterministic ranking is always available as fallback.
        result = rank_projection_local(
            graph, projection, query_obj, bounds=selected
        )
        # Re-stamp capability and mode awareness.
        status = result.status
        reason_code = result.reason_code
        reason = result.reason
        mode = ProjectionMode.LOCAL_FALLBACK
        if capability.health is CapabilityHealth.UNAVAILABLE:
            status = (
                ProjectionStatus.LOCAL_FALLBACK
                if self.policy.allow_local_fallback
                else ProjectionStatus.UNAVAILABLE
            )
            reason_code = (
                ReasonCode.LOCAL_FALLBACK_QUERY
                if self.policy.allow_local_fallback
                else ReasonCode.OPTIONAL_MODULE_UNAVAILABLE
            )
            reason = (
                f"optional GraphRAG unavailable; {result.reason}"
                if self.policy.allow_local_fallback
                else capability.reason
            )
        elif capability.health is CapabilityHealth.INCOMPATIBLE:
            status = (
                ProjectionStatus.LOCAL_FALLBACK
                if self.policy.allow_local_fallback
                else ProjectionStatus.INCOMPATIBLE
            )
            reason_code = (
                ReasonCode.LOCAL_FALLBACK_QUERY
                if self.policy.allow_local_fallback
                else ReasonCode.OPTIONAL_API_INCOMPATIBLE
            )
            reason = (
                f"optional GraphRAG incompatible; {result.reason}"
                if self.policy.allow_local_fallback
                else capability.reason
            )
        elif capability.health is CapabilityHealth.PARTIAL:
            status = ProjectionStatus.PARTIAL
            reason_code = ReasonCode.OPTIONAL_API_PARTIAL
            reason = f"{capability.reason}; {result.reason}"
            mode = ProjectionMode.MIXED
        elif capability.health is CapabilityHealth.HEALTHY:
            # Backend was preferred but did not produce a usable result —
            # still a valid local ranking over the same evidence.
            status = (
                ProjectionStatus.PARTIAL
                if result.truncated
                else ProjectionStatus.COMPLETED
            )
            reason_code = (
                ReasonCode.PARTIAL_TRUNCATION
                if result.truncated
                else ReasonCode.DETERMINISTIC_RANKING
            )
            mode = ProjectionMode.LOCAL_FALLBACK
            reason = f"backend unused or empty; {result.reason}"

        return GraphQueryResult(
            query_id=result.query_id,
            projection_id=result.projection_id,
            forest_id=result.forest_id,
            status=status,
            reason_code=reason_code,
            reason=reason,
            mode=mode,
            references=result.references,
            bounds=result.bounds,
            ranking_method=result.ranking_method,
            truncated=result.truncated,
            truncation_reason=result.truncation_reason,
            considered_count=result.considered_count,
            elapsed_ms=result.elapsed_ms,
            capability=capability,
        )

    def _query_backend(
        self,
        *,
        graph: ProgramGraph,
        projection: GraphProjection,
        query: GraphProjectionQuery,
        bounds: GraphProjectionBounds,
        capability: GraphProjectionCapability,
    ) -> GraphQueryResult | None:
        backend = self._cached_backend if self._backend is None else self._backend
        surface = self._cached_surface
        if backend is None:
            # Ensure probe populated caches.
            self.probe()
            backend = self._cached_backend if self._backend is None else self._backend
            surface = self._cached_surface
        if backend is None:
            return None

        started = self._clock()
        timeout_s = bounds.timeout_ms / 1000.0

        def invoke() -> Any:
            return self._dispatch_backend(backend, surface, query, bounds)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(invoke)
                raw = future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            return GraphQueryResult(
                query_id=query.query_id,
                projection_id=projection.projection_id,
                forest_id=projection.forest_id,
                status=ProjectionStatus.TIMED_OUT,
                reason_code=ReasonCode.TIMEOUT,
                reason=f"backend query exceeded {bounds.timeout_ms}ms",
                mode=ProjectionMode.GRAPHRAG,
                references=(),
                bounds=bounds,
                ranking_method="backend",
                considered_count=0,
                elapsed_ms=int((self._clock() - started) * 1000),
                capability=capability,
            )
        except Exception as exc:  # noqa: BLE001 — typed inconclusive
            return GraphQueryResult(
                query_id=query.query_id,
                projection_id=projection.projection_id,
                forest_id=projection.forest_id,
                status=ProjectionStatus.INCONCLUSIVE,
                reason_code=ReasonCode.INCONCLUSIVE_BACKEND,
                reason=f"backend error: {type(exc).__name__}: {exc}"[
                    : DEFAULT_MAX_REASON_BYTES
                ],
                mode=ProjectionMode.GRAPHRAG,
                references=(),
                bounds=bounds,
                ranking_method="backend",
                considered_count=0,
                elapsed_ms=int((self._clock() - started) * 1000),
                capability=capability,
            )

        refs, poison = _sanitize_backend_hits(
            raw, projection=projection, graph=graph, bounds=bounds
        )
        elapsed_ms = int((self._clock() - started) * 1000)
        if poison is not None:
            return GraphQueryResult(
                query_id=query.query_id,
                projection_id=projection.projection_id,
                forest_id=projection.forest_id,
                status=ProjectionStatus.POISONED,
                reason_code=ReasonCode.POISONED_BACKEND_RESULT,
                reason=f"backend result rejected: {poison}"[
                    : DEFAULT_MAX_REASON_BYTES
                ],
                mode=ProjectionMode.GRAPHRAG,
                references=(),
                bounds=bounds,
                ranking_method="backend",
                considered_count=0,
                elapsed_ms=elapsed_ms,
                capability=capability,
            )

        if not refs:
            # Empty backend result is not poison; fall through to local.
            return None

        return GraphQueryResult(
            query_id=query.query_id,
            projection_id=projection.projection_id,
            forest_id=projection.forest_id,
            status=ProjectionStatus.COMPLETED,
            reason_code=ReasonCode.BOUNDED_QUERY,
            reason="bounded GraphRAG ranking over canonical projection",
            mode=ProjectionMode.GRAPHRAG,
            references=refs,
            bounds=bounds,
            ranking_method="backend_filtered_canonical",
            truncated=False,
            considered_count=len(refs),
            elapsed_ms=elapsed_ms,
            capability=capability,
        )

    def _dispatch_backend(
        self,
        backend: Any,
        surface: _BackendSurface | None,
        query: GraphProjectionQuery,
        bounds: GraphProjectionBounds,
    ) -> Any:
        max_results = (
            query.max_results
            if query.max_results is not None
            else bounds.max_results
        )
        max_hops = (
            query.max_hops if query.max_hops is not None else bounds.max_hops
        )
        kwargs_candidates = (
            {
                "query_text": query.text,
                "top_k": max_results,
                "max_graph_hops": max_hops,
                "max_nodes_visited": bounds.max_items,
                "max_edges_traversed": bounds.max_items * 4,
            },
            {
                "query": query.text,
                "top_k": max_results,
                "max_hops": max_hops,
                "limit": max_results,
            },
            {
                "text": query.text,
                "max_results": max_results,
                "max_hops": max_hops,
            },
        )

        # Prefer explicit method names from the probed surface.
        method_names: list[str] = []
        if surface is not None:
            method_names.append(surface.method)
        method_names.extend(["query", "search", "retrieve", "rank"])

        if callable(backend) and not any(
            callable(getattr(backend, name, None)) for name in method_names
        ):
            # Bare callable backend.
            try:
                return backend(
                    query.text,
                    top_k=max_results,
                    max_graph_hops=max_hops,
                    limit=max_results,
                )
            except TypeError:
                return backend(query.text)

        last_error: Exception | None = None
        for method_name in method_names:
            method = getattr(backend, method_name, None)
            if not callable(method):
                # Class-like module: try constructing a default engine.
                continue
            for kwargs in kwargs_candidates:
                try:
                    params = _signature_params(method)
                except ProgramGraphProviderError:
                    params = frozenset()
                filtered = {
                    key: value
                    for key, value in kwargs.items()
                    if not params or key in params or any(
                        p.startswith("*") for p in params
                    )
                }
                # Always try to pass the query text positionally if needed.
                try:
                    if "query_text" in filtered:
                        return method(**filtered)
                    if "query" in filtered:
                        return method(**filtered)
                    if "text" in filtered:
                        return method(**filtered)
                    return method(query.text, **{
                        k: v
                        for k, v in filtered.items()
                        if k not in {"query_text", "query", "text"}
                    })
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception:
                    raise

        # Factory-style: GraphRAGQueryEngine class on a module.
        for class_name in _QUERY_CLASS_NAMES:
            cls = getattr(backend, class_name, None)
            if not inspect.isclass(cls):
                continue
            try:
                instance = cls()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            query_method = getattr(instance, "query", None)
            if not callable(query_method):
                continue
            try:
                return query_method(
                    query_text=query.text,
                    top_k=max_results,
                    max_graph_hops=max_hops,
                )
            except TypeError:
                try:
                    return query_method(query.text, top_k=max_results)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
            except Exception:
                raise

        if last_error is not None:
            raise last_error
        raise ProgramGraphProviderError(
            "backend has no dispatchable query method"
        )

    def project_and_query(
        self,
        graph: ProgramGraph | Mapping[str, Any],
        query: GraphProjectionQuery | str | Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[GraphProjection, GraphQueryResult]:
        """Project then query in one call; returns both artifacts."""

        projection = self.project(graph, bounds=kwargs.get("bounds"))
        result = self.query(
            graph,
            query,
            projection=projection,
            bounds=kwargs.get("bounds"),
            use_backend=kwargs.get("use_backend"),
        )
        return projection, result


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_OPTIONAL_ROOT",
    "DEFAULT_TIMEOUT_MS",
    "IPFS_DATASETS_PROGRAM_GRAPH_PROTOCOL_VERSION",
    "IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_ID",
    "IPFS_DATASETS_PROGRAM_GRAPH_PROVIDER_VERSION",
    "PROVIDER_CAPABILITY_SCHEMA",
    "PROVIDER_CHUNK_SCHEMA",
    "PROVIDER_PROJECTION_SCHEMA",
    "PROVIDER_QUERY_SCHEMA",
    "PROVIDER_RESULT_SCHEMA",
    "CapabilityHealth",
    "GraphProjection",
    "GraphProjectionBounds",
    "GraphProjectionCapability",
    "GraphProjectionPolicy",
    "GraphProjectionQuery",
    "GraphQueryResult",
    "IpfsDatasetsProgramGraphProvider",
    "ProjectedChunk",
    "ProgramGraphProviderBoundsError",
    "ProgramGraphProviderError",
    "ProgramGraphProviderPoisonError",
    "ProjectionMode",
    "ProjectionStatus",
    "ProvenanceLink",
    "RankedGraphReference",
    "ReasonCode",
    "inspect_program_graph_provider_capability",
    "project_program_graph_local",
    "rank_projection_local",
]
