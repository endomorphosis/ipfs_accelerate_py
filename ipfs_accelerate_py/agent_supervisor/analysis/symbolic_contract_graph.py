"""Typed, content-addressed contract graph and bounded GraphRAG projection.

``SymbolicContractGraph@1`` is the graph boundary between the complete
repository index and later contract/proof compilation.  Source observations
are represented by immutable typed nodes and edges.  Every record binds the
exact snapshot, a provenance receipt, an authority class, a schema version,
and a strict ``ContentIdentity@1`` identity.

Candidate retrieval is deliberately separate from graph authority.  It emits
compact, bounded references and context-only edge projections.  Proof callers
must subsequently request deterministic typed closure over the pinned graph.
Incomplete mandatory closure is never returned as successful: missing
requirements and count/depth/byte truncation raise typed fail-closed errors.

The optional :mod:`ipfs_datasets_py` GraphRAG provider is imported only when a
caller explicitly enables it for a retrieval.  Importing this module,
projecting an index, and computing deterministic closure do not load that
provider.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from .content_identity_bridge import (
    CONTENT_IDENTITY_INTERFACE,
    ContentIdentity,
    identify_strict_artifact,
)


SYMBOLIC_CONTRACT_GRAPH_INTERFACE: Final = "SymbolicContractGraph@1"
SYMBOLIC_CONTRACT_GRAPH_VERSION: Final = "symbolic-contract-graph@1"
SYMBOLIC_CONTRACT_GRAPH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-graph@1"
)
SYMBOLIC_CONTRACT_NODE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-node@1"
)
SYMBOLIC_CONTRACT_EDGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-edge@1"
)
SYMBOLIC_CLOSURE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-closure@1"
)
GRAPHRAG_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-graphrag-receipt@1"
)
DATASETS_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-graph-projection@1"
)

DEFAULT_MAX_GRAPH_NODES: Final = 100_000
DEFAULT_MAX_GRAPH_EDGES: Final = 300_000
DEFAULT_MAX_CLOSURE_NODES: Final = 16_384
DEFAULT_MAX_CLOSURE_EDGES: Final = 65_536
DEFAULT_MAX_CLOSURE_DEPTH: Final = 256
DEFAULT_MAX_CLOSURE_BYTES: Final = 8 * 1024 * 1024
DEFAULT_MAX_CANDIDATES: Final = 256
DEFAULT_MAX_RESULTS: Final = 32
DEFAULT_MAX_RETRIEVAL_BYTES: Final = 65_536
DEFAULT_MAX_RETRIEVAL_HOPS: Final = 2
HARD_MAX_METADATA_BYTES: Final = 65_536
HARD_MAX_RETRIEVAL_RESULTS: Final = 1_024
HARD_MAX_RETRIEVAL_BYTES: Final = 4 * 1024 * 1024

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/+-]*")
_SOURCE_SUFFIXES = (
    ".d.ts",
    ".tsx",
    ".jsx",
    ".mts",
    ".cts",
    ".mjs",
    ".cjs",
    ".ts",
    ".js",
    ".py",
)
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "source",
        "source_body",
        "source_code",
        "source_text",
        "contents",
        "body",
        "ast",
        "ast_body",
        "model_output",
        "model_response",
        "prompt",
        "completion",
    }
)


class SymbolicContractGraphError(ValueError):
    """Base error for malformed or incomplete graph evidence."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "symbolic_contract_graph_error",
        receipt: "ContractClosureReceipt | None" = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.receipt = receipt


class GraphBoundsError(SymbolicContractGraphError):
    """A graph or result exceeded a hard configured bound."""


class GraphClosureTruncatedError(GraphBoundsError):
    """Mandatory closure could not be completed within its bounds."""


class MissingMandatoryEdgeError(SymbolicContractGraphError):
    """One or more declared mandatory dependencies are absent."""

    def __init__(
        self,
        message: str,
        *,
        missing: Sequence[str] = (),
        receipt: "ContractClosureReceipt | None" = None,
    ) -> None:
        super().__init__(
            message,
            reason_code="missing_mandatory_edge",
            receipt=receipt,
        )
        self.missing = tuple(sorted({str(item) for item in missing if str(item)}))


class GraphIntegrityError(SymbolicContractGraphError):
    """A supplied record, identity, snapshot binding, or edge is inconsistent."""


class OptionalDatasetsProviderError(SymbolicContractGraphError):
    """The explicitly requested optional datasets provider failed."""


class ContractNodeKind(str, Enum):
    REPOSITORY_SNAPSHOT = "repository_snapshot"
    FILE = "file"
    MODULE = "module"
    SYMBOL = "symbol"
    IMPORT = "import"
    CALL = "call"
    EFFECT = "effect"
    SCHEMA = "schema"
    INTERFACE = "interface"
    METHOD = "method"
    TOOL = "tool"
    HANDLER = "handler"
    TEST = "test"
    POLICY = "policy"
    TRANSPORT = "transport"
    PROVENANCE = "provenance"
    CONTRACT = "contract"
    UNRESOLVED = "unresolved"


class ContractEdgeKind(str, Enum):
    CONTAINS = "contains"
    DEFINES = "defines"
    DECLARES = "declares"
    IMPORTS = "imports"
    CALLS = "calls"
    HAS_EFFECT = "has_effect"
    IMPLEMENTS = "implements"
    PUBLISHES_SCHEMA = "publishes_schema"
    VALIDATES = "validates"
    TESTS = "tests"
    GOVERNS = "governs"
    AUTHORIZES = "authorizes"
    DISPATCHES_TO = "dispatches_to"
    HANDLES = "handles"
    TRANSPORTS = "transports"
    REFERENCES = "references"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"


class ContractAuthority(str, Enum):
    REVIEWED_CONTRACT = "reviewed_contract"
    SOURCE_OBSERVATION = "source_observation"
    CONTEXT_ONLY = "context_only"
    UNRESOLVED = "unresolved"
    NONE = "none"

    @property
    def authority_bearing(self) -> bool:
        return self in {
            ContractAuthority.REVIEWED_CONTRACT,
            ContractAuthority.SOURCE_OBSERVATION,
        }


class ContractProvenance(str, Enum):
    REPOSITORY_INDEX = "repository_index"
    AST = "ast"
    SCHEMA = "schema"
    REVIEWED_CONTRACT = "reviewed_contract"
    GRAPHRAG = "graphrag"
    DATASETS_GRAPHRAG = "datasets_graphrag"
    UNRESOLVED = "unresolved"


class ClosureDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class DatasetsProviderState(str, Enum):
    DISABLED = "disabled"
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphIntegrityError("canonical graph values must be finite")
        return value
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise GraphIntegrityError("canonical graph keys must be strings")
        return {
            key: _canonical_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=canonical_symbolic_json)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical_value(converter())
    raise GraphIntegrityError(
        f"unsupported canonical graph value: {type(value).__name__}"
    )


def canonical_symbolic_json(value: Any) -> str:
    """Return the graph's deterministic strict JSON representation."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return canonical_symbolic_json(value).encode("utf-8")


def _identity(value: Any) -> ContentIdentity:
    # The bridge is imported above, but its datasets and multiformats providers
    # remain lazy until this operation is actually requested.
    return identify_strict_artifact(_canonical_value(value))


def _identity_dict(identity: ContentIdentity) -> dict[str, Any]:
    return identity.to_dict(include_canonical_bytes=False)


def _required_text(value: Any, label: str, *, maximum: int = 1_024) -> str:
    text = str(value or "").strip()
    if not text:
        raise GraphIntegrityError(f"{label} is required")
    if len(text.encode("utf-8")) > maximum:
        raise GraphIntegrityError(f"{label} exceeds {maximum} UTF-8 bytes")
    return text


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _canonical_value(dict(value or {}))
    if not isinstance(normalized, dict):
        raise GraphIntegrityError("metadata must be a mapping")

    def inspect(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in _FORBIDDEN_METADATA_KEYS:
                    raise GraphIntegrityError(
                        f"source/model body key is forbidden in graph metadata: {key}"
                    )
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(normalized)
    if len(_canonical_bytes(normalized)) > HARD_MAX_METADATA_BYTES:
        raise GraphBoundsError("graph metadata exceeds its hard byte bound")
    return normalized


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise GraphIntegrityError(f"invalid {label}: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ContractGraphNode:
    """One typed graph entity bound to exact source evidence."""

    kind: ContractNodeKind | str
    key: str
    label: str
    snapshot_id: str
    provenance: ContractProvenance | str
    provenance_id: str
    authority: ContractAuthority | str
    version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _content_identity: ContentIdentity = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ContractNodeKind, "node kind")
        )
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, ContractProvenance, "node provenance"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ContractAuthority, "node authority"),
        )
        for name in ("key", "label", "snapshot_id", "provenance_id", "version"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), f"node {name}")
            )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.provenance in {
            ContractProvenance.GRAPHRAG,
            ContractProvenance.DATASETS_GRAPHRAG,
        } and self.authority is not ContractAuthority.CONTEXT_ONLY:
            raise GraphIntegrityError(
                "GraphRAG node provenance must be context-only"
            )
        if (
            self.kind is ContractNodeKind.UNRESOLVED
            and self.authority.authority_bearing
        ):
            raise GraphIntegrityError(
                "an unresolved node cannot carry graph authority"
            )
        object.__setattr__(
            self, "_content_identity", _identity(self._identity_payload())
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_NODE_SCHEMA,
            "kind": self.kind.value,
            "key": self.key,
            "label": self.label,
            "snapshot_id": self.snapshot_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "authority": self.authority.value,
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @property
    def content_identity(self) -> ContentIdentity:
        return self._content_identity

    @property
    def node_id(self) -> str:
        return f"sca-node:{self.content_identity.cid}"

    @property
    def identity(self) -> Mapping[str, Any]:
        return _identity_dict(self.content_identity)

    @property
    def authoritative(self) -> bool:
        return self.authority.authority_bearing

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "node_id": self.node_id,
            "identity": dict(self.identity),
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractGraphNode":
        schema = str(payload.get("schema") or SYMBOLIC_CONTRACT_NODE_SCHEMA)
        if schema != SYMBOLIC_CONTRACT_NODE_SCHEMA:
            raise GraphIntegrityError(f"unsupported node schema: {schema}")
        node = cls(
            kind=payload.get("kind", ""),
            key=str(payload.get("key") or ""),
            label=str(payload.get("label") or ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            provenance=payload.get("provenance", ""),
            provenance_id=str(payload.get("provenance_id") or ""),
            authority=payload.get("authority", ""),
            version=str(payload.get("version") or ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = str(payload.get("node_id") or "")
        if claimed and claimed != node.node_id:
            raise GraphIntegrityError("node identity does not match its payload")
        identity = payload.get("identity")
        if isinstance(identity, Mapping):
            if str(identity.get("cid") or "") != node.content_identity.cid:
                raise GraphIntegrityError(
                    "node ContentIdentity does not match its payload"
                )
        if (
            "authoritative" in payload
            and bool(payload["authoritative"]) != node.authoritative
        ):
            raise GraphIntegrityError("node authority flag is inconsistent")
        return node


@dataclass(frozen=True, slots=True)
class ContractGraphEdge:
    """One typed relationship with its own source-evidence identity."""

    source: str
    target: str
    kind: ContractEdgeKind | str
    snapshot_id: str
    provenance: ContractProvenance | str
    provenance_id: str
    authority: ContractAuthority | str
    version: str
    mandatory: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _content_identity: ContentIdentity = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ContractEdgeKind, "edge kind")
        )
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, ContractProvenance, "edge provenance"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ContractAuthority, "edge authority"),
        )
        for name in (
            "source",
            "target",
            "snapshot_id",
            "provenance_id",
            "version",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), f"edge {name}")
            )
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.provenance in {
            ContractProvenance.GRAPHRAG,
            ContractProvenance.DATASETS_GRAPHRAG,
        } and self.authority is not ContractAuthority.CONTEXT_ONLY:
            raise GraphIntegrityError(
                "GraphRAG edge provenance must be context-only"
            )
        if self.authority is ContractAuthority.CONTEXT_ONLY and self.mandatory:
            raise GraphIntegrityError(
                "context-only edges cannot be mandatory proof dependencies"
            )
        if self.mandatory and not self.authority.authority_bearing:
            raise GraphIntegrityError(
                "mandatory edges must carry reviewed or source-observation authority"
            )
        object.__setattr__(
            self, "_content_identity", _identity(self._identity_payload())
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_EDGE_SCHEMA,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "snapshot_id": self.snapshot_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "authority": self.authority.value,
            "version": self.version,
            "mandatory": self.mandatory,
            "metadata": dict(self.metadata),
        }

    @property
    def content_identity(self) -> ContentIdentity:
        return self._content_identity

    @property
    def edge_id(self) -> str:
        return f"sca-edge:{self.content_identity.cid}"

    @property
    def identity(self) -> Mapping[str, Any]:
        return _identity_dict(self.content_identity)

    @property
    def authoritative(self) -> bool:
        return self.authority.authority_bearing

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "edge_id": self.edge_id,
            "identity": dict(self.identity),
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractGraphEdge":
        schema = str(payload.get("schema") or SYMBOLIC_CONTRACT_EDGE_SCHEMA)
        if schema != SYMBOLIC_CONTRACT_EDGE_SCHEMA:
            raise GraphIntegrityError(f"unsupported edge schema: {schema}")
        edge = cls(
            source=str(payload.get("source") or ""),
            target=str(payload.get("target") or ""),
            kind=payload.get("kind", ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            provenance=payload.get("provenance", ""),
            provenance_id=str(payload.get("provenance_id") or ""),
            authority=payload.get("authority", ""),
            version=str(payload.get("version") or ""),
            mandatory=bool(payload.get("mandatory", True)),
            metadata=payload.get("metadata") or {},
        )
        claimed = str(payload.get("edge_id") or "")
        if claimed and claimed != edge.edge_id:
            raise GraphIntegrityError("edge identity does not match its payload")
        identity = payload.get("identity")
        if isinstance(identity, Mapping):
            if str(identity.get("cid") or "") != edge.content_identity.cid:
                raise GraphIntegrityError(
                    "edge ContentIdentity does not match its payload"
                )
        if (
            "authoritative" in payload
            and bool(payload["authoritative"]) != edge.authoritative
        ):
            raise GraphIntegrityError("edge authority flag is inconsistent")
        return edge


@dataclass(frozen=True, slots=True)
class MandatoryEdgeRequirement:
    """A dependency that must exist before this graph can be consumed."""

    source: str
    target: str
    kind: ContractEdgeKind | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _required_text(self.source, "requirement source")
        )
        object.__setattr__(
            self, "target", _required_text(self.target, "requirement target")
        )
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, ContractEdgeKind, "requirement edge kind"),
        )

    @property
    def requirement_id(self) -> str:
        return (
            f"{self.source}|{self.kind.value}|{self.target}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "requirement_id": self.requirement_id,
        }


@dataclass(frozen=True, slots=True)
class ClosureBounds:
    max_nodes: int = DEFAULT_MAX_CLOSURE_NODES
    max_edges: int = DEFAULT_MAX_CLOSURE_EDGES
    max_depth: int = DEFAULT_MAX_CLOSURE_DEPTH
    max_bytes: int = DEFAULT_MAX_CLOSURE_BYTES

    def __post_init__(self) -> None:
        for name, hard in (
            ("max_nodes", DEFAULT_MAX_GRAPH_NODES),
            ("max_edges", DEFAULT_MAX_GRAPH_EDGES),
            ("max_depth", DEFAULT_MAX_GRAPH_NODES),
            ("max_bytes", HARD_MAX_RETRIEVAL_BYTES * 8),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise GraphBoundsError(f"{name} must be a positive integer")
            if int(value) > hard:
                raise GraphBoundsError(f"{name} exceeds its hard bound {hard}")
            object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, int]:
        return {
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_depth": self.max_depth,
            "max_bytes": self.max_bytes,
        }


@dataclass(frozen=True, slots=True)
class ContractClosureReceipt:
    graph_root: str
    snapshot_id: str
    seed_ids: tuple[str, ...]
    direction: ClosureDirection | str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    paths: Mapping[str, tuple[str, ...]]
    bounds: ClosureBounds
    complete: bool
    truncated: bool
    missing_mandatory_edges: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION

    def __post_init__(self) -> None:
        for name in ("graph_root", "snapshot_id", "version"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), f"closure {name}")
            )
        object.__setattr__(
            self,
            "direction",
            _enum(self.direction, ClosureDirection, "closure direction"),
        )
        for name in ("seed_ids", "node_ids", "edge_ids"):
            object.__setattr__(
                self,
                name,
                tuple(sorted({str(item) for item in getattr(self, name) if str(item)})),
            )
        object.__setattr__(
            self,
            "paths",
            {
                str(key): tuple(value)
                for key, value in sorted(self.paths.items())
            },
        )
        object.__setattr__(
            self,
            "missing_mandatory_edges",
            tuple(sorted(set(self.missing_mandatory_edges))),
        )
        object.__setattr__(
            self, "reason_codes", tuple(sorted(set(self.reason_codes)))
        )
        if self.complete and (
            self.truncated or self.missing_mandatory_edges
        ):
            raise GraphIntegrityError(
                "a complete closure cannot be truncated or have missing edges"
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CLOSURE_SCHEMA,
            "interface": SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "seed_ids": list(self.seed_ids),
            "direction": self.direction.value,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "paths": {key: list(value) for key, value in self.paths.items()},
            "bounds": self.bounds.to_dict(),
            "complete": self.complete,
            "truncated": self.truncated,
            "missing_mandatory_edges": list(self.missing_mandatory_edges),
            "reason_codes": list(self.reason_codes),
            "version": self.version,
        }

    @property
    def closure_id(self) -> str:
        return f"sca-closure:{_identity(self._identity_payload()).cid}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "closure_id": self.closure_id,
        }


@dataclass(frozen=True, slots=True)
class GraphRAGLimits:
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_results: int = DEFAULT_MAX_RESULTS
    max_bytes: int = DEFAULT_MAX_RETRIEVAL_BYTES
    max_hops: int = DEFAULT_MAX_RETRIEVAL_HOPS

    def __post_init__(self) -> None:
        for name, hard in (
            ("max_candidates", DEFAULT_MAX_GRAPH_NODES),
            ("max_results", HARD_MAX_RETRIEVAL_RESULTS),
            ("max_bytes", HARD_MAX_RETRIEVAL_BYTES),
            ("max_hops", 32),
        ):
            value = getattr(self, name)
            minimum = 512 if name == "max_bytes" else 0 if name == "max_hops" else 1
            if isinstance(value, bool) or int(value) < minimum:
                raise GraphBoundsError(
                    f"{name} must be an integer of at least {minimum}"
                )
            if int(value) > hard:
                raise GraphBoundsError(f"{name} exceeds its hard bound {hard}")
            object.__setattr__(self, name, int(value))
        if self.max_results > self.max_candidates:
            raise GraphBoundsError(
                "max_results cannot exceed max_candidates"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_results": self.max_results,
            "max_bytes": self.max_bytes,
            "max_hops": self.max_hops,
        }


@dataclass(frozen=True, slots=True)
class GraphRAGCandidate:
    node_id: str
    kind: str
    label: str
    score: int
    snapshot_id: str
    source_identity: str
    path: str = ""
    symbol: str = ""
    reason_codes: tuple[str, ...] = ()
    provenance: ContractProvenance = ContractProvenance.GRAPHRAG
    authority: ContractAuthority = ContractAuthority.CONTEXT_ONLY
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION
    _content_identity: ContentIdentity = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "node_id",
            "kind",
            "label",
            "snapshot_id",
            "source_identity",
            "version",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), f"candidate {name}"),
            )
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, ContractProvenance, "candidate provenance"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ContractAuthority, "candidate authority"),
        )
        if self.authority is not ContractAuthority.CONTEXT_ONLY:
            raise GraphIntegrityError("GraphRAG candidates are context-only")
        object.__setattr__(self, "score", max(0, int(self.score)))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({str(item) for item in self.reason_codes if str(item)})),
        )
        object.__setattr__(
            self, "_content_identity", _identity(self._identity_payload())
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "score": self.score,
            "snapshot_id": self.snapshot_id,
            "source_identity": self.source_identity,
            "path": self.path,
            "symbol": self.symbol,
            "reason_codes": list(self.reason_codes),
            "provenance": self.provenance.value,
            "authority": self.authority.value,
            "version": self.version,
        }

    @property
    def candidate_id(self) -> str:
        return "sca-graphrag-candidate:" + self._content_identity.cid

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "candidate_id": self.candidate_id,
            "identity": _identity_dict(self._content_identity),
            "authoritative": False,
        }


@dataclass(frozen=True, slots=True)
class GraphRAGContextEdge:
    source: str
    target: str
    kind: str
    source_edge_id: str
    snapshot_id: str
    provenance: ContractProvenance = ContractProvenance.GRAPHRAG
    authority: ContractAuthority = ContractAuthority.CONTEXT_ONLY
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION
    _content_identity: ContentIdentity = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source",
            "target",
            "kind",
            "source_edge_id",
            "snapshot_id",
            "version",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), f"context edge {name}"),
            )
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, ContractProvenance, "context edge provenance"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ContractAuthority, "context edge authority"),
        )
        if self.authority is not ContractAuthority.CONTEXT_ONLY:
            raise GraphIntegrityError("GraphRAG context edges are context-only")
        object.__setattr__(
            self, "_content_identity", _identity(self._identity_payload())
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "source_edge_id": self.source_edge_id,
            "snapshot_id": self.snapshot_id,
            "provenance": self.provenance.value,
            "authority": self.authority.value,
            "version": self.version,
        }

    @property
    def content_identity(self) -> ContentIdentity:
        return self._content_identity

    @property
    def edge_id(self) -> str:
        return "sca-graphrag-edge:" + self.content_identity.cid

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "edge_id": self.edge_id,
            "identity": _identity_dict(self.content_identity),
            "authoritative": False,
            "mandatory": False,
        }


@dataclass(frozen=True, slots=True)
class GraphRAGReceipt:
    graph_root: str
    snapshot_id: str
    query: str
    candidates: tuple[GraphRAGCandidate, ...]
    context_edges: tuple[GraphRAGContextEdge, ...]
    limits: GraphRAGLimits
    considered_count: int
    eligible_count: int
    provider_state: DatasetsProviderState | str
    provider_id: str
    provider_reason: str
    truncated: bool
    dropped_count: int
    output_bytes: int = 0
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_state",
            _enum(self.provider_state, DatasetsProviderState, "provider state"),
        )
        for name in ("graph_root", "snapshot_id", "version"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), f"receipt {name}")
            )
        object.__setattr__(self, "query", " ".join(str(self.query).split()))
        object.__setattr__(
            self,
            "candidates",
            tuple(self.candidates),
        )
        object.__setattr__(
            self,
            "context_edges",
            tuple(sorted(self.context_edges, key=lambda item: item.edge_id)),
        )
        for name in (
            "considered_count",
            "eligible_count",
            "dropped_count",
            "output_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise GraphIntegrityError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)

    def _identity_payload(self, *, output_bytes: int | None = None) -> dict[str, Any]:
        return {
            "schema": GRAPHRAG_RECEIPT_SCHEMA,
            "interface": SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "query": self.query,
            "candidates": [item.to_dict() for item in self.candidates],
            "context_edges": [item.to_dict() for item in self.context_edges],
            "limits": self.limits.to_dict(),
            "considered_count": self.considered_count,
            "eligible_count": self.eligible_count,
            "provider": {
                "state": self.provider_state.value,
                "provider_id": self.provider_id,
                "reason": self.provider_reason,
                "authority": ContractAuthority.CONTEXT_ONLY.value,
            },
            "truncated": self.truncated,
            "dropped_count": self.dropped_count,
            "output_bytes": (
                self.output_bytes if output_bytes is None else int(output_bytes)
            ),
            "version": self.version,
            "completion_authoritative": False,
        }

    @property
    def receipt_id(self) -> str:
        return (
            "sca-graphrag-receipt:"
            + _identity(self._identity_payload(output_bytes=0)).cid
        )

    @property
    def results(self) -> tuple[GraphRAGCandidate, ...]:
        return self.candidates

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_symbolic_json(self.to_dict())
        return json.dumps(
            _canonical_value(self.to_dict()),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )


class LazyDatasetsGraphRAGProvider:
    """Opt-in adapter whose module import occurs on the first retrieval only.

    The configured module may expose ``retrieve_symbolic_candidates``.  A
    caller may also inject a callable, which is useful for a reviewed provider
    adapter and for deterministic conformance tests.  Provider output only
    nominates existing node IDs; unknown IDs are discarded.
    """

    DEFAULT_MODULE: Final = (
        "ipfs_datasets_py.knowledge_graphs.query.unified_engine"
    )

    def __init__(
        self,
        *,
        module_name: str = DEFAULT_MODULE,
        candidate_callable: Callable[..., Iterable[Any]] | None = None,
        provider_id: str = "ipfs-datasets-graphrag@proposal",
    ) -> None:
        self.module_name = _required_text(module_name, "datasets module name")
        self.candidate_callable = candidate_callable
        self.provider_id = _required_text(provider_id, "datasets provider id")
        self._module: Any = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def retrieve(
        self,
        query: str,
        *,
        graph_records: Mapping[str, Any],
        limit: int,
    ) -> tuple[str, ...]:
        callback = self.candidate_callable
        if callback is None:
            try:
                self._module = importlib.import_module(self.module_name)
                self._loaded = True
            except Exception as exc:  # noqa: BLE001 - optional provider boundary
                self._loaded = True
                raise OptionalDatasetsProviderError(
                    f"optional datasets GraphRAG provider unavailable: {exc}",
                    reason_code="datasets_provider_unavailable",
                ) from exc
            callback = getattr(
                self._module, "retrieve_symbolic_candidates", None
            )
            if not callable(callback):
                raise OptionalDatasetsProviderError(
                    "datasets provider does not expose "
                    "retrieve_symbolic_candidates",
                    reason_code="datasets_provider_capability_missing",
                )
        else:
            self._loaded = True
        try:
            values = callback(
                query=query,
                graph=graph_records,
                limit=int(limit),
            )
        except TypeError:
            values = callback(query, graph_records, int(limit))
        except Exception as exc:  # noqa: BLE001 - normalize proposal provider
            raise OptionalDatasetsProviderError(
                f"datasets provider retrieval failed: {exc}",
                reason_code="datasets_provider_failed",
            ) from exc
        result: list[str] = []
        for value in values or ():
            if isinstance(value, Mapping):
                node_id = str(
                    value.get("node_id")
                    or value.get("id")
                    or value.get("source_id")
                    or ""
                )
            else:
                node_id = str(value or "")
            if node_id and node_id not in result:
                result.append(node_id)
            if len(result) >= limit:
                break
        return tuple(result)


class BoundedGraphRAGRetriever:
    """Small graph-bound facade over :meth:`retrieve_candidates`.

    This keeps the established ``BoundedGraphRAGRetriever`` interface shape
    available to consumers while making the pinned symbolic graph root an
    explicit constructor dependency.
    """

    def __init__(
        self,
        graph: "SymbolicContractGraph",
        *,
        datasets_provider: LazyDatasetsGraphRAGProvider | bool | None = None,
    ) -> None:
        if not isinstance(graph, SymbolicContractGraph):
            raise GraphIntegrityError(
                "BoundedGraphRAGRetriever requires a SymbolicContractGraph"
            )
        self.graph = graph
        self.datasets_provider = datasets_provider

    def retrieve(
        self,
        query: str,
        *,
        limits: GraphRAGLimits | Mapping[str, Any] | None = None,
    ) -> GraphRAGReceipt:
        return self.graph.retrieve_candidates(
            query,
            limits=limits,
            datasets_provider=self.datasets_provider,
        )

    query = retrieve
    search = retrieve


@dataclass(frozen=True)
class SymbolicContractGraph:
    """Canonical typed graph pinned to one repository snapshot."""

    snapshot_id: str
    source_index_id: str
    nodes: tuple[ContractGraphNode, ...] = ()
    edges: tuple[ContractGraphEdge, ...] = ()
    mandatory_requirements: tuple[MandatoryEdgeRequirement, ...] = ()
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION
    _content_identity: ContentIdentity = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "source_index_id", "version"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), f"graph {name}")
            )
        node_map: dict[str, ContractGraphNode] = {}
        for value in self.nodes:
            node = (
                value
                if isinstance(value, ContractGraphNode)
                else ContractGraphNode.from_dict(value)
            )
            if node.snapshot_id != self.snapshot_id:
                raise GraphIntegrityError(
                    f"node {node.node_id} is bound to a foreign snapshot"
                )
            previous = node_map.get(node.node_id)
            if previous is not None and previous.to_dict() != node.to_dict():
                raise GraphIntegrityError(
                    f"conflicting graph node {node.node_id}"
                )
            node_map[node.node_id] = node
        if len(node_map) > DEFAULT_MAX_GRAPH_NODES:
            raise GraphBoundsError("symbolic graph has too many nodes")

        edge_map: dict[str, ContractGraphEdge] = {}
        for value in self.edges:
            edge = (
                value
                if isinstance(value, ContractGraphEdge)
                else ContractGraphEdge.from_dict(value)
            )
            if edge.snapshot_id != self.snapshot_id:
                raise GraphIntegrityError(
                    f"edge {edge.edge_id} is bound to a foreign snapshot"
                )
            source = node_map.get(edge.source)
            target = node_map.get(edge.target)
            if source is None or target is None:
                missing = [
                    node_id
                    for node_id, node in (
                        (edge.source, source),
                        (edge.target, target),
                    )
                    if node is None
                ]
                if edge.mandatory:
                    raise MissingMandatoryEdgeError(
                        f"mandatory edge {edge.edge_id} references unknown nodes",
                        missing=missing,
                    )
                raise GraphIntegrityError(
                    f"edge {edge.edge_id} references unknown nodes: {missing}"
                )
            if edge.mandatory and (
                not source.authoritative or not target.authoritative
            ):
                raise GraphIntegrityError(
                    "mandatory edge endpoints must both be authority-bearing"
                )
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_GRAPH_EDGES:
            raise GraphBoundsError("symbolic graph has too many edges")

        requirements = tuple(
            sorted(
                (
                    item
                    if isinstance(item, MandatoryEdgeRequirement)
                    else MandatoryEdgeRequirement(**dict(item))
                    for item in self.mandatory_requirements
                ),
                key=lambda item: item.requirement_id,
            )
        )
        present = {
            (edge.source, edge.target, edge.kind)
            for edge in edge_map.values()
            if edge.mandatory and edge.authoritative
        }
        missing_requirements = tuple(
            item.requirement_id
            for item in requirements
            if (item.source, item.target, item.kind) not in present
        )
        if missing_requirements:
            raise MissingMandatoryEdgeError(
                "declared mandatory graph edges are missing",
                missing=missing_requirements,
            )

        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        object.__setattr__(self, "mandatory_requirements", requirements)
        object.__setattr__(
            self, "_content_identity", _identity(self._root_payload())
        )

    def _root_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_GRAPH_SCHEMA,
            "interface": SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "source_index_id": self.source_index_id,
            "node_identities": [
                node.content_identity.cid for node in self.nodes
            ],
            "edge_identities": [
                edge.content_identity.cid for edge in self.edges
            ],
            "mandatory_requirements": [
                item.to_dict() for item in self.mandatory_requirements
            ],
        }

    @property
    def content_identity(self) -> ContentIdentity:
        return self._content_identity

    @property
    def root_id(self) -> str:
        return self.content_identity.cid

    @property
    def graph_root(self) -> str:
        return self.root_id

    @property
    def graph_id(self) -> str:
        return f"sca-graph:{self.root_id}"

    @property
    def identity(self) -> Mapping[str, Any]:
        return _identity_dict(self.content_identity)

    def node(self, node_id: str) -> ContractGraphNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def nodes_by_kind(
        self, kind: ContractNodeKind | str
    ) -> tuple[ContractGraphNode, ...]:
        expected = _enum(kind, ContractNodeKind, "node kind")
        return tuple(node for node in self.nodes if node.kind is expected)

    def edges_by_kind(
        self, kind: ContractEdgeKind | str
    ) -> tuple[ContractGraphEdge, ...]:
        expected = _enum(kind, ContractEdgeKind, "edge kind")
        return tuple(edge for edge in self.edges if edge.kind is expected)

    def _missing_requirements(
        self,
        requirements: Iterable[MandatoryEdgeRequirement | Mapping[str, Any]],
    ) -> tuple[str, ...]:
        present = {
            (edge.source, edge.target, edge.kind)
            for edge in self.edges
            if edge.mandatory and edge.authoritative
        }
        missing: list[str] = []
        for value in requirements:
            requirement = (
                value
                if isinstance(value, MandatoryEdgeRequirement)
                else MandatoryEdgeRequirement(**dict(value))
            )
            if (
                requirement.source,
                requirement.target,
                requirement.kind,
            ) not in present:
                missing.append(requirement.requirement_id)
        return tuple(sorted(set(missing)))

    def closure(
        self,
        seed_ids: str | Iterable[str],
        *,
        direction: ClosureDirection | str = ClosureDirection.FORWARD,
        edge_kinds: Iterable[ContractEdgeKind | str] | None = None,
        mandatory_only: bool = False,
        bounds: ClosureBounds | None = None,
        required_edges: Iterable[
            MandatoryEdgeRequirement | Mapping[str, Any]
        ] = (),
    ) -> ContractClosureReceipt:
        """Compute exact deterministic forward or reverse typed-edge closure.

        The result is successful only if every selected edge can be traversed
        within bounds and all explicitly required mandatory edges exist.
        """

        limits = bounds or ClosureBounds()
        mode = _enum(direction, ClosureDirection, "closure direction")
        seeds = (
            (seed_ids,)
            if isinstance(seed_ids, str)
            else tuple(str(item) for item in seed_ids)
        )
        seeds = tuple(sorted({item for item in seeds if item}))
        if not seeds:
            raise GraphIntegrityError("closure requires at least one seed")
        node_ids = {node.node_id for node in self.nodes}
        unknown = tuple(sorted(set(seeds) - node_ids))
        requirements = (*self.mandatory_requirements, *tuple(required_edges))
        missing = (*unknown, *self._missing_requirements(requirements))
        if missing:
            receipt = ContractClosureReceipt(
                graph_root=self.root_id,
                snapshot_id=self.snapshot_id,
                seed_ids=seeds,
                direction=mode,
                node_ids=tuple(item for item in seeds if item in node_ids),
                edge_ids=(),
                paths={
                    item: (item,) for item in seeds if item in node_ids
                },
                bounds=limits,
                complete=False,
                truncated=False,
                missing_mandatory_edges=tuple(missing),
                reason_codes=("missing_mandatory_edge",),
            )
            raise MissingMandatoryEdgeError(
                "mandatory closure inputs are missing",
                missing=missing,
                receipt=receipt,
            )

        selected_kinds = (
            None
            if edge_kinds is None
            else frozenset(
                _enum(item, ContractEdgeKind, "closure edge kind")
                for item in edge_kinds
            )
        )
        adjacency: dict[str, list[tuple[ContractGraphEdge, str]]] = {}
        for edge in self.edges:
            if mandatory_only and not edge.mandatory:
                continue
            if selected_kinds is not None and edge.kind not in selected_kinds:
                continue
            source, target = (
                (edge.source, edge.target)
                if mode is ClosureDirection.FORWARD
                else (edge.target, edge.source)
            )
            adjacency.setdefault(source, []).append((edge, target))
        for values in adjacency.values():
            values.sort(
                key=lambda item: (
                    item[0].kind.value,
                    item[1],
                    item[0].edge_id,
                )
            )

        paths: dict[str, tuple[str, ...]] = {
            seed: (seed,) for seed in seeds
        }
        depths = {seed: 0 for seed in seeds}
        included_edges: set[str] = set()
        queue: deque[str] = deque(seeds)

        def truncated(reason: str) -> None:
            receipt = ContractClosureReceipt(
                graph_root=self.root_id,
                snapshot_id=self.snapshot_id,
                seed_ids=seeds,
                direction=mode,
                node_ids=tuple(paths),
                edge_ids=tuple(included_edges),
                paths=paths,
                bounds=limits,
                complete=False,
                truncated=True,
                reason_codes=(reason,),
            )
            raise GraphClosureTruncatedError(
                f"mandatory closure {reason.replace('_', ' ')}",
                reason_code=reason,
                receipt=receipt,
            )

        if len(paths) > limits.max_nodes:
            truncated("max_nodes_exceeded")
        while queue:
            current = queue.popleft()
            for edge, target in adjacency.get(current, ()):
                depth = depths[current] + 1
                if depth > limits.max_depth:
                    truncated("max_depth_exceeded")
                included_edges.add(edge.edge_id)
                if len(included_edges) > limits.max_edges:
                    truncated("max_edges_exceeded")
                candidate_path = (*paths[current], target)
                previous = paths.get(target)
                if previous is None:
                    paths[target] = candidate_path
                    depths[target] = depth
                    if len(paths) > limits.max_nodes:
                        truncated("max_nodes_exceeded")
                    queue.append(target)
                elif (len(candidate_path), candidate_path) < (
                    len(previous),
                    previous,
                ):
                    paths[target] = candidate_path
                    depths[target] = depth

        receipt = ContractClosureReceipt(
            graph_root=self.root_id,
            snapshot_id=self.snapshot_id,
            seed_ids=seeds,
            direction=mode,
            node_ids=tuple(paths),
            edge_ids=tuple(included_edges),
            paths=paths,
            bounds=limits,
            complete=True,
            truncated=False,
            reason_codes=("exact_typed_closure",),
        )
        if len(_canonical_bytes(receipt.to_dict())) > limits.max_bytes:
            truncated("max_bytes_exceeded")
        return receipt

    def mandatory_closure(
        self,
        seed_ids: str | Iterable[str],
        **kwargs: Any,
    ) -> ContractClosureReceipt:
        kwargs["mandatory_only"] = True
        return self.closure(seed_ids, **kwargs)

    def forward_mandatory_closure(
        self, seed_ids: str | Iterable[str], **kwargs: Any
    ) -> ContractClosureReceipt:
        kwargs["direction"] = ClosureDirection.FORWARD
        kwargs["mandatory_only"] = True
        return self.closure(seed_ids, **kwargs)

    def reverse_mandatory_closure(
        self, seed_ids: str | Iterable[str], **kwargs: Any
    ) -> ContractClosureReceipt:
        kwargs["direction"] = ClosureDirection.REVERSE
        kwargs["mandatory_only"] = True
        return self.closure(seed_ids, **kwargs)

    def forward_closure(
        self, seed_ids: str | Iterable[str], **kwargs: Any
    ) -> ContractClosureReceipt:
        kwargs["direction"] = ClosureDirection.FORWARD
        return self.closure(seed_ids, **kwargs)

    def reverse_closure(
        self, seed_ids: str | Iterable[str], **kwargs: Any
    ) -> ContractClosureReceipt:
        kwargs["direction"] = ClosureDirection.REVERSE
        return self.closure(seed_ids, **kwargs)

    closure_forward = forward_closure
    closure_reverse = reverse_closure

    def _search_document(self, node: ContractGraphNode) -> str:
        safe_fields = (
            "path",
            "module",
            "symbol",
            "target",
            "language",
            "role",
        )
        values = [
            node.kind.value,
            node.key,
            node.label,
            *(
                str(node.metadata.get(name) or "")
                for name in safe_fields
            ),
        ]
        return " ".join(value for value in values if value)

    def retrieve_candidates(
        self,
        query: str,
        *,
        limits: GraphRAGLimits | Mapping[str, Any] | None = None,
        datasets_provider: LazyDatasetsGraphRAGProvider | bool | None = None,
    ) -> GraphRAGReceipt:
        """Return bounded context-only candidates with an auditable receipt."""

        bounds = (
            limits
            if isinstance(limits, GraphRAGLimits)
            else GraphRAGLimits(**dict(limits or {}))
        )
        normalized_query = " ".join(str(query or "").split())
        query_tokens = {
            token.casefold() for token in _TOKEN_RE.findall(normalized_query)
        }
        scores: dict[str, tuple[int, tuple[str, ...]]] = {}
        node_map = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            document = self._search_document(node)
            lowered = document.casefold()
            document_tokens = {
                token.casefold() for token in _TOKEN_RE.findall(document)
            }
            overlap = sorted(query_tokens.intersection(document_tokens))
            score = 0
            reasons: list[str] = []
            if not query_tokens:
                score = 1
                reasons.append("all_nodes")
            if normalized_query and normalized_query.casefold() in lowered:
                score = max(score, 100)
                reasons.append("exact_phrase")
            if overlap:
                coverage = len(overlap) / max(1, len(query_tokens))
                score = max(score, 50 + int(40 * coverage))
                reasons.append(f"token_overlap:{len(overlap)}")
            if score:
                scores[node.node_id] = (score, tuple(reasons))

        provider_state = DatasetsProviderState.DISABLED
        provider_id = ""
        provider_reason = "not_requested"
        if datasets_provider:
            provider = (
                LazyDatasetsGraphRAGProvider()
                if datasets_provider is True
                else datasets_provider
            )
            if not isinstance(provider, LazyDatasetsGraphRAGProvider):
                raise GraphIntegrityError(
                    "datasets_provider must be a lazy provider or true"
                )
            provider_id = provider.provider_id
            try:
                provider_nodes = provider.retrieve(
                    normalized_query,
                    graph_records=self.to_datasets_projection(),
                    limit=bounds.max_candidates,
                )
                provider_state = DatasetsProviderState.HEALTHY
                provider_reason = "candidate_seeds_received"
                for node_id in provider_nodes:
                    if node_id in node_map:
                        old_score, old_reasons = scores.get(node_id, (0, ()))
                        scores[node_id] = (
                            max(old_score, 80),
                            tuple(
                                dict.fromkeys(
                                    (*old_reasons, "datasets_candidate")
                                )
                            ),
                        )
            except OptionalDatasetsProviderError as exc:
                provider_state = (
                    DatasetsProviderState.UNAVAILABLE
                    if exc.reason_code
                    in {
                        "datasets_provider_unavailable",
                        "datasets_provider_capability_missing",
                    }
                    else DatasetsProviderState.FAILED
                )
                provider_reason = f"{exc.reason_code}:{exc}"

        ranked_seed_ids = [
            node_id
            for node_id, _ in sorted(
                scores.items(),
                key=lambda item: (-item[1][0], item[0]),
            )
        ]
        total_eligible = len(ranked_seed_ids)
        selected_ids = ranked_seed_ids[: bounds.max_candidates]

        # Expand a deterministic undirected neighborhood for candidate context.
        neighborhood: dict[str, list[str]] = {}
        for edge in self.edges:
            neighborhood.setdefault(edge.source, []).append(edge.target)
            neighborhood.setdefault(edge.target, []).append(edge.source)
        for values in neighborhood.values():
            values.sort()
        distances = {node_id: 0 for node_id in selected_ids}
        queue: deque[str] = deque(selected_ids)
        while queue and len(distances) < bounds.max_candidates:
            current = queue.popleft()
            depth = distances[current]
            if depth >= bounds.max_hops:
                continue
            for neighbor in neighborhood.get(current, ()):
                if neighbor in distances:
                    continue
                distances[neighbor] = depth + 1
                old_score, old_reasons = scores.get(neighbor, (0, ()))
                scores[neighbor] = (
                    old_score,
                    tuple(dict.fromkeys((*old_reasons, f"graph_hop:{depth + 1}"))),
                )
                queue.append(neighbor)
                if len(distances) >= bounds.max_candidates:
                    break

        ranked = sorted(
            distances,
            key=lambda node_id: (
                -scores.get(node_id, (0, ()))[0],
                distances[node_id],
                node_id,
            ),
        )
        result_ids = ranked[: bounds.max_results]

        def candidate(node_id: str) -> GraphRAGCandidate:
            node = node_map[node_id]
            return GraphRAGCandidate(
                node_id=node.node_id,
                kind=node.kind.value,
                label=node.label[:320],
                score=scores.get(node_id, (0, ()))[0],
                snapshot_id=self.snapshot_id,
                source_identity=node.content_identity.cid,
                path=str(node.metadata.get("path") or "")[:320],
                symbol=str(node.metadata.get("symbol") or "")[:320],
                reason_codes=scores.get(node_id, (0, ()))[1],
            )

        candidates = [candidate(node_id) for node_id in result_ids]
        result_set = set(result_ids)
        context_edges = [
            GraphRAGContextEdge(
                source=edge.source,
                target=edge.target,
                kind=edge.kind.value,
                source_edge_id=edge.edge_id,
                snapshot_id=self.snapshot_id,
                provenance=(
                    ContractProvenance.DATASETS_GRAPHRAG
                    if provider_state is DatasetsProviderState.HEALTHY
                    else ContractProvenance.GRAPHRAG
                ),
            )
            for edge in self.edges
            if edge.source in result_set and edge.target in result_set
        ]

        neighborhood_truncated = bool(
            len(distances) >= bounds.max_candidates
            and any(
                distances[node_id] < bounds.max_hops
                and any(
                    neighbor not in distances
                    for neighbor in neighborhood.get(node_id, ())
                )
                for node_id in distances
            )
        )
        truncated = (
            total_eligible > bounds.max_candidates
            or len(ranked) > bounds.max_results
            or neighborhood_truncated
        )
        dropped = max(
            int(neighborhood_truncated),
            len(ranked) - len(candidates),
        )

        def make(
            values: Sequence[GraphRAGCandidate],
            *,
            byte_drops: int,
            output_bytes: int,
        ) -> GraphRAGReceipt:
            ids = {item.node_id for item in values}
            edges = tuple(
                edge
                for edge in context_edges
                if edge.source in ids and edge.target in ids
            )
            return GraphRAGReceipt(
                graph_root=self.root_id,
                snapshot_id=self.snapshot_id,
                query=normalized_query,
                candidates=tuple(values),
                context_edges=edges,
                limits=bounds,
                considered_count=len(self.nodes),
                eligible_count=total_eligible,
                provider_state=provider_state,
                provider_id=provider_id,
                provider_reason=provider_reason,
                truncated=truncated or byte_drops > 0,
                dropped_count=dropped + byte_drops,
                output_bytes=output_bytes,
            )

        byte_drops = 0
        output_bytes = 0
        receipt = make(candidates, byte_drops=byte_drops, output_bytes=0)
        while (
            len(receipt.to_json().encode("utf-8")) > bounds.max_bytes
            and candidates
        ):
            candidates.pop()
            byte_drops += 1
            receipt = make(candidates, byte_drops=byte_drops, output_bytes=0)
        for _ in range(8):
            size = len(receipt.to_json().encode("utf-8"))
            if size == output_bytes:
                break
            output_bytes = size
            receipt = make(
                candidates,
                byte_drops=byte_drops,
                output_bytes=output_bytes,
            )
        final_size = len(receipt.to_json().encode("utf-8"))
        if final_size > bounds.max_bytes:
            raise GraphBoundsError(
                "max_bytes is too small for the mandatory GraphRAG receipt"
            )
        if final_size != receipt.output_bytes:
            receipt = make(
                candidates,
                byte_drops=byte_drops,
                output_bytes=final_size,
            )
        return receipt

    candidate_retrieval = retrieve_candidates
    search = retrieve_candidates

    def to_datasets_projection(self) -> dict[str, Any]:
        """Return provider-neutral KG rows without importing datasets."""

        return {
            "schema": DATASETS_PROJECTION_SCHEMA,
            "graph_root": self.root_id,
            "snapshot_id": self.snapshot_id,
            "version": self.version,
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.kind.value,
                    "label": node.label,
                    "properties": {
                        "content_identity": node.content_identity.cid,
                        "provenance": node.provenance.value,
                        "provenance_id": node.provenance_id,
                        "source_authority": node.authority.value,
                        "snapshot_id": node.snapshot_id,
                        "version": node.version,
                        **dict(node.metadata),
                    },
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.kind.value,
                    "properties": {
                        "content_identity": edge.content_identity.cid,
                        "provenance": edge.provenance.value,
                        "provenance_id": edge.provenance_id,
                        "source_authority": edge.authority.value,
                        "snapshot_id": edge.snapshot_id,
                        "version": edge.version,
                        "mandatory": edge.mandatory,
                        **dict(edge.metadata),
                    },
                }
                for edge in self.edges
            ],
        }

    def to_code_evidence_graph(self) -> Any:
        """Project into the established ``CodeEvidenceGraph`` container.

        The established graph has a smaller historical type vocabulary, so the
        exact SCA type, identity, authority, snapshot, and version remain in
        each durable evidence record.  This adapter does not promote GraphRAG
        context.
        """

        from .code_evidence_graph import (
            CodeEvidenceGraph,
            EvidenceEdgeKind,
            EvidenceNode,
            EvidenceNodeKind,
            EvidenceProvenance,
            ProvenanceEdge,
        )

        evidence_nodes: list[Any] = []
        id_map: dict[str, str] = {}
        for node in self.nodes:
            kind = (
                EvidenceNodeKind.SYMBOL
                if node.kind is ContractNodeKind.SYMBOL
                else EvidenceNodeKind.AST_SCOPE
                if node.kind in {ContractNodeKind.FILE, ContractNodeKind.MODULE}
                else EvidenceNodeKind.EVIDENCE
            )
            evidence = EvidenceNode(
                kind=kind,
                record_key=node.node_id,
                provenance=EvidenceProvenance.AST,
                record=node.to_dict(),
                tree_id=self.snapshot_id,
                symbol=(
                    str(node.metadata.get("symbol") or node.label)
                    if node.kind is ContractNodeKind.SYMBOL
                    else ""
                ),
            )
            evidence_nodes.append(evidence)
            id_map[node.node_id] = evidence.node_id
        evidence_edges = [
            ProvenanceEdge(
                source=id_map[edge.source],
                target=id_map[edge.target],
                kind=EvidenceEdgeKind.DERIVED_FROM,
                provenance=EvidenceProvenance.AST,
                provenance_record_id=edge.provenance_id,
                metadata={
                    "symbolic_edge_id": edge.edge_id,
                    "symbolic_kind": edge.kind.value,
                    "authority": edge.authority.value,
                    "snapshot_id": edge.snapshot_id,
                    "version": edge.version,
                    "mandatory": edge.mandatory,
                },
            )
            for edge in self.edges
        ]
        return CodeEvidenceGraph(
            nodes=tuple(evidence_nodes),
            edges=tuple(evidence_edges),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_GRAPH_SCHEMA,
            "interface": SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
            "version": self.version,
            "graph_id": self.graph_id,
            "graph_root": self.root_id,
            "identity": dict(self.identity),
            "snapshot_id": self.snapshot_id,
            "source_index_id": self.source_index_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "mandatory_requirements": [
                item.to_dict() for item in self.mandatory_requirements
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_symbolic_json(self.to_dict())
        return json.dumps(
            _canonical_value(self.to_dict()),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicContractGraph":
        schema = str(payload.get("schema") or SYMBOLIC_CONTRACT_GRAPH_SCHEMA)
        if schema != SYMBOLIC_CONTRACT_GRAPH_SCHEMA:
            raise GraphIntegrityError(f"unsupported graph schema: {schema}")
        graph = cls(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            source_index_id=str(payload.get("source_index_id") or ""),
            nodes=tuple(
                ContractGraphNode.from_dict(item)
                for item in payload.get("nodes") or ()
            ),
            edges=tuple(
                ContractGraphEdge.from_dict(item)
                for item in payload.get("edges") or ()
            ),
            mandatory_requirements=tuple(
                MandatoryEdgeRequirement(
                    source=str(item.get("source") or ""),
                    target=str(item.get("target") or ""),
                    kind=item.get("kind", ""),
                )
                for item in payload.get("mandatory_requirements") or ()
            ),
            version=str(
                payload.get("version") or SYMBOLIC_CONTRACT_GRAPH_VERSION
            ),
        )
        claimed_root = str(payload.get("graph_root") or "")
        if claimed_root and claimed_root != graph.root_id:
            raise GraphIntegrityError("graph root does not match graph records")
        claimed_id = str(payload.get("graph_id") or "")
        if claimed_id and claimed_id != graph.graph_id:
            raise GraphIntegrityError("graph identity does not match graph records")
        identity = payload.get("identity")
        if isinstance(identity, Mapping):
            if str(identity.get("cid") or "") != graph.root_id:
                raise GraphIntegrityError(
                    "graph ContentIdentity does not match graph records"
                )
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "SymbolicContractGraph":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GraphIntegrityError("symbolic graph JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise GraphIntegrityError("symbolic graph JSON must be an object")
        return cls.from_dict(value)


class _GraphBuilder:
    def __init__(self, snapshot_id: str, source_index_id: str) -> None:
        self.snapshot_id = snapshot_id
        self.source_index_id = source_index_id
        self.nodes: dict[str, ContractGraphNode] = {}
        self.edges: dict[str, ContractGraphEdge] = {}

    def node(
        self,
        *,
        kind: ContractNodeKind,
        key: str,
        label: str,
        provenance: ContractProvenance,
        provenance_id: str,
        authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
        version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContractGraphNode:
        node = ContractGraphNode(
            kind=kind,
            key=key,
            label=label,
            snapshot_id=self.snapshot_id,
            provenance=provenance,
            provenance_id=provenance_id,
            authority=authority,
            version=version,
            metadata=metadata or {},
        )
        self.nodes.setdefault(node.node_id, node)
        return self.nodes[node.node_id]

    def edge(
        self,
        source: ContractGraphNode,
        target: ContractGraphNode,
        *,
        kind: ContractEdgeKind,
        provenance: ContractProvenance,
        provenance_id: str,
        authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
        mandatory: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContractGraphEdge:
        edge = ContractGraphEdge(
            source=source.node_id,
            target=target.node_id,
            kind=kind,
            snapshot_id=self.snapshot_id,
            provenance=provenance,
            provenance_id=provenance_id,
            authority=authority,
            version=SYMBOLIC_CONTRACT_GRAPH_VERSION,
            mandatory=mandatory,
            metadata=metadata or {},
        )
        self.edges.setdefault(edge.edge_id, edge)
        return self.edges[edge.edge_id]


def _module_name(path: str) -> str:
    value = str(path).replace("\\", "/")
    for suffix in _SOURCE_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    parts = list(PurePosixPath(value).parts)
    if parts and parts[-1] in {"__init__", "index"}:
        parts.pop()
    return ".".join(parts)


def _row_kind(row: Any) -> str:
    value = getattr(row, "disposition_kind", "")
    return str(getattr(value, "value", value) or "")


def _role_kind(symbol: str, path: str) -> ContractNodeKind | None:
    value = f"{path} {symbol}".casefold()
    leaf = symbol.rsplit(".", 1)[-1].casefold()
    if leaf.startswith("test") or "/test" in value or ".test." in value:
        return ContractNodeKind.TEST
    if "handler" in leaf or leaf.startswith("handle_"):
        return ContractNodeKind.HANDLER
    if "policy" in value or "authorize" in leaf or "permission" in leaf:
        return ContractNodeKind.POLICY
    if any(
        token in value
        for token in ("websocket", "libp2p", "stdio", "transport", "http")
    ):
        return ContractNodeKind.TRANSPORT
    if "tool" in leaf or leaf.startswith("register_"):
        return ContractNodeKind.TOOL
    return None


def project_repository_index(
    repository_index: Any,
    *,
    require_healthy: bool = True,
    additional_nodes: Iterable[ContractGraphNode | Mapping[str, Any]] = (),
    additional_edges: Iterable[ContractGraphEdge | Mapping[str, Any]] = (),
    mandatory_requirements: Iterable[
        MandatoryEdgeRequirement | Mapping[str, Any]
    ] = (),
) -> SymbolicContractGraph:
    """Project a complete ``RepositoryIndex`` into the typed contract graph."""

    snapshot_id = _required_text(
        getattr(repository_index, "snapshot_id", ""),
        "repository index snapshot_id",
    )
    index_id = _required_text(
        getattr(repository_index, "index_id", ""),
        "repository index index_id",
    )
    if require_healthy and not bool(
        getattr(repository_index, "safe_for_completion_reasoning", False)
    ):
        raise GraphIntegrityError(
            "repository index is not healthy/complete enough for graph projection",
            reason_code="repository_index_not_complete",
        )

    builder = _GraphBuilder(snapshot_id, index_id)
    snapshot_node = builder.node(
        kind=ContractNodeKind.REPOSITORY_SNAPSHOT,
        key=f"snapshot:{snapshot_id}",
        label=snapshot_id,
        provenance=ContractProvenance.REPOSITORY_INDEX,
        provenance_id=index_id,
        metadata={"source_index_id": index_id},
    )

    file_nodes: dict[str, ContractGraphNode] = {}
    module_nodes: dict[str, list[ContractGraphNode]] = {}
    rows = tuple(getattr(repository_index, "rows", ()))
    for row in rows:
        path = str(getattr(row, "path", "") or "")
        row_id = str(getattr(row, "row_id", "") or "")
        if not path or not row_id:
            raise GraphIntegrityError(
                "repository index rows require path and row_id"
            )
        file_node = builder.node(
            kind=ContractNodeKind.FILE,
            key=f"file:{path}",
            label=path,
            provenance=ContractProvenance.REPOSITORY_INDEX,
            provenance_id=row_id,
            metadata={
                "path": path,
                "coverage_kind": _row_kind(row),
                "content_digest": str(
                    getattr(row, "content_digest", "") or ""
                ),
                "git_status": str(
                    getattr(getattr(row, "git_status", ""), "value", "")
                    or getattr(row, "git_status", "")
                ),
            },
        )
        file_nodes[path] = file_node
        builder.edge(
            snapshot_node,
            file_node,
            kind=ContractEdgeKind.CONTAINS,
            provenance=ContractProvenance.REPOSITORY_INDEX,
            provenance_id=row_id,
        )

        if _row_kind(row) == "structured_data":
            schema_node = builder.node(
                kind=ContractNodeKind.SCHEMA,
                key=f"schema:{path}",
                label=path,
                provenance=ContractProvenance.SCHEMA,
                provenance_id=(
                    str(getattr(row, "ast_record_id", "") or "") or row_id
                ),
                metadata={"path": path},
            )
            builder.edge(
                file_node,
                schema_node,
                kind=ContractEdgeKind.DECLARES,
                provenance=ContractProvenance.SCHEMA,
                provenance_id=schema_node.provenance_id,
            )

    ast_index = getattr(repository_index, "ast_index", None)
    indexed_records = tuple(getattr(ast_index, "path_records", ()))
    symbol_aliases: dict[str, list[ContractGraphNode]] = {}
    pending_imports: list[
        tuple[ContractGraphNode, ContractGraphNode, str, str]
    ] = []
    pending_calls: list[
        tuple[ContractGraphNode, ContractGraphNode, str, str]
    ] = []

    for indexed in indexed_records:
        path = str(getattr(indexed, "path", "") or "")
        file_node = file_nodes.get(path)
        if file_node is None:
            raise GraphIntegrityError(
                f"AST path is absent from repository ledger: {path}"
            )
        record = getattr(indexed, "ast_record", None)
        record_id = _required_text(
            getattr(record, "record_id", ""), "AST record_id"
        )
        module = _module_name(path)
        module_node = builder.node(
            kind=ContractNodeKind.MODULE,
            key=f"module:{path}:{module}",
            label=module or path,
            provenance=ContractProvenance.AST,
            provenance_id=record_id,
            version=f"ast-blob-record@{getattr(record, 'record_schema_version', 1)}",
            metadata={
                "path": path,
                "module": module,
                "language": str(getattr(record, "language", "") or ""),
                "ast_record_id": record_id,
            },
        )
        module_nodes.setdefault(module, []).append(module_node)
        builder.edge(
            file_node,
            module_node,
            kind=ContractEdgeKind.DEFINES,
            provenance=ContractProvenance.AST,
            provenance_id=record_id,
        )

        for symbol in tuple(getattr(record, "qualified_symbols", ())):
            symbol = str(symbol)
            start, end = dict(getattr(record, "symbol_lines", {})).get(
                symbol, (0, 0)
            )
            symbol_node = builder.node(
                kind=ContractNodeKind.SYMBOL,
                key=f"symbol:{path}:{symbol}",
                label=symbol,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
                version=f"ast-blob-record@{getattr(record, 'record_schema_version', 1)}",
                metadata={
                    "path": path,
                    "module": module,
                    "symbol": symbol,
                    "symbol_hash": str(
                        dict(getattr(record, "symbol_hashes", {})).get(
                            symbol, ""
                        )
                    ),
                    "line_start": int(start),
                    "line_end": int(end),
                },
            )
            builder.edge(
                module_node,
                symbol_node,
                kind=ContractEdgeKind.DEFINES,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )
            for alias in {symbol, f"{module}.{symbol}" if module else symbol}:
                symbol_aliases.setdefault(alias, []).append(symbol_node)
            role = _role_kind(symbol, path)
            if role is not None:
                role_node = builder.node(
                    kind=role,
                    key=f"{role.value}:{path}:{symbol}",
                    label=symbol,
                    provenance=ContractProvenance.AST,
                    provenance_id=record_id,
                    metadata={
                        "path": path,
                        "module": module,
                        "symbol": symbol,
                        "role": role.value,
                    },
                )
                builder.edge(
                    symbol_node,
                    role_node,
                    kind=ContractEdgeKind.IMPLEMENTS,
                    provenance=ContractProvenance.AST,
                    provenance_id=record_id,
                )

        for position, target in enumerate(
            tuple(getattr(record, "imports", ()))
        ):
            target = str(target)
            import_node = builder.node(
                kind=ContractNodeKind.IMPORT,
                key=f"import:{path}:{position}:{target}",
                label=target,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
                metadata={"path": path, "module": module, "target": target},
            )
            builder.edge(
                module_node,
                import_node,
                kind=ContractEdgeKind.CONTAINS,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )
            pending_imports.append((module_node, import_node, target, record_id))

        for position, target in enumerate(tuple(getattr(record, "calls", ()))):
            target = str(target)
            call_node = builder.node(
                kind=ContractNodeKind.CALL,
                key=f"call:{path}:{position}:{target}",
                label=target,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
                metadata={"path": path, "module": module, "target": target},
            )
            builder.edge(
                module_node,
                call_node,
                kind=ContractEdgeKind.CONTAINS,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )
            pending_calls.append((module_node, call_node, target, record_id))

        for position, effect in enumerate(
            tuple(getattr(record, "state_transitions", ()))
        ):
            effect = str(effect)
            effect_node = builder.node(
                kind=ContractNodeKind.EFFECT,
                key=f"effect:{path}:{position}:{effect}",
                label=effect,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
                metadata={"path": path, "module": module, "effect": effect},
            )
            builder.edge(
                module_node,
                effect_node,
                kind=ContractEdgeKind.HAS_EFFECT,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )

        for position, interface in enumerate(
            tuple(getattr(record, "interfaces", ()))
        ):
            interface = str(interface)
            interface_node = builder.node(
                kind=ContractNodeKind.INTERFACE,
                key=f"interface:{path}:{position}:{interface}",
                label=interface,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
                metadata={
                    "path": path,
                    "module": module,
                    "signature": interface,
                },
            )
            builder.edge(
                module_node,
                interface_node,
                kind=ContractEdgeKind.DECLARES,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )

    unresolved: dict[tuple[str, str], ContractGraphNode] = {}

    def unresolved_node(kind: str, target: str, record_id: str) -> ContractGraphNode:
        key = (kind, target)
        if key not in unresolved:
            unresolved[key] = builder.node(
                kind=ContractNodeKind.UNRESOLVED,
                key=f"unresolved:{kind}:{target}",
                label=target,
                provenance=ContractProvenance.UNRESOLVED,
                provenance_id=record_id,
                authority=ContractAuthority.UNRESOLVED,
                metadata={"target": target, "relationship": kind},
            )
        return unresolved[key]

    for _module, import_node, target, record_id in pending_imports:
        target_modules = module_nodes.get(target, ())
        if len(target_modules) == 1:
            builder.edge(
                import_node,
                target_modules[0],
                kind=ContractEdgeKind.IMPORTS,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )
        else:
            builder.edge(
                import_node,
                unresolved_node("import", target, record_id),
                kind=ContractEdgeKind.IMPORTS,
                provenance=ContractProvenance.UNRESOLVED,
                provenance_id=record_id,
                authority=ContractAuthority.UNRESOLVED,
                mandatory=False,
                metadata={
                    "reason_code": (
                        "import_target_ambiguous"
                        if len(target_modules) > 1
                        else "import_target_unresolved"
                    ),
                    "match_count": len(target_modules),
                },
            )

    for _module, call_node, target, record_id in pending_calls:
        matches = symbol_aliases.get(target, ())
        if len(matches) == 1:
            builder.edge(
                call_node,
                matches[0],
                kind=ContractEdgeKind.CALLS,
                provenance=ContractProvenance.AST,
                provenance_id=record_id,
            )
        else:
            reason = (
                "call_target_ambiguous" if len(matches) > 1
                else "call_target_unresolved"
            )
            builder.edge(
                call_node,
                unresolved_node("call", target, record_id),
                kind=ContractEdgeKind.CALLS,
                provenance=ContractProvenance.UNRESOLVED,
                provenance_id=record_id,
                authority=ContractAuthority.UNRESOLVED,
                mandatory=False,
                metadata={"reason_code": reason, "match_count": len(matches)},
            )

    nodes = list(builder.nodes.values())
    for value in additional_nodes:
        node = (
            value
            if isinstance(value, ContractGraphNode)
            else ContractGraphNode.from_dict(value)
        )
        if node.snapshot_id != snapshot_id:
            raise GraphIntegrityError(
                "additional node is bound to a foreign snapshot"
            )
        nodes.append(node)
    node_ids = {node.node_id for node in nodes}

    edges = list(builder.edges.values())
    for value in additional_edges:
        edge = (
            value
            if isinstance(value, ContractGraphEdge)
            else ContractGraphEdge.from_dict(value)
        )
        if edge.snapshot_id != snapshot_id:
            raise GraphIntegrityError(
                "additional edge is bound to a foreign snapshot"
            )
        if edge.source not in node_ids or edge.target not in node_ids:
            if edge.mandatory:
                raise MissingMandatoryEdgeError(
                    "additional mandatory edge references an unknown node",
                    missing=(edge.source, edge.target),
                )
            raise GraphIntegrityError(
                "additional edge references an unknown node"
            )
        edges.append(edge)

    return SymbolicContractGraph(
        snapshot_id=snapshot_id,
        source_index_id=index_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        mandatory_requirements=tuple(mandatory_requirements),
    )


build_symbolic_contract_graph = project_repository_index
project_typed_contract_graph = project_repository_index

# Concise compatibility names for callers that prefer the interface name.
SymbolicGraphNode = ContractGraphNode
SymbolicGraphEdge = ContractGraphEdge
SymbolicGraph = SymbolicContractGraph
BoundedGraphRAGView = GraphRAGReceipt
SymbolicNode = ContractGraphNode
SymbolicEdge = ContractGraphEdge
SymbolicNodeKind = ContractNodeKind
SymbolicEdgeKind = ContractEdgeKind
GraphAuthority = ContractAuthority
GraphProvenance = ContractProvenance
MandatoryClosureReceipt = ContractClosureReceipt
GraphRAGCandidateReceipt = GraphRAGReceipt


__all__ = [
    "CONTENT_IDENTITY_INTERFACE",
    "DATASETS_PROJECTION_SCHEMA",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_CLOSURE_DEPTH",
    "DEFAULT_MAX_CLOSURE_EDGES",
    "DEFAULT_MAX_CLOSURE_NODES",
    "DEFAULT_MAX_GRAPH_EDGES",
    "DEFAULT_MAX_GRAPH_NODES",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MAX_RETRIEVAL_BYTES",
    "BoundedGraphRAGRetriever",
    "BoundedGraphRAGView",
    "ClosureBounds",
    "ClosureDirection",
    "ContractAuthority",
    "ContractClosureReceipt",
    "ContractEdgeKind",
    "ContractGraphEdge",
    "ContractGraphNode",
    "ContractNodeKind",
    "ContractProvenance",
    "DatasetsProviderState",
    "GraphBoundsError",
    "GraphAuthority",
    "GraphClosureTruncatedError",
    "GraphIntegrityError",
    "GraphProvenance",
    "GraphRAGCandidate",
    "GraphRAGCandidateReceipt",
    "GraphRAGContextEdge",
    "GraphRAGLimits",
    "GraphRAGReceipt",
    "LazyDatasetsGraphRAGProvider",
    "MandatoryEdgeRequirement",
    "MandatoryClosureReceipt",
    "MissingMandatoryEdgeError",
    "OptionalDatasetsProviderError",
    "SYMBOLIC_CLOSURE_SCHEMA",
    "SYMBOLIC_CONTRACT_EDGE_SCHEMA",
    "SYMBOLIC_CONTRACT_GRAPH_INTERFACE",
    "SYMBOLIC_CONTRACT_GRAPH_SCHEMA",
    "SYMBOLIC_CONTRACT_GRAPH_VERSION",
    "SYMBOLIC_CONTRACT_NODE_SCHEMA",
    "SymbolicContractGraph",
    "SymbolicContractGraphError",
    "SymbolicGraph",
    "SymbolicGraphEdge",
    "SymbolicGraphNode",
    "SymbolicEdge",
    "SymbolicEdgeKind",
    "SymbolicNode",
    "SymbolicNodeKind",
    "build_symbolic_contract_graph",
    "canonical_symbolic_json",
    "project_repository_index",
    "project_typed_contract_graph",
]
