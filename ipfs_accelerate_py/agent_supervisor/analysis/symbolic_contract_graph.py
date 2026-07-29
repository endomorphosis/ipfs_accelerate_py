"""Typed, content-addressed contract graph and bounded GraphRAG projection.

This module is the deterministic boundary between repository facts and
candidate retrieval.  Source/index/catalog facts may create authority-bearing
nodes and mandatory edges.  Retrieval and GraphRAG may only nominate existing
nodes or add context-only edges; they never create a proof dependency.

Edges are directed from a subject to a dependency.  Consequently, a forward
closure contains everything a subject depends on, while a reverse closure
contains every subject which depends on a seed.  Both closures are exact:
exceeding a bound or encountering a declared-but-missing mandatory edge raises
an incomplete-closure error instead of returning a partial authoritative view.

The optional ``ipfs_datasets_py`` analysis adapter is deliberately imported
only inside an explicitly requested provider dispatch.  Importing this module,
constructing a graph, constructing a retriever, and local retrieval do not
load or probe that optional analysis provider.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from .content_identity_bridge import ContentIdentity, identify_strict_artifact


SYMBOLIC_CONTRACT_GRAPH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-graph@1"
)
SYMBOLIC_CONTRACT_NODE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-node@1"
)
SYMBOLIC_CONTRACT_EDGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-edge@1"
)
MANDATORY_EDGE_REQUIREMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mandatory-contract-edge@1"
)
CONTRACT_GRAPH_CLOSURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-graph-closure@1"
)
GRAPH_RAG_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/graph-rag-candidate@1"
)
GRAPH_RAG_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/bounded-graph-rag-receipt@1"
)
GRAPH_RAG_VIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/bounded-graph-rag-view@1"
)

SYMBOLIC_CONTRACT_GRAPH_VERSION = "SymbolicContractGraph@1"
BOUNDED_GRAPH_RAG_RETRIEVER_VERSION = "BoundedGraphRAGRetriever@1"
SYMBOLIC_CONTRACT_GRAPH_INTERFACE = "SymbolicContractGraph@1"
BOUNDED_GRAPH_RAG_RETRIEVER_INTERFACE = "BoundedGraphRAGRetriever@1"

DEFAULT_MAX_GRAPH_NODES = 100_000
DEFAULT_MAX_GRAPH_EDGES = 250_000
DEFAULT_MAX_CLOSURE_NODES = 16_384
DEFAULT_MAX_CLOSURE_EDGES = 65_536
DEFAULT_MAX_CLOSURE_DEPTH = 256
DEFAULT_MAX_CANDIDATES = 32
DEFAULT_MAX_QUERY_BYTES = 16 * 1024
HARD_MAX_CANDIDATES = 1_024
HARD_MAX_QUERY_BYTES = 64 * 1024


class SymbolicContractGraphError(ValueError):
    """A graph record violates the typed contract or authority boundary."""


class ContractGraphBoundsError(SymbolicContractGraphError):
    """A graph, closure, query, or candidate set exceeded a hard bound."""


class IncompleteContractClosureError(SymbolicContractGraphError):
    """An exact mandatory closure could not be produced."""


class MissingMandatoryEdgeError(IncompleteContractClosureError):
    """A node declares a mandatory relationship absent from the graph."""

    def __init__(self, requirements: Sequence["MandatoryEdgeRequirement"]) -> None:
        self.requirements = tuple(requirements)
        rendered = ", ".join(item.requirement_id for item in self.requirements[:8])
        super().__init__(f"mandatory contract edges are missing: {rendered}")


class TruncatedContractClosureError(
    ContractGraphBoundsError, IncompleteContractClosureError
):
    """A closure bound would have caused a partial result."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"mandatory contract closure truncated: {reason_code}")


class ContractNodeKind(str, Enum):
    REPOSITORY_SNAPSHOT = "repository_snapshot"
    FILE = "file"
    MODULE = "module"
    SYMBOL = "symbol"
    CALL = "call"
    IMPORT = "import"
    EFFECT = "effect"
    SCHEMA = "schema"
    IDL = "idl"
    INTERFACE = "interface"
    METHOD = "method"
    TOOL = "tool"
    HANDLER = "handler"
    IMPLEMENTATION = "implementation"
    TEST = "test"
    POLICY = "policy"
    AUTHORIZATION = "authorization"
    TRANSPORT = "transport"
    PROVENANCE = "provenance"
    CONTRACT = "contract"
    CLAIM = "claim"
    OBLIGATION = "obligation"


class ContractEdgeKind(str, Enum):
    CONTAINS = "contains"
    DEFINES = "defines"
    IMPORTS = "imports"
    CALLS = "calls"
    HAS_EFFECT = "has_effect"
    EXPOSES = "exposes"
    BINDS_SCHEMA = "binds_schema"
    IMPLEMENTS = "implements"
    HANDLED_BY = "handled_by"
    VALIDATES = "validates"
    TESTS = "tests"
    GOVERNED_BY = "governed_by"
    AUTHORIZED_BY = "authorized_by"
    TRANSPORTED_BY = "transported_by"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    PROVENANCE_FOR = "provenance_for"
    RELATED_TO = "related_to"
    NOMINATES = "nominates"


class ContractProvenance(str, Enum):
    SNAPSHOT = "snapshot"
    INDEX = "index"
    AST = "ast"
    SCHEMA = "schema"
    CATALOG = "catalog"
    SOURCE = "source"
    POLICY = "policy"
    TEST = "test"
    RUNTIME = "runtime"
    PROOF = "proof"
    RETRIEVAL = "retrieval"
    GRAPHRAG = "graphrag"
    MODEL = "model"

    @property
    def trusted_channel(self) -> bool:
        return self not in {
            ContractProvenance.RETRIEVAL,
            ContractProvenance.GRAPHRAG,
            ContractProvenance.MODEL,
        }


class ContractAuthority(str, Enum):
    OBSERVATION = "observation"
    REVIEWED_CONTRACT = "reviewed_contract"
    POLICY = "policy"
    IMPLEMENTATION = "implementation"
    PROOF_INPUT = "proof_input"
    CONTEXT_ONLY = "context_only"
    PROPOSAL_ONLY = "proposal_only"
    NONE = "none"

    @property
    def authority_bearing(self) -> bool:
        return self in {
            ContractAuthority.OBSERVATION,
            ContractAuthority.REVIEWED_CONTRACT,
            ContractAuthority.POLICY,
            ContractAuthority.IMPLEMENTATION,
            ContractAuthority.PROOF_INPUT,
        }


class ClosureDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class RetrievalStatus(str, Enum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    INCOMPLETE = "incomplete"


_TOKEN = re.compile(r"[A-Za-z0-9_./:+-]+")


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise SymbolicContractGraphError(f"invalid {name}: {value!r}") from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = 8_192,
) -> str:
    if not isinstance(value, str):
        raise SymbolicContractGraphError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise SymbolicContractGraphError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise SymbolicContractGraphError(f"{name} is required")
    if len(value.encode("utf-8")) > max_bytes:
        raise ContractGraphBoundsError(f"{name} exceeds {max_bytes} bytes")
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise ContractGraphBoundsError("contract graph record nesting is excessive")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SymbolicContractGraphError(
                "non-finite values are not canonical graph data"
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > 4_096 or not all(isinstance(key, str) for key in value):
            raise ContractGraphBoundsError("contract graph mapping is invalid")
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 65_536:
            raise ContractGraphBoundsError("contract graph sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise SymbolicContractGraphError(
        f"unsupported contract graph value: {type(value).__name__}"
    )


def canonical_contract_graph_bytes(value: Any) -> bytes:
    """Return the strict deterministic JSON preimage used for local checks."""

    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_contract_graph_bytes(value)).hexdigest()


def _identity(value: Any) -> ContentIdentity:
    # ContentIdentity@1 validates CID version, codec, multihash, and the
    # retained canonical preimage.  Missing CID support therefore fails closed.
    return identify_strict_artifact(_plain(value))


def _identity_dict(value: Any) -> dict[str, Any]:
    return _identity(value).to_dict()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise SymbolicContractGraphError(f"{name} must be a mapping")
        value = to_dict()
    normalized = _plain(value)
    if not isinstance(normalized, dict):
        raise SymbolicContractGraphError(f"{name} must be a mapping")
    return MappingProxyType(normalized)


def _verify_bool_claim(
    payload: Mapping[str, Any],
    name: str,
    expected: bool,
    error: str,
) -> None:
    if name not in payload:
        return
    value = payload[name]
    if not isinstance(value, bool) or value is not expected:
        raise SymbolicContractGraphError(error)


@dataclass(frozen=True)
class SymbolicContractNode:
    """One snapshot-bound code, contract, policy, or provenance fact."""

    node_id: str
    kind: ContractNodeKind
    snapshot_id: str
    provenance: ContractProvenance
    provenance_id: str
    authority: ContractAuthority
    version: str
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("node_id", "snapshot_id", "provenance_id", "version"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"node {name}")
            )
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
        object.__setattr__(self, "record", _mapping(self.record, "node record"))
        if (
            not self.provenance.trusted_channel
            and self.authority is not ContractAuthority.CONTEXT_ONLY
        ):
            raise SymbolicContractGraphError(
                "retrieval, GraphRAG, and model nodes must be context_only"
            )

    @property
    def authoritative(self) -> bool:
        return (
            self.provenance.trusted_channel
            and self.authority.authority_bearing
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_NODE_SCHEMA,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "snapshot_id": self.snapshot_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "authority": self.authority.value,
            "version": self.version,
            "record": _plain(self.record),
        }

    @property
    def identity(self) -> ContentIdentity:
        return _identity(self._identity_payload())

    @property
    def content_id(self) -> str:
        return self.identity.cid

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "content_id": self.content_id,
            "identity": self.identity.to_dict(),
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicContractNode":
        if str(payload.get("schema") or SYMBOLIC_CONTRACT_NODE_SCHEMA) != (
            SYMBOLIC_CONTRACT_NODE_SCHEMA
        ):
            raise SymbolicContractGraphError("unsupported symbolic node schema")
        node = cls(
            node_id=str(payload.get("node_id") or ""),
            kind=payload.get("kind", ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            provenance=payload.get("provenance", ""),
            provenance_id=str(payload.get("provenance_id") or ""),
            authority=payload.get("authority", ""),
            version=str(payload.get("version") or ""),
            record=payload.get("record") or {},
        )
        claimed = str(payload.get("content_id") or "")
        if claimed and claimed != node.content_id:
            raise SymbolicContractGraphError("symbolic node identity mismatch")
        claimed_identity = payload.get("identity")
        if claimed_identity and _plain(claimed_identity) != node.identity.to_dict():
            raise SymbolicContractGraphError(
                "symbolic node ContentIdentity metadata mismatch"
            )
        _verify_bool_claim(
            payload,
            "authoritative",
            node.authoritative,
            "forged symbolic node authority",
        )
        return node


@dataclass(frozen=True)
class SymbolicContractEdge:
    """One typed relationship, directed from subject to dependency."""

    source: str
    target: str
    kind: ContractEdgeKind
    snapshot_id: str
    provenance: ContractProvenance
    provenance_id: str
    authority: ContractAuthority
    version: str
    mandatory: bool = True
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source",
            "target",
            "snapshot_id",
            "provenance_id",
            "version",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"edge {name}")
            )
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
        if not isinstance(self.mandatory, bool):
            raise SymbolicContractGraphError("edge mandatory must be a boolean")
        object.__setattr__(self, "record", _mapping(self.record, "edge record"))
        if not self.provenance.trusted_channel:
            if self.authority is not ContractAuthority.CONTEXT_ONLY:
                raise SymbolicContractGraphError(
                    "GraphRAG/retrieval/model edges must be context_only"
                )
            if self.mandatory:
                raise SymbolicContractGraphError(
                    "GraphRAG/retrieval/model edges cannot be mandatory"
                )
        if self.mandatory and not self.authority.authority_bearing:
            raise SymbolicContractGraphError(
                "mandatory edges must carry source-derived authority"
            )

    @property
    def authoritative(self) -> bool:
        return (
            self.provenance.trusted_channel
            and self.authority.authority_bearing
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
            "record": _plain(self.record),
        }

    @property
    def identity(self) -> ContentIdentity:
        return _identity(self._identity_payload())

    @property
    def edge_id(self) -> str:
        return self.identity.cid

    @property
    def content_id(self) -> str:
        return self.edge_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "edge_id": self.edge_id,
            "content_id": self.content_id,
            "identity": self.identity.to_dict(),
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicContractEdge":
        if str(payload.get("schema") or SYMBOLIC_CONTRACT_EDGE_SCHEMA) != (
            SYMBOLIC_CONTRACT_EDGE_SCHEMA
        ):
            raise SymbolicContractGraphError("unsupported symbolic edge schema")
        edge = cls(
            source=str(
                payload.get("source") or payload.get("source_node_id") or ""
            ),
            target=str(
                payload.get("target") or payload.get("target_node_id") or ""
            ),
            kind=payload.get("kind", payload.get("edge_kind", "")),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            provenance=payload.get("provenance", ""),
            provenance_id=str(payload.get("provenance_id") or ""),
            authority=payload.get("authority", ""),
            version=str(payload.get("version") or ""),
            mandatory=payload.get("mandatory", True),
            record=payload.get("record") or {},
        )
        claimed = str(payload.get("edge_id") or payload.get("content_id") or "")
        if claimed and claimed != edge.edge_id:
            raise SymbolicContractGraphError("symbolic edge identity mismatch")
        claimed_identity = payload.get("identity")
        if claimed_identity and _plain(claimed_identity) != edge.identity.to_dict():
            raise SymbolicContractGraphError(
                "symbolic edge ContentIdentity metadata mismatch"
            )
        _verify_bool_claim(
            payload,
            "authoritative",
            edge.authoritative,
            "forged symbolic edge authority",
        )
        return edge


@dataclass(frozen=True)
class MandatoryEdgeRequirement:
    """A declared dependency which must be represented by exactly one edge."""

    source: str
    kind: ContractEdgeKind
    target: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _text(self.source, "requirement source")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ContractEdgeKind, "requirement kind")
        )
        object.__setattr__(
            self,
            "target",
            _text(
                self.target,
                "requirement target",
                required=False,
            ),
        )

    @property
    def requirement_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANDATORY_EDGE_REQUIREMENT_SCHEMA,
            "source": self.source,
            "kind": self.kind.value,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MandatoryEdgeRequirement":
        return cls(
            source=str(payload.get("source") or ""),
            kind=payload.get("kind", ""),
            target=str(payload.get("target") or ""),
        )


@dataclass(frozen=True)
class ContractClosureBounds:
    max_nodes: int = DEFAULT_MAX_CLOSURE_NODES
    max_edges: int = DEFAULT_MAX_CLOSURE_EDGES
    max_depth: int = DEFAULT_MAX_CLOSURE_DEPTH

    def __post_init__(self) -> None:
        limits = {
            "max_nodes": (self.max_nodes, DEFAULT_MAX_GRAPH_NODES),
            "max_edges": (self.max_edges, DEFAULT_MAX_GRAPH_EDGES),
            "max_depth": (self.max_depth, 4_096),
        }
        for name, (value, hard_maximum) in limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > hard_maximum
            ):
                raise ContractGraphBoundsError(
                    f"{name} must be an integer from 1 through {hard_maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_depth": self.max_depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractClosureBounds":
        if not isinstance(payload, Mapping):
            raise SymbolicContractGraphError("closure bounds must be a mapping")
        return cls(
            max_nodes=payload.get("max_nodes", DEFAULT_MAX_CLOSURE_NODES),
            max_edges=payload.get("max_edges", DEFAULT_MAX_CLOSURE_EDGES),
            max_depth=payload.get("max_depth", DEFAULT_MAX_CLOSURE_DEPTH),
        )


@dataclass(frozen=True)
class ContractGraphClosure:
    graph_root: str
    snapshot_id: str
    direction: ClosureDirection
    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    paths: Mapping[str, tuple[str, ...]]
    bounds: ContractClosureBounds = field(default_factory=ContractClosureBounds)

    def __post_init__(self) -> None:
        for name in ("graph_root", "snapshot_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"closure {name}")
            )
        object.__setattr__(
            self,
            "direction",
            _enum(self.direction, ClosureDirection, "closure direction"),
        )
        seeds = tuple(sorted({_text(item, "closure seed") for item in self.seed_node_ids}))
        nodes = tuple(sorted({_text(item, "closure node") for item in self.node_ids}))
        edges = tuple(sorted({_text(item, "closure edge") for item in self.edge_ids}))
        object.__setattr__(self, "seed_node_ids", seeds)
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "edge_ids", edges)
        normalized_paths = {
            str(key): tuple(str(part) for part in value)
            for key, value in sorted(self.paths.items())
        }
        object.__setattr__(self, "paths", MappingProxyType(normalized_paths))
        if not seeds or not set(seeds).issubset(nodes):
            raise SymbolicContractGraphError(
                "closure must contain at least one declared seed"
            )
        if set(normalized_paths) != set(nodes):
            raise SymbolicContractGraphError(
                "closure paths must cover exactly the closure nodes"
            )
        for node_id, path in normalized_paths.items():
            if (
                not path
                or path[0] not in seeds
                or path[-1] != node_id
                or len(path) != len(set(path))
            ):
                raise SymbolicContractGraphError(
                    f"invalid closure path for {node_id!r}"
                )
        if len(nodes) > self.bounds.max_nodes:
            raise TruncatedContractClosureError("max_nodes_exceeded")
        if len(edges) > self.bounds.max_edges:
            raise TruncatedContractClosureError("max_edges_exceeded")
        if max((len(path) - 1 for path in normalized_paths.values()), default=0) > (
            self.bounds.max_depth
        ):
            raise TruncatedContractClosureError("max_depth_exceeded")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_GRAPH_CLOSURE_SCHEMA,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "direction": self.direction.value,
            "seed_node_ids": list(self.seed_node_ids),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "paths": {key: list(value) for key, value in self.paths.items()},
        }

    @property
    def closure_id(self) -> str:
        return _identity(self._identity_payload()).cid

    @property
    def complete(self) -> bool:
        return True

    @property
    def truncated(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "closure_id": self.closure_id,
            "identity": _identity_dict(self._identity_payload()),
            "bounds": self.bounds.to_dict(),
            "complete": True,
            "truncated": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractGraphClosure":
        if str(payload.get("schema") or CONTRACT_GRAPH_CLOSURE_SCHEMA) != (
            CONTRACT_GRAPH_CLOSURE_SCHEMA
        ):
            raise SymbolicContractGraphError("unsupported contract closure schema")
        raw_paths = payload.get("paths") or {}
        if not isinstance(raw_paths, Mapping):
            raise SymbolicContractGraphError("closure paths must be a mapping")
        closure = cls(
            graph_root=str(payload.get("graph_root") or ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            direction=payload.get("direction", ""),
            seed_node_ids=tuple(payload.get("seed_node_ids") or ()),
            node_ids=tuple(payload.get("node_ids") or ()),
            edge_ids=tuple(payload.get("edge_ids") or ()),
            paths={
                str(key): tuple(value)
                for key, value in raw_paths.items()
            },
            bounds=ContractClosureBounds.from_dict(payload.get("bounds") or {}),
        )
        claimed = str(payload.get("closure_id") or "")
        if claimed and claimed != closure.closure_id:
            raise SymbolicContractGraphError("contract closure identity mismatch")
        claimed_identity = payload.get("identity")
        if (
            claimed_identity
            and _plain(claimed_identity)
            != _identity_dict(closure._identity_payload())
        ):
            raise SymbolicContractGraphError(
                "contract closure ContentIdentity metadata mismatch"
            )
        for name, expected in (("complete", True), ("truncated", False)):
            if name in payload and (
                not isinstance(payload[name], bool)
                or payload[name] is not expected
            ):
                raise IncompleteContractClosureError(
                    "serialized mandatory closure is incomplete"
                )
        return closure

    @classmethod
    def from_json(cls, payload: str) -> "ContractGraphClosure":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SymbolicContractGraphError(
                "contract closure JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise SymbolicContractGraphError(
                "contract closure JSON must contain an object"
            )
        return cls.from_dict(value)


def _requirements_from_node(
    node: SymbolicContractNode,
) -> tuple[MandatoryEdgeRequirement, ...]:
    """Accept compact producer declarations without coupling to one AST shape."""

    record = node.record
    requirements: list[MandatoryEdgeRequirement] = []
    for value in record.get("mandatory_edges", ()):
        if not isinstance(value, Mapping):
            raise SymbolicContractGraphError(
                f"node {node.node_id!r} mandatory_edges must contain mappings"
            )
        requirements.append(
            MandatoryEdgeRequirement(
                source=node.node_id,
                kind=value.get("kind", ContractEdgeKind.DEPENDS_ON.value),
                target=str(value.get("target") or ""),
            )
        )
    for value in record.get("mandatory_dependency_ids", ()):
        requirements.append(
            MandatoryEdgeRequirement(
                source=node.node_id,
                kind=ContractEdgeKind.DEPENDS_ON,
                target=str(value),
            )
        )
    for value in record.get("required_edge_kinds", ()):
        requirements.append(
            MandatoryEdgeRequirement(source=node.node_id, kind=value)
        )
    return tuple(requirements)


@dataclass(frozen=True)
class SymbolicContractGraph:
    """Canonical typed projection pinned to exactly one repository snapshot."""

    snapshot_id: str
    nodes: tuple[SymbolicContractNode, ...] = ()
    edges: tuple[SymbolicContractEdge, ...] = ()
    mandatory_edge_requirements: tuple[MandatoryEdgeRequirement, ...] = ()
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION

    def __post_init__(self) -> None:
        snapshot_id = _text(self.snapshot_id, "graph snapshot_id")
        version = _text(self.version, "graph version")
        node_map: dict[str, SymbolicContractNode] = {}
        for value in self.nodes:
            node = (
                value
                if isinstance(value, SymbolicContractNode)
                else SymbolicContractNode.from_dict(value)
            )
            if node.snapshot_id != snapshot_id:
                raise SymbolicContractGraphError(
                    f"node {node.node_id!r} is bound to a foreign snapshot"
                )
            previous = node_map.get(node.node_id)
            if previous is not None and previous.to_dict() != node.to_dict():
                raise SymbolicContractGraphError(
                    f"conflicting symbolic node: {node.node_id}"
                )
            node_map[node.node_id] = node
        if len(node_map) > DEFAULT_MAX_GRAPH_NODES:
            raise ContractGraphBoundsError("symbolic graph has too many nodes")

        edge_map: dict[str, SymbolicContractEdge] = {}
        edge_keys: set[tuple[str, ContractEdgeKind, str]] = set()
        for value in self.edges:
            edge = (
                value
                if isinstance(value, SymbolicContractEdge)
                else SymbolicContractEdge.from_dict(value)
            )
            if edge.snapshot_id != snapshot_id:
                raise SymbolicContractGraphError(
                    f"edge {edge.edge_id!r} is bound to a foreign snapshot"
                )
            if edge.source not in node_map or edge.target not in node_map:
                if edge.mandatory:
                    requirement = MandatoryEdgeRequirement(
                        edge.source, edge.kind, edge.target
                    )
                    raise MissingMandatoryEdgeError((requirement,))
                raise SymbolicContractGraphError(
                    f"edge {edge.edge_id!r} references an unknown node"
                )
            if edge.authoritative and (
                not node_map[edge.source].authoritative
                or not node_map[edge.target].authoritative
            ):
                raise SymbolicContractGraphError(
                    "authoritative edge cannot promote a context-only endpoint"
                )
            key = (edge.source, edge.kind, edge.target)
            if edge.mandatory and key in edge_keys:
                raise SymbolicContractGraphError(
                    "mandatory relationship must have exactly one typed edge"
                )
            if edge.mandatory:
                edge_keys.add(key)
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_GRAPH_EDGES:
            raise ContractGraphBoundsError("symbolic graph has too many edges")

        requirements: dict[str, MandatoryEdgeRequirement] = {}
        raw_requirements: Iterable[Any] = self.mandatory_edge_requirements
        for value in raw_requirements:
            requirement = (
                value
                if isinstance(value, MandatoryEdgeRequirement)
                else MandatoryEdgeRequirement.from_dict(value)
            )
            requirements[requirement.requirement_id] = requirement
        for node in node_map.values():
            for requirement in _requirements_from_node(node):
                requirements[requirement.requirement_id] = requirement

        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        object.__setattr__(
            self,
            "mandatory_edge_requirements",
            tuple(requirements[key] for key in sorted(requirements)),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_GRAPH_SCHEMA,
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "mandatory_edge_requirements": [
                item.to_dict() for item in self.mandatory_edge_requirements
            ],
        }

    @property
    def identity(self) -> ContentIdentity:
        return _identity(self._identity_payload())

    @property
    def graph_root(self) -> str:
        return self.identity.cid

    @property
    def root_id(self) -> str:
        return self.graph_root

    @property
    def graph_id(self) -> str:
        return self.graph_root

    def node(self, node_id: str) -> SymbolicContractNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def nodes_by_kind(
        self, kind: ContractNodeKind | str
    ) -> tuple[SymbolicContractNode, ...]:
        expected = _enum(kind, ContractNodeKind, "node kind")
        return tuple(node for node in self.nodes if node.kind is expected)

    def edges_by_kind(
        self, kind: ContractEdgeKind | str
    ) -> tuple[SymbolicContractEdge, ...]:
        expected = _enum(kind, ContractEdgeKind, "edge kind")
        return tuple(edge for edge in self.edges if edge.kind is expected)

    def _missing_requirements(
        self, source_ids: Iterable[str]
    ) -> tuple[MandatoryEdgeRequirement, ...]:
        sources = set(source_ids)
        present = {
            (edge.source, edge.kind, edge.target)
            for edge in self.edges
            if edge.mandatory and edge.authoritative
        }
        missing: list[MandatoryEdgeRequirement] = []
        for requirement in self.mandatory_edge_requirements:
            if requirement.source not in sources:
                continue
            matched = any(
                source == requirement.source
                and kind is requirement.kind
                and (not requirement.target or target == requirement.target)
                for source, kind, target in present
            )
            if not matched:
                missing.append(requirement)
        return tuple(sorted(missing, key=lambda item: item.requirement_id))

    def validate_mandatory_edges(
        self, node_ids: Iterable[str] | None = None
    ) -> None:
        selected = (
            {node.node_id for node in self.nodes}
            if node_ids is None
            else {_text(item, "node_id") for item in node_ids}
        )
        unknown = selected - {node.node_id for node in self.nodes}
        if unknown:
            raise KeyError(sorted(unknown)[0])
        missing = self._missing_requirements(selected)
        if missing:
            raise MissingMandatoryEdgeError(missing)

    def exact_closure(
        self,
        seed_node_ids: str | Iterable[str],
        *,
        direction: ClosureDirection | str = ClosureDirection.FORWARD,
        bounds: ContractClosureBounds | None = None,
    ) -> ContractGraphClosure:
        limits = bounds or ContractClosureBounds()
        selected_direction = _enum(
            direction, ClosureDirection, "closure direction"
        )
        if isinstance(seed_node_ids, str):
            seeds = (seed_node_ids,)
        else:
            seeds = tuple(seed_node_ids)
        seeds = tuple(sorted({_text(item, "closure seed") for item in seeds}))
        if not seeds:
            raise SymbolicContractGraphError("closure requires at least one seed")
        node_by_id = {node.node_id: node for node in self.nodes}
        unknown = set(seeds) - set(node_by_id)
        if unknown:
            raise KeyError(sorted(unknown)[0])
        if any(not node_by_id[item].authoritative for item in seeds):
            raise SymbolicContractGraphError(
                "mandatory closure seeds must be authority-bearing"
            )

        adjacency: dict[str, list[tuple[str, SymbolicContractEdge]]] = {}
        for edge in self.edges:
            if not edge.mandatory or not edge.authoritative:
                continue
            source, target = (
                (edge.source, edge.target)
                if selected_direction is ClosureDirection.FORWARD
                else (edge.target, edge.source)
            )
            adjacency.setdefault(source, []).append((target, edge))
        for values in adjacency.values():
            values.sort(
                key=lambda pair: (
                    pair[1].kind.value,
                    pair[0],
                    pair[1].edge_id,
                )
            )

        paths: dict[str, tuple[str, ...]] = {seed: (seed,) for seed in seeds}
        depths = {seed: 0 for seed in seeds}
        included_edges: set[str] = set()
        queue: deque[str] = deque(seeds)
        while queue:
            current = queue.popleft()
            # A reverse walk discovers dependents; each dependent's declared
            # dependency edge still has to exist even if its target is not a
            # further reverse neighbor.
            missing = self._missing_requirements((current,))
            if missing:
                raise MissingMandatoryEdgeError(missing)
            for target, edge in adjacency.get(current, ()):
                if not node_by_id[target].authoritative:
                    raise IncompleteContractClosureError(
                        "mandatory edge reached a non-authoritative node"
                    )
                depth = depths[current] + 1
                if depth > limits.max_depth:
                    raise TruncatedContractClosureError("max_depth_exceeded")
                included_edges.add(edge.edge_id)
                if len(included_edges) > limits.max_edges:
                    raise TruncatedContractClosureError("max_edges_exceeded")
                candidate_path = (*paths[current], target)
                previous = paths.get(target)
                if previous is None:
                    paths[target] = candidate_path
                    depths[target] = depth
                    if len(paths) > limits.max_nodes:
                        raise TruncatedContractClosureError("max_nodes_exceeded")
                    queue.append(target)
                elif (len(candidate_path), candidate_path) < (
                    len(previous),
                    previous,
                ):
                    paths[target] = candidate_path
                    depths[target] = depth

        return ContractGraphClosure(
            graph_root=self.graph_root,
            snapshot_id=self.snapshot_id,
            direction=selected_direction,
            seed_node_ids=seeds,
            node_ids=tuple(paths),
            edge_ids=tuple(included_edges),
            paths=paths,
            bounds=limits,
        )

    def forward_closure(
        self,
        seed_node_ids: str | Iterable[str],
        *,
        bounds: ContractClosureBounds | None = None,
    ) -> ContractGraphClosure:
        return self.exact_closure(
            seed_node_ids, direction=ClosureDirection.FORWARD, bounds=bounds
        )

    def reverse_closure(
        self,
        seed_node_ids: str | Iterable[str],
        *,
        bounds: ContractClosureBounds | None = None,
    ) -> ContractGraphClosure:
        return self.exact_closure(
            seed_node_ids, direction=ClosureDirection.REVERSE, bounds=bounds
        )

    mandatory_closure = forward_closure

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "graph_root": self.graph_root,
            "graph_id": self.graph_root,
            "identity": self.identity.to_dict(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_contract_graph_bytes(self.to_dict()).decode("utf-8")
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicContractGraph":
        if str(payload.get("schema") or SYMBOLIC_CONTRACT_GRAPH_SCHEMA) != (
            SYMBOLIC_CONTRACT_GRAPH_SCHEMA
        ):
            raise SymbolicContractGraphError("unsupported symbolic graph schema")
        graph = cls(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            nodes=tuple(payload.get("nodes") or ()),
            edges=tuple(payload.get("edges") or ()),
            mandatory_edge_requirements=tuple(
                payload.get("mandatory_edge_requirements") or ()
            ),
            version=str(
                payload.get("version") or SYMBOLIC_CONTRACT_GRAPH_VERSION
            ),
        )
        claimed = str(payload.get("graph_root") or payload.get("graph_id") or "")
        if claimed and claimed != graph.graph_root:
            raise SymbolicContractGraphError("symbolic graph root mismatch")
        claimed_identity = payload.get("identity")
        if claimed_identity and _plain(claimed_identity) != graph.identity.to_dict():
            raise SymbolicContractGraphError(
                "symbolic graph ContentIdentity metadata mismatch"
            )
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "SymbolicContractGraph":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SymbolicContractGraphError(
                "symbolic contract graph JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise SymbolicContractGraphError(
                "symbolic contract graph JSON must contain an object"
            )
        return cls.from_dict(value)


@dataclass(frozen=True)
class GraphRAGCandidate:
    node_id: str
    score_millionths: int
    nomination_source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _text(self.node_id, "candidate node_id")
        )
        if (
            isinstance(self.score_millionths, bool)
            or not isinstance(self.score_millionths, int)
            or not 0 <= self.score_millionths <= 1_000_000
        ):
            raise SymbolicContractGraphError(
                "candidate score_millionths must be from 0 through 1000000"
            )
        object.__setattr__(
            self,
            "nomination_source",
            _text(self.nomination_source, "candidate nomination_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_RAG_CANDIDATE_SCHEMA,
            "node_id": self.node_id,
            "score_millionths": self.score_millionths,
            "nomination_source": self.nomination_source,
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "mandatory": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphRAGCandidate":
        if str(payload.get("schema") or GRAPH_RAG_CANDIDATE_SCHEMA) != (
            GRAPH_RAG_CANDIDATE_SCHEMA
        ):
            raise SymbolicContractGraphError("unsupported GraphRAG candidate schema")
        if payload.get("authority", ContractAuthority.CONTEXT_ONLY.value) != (
            ContractAuthority.CONTEXT_ONLY.value
        ):
            raise SymbolicContractGraphError(
                "GraphRAG candidates must remain context_only"
            )
        _verify_bool_claim(
            payload,
            "mandatory",
            False,
            "GraphRAG candidates cannot become mandatory",
        )
        return cls(
            node_id=str(payload.get("node_id") or ""),
            score_millionths=payload.get("score_millionths", -1),
            nomination_source=str(payload.get("nomination_source") or ""),
        )


@dataclass(frozen=True)
class GraphRAGRetrievalBounds:
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_nodes: int = DEFAULT_MAX_CLOSURE_NODES
    max_edges: int = DEFAULT_MAX_CLOSURE_EDGES
    max_depth: int = DEFAULT_MAX_CLOSURE_DEPTH
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES

    def __post_init__(self) -> None:
        checks = {
            "max_candidates": (self.max_candidates, HARD_MAX_CANDIDATES),
            "max_nodes": (self.max_nodes, DEFAULT_MAX_GRAPH_NODES),
            "max_edges": (self.max_edges, DEFAULT_MAX_GRAPH_EDGES),
            "max_depth": (self.max_depth, 4_096),
            "max_query_bytes": (self.max_query_bytes, HARD_MAX_QUERY_BYTES),
        }
        for name, (value, maximum) in checks.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise ContractGraphBoundsError(
                    f"{name} must be an integer from 1 through {maximum}"
                )

    @property
    def closure_bounds(self) -> ContractClosureBounds:
        return ContractClosureBounds(
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            max_depth=self.max_depth,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_depth": self.max_depth,
            "max_query_bytes": self.max_query_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphRAGRetrievalBounds":
        if not isinstance(payload, Mapping):
            raise SymbolicContractGraphError(
                "GraphRAG retrieval bounds must be a mapping"
            )
        return cls(
            max_candidates=payload.get(
                "max_candidates", DEFAULT_MAX_CANDIDATES
            ),
            max_nodes=payload.get("max_nodes", DEFAULT_MAX_CLOSURE_NODES),
            max_edges=payload.get("max_edges", DEFAULT_MAX_CLOSURE_EDGES),
            max_depth=payload.get("max_depth", DEFAULT_MAX_CLOSURE_DEPTH),
            max_query_bytes=payload.get(
                "max_query_bytes", DEFAULT_MAX_QUERY_BYTES
            ),
        )


@dataclass(frozen=True)
class GraphRAGRetrievalReceipt:
    graph_root: str
    snapshot_id: str
    query_digest: str
    candidates: tuple[GraphRAGCandidate, ...]
    bounds: GraphRAGRetrievalBounds
    status: RetrievalStatus
    reason_code: str
    forward_closure_id: str = ""
    reverse_closure_id: str = ""
    provider_requested: bool = False
    provider_imported: bool = False
    provider_result_id: str = ""
    provider_status: str = "not_requested"
    missing_requirement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "graph_root",
            "snapshot_id",
            "query_digest",
            "reason_code",
            "provider_status",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"receipt {name}")
            )
        for name in (
            "forward_closure_id",
            "reverse_closure_id",
            "provider_result_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    f"receipt {name}",
                    required=False,
                ),
            )
        object.__setattr__(
            self, "status", _enum(self.status, RetrievalStatus, "receipt status")
        )
        for name in ("provider_requested", "provider_imported"):
            if not isinstance(getattr(self, name), bool):
                raise SymbolicContractGraphError(
                    f"receipt {name} must be a boolean"
                )
        candidates = tuple(
            sorted(
                (
                    item
                    if isinstance(item, GraphRAGCandidate)
                    else GraphRAGCandidate.from_dict(item)
                    for item in self.candidates
                ),
                key=lambda item: (
                    -item.score_millionths,
                    item.node_id,
                    item.nomination_source,
                ),
            )
        )
        if len(candidates) > self.bounds.max_candidates:
            raise ContractGraphBoundsError("receipt candidate bound exceeded")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "missing_requirement_ids",
            tuple(
                sorted(
                    {
                        _text(item, "receipt missing requirement id")
                        for item in self.missing_requirement_ids
                    }
                )
            ),
        )
        if self.status is RetrievalStatus.COMPLETE:
            if self.missing_requirement_ids:
                raise SymbolicContractGraphError(
                    "complete receipt cannot report missing mandatory edges"
                )
            if candidates and (
                not self.forward_closure_id or not self.reverse_closure_id
            ):
                raise SymbolicContractGraphError(
                    "complete non-empty retrieval requires both exact closures"
                )
        elif self.forward_closure_id or self.reverse_closure_id:
            raise SymbolicContractGraphError(
                "incomplete receipt cannot publish partial closure identities"
            )
        if self.provider_imported and not self.provider_requested:
            raise SymbolicContractGraphError(
                "provider cannot be imported without an explicit request"
            )

    @property
    def complete(self) -> bool:
        return self.status is RetrievalStatus.COMPLETE

    @property
    def truncated(self) -> bool:
        return self.status is RetrievalStatus.TRUNCATED

    @property
    def safe_for_completion_reasoning(self) -> bool:
        # GraphRAG is candidate context even when its deterministic closure is
        # structurally complete.
        return False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_RAG_RECEIPT_SCHEMA,
            "retriever_version": BOUNDED_GRAPH_RAG_RETRIEVER_VERSION,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "query_digest": self.query_digest,
            "candidates": [item.to_dict() for item in self.candidates],
            "bounds": self.bounds.to_dict(),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "forward_closure_id": self.forward_closure_id,
            "reverse_closure_id": self.reverse_closure_id,
            "provider_requested": self.provider_requested,
            "provider_imported": self.provider_imported,
            "provider_result_id": self.provider_result_id,
            "provider_status": self.provider_status,
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "completion_authority": False,
            "proof_authority": False,
        }

    @property
    def identity(self) -> ContentIdentity:
        return _identity(self._identity_payload())

    @property
    def receipt_id(self) -> str:
        return self.identity.cid

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "receipt_id": self.receipt_id,
            "identity": self.identity.to_dict(),
            "complete": self.complete,
            "truncated": self.truncated,
            "safe_for_completion_reasoning": False,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_contract_graph_bytes(self.to_dict()).decode("utf-8")
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphRAGRetrievalReceipt":
        if str(payload.get("schema") or GRAPH_RAG_RECEIPT_SCHEMA) != (
            GRAPH_RAG_RECEIPT_SCHEMA
        ):
            raise SymbolicContractGraphError("unsupported GraphRAG receipt schema")
        if payload.get("authority", ContractAuthority.CONTEXT_ONLY.value) != (
            ContractAuthority.CONTEXT_ONLY.value
        ):
            raise SymbolicContractGraphError(
                "GraphRAG receipt authority must remain context_only"
            )
        _verify_bool_claim(
            payload,
            "completion_authority",
            False,
            "GraphRAG receipt cannot grant completion authority",
        )
        _verify_bool_claim(
            payload,
            "proof_authority",
            False,
            "GraphRAG receipt cannot grant proof authority",
        )
        receipt = cls(
            graph_root=str(payload.get("graph_root") or ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            query_digest=str(payload.get("query_digest") or ""),
            candidates=tuple(payload.get("candidates") or ()),
            bounds=GraphRAGRetrievalBounds.from_dict(
                payload.get("bounds") or {}
            ),
            status=payload.get("status", ""),
            reason_code=str(payload.get("reason_code") or ""),
            forward_closure_id=str(
                payload.get("forward_closure_id") or ""
            ),
            reverse_closure_id=str(
                payload.get("reverse_closure_id") or ""
            ),
            provider_requested=payload.get("provider_requested", False),
            provider_imported=payload.get("provider_imported", False),
            provider_result_id=str(payload.get("provider_result_id") or ""),
            provider_status=str(payload.get("provider_status") or ""),
            missing_requirement_ids=tuple(
                payload.get("missing_requirement_ids") or ()
            ),
        )
        claimed = str(payload.get("receipt_id") or "")
        if claimed and claimed != receipt.receipt_id:
            raise SymbolicContractGraphError("GraphRAG receipt identity mismatch")
        claimed_identity = payload.get("identity")
        if (
            claimed_identity
            and _plain(claimed_identity) != receipt.identity.to_dict()
        ):
            raise SymbolicContractGraphError(
                "GraphRAG receipt ContentIdentity metadata mismatch"
            )
        _verify_bool_claim(
            payload,
            "complete",
            receipt.complete,
            "forged GraphRAG receipt completeness",
        )
        _verify_bool_claim(
            payload,
            "truncated",
            receipt.truncated,
            "forged GraphRAG truncation status",
        )
        _verify_bool_claim(
            payload,
            "safe_for_completion_reasoning",
            False,
            "GraphRAG receipt cannot be completion evidence",
        )
        return receipt

    @classmethod
    def from_json(cls, payload: str) -> "GraphRAGRetrievalReceipt":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SymbolicContractGraphError(
                "GraphRAG receipt JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise SymbolicContractGraphError(
                "GraphRAG receipt JSON must contain an object"
            )
        return cls.from_dict(value)


@dataclass(frozen=True)
class BoundedGraphRAGView:
    graph_root: str
    snapshot_id: str
    candidates: tuple[GraphRAGCandidate, ...]
    nodes: tuple[SymbolicContractNode, ...]
    edges: tuple[SymbolicContractEdge, ...]
    receipt: GraphRAGRetrievalReceipt
    forward_closure: ContractGraphClosure | None = None
    reverse_closure: ContractGraphClosure | None = None

    @property
    def complete(self) -> bool:
        return self.receipt.complete

    @property
    def truncated(self) -> bool:
        return self.receipt.truncated

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    def require_complete(self) -> "BoundedGraphRAGView":
        if self.complete:
            return self
        if self.truncated:
            raise TruncatedContractClosureError(self.receipt.reason_code)
        requirements = tuple(
            MandatoryEdgeRequirement(
                source=requirement_id,
                kind=ContractEdgeKind.DEPENDS_ON,
            )
            for requirement_id in self.receipt.missing_requirement_ids
        )
        if requirements:
            raise MissingMandatoryEdgeError(requirements)
        raise IncompleteContractClosureError(self.receipt.reason_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_RAG_VIEW_SCHEMA,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "candidates": [item.to_dict() for item in self.candidates],
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "forward_closure": (
                self.forward_closure.to_dict() if self.forward_closure else None
            ),
            "reverse_closure": (
                self.reverse_closure.to_dict() if self.reverse_closure else None
            ),
            "receipt": self.receipt.to_dict(),
            "complete": self.complete,
            "truncated": self.truncated,
            "completion_authority": False,
            "proof_authority": False,
        }


class BoundedGraphRAGRetriever:
    """Nominate bounded candidates, then add deterministic mandatory closure."""

    def __init__(
        self,
        graph: SymbolicContractGraph,
        *,
        bounds: GraphRAGRetrievalBounds | None = None,
        provider: Any = None,
        provider_factory: Callable[[], Any] | None = None,
        repository_id: str = "repository:unknown",
        objective_revision: str = "objective:unknown",
    ) -> None:
        if provider is not None and provider_factory is not None:
            raise SymbolicContractGraphError(
                "provider and provider_factory cannot both be supplied"
            )
        self.graph = graph
        self.bounds = bounds or GraphRAGRetrievalBounds()
        self._provider = provider
        self._provider_factory = provider_factory
        self.repository_id = _text(repository_id, "repository_id")
        self.objective_revision = _text(
            objective_revision, "objective_revision"
        )

    @staticmethod
    def _search_text(node: SymbolicContractNode) -> str:
        return " ".join(
            (
                node.node_id,
                node.kind.value,
                json.dumps(
                    _plain(node.record),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        ).lower()

    def _local_candidates(self, query: str) -> tuple[GraphRAGCandidate, ...]:
        query_tokens = tuple(sorted(set(_TOKEN.findall(query.lower()))))
        exact = query.lower()
        ranked: list[GraphRAGCandidate] = []
        for node in self.graph.nodes:
            if not node.authoritative:
                continue
            haystack = self._search_text(node)
            matched = sum(1 for token in query_tokens if token in haystack)
            exact_bonus = 1 if exact in haystack else 0
            if not matched and not exact_bonus:
                continue
            denominator = max(1, len(query_tokens) + 1)
            score = min(
                1_000_000,
                ((matched + exact_bonus) * 1_000_000) // denominator,
            )
            ranked.append(
                GraphRAGCandidate(
                    node_id=node.node_id,
                    score_millionths=score,
                    nomination_source="local_deterministic",
                )
            )
        return tuple(
            sorted(
                ranked,
                key=lambda item: (-item.score_millionths, item.node_id),
            )
        )

    def _load_provider(self) -> Any:
        if self._provider is not None:
            return self._provider
        if self._provider_factory is not None:
            self._provider = self._provider_factory()
            return self._provider
        # The adapter module is local, but it and the optional package are not
        # loaded until this explicit provider path is selected.
        from ..integrations.ipfs_datasets_analysis_provider import (
            IpfsDatasetsAnalysisProvider,
        )

        self._provider = IpfsDatasetsAnalysisProvider()
        return self._provider

    def _provider_candidates(
        self, query: str
    ) -> tuple[tuple[GraphRAGCandidate, ...], bool, str, str, bool]:
        provider = self._load_provider()
        request_payload = {
            "operation": "graph_retrieval",
            "repository_id": self.repository_id,
            "tree_id": self.graph.snapshot_id,
            "objective_revision": self.objective_revision,
            "query": {"text": query},
            "artifact_references": [
                {
                    "record_id": node.node_id,
                    "digest": node.identity.digest,
                    "kind": node.kind.value,
                }
                for node in self.graph.nodes[: self.bounds.max_candidates]
            ],
        }
        request = (
            provider.build_request(request_payload)
            if callable(getattr(provider, "build_request", None))
            else request_payload
        )
        result = provider.analyze(request)
        status_value = getattr(getattr(result, "status", ""), "value", None)
        status = str(status_value or getattr(result, "status", "unknown"))
        result_id = str(
            getattr(result, "result_id", "")
            or getattr(result, "content_id", "")
        )
        truncated = bool(getattr(result, "truncated", False))
        references = getattr(result, "evidence_references", ())
        successful = bool(
            getattr(result, "successful", status in {"completed", "complete"})
        )
        known = {node.node_id for node in self.graph.nodes}
        candidates: list[GraphRAGCandidate] = []
        if successful:
            for index, reference in enumerate(references):
                if not isinstance(reference, Mapping):
                    continue
                node_id = str(
                    reference.get("node_id")
                    or reference.get("record_id")
                    or reference.get("reference_id")
                    or reference.get("evidence_id")
                    or ""
                )
                if node_id not in known:
                    continue
                raw_score = reference.get("score_millionths")
                score = (
                    int(raw_score)
                    if isinstance(raw_score, int) and not isinstance(raw_score, bool)
                    else max(1, 900_000 - index)
                )
                candidates.append(
                    GraphRAGCandidate(
                        node_id=node_id,
                        score_millionths=max(0, min(1_000_000, score)),
                        nomination_source="ipfs_datasets_graphrag",
                    )
                )
        return tuple(candidates), truncated, status, result_id, successful

    @staticmethod
    def _merge_candidates(
        local: Sequence[GraphRAGCandidate],
        provider: Sequence[GraphRAGCandidate],
    ) -> tuple[GraphRAGCandidate, ...]:
        by_node: dict[str, GraphRAGCandidate] = {}
        for candidate in (*provider, *local):
            previous = by_node.get(candidate.node_id)
            if previous is None or (
                candidate.score_millionths,
                candidate.nomination_source,
            ) > (
                previous.score_millionths,
                previous.nomination_source,
            ):
                by_node[candidate.node_id] = candidate
        return tuple(
            sorted(
                by_node.values(),
                key=lambda item: (
                    -item.score_millionths,
                    item.node_id,
                    item.nomination_source,
                ),
            )
        )

    def _failed_view(
        self,
        *,
        query_digest: str,
        candidates: tuple[GraphRAGCandidate, ...],
        status: RetrievalStatus,
        reason_code: str,
        provider_requested: bool,
        provider_imported: bool,
        provider_result_id: str,
        provider_status: str,
        missing_requirement_ids: tuple[str, ...] = (),
    ) -> BoundedGraphRAGView:
        selected = candidates[: self.bounds.max_candidates]
        receipt = GraphRAGRetrievalReceipt(
            graph_root=self.graph.graph_root,
            snapshot_id=self.graph.snapshot_id,
            query_digest=query_digest,
            candidates=selected,
            bounds=self.bounds,
            status=status,
            reason_code=reason_code,
            provider_requested=provider_requested,
            provider_imported=provider_imported,
            provider_result_id=provider_result_id,
            provider_status=provider_status,
            missing_requirement_ids=missing_requirement_ids,
        )
        return BoundedGraphRAGView(
            graph_root=self.graph.graph_root,
            snapshot_id=self.graph.snapshot_id,
            candidates=selected,
            nodes=(),
            edges=(),
            receipt=receipt,
        )

    def retrieve(
        self,
        query: str,
        *,
        use_optional_provider: bool = False,
        candidate_node_ids: Iterable[str] | None = None,
    ) -> BoundedGraphRAGView:
        query = _text(
            query,
            "retrieval query",
            max_bytes=self.bounds.max_query_bytes,
        )
        query_digest = _digest(
            {
                "query": query,
                "graph_root": self.graph.graph_root,
                "bounds": self.bounds.to_dict(),
            }
        )
        if candidate_node_ids is None:
            local = self._local_candidates(query)
        else:
            known = {node.node_id for node in self.graph.nodes}
            requested = tuple(
                sorted({_text(item, "candidate node_id") for item in candidate_node_ids})
            )
            unknown = set(requested) - known
            if unknown:
                raise KeyError(sorted(unknown)[0])
            local = tuple(
                GraphRAGCandidate(node_id, 1_000_000, "caller_candidate")
                for node_id in requested
            )

        provider_candidates: tuple[GraphRAGCandidate, ...] = ()
        provider_truncated = False
        provider_status = "not_requested"
        provider_result_id = ""
        provider_successful = False
        provider_imported = False
        if use_optional_provider:
            try:
                (
                    provider_candidates,
                    provider_truncated,
                    provider_status,
                    provider_result_id,
                    provider_successful,
                ) = self._provider_candidates(query)
                provider_imported = True
            except Exception:
                # Optional provider failure is explicitly recorded and local
                # deterministic retrieval remains available.
                provider_status = "provider_dispatch_failed"
                provider_imported = True

        candidates = self._merge_candidates(local, provider_candidates)
        if provider_truncated:
            return self._failed_view(
                query_digest=query_digest,
                candidates=candidates,
                status=RetrievalStatus.TRUNCATED,
                reason_code="optional_provider_truncated",
                provider_requested=True,
                provider_imported=provider_imported,
                provider_result_id=provider_result_id,
                provider_status=provider_status,
            )
        if len(candidates) > self.bounds.max_candidates:
            return self._failed_view(
                query_digest=query_digest,
                candidates=candidates,
                status=RetrievalStatus.TRUNCATED,
                reason_code="candidate_bound_exceeded",
                provider_requested=use_optional_provider,
                provider_imported=provider_imported,
                provider_result_id=provider_result_id,
                provider_status=provider_status,
            )

        seed_ids = tuple(candidate.node_id for candidate in candidates)
        if not seed_ids:
            receipt = GraphRAGRetrievalReceipt(
                graph_root=self.graph.graph_root,
                snapshot_id=self.graph.snapshot_id,
                query_digest=query_digest,
                candidates=(),
                bounds=self.bounds,
                status=RetrievalStatus.COMPLETE,
                reason_code=(
                    "no_candidates"
                    if not use_optional_provider or provider_successful
                    else "provider_degraded_local_fallback"
                ),
                provider_requested=use_optional_provider,
                provider_imported=provider_imported,
                provider_result_id=provider_result_id,
                provider_status=provider_status,
            )
            return BoundedGraphRAGView(
                graph_root=self.graph.graph_root,
                snapshot_id=self.graph.snapshot_id,
                candidates=(),
                nodes=(),
                edges=(),
                receipt=receipt,
            )

        try:
            forward = self.graph.forward_closure(
                seed_ids, bounds=self.bounds.closure_bounds
            )
            reverse = self.graph.reverse_closure(
                seed_ids, bounds=self.bounds.closure_bounds
            )
        except MissingMandatoryEdgeError as exc:
            return self._failed_view(
                query_digest=query_digest,
                candidates=candidates,
                status=RetrievalStatus.INCOMPLETE,
                reason_code="missing_mandatory_edges",
                provider_requested=use_optional_provider,
                provider_imported=provider_imported,
                provider_result_id=provider_result_id,
                provider_status=provider_status,
                missing_requirement_ids=tuple(
                    item.requirement_id for item in exc.requirements
                ),
            )
        except TruncatedContractClosureError as exc:
            return self._failed_view(
                query_digest=query_digest,
                candidates=candidates,
                status=RetrievalStatus.TRUNCATED,
                reason_code=exc.reason_code,
                provider_requested=use_optional_provider,
                provider_imported=provider_imported,
                provider_result_id=provider_result_id,
                provider_status=provider_status,
            )

        node_ids = set(forward.node_ids) | set(reverse.node_ids)
        edge_ids = set(forward.edge_ids) | set(reverse.edge_ids)
        receipt = GraphRAGRetrievalReceipt(
            graph_root=self.graph.graph_root,
            snapshot_id=self.graph.snapshot_id,
            query_digest=query_digest,
            candidates=candidates,
            bounds=self.bounds,
            status=RetrievalStatus.COMPLETE,
            reason_code=(
                "complete"
                if not use_optional_provider or provider_successful
                else "provider_degraded_local_fallback"
            ),
            forward_closure_id=forward.closure_id,
            reverse_closure_id=reverse.closure_id,
            provider_requested=use_optional_provider,
            provider_imported=provider_imported,
            provider_result_id=provider_result_id,
            provider_status=provider_status,
        )
        return BoundedGraphRAGView(
            graph_root=self.graph.graph_root,
            snapshot_id=self.graph.snapshot_id,
            candidates=candidates,
            nodes=tuple(node for node in self.graph.nodes if node.node_id in node_ids),
            edges=tuple(edge for edge in self.graph.edges if edge.edge_id in edge_ids),
            receipt=receipt,
            forward_closure=forward,
            reverse_closure=reverse,
        )


def build_symbolic_contract_graph(
    *,
    snapshot_id: str,
    nodes: Iterable[SymbolicContractNode | Mapping[str, Any]] = (),
    edges: Iterable[SymbolicContractEdge | Mapping[str, Any]] = (),
    mandatory_edge_requirements: Iterable[
        MandatoryEdgeRequirement | Mapping[str, Any]
    ] = (),
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION,
) -> SymbolicContractGraph:
    """Build a canonical graph from typed producer projections."""

    return SymbolicContractGraph(
        snapshot_id=snapshot_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        mandatory_edge_requirements=tuple(mandatory_edge_requirements),
        version=version,
    )


def project_symbolic_contract_graph(
    *,
    snapshot_id: str,
    node_records: Iterable[SymbolicContractNode | Mapping[str, Any]] = (),
    edge_records: Iterable[SymbolicContractEdge | Mapping[str, Any]] = (),
    mandatory_edge_requirements: Iterable[
        MandatoryEdgeRequirement | Mapping[str, Any]
    ] = (),
    version: str = SYMBOLIC_CONTRACT_GRAPH_VERSION,
) -> SymbolicContractGraph:
    """Compatibility spelling emphasizing projection from indexed facts."""

    return build_symbolic_contract_graph(
        snapshot_id=snapshot_id,
        nodes=node_records,
        edges=edge_records,
        mandatory_edge_requirements=mandatory_edge_requirements,
        version=version,
    )


# Interface-friendly aliases used by adjacent graph/proof components.
ContractGraphNode = SymbolicContractNode
ContractGraphEdge = SymbolicContractEdge
CodeEvidenceGraphNode = SymbolicContractNode
CodeEvidenceGraphEdge = SymbolicContractEdge
GraphNode = SymbolicContractNode
GraphEdge = SymbolicContractEdge
GraphNodeKind = ContractNodeKind
GraphEdgeKind = ContractEdgeKind
GraphProvenance = ContractProvenance
GraphAuthority = ContractAuthority
ClosureBounds = ContractClosureBounds
MandatoryClosure = ContractGraphClosure
BoundedGraphRAGResult = BoundedGraphRAGView
GraphRAGView = BoundedGraphRAGView
CandidateRetrievalReceipt = GraphRAGRetrievalReceipt
RetrievalBounds = GraphRAGRetrievalBounds
RetrievalReceipt = GraphRAGRetrievalReceipt


__all__ = [
    "BOUNDED_GRAPH_RAG_RETRIEVER_VERSION",
    "BOUNDED_GRAPH_RAG_RETRIEVER_INTERFACE",
    "CONTRACT_GRAPH_CLOSURE_SCHEMA",
    "GRAPH_RAG_RECEIPT_SCHEMA",
    "SYMBOLIC_CONTRACT_GRAPH_SCHEMA",
    "SYMBOLIC_CONTRACT_GRAPH_INTERFACE",
    "SYMBOLIC_CONTRACT_GRAPH_VERSION",
    "BoundedGraphRAGResult",
    "BoundedGraphRAGRetriever",
    "BoundedGraphRAGView",
    "CandidateRetrievalReceipt",
    "ClosureBounds",
    "ClosureDirection",
    "CodeEvidenceGraphEdge",
    "CodeEvidenceGraphNode",
    "ContractAuthority",
    "ContractClosureBounds",
    "ContractEdgeKind",
    "ContractGraphBoundsError",
    "ContractGraphClosure",
    "ContractGraphEdge",
    "ContractGraphNode",
    "ContractNodeKind",
    "ContractProvenance",
    "GraphRAGCandidate",
    "GraphRAGView",
    "GraphRAGRetrievalBounds",
    "GraphRAGRetrievalReceipt",
    "GraphAuthority",
    "GraphEdge",
    "GraphEdgeKind",
    "GraphNode",
    "GraphNodeKind",
    "GraphProvenance",
    "IncompleteContractClosureError",
    "MandatoryClosure",
    "MandatoryEdgeRequirement",
    "MissingMandatoryEdgeError",
    "RetrievalBounds",
    "RetrievalReceipt",
    "RetrievalStatus",
    "SymbolicContractEdge",
    "SymbolicContractGraph",
    "SymbolicContractGraphError",
    "SymbolicContractNode",
    "TruncatedContractClosureError",
    "build_symbolic_contract_graph",
    "canonical_contract_graph_bytes",
    "project_symbolic_contract_graph",
]
