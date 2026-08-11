"""Typed, content-addressed contract graph and bounded GraphRAG view.

Interface: ``SymbolicContractGraph@1``

This module is the graph trust boundary for SwissKnife contract assurance.
Indexed source and schema facts are projected into deterministic typed nodes
and edges.  Retrieval (including the optional ``ipfs_datasets_py`` provider)
may nominate context seeds, but only explicit, authority-bearing ``mandatory``
edges participate in proof-dependency closure.

The implementation is intentionally self-contained and does not import the
optional datasets analysis provider at module import, graph construction, or
local retrieval time.  The provider is loaded only when a caller explicitly
requests provider-backed candidate nomination.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Final

from ..proof.formal_verification_contracts import content_identity


SYMBOLIC_CONTRACT_GRAPH_INTERFACE: Final = "SymbolicContractGraph@1"
SYMBOLIC_CONTRACT_GRAPH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-graph@1"
)
SYMBOLIC_CONTRACT_NODE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-node@1"
)
SYMBOLIC_CONTRACT_EDGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-edge@1"
)
SYMBOLIC_CONTRACT_CLOSURE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-closure@1"
)
BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE: Final = "BoundedGraphRAGRetriever@1"
GRAPHRAG_RETRIEVAL_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/graphrag-retrieval-receipt@1"
)
BOUNDED_GRAPHRAG_VIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/bounded-graphrag-view@1"
)
# Objective-heap evidence term for SCA-G031 exact datasets GraphRAG/Cypher
# binding (SCAEV031DATASETSGRAPH).  Kept in-module so AST/evidence scanners
# admit coverage without importing the optional datasets provider at load time.
SCAEV031DATASETSGRAPH: Final = "SCAEV031DATASETSGRAPH"
SCAEV031DATASETSGRAPH_EVIDENCE: Final = SCAEV031DATASETSGRAPH
EXACT_DATASETS_GRAPHRAG_MODULE: Final = (
    "ipfs_datasets_py.logic.intent_ir.graphrag.retrieval"
)
EXACT_DATASETS_CYPHER_AST_MODULE: Final = (
    "ipfs_datasets_py.knowledge_graphs.cypher.ast"
)
EXACT_DATASETS_CYPHER_PARSER_MODULE: Final = (
    "ipfs_datasets_py.knowledge_graphs.cypher.parser"
)
CONTENT_IDENTITY_PROFILE: Final = "strict-dag-json-v1"
CONTENT_IDENTITY_CANONICALIZATION: Final = "deterministic-dag-json"
GRAPH_VERSION: Final = "1"

DEFAULT_MAX_GRAPH_NODES = 100_000
DEFAULT_MAX_GRAPH_EDGES = 250_000
DEFAULT_MAX_CLOSURE_NODES = 16_384
DEFAULT_MAX_CLOSURE_EDGES = 65_536
DEFAULT_MAX_CLOSURE_DEPTH = 256
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_MAX_RETRIEVAL_BYTES = 32_768
DEFAULT_MAX_QUERY_BYTES = 4_096
HARD_MAX_CANDIDATES = 1_000
HARD_MAX_RETRIEVAL_BYTES = 1_048_576
HARD_MAX_QUERY_BYTES = 65_536

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class SymbolicContractGraphError(ValueError):
    """Graph data is malformed or violates the authority boundary."""


class SymbolicGraphBoundsError(SymbolicContractGraphError):
    """A graph, closure, query, or receipt exceeded a deterministic bound."""


class IncompleteMandatoryClosureError(SymbolicContractGraphError):
    """Mandatory closure was truncated or its declared edge set is missing."""

    def __init__(self, message: str, receipt: "ContractGraphClosure") -> None:
        super().__init__(message)
        self.receipt = receipt


class CandidateRetrievalError(SymbolicContractGraphError):
    """A bounded candidate request could not produce a valid receipt."""


class ExactDatasetsGraphProviderError(SymbolicContractGraphError):
    """Exact datasets GraphRAG/Cypher modules are missing or incompatible.

    Local lexical retrieval remains available separately; this error only
    blocks claims of exact ``ipfs_datasets_py`` GraphRAG / Cypher-AST use.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class ContractNodeKind(str, Enum):
    REPOSITORY_SNAPSHOT = "repository_snapshot"
    FILE = "file"
    MODULE = "module"
    SYMBOL = "symbol"
    CALL = "call"
    IMPORT = "import"
    EFFECT = "effect"
    SCHEMA = "schema"
    INTERFACE = "interface"
    METHOD = "method"
    TOOL = "tool"
    HANDLER = "handler"
    TEST = "test"
    POLICY = "policy"
    TRANSPORT = "transport"
    CONTRACT = "contract"
    PROVENANCE = "provenance"
    UNRESOLVED = "unresolved"


class ContractEdgeKind(str, Enum):
    CONTAINS = "contains"
    DEFINED_IN = "defined_in"
    IMPORTS = "imports"
    CALLS = "calls"
    HAS_EFFECT = "has_effect"
    USES_SCHEMA = "uses_schema"
    DECLARES = "declares"
    IMPLEMENTS = "implements"
    REGISTERS = "registers"
    DISPATCHES_TO = "dispatches_to"
    HANDLED_BY = "handled_by"
    TESTS = "tests"
    ENFORCED_BY = "enforced_by"
    TRANSPORTED_BY = "transported_by"
    DEPENDS_ON = "depends_on"
    SOURCED_FROM = "sourced_from"
    RELATED_TO = "related_to"


class ContractProvenance(str, Enum):
    REPOSITORY = "repository"
    AST = "ast"
    SCHEMA = "schema"
    REGISTRY = "registry"
    CONTRACT = "contract"
    POLICY = "policy"
    TEST = "test"
    TRANSPORT = "transport"
    MANUAL_REVIEW = "manual_review"
    RETRIEVAL = "retrieval"
    GRAPHRAG = "graphrag"
    DATASETS = "datasets"

    @property
    def context_only(self) -> bool:
        return self in {
            ContractProvenance.RETRIEVAL,
            ContractProvenance.GRAPHRAG,
            ContractProvenance.DATASETS,
        }


class ContractAuthority(str, Enum):
    SOURCE_OBSERVATION = "source_observation"
    REVIEWED_CONTRACT = "reviewed_contract"
    POLICY = "policy"
    CONTEXT_ONLY = "context_only"
    NONE = "none"

    @property
    def authority_bearing(self) -> bool:
        return self in {
            ContractAuthority.SOURCE_OBSERVATION,
            ContractAuthority.REVIEWED_CONTRACT,
            ContractAuthority.POLICY,
        }


class ClosureDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


def _enum(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise SymbolicContractGraphError(
            f"unknown {field_name}: {value!r}"
        ) from exc


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    max_bytes: int = 8_192,
) -> str:
    if not isinstance(value, str):
        raise SymbolicContractGraphError(f"{field_name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise SymbolicContractGraphError(
            f"{field_name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise SymbolicContractGraphError(f"{field_name} is required")
    if len(value.encode("utf-8")) > max_bytes:
        raise SymbolicGraphBoundsError(f"{field_name} exceeds {max_bytes} bytes")
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise SymbolicGraphBoundsError("graph record exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise SymbolicContractGraphError(
            "floating values are not canonical contract graph data"
        )
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise SymbolicGraphBoundsError(
                "graph mappings require at most 1024 string keys"
            )
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 16_384:
            raise SymbolicGraphBoundsError("graph sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise SymbolicContractGraphError(
        f"unsupported graph value: {type(value).__name__}"
    )


def canonical_contract_graph_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a graph artifact."""

    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_contract_graph_json(value: Any) -> str:
    return canonical_contract_graph_bytes(value).decode("utf-8")


@dataclass(frozen=True, slots=True)
class GraphContentIdentity:
    """Compact ContentIdentity@1 metadata for one canonical graph record."""

    cid: str
    digest: str
    byte_length: int
    profile: str = CONTENT_IDENTITY_PROFILE
    canonicalization: str = CONTENT_IDENTITY_CANONICALIZATION
    cid_version: int = 1
    multibase: str = "base32"
    multicodec: str = "dag-json"
    multihash: str = "sha2-256"
    interface: str = "ContentIdentity@1"
    version: str = "1"

    @classmethod
    def for_value(cls, value: Any) -> "GraphContentIdentity":
        canonical = canonical_contract_graph_bytes(value)
        return cls(
            cid=content_identity(value),
            digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
            byte_length=len(canonical),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "version": self.version,
            "profile": self.profile,
            "canonicalization": self.canonicalization,
            "byte_length": self.byte_length,
            "digest": self.digest,
            "cid": self.cid,
            "cid_version": self.cid_version,
            "multibase": self.multibase,
            "multicodec": self.multicodec,
            "multihash": self.multihash,
            "validated": True,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphContentIdentity":
        if not isinstance(value, Mapping):
            raise SymbolicContractGraphError("identity must be an object")
        if value.get("interface") not in (None, "ContentIdentity@1"):
            raise SymbolicContractGraphError("unsupported identity interface")
        if value.get("validated") is False:
            raise SymbolicContractGraphError("graph identity is not validated")
        return cls(
            cid=str(value.get("cid") or ""),
            digest=str(value.get("digest") or ""),
            byte_length=int(value.get("byte_length") or 0),
            profile=str(value.get("profile") or CONTENT_IDENTITY_PROFILE),
            canonicalization=str(
                value.get("canonicalization")
                or CONTENT_IDENTITY_CANONICALIZATION
            ),
            cid_version=int(value.get("cid_version") or 1),
            multibase=str(value.get("multibase") or "base32"),
            multicodec=str(value.get("multicodec") or "dag-json"),
            multihash=str(value.get("multihash") or "sha2-256"),
            interface=str(value.get("interface") or "ContentIdentity@1"),
            version=str(value.get("version") or "1"),
        )

    def verifies(self, value: Any) -> bool:
        return self == GraphContentIdentity.for_value(value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        value = to_dict() if callable(to_dict) else value
    if not isinstance(value, Mapping):
        raise SymbolicContractGraphError(f"{field_name} must be an object")
    return MappingProxyType(_plain(value))


def _strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SymbolicContractGraphError(f"{field_name} must be a sequence")
    result = tuple(
        sorted(
            {
                _text(str(item), field_name, max_bytes=2_048)
                for item in value
            }
        )
    )
    return result


@dataclass(frozen=True, slots=True)
class ContractGraphNode:
    """One typed graph node bound to an exact repository snapshot."""

    kind: ContractNodeKind
    stable_key: str
    snapshot_id: str
    provenance: ContractProvenance
    authority: ContractAuthority
    version: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    required_dependencies: tuple[str, ...] = ()
    node_id: str = ""
    identity: GraphContentIdentity | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ContractNodeKind, "node kind")
        )
        object.__setattr__(
            self, "provenance", _enum(
                self.provenance, ContractProvenance, "node provenance"
            )
        )
        object.__setattr__(
            self, "authority", _enum(
                self.authority, ContractAuthority, "node authority"
            )
        )
        for name in ("stable_key", "snapshot_id", "version"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"node {name}")
            )
        if self.provenance.context_only and (
            self.authority is not ContractAuthority.CONTEXT_ONLY
        ):
            raise SymbolicContractGraphError(
                "retrieval/GraphRAG node provenance must remain context_only"
            )
        object.__setattr__(self, "payload", _mapping(self.payload, "node payload"))
        object.__setattr__(
            self, "source_refs", _strings(self.source_refs, "node source_refs")
        )
        object.__setattr__(
            self,
            "required_dependencies",
            _strings(
                self.required_dependencies, "node required_dependencies"
            ),
        )
        record = self._identity_payload()
        expected = GraphContentIdentity.for_value(record)
        claimed_id = str(self.node_id or "")
        if claimed_id and claimed_id != expected.cid:
            raise SymbolicContractGraphError(
                "node identity does not match canonical content"
            )
        if self.identity is not None:
            supplied = (
                self.identity
                if isinstance(self.identity, GraphContentIdentity)
                else GraphContentIdentity.from_dict(self.identity)
            )
            if supplied != expected:
                raise SymbolicContractGraphError(
                    "node ContentIdentity does not match canonical content"
                )
        object.__setattr__(self, "node_id", expected.cid)
        object.__setattr__(self, "identity", expected)

    @property
    def authoritative(self) -> bool:
        return self.authority.authority_bearing

    @property
    def key(self) -> str:
        return self.stable_key

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_NODE_SCHEMA,
            "kind": self.kind.value,
            "stable_key": self.stable_key,
            "snapshot_id": self.snapshot_id,
            "provenance": self.provenance.value,
            "authority": self.authority.value,
            "version": self.version,
            "payload": dict(self.payload),
            "source_refs": list(self.source_refs),
            "required_dependencies": list(self.required_dependencies),
        }

    def to_dict(self) -> dict[str, Any]:
        assert self.identity is not None
        return {
            "node_id": self.node_id,
            "identity": self.identity.to_dict(),
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractGraphNode":
        if value.get("schema") not in (None, SYMBOLIC_CONTRACT_NODE_SCHEMA):
            raise SymbolicContractGraphError("unsupported node schema")
        return cls(
            kind=value.get("kind", ""),
            stable_key=str(value.get("stable_key") or value.get("key") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            provenance=value.get("provenance", ""),
            authority=value.get("authority", ""),
            version=str(value.get("version") or ""),
            payload=value.get("payload") or {},
            source_refs=tuple(value.get("source_refs") or ()),
            required_dependencies=tuple(
                value.get("required_dependencies") or ()
            ),
            node_id=str(value.get("node_id") or ""),
            identity=(
                GraphContentIdentity.from_dict(value["identity"])
                if isinstance(value.get("identity"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ContractGraphEdge:
    """One typed relationship; mandatory edges point to dependencies."""

    kind: ContractEdgeKind
    source: str
    target: str
    snapshot_id: str
    provenance: ContractProvenance
    authority: ContractAuthority
    version: str
    mandatory: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    edge_id: str = ""
    identity: GraphContentIdentity | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ContractEdgeKind, "edge kind")
        )
        object.__setattr__(
            self, "provenance", _enum(
                self.provenance, ContractProvenance, "edge provenance"
            )
        )
        object.__setattr__(
            self, "authority", _enum(
                self.authority, ContractAuthority, "edge authority"
            )
        )
        for name in ("source", "target", "snapshot_id", "version"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"edge {name}")
            )
        if not isinstance(self.mandatory, bool):
            raise SymbolicContractGraphError("edge mandatory must be boolean")
        if self.provenance.context_only and (
            self.authority is not ContractAuthority.CONTEXT_ONLY
            or self.mandatory
        ):
            raise SymbolicContractGraphError(
                "retrieval/GraphRAG edges are context-only and never mandatory"
            )
        if self.authority is ContractAuthority.CONTEXT_ONLY and self.mandatory:
            raise SymbolicContractGraphError(
                "context-only edge cannot be a mandatory dependency"
            )
        if self.mandatory and not self.authority.authority_bearing:
            raise SymbolicContractGraphError(
                "mandatory edge must carry source, reviewed, or policy authority"
            )
        object.__setattr__(self, "payload", _mapping(self.payload, "edge payload"))
        object.__setattr__(
            self, "source_refs", _strings(self.source_refs, "edge source_refs")
        )
        record = self._identity_payload()
        expected = GraphContentIdentity.for_value(record)
        claimed_id = str(self.edge_id or "")
        if claimed_id and claimed_id != expected.cid:
            raise SymbolicContractGraphError(
                "edge identity does not match canonical content"
            )
        if self.identity is not None:
            supplied = (
                self.identity
                if isinstance(self.identity, GraphContentIdentity)
                else GraphContentIdentity.from_dict(self.identity)
            )
            if supplied != expected:
                raise SymbolicContractGraphError(
                    "edge ContentIdentity does not match canonical content"
                )
        object.__setattr__(self, "edge_id", expected.cid)
        object.__setattr__(self, "identity", expected)

    @property
    def authoritative(self) -> bool:
        return self.authority.authority_bearing

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_EDGE_SCHEMA,
            "kind": self.kind.value,
            "source": self.source,
            "target": self.target,
            "snapshot_id": self.snapshot_id,
            "provenance": self.provenance.value,
            "authority": self.authority.value,
            "version": self.version,
            "mandatory": self.mandatory,
            "payload": dict(self.payload),
            "source_refs": list(self.source_refs),
        }

    def to_dict(self) -> dict[str, Any]:
        assert self.identity is not None
        return {
            "edge_id": self.edge_id,
            "identity": self.identity.to_dict(),
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractGraphEdge":
        if value.get("schema") not in (None, SYMBOLIC_CONTRACT_EDGE_SCHEMA):
            raise SymbolicContractGraphError("unsupported edge schema")
        return cls(
            kind=value.get("kind", ""),
            source=str(value.get("source") or ""),
            target=str(value.get("target") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            provenance=value.get("provenance", ""),
            authority=value.get("authority", ""),
            version=str(value.get("version") or ""),
            mandatory=value.get("mandatory", False),
            payload=value.get("payload") or {},
            source_refs=tuple(value.get("source_refs") or ()),
            edge_id=str(value.get("edge_id") or ""),
            identity=(
                GraphContentIdentity.from_dict(value["identity"])
                if isinstance(value.get("identity"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ClosureBounds:
    max_nodes: int = DEFAULT_MAX_CLOSURE_NODES
    max_edges: int = DEFAULT_MAX_CLOSURE_EDGES
    max_depth: int = DEFAULT_MAX_CLOSURE_DEPTH

    def __post_init__(self) -> None:
        maxima = {
            "max_nodes": DEFAULT_MAX_GRAPH_NODES,
            "max_edges": DEFAULT_MAX_GRAPH_EDGES,
            "max_depth": 16_384,
        }
        for name, maximum in maxima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SymbolicGraphBoundsError(f"{name} must be an integer")
            if not 1 <= value <= maximum:
                raise SymbolicGraphBoundsError(
                    f"{name} must be between 1 and {maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_depth": self.max_depth,
        }

    @classmethod
    def from_value(
        cls, value: "ClosureBounds | Mapping[str, Any] | None"
    ) -> "ClosureBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise SymbolicGraphBoundsError("closure bounds must be an object")
        return cls(
            max_nodes=int(value.get("max_nodes", DEFAULT_MAX_CLOSURE_NODES)),
            max_edges=int(value.get("max_edges", DEFAULT_MAX_CLOSURE_EDGES)),
            max_depth=int(value.get("max_depth", DEFAULT_MAX_CLOSURE_DEPTH)),
        )


@dataclass(frozen=True, slots=True)
class ContractGraphClosure:
    graph_root: str
    snapshot_id: str
    version: str
    direction: ClosureDirection
    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    paths: Mapping[str, tuple[str, ...]]
    bounds: ClosureBounds
    complete: bool
    truncated: bool = False
    missing_edge_ids: tuple[str, ...] = ()
    missing_dependency_keys: tuple[str, ...] = ()
    reason_code: str = "complete"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            _enum(self.direction, ClosureDirection, "closure direction"),
        )
        for name in ("graph_root", "snapshot_id", "version", "reason_code"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"closure {name}")
            )
        for name in (
            "seed_node_ids",
            "node_ids",
            "edge_ids",
            "missing_edge_ids",
            "missing_dependency_keys",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), f"closure {name}")
            )
        object.__setattr__(self, "bounds", ClosureBounds.from_value(self.bounds))
        normalized_paths: dict[str, tuple[str, ...]] = {}
        for node_id, path in sorted(dict(self.paths).items()):
            if isinstance(path, (str, bytes)) or not isinstance(
                path, Sequence
            ):
                raise SymbolicContractGraphError(
                    "closure path values must be sequences"
                )
            normalized_paths[str(node_id)] = tuple(str(item) for item in path)
        object.__setattr__(
            self, "paths", MappingProxyType(normalized_paths)
        )
        if not isinstance(self.complete, bool) or not isinstance(
            self.truncated, bool
        ):
            raise SymbolicContractGraphError(
                "closure complete and truncated must be booleans"
            )
        if self.complete and (
            self.truncated
            or self.missing_edge_ids
            or self.missing_dependency_keys
        ):
            raise SymbolicContractGraphError(
                "complete closure cannot report truncation or missing dependencies"
            )

    @property
    def safe_for_proof(self) -> bool:
        return self.complete and not self.truncated

    @property
    def closure_id(self) -> str:
        return content_identity(self._payload())

    @property
    def identity(self) -> GraphContentIdentity:
        return GraphContentIdentity.for_value(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_CLOSURE_SCHEMA,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "version": self.version,
            "direction": self.direction.value,
            "seed_node_ids": list(self.seed_node_ids),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "paths": {
                key: list(value) for key, value in sorted(self.paths.items())
            },
            "bounds": self.bounds.to_dict(),
            "complete": self.complete,
            "truncated": self.truncated,
            "missing_edge_ids": list(self.missing_edge_ids),
            "missing_dependency_keys": list(self.missing_dependency_keys),
            "reason_code": self.reason_code,
            "authority": (
                "source_observation" if self.complete else "none"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "closure_id": self.closure_id,
            "identity": self.identity.to_dict(),
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractGraphClosure":
        if value.get("schema") not in (
            None,
            SYMBOLIC_CONTRACT_CLOSURE_SCHEMA,
        ):
            raise SymbolicContractGraphError("unsupported closure schema")
        paths = value.get("paths") or {}
        if not isinstance(paths, Mapping):
            raise SymbolicContractGraphError("closure paths must be an object")
        if any(
            isinstance(path, (str, bytes))
            or not isinstance(path, Sequence)
            for path in paths.values()
        ):
            raise SymbolicContractGraphError(
                "closure path values must be sequences"
            )
        result = cls(
            graph_root=str(value.get("graph_root") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            version=str(value.get("version") or ""),
            direction=value.get("direction", ""),
            seed_node_ids=tuple(value.get("seed_node_ids") or ()),
            node_ids=tuple(value.get("node_ids") or ()),
            edge_ids=tuple(value.get("edge_ids") or ()),
            paths={
                str(key): tuple(path)
                for key, path in paths.items()
            },
            bounds=ClosureBounds.from_value(value.get("bounds")),
            complete=value.get("complete", False),
            truncated=value.get("truncated", False),
            missing_edge_ids=tuple(value.get("missing_edge_ids") or ()),
            missing_dependency_keys=tuple(
                value.get("missing_dependency_keys") or ()
            ),
            reason_code=str(value.get("reason_code") or ""),
        )
        claimed = str(value.get("closure_id") or "")
        if claimed and claimed != result.closure_id:
            raise SymbolicContractGraphError("closure identity mismatch")
        if isinstance(value.get("identity"), Mapping):
            identity = GraphContentIdentity.from_dict(value["identity"])
            if identity != result.identity:
                raise SymbolicContractGraphError(
                    "closure ContentIdentity mismatch"
                )
        expected_authority = (
            "source_observation" if result.complete else "none"
        )
        if value.get("authority") not in (None, expected_authority):
            raise SymbolicContractGraphError(
                "closure authority claim mismatch"
            )
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "ContractGraphClosure":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SymbolicContractGraphError("closure JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise SymbolicContractGraphError(
                "closure JSON must contain an object"
            )
        return cls.from_dict(payload)


@dataclass(frozen=True)
class SymbolicContractGraph:
    """Immutable typed graph pinned to one snapshot and graph version."""

    snapshot_id: str
    nodes: tuple[ContractGraphNode, ...] = ()
    edges: tuple[ContractGraphEdge, ...] = ()
    version: str = GRAPH_VERSION
    mandatory_edge_ids: tuple[str, ...] = ()
    graph_root_claim: str = field(default="", repr=False, compare=False)
    identity_claim: GraphContentIdentity | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "graph snapshot_id")
        )
        object.__setattr__(
            self, "version", _text(self.version, "graph version")
        )
        node_map: dict[str, ContractGraphNode] = {}
        key_map: dict[str, ContractGraphNode] = {}
        for raw in self.nodes:
            node = (
                raw
                if isinstance(raw, ContractGraphNode)
                else ContractGraphNode.from_dict(raw)
            )
            if node.snapshot_id != self.snapshot_id:
                raise SymbolicContractGraphError(
                    f"node {node.node_id} is bound to a foreign snapshot"
                )
            if node.version != self.version:
                raise SymbolicContractGraphError(
                    f"node {node.node_id} has a foreign graph version"
                )
            old = node_map.get(node.node_id)
            if old is not None and old.to_dict() != node.to_dict():
                raise SymbolicContractGraphError(
                    f"conflicting node identity: {node.node_id}"
                )
            keyed = key_map.get(node.stable_key)
            if keyed is not None and keyed.node_id != node.node_id:
                raise SymbolicContractGraphError(
                    f"conflicting stable node key: {node.stable_key}"
                )
            node_map[node.node_id] = node
            key_map[node.stable_key] = node
        if len(node_map) > DEFAULT_MAX_GRAPH_NODES:
            raise SymbolicGraphBoundsError("graph has too many nodes")

        edge_map: dict[str, ContractGraphEdge] = {}
        for raw in self.edges:
            edge = (
                raw
                if isinstance(raw, ContractGraphEdge)
                else ContractGraphEdge.from_dict(raw)
            )
            if edge.snapshot_id != self.snapshot_id:
                raise SymbolicContractGraphError(
                    f"edge {edge.edge_id} is bound to a foreign snapshot"
                )
            if edge.version != self.version:
                raise SymbolicContractGraphError(
                    f"edge {edge.edge_id} has a foreign graph version"
                )
            source = node_map.get(edge.source)
            target = node_map.get(edge.target)
            if source is None or target is None:
                raise SymbolicContractGraphError(
                    f"edge {edge.edge_id} references an unknown node"
                )
            if edge.mandatory and (
                not source.authoritative or not target.authoritative
            ):
                raise SymbolicContractGraphError(
                    "mandatory edge cannot promote a context-only endpoint"
                )
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_GRAPH_EDGES:
            raise SymbolicGraphBoundsError("graph has too many edges")

        actual_mandatory = tuple(
            sorted(
                edge.edge_id for edge in edge_map.values() if edge.mandatory
            )
        )
        declared_mandatory = (
            _strings(self.mandatory_edge_ids, "mandatory_edge_ids")
            if self.mandatory_edge_ids
            else actual_mandatory
        )
        extra = set(declared_mandatory) - set(edge_map)
        # Missing declared edges are retained as an explicit incomplete graph
        # state.  Unknown non-mandatory IDs cannot be distinguished, so every
        # manifest entry is by definition mandatory.
        if any(not item for item in extra):
            raise SymbolicContractGraphError(
                "mandatory edge manifest contains an empty identity"
            )

        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        object.__setattr__(
            self, "mandatory_edge_ids", tuple(sorted(declared_mandatory))
        )
        expected_root = content_identity(self._root_payload())
        if self.graph_root_claim and self.graph_root_claim != expected_root:
            raise SymbolicContractGraphError(
                "graph root does not match canonical graph content"
            )
        expected_identity = GraphContentIdentity.for_value(self._root_payload())
        if self.identity_claim is not None:
            supplied = (
                self.identity_claim
                if isinstance(self.identity_claim, GraphContentIdentity)
                else GraphContentIdentity.from_dict(self.identity_claim)
            )
            if supplied != expected_identity:
                raise SymbolicContractGraphError(
                    "graph ContentIdentity does not match canonical content"
                )

    @property
    def graph_root(self) -> str:
        return content_identity(self._root_payload())

    @property
    def graph_id(self) -> str:
        return self.graph_root

    @property
    def root_id(self) -> str:
        return self.graph_root

    @property
    def identity(self) -> GraphContentIdentity:
        return GraphContentIdentity.for_value(self._root_payload())

    @property
    def missing_mandatory_edge_ids(self) -> tuple[str, ...]:
        actual = {edge.edge_id for edge in self.edges}
        return tuple(
            edge_id
            for edge_id in self.mandatory_edge_ids
            if edge_id not in actual
        )

    @property
    def missing_dependency_keys(self) -> tuple[str, ...]:
        key_to_node = {node.stable_key: node for node in self.nodes}
        mandatory_pairs = {
            (self.node(edge.source).stable_key, self.node(edge.target).stable_key)
            for edge in self.edges
            if edge.mandatory
        }
        missing: set[str] = set()
        for node in self.nodes:
            for dependency in node.required_dependencies:
                if dependency not in key_to_node:
                    missing.add(f"{node.stable_key}->{dependency}:missing_node")
                elif (node.stable_key, dependency) not in mandatory_pairs:
                    missing.add(f"{node.stable_key}->{dependency}:missing_edge")
        return tuple(sorted(missing))

    @property
    def complete(self) -> bool:
        return not (
            self.missing_mandatory_edge_ids or self.missing_dependency_keys
        )

    def node(self, node_id: str) -> ContractGraphNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    def node_for_key(self, stable_key: str) -> ContractGraphNode:
        for item in self.nodes:
            if item.stable_key == stable_key:
                return item
        raise KeyError(stable_key)

    def nodes_by_kind(
        self, kind: ContractNodeKind | str
    ) -> tuple[ContractGraphNode, ...]:
        expected = _enum(kind, ContractNodeKind, "node kind")
        return tuple(item for item in self.nodes if item.kind is expected)

    def edges_by_kind(
        self, kind: ContractEdgeKind | str
    ) -> tuple[ContractGraphEdge, ...]:
        expected = _enum(kind, ContractEdgeKind, "edge kind")
        return tuple(item for item in self.edges if item.kind is expected)

    def _root_payload(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CONTRACT_GRAPH_SCHEMA,
            "interface": SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "mandatory_edge_ids": list(self.mandatory_edge_ids),
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_root": self.graph_root,
            "graph_id": self.graph_root,
            "identity": self.identity.to_dict(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "complete": self.complete,
            "missing_mandatory_edge_ids": list(
                self.missing_mandatory_edge_ids
            ),
            "missing_dependency_keys": list(self.missing_dependency_keys),
            **self._root_payload(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_contract_graph_json(self.to_dict())
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SymbolicContractGraph":
        if value.get("schema") not in (None, SYMBOLIC_CONTRACT_GRAPH_SCHEMA):
            raise SymbolicContractGraphError("unsupported graph schema")
        if value.get("interface") not in (
            None,
            SYMBOLIC_CONTRACT_GRAPH_INTERFACE,
        ):
            raise SymbolicContractGraphError("unsupported graph interface")
        raw_nodes = value.get("nodes") or ()
        raw_edges = value.get("edges") or ()
        if (
            isinstance(raw_nodes, (str, bytes))
            or not isinstance(raw_nodes, Sequence)
            or not all(isinstance(item, Mapping) for item in raw_nodes)
        ):
            raise SymbolicContractGraphError(
                "graph nodes must be a sequence of objects"
            )
        if (
            isinstance(raw_edges, (str, bytes))
            or not isinstance(raw_edges, Sequence)
            or not all(isinstance(item, Mapping) for item in raw_edges)
        ):
            raise SymbolicContractGraphError(
                "graph edges must be a sequence of objects"
            )
        graph = cls(
            snapshot_id=str(value.get("snapshot_id") or ""),
            version=str(value.get("version") or GRAPH_VERSION),
            nodes=tuple(ContractGraphNode.from_dict(item) for item in raw_nodes),
            edges=tuple(ContractGraphEdge.from_dict(item) for item in raw_edges),
            mandatory_edge_ids=tuple(
                value.get("mandatory_edge_ids") or ()
            ),
            graph_root_claim=str(
                value.get("graph_root") or value.get("graph_id") or ""
            ),
            identity_claim=(
                GraphContentIdentity.from_dict(value["identity"])
                if isinstance(value.get("identity"), Mapping)
                else None
            ),
        )
        if "node_count" in value and int(value["node_count"]) != len(graph.nodes):
            raise SymbolicContractGraphError("graph node_count mismatch")
        if "edge_count" in value and int(value["edge_count"]) != len(graph.edges):
            raise SymbolicContractGraphError("graph edge_count mismatch")
        if "complete" in value and bool(value["complete"]) != graph.complete:
            raise SymbolicContractGraphError("graph completeness claim mismatch")
        return graph

    @classmethod
    def from_json(cls, value: str | bytes) -> "SymbolicContractGraph":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SymbolicContractGraphError("graph JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise SymbolicContractGraphError("graph JSON must contain an object")
        return cls.from_dict(payload)

    def closure(
        self,
        seed_node_ids: str | Sequence[str],
        *,
        direction: ClosureDirection | str = ClosureDirection.FORWARD,
        bounds: ClosureBounds | Mapping[str, Any] | None = None,
        edge_kinds: Iterable[ContractEdgeKind | str] | None = None,
        mandatory_only: bool = True,
        required_edge_ids: Sequence[str] = (),
        fail_closed: bool = True,
    ) -> ContractGraphClosure:
        """Compute deterministic exact forward or reverse typed-edge closure.

        Mandatory closure never returns a partial result as proof-capable.
        With ``fail_closed=True`` (the default), truncation or missing declared
        dependencies raises :class:`IncompleteMandatoryClosureError` carrying
        the explicit incomplete receipt.
        """

        orientation = _enum(
            direction, ClosureDirection, "closure direction"
        )
        limits = ClosureBounds.from_value(bounds)
        seeds = (
            (seed_node_ids,)
            if isinstance(seed_node_ids, str)
            else tuple(seed_node_ids)
        )
        seed_ids = _strings(seeds, "closure seed_node_ids")
        if not seed_ids:
            raise SymbolicContractGraphError(
                "closure requires at least one seed node"
            )
        node_map = {node.node_id: node for node in self.nodes}
        unknown = tuple(item for item in seed_ids if item not in node_map)
        if unknown:
            raise SymbolicContractGraphError(
                "closure contains unknown seeds: " + ", ".join(unknown)
            )
        kinds = (
            frozenset(
                _enum(item, ContractEdgeKind, "closure edge kind")
                for item in edge_kinds
            )
            if edge_kinds is not None
            else None
        )
        missing_edges = tuple(
            sorted(
                set(self.missing_mandatory_edge_ids).union(
                    edge_id
                    for edge_id in required_edge_ids
                    if edge_id not in {edge.edge_id for edge in self.edges}
                )
            )
        )
        missing_dependencies = self.missing_dependency_keys
        incomplete_reason = (
            "missing_mandatory_edges"
            if missing_edges
            else "missing_mandatory_dependencies"
            if missing_dependencies
            else ""
        )

        adjacency: dict[str, list[tuple[str, ContractGraphEdge]]] = {}
        for edge in self.edges:
            if mandatory_only and not edge.mandatory:
                continue
            if kinds is not None and edge.kind not in kinds:
                continue
            if mandatory_only and not edge.authoritative:
                continue
            if orientation is ClosureDirection.FORWARD:
                origin, neighbor = edge.source, edge.target
            else:
                origin, neighbor = edge.target, edge.source
            adjacency.setdefault(origin, []).append((neighbor, edge))
        for values in adjacency.values():
            values.sort(
                key=lambda item: (
                    item[1].kind.value,
                    item[0],
                    item[1].edge_id,
                )
            )

        paths: dict[str, tuple[str, ...]] = {
            seed: (seed,) for seed in seed_ids
        }
        depths = {seed: 0 for seed in seed_ids}
        included_edges: set[str] = set()
        queue: deque[str] = deque(seed_ids)
        truncated = False
        reason = incomplete_reason
        while queue and not truncated:
            current = queue.popleft()
            for neighbor, edge in adjacency.get(current, ()):
                depth = depths[current] + 1
                if depth > limits.max_depth:
                    truncated = True
                    reason = "max_depth_exceeded"
                    break
                included_edges.add(edge.edge_id)
                if len(included_edges) > limits.max_edges:
                    included_edges.remove(edge.edge_id)
                    truncated = True
                    reason = "max_edges_exceeded"
                    break
                candidate = (*paths[current], neighbor)
                previous = paths.get(neighbor)
                if previous is None:
                    if len(paths) >= limits.max_nodes:
                        truncated = True
                        reason = "max_nodes_exceeded"
                        break
                    paths[neighbor] = candidate
                    depths[neighbor] = depth
                    queue.append(neighbor)
                elif (len(candidate), candidate) < (len(previous), previous):
                    paths[neighbor] = candidate
                    depths[neighbor] = depth

        complete = not (
            truncated or missing_edges or missing_dependencies
        )
        receipt = ContractGraphClosure(
            graph_root=self.graph_root,
            snapshot_id=self.snapshot_id,
            version=self.version,
            direction=orientation,
            seed_node_ids=seed_ids,
            node_ids=tuple(paths),
            edge_ids=tuple(included_edges),
            paths=paths,
            bounds=limits,
            complete=complete,
            truncated=truncated,
            missing_edge_ids=missing_edges,
            missing_dependency_keys=missing_dependencies,
            reason_code=reason or "complete",
        )
        if fail_closed and not receipt.complete:
            raise IncompleteMandatoryClosureError(
                f"mandatory closure is incomplete: {receipt.reason_code}",
                receipt,
            )
        return receipt

    def forward_closure(
        self, seed_node_ids: str | Sequence[str], **kwargs: Any
    ) -> ContractGraphClosure:
        return self.closure(
            seed_node_ids, direction=ClosureDirection.FORWARD, **kwargs
        )

    def reverse_closure(
        self, seed_node_ids: str | Sequence[str], **kwargs: Any
    ) -> ContractGraphClosure:
        return self.closure(
            seed_node_ids, direction=ClosureDirection.REVERSE, **kwargs
        )

    mandatory_closure = forward_closure


@dataclass(frozen=True, slots=True)
class RetrievalBounds:
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_bytes: int = DEFAULT_MAX_RETRIEVAL_BYTES
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES

    def __post_init__(self) -> None:
        values = {
            "max_candidates": (
                self.max_candidates,
                1,
                HARD_MAX_CANDIDATES,
            ),
            "max_bytes": (
                self.max_bytes,
                1_024,
                HARD_MAX_RETRIEVAL_BYTES,
            ),
            "max_query_bytes": (
                self.max_query_bytes,
                1,
                HARD_MAX_QUERY_BYTES,
            ),
        }
        for name, (value, minimum, maximum) in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise SymbolicGraphBoundsError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise SymbolicGraphBoundsError(
                    f"{name} must be between {minimum} and {maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_bytes": self.max_bytes,
            "max_query_bytes": self.max_query_bytes,
        }

    @classmethod
    def from_value(
        cls, value: "RetrievalBounds | Mapping[str, Any] | None"
    ) -> "RetrievalBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise SymbolicGraphBoundsError("retrieval bounds must be an object")
        return cls(
            max_candidates=int(
                value.get("max_candidates", DEFAULT_MAX_CANDIDATES)
            ),
            max_bytes=int(
                value.get("max_bytes", DEFAULT_MAX_RETRIEVAL_BYTES)
            ),
            max_query_bytes=int(
                value.get("max_query_bytes", DEFAULT_MAX_QUERY_BYTES)
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphRAGCandidate:
    node_id: str
    stable_key: str
    kind: ContractNodeKind
    score: int
    rank: int
    nominated_by: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("node_id", "stable_key"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"candidate {name}")
            )
        object.__setattr__(
            self, "kind", _enum(self.kind, ContractNodeKind, "candidate kind")
        )
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, int)
            or self.score < 0
        ):
            raise CandidateRetrievalError(
                "candidate score must be a non-negative integer"
            )
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise CandidateRetrievalError(
                "candidate rank must be a positive integer"
            )
        object.__setattr__(
            self,
            "nominated_by",
            _strings(self.nominated_by, "candidate nominated_by"),
        )
        if not self.nominated_by:
            raise CandidateRetrievalError(
                "candidate must declare at least one nomination source"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "stable_key": self.stable_key,
            "kind": self.kind.value,
            "score": self.score,
            "rank": self.rank,
            "nominated_by": list(self.nominated_by),
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "provenance": ContractProvenance.GRAPHRAG.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphRAGCandidate":
        if value.get("authority") not in (
            None,
            ContractAuthority.CONTEXT_ONLY.value,
        ):
            raise CandidateRetrievalError(
                "candidate authority must remain context_only"
            )
        if value.get("provenance") not in (
            None,
            ContractProvenance.GRAPHRAG.value,
        ):
            raise CandidateRetrievalError(
                "candidate provenance must remain graphrag"
            )
        return cls(
            node_id=str(value.get("node_id") or ""),
            stable_key=str(value.get("stable_key") or ""),
            kind=value.get("kind", ""),
            score=value.get("score", -1),
            rank=value.get("rank", 0),
            nominated_by=tuple(value.get("nominated_by") or ()),
        )


@dataclass(frozen=True, slots=True)
class GraphRAGRetrievalReceipt:
    graph_root: str
    snapshot_id: str
    graph_version: str
    query: str
    bounds: RetrievalBounds
    candidates: tuple[GraphRAGCandidate, ...]
    total_matches: int
    truncated: bool
    provider_requested: bool
    provider_loaded: bool
    provider_status: str
    provider_receipt_id: str = ""
    reason_code: str = "bounded_candidates"

    def __post_init__(self) -> None:
        for name in (
            "graph_root",
            "snapshot_id",
            "graph_version",
            "query",
            "provider_status",
            "provider_receipt_id",
            "reason_code",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    f"retrieval {name}",
                    required=name
                    not in {"query", "provider_receipt_id"},
                    max_bytes=HARD_MAX_QUERY_BYTES
                    if name == "query"
                    else 8_192,
                ),
            )
        object.__setattr__(
            self, "bounds", RetrievalBounds.from_value(self.bounds)
        )
        for name in (
            "truncated",
            "provider_requested",
            "provider_loaded",
        ):
            if not isinstance(getattr(self, name), bool):
                raise CandidateRetrievalError(
                    f"retrieval {name} must be a boolean"
                )
        if (
            isinstance(self.total_matches, bool)
            or not isinstance(self.total_matches, int)
            or self.total_matches < 0
        ):
            raise CandidateRetrievalError(
                "total_matches must be a non-negative integer"
            )
        if len(self.candidates) > self.bounds.max_candidates:
            raise CandidateRetrievalError(
                "candidate receipt exceeds max_candidates"
            )
        if len({item.node_id for item in self.candidates}) != len(
            self.candidates
        ):
            raise CandidateRetrievalError(
                "candidate receipt contains duplicate nodes"
            )
        if tuple(item.rank for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise CandidateRetrievalError(
                "candidate ranks must be consecutive and start at one"
            )
        if self.total_matches < len(self.candidates):
            raise CandidateRetrievalError(
                "total_matches is less than returned candidates"
            )
        if not self.truncated and self.total_matches > len(self.candidates):
            raise CandidateRetrievalError(
                "omitted candidates require explicit truncation"
            )

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def safe_for_proof(self) -> bool:
        return False

    @property
    def receipt_id(self) -> str:
        return content_identity(self._payload())

    @property
    def identity(self) -> GraphContentIdentity:
        return GraphContentIdentity.for_value(self._payload())

    @property
    def byte_count(self) -> int:
        return len(canonical_contract_graph_bytes(self.to_dict()))

    @property
    def candidate_node_ids(self) -> tuple[str, ...]:
        return tuple(item.node_id for item in self.candidates)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": GRAPHRAG_RETRIEVAL_RECEIPT_SCHEMA,
            "interface": BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE,
            "version": "1",
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "graph_version": self.graph_version,
            "query": self.query,
            "bounds": self.bounds.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "returned_candidates": len(self.candidates),
            "total_matches": self.total_matches,
            "omitted_candidates": self.total_matches - len(self.candidates),
            "truncated": self.truncated,
            "provider_requested": self.provider_requested,
            "provider_loaded": self.provider_loaded,
            "provider_status": self.provider_status,
            "provider_receipt_id": self.provider_receipt_id,
            "reason_code": self.reason_code,
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "provenance": ContractProvenance.GRAPHRAG.value,
            "non_authoritative": True,
            "safe_for_proof": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "identity": self.identity.to_dict(),
            **self._payload(),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "GraphRAGRetrievalReceipt":
        if value.get("schema") not in (
            None,
            GRAPHRAG_RETRIEVAL_RECEIPT_SCHEMA,
        ):
            raise CandidateRetrievalError(
                "unsupported retrieval receipt schema"
            )
        if value.get("interface") not in (
            None,
            BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE,
        ):
            raise CandidateRetrievalError(
                "unsupported retrieval receipt interface"
            )
        raw_candidates = value.get("candidates") or ()
        if (
            isinstance(raw_candidates, (str, bytes))
            or not isinstance(raw_candidates, Sequence)
            or not all(isinstance(item, Mapping) for item in raw_candidates)
        ):
            raise CandidateRetrievalError(
                "receipt candidates must be a sequence of objects"
            )
        result = cls(
            graph_root=str(value.get("graph_root") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            graph_version=str(value.get("graph_version") or ""),
            query=str(value.get("query") or ""),
            bounds=RetrievalBounds.from_value(value.get("bounds")),
            candidates=tuple(
                GraphRAGCandidate.from_dict(item) for item in raw_candidates
            ),
            total_matches=value.get("total_matches", 0),
            truncated=value.get("truncated", False),
            provider_requested=value.get("provider_requested", False),
            provider_loaded=value.get("provider_loaded", False),
            provider_status=str(value.get("provider_status") or ""),
            provider_receipt_id=str(
                value.get("provider_receipt_id") or ""
            ),
            reason_code=str(value.get("reason_code") or ""),
        )
        fixed_claims = {
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "provenance": ContractProvenance.GRAPHRAG.value,
            "non_authoritative": True,
            "safe_for_proof": False,
            "returned_candidates": len(result.candidates),
            "omitted_candidates": (
                result.total_matches - len(result.candidates)
            ),
        }
        for name, expected in fixed_claims.items():
            if name in value and value[name] != expected:
                raise CandidateRetrievalError(
                    f"retrieval receipt {name} claim mismatch"
                )
        claimed = str(value.get("receipt_id") or "")
        if claimed and claimed != result.receipt_id:
            raise CandidateRetrievalError("retrieval receipt identity mismatch")
        if isinstance(value.get("identity"), Mapping):
            identity = GraphContentIdentity.from_dict(value["identity"])
            if identity != result.identity:
                raise CandidateRetrievalError(
                    "retrieval receipt ContentIdentity mismatch"
                )
        if result.byte_count > result.bounds.max_bytes:
            raise CandidateRetrievalError(
                "retrieval receipt exceeds its declared max_bytes"
            )
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes
    ) -> "GraphRAGRetrievalReceipt":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CandidateRetrievalError(
                "retrieval receipt JSON is malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise CandidateRetrievalError(
                "retrieval receipt JSON must contain an object"
            )
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class BoundedGraphRAGView:
    retrieval: GraphRAGRetrievalReceipt
    mandatory_closure: ContractGraphClosure

    @property
    def view_id(self) -> str:
        return content_identity(self._payload())

    @property
    def graph_root(self) -> str:
        return self.retrieval.graph_root

    @property
    def safe_for_proof(self) -> bool:
        # The closure may be used by a proof compiler, but the ranked view
        # itself remains context and can never constitute proof.
        return False

    @property
    def mandatory_closure_complete(self) -> bool:
        return self.mandatory_closure.complete

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": BOUNDED_GRAPHRAG_VIEW_SCHEMA,
            "version": "1",
            "graph_root": self.graph_root,
            "retrieval_receipt_id": self.retrieval.receipt_id,
            "mandatory_closure_id": self.mandatory_closure.closure_id,
            "candidate_node_ids": list(self.retrieval.candidate_node_ids),
            "mandatory_node_ids": list(self.mandatory_closure.node_ids),
            "mandatory_edge_ids": list(self.mandatory_closure.edge_ids),
            "retrieval_truncated": self.retrieval.truncated,
            "mandatory_closure_complete": self.mandatory_closure.complete,
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "non_authoritative": True,
            "safe_for_proof": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"view_id": self.view_id, **self._payload()}


def _tokens(value: str) -> tuple[str, ...]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return tuple(item.casefold() for item in _WORD_RE.findall(expanded))


def _node_search_text(node: ContractGraphNode) -> str:
    return " ".join(
        (
            node.kind.value,
            node.stable_key,
            canonical_contract_graph_json(dict(node.payload)),
            " ".join(node.source_refs),
        )
    )


def _local_score(query: str, node: ContractGraphNode) -> int:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 1
    text = _node_search_text(node).casefold()
    tokens = set(_tokens(text))
    score = 0
    for token in query_tokens:
        if token in tokens:
            score += 100
        elif token in text:
            score += 25
    if query.casefold() in text:
        score += 250
    return score


class BoundedGraphRAGRetriever:
    """Candidate nominator with a lazy optional datasets provider.

    Local lexical scoring is always available and never claims exact datasets
    use.  Exact ``IntentGraphRetriever`` / Cypher-AST binding is a separate
    capability gate: package-root fallback, fixture-only backends, and the
    local lexical path cannot satisfy it.
    """

    def __init__(
        self,
        graph: SymbolicContractGraph,
        *,
        provider: Any = None,
        provider_factory: Callable[[], Any] | None = None,
        exact_datasets_importer: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(graph, SymbolicContractGraph):
            raise TypeError("graph must be a SymbolicContractGraph")
        self.graph = graph
        self._provider = provider
        self._provider_factory = provider_factory
        self._provider_loaded = provider is not None
        self._exact_datasets_importer = exact_datasets_importer
        self._exact_adapter: Any = None
        self._exact_adapter_loaded = False
        self._exact_capability: Mapping[str, Any] | None = None

    @property
    def provider_loaded(self) -> bool:
        return self._provider_loaded

    @property
    def exact_datasets_loaded(self) -> bool:
        return self._exact_adapter_loaded

    def capability(self) -> dict[str, Any]:
        """Return local capability metadata without loading optional code."""

        return {
            "interface": BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE,
            "version": "1",
            "evidence_id": SCAEV031DATASETSGRAPH,
            "evidence": {
                "requirement_ids": [SCAEV031DATASETSGRAPH],
                "evidence_id": SCAEV031DATASETSGRAPH_EVIDENCE,
            },
            "operations": [
                "local_candidate_retrieval",
                "graph_retrieval",
                "exact_datasets_graph_retrieval",
            ],
            "provider_loaded": self.provider_loaded,
            "exact_datasets_loaded": self.exact_datasets_loaded,
            "optional_provider_lazy": True,
            "exact_datasets_lazy": True,
            "package_root_fallback_accepted": False,
            "fixture_only_accepted": False,
            "local_lexical_claims_exact_datasets": False,
            "exact_modules": {
                "graphrag": EXACT_DATASETS_GRAPHRAG_MODULE,
                "cypher_ast": EXACT_DATASETS_CYPHER_AST_MODULE,
                "cypher_parser": EXACT_DATASETS_CYPHER_PARSER_MODULE,
            },
            "authority": ContractAuthority.CONTEXT_ONLY.value,
            "non_authoritative": True,
            "proof_authority": False,
        }

    capabilities = capability

    def _analysis_provider_module(self) -> Any:
        return importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.integrations."
            "ipfs_datasets_analysis_provider"
        )

    def exact_datasets_capability(
        self, *, probe: bool = True
    ) -> dict[str, Any]:
        """Return exact GraphRAG/Cypher capability receipts (lazy optional)."""

        if self._exact_capability is not None and not probe:
            return dict(self._exact_capability)
        module = self._analysis_provider_module()
        capability = module.inspect_exact_datasets_graph_capability(
            importer=self._exact_datasets_importer
        )
        self._exact_capability = capability
        return dict(capability)

    def _load_provider(self) -> Any:
        if self._provider_loaded:
            return self._provider
        if self._provider_factory is not None:
            provider = self._provider_factory()
        else:
            module = self._analysis_provider_module()
            # Default optional path remains the lazy analysis provider.  Exact
            # IntentGraphRetriever binding is explicit via use_exact_datasets.
            provider = module.IpfsDatasetsAnalysisProvider()
        self._provider = provider
        self._provider_loaded = True
        return provider

    def _load_exact_datasets_adapter(self) -> Any:
        if self._exact_adapter_loaded:
            if self._exact_adapter is None:
                raise ExactDatasetsGraphProviderError(
                    "exact datasets GraphRAG adapter previously failed to load",
                    reason_code="exact_datasets_unavailable",
                    details=dict(self._exact_capability or {}),
                )
            return self._exact_adapter
        module = self._analysis_provider_module()
        capability = module.inspect_exact_datasets_graph_capability(
            importer=self._exact_datasets_importer
        )
        self._exact_capability = capability
        if not capability.get("available"):
            self._exact_adapter_loaded = True
            self._exact_adapter = None
            raise ExactDatasetsGraphProviderError(
                "exact datasets GraphRAG/Cypher modules unavailable or incompatible",
                reason_code="exact_datasets_unavailable",
                details=capability,
            )
        try:
            adapter = module.create_exact_datasets_graphrag_adapter(
                importer=self._exact_datasets_importer
            )
        except module.DatasetsGraphBackendError as exc:
            self._exact_adapter_loaded = True
            self._exact_adapter = None
            raise ExactDatasetsGraphProviderError(
                str(exc),
                reason_code=getattr(exc, "reason_code", "exact_datasets_unavailable"),
                details=getattr(exc, "details", {}) or capability,
            ) from exc
        self._exact_adapter = adapter
        self._exact_adapter_loaded = True
        return adapter

    def _map_provider_references(
        self, references: Sequence[Any]
    ) -> set[str]:
        by_key = {node.stable_key: node.node_id for node in self.graph.nodes}
        node_ids = {node.node_id for node in self.graph.nodes}
        by_path: dict[str, set[str]] = {}
        by_symbol: dict[str, set[str]] = {}
        for node in self.graph.nodes:
            path = str(node.payload.get("path") or "")
            symbol = str(node.payload.get("symbol") or "")
            if path:
                by_path.setdefault(path, set()).add(node.node_id)
            if symbol:
                by_symbol.setdefault(symbol, set()).add(node.node_id)
        nominated: set[str] = set()
        for raw in references:
            if not isinstance(raw, Mapping):
                continue
            for field_name in ("node_id", "evidence_id", "artifact_id"):
                candidate = str(raw.get(field_name) or "")
                if candidate in node_ids:
                    nominated.add(candidate)
            key = str(raw.get("stable_key") or "")
            if key in by_key:
                nominated.add(by_key[key])
            nominated.update(by_path.get(str(raw.get("path") or ""), ()))
            nominated.update(by_symbol.get(str(raw.get("symbol") or ""), ()))
        return nominated

    def _provider_candidates(
        self,
        query: str,
        bounds: RetrievalBounds,
    ) -> tuple[set[str], bool, str, str]:
        provider = self._load_provider()
        request = {
            "operation": "graph_retrieval",
            "repository_id": self.graph.graph_root,
            "tree_id": self.graph.snapshot_id,
            "objective_revision": f"{SYMBOLIC_CONTRACT_GRAPH_INTERFACE}:"
            f"{self.graph.version}",
            "query": {"text": query},
            "payload": {"graph_root": self.graph.graph_root},
            "bounds": {
                "max_results": bounds.max_candidates,
                "max_batch_requests": 1,
                "max_query_bytes": bounds.max_query_bytes,
                "max_request_bytes": max(4_096, bounds.max_bytes * 2),
                "max_response_bytes": bounds.max_bytes,
                "max_reference_bytes": min(bounds.max_bytes, 262_144),
                "timeout_ms": 30_000,
            },
        }
        result = provider.analyze(request)
        if isinstance(result, Mapping):
            references = result.get("evidence_references") or result.get(
                "results"
            ) or ()
            truncated = bool(result.get("truncated", False))
            status = str(result.get("status") or "unknown")
            receipt_id = str(
                result.get("result_id") or result.get("receipt_id") or ""
            )
        else:
            references = getattr(result, "evidence_references", ())
            truncated = bool(getattr(result, "truncated", False))
            raw_status = getattr(result, "status", "unknown")
            status = str(getattr(raw_status, "value", raw_status))
            receipt_id = str(
                getattr(result, "result_id", "")
                or getattr(result, "receipt_id", "")
            )
        nominated = self._map_provider_references(
            references if isinstance(references, Sequence) else ()
        )
        # Optional package-root style provider never claims exact datasets use.
        # Keep the backend status string intact for existing receipts; exact
        # mode uses a separate provider_status vocabulary.
        return nominated, truncated, status, receipt_id

    def _exact_datasets_candidates(
        self,
        query: str,
        bounds: RetrievalBounds,
    ) -> tuple[set[str], bool, str, str, Mapping[str, Any]]:
        adapter = self._load_exact_datasets_adapter()
        result = adapter.retrieve_candidates(
            query=query,
            graph_root=self.graph.graph_root,
            snapshot_id=self.graph.snapshot_id,
            bounds={
                "max_results": bounds.max_candidates,
                "max_bytes": bounds.max_bytes,
                "timeout_ms": 30_000,
            },
        )
        if not isinstance(result, Mapping):
            raise ExactDatasetsGraphProviderError(
                "exact datasets adapter returned a non-object result",
                reason_code="exact_datasets_malformed_result",
            )
        if result.get("exact_module") is not True:
            raise ExactDatasetsGraphProviderError(
                "exact datasets adapter did not claim exact module use",
                reason_code="exact_module_claim_missing",
                details=dict(result),
            )
        if result.get("fixture_only") is True or result.get(
            "package_root_fallback"
        ):
            raise ExactDatasetsGraphProviderError(
                "fixture-only or package-root results cannot pass the exact gate",
                reason_code="exact_source_rejected",
                details=dict(result),
            )
        if result.get("non_authoritative") is not True or result.get(
            "proof_authority"
        ):
            raise ExactDatasetsGraphProviderError(
                "exact GraphRAG results must remain non-authoritative",
                reason_code="graphrag_authoritative_claim",
                details=dict(result),
            )
        references = result.get("evidence_references") or result.get("results") or ()
        nominated = self._map_provider_references(
            references if isinstance(references, Sequence) else ()
        )
        truncated = bool(result.get("truncated", False))
        receipt_id = str(
            result.get("receipt_id")
            or result.get("canary_receipt_id")
            or result.get("result_id")
            or ""
        )
        capability = result.get("capability") if isinstance(
            result.get("capability"), Mapping
        ) else {}
        return (
            nominated,
            truncated,
            "exact_datasets:completed",
            receipt_id,
            {
                "capability": dict(capability),
                "capability_revision": result.get("capability_revision", ""),
                "package_version": result.get("package_version", ""),
                "package_tree": result.get("package_tree", ""),
                "module_paths": list(result.get("module_paths") or ()),
                "canary_receipt_id": result.get("canary_receipt_id", ""),
                "graph_root": result.get("graph_root", self.graph.graph_root),
                "bounds": dict(result.get("bounds") or {}),
                "non_authoritative": True,
                "proof_authority": False,
                "exact_module": True,
            },
        )

    def retrieve(
        self,
        query: str,
        *,
        bounds: RetrievalBounds | Mapping[str, Any] | None = None,
        use_optional_provider: bool = False,
        use_exact_datasets: bool = False,
        require_exact_datasets: bool = False,
    ) -> GraphRAGRetrievalReceipt:
        if require_exact_datasets:
            use_exact_datasets = True
        if use_exact_datasets and use_optional_provider:
            # Exact IntentGraphRetriever binding supersedes package-root optional.
            use_optional_provider = False

        limits = RetrievalBounds.from_value(bounds)
        normalized_query = " ".join(
            _text(
                query,
                "retrieval query",
                required=False,
                max_bytes=limits.max_query_bytes,
            ).split()
        )
        local_scores = {
            node.node_id: _local_score(normalized_query, node)
            for node in self.graph.nodes
        }
        local_scores = {
            node_id: score
            for node_id, score in local_scores.items()
            if score > 0
        }
        provider_ids: set[str] = set()
        provider_truncated = False
        provider_status = "not_requested"
        provider_receipt_id = ""
        exact_meta: Mapping[str, Any] = {}
        provider_requested = use_optional_provider or use_exact_datasets

        if use_exact_datasets:
            try:
                (
                    provider_ids,
                    provider_truncated,
                    provider_status,
                    provider_receipt_id,
                    exact_meta,
                ) = self._exact_datasets_candidates(normalized_query, limits)
            except ExactDatasetsGraphProviderError as exc:
                if require_exact_datasets:
                    raise
                # Soft exact request degrades without claiming exact success.
                provider_status = f"exact_blocked:{exc.reason_code}"
                provider_receipt_id = content_identity(
                    {
                        "reason_code": exc.reason_code,
                        "details": dict(exc.details),
                    }
                )
            except Exception as exc:
                if require_exact_datasets:
                    raise ExactDatasetsGraphProviderError(
                        f"exact datasets retrieval failed: {exc}",
                        reason_code="exact_datasets_failed",
                        details={"error_type": type(exc).__name__},
                    ) from exc
                provider_status = "exact_degraded:" + type(exc).__name__
        elif use_optional_provider:
            try:
                (
                    provider_ids,
                    provider_truncated,
                    provider_status,
                    provider_receipt_id,
                ) = self._provider_candidates(normalized_query, limits)
            except Exception as exc:
                # Optional retrieval failure is typed context degradation, not
                # a reason to lose deterministic local candidate retrieval.
                provider_status = "degraded:" + type(exc).__name__

        ranked: list[tuple[int, str, set[str]]] = []
        for node in self.graph.nodes:
            score = local_scores.get(node.node_id, 0)
            sources: set[str] = set()
            if score:
                sources.add("local")
            if node.node_id in provider_ids:
                score += 50
                if use_exact_datasets and provider_status.startswith(
                    "exact_datasets:"
                ):
                    sources.add("ipfs_datasets_exact")
                else:
                    sources.add("ipfs_datasets")
            if score:
                ranked.append((score, node.node_id, sources))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        # Local lexical hits never claim exact datasets use even when an exact
        # request was made; only provider-nominated nodes carry that source.
        if use_exact_datasets and not provider_status.startswith(
            "exact_datasets:"
        ):
            # No successful exact nomination: strip any accidental exact labels.
            ranked = [
                (
                    score,
                    node_id,
                    frozenset(
                        source
                        for source in sources
                        if source != "ipfs_datasets_exact"
                    ),
                )
                for score, node_id, sources in ranked
            ]
            ranked = [
                (score, node_id, set(sources))
                for score, node_id, sources in ranked
                if sources
            ]
            ranked.sort(key=lambda item: (-item[0], item[1]))

        node_map = {node.node_id: node for node in self.graph.nodes}
        total_matches = len(ranked)
        accepted: list[GraphRAGCandidate] = []

        def make_receipt(
            values: Sequence[GraphRAGCandidate], truncated: bool
        ) -> GraphRAGRetrievalReceipt:
            reason = "bounded_candidates"
            if truncated:
                reason = "bounded_candidates_truncated"
            if use_exact_datasets and provider_status.startswith(
                "exact_datasets:"
            ):
                reason = (
                    "exact_datasets_bounded_candidates_truncated"
                    if truncated
                    else "exact_datasets_bounded_candidates"
                )
            elif use_exact_datasets and provider_status.startswith(
                "exact_blocked:"
            ):
                reason = "exact_datasets_blocked_local_fallback"
            return GraphRAGRetrievalReceipt(
                graph_root=self.graph.graph_root,
                snapshot_id=self.graph.snapshot_id,
                graph_version=self.graph.version,
                query=normalized_query,
                bounds=limits,
                candidates=tuple(values),
                total_matches=total_matches,
                truncated=truncated,
                provider_requested=provider_requested,
                provider_loaded=(
                    self.provider_loaded or self.exact_datasets_loaded
                ),
                provider_status=provider_status,
                provider_receipt_id=provider_receipt_id,
                reason_code=reason,
            )

        count_limited = total_matches > limits.max_candidates
        for score, node_id, sources in ranked[: limits.max_candidates]:
            node = node_map[node_id]
            # Local lexical never masquerades as exact datasets nomination.
            nominated_by = tuple(sorted(sources))
            if "local" in nominated_by and "ipfs_datasets_exact" not in nominated_by:
                nominated_by = tuple(
                    item
                    for item in nominated_by
                    if item != "ipfs_datasets_exact"
                )
            candidate = GraphRAGCandidate(
                node_id=node.node_id,
                stable_key=node.stable_key,
                kind=node.kind,
                score=score,
                rank=len(accepted) + 1,
                nominated_by=nominated_by,
            )
            trial = make_receipt(
                (*accepted, candidate),
                count_limited
                or provider_truncated
                or len(accepted) + 1 < total_matches,
            )
            if trial.byte_count > limits.max_bytes:
                break
            accepted.append(candidate)

        truncated = (
            count_limited
            or provider_truncated
            or len(accepted) < total_matches
        )
        receipt = make_receipt(accepted, truncated)
        while accepted and receipt.byte_count > limits.max_bytes:
            accepted.pop()
            truncated = True
            receipt = make_receipt(accepted, truncated)
        if receipt.byte_count > limits.max_bytes:
            raise CandidateRetrievalError(
                "max_bytes is too small for retrieval receipt metadata"
            )
        # Exact metadata is retained on the provider receipt id / status only;
        # GraphRAG remains context-only and never gains proof authority.
        _ = exact_meta
        return receipt

    def view(
        self,
        query: str,
        *,
        retrieval_bounds: RetrievalBounds | Mapping[str, Any] | None = None,
        closure_bounds: ClosureBounds | Mapping[str, Any] | None = None,
        use_optional_provider: bool = False,
        use_exact_datasets: bool = False,
        require_exact_datasets: bool = False,
    ) -> BoundedGraphRAGView:
        retrieval = self.retrieve(
            query,
            bounds=retrieval_bounds,
            use_optional_provider=use_optional_provider,
            use_exact_datasets=use_exact_datasets,
            require_exact_datasets=require_exact_datasets,
        )
        if not retrieval.candidate_node_ids:
            closure = ContractGraphClosure(
                graph_root=self.graph.graph_root,
                snapshot_id=self.graph.snapshot_id,
                version=self.graph.version,
                direction=ClosureDirection.FORWARD,
                seed_node_ids=(),
                node_ids=(),
                edge_ids=(),
                paths={},
                bounds=ClosureBounds.from_value(closure_bounds),
                complete=self.graph.complete,
                missing_edge_ids=self.graph.missing_mandatory_edge_ids,
                missing_dependency_keys=self.graph.missing_dependency_keys,
                reason_code=(
                    "complete_empty_candidates"
                    if self.graph.complete
                    else "missing_mandatory_dependencies"
                ),
            )
        else:
            closure = self.graph.forward_closure(
                retrieval.candidate_node_ids,
                bounds=closure_bounds,
            )
        return BoundedGraphRAGView(
            retrieval=retrieval,
            mandatory_closure=closure,
        )


def _value(value: Any, *names: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _as_records(value: Any) -> tuple[Any, ...]:
    records = _value(value, "records", "path_records", default=())
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        return tuple(records)
    return ()


def _index_records(value: Any, *names: str) -> tuple[Any, ...]:
    """Read one of several compact index collection spellings."""

    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(value)
    for name in names:
        records = _value(value, name, default=None)
        if isinstance(records, Sequence) and not isinstance(
            records, (str, bytes, bytearray)
        ):
            return tuple(records)
    return ()


def _compact_schema_payload(record: Any) -> dict[str, Any]:
    """Retain schema identity and location without copying schema bodies."""

    fields = (
        "schema_id",
        "record_id",
        "name",
        "subject",
        "path",
        "kind",
        "language",
        "schema_version",
        "source_version",
        "content_digest",
        "blob_identity",
        "source_sha256",
    )
    return {
        name: _plain(_value(record, name, default=""))
        for name in fields
        if _value(record, name, default="") not in (None, "")
    }


def _code_evidence_provenance(value: Any) -> ContractProvenance:
    raw = str(getattr(value, "value", value) or "").casefold()
    if raw == "ast":
        return ContractProvenance.AST
    if raw == "validation":
        return ContractProvenance.TEST
    if raw == "task":
        return ContractProvenance.CONTRACT
    if raw == "proof":
        return ContractProvenance.CONTRACT
    if raw == "merge":
        return ContractProvenance.REPOSITORY
    if raw in {"enrichment", "llm", "graphrag"}:
        return ContractProvenance.GRAPHRAG
    return ContractProvenance.REPOSITORY


def _code_evidence_node_kind(value: Any) -> ContractNodeKind:
    raw = str(getattr(value, "value", value) or "").casefold()
    return {
        "tree": ContractNodeKind.REPOSITORY_SNAPSHOT,
        "symbol": ContractNodeKind.SYMBOL,
        "ast_scope": ContractNodeKind.MODULE,
        "validation": ContractNodeKind.TEST,
        "obligation": ContractNodeKind.CONTRACT,
        "proof": ContractNodeKind.CONTRACT,
        "enrichment": ContractNodeKind.PROVENANCE,
    }.get(raw, ContractNodeKind.PROVENANCE)


def _module_for_path(path: str) -> str:
    pure = PurePosixPath(path)
    suffixes = {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
    }
    name = pure.as_posix()
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = list(PurePosixPath(name).parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class _ProjectionBuilder:
    def __init__(self, snapshot_id: str, version: str) -> None:
        self.snapshot_id = snapshot_id
        self.version = version
        self.nodes: dict[str, ContractGraphNode] = {}
        self.edges: dict[str, ContractGraphEdge] = {}

    def node(
        self,
        kind: ContractNodeKind,
        stable_key: str,
        *,
        provenance: ContractProvenance = ContractProvenance.AST,
        authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
        required_dependencies: Sequence[str] = (),
    ) -> ContractGraphNode:
        result = ContractGraphNode(
            kind=kind,
            stable_key=stable_key,
            snapshot_id=self.snapshot_id,
            provenance=provenance,
            authority=authority,
            version=self.version,
            payload=payload or {},
            source_refs=tuple(source_refs),
            required_dependencies=tuple(required_dependencies),
        )
        old = self.nodes.get(stable_key)
        if old is not None and old.node_id != result.node_id:
            raise SymbolicContractGraphError(
                f"projection produced conflicting node {stable_key!r}"
            )
        self.nodes[stable_key] = result
        return result

    def edge(
        self,
        kind: ContractEdgeKind,
        source: ContractGraphNode,
        target: ContractGraphNode,
        *,
        provenance: ContractProvenance = ContractProvenance.AST,
        authority: ContractAuthority = ContractAuthority.SOURCE_OBSERVATION,
        mandatory: bool = True,
        payload: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
    ) -> ContractGraphEdge:
        result = ContractGraphEdge(
            kind=kind,
            source=source.node_id,
            target=target.node_id,
            snapshot_id=self.snapshot_id,
            provenance=provenance,
            authority=authority,
            version=self.version,
            mandatory=mandatory,
            payload=payload or {},
            source_refs=tuple(source_refs),
        )
        self.edges[result.edge_id] = result
        return result


def project_symbolic_contract_graph(
    repository_index: Any,
    *,
    schema_index: Any = None,
    code_evidence_graph: Any = None,
    nodes: Iterable[ContractGraphNode | Mapping[str, Any]] = (),
    edges: Iterable[ContractGraphEdge | Mapping[str, Any]] = (),
    version: str = GRAPH_VERSION,
) -> SymbolicContractGraph:
    """Project a RepositoryIndex/AnalysisASTIndex into typed graph facts.

    ``repository_index`` may be the complete SCA ``RepositoryIndex`` or a
    compact mapping/object exposing ``snapshot_id`` and an AST index.
    ``schema_index`` and ``code_evidence_graph`` are compact optional inputs;
    when omitted they are discovered on ``repository_index``.  Extra MCP,
    handler, policy, transport, contract, and test records can be supplied as
    already-typed nodes and edges without weakening validation.
    """

    snapshot_id = str(
        _value(
            repository_index,
            "snapshot_id",
            "tree_id",
            "repository_tree_id",
            default="",
        )
        or ""
    )
    if not snapshot_id:
        snapshot = _value(repository_index, "snapshot", default=None)
        snapshot_id = str(
            _value(snapshot, "snapshot_id", "tree_id", default="") or ""
        )
    snapshot_id = _text(snapshot_id, "repository snapshot_id")
    version = _text(version, "graph version")
    ast_index = _value(
        repository_index, "ast_index", "analysis_ast_index", default=None
    )
    if ast_index is None:
        ast_index = repository_index
    ast_index_id = str(
        _value(ast_index, "index_id", "ast_index_id", default="") or ""
    )

    builder = _ProjectionBuilder(snapshot_id, version)
    snapshot_node = builder.node(
        ContractNodeKind.REPOSITORY_SNAPSHOT,
        f"snapshot:{snapshot_id}",
        provenance=ContractProvenance.REPOSITORY,
        payload={"snapshot_id": snapshot_id, "ast_index_id": ast_index_id},
        source_refs=(ast_index_id,) if ast_index_id else (),
    )

    module_nodes: dict[str, ContractGraphNode] = {}
    symbol_nodes: dict[str, ContractGraphNode] = {}
    records = _as_records(ast_index)
    for indexed in records:
        path = str(_value(indexed, "path", default="") or "")
        if not path:
            raise SymbolicContractGraphError(
                "indexed AST record is missing a path"
            )
        ast_record = _value(indexed, "ast_record", default=indexed)
        record_id = str(
            _value(indexed, "record_id", default="")
            or _value(ast_record, "record_id", default="")
            or ""
        )
        blob_identity = str(
            _value(indexed, "blob_identity", default="")
            or _value(ast_record, "blob_identity", default="")
            or ""
        )
        source_refs = tuple(
            item for item in (record_id, blob_identity) if item
        )
        module_name = str(
            _value(indexed, "module", default="") or _module_for_path(path)
        )
        file_node = builder.node(
            ContractNodeKind.FILE,
            f"file:{path}",
            payload={
                "path": path,
                "record_id": record_id,
                "blob_identity": blob_identity,
                "language": str(
                    _value(ast_record, "language", default="") or ""
                ),
            },
            source_refs=source_refs,
            required_dependencies=(snapshot_node.stable_key,),
        )
        builder.edge(
            ContractEdgeKind.SOURCED_FROM,
            file_node,
            snapshot_node,
            provenance=ContractProvenance.REPOSITORY,
            source_refs=source_refs,
        )
        module_node = builder.node(
            ContractNodeKind.MODULE,
            f"module:{module_name}",
            payload={"module": module_name, "path": path},
            source_refs=source_refs,
            required_dependencies=(file_node.stable_key,),
        )
        module_nodes[module_name] = module_node
        builder.edge(
            ContractEdgeKind.DEFINED_IN,
            module_node,
            file_node,
            source_refs=source_refs,
        )

        qualified_symbols = tuple(
            _value(ast_record, "qualified_symbols", default=()) or ()
        )
        symbol_lines = _value(ast_record, "symbol_lines", default={}) or {}
        symbol_hashes = _value(ast_record, "symbol_hashes", default={}) or {}
        for symbol in sorted({str(item) for item in qualified_symbols}):
            qualified = (
                symbol
                if symbol.startswith(module_name + ".")
                else f"{module_name}.{symbol}"
                if module_name
                else symbol
            )
            span = (
                symbol_lines.get(symbol, (0, 0))
                if isinstance(symbol_lines, Mapping)
                else (0, 0)
            )
            node = builder.node(
                ContractNodeKind.SYMBOL,
                f"symbol:{qualified}",
                payload={
                    "symbol": symbol,
                    "qualified_symbol": qualified,
                    "path": path,
                    "line_start": int(span[0]) if len(span) > 0 else 0,
                    "line_end": int(span[1]) if len(span) > 1 else 0,
                    "symbol_hash": (
                        str(symbol_hashes.get(symbol) or "")
                        if isinstance(symbol_hashes, Mapping)
                        else ""
                    ),
                },
                source_refs=source_refs,
                required_dependencies=(module_node.stable_key,),
            )
            symbol_nodes[qualified] = node
            symbol_nodes.setdefault(symbol, node)
            builder.edge(
                ContractEdgeKind.DEFINED_IN,
                node,
                module_node,
                source_refs=source_refs,
            )

    # Calls/imports can target records defined later, so project them after all
    # module and symbol identities are known.
    for indexed in records:
        path = str(_value(indexed, "path", default="") or "")
        ast_record = _value(indexed, "ast_record", default=indexed)
        record_id = str(
            _value(indexed, "record_id", default="")
            or _value(ast_record, "record_id", default="")
            or ""
        )
        module_name = str(
            _value(indexed, "module", default="") or _module_for_path(path)
        )
        module_node = module_nodes[module_name]
        source_refs = (record_id,) if record_id else ()

        for imported in sorted(
            {str(item) for item in _value(ast_record, "imports", default=()) or ()}
        ):
            import_node = builder.node(
                ContractNodeKind.IMPORT,
                f"import:{module_name}:{imported}",
                payload={
                    "module": module_name,
                    "target": imported,
                    "path": path,
                },
                source_refs=source_refs,
            )
            builder.edge(
                ContractEdgeKind.IMPORTS,
                module_node,
                import_node,
                source_refs=source_refs,
            )
            target = module_nodes.get(imported)
            if target is None:
                target = builder.node(
                    ContractNodeKind.UNRESOLVED,
                    f"unresolved:module:{imported}",
                    payload={
                        "target": imported,
                        "reason_code": "external_or_dynamic_import",
                    },
                    source_refs=source_refs,
                )
            builder.edge(
                ContractEdgeKind.DEPENDS_ON,
                import_node,
                target,
                source_refs=source_refs,
            )

        for call in sorted(
            {str(item) for item in _value(ast_record, "calls", default=()) or ()}
        ):
            owner, separator, callee = call.partition("->")
            callee = callee if separator else call
            owner_node = (
                symbol_nodes.get(owner)
                or symbol_nodes.get(
                    f"{module_name}.{owner}" if module_name else owner
                )
                or module_node
            )
            call_node = builder.node(
                ContractNodeKind.CALL,
                f"call:{module_name}:{call}",
                payload={
                    "module": module_name,
                    "owner": owner if separator else "",
                    "callee": callee,
                    "path": path,
                },
                source_refs=source_refs,
            )
            builder.edge(
                ContractEdgeKind.CALLS,
                owner_node,
                call_node,
                source_refs=source_refs,
            )
            target = (
                symbol_nodes.get(callee)
                or symbol_nodes.get(
                    f"{module_name}.{callee}" if module_name else callee
                )
            )
            if target is None:
                target = builder.node(
                    ContractNodeKind.UNRESOLVED,
                    f"unresolved:symbol:{callee}",
                    payload={
                        "target": callee,
                        "reason_code": "external_or_dynamic_call",
                    },
                    source_refs=source_refs,
                )
            builder.edge(
                ContractEdgeKind.DEPENDS_ON,
                call_node,
                target,
                source_refs=source_refs,
            )

        for effect in sorted(
            {
                str(item)
                for item in _value(
                    ast_record, "state_transitions", default=()
                )
                or ()
            }
        ):
            effect_node = builder.node(
                ContractNodeKind.EFFECT,
                f"effect:{module_name}:{effect}",
                payload={
                    "module": module_name,
                    "effect": effect,
                    "path": path,
                },
                source_refs=source_refs,
            )
            builder.edge(
                ContractEdgeKind.HAS_EFFECT,
                module_node,
                effect_node,
                source_refs=source_refs,
            )

        for interface in sorted(
            {
                str(item)
                for item in _value(ast_record, "interfaces", default=()) or ()
            }
        ):
            interface_node = builder.node(
                ContractNodeKind.INTERFACE,
                f"interface:{module_name}:{interface}",
                payload={
                    "module": module_name,
                    "interface": interface,
                    "path": path,
                },
                source_refs=source_refs,
            )
            builder.edge(
                ContractEdgeKind.DECLARES,
                module_node,
                interface_node,
                source_refs=source_refs,
            )

    schema_source = (
        schema_index
        if schema_index is not None
        else _value(
            repository_index,
            "schema_index",
            "schemas",
            default=None,
        )
    )
    for schema_record in _index_records(
        schema_source, "records", "schemas", "entries", "items"
    ):
        compact = _compact_schema_payload(schema_record)
        schema_key = str(
            _value(
                schema_record,
                "schema_id",
                "record_id",
                "content_id",
                "name",
                "path",
                default="",
            )
            or ""
        )
        if not schema_key:
            raise SymbolicContractGraphError(
                "schema index record lacks a stable identity"
            )
        path = str(_value(schema_record, "path", default="") or "")
        file_node = builder.nodes.get(f"file:{path}") if path else None
        source_refs = tuple(
            item
            for item in (
                str(_value(schema_record, "record_id", default="") or ""),
                str(
                    _value(
                        schema_record,
                        "content_digest",
                        "blob_identity",
                        "source_sha256",
                        default="",
                    )
                    or ""
                ),
            )
            if item
        )
        schema_node = builder.node(
            ContractNodeKind.SCHEMA,
            f"schema:{schema_key}",
            provenance=ContractProvenance.SCHEMA,
            payload=compact,
            source_refs=source_refs,
            required_dependencies=(
                (file_node.stable_key,) if file_node is not None else ()
            ),
        )
        if file_node is not None:
            builder.edge(
                ContractEdgeKind.SOURCED_FROM,
                schema_node,
                file_node,
                provenance=ContractProvenance.SCHEMA,
                source_refs=source_refs,
            )

    evidence_source = (
        code_evidence_graph
        if code_evidence_graph is not None
        else _value(repository_index, "code_evidence_graph", default=None)
    )
    evidence_nodes = _index_records(evidence_source, "nodes")
    projected_evidence: dict[str, ContractGraphNode] = {}
    for evidence_node in evidence_nodes:
        evidence_id = str(
            _value(evidence_node, "node_id", default="") or ""
        )
        evidence_key = str(
            _value(evidence_node, "record_key", default="") or evidence_id
        )
        if not evidence_key:
            raise SymbolicContractGraphError(
                "code evidence node lacks a stable identity"
            )
        raw_provenance = _value(
            evidence_node, "provenance", default="repository"
        )
        provenance = _code_evidence_provenance(raw_provenance)
        authority = (
            ContractAuthority.CONTEXT_ONLY
            if provenance.context_only
            else ContractAuthority.SOURCE_OBSERVATION
        )
        raw_record = _value(evidence_node, "record", default={}) or {}
        payload = {
            "code_evidence_kind": str(
                getattr(
                    _value(evidence_node, "kind", default=""),
                    "value",
                    _value(evidence_node, "kind", default=""),
                )
            ),
            "record_key": evidence_key,
            "record": _plain(raw_record),
            **{
                name: str(_value(evidence_node, name, default="") or "")
                for name in (
                    "task_id",
                    "tree_id",
                    "symbol",
                    "obligation_id",
                    "assurance",
                    "freshness",
                )
                if _value(evidence_node, name, default="") not in (None, "")
            },
        }
        projected = builder.node(
            _code_evidence_node_kind(
                _value(evidence_node, "kind", default="")
            ),
            f"code-evidence:{evidence_key}",
            provenance=provenance,
            authority=authority,
            payload=payload,
            source_refs=(evidence_id,) if evidence_id else (),
        )
        projected_evidence[evidence_id or evidence_key] = projected

    for evidence_edge in _index_records(evidence_source, "edges"):
        source_id = str(_value(evidence_edge, "source", default="") or "")
        target_id = str(_value(evidence_edge, "target", default="") or "")
        source = projected_evidence.get(source_id)
        target = projected_evidence.get(target_id)
        if source is None or target is None:
            raise SymbolicContractGraphError(
                "code evidence edge references an unprojected node"
            )
        provenance = _code_evidence_provenance(
            _value(evidence_edge, "provenance", default="repository")
        )
        authoritative = bool(
            _value(evidence_edge, "authoritative", default=False)
        )
        authority = (
            ContractAuthority.SOURCE_OBSERVATION
            if authoritative and not provenance.context_only
            else ContractAuthority.CONTEXT_ONLY
        )
        raw_kind = str(
            getattr(
                _value(evidence_edge, "kind", default=""),
                "value",
                _value(evidence_edge, "kind", default=""),
            )
        )
        kind = {
            "depends_on": ContractEdgeKind.DEPENDS_ON,
            "derived_from": ContractEdgeKind.SOURCED_FROM,
            "targets_tree": ContractEdgeKind.SOURCED_FROM,
            "related_to": ContractEdgeKind.RELATED_TO,
            "mentions": ContractEdgeKind.RELATED_TO,
            "suggests": ContractEdgeKind.RELATED_TO,
        }.get(raw_kind, ContractEdgeKind.DEPENDS_ON)
        edge_id = str(_value(evidence_edge, "edge_id", default="") or "")
        builder.edge(
            kind,
            source,
            target,
            provenance=provenance,
            authority=authority,
            mandatory=authority.authority_bearing,
            payload={
                "code_evidence_kind": raw_kind,
                "metadata": _plain(
                    _value(evidence_edge, "metadata", default={}) or {}
                ),
            },
            source_refs=tuple(
                item
                for item in (
                    edge_id,
                    str(
                        _value(
                            evidence_edge,
                            "provenance_record_id",
                            default="",
                        )
                        or ""
                    ),
                )
                if item
            ),
        )

    for raw in nodes:
        node = (
            raw
            if isinstance(raw, ContractGraphNode)
            else ContractGraphNode.from_dict(raw)
        )
        if node.snapshot_id != snapshot_id or node.version != version:
            raise SymbolicContractGraphError(
                "extra node is bound to a foreign snapshot or graph version"
            )
        old = builder.nodes.get(node.stable_key)
        if old is not None and old.node_id != node.node_id:
            raise SymbolicContractGraphError(
                f"extra node conflicts with {node.stable_key!r}"
            )
        builder.nodes[node.stable_key] = node
    for raw in edges:
        edge = (
            raw
            if isinstance(raw, ContractGraphEdge)
            else ContractGraphEdge.from_dict(raw)
        )
        if edge.snapshot_id != snapshot_id or edge.version != version:
            raise SymbolicContractGraphError(
                "extra edge is bound to a foreign snapshot or graph version"
            )
        builder.edges[edge.edge_id] = edge

    return SymbolicContractGraph(
        snapshot_id=snapshot_id,
        version=version,
        nodes=tuple(builder.nodes.values()),
        edges=tuple(builder.edges.values()),
    )


def build_symbolic_contract_graph(
    *,
    snapshot_id: str,
    nodes: Iterable[ContractGraphNode | Mapping[str, Any]] = (),
    edges: Iterable[ContractGraphEdge | Mapping[str, Any]] = (),
    version: str = GRAPH_VERSION,
    mandatory_edge_ids: Sequence[str] = (),
) -> SymbolicContractGraph:
    """Build and validate a graph from already-typed records."""

    return SymbolicContractGraph(
        snapshot_id=snapshot_id,
        version=version,
        nodes=tuple(
            item
            if isinstance(item, ContractGraphNode)
            else ContractGraphNode.from_dict(item)
            for item in nodes
        ),
        edges=tuple(
            item
            if isinstance(item, ContractGraphEdge)
            else ContractGraphEdge.from_dict(item)
            for item in edges
        ),
        mandatory_edge_ids=tuple(mandatory_edge_ids),
    )


# Compact compatibility aliases for downstream graph consumers.
SymbolicContractNode = ContractGraphNode
SymbolicContractEdge = ContractGraphEdge
MandatoryContractClosure = ContractGraphClosure
GraphRAGReceipt = GraphRAGRetrievalReceipt
GraphNode = ContractGraphNode
GraphEdge = ContractGraphEdge
GraphNodeKind = ContractNodeKind
GraphEdgeKind = ContractEdgeKind
GraphAuthority = ContractAuthority
GraphProvenance = ContractProvenance
MandatoryClosureError = IncompleteMandatoryClosureError
RetrievalReceipt = GraphRAGRetrievalReceipt
project_contract_graph = project_symbolic_contract_graph


__all__ = [
    "BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE",
    "BOUNDED_GRAPHRAG_VIEW_SCHEMA",
    "CONTENT_IDENTITY_CANONICALIZATION",
    "CONTENT_IDENTITY_PROFILE",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_CLOSURE_DEPTH",
    "DEFAULT_MAX_CLOSURE_EDGES",
    "DEFAULT_MAX_CLOSURE_NODES",
    "DEFAULT_MAX_RETRIEVAL_BYTES",
    "EXACT_DATASETS_CYPHER_AST_MODULE",
    "EXACT_DATASETS_CYPHER_PARSER_MODULE",
    "EXACT_DATASETS_GRAPHRAG_MODULE",
    "GRAPH_VERSION",
    "GRAPHRAG_RETRIEVAL_RECEIPT_SCHEMA",
    "SCAEV031DATASETSGRAPH",
    "SCAEV031DATASETSGRAPH_EVIDENCE",
    "SYMBOLIC_CONTRACT_CLOSURE_SCHEMA",
    "SYMBOLIC_CONTRACT_EDGE_SCHEMA",
    "SYMBOLIC_CONTRACT_GRAPH_INTERFACE",
    "SYMBOLIC_CONTRACT_GRAPH_SCHEMA",
    "SYMBOLIC_CONTRACT_NODE_SCHEMA",
    "BoundedGraphRAGRetriever",
    "BoundedGraphRAGView",
    "CandidateRetrievalError",
    "ClosureBounds",
    "ClosureDirection",
    "ContractAuthority",
    "ContractEdgeKind",
    "ContractGraphClosure",
    "ContractGraphEdge",
    "ContractGraphNode",
    "ContractNodeKind",
    "ContractProvenance",
    "ExactDatasetsGraphProviderError",
    "GraphContentIdentity",
    "GraphAuthority",
    "GraphEdge",
    "GraphEdgeKind",
    "GraphNode",
    "GraphNodeKind",
    "GraphProvenance",
    "GraphRAGCandidate",
    "GraphRAGReceipt",
    "GraphRAGRetrievalReceipt",
    "IncompleteMandatoryClosureError",
    "MandatoryClosureError",
    "MandatoryContractClosure",
    "RetrievalReceipt",
    "RetrievalBounds",
    "SymbolicContractEdge",
    "SymbolicContractGraph",
    "SymbolicContractGraphError",
    "SymbolicContractNode",
    "SymbolicGraphBoundsError",
    "build_symbolic_contract_graph",
    "canonical_contract_graph_bytes",
    "canonical_contract_graph_json",
    "project_contract_graph",
    "project_symbolic_contract_graph",
]
