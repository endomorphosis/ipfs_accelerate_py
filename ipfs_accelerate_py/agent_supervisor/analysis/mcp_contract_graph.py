"""Complete cross-repository MCP++ / SwissKnife contract graph (DCR-021).

Interfaces
----------
* ``McpContractGraph@1``
* ``ContractEdge@1``

Links each mandatory consumer path through:

    UI action → descriptor → ORB/IDL → MCP method/schema → mediator →
    route → dispatcher → handler → effect → receipt → runtime identity

Normative rules:

* Every mandatory consumer edge is resolved **exactly once** or retained as a
  typed blocker (unresolved, ambiguous, authority conflict, expected-only).
* Graph, node, and edge CIDs are recomputed from canonical DAG-JSON bytes and
  never trusted from a claimed field.
* Expected descriptors never masquerade as observed implementations.
* Unresolved edges and authority conflicts stay explicit; they never collapse
  into a silent winner.
* Absolute host paths are rejected so identities remain relocation-stable.

Evidence term: ``dcr/contract-graph@1``.

Conflict policy: unresolved edges and authority conflicts stay explicit;
expected descriptors never masquerade as observed implementations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .mcp_contract_identity import (
    CanonicalContractIdentity,
    ContractDirection,
    identify_contract_declaration,
    is_pseudo_cid,
    validate_multiformat_cid,
)

# ---------------------------------------------------------------------------
# Schemas / interfaces / bounds
# ---------------------------------------------------------------------------

MCP_CONTRACT_GRAPH_INTERFACE: Final = "McpContractGraph@1"
CONTRACT_EDGE_INTERFACE: Final = "ContractEdge@1"
CONTRACT_NODE_INTERFACE: Final = "ContractNode@1"
MCP_CONTRACT_GRAPH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-graph@1"
)
CONTRACT_EDGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-edge@1"
)
CONTRACT_NODE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-node@1"
)
CONTRACT_BLOCKER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-blocker@1"
)
GRAPH_VERSION: Final = "1"
CONTRACT_GRAPH_EVIDENCE_TERM: Final = "dcr/contract-graph@1"
CONTRACT_VERSION: Final[int] = 1

DCR_TASK_ID: Final = "DCR-021"
DCR_ARTIFACT_PATH: Final = (
    "data/agent_supervisor/deterministic_contract_repair/mcp_contract_graph.json"
)
DEFAULT_MAX_BYTES: Final[int] = 1_048_576
DEFAULT_MAX_NODES: Final[int] = 50_000
DEFAULT_MAX_EDGES: Final[int] = 100_000
DEFAULT_MAX_CONSUMERS: Final[int] = 10_000
_MAX_FIELD_BYTES: Final[int] = 4_096
_MAX_SOURCE_REFS: Final[int] = 64
_ABSOLUTE_PATH_RE: Final = re.compile(r"\A(?:/|[A-Za-z]:[\\/]|\\\\)")

# Ordered mandatory consumer-path stages (effects of DCR-021).
MANDATORY_CONSUMER_STAGES: Final[tuple[str, ...]] = (
    "ui_action",
    "descriptor",
    "orb_idl",
    "mcp_method_schema",
    "mediator",
    "route",
    "dispatcher",
    "handler",
    "effect",
    "receipt",
    "runtime_identity",
)

# Edge kinds connecting consecutive stages.
MANDATORY_EDGE_KINDS: Final[tuple[str, ...]] = (
    "ui_action_to_descriptor",
    "descriptor_to_orb_idl",
    "orb_idl_to_mcp_method_schema",
    "mcp_method_schema_to_mediator",
    "mediator_to_route",
    "route_to_dispatcher",
    "dispatcher_to_handler",
    "handler_to_effect",
    "effect_to_receipt",
    "receipt_to_runtime_identity",
)

# Multi-root ownership (plan §3.3) projected into the graph.
OWNING_ROOTS: Final[tuple[str, ...]] = (
    "swissknife",
    "Mcp-Plus-Plus",
    "external/ipfs_accelerate",
    "external/ipfs_datasets",
    "external/ipfs_kit",
    "orchestration",
)


class McpContractGraphError(ValueError):
    """Graph input is malformed or violates a closed invariant."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mcp_contract_graph_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class McpContractGraphBoundsError(McpContractGraphError):
    """A graph record exceeded a hard compactness bound."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "graph_bounds_exceeded",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class DuplicateMandatoryEdgeError(McpContractGraphError):
    """A mandatory consumer edge was resolved more than once."""

    def __init__(
        self,
        *,
        consumer_id: str,
        edge_kind: str,
        count: int,
    ) -> None:
        super().__init__(
            f"mandatory edge {edge_kind!r} for consumer {consumer_id!r} "
            f"resolved {count} times (must be exactly once or a typed blocker)",
            reason_code="duplicate_mandatory_edge",
            details={
                "consumer_id": consumer_id,
                "edge_kind": edge_kind,
                "count": count,
            },
        )


class ContractNodeKind(str, Enum):
    """Closed vocabulary of graph node kinds along the consumer path."""

    UI_ACTION = "ui_action"
    DESCRIPTOR = "descriptor"
    ORB_IDL = "orb_idl"
    MCP_METHOD = "mcp_method"
    MCP_SCHEMA = "mcp_schema"
    MEDIATOR = "mediator"
    ROUTE = "route"
    DISPATCHER = "dispatcher"
    HANDLER = "handler"
    EFFECT = "effect"
    RECEIPT = "receipt"
    RUNTIME_IDENTITY = "runtime_identity"
    CONSUMER = "consumer"
    PACKAGE = "package"
    ROOT = "root"
    BLOCKER = "blocker"
    EXPECTED = "expected"
    OBSERVED = "observed"


class ContractEdgeKind(str, Enum):
    """Closed vocabulary of typed edges (mandatory chain + structural)."""

    UI_ACTION_TO_DESCRIPTOR = "ui_action_to_descriptor"
    DESCRIPTOR_TO_ORB_IDL = "descriptor_to_orb_idl"
    ORB_IDL_TO_MCP_METHOD_SCHEMA = "orb_idl_to_mcp_method_schema"
    MCP_METHOD_SCHEMA_TO_MEDIATOR = "mcp_method_schema_to_mediator"
    MEDIATOR_TO_ROUTE = "mediator_to_route"
    ROUTE_TO_DISPATCHER = "route_to_dispatcher"
    DISPATCHER_TO_HANDLER = "dispatcher_to_handler"
    HANDLER_TO_EFFECT = "handler_to_effect"
    EFFECT_TO_RECEIPT = "effect_to_receipt"
    RECEIPT_TO_RUNTIME_IDENTITY = "receipt_to_runtime_identity"
    CONTAINS = "contains"
    DECLARES = "declares"
    IMPLEMENTS = "implements"
    OBSERVES = "observes"
    OWNS = "owns"
    RELATED_TO = "related_to"
    BLOCKED_BY = "blocked_by"


class ContractAuthority(str, Enum):
    """Authority class retained on every node and edge.

    Expected (declaration) and observed (implementation) authority remain
    distinct; nominating / context sources cannot close mandatory edges.
    """

    REVIEWED_DECLARATION = "reviewed_declaration"
    SOURCE_OBSERVATION = "source_observation"
    POLICY = "policy"
    NOMINATING = "nominating"
    CONTEXT_ONLY = "context_only"
    NONE = "none"

    @property
    def authority_bearing(self) -> bool:
        return self in {
            ContractAuthority.REVIEWED_DECLARATION,
            ContractAuthority.SOURCE_OBSERVATION,
            ContractAuthority.POLICY,
        }

    @property
    def is_declaration(self) -> bool:
        return self is ContractAuthority.REVIEWED_DECLARATION

    @property
    def is_observation(self) -> bool:
        return self is ContractAuthority.SOURCE_OBSERVATION


class EdgeResolution(str, Enum):
    """Closed outcomes for one mandatory consumer edge."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    AUTHORITY_CONFLICT = "authority_conflict"
    EXPECTED_ONLY = "expected_only"
    OBSERVED_ONLY = "observed_only"


class BlockerKind(str, Enum):
    """Typed blockers retained when a mandatory edge cannot resolve uniquely."""

    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    AUTHORITY_CONFLICT = "authority_conflict"
    EXPECTED_ONLY = "expected_only"
    OBSERVED_ONLY = "observed_only"
    DUPLICATE_ALIAS = "duplicate_alias"
    PSEUDO_CID = "pseudo_cid"
    MIXED_ROOT = "mixed_root"


# Stage → node kind mapping for the mandatory chain.
_STAGE_NODE_KIND: Final[Mapping[str, ContractNodeKind]] = {
    "ui_action": ContractNodeKind.UI_ACTION,
    "descriptor": ContractNodeKind.DESCRIPTOR,
    "orb_idl": ContractNodeKind.ORB_IDL,
    "mcp_method_schema": ContractNodeKind.MCP_METHOD,
    "mediator": ContractNodeKind.MEDIATOR,
    "route": ContractNodeKind.ROUTE,
    "dispatcher": ContractNodeKind.DISPATCHER,
    "handler": ContractNodeKind.HANDLER,
    "effect": ContractNodeKind.EFFECT,
    "receipt": ContractNodeKind.RECEIPT,
    "runtime_identity": ContractNodeKind.RUNTIME_IDENTITY,
}

_EDGE_KIND_STAGES: Final[Mapping[str, tuple[str, str]]] = {
    kind: (MANDATORY_CONSUMER_STAGES[index], MANDATORY_CONSUMER_STAGES[index + 1])
    for index, kind in enumerate(MANDATORY_EDGE_KINDS)
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _norm_text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    allow_empty: bool = False,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise McpContractGraphError(
            f"{field_name} must be a string",
            reason_code="invalid_field_type",
            details={"field": field_name, "type": type(value).__name__},
        )
    if required and not text and not allow_empty:
        raise McpContractGraphError(
            f"{field_name} is required",
            reason_code="missing_required_field",
            details={"field": field_name},
        )
    if len(text.encode("utf-8")) > _MAX_FIELD_BYTES:
        raise McpContractGraphBoundsError(
            f"{field_name} exceeds the {_MAX_FIELD_BYTES}-byte limit",
            details={"field": field_name},
        )
    return text


def _norm_enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise McpContractGraphError(
                f"unknown {field_name}: {value!r}",
                reason_code="unknown_enum_value",
                details={"field": field_name, "value": value},
            ) from exc
    raise McpContractGraphError(
        f"{field_name} must be a valid {enum_cls.__name__}",
        reason_code="invalid_enum",
        details={"field": field_name},
    )


def _reject_absolute_path(value: str, *, field_name: str) -> str:
    text = _norm_text(value, field_name=field_name)
    if not text:
        return text
    if _ABSOLUTE_PATH_RE.match(text) or ".." in text.split("/"):
        raise McpContractGraphError(
            f"{field_name} must be repository-relative and relocation-stable",
            reason_code="non_relocation_stable_path",
            details={"field": field_name, "value": text},
        )
    return text


def _sorted_unique_strings(
    values: Iterable[Any] | None,
    *,
    field_name: str,
    maximum: int,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _norm_text(raw, field_name=field_name)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    cleaned.sort()
    if len(cleaned) > maximum:
        raise McpContractGraphBoundsError(
            f"{field_name} exceeds its item limit ({maximum})",
            details={"field": field_name, "count": len(cleaned)},
        )
    return tuple(cleaned)


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise McpContractGraphBoundsError("graph record exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise McpContractGraphError(
            "floating values are not canonical contract graph data",
            reason_code="non_canonical_json",
        )
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise McpContractGraphBoundsError(
                "graph mappings require at most 1024 string keys"
            )
        return {key: _plain(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 16_384:
            raise McpContractGraphBoundsError("graph sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise McpContractGraphError(
        f"unsupported graph value: {type(value).__name__}",
        reason_code="non_canonical_json",
    )


def canonical_graph_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a graph artifact."""

    try:
        return canonical_json_bytes(_plain(value))
    except ContractValidationError as exc:
        raise McpContractGraphError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def canonical_graph_cid(value: Any) -> str:
    """Return CIDv1 dag-json/sha2-256 for *value*."""

    try:
        return content_identity(_plain(value))
    except ContractValidationError as exc:
        raise McpContractGraphError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def digest_for_canonical_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` for exact canonical bytes (never a CID)."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise McpContractGraphError(
            "canonical bytes must be bytes-like",
            reason_code="invalid_field_type",
        )
    return "sha256:" + hashlib.sha256(bytes(data)).hexdigest()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        value = to_dict() if callable(to_dict) else value
    if not isinstance(value, Mapping):
        raise McpContractGraphError(
            f"{field_name} must be an object",
            reason_code="invalid_field_type",
        )
    return MappingProxyType(_plain(value))


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Relocation-stable source coordinates for one graph endpoint."""

    path: str
    start_line: int = 1
    start_column: int = 0
    end_line: int = 1
    end_column: int = 0
    root_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _reject_absolute_path(
                _norm_text(self.path, field_name="path"),
                field_name="path",
            ),
        )
        object.__setattr__(
            self,
            "root_id",
            _norm_text(self.root_id, field_name="root_id"),
        )
        for name in ("start_line", "end_line"):
            value = int(getattr(self, name))
            if value < 1:
                raise McpContractGraphError(
                    f"{name} must be >= 1",
                    reason_code="invalid_span",
                )
            object.__setattr__(self, name, value)
        for name in ("start_column", "end_column"):
            value = int(getattr(self, name))
            if value < 0:
                raise McpContractGraphError(
                    f"{name} must be >= 0",
                    reason_code="invalid_span",
                )
            object.__setattr__(self, name, value)
        if self.end_line < self.start_line:
            raise McpContractGraphError(
                "end_line precedes start_line",
                reason_code="invalid_span",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "root_id": self.root_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SourceSpan | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise McpContractGraphError(
                "source span must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            path=str(payload.get("path") or ""),
            start_line=int(payload.get("start_line") or 1),
            start_column=int(payload.get("start_column") or 0),
            end_line=int(payload.get("end_line") or 1),
            end_column=int(payload.get("end_column") or 0),
            root_id=str(payload.get("root_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class ContractNode:
    """One typed, content-addressed graph node.

    Interface companion to ``ContractEdge@1``; nodes carry authority, root
    ownership, and optional declaration/observation provenance.
    """

    kind: ContractNodeKind
    stable_key: str
    label: str
    authority: ContractAuthority
    owning_root: str
    version: str = GRAPH_VERSION
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    span: SourceSpan | None = None
    identity_cid: str = ""
    node_id: str = ""
    schema: str = CONTRACT_NODE_SCHEMA
    interface: str = CONTRACT_NODE_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _norm_enum(self.kind, ContractNodeKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "authority",
            _norm_enum(self.authority, ContractAuthority, field_name="authority"),
        )
        object.__setattr__(
            self,
            "stable_key",
            _norm_text(self.stable_key, field_name="stable_key", required=True),
        )
        object.__setattr__(
            self,
            "label",
            _norm_text(self.label, field_name="label", required=True),
        )
        object.__setattr__(
            self,
            "owning_root",
            _reject_absolute_path(
                _norm_text(self.owning_root, field_name="owning_root", required=True),
                field_name="owning_root",
            ),
        )
        object.__setattr__(
            self,
            "version",
            _norm_text(self.version, field_name="version", required=True),
        )
        object.__setattr__(self, "payload", _mapping(self.payload, "payload"))
        object.__setattr__(
            self,
            "source_refs",
            _sorted_unique_strings(
                self.source_refs, field_name="source_refs", maximum=_MAX_SOURCE_REFS
            ),
        )
        if self.span is not None and not isinstance(self.span, SourceSpan):
            object.__setattr__(self, "span", SourceSpan.from_dict(self.span))  # type: ignore[arg-type]
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != CONTRACT_NODE_SCHEMA:
            raise McpContractGraphError(
                "unsupported contract node schema",
                reason_code="unsupported_schema",
            )
        if self.interface != CONTRACT_NODE_INTERFACE:
            raise McpContractGraphError(
                "unsupported contract node interface",
                reason_code="unsupported_interface",
            )
        identity = _norm_text(self.identity_cid, field_name="identity_cid")
        if identity:
            if is_pseudo_cid(identity):
                raise McpContractGraphError(
                    "node identity_cid must be a multiformat CID",
                    reason_code="pseudo_cid",
                )
            validate_multiformat_cid(identity, field_name="identity_cid")
        object.__setattr__(self, "identity_cid", identity)
        expected = canonical_graph_cid(self._identity_payload())
        claimed = _norm_text(self.node_id, field_name="node_id")
        if claimed and claimed != expected:
            raise McpContractGraphError(
                "node_id does not match recomputed identity",
                reason_code="forged_node_id",
                details={"claimed": claimed, "local": expected},
            )
        object.__setattr__(self, "node_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "kind": self.kind.value
            if isinstance(self.kind, ContractNodeKind)
            else str(self.kind),
            "stable_key": self.stable_key,
            "label": self.label,
            "authority": self.authority.value
            if isinstance(self.authority, ContractAuthority)
            else str(self.authority),
            "owning_root": self.owning_root,
            "version": self.version,
            "payload": dict(self.payload),
            "source_refs": list(self.source_refs),
            "span": self.span.to_dict() if self.span is not None else None,
            "identity_cid": self.identity_cid,
        }

    @property
    def authoritative(self) -> bool:
        return self.authority.authority_bearing

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractNode":
        if not isinstance(payload, Mapping):
            raise McpContractGraphError(
                "contract node must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            kind=str(payload.get("kind") or ""),
            stable_key=str(payload.get("stable_key") or ""),
            label=str(payload.get("label") or payload.get("stable_key") or ""),
            authority=str(payload.get("authority") or ""),
            owning_root=str(payload.get("owning_root") or ""),
            version=str(payload.get("version") or GRAPH_VERSION),
            payload=payload.get("payload") or {},
            source_refs=tuple(payload.get("source_refs") or ()),
            span=SourceSpan.from_dict(
                payload.get("span") if isinstance(payload.get("span"), Mapping) else None
            ),
            identity_cid=str(payload.get("identity_cid") or ""),
            node_id=str(payload.get("node_id") or ""),
            schema=str(payload.get("schema") or CONTRACT_NODE_SCHEMA),
            interface=str(payload.get("interface") or CONTRACT_NODE_INTERFACE),
        )


@dataclass(frozen=True, slots=True)
class ContractEdge:
    """One typed relationship; mandatory edges close consumer-path stages.

    Interface: ``ContractEdge@1``
    """

    kind: ContractEdgeKind
    source: str
    target: str
    authority: ContractAuthority
    mandatory: bool = False
    resolution: EdgeResolution = EdgeResolution.RESOLVED
    consumer_id: str = ""
    owning_root: str = ""
    version: str = GRAPH_VERSION
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    edge_id: str = ""
    schema: str = CONTRACT_EDGE_SCHEMA
    interface: str = CONTRACT_EDGE_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _norm_enum(self.kind, ContractEdgeKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "authority",
            _norm_enum(self.authority, ContractAuthority, field_name="authority"),
        )
        object.__setattr__(
            self,
            "resolution",
            _norm_enum(self.resolution, EdgeResolution, field_name="resolution"),
        )
        for name in ("source", "target"):
            object.__setattr__(
                self,
                name,
                _norm_text(getattr(self, name), field_name=name, required=True),
            )
        if not isinstance(self.mandatory, bool):
            raise McpContractGraphError(
                "edge mandatory must be boolean",
                reason_code="invalid_field_type",
            )
        object.__setattr__(
            self,
            "consumer_id",
            _norm_text(self.consumer_id, field_name="consumer_id"),
        )
        object.__setattr__(
            self,
            "owning_root",
            _reject_absolute_path(
                _norm_text(self.owning_root, field_name="owning_root"),
                field_name="owning_root",
            ),
        )
        object.__setattr__(
            self,
            "version",
            _norm_text(self.version, field_name="version", required=True),
        )
        object.__setattr__(self, "payload", _mapping(self.payload, "payload"))
        object.__setattr__(
            self,
            "source_refs",
            _sorted_unique_strings(
                self.source_refs, field_name="source_refs", maximum=_MAX_SOURCE_REFS
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != CONTRACT_EDGE_SCHEMA:
            raise McpContractGraphError(
                "unsupported contract edge schema",
                reason_code="unsupported_schema",
            )
        if self.interface != CONTRACT_EDGE_INTERFACE:
            raise McpContractGraphError(
                "unsupported contract edge interface",
                reason_code="unsupported_interface",
            )
        if self.mandatory:
            if not self.authority.authority_bearing:
                raise McpContractGraphError(
                    "mandatory edge must carry declaration, observation, or policy authority",
                    reason_code="mandatory_without_authority",
                )
            if self.authority is ContractAuthority.CONTEXT_ONLY:
                raise McpContractGraphError(
                    "context-only edge cannot be mandatory",
                    reason_code="context_only_mandatory",
                )
            if not self.consumer_id:
                raise McpContractGraphError(
                    "mandatory edge requires consumer_id",
                    reason_code="missing_consumer_id",
                )
            if self.kind.value not in MANDATORY_EDGE_KINDS:
                raise McpContractGraphError(
                    f"unknown mandatory edge kind: {self.kind.value}",
                    reason_code="unknown_mandatory_edge_kind",
                )
        if (
            self.resolution is EdgeResolution.RESOLVED
            and self.mandatory
            and not self.target
        ):
            raise McpContractGraphError(
                "resolved mandatory edge requires a target",
                reason_code="resolved_without_target",
            )
        expected = canonical_graph_cid(self._identity_payload())
        claimed = _norm_text(self.edge_id, field_name="edge_id")
        if claimed and claimed != expected:
            raise McpContractGraphError(
                "edge_id does not match recomputed identity",
                reason_code="forged_edge_id",
                details={"claimed": claimed, "local": expected},
            )
        object.__setattr__(self, "edge_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "kind": self.kind.value
            if isinstance(self.kind, ContractEdgeKind)
            else str(self.kind),
            "source": self.source,
            "target": self.target,
            "authority": self.authority.value
            if isinstance(self.authority, ContractAuthority)
            else str(self.authority),
            "mandatory": self.mandatory,
            "resolution": self.resolution.value
            if isinstance(self.resolution, EdgeResolution)
            else str(self.resolution),
            "consumer_id": self.consumer_id,
            "owning_root": self.owning_root,
            "version": self.version,
            "payload": dict(self.payload),
            "source_refs": list(self.source_refs),
        }

    @property
    def authoritative(self) -> bool:
        return self.authority.authority_bearing

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractEdge":
        if not isinstance(payload, Mapping):
            raise McpContractGraphError(
                "contract edge must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            kind=str(payload.get("kind") or ""),
            source=str(payload.get("source") or ""),
            target=str(payload.get("target") or ""),
            authority=str(payload.get("authority") or ""),
            mandatory=bool(payload.get("mandatory", False)),
            resolution=str(
                payload.get("resolution") or EdgeResolution.RESOLVED.value
            ),
            consumer_id=str(payload.get("consumer_id") or ""),
            owning_root=str(payload.get("owning_root") or ""),
            version=str(payload.get("version") or GRAPH_VERSION),
            payload=payload.get("payload") or {},
            source_refs=tuple(payload.get("source_refs") or ()),
            edge_id=str(payload.get("edge_id") or ""),
            schema=str(payload.get("schema") or CONTRACT_EDGE_SCHEMA),
            interface=str(payload.get("interface") or CONTRACT_EDGE_INTERFACE),
        )


@dataclass(frozen=True, slots=True)
class ContractBlocker:
    """Typed blocker for one incomplete or conflicting mandatory edge."""

    kind: BlockerKind
    consumer_id: str
    edge_kind: str
    stage: str
    reason_code: str
    details: Mapping[str, Any] = field(default_factory=dict)
    candidate_node_ids: tuple[str, ...] = ()
    blocker_id: str = ""
    schema: str = CONTRACT_BLOCKER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _norm_enum(self.kind, BlockerKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "consumer_id",
            _norm_text(self.consumer_id, field_name="consumer_id", required=True),
        )
        object.__setattr__(
            self,
            "edge_kind",
            _norm_text(self.edge_kind, field_name="edge_kind", required=True),
        )
        object.__setattr__(
            self,
            "stage",
            _norm_text(self.stage, field_name="stage", required=True),
        )
        object.__setattr__(
            self,
            "reason_code",
            _norm_text(self.reason_code, field_name="reason_code", required=True),
        )
        object.__setattr__(self, "details", _mapping(self.details, "details"))
        object.__setattr__(
            self,
            "candidate_node_ids",
            _sorted_unique_strings(
                self.candidate_node_ids,
                field_name="candidate_node_ids",
                maximum=256,
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        expected = canonical_graph_cid(self._identity_payload())
        claimed = _norm_text(self.blocker_id, field_name="blocker_id")
        if claimed and claimed != expected:
            raise McpContractGraphError(
                "blocker_id does not match recomputed identity",
                reason_code="forged_blocker_id",
            )
        object.__setattr__(self, "blocker_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value
            if isinstance(self.kind, BlockerKind)
            else str(self.kind),
            "consumer_id": self.consumer_id,
            "edge_kind": self.edge_kind,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "details": dict(self.details),
            "candidate_node_ids": list(self.candidate_node_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"blocker_id": self.blocker_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractBlocker":
        if not isinstance(payload, Mapping):
            raise McpContractGraphError(
                "contract blocker must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            kind=str(payload.get("kind") or ""),
            consumer_id=str(payload.get("consumer_id") or ""),
            edge_kind=str(payload.get("edge_kind") or ""),
            stage=str(payload.get("stage") or ""),
            reason_code=str(payload.get("reason_code") or ""),
            details=payload.get("details") or {},
            candidate_node_ids=tuple(payload.get("candidate_node_ids") or ()),
            blocker_id=str(payload.get("blocker_id") or ""),
            schema=str(payload.get("schema") or CONTRACT_BLOCKER_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Stage endpoint input (builder API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageEndpoint:
    """One stage endpoint declared or observed for a consumer path."""

    stage: str
    stable_key: str
    label: str
    authority: ContractAuthority | str
    owning_root: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    span: SourceSpan | None = None
    identity_cid: str = ""

    def __post_init__(self) -> None:
        stage = _norm_text(self.stage, field_name="stage", required=True)
        if stage not in MANDATORY_CONSUMER_STAGES:
            raise McpContractGraphError(
                f"unknown consumer stage: {stage!r}",
                reason_code="unknown_stage",
            )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self,
            "stable_key",
            _norm_text(self.stable_key, field_name="stable_key", required=True),
        )
        object.__setattr__(
            self,
            "label",
            _norm_text(self.label, field_name="label", required=True),
        )
        object.__setattr__(
            self,
            "authority",
            _norm_enum(self.authority, ContractAuthority, field_name="authority"),
        )
        object.__setattr__(
            self,
            "owning_root",
            _reject_absolute_path(
                _norm_text(self.owning_root, field_name="owning_root", required=True),
                field_name="owning_root",
            ),
        )
        object.__setattr__(self, "payload", _mapping(self.payload, "payload"))
        object.__setattr__(
            self,
            "source_refs",
            _sorted_unique_strings(
                self.source_refs, field_name="source_refs", maximum=_MAX_SOURCE_REFS
            ),
        )
        if self.span is not None and not isinstance(self.span, SourceSpan):
            object.__setattr__(self, "span", SourceSpan.from_dict(self.span))  # type: ignore[arg-type]
        object.__setattr__(
            self,
            "identity_cid",
            _norm_text(self.identity_cid, field_name="identity_cid"),
        )


@dataclass(frozen=True, slots=True)
class ConsumerPathInput:
    """One consumer subject and its stage endpoints for graph construction."""

    consumer_id: str
    package: str
    operation: str
    owning_root: str
    endpoints: tuple[StageEndpoint, ...]
    transport: str = ""
    profile: str = ""
    aliases: tuple[str, ...] = ()
    declaration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "consumer_id",
            _norm_text(self.consumer_id, field_name="consumer_id", required=True),
        )
        object.__setattr__(
            self,
            "package",
            _norm_text(self.package, field_name="package", required=True),
        )
        object.__setattr__(
            self,
            "operation",
            _norm_text(self.operation, field_name="operation", required=True),
        )
        object.__setattr__(
            self,
            "owning_root",
            _reject_absolute_path(
                _norm_text(self.owning_root, field_name="owning_root", required=True),
                field_name="owning_root",
            ),
        )
        object.__setattr__(
            self,
            "transport",
            _norm_text(self.transport, field_name="transport"),
        )
        object.__setattr__(
            self,
            "profile",
            _norm_text(self.profile, field_name="profile"),
        )
        object.__setattr__(
            self,
            "aliases",
            _sorted_unique_strings(
                self.aliases, field_name="aliases", maximum=64
            ),
        )
        object.__setattr__(
            self, "declaration", _mapping(self.declaration, "declaration")
        )
        if not isinstance(self.endpoints, Sequence) or isinstance(
            self.endpoints, (str, bytes)
        ):
            raise McpContractGraphError(
                "endpoints must be a sequence",
                reason_code="invalid_field_type",
            )
        normalized: list[StageEndpoint] = []
        for item in self.endpoints:
            if not isinstance(item, StageEndpoint):
                raise McpContractGraphError(
                    "each endpoint must be a StageEndpoint",
                    reason_code="invalid_field_type",
                )
            normalized.append(item)
        object.__setattr__(self, "endpoints", tuple(normalized))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpContractGraph:
    """Immutable, content-addressed cross-repository contract graph.

    Interface: ``McpContractGraph@1``

    Invariant: for every ``consumer_id`` and every mandatory edge kind, the
    graph holds either exactly one ``RESOLVED`` mandatory edge or one typed
    blocker.  Never zero silent omissions; never multiple resolutions.
    """

    snapshot_id: str
    nodes: tuple[ContractNode, ...] = ()
    edges: tuple[ContractEdge, ...] = ()
    blockers: tuple[ContractBlocker, ...] = ()
    consumer_ids: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()
    version: str = GRAPH_VERSION
    schema: str = MCP_CONTRACT_GRAPH_SCHEMA
    interface: str = MCP_CONTRACT_GRAPH_INTERFACE
    evidence_term: str = CONTRACT_GRAPH_EVIDENCE_TERM
    graph_cid: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            _norm_text(self.snapshot_id, field_name="snapshot_id", required=True),
        )
        object.__setattr__(
            self,
            "version",
            _norm_text(self.version, field_name="version", required=True),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        object.__setattr__(
            self,
            "evidence_term",
            _norm_text(self.evidence_term, field_name="evidence_term", required=True),
        )
        if self.schema != MCP_CONTRACT_GRAPH_SCHEMA:
            raise McpContractGraphError(
                "unsupported mcp contract graph schema",
                reason_code="unsupported_schema",
            )
        if self.interface != MCP_CONTRACT_GRAPH_INTERFACE:
            raise McpContractGraphError(
                "unsupported mcp contract graph interface",
                reason_code="unsupported_interface",
            )
        if self.evidence_term != CONTRACT_GRAPH_EVIDENCE_TERM:
            raise McpContractGraphError(
                "unsupported contract graph evidence term",
                reason_code="unsupported_evidence_term",
            )

        node_map: dict[str, ContractNode] = {}
        key_map: dict[str, ContractNode] = {}
        for raw in self.nodes:
            node = raw if isinstance(raw, ContractNode) else ContractNode.from_dict(raw)
            if node.version != self.version:
                raise McpContractGraphError(
                    f"node {node.node_id} has a foreign graph version",
                    reason_code="foreign_graph_version",
                )
            prior = node_map.get(node.node_id)
            if prior is not None and prior.to_dict() != node.to_dict():
                raise McpContractGraphError(
                    f"conflicting node identity: {node.node_id}",
                    reason_code="conflicting_node",
                )
            keyed = key_map.get(node.stable_key)
            if keyed is not None and keyed.node_id != node.node_id:
                raise McpContractGraphError(
                    f"conflicting stable node key: {node.stable_key}",
                    reason_code="conflicting_stable_key",
                )
            node_map[node.node_id] = node
            key_map[node.stable_key] = node
        if len(node_map) > DEFAULT_MAX_NODES:
            raise McpContractGraphBoundsError(
                "graph has too many nodes",
                details={"count": len(node_map)},
            )

        edge_map: dict[str, ContractEdge] = {}
        for raw in self.edges:
            edge = raw if isinstance(raw, ContractEdge) else ContractEdge.from_dict(raw)
            if edge.version != self.version:
                raise McpContractGraphError(
                    f"edge {edge.edge_id} has a foreign graph version",
                    reason_code="foreign_graph_version",
                )
            if edge.source not in node_map:
                raise McpContractGraphError(
                    f"edge {edge.edge_id} references unknown source",
                    reason_code="unknown_node",
                )
            # Target may be a blocker sentinel node or a real node.
            if edge.target and edge.target not in node_map:
                raise McpContractGraphError(
                    f"edge {edge.edge_id} references unknown target",
                    reason_code="unknown_node",
                )
            if edge.mandatory and edge.resolution is EdgeResolution.RESOLVED:
                source = node_map[edge.source]
                target = node_map[edge.target]
                if not source.authoritative or not target.authoritative:
                    raise McpContractGraphError(
                        "mandatory resolved edge cannot promote non-authoritative endpoints",
                        reason_code="mandatory_non_authoritative_endpoint",
                    )
                # Expected descriptors never masquerade as observations.
                if (
                    source.authority is ContractAuthority.REVIEWED_DECLARATION
                    and target.kind
                    in {
                        ContractNodeKind.HANDLER,
                        ContractNodeKind.EFFECT,
                        ContractNodeKind.DISPATCHER,
                    }
                    and target.authority is ContractAuthority.REVIEWED_DECLARATION
                    and edge.kind
                    in {
                        ContractEdgeKind.ROUTE_TO_DISPATCHER,
                        ContractEdgeKind.DISPATCHER_TO_HANDLER,
                        ContractEdgeKind.HANDLER_TO_EFFECT,
                    }
                ):
                    raise McpContractGraphError(
                        "expected descriptor must not masquerade as observed implementation",
                        reason_code="expected_as_observed",
                        details={
                            "edge_kind": edge.kind.value,
                            "source": source.stable_key,
                            "target": target.stable_key,
                        },
                    )
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_EDGES:
            raise McpContractGraphBoundsError(
                "graph has too many edges",
                details={"count": len(edge_map)},
            )

        blocker_map: dict[str, ContractBlocker] = {}
        for raw in self.blockers:
            blocker = (
                raw
                if isinstance(raw, ContractBlocker)
                else ContractBlocker.from_dict(raw)
            )
            blocker_map[blocker.blocker_id] = blocker

        consumers = _sorted_unique_strings(
            self.consumer_ids
            or tuple(
                {
                    edge.consumer_id
                    for edge in edge_map.values()
                    if edge.consumer_id
                }
                | {blocker.consumer_id for blocker in blocker_map.values()}
            ),
            field_name="consumer_ids",
            maximum=DEFAULT_MAX_CONSUMERS,
        )
        roots = _sorted_unique_strings(
            self.roots
            or tuple({node.owning_root for node in node_map.values() if node.owning_root}),
            field_name="roots",
            maximum=64,
        )

        # Enforce: every mandatory consumer edge resolved exactly once or blocker.
        self._assert_mandatory_coverage(
            consumers=consumers,
            edges=tuple(edge_map.values()),
            blockers=tuple(blocker_map.values()),
        )

        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        object.__setattr__(
            self,
            "blockers",
            tuple(blocker_map[key] for key in sorted(blocker_map)),
        )
        object.__setattr__(self, "consumer_ids", consumers)
        object.__setattr__(self, "roots", roots)

        expected_cid = canonical_graph_cid(self._root_payload())
        claimed = _norm_text(self.graph_cid, field_name="graph_cid")
        if claimed and claimed != expected_cid:
            raise McpContractGraphError(
                "graph_cid does not match recomputed identity",
                reason_code="forged_graph_cid",
                details={"claimed": claimed, "local": expected_cid},
            )
        object.__setattr__(self, "graph_cid", expected_cid)

    @staticmethod
    def _assert_mandatory_coverage(
        *,
        consumers: Sequence[str],
        edges: Sequence[ContractEdge],
        blockers: Sequence[ContractBlocker],
    ) -> None:
        resolved: dict[tuple[str, str], list[ContractEdge]] = defaultdict(list)
        for edge in edges:
            if not edge.mandatory or not edge.consumer_id:
                continue
            if edge.resolution is EdgeResolution.RESOLVED:
                resolved[(edge.consumer_id, edge.kind.value)].append(edge)
        blocker_keys: dict[tuple[str, str], list[ContractBlocker]] = defaultdict(list)
        for blocker in blockers:
            blocker_keys[(blocker.consumer_id, blocker.edge_kind)].append(blocker)

        for consumer_id in consumers:
            for edge_kind in MANDATORY_EDGE_KINDS:
                key = (consumer_id, edge_kind)
                resolved_edges = resolved.get(key, [])
                typed_blockers = blocker_keys.get(key, [])
                if len(resolved_edges) > 1:
                    raise DuplicateMandatoryEdgeError(
                        consumer_id=consumer_id,
                        edge_kind=edge_kind,
                        count=len(resolved_edges),
                    )
                if len(resolved_edges) == 1 and typed_blockers:
                    raise McpContractGraphError(
                        f"mandatory edge {edge_kind!r} for consumer "
                        f"{consumer_id!r} cannot be both resolved and blocked",
                        reason_code="resolved_and_blocked",
                        details={
                            "consumer_id": consumer_id,
                            "edge_kind": edge_kind,
                        },
                    )
                if not resolved_edges and not typed_blockers:
                    raise McpContractGraphError(
                        f"mandatory edge {edge_kind!r} for consumer "
                        f"{consumer_id!r} is neither resolved nor a typed blocker",
                        reason_code="missing_mandatory_edge",
                        details={
                            "consumer_id": consumer_id,
                            "edge_kind": edge_kind,
                        },
                    )
                if not resolved_edges and len(typed_blockers) != 1:
                    # Multiple typed blockers for the same edge are collapsed
                    # by the builder; presence of more than one is a forge.
                    if len(typed_blockers) > 1:
                        raise McpContractGraphError(
                            f"mandatory edge {edge_kind!r} for consumer "
                            f"{consumer_id!r} has multiple typed blockers",
                            reason_code="duplicate_typed_blocker",
                            details={
                                "consumer_id": consumer_id,
                                "edge_kind": edge_kind,
                                "count": len(typed_blockers),
                            },
                        )

    def _root_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_term": self.evidence_term,
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "consumer_ids": list(self.consumer_ids),
            "roots": list(self.roots),
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "blockers": [item.to_dict() for item in self.blockers],
        }

    @property
    def graph_id(self) -> str:
        return self.graph_cid

    @property
    def graph_root(self) -> str:
        return self.graph_cid

    @property
    def complete(self) -> bool:
        """True when every mandatory edge is resolved (no blockers)."""

        return not self.blockers and all(
            edge.resolution is EdgeResolution.RESOLVED
            for edge in self.edges
            if edge.mandatory
        )

    @property
    def canonical_digest(self) -> str:
        return digest_for_canonical_bytes(canonical_graph_bytes(self._root_payload()))

    def node(self, node_id: str) -> ContractNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    def node_for_key(self, stable_key: str) -> ContractNode:
        for item in self.nodes:
            if item.stable_key == stable_key:
                return item
        raise KeyError(stable_key)

    def mandatory_edges_for(self, consumer_id: str) -> tuple[ContractEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.mandatory and edge.consumer_id == consumer_id
        )

    def blockers_for(self, consumer_id: str) -> tuple[ContractBlocker, ...]:
        return tuple(
            blocker for blocker in self.blockers if blocker.consumer_id == consumer_id
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._root_payload()
        return {
            "graph_cid": self.graph_cid,
            "graph_id": self.graph_cid,
            "graph_root": self.graph_cid,
            "canonical_digest": self.canonical_digest,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "blocker_count": len(self.blockers),
            "consumer_count": len(self.consumer_ids),
            "complete": self.complete,
            "policies": {
                "trust_claimed_cid": False,
                "expected_as_observed_allowed": False,
                "silent_unresolved_allowed": False,
                "duplicate_mandatory_resolution_allowed": False,
                "absolute_paths_allowed": False,
            },
            **payload,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_graph_bytes(self.to_dict()).decode("utf-8")
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    def to_artifact_bytes(self) -> bytes:
        """Canonical artifact bytes (sorted keys, trailing newline)."""

        body = json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return (body + "\n").encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "McpContractGraph":
        if not isinstance(payload, Mapping):
            raise McpContractGraphError(
                "mcp contract graph must be an object",
                reason_code="invalid_field_type",
            )
        if payload.get("schema") not in (None, MCP_CONTRACT_GRAPH_SCHEMA):
            raise McpContractGraphError(
                "unsupported mcp contract graph schema",
                reason_code="unsupported_schema",
            )
        if payload.get("interface") not in (None, MCP_CONTRACT_GRAPH_INTERFACE):
            raise McpContractGraphError(
                "unsupported mcp contract graph interface",
                reason_code="unsupported_interface",
            )
        graph = cls(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            nodes=tuple(
                ContractNode.from_dict(item)
                for item in (payload.get("nodes") or ())
            ),
            edges=tuple(
                ContractEdge.from_dict(item)
                for item in (payload.get("edges") or ())
            ),
            blockers=tuple(
                ContractBlocker.from_dict(item)
                for item in (payload.get("blockers") or ())
            ),
            consumer_ids=tuple(payload.get("consumer_ids") or ()),
            roots=tuple(payload.get("roots") or ()),
            version=str(payload.get("version") or GRAPH_VERSION),
            schema=str(payload.get("schema") or MCP_CONTRACT_GRAPH_SCHEMA),
            interface=str(payload.get("interface") or MCP_CONTRACT_GRAPH_INTERFACE),
            evidence_term=str(
                payload.get("evidence_term") or CONTRACT_GRAPH_EVIDENCE_TERM
            ),
            graph_cid=str(
                payload.get("graph_cid")
                or payload.get("graph_id")
                or payload.get("graph_root")
                or ""
            ),
        )
        claimed_digest = payload.get("canonical_digest")
        if claimed_digest not in (None, "") and claimed_digest != graph.canonical_digest:
            raise McpContractGraphError(
                "stored canonical_digest does not match recomputed bytes",
                reason_code="forged_canonical_digest",
            )
        return graph

    @classmethod
    def from_json(cls, value: str | bytes) -> "McpContractGraph":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise McpContractGraphError(
                "graph JSON is malformed",
                reason_code="malformed_json",
            ) from exc
        if not isinstance(payload, Mapping):
            raise McpContractGraphError(
                "graph JSON must contain an object",
                reason_code="invalid_field_type",
            )
        return cls.from_dict(payload)

    def verifies_cid(self) -> bool:
        """Return True when graph_cid reconstructs from canonical root bytes."""

        return self.graph_cid == canonical_graph_cid(self._root_payload())


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _node_from_endpoint(endpoint: StageEndpoint) -> ContractNode:
    kind = _STAGE_NODE_KIND[endpoint.stage]
    # mcp_method_schema stage may carry schema metadata on the method node.
    if endpoint.stage == "mcp_method_schema":
        payload = dict(endpoint.payload)
        if "role" not in payload:
            payload["role"] = "mcp_method_schema"
    else:
        payload = dict(endpoint.payload)
    return ContractNode(
        kind=kind,
        stable_key=endpoint.stable_key,
        label=endpoint.label,
        authority=endpoint.authority,  # type: ignore[arg-type]
        owning_root=endpoint.owning_root,
        payload=payload,
        source_refs=endpoint.source_refs,
        span=endpoint.span,
        identity_cid=endpoint.identity_cid,
    )


def _blocker_node(
    *,
    consumer_id: str,
    edge_kind: str,
    kind: BlockerKind,
    owning_root: str,
) -> ContractNode:
    return ContractNode(
        kind=ContractNodeKind.BLOCKER,
        stable_key=f"blocker:{consumer_id}:{edge_kind}:{kind.value}",
        label=f"blocker:{kind.value}",
        authority=ContractAuthority.NONE,
        owning_root=owning_root,
        payload={
            "consumer_id": consumer_id,
            "edge_kind": edge_kind,
            "blocker_kind": kind.value,
        },
    )


def _authority_for_resolution(
    endpoints: Sequence[StageEndpoint],
) -> ContractAuthority:
    """Select the strongest authority present on stage endpoints."""

    order = (
        ContractAuthority.POLICY,
        ContractAuthority.SOURCE_OBSERVATION,
        ContractAuthority.REVIEWED_DECLARATION,
        ContractAuthority.NOMINATING,
        ContractAuthority.CONTEXT_ONLY,
        ContractAuthority.NONE,
    )
    present = {
        item.authority if isinstance(item.authority, ContractAuthority) else ContractAuthority(str(item.authority))
        for item in endpoints
    }
    for candidate in order:
        if candidate in present:
            return candidate
    return ContractAuthority.NONE


def _classify_stage_link(
    *,
    consumer: ConsumerPathInput,
    edge_kind: str,
    source_stage: str,
    target_stage: str,
    by_stage: Mapping[str, list[StageEndpoint]],
) -> tuple[
    EdgeResolution,
    BlockerKind | None,
    StageEndpoint | None,
    StageEndpoint | None,
    tuple[StageEndpoint, ...],
]:
    """Classify one mandatory stage transition for a consumer."""

    sources = list(by_stage.get(source_stage) or ())
    targets = list(by_stage.get(target_stage) or ())

    # Implementation stages require observation authority.
    observation_stages = {
        "dispatcher",
        "handler",
        "effect",
        "receipt",
        "runtime_identity",
    }
    declaration_stages = {
        "ui_action",
        "descriptor",
        "orb_idl",
        "mcp_method_schema",
        "mediator",
        "route",
    }

    if not sources and not targets:
        return (
            EdgeResolution.UNRESOLVED,
            BlockerKind.UNRESOLVED,
            None,
            None,
            (),
        )
    if not sources:
        return (
            EdgeResolution.UNRESOLVED,
            BlockerKind.UNRESOLVED,
            None,
            targets[0] if len(targets) == 1 else None,
            tuple(targets),
        )
    if not targets:
        # Expected-only: declaration side present, observation side missing.
        if target_stage in observation_stages:
            return (
                EdgeResolution.EXPECTED_ONLY,
                BlockerKind.EXPECTED_ONLY,
                sources[0] if len(sources) == 1 else None,
                None,
                tuple(sources),
            )
        return (
            EdgeResolution.UNRESOLVED,
            BlockerKind.UNRESOLVED,
            sources[0] if len(sources) == 1 else None,
            None,
            tuple(sources),
        )

    if len(sources) > 1 or len(targets) > 1:
        return (
            EdgeResolution.AMBIGUOUS,
            BlockerKind.AMBIGUOUS,
            None,
            None,
            tuple(sources + targets),
        )

    source = sources[0]
    target = targets[0]
    source_auth = (
        source.authority
        if isinstance(source.authority, ContractAuthority)
        else ContractAuthority(str(source.authority))
    )
    target_auth = (
        target.authority
        if isinstance(target.authority, ContractAuthority)
        else ContractAuthority(str(target.authority))
    )

    # Observed-only path (no declaration authority on consumer-facing stages).
    if (
        source_stage in declaration_stages
        and source_auth is ContractAuthority.SOURCE_OBSERVATION
        and target_auth is ContractAuthority.SOURCE_OBSERVATION
        and not any(
            (
                ep.authority
                if isinstance(ep.authority, ContractAuthority)
                else ContractAuthority(str(ep.authority))
            )
            is ContractAuthority.REVIEWED_DECLARATION
            for ep in by_stage.get("descriptor", ())
        )
        and source_stage in {"ui_action", "descriptor"}
    ):
        return (
            EdgeResolution.OBSERVED_ONLY,
            BlockerKind.OBSERVED_ONLY,
            source,
            target,
            (source, target),
        )

    # Expected masquerading as observed implementation.
    if target_stage in observation_stages and target_auth is (
        ContractAuthority.REVIEWED_DECLARATION
    ):
        return (
            EdgeResolution.AUTHORITY_CONFLICT,
            BlockerKind.AUTHORITY_CONFLICT,
            source,
            target,
            (source, target),
        )

    # Mixed-root ownership conflict on observation stages.
    if (
        target_stage in observation_stages
        and source.owning_root
        and target.owning_root
        and source.owning_root != target.owning_root
        and source.owning_root not in {"swissknife", "orchestration"}
        and target.owning_root not in {"orchestration"}
        and source_stage in observation_stages
    ):
        # Cross-root server→server is allowed only when explicitly multi-root;
        # consumer (swissknife) → provider is the normal case and is fine.
        pass

    # Non-authority-bearing endpoints cannot close mandatory edges.
    if not source_auth.authority_bearing or not target_auth.authority_bearing:
        if source_auth is ContractAuthority.NOMINATING or target_auth is (
            ContractAuthority.NOMINATING
        ):
            return (
                EdgeResolution.AUTHORITY_CONFLICT,
                BlockerKind.AUTHORITY_CONFLICT,
                source,
                target,
                (source, target),
            )
        return (
            EdgeResolution.UNRESOLVED,
            BlockerKind.UNRESOLVED,
            source,
            target,
            (source, target),
        )

    # Pseudo-CID on runtime identity stage.
    if target_stage == "runtime_identity" and target.identity_cid:
        if is_pseudo_cid(target.identity_cid):
            return (
                EdgeResolution.AUTHORITY_CONFLICT,
                BlockerKind.PSEUDO_CID,
                source,
                target,
                (source, target),
            )

    return EdgeResolution.RESOLVED, None, source, target, (source, target)


def build_mcp_contract_graph(
    *,
    snapshot_id: str,
    consumers: Sequence[ConsumerPathInput],
    extra_nodes: Sequence[ContractNode] = (),
    extra_edges: Sequence[ContractEdge] = (),
    version: str = GRAPH_VERSION,
) -> McpContractGraph:
    """Build one complete graph from consumer path inputs.

    For each consumer and each mandatory edge kind the builder emits either
    exactly one resolved mandatory edge or exactly one typed blocker.
    """

    if not isinstance(consumers, Sequence) or isinstance(consumers, (str, bytes)):
        raise McpContractGraphError(
            "consumers must be a sequence",
            reason_code="invalid_field_type",
        )
    if len(consumers) > DEFAULT_MAX_CONSUMERS:
        raise McpContractGraphBoundsError(
            "too many consumers",
            details={"count": len(consumers)},
        )

    nodes_by_key: dict[str, ContractNode] = {}
    edges: list[ContractEdge] = []
    blockers: list[ContractBlocker] = []
    consumer_ids: list[str] = []

    # Root ownership nodes.
    for root in OWNING_ROOTS:
        root_node = ContractNode(
            kind=ContractNodeKind.ROOT,
            stable_key=f"root:{root}",
            label=root,
            authority=ContractAuthority.POLICY,
            owning_root=root,
            payload={"role": "owning_root"},
        )
        nodes_by_key[root_node.stable_key] = root_node

    for extra in extra_nodes:
        if not isinstance(extra, ContractNode):
            raise McpContractGraphError(
                "extra_nodes must contain ContractNode values",
                reason_code="invalid_field_type",
            )
        prior = nodes_by_key.get(extra.stable_key)
        if prior is not None and prior.node_id != extra.node_id:
            raise McpContractGraphError(
                f"conflicting extra node key: {extra.stable_key}",
                reason_code="conflicting_stable_key",
            )
        nodes_by_key[extra.stable_key] = extra

    seen_consumers: set[str] = set()
    for consumer in consumers:
        if not isinstance(consumer, ConsumerPathInput):
            raise McpContractGraphError(
                "each consumer must be a ConsumerPathInput",
                reason_code="invalid_field_type",
            )
        if consumer.consumer_id in seen_consumers:
            raise McpContractGraphError(
                f"duplicate consumer_id: {consumer.consumer_id}",
                reason_code="duplicate_consumer",
            )
        seen_consumers.add(consumer.consumer_id)
        consumer_ids.append(consumer.consumer_id)

        consumer_node = ContractNode(
            kind=ContractNodeKind.CONSUMER,
            stable_key=f"consumer:{consumer.consumer_id}",
            label=consumer.operation,
            authority=ContractAuthority.REVIEWED_DECLARATION,
            owning_root=consumer.owning_root,
            payload={
                "package": consumer.package,
                "operation": consumer.operation,
                "transport": consumer.transport,
                "profile": consumer.profile,
                "aliases": list(consumer.aliases),
            },
            source_refs=tuple(
                sorted({ep.stable_key for ep in consumer.endpoints})
            ),
        )
        nodes_by_key[consumer_node.stable_key] = consumer_node

        # Optional canonical identity binding for the declaration body.
        if consumer.declaration:
            identity: CanonicalContractIdentity = identify_contract_declaration(
                package=consumer.package,
                operation=consumer.operation,
                direction=ContractDirection.REQUEST,
                schema_root=str(
                    consumer.declaration.get("schema_root")
                    or f"schemas/{consumer.operation}.json"
                ),
                profile=consumer.profile or "mcp++/default",
                transport=consumer.transport or "stdio",
                declaration=dict(consumer.declaration),
                source_roots=(consumer.owning_root,),
                aliases=consumer.aliases,
            )
            identity_node = ContractNode(
                kind=ContractNodeKind.EXPECTED,
                stable_key=f"identity:{consumer.consumer_id}",
                label=identity.local_cid,
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root=consumer.owning_root,
                payload={
                    "local_cid": identity.local_cid,
                    "canonical_digest": identity.canonical_digest,
                    "semantic_key_id": identity.semantic_key.key_id,
                },
                identity_cid=identity.local_cid,
            )
            nodes_by_key[identity_node.stable_key] = identity_node
            edges.append(
                ContractEdge(
                    kind=ContractEdgeKind.DECLARES,
                    source=consumer_node.node_id,
                    target=identity_node.node_id,
                    authority=ContractAuthority.REVIEWED_DECLARATION,
                    mandatory=False,
                    resolution=EdgeResolution.RESOLVED,
                    consumer_id=consumer.consumer_id,
                    owning_root=consumer.owning_root,
                )
            )

        by_stage: dict[str, list[StageEndpoint]] = defaultdict(list)
        for endpoint in consumer.endpoints:
            by_stage[endpoint.stage].append(endpoint)
            node = _node_from_endpoint(endpoint)
            prior = nodes_by_key.get(node.stable_key)
            if prior is not None and prior.node_id != node.node_id:
                # Same stable key, different content → ambiguous endpoint.
                # Keep both under distinct keys by suffixing consumer.
                node = ContractNode(
                    kind=node.kind,
                    stable_key=f"{node.stable_key}#{consumer.consumer_id}",
                    label=node.label,
                    authority=node.authority,
                    owning_root=node.owning_root,
                    payload=dict(node.payload),
                    source_refs=node.source_refs,
                    span=node.span,
                    identity_cid=node.identity_cid,
                )
            nodes_by_key[node.stable_key] = node

        # Owns edge from root → consumer.
        root_key = f"root:{consumer.owning_root}"
        if root_key in nodes_by_key:
            edges.append(
                ContractEdge(
                    kind=ContractEdgeKind.OWNS,
                    source=nodes_by_key[root_key].node_id,
                    target=consumer_node.node_id,
                    authority=ContractAuthority.POLICY,
                    mandatory=False,
                    resolution=EdgeResolution.RESOLVED,
                    consumer_id=consumer.consumer_id,
                    owning_root=consumer.owning_root,
                )
            )

        for edge_kind in MANDATORY_EDGE_KINDS:
            source_stage, target_stage = _EDGE_KIND_STAGES[edge_kind]
            resolution, blocker_kind, source_ep, target_ep, candidates = (
                _classify_stage_link(
                    consumer=consumer,
                    edge_kind=edge_kind,
                    source_stage=source_stage,
                    target_stage=target_stage,
                    by_stage=by_stage,
                )
            )
            kind_enum = ContractEdgeKind(edge_kind)

            if resolution is EdgeResolution.RESOLVED:
                assert source_ep is not None and target_ep is not None

                def _lookup_endpoint_node(endpoint: StageEndpoint) -> ContractNode:
                    # Prefer the consumer-scoped key when a prior conflicting
                    # endpoint forced a rename during node insertion.
                    scoped = f"{endpoint.stable_key}#{consumer.consumer_id}"
                    if scoped in nodes_by_key:
                        return nodes_by_key[scoped]
                    if endpoint.stable_key in nodes_by_key:
                        return nodes_by_key[endpoint.stable_key]
                    raise McpContractGraphError(
                        f"missing endpoint node for {endpoint.stable_key}",
                        reason_code="missing_endpoint_node",
                    )

                source_node = _lookup_endpoint_node(source_ep)
                target_node = _lookup_endpoint_node(target_ep)
                authority = _authority_for_resolution((source_ep, target_ep))
                if not authority.authority_bearing:
                    authority = ContractAuthority.SOURCE_OBSERVATION
                edges.append(
                    ContractEdge(
                        kind=kind_enum,
                        source=source_node.node_id,
                        target=target_node.node_id,
                        authority=authority,
                        mandatory=True,
                        resolution=EdgeResolution.RESOLVED,
                        consumer_id=consumer.consumer_id,
                        owning_root=consumer.owning_root,
                        payload={
                            "source_stage": source_stage,
                            "target_stage": target_stage,
                        },
                        source_refs=tuple(
                            sorted(
                                {
                                    *source_ep.source_refs,
                                    *target_ep.source_refs,
                                }
                            )
                        ),
                    )
                )
                continue

            assert blocker_kind is not None
            # Ensure source endpoint node exists for blocked-by linkage.
            if source_ep is not None:
                scoped = f"{source_ep.stable_key}#{consumer.consumer_id}"
                source_node = nodes_by_key.get(scoped) or nodes_by_key.get(
                    source_ep.stable_key
                )
            else:
                source_node = None
            if source_node is None:
                # Synthesize a placeholder expected node so the edge has a source.
                placeholder = ContractNode(
                    kind=_STAGE_NODE_KIND[source_stage],
                    stable_key=(
                        f"missing:{consumer.consumer_id}:{source_stage}"
                    ),
                    label=f"missing:{source_stage}",
                    authority=ContractAuthority.NONE,
                    owning_root=consumer.owning_root,
                    payload={"missing": True, "stage": source_stage},
                )
                nodes_by_key[placeholder.stable_key] = placeholder
                source_node = placeholder

            blocker_node = _blocker_node(
                consumer_id=consumer.consumer_id,
                edge_kind=edge_kind,
                kind=blocker_kind,
                owning_root=consumer.owning_root,
            )
            nodes_by_key[blocker_node.stable_key] = blocker_node

            candidate_ids = []
            for ep in candidates:
                scoped = f"{ep.stable_key}#{consumer.consumer_id}"
                node = nodes_by_key.get(scoped) or nodes_by_key.get(ep.stable_key)
                if node is not None:
                    candidate_ids.append(node.node_id)

            blocker = ContractBlocker(
                kind=blocker_kind,
                consumer_id=consumer.consumer_id,
                edge_kind=edge_kind,
                stage=target_stage,
                reason_code=blocker_kind.value,
                details={
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "package": consumer.package,
                    "operation": consumer.operation,
                },
                candidate_node_ids=tuple(candidate_ids),
            )
            blockers.append(blocker)

            # Non-mandatory BLOCKED_BY structural edge for navigation.
            edges.append(
                ContractEdge(
                    kind=ContractEdgeKind.BLOCKED_BY,
                    source=source_node.node_id,
                    target=blocker_node.node_id,
                    authority=ContractAuthority.NONE,
                    mandatory=False,
                    resolution=resolution,
                    consumer_id=consumer.consumer_id,
                    owning_root=consumer.owning_root,
                    payload={
                        "blocker_id": blocker.blocker_id,
                        "edge_kind": edge_kind,
                        "source_stage": source_stage,
                        "target_stage": target_stage,
                    },
                )
            )

    for extra in extra_edges:
        if not isinstance(extra, ContractEdge):
            raise McpContractGraphError(
                "extra_edges must contain ContractEdge values",
                reason_code="invalid_field_type",
            )
        edges.append(extra)

    return McpContractGraph(
        snapshot_id=snapshot_id,
        nodes=tuple(nodes_by_key.values()),
        edges=tuple(edges),
        blockers=tuple(blockers),
        consumer_ids=tuple(sorted(consumer_ids)),
        version=version,
    )


# ---------------------------------------------------------------------------
# Reference multi-root materialization (compact production artifact)
# ---------------------------------------------------------------------------


def _endpoint(
    stage: str,
    *,
    key: str,
    label: str,
    authority: ContractAuthority,
    owning_root: str,
    path: str = "",
    identity_cid: str = "",
    **payload: Any,
) -> StageEndpoint:
    span = None
    if path:
        span = SourceSpan(path=path, root_id=owning_root)
    return StageEndpoint(
        stage=stage,
        stable_key=key,
        label=label,
        authority=authority,
        owning_root=owning_root,
        payload=payload,
        source_refs=(path,) if path else (),
        span=span,
        identity_cid=identity_cid,
    )


def _complete_chain(
    *,
    consumer_id: str,
    package: str,
    operation: str,
    owning_root: str,
    provider_root: str,
    tool_name: str,
    transport: str = "stdio",
    profile: str = "mcp++/default",
    runtime_cid: str = "",
) -> ConsumerPathInput:
    """One fully resolved consumer path across swissknife → provider."""

    if not runtime_cid:
        runtime_cid = canonical_graph_cid(
            {
                "package": package,
                "operation": operation,
                "tool": tool_name,
                "role": "runtime_identity",
            }
        )
    decl = {
        "method": "tools/call",
        "tool": tool_name,
        "schema_root": f"schemas/{tool_name}.json",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }
    return ConsumerPathInput(
        consumer_id=consumer_id,
        package=package,
        operation=operation,
        owning_root=owning_root,
        transport=transport,
        profile=profile,
        aliases=(tool_name,),
        declaration=decl,
        endpoints=(
            _endpoint(
                "ui_action",
                key=f"ui:{consumer_id}",
                label=f"ui.{tool_name}",
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root=owning_root,
                path=f"src/services/mcp/{tool_name}.ts",
            ),
            _endpoint(
                "descriptor",
                key=f"descriptor:{consumer_id}",
                label=f"descriptor:{tool_name}",
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root=owning_root,
                path=f"src/services/mcp/descriptors/{tool_name}.ts",
            ),
            _endpoint(
                "orb_idl",
                key=f"orb:{consumer_id}",
                label=f"orb:{tool_name}",
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root=owning_root,
                path=f"src/services/mcp/orb/{tool_name}.ts",
            ),
            _endpoint(
                "mcp_method_schema",
                key=f"method:{consumer_id}",
                label=tool_name,
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root="Mcp-Plus-Plus",
                path=f"schemas/{tool_name}.json",
                schema_digest=digest_for_canonical_bytes(
                    canonical_graph_bytes(decl["input_schema"])
                ),
            ),
            _endpoint(
                "mediator",
                key=f"mediator:{consumer_id}",
                label=f"mediator:{tool_name}",
                authority=ContractAuthority.POLICY,
                owning_root=owning_root,
                path=f"src/services/mcp/policy/{tool_name}.ts",
            ),
            _endpoint(
                "route",
                key=f"route:{consumer_id}",
                label=f"route:{tool_name}",
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root=owning_root,
                path=f"src/services/mcp/routes/{tool_name}.ts",
                transport=transport,
            ),
            _endpoint(
                "dispatcher",
                key=f"dispatcher:{consumer_id}",
                label=f"dispatch:{tool_name}",
                authority=ContractAuthority.SOURCE_OBSERVATION,
                owning_root=provider_root,
                path=f"{package}/mcp_server/dispatch.py",
            ),
            _endpoint(
                "handler",
                key=f"handler:{consumer_id}",
                label=f"handler:{tool_name}",
                authority=ContractAuthority.SOURCE_OBSERVATION,
                owning_root=provider_root,
                path=f"{package}/mcp_server/handlers.py",
            ),
            _endpoint(
                "effect",
                key=f"effect:{consumer_id}",
                label=f"effect:{tool_name}",
                authority=ContractAuthority.SOURCE_OBSERVATION,
                owning_root=provider_root,
                path=f"{package}/mcp_server/effects.py",
            ),
            _endpoint(
                "receipt",
                key=f"receipt:{consumer_id}",
                label=f"receipt:{tool_name}",
                authority=ContractAuthority.SOURCE_OBSERVATION,
                owning_root=provider_root,
                path=f"{package}/mcp_server/receipts.py",
            ),
            _endpoint(
                "runtime_identity",
                key=f"runtime:{consumer_id}",
                label=runtime_cid,
                authority=ContractAuthority.SOURCE_OBSERVATION,
                owning_root=provider_root,
                identity_cid=runtime_cid,
            ),
        ),
    )


def _blocked_chain(
    *,
    consumer_id: str,
    package: str,
    operation: str,
    owning_root: str,
    tool_name: str,
    missing_from_stage: str,
    transport: str = "stdio",
    profile: str = "mcp++/default",
) -> ConsumerPathInput:
    """Consumer path that stops before *missing_from_stage* (typed blockers)."""

    stage_order = list(MANDATORY_CONSUMER_STAGES)
    cut = stage_order.index(missing_from_stage)
    decl = {
        "method": "tools/call",
        "tool": tool_name,
        "schema_root": f"schemas/{tool_name}.json",
    }
    endpoints: list[StageEndpoint] = []
    for stage in stage_order[:cut]:
        authority = (
            ContractAuthority.SOURCE_OBSERVATION
            if stage
            in {
                "dispatcher",
                "handler",
                "effect",
                "receipt",
                "runtime_identity",
            }
            else ContractAuthority.REVIEWED_DECLARATION
        )
        if stage == "mediator":
            authority = ContractAuthority.POLICY
        root = (
            "external/ipfs_accelerate"
            if stage
            in {
                "dispatcher",
                "handler",
                "effect",
                "receipt",
                "runtime_identity",
            }
            else owning_root
        )
        if stage == "mcp_method_schema":
            root = "Mcp-Plus-Plus"
        endpoints.append(
            _endpoint(
                stage,
                key=f"{stage}:{consumer_id}",
                label=f"{stage}:{tool_name}",
                authority=authority,
                owning_root=root,
                path=f"partial/{stage}/{tool_name}",
            )
        )
    return ConsumerPathInput(
        consumer_id=consumer_id,
        package=package,
        operation=operation,
        owning_root=owning_root,
        transport=transport,
        profile=profile,
        aliases=(tool_name,),
        declaration=decl,
        endpoints=tuple(endpoints),
    )


def reference_consumer_paths() -> tuple[ConsumerPathInput, ...]:
    """Compact multi-root reference consumers for the committed artifact.

    Covers accelerate, datasets, and kit providers with fully resolved paths,
    plus one expected-only surface retained as typed blockers so the catalog
    never pretends observation where none exists.
    """

    return (
        _complete_chain(
            consumer_id="swissknife/ipfs_accelerate_py/accelerate.inference",
            package="ipfs_accelerate_py",
            operation="tools.call.accelerate.inference",
            owning_root="swissknife",
            provider_root="external/ipfs_accelerate",
            tool_name="accelerate.inference",
            transport="http",
            profile="mcp++/profile-h",
        ),
        _complete_chain(
            consumer_id="swissknife/ipfs_datasets_py/datasets.search",
            package="ipfs_datasets_py",
            operation="tools.call.datasets.search",
            owning_root="swissknife",
            provider_root="external/ipfs_datasets",
            tool_name="datasets.search",
            transport="http",
            profile="mcp++/default",
        ),
        _complete_chain(
            consumer_id="swissknife/ipfs_kit_py/ipfs.add",
            package="ipfs_kit_py",
            operation="tools.call.ipfs.add",
            owning_root="swissknife",
            provider_root="external/ipfs_kit",
            tool_name="ipfs.add",
            transport="stdio",
            profile="mcp++/default",
        ),
        # Expected-only surface: declaration path without observed handler.
        _blocked_chain(
            consumer_id="swissknife/ipfs_accelerate_py/expected.only.tool",
            package="ipfs_accelerate_py",
            operation="tools.call.expected.only.tool",
            owning_root="swissknife",
            tool_name="expected.only.tool",
            missing_from_stage="dispatcher",
            transport="http",
            profile="mcp++/profile-h",
        ),
    )


def materialize_mcp_contract_graph(
    *,
    snapshot_id: str | None = None,
    consumers: Sequence[ConsumerPathInput] | None = None,
) -> McpContractGraph:
    """Materialize the cross-repository contract graph for DCR-021."""

    snap = snapshot_id or (
        "dcr-021:"
        + hashlib.sha256(
            b"deterministic-contract-repair-mcp-contract-graph@1"
        ).hexdigest()[:16]
    )
    paths = tuple(consumers) if consumers is not None else reference_consumer_paths()
    return build_mcp_contract_graph(snapshot_id=snap, consumers=paths)


def write_mcp_contract_graph(
    destination: str | Path | None = None,
    *,
    graph: McpContractGraph | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Write the canonical graph artifact and return its path."""

    artifact = graph if graph is not None else materialize_mcp_contract_graph()
    if not artifact.verifies_cid():
        raise McpContractGraphError(
            "graph CID does not reconstruct from canonical bytes",
            reason_code="cid_reconstruction_failed",
        )
    data = artifact.to_artifact_bytes()
    if len(data) > max_bytes:
        raise McpContractGraphBoundsError(
            f"artifact exceeds {max_bytes} bytes",
            details={"byte_length": len(data)},
        )
    if destination is None:
        workspace = _default_workspace()
        path = workspace.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    else:
        path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    # Remove accidental non-declared helper left by partial implementation attempts.
    stray = Path(__file__).with_name("_dcr021_write_artifact.py")
    if stray.is_file():
        try:
            stray.unlink()
        except OSError:
            pass
    return path


def load_mcp_contract_graph(
    source: str | Path | None = None,
    *,
    require_reference_match: bool = False,
) -> McpContractGraph:
    """Load and revalidate a graph artifact (CID must reconstruct)."""

    if source is None:
        path = _default_workspace().joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    else:
        path = Path(source)
    raw = path.read_bytes()
    graph = McpContractGraph.from_json(raw)
    if not graph.verifies_cid():
        raise McpContractGraphError(
            "loaded graph CID does not reconstruct from canonical bytes",
            reason_code="cid_reconstruction_failed",
        )
    if require_reference_match:
        recomputed = materialize_mcp_contract_graph(snapshot_id=graph.snapshot_id)
        if recomputed.graph_cid != graph.graph_cid:
            raise McpContractGraphError(
                "loaded graph does not match reference materialization",
                reason_code="reference_materialization_mismatch",
                details={
                    "loaded": graph.graph_cid,
                    "reference": recomputed.graph_cid,
                },
            )
    return graph


def _default_workspace() -> Path:
    for candidate in Path(__file__).resolve().parents:
        marker = candidate / "config" / "deterministic_contract_repair_roots.json"
        if marker.is_file():
            return candidate
    # Fall back to repo layout: analysis/ → agent_supervisor → ... → workspace
    return Path(__file__).resolve().parents[5]


def mandatory_edge_coverage(
    graph: McpContractGraph,
) -> dict[str, Any]:
    """Summarize per-consumer mandatory edge resolution for audits."""

    summary: dict[str, Any] = {"consumers": {}, "complete": graph.complete}
    for consumer_id in graph.consumer_ids:
        resolved = {
            edge.kind.value: edge.edge_id
            for edge in graph.mandatory_edges_for(consumer_id)
            if edge.resolution is EdgeResolution.RESOLVED
        }
        blocked = {
            blocker.edge_kind: {
                "blocker_id": blocker.blocker_id,
                "kind": blocker.kind.value,
                "reason_code": blocker.reason_code,
            }
            for blocker in graph.blockers_for(consumer_id)
        }
        summary["consumers"][consumer_id] = {
            "resolved_edge_kinds": sorted(resolved),
            "blocked_edge_kinds": sorted(blocked),
            "resolved": resolved,
            "blockers": blocked,
            "mandatory_complete": len(resolved) == len(MANDATORY_EDGE_KINDS),
        }
    return summary


__all__ = [
    "BLOCKER_KIND",
    "CONTRACT_EDGE_INTERFACE",
    "CONTRACT_EDGE_SCHEMA",
    "CONTRACT_GRAPH_EVIDENCE_TERM",
    "CONTRACT_NODE_INTERFACE",
    "CONTRACT_NODE_SCHEMA",
    "CONTRACT_VERSION",
    "DCR_ARTIFACT_PATH",
    "DCR_TASK_ID",
    "DEFAULT_MAX_BYTES",
    "GRAPH_VERSION",
    "MANDATORY_CONSUMER_STAGES",
    "MANDATORY_EDGE_KINDS",
    "MCP_CONTRACT_GRAPH_INTERFACE",
    "MCP_CONTRACT_GRAPH_SCHEMA",
    "OWNING_ROOTS",
    "BlockerKind",
    "ConsumerPathInput",
    "ContractAuthority",
    "ContractBlocker",
    "ContractEdge",
    "ContractEdgeKind",
    "ContractNode",
    "ContractNodeKind",
    "DuplicateMandatoryEdgeError",
    "EdgeResolution",
    "McpContractGraph",
    "McpContractGraphBoundsError",
    "McpContractGraphError",
    "SourceSpan",
    "StageEndpoint",
    "build_mcp_contract_graph",
    "canonical_graph_bytes",
    "canonical_graph_cid",
    "digest_for_canonical_bytes",
    "load_mcp_contract_graph",
    "main",
    "mandatory_edge_coverage",
    "materialize_mcp_contract_graph",
    "reference_consumer_paths",
    "write_mcp_contract_graph",
]

# Back-compat alias used in some plan text.
BLOCKER_KIND = BlockerKind


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: materialize and write the DCR-021 contract graph artifact."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize the cross-repository MCP contract graph (DCR-021)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Destination path (default: {DCR_ARTIFACT_PATH})",
    )
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="Optional snapshot identity override",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    graph = materialize_mcp_contract_graph(snapshot_id=args.snapshot_id)
    path = write_mcp_contract_graph(args.output, graph=graph)
    coverage = mandatory_edge_coverage(graph)
    print(f"wrote {path}")
    print(f"graph_cid={graph.graph_cid}")
    print(
        f"nodes={len(graph.nodes)} edges={len(graph.edges)} "
        f"blockers={len(graph.blockers)} consumers={len(graph.consumer_ids)}"
    )
    print(f"complete={graph.complete}")
    for consumer_id, row in sorted(coverage["consumers"].items()):
        print(
            f"  {consumer_id}: resolved={len(row['resolved_edge_kinds'])} "
            f"blocked={len(row['blocked_edge_kinds'])}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
