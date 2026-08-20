"""Exact, fail-closed MCP++ invocation reachability.

Interface: ``McpInvocationTrace@1``

The tracer consumes a pinned :class:`SymbolicContractGraph` and computes
deterministic paths from a SwissKnife declaration/registry node to one or more
package handler or implementation nodes.  Only mandatory, authority-bearing,
allowlisted edges can prove reachability.  Candidate paths through unresolved
nodes remain visible, but can never become a reachability proof.

This module proves only structural reachability.  Argument, envelope, policy,
and behavioral properties are separate contract obligations.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .symbolic_contract_graph import (
    GRAPH_VERSION,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    SymbolicContractGraph,
    canonical_contract_graph_bytes,
)


MCP_INVOCATION_TRACE_INTERFACE: Final = "McpInvocationTrace@1"
MCP_INVOCATION_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-invocation-trace@1"
)
MCP_INVOCATION_PATH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-invocation-path@1"
)
MCP_INVOCATION_SEGMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-invocation-segment@1"
)
MCP_INVOCATION_TRACE_VERSION: Final = "1"

DEFAULT_MAX_TRACE_DEPTH: Final = 64
DEFAULT_MAX_TRACE_PATHS: Final = 4_096
DEFAULT_MAX_TRACE_STATES: Final = 100_000
HARD_MAX_TRACE_DEPTH: Final = 512
HARD_MAX_TRACE_PATHS: Final = 65_536
HARD_MAX_TRACE_STATES: Final = 1_000_000

# These are the structural relationships that may constitute invocation.
# RELATED_TO and evidence/schema-only edges are deliberately absent.
DEFAULT_INVOCATION_EDGE_KINDS: Final[tuple[ContractEdgeKind, ...]] = (
    ContractEdgeKind.DECLARES,
    ContractEdgeKind.REGISTERS,
    ContractEdgeKind.DISPATCHES_TO,
    ContractEdgeKind.HANDLED_BY,
    ContractEdgeKind.IMPLEMENTS,
    ContractEdgeKind.CALLS,
    ContractEdgeKind.TRANSPORTED_BY,
    ContractEdgeKind.ENFORCED_BY,
    ContractEdgeKind.DEPENDS_ON,
)

_DYNAMIC_WORDS: Final[frozenset[str]] = frozenset(
    {
        "ambiguous",
        "computed",
        "dynamic",
        "indeterminate",
        "unknown",
        "unresolved",
    }
)
_COMPATIBILITY_WORDS: Final[tuple[str, ...]] = (
    "compat",
    "legacy",
    "shim",
    "bypass",
    "/api/v0/",
    "direct_fetch",
    "direct_rest",
    "compatibility_route",
)


class McpInvocationTraceError(ValueError):
    """A trace request or serialized trace is malformed."""


class InvocationTerminalState(str, Enum):
    """The closed terminal-state set for one declared operation."""

    REACHABLE = "reachable"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"


class InvocationPathClass(str, Enum):
    """Keep canonical/direct and compatibility routes distinguishable."""

    DIRECT = "direct"
    COMPATIBILITY = "compatibility"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise McpInvocationTraceError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise McpInvocationTraceError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise McpInvocationTraceError(f"{name} is required")
    if len(value.encode("utf-8")) > 16_384:
        raise McpInvocationTraceError(f"{name} is oversized")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise McpInvocationTraceError(f"{name} must be a sequence")
    return tuple(
        sorted({_text(str(item), name) for item in value})
    )


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise McpInvocationTraceError("trace value exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise McpInvocationTraceError(
            "floating values are not canonical trace data"
        )
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(
            isinstance(key, str) for key in value
        ):
            raise McpInvocationTraceError(
                "trace objects require at most 1024 string keys"
            )
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > HARD_MAX_TRACE_STATES:
            raise McpInvocationTraceError("trace sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise McpInvocationTraceError(
        f"unsupported trace value: {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise McpInvocationTraceError(f"{name} must be an object")
    return MappingProxyType(_plain(value))


def _canonical(value: Any) -> bytes:
    return canonical_contract_graph_bytes(_plain(value))


def _cid(value: Any) -> str:
    return content_identity(_plain(value))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise McpInvocationTraceError(
            f"unknown {name}: {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class TraceBounds:
    """Deterministic resource bounds for simple-path enumeration."""

    max_depth: int = DEFAULT_MAX_TRACE_DEPTH
    max_paths: int = DEFAULT_MAX_TRACE_PATHS
    max_states: int = DEFAULT_MAX_TRACE_STATES

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_depth", HARD_MAX_TRACE_DEPTH),
            ("max_paths", HARD_MAX_TRACE_PATHS),
            ("max_states", HARD_MAX_TRACE_STATES),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise McpInvocationTraceError(
                    f"{name} must be between 1 and {maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_depth": self.max_depth,
            "max_paths": self.max_paths,
            "max_states": self.max_states,
        }

    @classmethod
    def from_value(
        cls, value: "TraceBounds | Mapping[str, Any] | None"
    ) -> "TraceBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise McpInvocationTraceError("bounds must be an object")
        return cls(
            max_depth=int(value.get("max_depth", DEFAULT_MAX_TRACE_DEPTH)),
            max_paths=int(value.get("max_paths", DEFAULT_MAX_TRACE_PATHS)),
            max_states=int(value.get("max_states", DEFAULT_MAX_TRACE_STATES)),
        )


def _span_candidates(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Extract exact span objects without inventing missing coordinates."""

    if not isinstance(value, Mapping):
        return ()
    raw: list[Mapping[str, Any]] = []
    for key in ("source_span", "span", "registration_span"):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            raw.append(candidate)
    many = value.get("source_spans")
    if isinstance(many, Sequence) and not isinstance(many, (str, bytes)):
        raw.extend(item for item in many if isinstance(item, Mapping))
    handler = value.get("handler")
    if isinstance(handler, Mapping):
        candidate = handler.get("span")
        if isinstance(candidate, Mapping):
            raw.append(candidate)
    unique: dict[bytes, Mapping[str, Any]] = {}
    for span in raw:
        normalized = _mapping(span, "source span")
        path = normalized.get("path")
        if not isinstance(path, str) or not path:
            continue
        unique[_canonical(normalized)] = normalized
    return tuple(unique[key] for key in sorted(unique))


def _source_evidence(
    edge: ContractGraphEdge,
    source: ContractGraphNode,
    target: ContractGraphNode,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    ids = set(edge.source_refs)
    spans = list(_span_candidates(edge.payload))
    # Projection code may attach a call's coordinates to its endpoint nodes.
    # Retain these as a fallback while keeping the edge identity explicit.
    if not spans:
        spans.extend(_span_candidates(source.payload))
        spans.extend(_span_candidates(target.payload))
    if not ids:
        ids.update(source.source_refs)
        ids.update(target.source_refs)
    for span in spans:
        digest = span.get("source_sha256")
        if isinstance(digest, str) and digest:
            ids.add(digest)
    unique_spans = {_canonical(span): span for span in spans}
    return (
        tuple(sorted(ids)),
        tuple(unique_spans[key] for key in sorted(unique_spans)),
    )


def _dynamic_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _DYNAMIC_WORDS
    return False


def _node_is_dynamic(node: ContractGraphNode) -> bool:
    if node.kind is ContractNodeKind.UNRESOLVED:
        return True
    if node.stable_key.lower().startswith(("unresolved:", "dynamic:")):
        return True
    payload = node.payload
    return bool(
        payload.get("dynamic") is True
        or payload.get("unresolved") is True
        or payload.get("unresolved_id")
        or _dynamic_value(payload.get("state"))
        or _dynamic_value(payload.get("resolution_state"))
        or _dynamic_value(payload.get("status"))
    )


def _edge_is_dynamic(edge: ContractGraphEdge) -> bool:
    payload = edge.payload
    return bool(
        payload.get("dynamic") is True
        or payload.get("unresolved") is True
        or payload.get("unresolved_id")
        or _dynamic_value(payload.get("state"))
        or _dynamic_value(payload.get("resolution_state"))
        or _dynamic_value(payload.get("target_state"))
    )


def _edge_is_compatibility(edge: ContractGraphEdge) -> bool:
    payload = edge.payload
    if payload.get("compatibility") is True or payload.get(
        "bypass_candidate"
    ) is True:
        return True
    values = (
        payload.get("path_class"),
        payload.get("path_kind"),
        payload.get("route_kind"),
        payload.get("kind"),
        payload.get("target"),
    )
    return any(
        marker in str(value or "").lower()
        for marker in _COMPATIBILITY_WORDS
        for value in values
    )


@dataclass(frozen=True, slots=True)
class InvocationPathSegment:
    """One exact graph edge plus its source provenance."""

    edge_id: str
    edge_kind: ContractEdgeKind
    source_node_id: str
    target_node_id: str
    source_ids: tuple[str, ...]
    source_spans: tuple[Mapping[str, Any], ...]
    dynamic: bool = False
    compatibility: bool = False
    segment_id: str = ""

    def __post_init__(self) -> None:
        for name in ("edge_id", "source_node_id", "target_node_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "edge_kind",
            _enum(self.edge_kind, ContractEdgeKind, "edge kind"),
        )
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, "source_ids")
        )
        spans = tuple(
            _mapping(item, "source span") for item in self.source_spans
        )
        object.__setattr__(
            self,
            "source_spans",
            tuple(
                {_canonical(item): item for item in spans}[key]
                for key in sorted({_canonical(item) for item in spans})
            ),
        )
        object.__setattr__(self, "dynamic", bool(self.dynamic))
        object.__setattr__(self, "compatibility", bool(self.compatibility))
        expected = _cid(self._identity_payload())
        claimed = str(self.segment_id or "")
        if claimed and claimed != expected:
            raise McpInvocationTraceError("segment identity mismatch")
        object.__setattr__(self, "segment_id", expected)

    @property
    def has_exact_source(self) -> bool:
        return bool(self.source_ids and self.source_spans)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_INVOCATION_SEGMENT_SCHEMA,
            "edge_id": self.edge_id,
            "edge_kind": self.edge_kind.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "source_ids": list(self.source_ids),
            "source_spans": [dict(item) for item in self.source_spans],
            "dynamic": self.dynamic,
            "compatibility": self.compatibility,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, **self._identity_payload()}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "InvocationPathSegment":
        if value.get("schema") not in (
            None,
            MCP_INVOCATION_SEGMENT_SCHEMA,
        ):
            raise McpInvocationTraceError("unsupported segment schema")
        return cls(
            edge_id=str(value.get("edge_id") or ""),
            edge_kind=value.get("edge_kind", ""),
            source_node_id=str(value.get("source_node_id") or ""),
            target_node_id=str(value.get("target_node_id") or ""),
            source_ids=tuple(value.get("source_ids") or ()),
            source_spans=tuple(value.get("source_spans") or ()),
            dynamic=value.get("dynamic", False),
            compatibility=value.get("compatibility", False),
            segment_id=str(value.get("segment_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class InvocationPath:
    """A deterministic simple path through the pinned graph."""

    node_ids: tuple[str, ...]
    segments: tuple[InvocationPathSegment, ...]
    path_class: InvocationPathClass
    reaches_target: bool
    path_id: str = ""

    def __post_init__(self) -> None:
        nodes = tuple(
            _text(str(item), "path node_id") for item in self.node_ids
        )
        segments = tuple(
            item
            if isinstance(item, InvocationPathSegment)
            else InvocationPathSegment.from_dict(item)
            for item in self.segments
        )
        if not nodes:
            raise McpInvocationTraceError("path requires at least one node")
        if len(segments) != len(nodes) - 1:
            raise McpInvocationTraceError(
                "path segments must connect every adjacent node"
            )
        for index, segment in enumerate(segments):
            if (
                segment.source_node_id != nodes[index]
                or segment.target_node_id != nodes[index + 1]
            ):
                raise McpInvocationTraceError(
                    "path segment endpoints are not contiguous"
                )
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(
            self,
            "path_class",
            _enum(self.path_class, InvocationPathClass, "path class"),
        )
        expected_class = (
            InvocationPathClass.COMPATIBILITY
            if any(segment.compatibility for segment in segments)
            else InvocationPathClass.DIRECT
        )
        if self.path_class is not expected_class:
            raise McpInvocationTraceError(
                "path class does not match its segments"
            )
        object.__setattr__(self, "reaches_target", bool(self.reaches_target))
        expected = _cid(self._identity_payload())
        claimed = str(self.path_id or "")
        if claimed and claimed != expected:
            raise McpInvocationTraceError("path identity mismatch")
        object.__setattr__(self, "path_id", expected)

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(segment.edge_id for segment in self.segments)

    @property
    def source_spans(self) -> tuple[Mapping[str, Any], ...]:
        values = {
            _canonical(span): span
            for segment in self.segments
            for span in segment.source_spans
        }
        return tuple(values[key] for key in sorted(values))

    @property
    def dynamic(self) -> bool:
        return any(segment.dynamic for segment in self.segments)

    @property
    def proof_eligible(self) -> bool:
        return bool(
            self.reaches_target
            and self.segments
            and not self.dynamic
            and all(segment.has_exact_source for segment in self.segments)
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_INVOCATION_PATH_SCHEMA,
            "node_ids": list(self.node_ids),
            "segments": [item.to_dict() for item in self.segments],
            "path_class": self.path_class.value,
            "reaches_target": self.reaches_target,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "edge_ids": list(self.edge_ids),
            "source_spans": [dict(item) for item in self.source_spans],
            "dynamic": self.dynamic,
            "proof_eligible": self.proof_eligible,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvocationPath":
        if value.get("schema") not in (None, MCP_INVOCATION_PATH_SCHEMA):
            raise McpInvocationTraceError("unsupported path schema")
        result = cls(
            node_ids=tuple(value.get("node_ids") or ()),
            segments=tuple(
                InvocationPathSegment.from_dict(item)
                for item in value.get("segments") or ()
            ),
            path_class=value.get("path_class", ""),
            reaches_target=value.get("reaches_target", False),
            path_id=str(value.get("path_id") or ""),
        )
        if "edge_ids" in value and tuple(value["edge_ids"]) != result.edge_ids:
            raise McpInvocationTraceError("path edge_ids mismatch")
        if "dynamic" in value and bool(value["dynamic"]) != result.dynamic:
            raise McpInvocationTraceError("path dynamic claim mismatch")
        if (
            "proof_eligible" in value
            and bool(value["proof_eligible"]) != result.proof_eligible
        ):
            raise McpInvocationTraceError(
                "path proof_eligible claim mismatch"
            )
        return result


@dataclass(frozen=True, slots=True)
class McpInvocationTrace:
    """One operation with exactly one closed terminal state."""

    operation_id: str
    graph_root: str
    snapshot_id: str
    source_node_id: str
    target_node_ids: tuple[str, ...]
    terminal_state: InvocationTerminalState
    reason_code: str
    direct_paths: tuple[InvocationPath, ...] = ()
    compatibility_paths: tuple[InvocationPath, ...] = ()
    unresolved_paths: tuple[InvocationPath, ...] = ()
    complete: bool = True
    bounds: TraceBounds = field(default_factory=TraceBounds)
    trace_id: str = ""
    version: str = MCP_INVOCATION_TRACE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "graph_root",
            "snapshot_id",
            "source_node_id",
            "reason_code",
            "version",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "target_node_ids",
            _strings(self.target_node_ids, "target_node_ids"),
        )
        object.__setattr__(
            self,
            "terminal_state",
            _enum(
                self.terminal_state,
                InvocationTerminalState,
                "terminal state",
            ),
        )
        for name, expected_class in (
            ("direct_paths", InvocationPathClass.DIRECT),
            ("compatibility_paths", InvocationPathClass.COMPATIBILITY),
            ("unresolved_paths", None),
        ):
            paths = tuple(
                item
                if isinstance(item, InvocationPath)
                else InvocationPath.from_dict(item)
                for item in getattr(self, name)
            )
            if expected_class is not None and any(
                item.path_class is not expected_class for item in paths
            ):
                raise McpInvocationTraceError(
                    f"{name} contains the wrong path class"
                )
            if name == "unresolved_paths" and any(
                not item.dynamic for item in paths
            ):
                raise McpInvocationTraceError(
                    "unresolved_paths must contain a dynamic segment"
                )
            unique = {item.path_id: item for item in paths}
            object.__setattr__(
                self, name, tuple(unique[key] for key in sorted(unique))
            )
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(
            self, "bounds", TraceBounds.from_value(self.bounds)
        )
        proof_paths = self.proved_paths
        if (
            self.terminal_state is InvocationTerminalState.REACHABLE
            and not proof_paths
        ):
            raise McpInvocationTraceError(
                "reachable trace requires a proof-eligible path"
            )
        if (
            self.terminal_state is not InvocationTerminalState.REACHABLE
            and proof_paths
            and self.terminal_state
            not in {
                InvocationTerminalState.AMBIGUOUS,
                InvocationTerminalState.NOT_MEASURED,
            }
        ):
            raise McpInvocationTraceError(
                "non-reachability terminal state contradicts a proved path"
            )
        expected = _cid(self._identity_payload())
        claimed = str(self.trace_id or "")
        if claimed and claimed != expected:
            raise McpInvocationTraceError("trace identity mismatch")
        object.__setattr__(self, "trace_id", expected)

    @property
    def state(self) -> InvocationTerminalState:
        return self.terminal_state

    @property
    def all_paths(self) -> tuple[InvocationPath, ...]:
        values = {
            item.path_id: item
            for item in (
                *self.direct_paths,
                *self.compatibility_paths,
                *self.unresolved_paths,
            )
        }
        return tuple(values[key] for key in sorted(values))

    @property
    def proved_paths(self) -> tuple[InvocationPath, ...]:
        return tuple(item for item in self.all_paths if item.proof_eligible)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_INVOCATION_TRACE_SCHEMA,
            "interface": MCP_INVOCATION_TRACE_INTERFACE,
            "version": self.version,
            "operation_id": self.operation_id,
            "graph_root": self.graph_root,
            "snapshot_id": self.snapshot_id,
            "source_node_id": self.source_node_id,
            "target_node_ids": list(self.target_node_ids),
            "terminal_state": self.terminal_state.value,
            "reason_code": self.reason_code,
            "direct_paths": [item.to_dict() for item in self.direct_paths],
            "compatibility_paths": [
                item.to_dict() for item in self.compatibility_paths
            ],
            "unresolved_paths": [
                item.to_dict() for item in self.unresolved_paths
            ],
            "complete": self.complete,
            "bounds": self.bounds.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path_count": len(self.all_paths),
            "proved_path_ids": [
                item.path_id for item in self.proved_paths
            ],
            **self._identity_payload(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McpInvocationTrace":
        if value.get("schema") not in (None, MCP_INVOCATION_TRACE_SCHEMA):
            raise McpInvocationTraceError("unsupported trace schema")
        if value.get("interface") not in (
            None,
            MCP_INVOCATION_TRACE_INTERFACE,
        ):
            raise McpInvocationTraceError("unsupported trace interface")
        result = cls(
            operation_id=str(value.get("operation_id") or ""),
            graph_root=str(value.get("graph_root") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            source_node_id=str(value.get("source_node_id") or ""),
            target_node_ids=tuple(value.get("target_node_ids") or ()),
            terminal_state=value.get("terminal_state", ""),
            reason_code=str(value.get("reason_code") or ""),
            direct_paths=tuple(
                InvocationPath.from_dict(item)
                for item in value.get("direct_paths") or ()
            ),
            compatibility_paths=tuple(
                InvocationPath.from_dict(item)
                for item in value.get("compatibility_paths") or ()
            ),
            unresolved_paths=tuple(
                InvocationPath.from_dict(item)
                for item in value.get("unresolved_paths") or ()
            ),
            complete=value.get("complete", False),
            bounds=TraceBounds.from_value(value.get("bounds")),
            trace_id=str(value.get("trace_id") or ""),
            version=str(
                value.get("version") or MCP_INVOCATION_TRACE_VERSION
            ),
        )
        if "path_count" in value and int(value["path_count"]) != len(
            result.all_paths
        ):
            raise McpInvocationTraceError("trace path_count mismatch")
        if "proved_path_ids" in value and tuple(
            value["proved_path_ids"]
        ) != tuple(item.path_id for item in result.proved_paths):
            raise McpInvocationTraceError(
                "trace proved_path_ids mismatch"
            )
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "McpInvocationTrace":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise McpInvocationTraceError("trace JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise McpInvocationTraceError(
                "trace JSON must contain an object"
            )
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class InvocationTraceRequest:
    """A normalized operation-to-handler reachability query."""

    operation_id: str
    source_node_id: str
    target_node_ids: tuple[str, ...]
    supported: bool = True
    measured: bool = True
    allowed_edge_kinds: tuple[ContractEdgeKind, ...] = (
        DEFAULT_INVOCATION_EDGE_KINDS
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _text(self.operation_id, "operation_id"),
        )
        object.__setattr__(
            self,
            "source_node_id",
            _text(self.source_node_id, "source_node_id"),
        )
        object.__setattr__(
            self,
            "target_node_ids",
            _strings(self.target_node_ids, "target_node_ids"),
        )
        object.__setattr__(self, "supported", bool(self.supported))
        object.__setattr__(self, "measured", bool(self.measured))
        kinds = tuple(
            sorted(
                {
                    _enum(item, ContractEdgeKind, "allowed edge kind")
                    for item in self.allowed_edge_kinds
                },
                key=lambda item: item.value,
            )
        )
        if self.supported and not kinds:
            raise McpInvocationTraceError(
                "supported trace requires allowlisted edge kinds"
            )
        object.__setattr__(self, "allowed_edge_kinds", kinds)


@dataclass(frozen=True, slots=True)
class _SearchResult:
    paths: tuple[InvocationPath, ...]
    unresolved_paths: tuple[InvocationPath, ...]
    complete: bool
    reason_code: str


class McpInvocationTracer:
    """Compute exact MCP++ invocation paths over ``SymbolicContractGraph@1``."""

    interface = MCP_INVOCATION_TRACE_INTERFACE

    def __init__(
        self,
        graph: SymbolicContractGraph,
        *,
        bounds: TraceBounds | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(graph, SymbolicContractGraph):
            raise McpInvocationTraceError(
                "graph must implement SymbolicContractGraph@1"
            )
        if graph.version != GRAPH_VERSION:
            raise McpInvocationTraceError(
                "unsupported SymbolicContractGraph version"
            )
        self.graph = graph
        self.bounds = TraceBounds.from_value(bounds)
        self._nodes = {item.node_id: item for item in graph.nodes}
        self._keys = {item.stable_key: item.node_id for item in graph.nodes}

    def _resolve_node(self, value: str, name: str) -> str:
        if value in self._nodes:
            return value
        if value in self._keys:
            return self._keys[value]
        raise McpInvocationTraceError(f"{name} is not in the pinned graph")

    def _segment(self, edge: ContractGraphEdge) -> InvocationPathSegment:
        source = self._nodes[edge.source]
        target = self._nodes[edge.target]
        source_ids, spans = _source_evidence(edge, source, target)
        return InvocationPathSegment(
            edge_id=edge.edge_id,
            edge_kind=edge.kind,
            source_node_id=edge.source,
            target_node_id=edge.target,
            source_ids=source_ids,
            source_spans=spans,
            dynamic=(
                _edge_is_dynamic(edge)
                or _node_is_dynamic(source)
                or _node_is_dynamic(target)
            ),
            compatibility=_edge_is_compatibility(edge),
        )

    @staticmethod
    def _path(
        nodes: tuple[str, ...],
        segments: tuple[InvocationPathSegment, ...],
        *,
        reaches_target: bool,
    ) -> InvocationPath:
        return InvocationPath(
            node_ids=nodes,
            segments=segments,
            path_class=(
                InvocationPathClass.COMPATIBILITY
                if any(item.compatibility for item in segments)
                else InvocationPathClass.DIRECT
            ),
            reaches_target=reaches_target,
        )

    def _search(
        self,
        source: str,
        targets: frozenset[str],
        allowed: frozenset[ContractEdgeKind],
    ) -> _SearchResult:
        adjacency: dict[str, list[ContractGraphEdge]] = {}
        for edge in self.graph.edges:
            if (
                edge.kind not in allowed
                or not edge.mandatory
                or not edge.authority.authority_bearing
                or edge.authority is ContractAuthority.CONTEXT_ONLY
            ):
                continue
            adjacency.setdefault(edge.source, []).append(edge)
        for edges in adjacency.values():
            edges.sort(
                key=lambda item: (
                    item.kind.value,
                    item.target,
                    item.edge_id,
                )
            )

        queue = deque([(source, (source,), ())])
        paths: dict[str, InvocationPath] = {}
        unresolved: dict[str, InvocationPath] = {}
        states = 0
        complete = True
        reason = "complete"
        while queue:
            current, node_path, segment_path = queue.popleft()
            states += 1
            if states > self.bounds.max_states:
                complete = False
                reason = "max_states_exceeded"
                break
            if current in targets and current != source:
                path = self._path(
                    node_path, segment_path, reaches_target=True
                )
                collection = unresolved if path.dynamic else paths
                collection[path.path_id] = path
                if len(paths) + len(unresolved) >= self.bounds.max_paths:
                    complete = False
                    reason = "max_paths_exceeded"
                    break
                continue
            if len(segment_path) >= self.bounds.max_depth:
                if adjacency.get(current):
                    complete = False
                    reason = "max_depth_exceeded"
                    break
                continue
            outgoing = adjacency.get(current, ())
            if not outgoing and segment_path and (
                _node_is_dynamic(self._nodes[current])
                or any(item.dynamic for item in segment_path)
            ):
                path = self._path(
                    node_path, segment_path, reaches_target=False
                )
                unresolved[path.path_id] = path
                continue
            for edge in outgoing:
                if edge.target in node_path:
                    # Simple paths are sufficient for reachability; revisiting
                    # a node can only add a cycle, never a new endpoint.
                    continue
                segment = self._segment(edge)
                next_nodes = (*node_path, edge.target)
                next_segments = (*segment_path, segment)
                if segment.dynamic and edge.target not in targets:
                    candidate = self._path(
                        next_nodes,
                        next_segments,
                        reaches_target=False,
                    )
                    unresolved[candidate.path_id] = candidate
                queue.append((edge.target, next_nodes, next_segments))
        return _SearchResult(
            paths=tuple(paths[key] for key in sorted(paths)),
            unresolved_paths=tuple(
                unresolved[key] for key in sorted(unresolved)
            ),
            complete=complete,
            reason_code=reason,
        )

    def trace(
        self,
        operation_id: str | InvocationTraceRequest,
        source_node_id: str = "",
        target_node_ids: Sequence[str] = (),
        *,
        supported: bool = True,
        measured: bool = True,
        allowed_edge_kinds: Iterable[ContractEdgeKind | str] = (
            DEFAULT_INVOCATION_EDGE_KINDS
        ),
    ) -> McpInvocationTrace:
        """Trace one operation and assign exactly one terminal state.

        Node identities may be content IDs or exact stable keys.  Targets are
        explicit: the tracer never guesses a handler from a similar label.
        """

        if isinstance(operation_id, InvocationTraceRequest):
            if source_node_id or target_node_ids:
                raise McpInvocationTraceError(
                    "request object cannot be combined with node arguments"
                )
            request = operation_id
        else:
            request = InvocationTraceRequest(
                operation_id=operation_id,
                source_node_id=source_node_id,
                target_node_ids=tuple(target_node_ids),
                supported=supported,
                measured=measured,
                allowed_edge_kinds=tuple(allowed_edge_kinds),
            )
        source = self._resolve_node(
            request.source_node_id, "source_node_id"
        )
        targets = tuple(
            sorted(
                {
                    self._resolve_node(item, "target_node_id")
                    for item in request.target_node_ids
                }
            )
        )

        if not request.supported:
            return self._result(
                request,
                source,
                targets,
                InvocationTerminalState.UNSUPPORTED,
                "operation_family_unsupported",
                (),
                (),
                True,
            )
        if not request.measured:
            return self._result(
                request,
                source,
                targets,
                InvocationTerminalState.NOT_MEASURED,
                "measurement_not_requested",
                (),
                (),
                True,
            )
        if not targets:
            return self._result(
                request,
                source,
                targets,
                InvocationTerminalState.UNSUPPORTED,
                "no_explicit_handler_target",
                (),
                (),
                True,
            )
        if not self.graph.complete:
            return self._result(
                request,
                source,
                targets,
                InvocationTerminalState.NOT_MEASURED,
                "incomplete_symbolic_contract_graph",
                (),
                (),
                False,
            )

        search = self._search(
            source,
            frozenset(targets),
            frozenset(request.allowed_edge_kinds),
        )
        resolved = search.paths
        proof_paths = tuple(
            item for item in resolved if item.proof_eligible
        )
        reached_targets = {
            item.node_ids[-1] for item in proof_paths
        }
        if not search.complete:
            state = InvocationTerminalState.NOT_MEASURED
            reason = search.reason_code
        elif len(reached_targets) > 1:
            state = InvocationTerminalState.AMBIGUOUS
            reason = "multiple_concrete_handlers"
        elif proof_paths:
            state = InvocationTerminalState.REACHABLE
            reason = "exact_authoritative_path"
        elif search.unresolved_paths:
            state = InvocationTerminalState.AMBIGUOUS
            reason = "unresolved_dynamic_segment"
        elif resolved:
            state = InvocationTerminalState.NOT_MEASURED
            reason = "path_source_provenance_incomplete"
        else:
            state = InvocationTerminalState.REFUTED
            reason = "no_allowlisted_authoritative_path"
        return self._result(
            request,
            source,
            targets,
            state,
            reason,
            resolved,
            search.unresolved_paths,
            search.complete,
        )

    def _result(
        self,
        request: InvocationTraceRequest,
        source: str,
        targets: tuple[str, ...],
        state: InvocationTerminalState,
        reason: str,
        paths: tuple[InvocationPath, ...],
        unresolved_paths: tuple[InvocationPath, ...],
        complete: bool,
    ) -> McpInvocationTrace:
        return McpInvocationTrace(
            operation_id=request.operation_id,
            graph_root=self.graph.graph_root,
            snapshot_id=self.graph.snapshot_id,
            source_node_id=source,
            target_node_ids=targets,
            terminal_state=state,
            reason_code=reason,
            direct_paths=tuple(
                item
                for item in paths
                if item.path_class is InvocationPathClass.DIRECT
            ),
            compatibility_paths=tuple(
                item
                for item in paths
                if item.path_class is InvocationPathClass.COMPATIBILITY
            ),
            unresolved_paths=unresolved_paths,
            complete=complete,
            bounds=self.bounds,
        )

    def trace_many(
        self, requests: Iterable[InvocationTraceRequest]
    ) -> tuple[McpInvocationTrace, ...]:
        """Trace requests deterministically, rejecting duplicate operations."""

        by_operation: dict[str, InvocationTraceRequest] = {}
        for request in requests:
            if not isinstance(request, InvocationTraceRequest):
                raise McpInvocationTraceError(
                    "trace_many requires InvocationTraceRequest values"
                )
            if request.operation_id in by_operation:
                raise McpInvocationTraceError(
                    f"duplicate operation_id: {request.operation_id}"
                )
            by_operation[request.operation_id] = request
        return tuple(
            self.trace(by_operation[key]) for key in sorted(by_operation)
        )


def compute_mcp_invocation_trace(
    graph: SymbolicContractGraph,
    operation_id: str,
    source_node_id: str,
    target_node_ids: Sequence[str],
    *,
    bounds: TraceBounds | Mapping[str, Any] | None = None,
    supported: bool = True,
    measured: bool = True,
    allowed_edge_kinds: Iterable[ContractEdgeKind | str] = (
        DEFAULT_INVOCATION_EDGE_KINDS
    ),
) -> McpInvocationTrace:
    """Convenience entry point for a single exact invocation trace."""

    return McpInvocationTracer(graph, bounds=bounds).trace(
        operation_id,
        source_node_id,
        target_node_ids,
        supported=supported,
        measured=measured,
        allowed_edge_kinds=allowed_edge_kinds,
    )


def compute_mcp_invocation_traces(
    graph: SymbolicContractGraph,
    requests: Iterable[InvocationTraceRequest],
    *,
    bounds: TraceBounds | Mapping[str, Any] | None = None,
) -> tuple[McpInvocationTrace, ...]:
    """Convenience entry point for a deterministic operation batch."""

    return McpInvocationTracer(graph, bounds=bounds).trace_many(requests)


# Compact spellings for downstream obligation compilers.
ReachabilityState = InvocationTerminalState
McpInvocationReachability = McpInvocationTrace
InvocationTrace = McpInvocationTrace
InvocationTracer = McpInvocationTracer
trace_mcp_invocation = compute_mcp_invocation_trace


__all__ = [
    "DEFAULT_INVOCATION_EDGE_KINDS",
    "DEFAULT_MAX_TRACE_DEPTH",
    "DEFAULT_MAX_TRACE_PATHS",
    "DEFAULT_MAX_TRACE_STATES",
    "HARD_MAX_TRACE_DEPTH",
    "HARD_MAX_TRACE_PATHS",
    "HARD_MAX_TRACE_STATES",
    "MCP_INVOCATION_PATH_SCHEMA",
    "MCP_INVOCATION_SEGMENT_SCHEMA",
    "MCP_INVOCATION_TRACE_INTERFACE",
    "MCP_INVOCATION_TRACE_SCHEMA",
    "MCP_INVOCATION_TRACE_VERSION",
    "InvocationPath",
    "InvocationPathClass",
    "InvocationPathSegment",
    "InvocationTerminalState",
    "InvocationTrace",
    "InvocationTraceRequest",
    "InvocationTracer",
    "McpInvocationReachability",
    "McpInvocationTrace",
    "McpInvocationTraceError",
    "McpInvocationTracer",
    "ReachabilityState",
    "TraceBounds",
    "compute_mcp_invocation_trace",
    "compute_mcp_invocation_traces",
    "trace_mcp_invocation",
]
