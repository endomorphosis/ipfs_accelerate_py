"""Exact cross-component MCP++ mediation tracing.

Interface: ``RuntimeMcpInvocationTrace@1``

Structural reachability (SCA-050) is necessary but not sufficient.  A call is
**mediated** only when a primary ``tools_dispatch`` or HTTP path traverses a
reviewed :class:`DispatchPipeline` with matching :class:`InterfaceDescriptor`
route, schema, and function identities.  Post-hoc descriptor name matches and
dynamic/unknown segments never prove mediation.

Direct fetch, direct import, and compatibility routes remain first-class,
visible path classes.  They can never become a mediation proof.  Native
supervisor operations and the three package targets
(``ipfs_kit_py``, ``ipfs_datasets_py``, ``ipfs_accelerate_py``) each receive
exactly one closed terminal state.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .mcp_invocation_trace import (
    InvocationPath,
    InvocationPathClass,
    InvocationTerminalState,
    McpInvocationTrace,
    TraceBounds,
    compute_mcp_invocation_trace,
)
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


RUNTIME_MCP_INVOCATION_TRACE_INTERFACE: Final = "RuntimeMcpInvocationTrace@1"
RUNTIME_MCP_INVOCATION_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-mcp-invocation-trace@1"
)
RUNTIME_MCP_MEDIATION_PATH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-mcp-mediation-path@1"
)
RUNTIME_MCP_MEDIATION_SEGMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-mcp-mediation-segment@1"
)
DISPATCH_PIPELINE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dispatch-pipeline@1"
)
INTERFACE_DESCRIPTOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/interface-descriptor@1"
)
RUNTIME_MCP_BATCH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-mcp-invocation-trace-batch@1"
)
RUNTIME_MCP_INVOCATION_TRACE_VERSION: Final = "1"

# Reviewed mediation stages that a primary tools_dispatch / HTTP path must
# close through.  Health and discovery are first-class sibling surfaces; they
# are recorded when present but are not required on every call path.
REQUIRED_MEDIATION_STAGES: Final[tuple[str, ...]] = (
    "capability",
    "connector",
    "call",
    "policy",
    "transport",
    "handler",
    "implementation",
)

# Sibling surfaces closed by the tracer (effects inventory).
PIPELINE_SURFACE_STAGES: Final[tuple[str, ...]] = (
    "health",
    "discovery",
    "call",
    "policy",
    "transport",
    "handler",
    "implementation",
)

KNOWN_PACKAGE_TARGETS: Final[tuple[str, ...]] = (
    "ipfs_kit_py",
    "ipfs_datasets_py",
    "ipfs_accelerate_py",
    "agent_supervisor",
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

_STAGE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "capability": "capability",
        "declaration": "capability",
        "descriptor": "capability",
        "swissknife_capability": "capability",
        "connector": "connector",
        "mcp_connector": "connector",
        "mcp_plus_plus": "connector",
        "mcplusplus": "connector",
        "health": "health",
        "health_check": "health",
        "discovery": "discovery",
        "tools_list": "discovery",
        "tools/list": "discovery",
        "list_tools": "discovery",
        "call": "call",
        "tools_call": "call",
        "tools/call": "call",
        "tools_dispatch": "call",
        "hierarchical_dispatch": "call",
        "dispatch": "call",
        "policy": "policy",
        "policy_mediation": "policy",
        "authorization": "policy",
        "ucan": "policy",
        "transport": "transport",
        "http": "transport",
        "stdio": "transport",
        "websocket": "transport",
        "libp2p": "transport",
        "handler": "handler",
        "registration": "handler",
        "package_handler": "handler",
        "implementation": "implementation",
        "function": "implementation",
        "package_function": "implementation",
        "schema": "schema",
        "event": "event",
        "receipt": "receipt",
        "behavior": "behavior",
    }
)

_BYPASS_MARKERS: Final[tuple[str, ...]] = (
    "compat",
    "legacy",
    "shim",
    "/api/v0/",
    "direct_fetch",
    "direct_rest",
    "direct_import",
    "compatibility_route",
    "bypass",
)

_DIRECT_FETCH_MARKERS: Final[tuple[str, ...]] = (
    "direct_fetch",
    "direct_rest",
    "raw_http_fetch",
    "fetch(",
)

_DIRECT_IMPORT_MARKERS: Final[tuple[str, ...]] = (
    "direct_import",
    "package_import",
    "from ipfs_",
    "import ipfs_",
)


class RuntimeMcpInvocationTraceError(ValueError):
    """A mediation request or serialized trace is malformed."""


class MediationTerminalState(str, Enum):
    """Closed terminal-state set for one runtime mediation claim.

    ``mediated`` is the only acceptance result.  It requires a reviewed
    pipeline path with matching interface identities; structural reachability
    alone is insufficient.
    """

    MEDIATED = "mediated"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"


class MediationPathClass(str, Enum):
    """Primary surfaces and visible bypass classes."""

    TOOLS_DISPATCH = "tools_dispatch"
    HTTP = "http"
    DIRECT_FETCH = "direct_fetch"
    DIRECT_IMPORT = "direct_import"
    COMPATIBILITY = "compatibility"
    OTHER = "other"


class DispatchPipelineStage(str, Enum):
    """Reviewed pipeline stages for cross-component mediation."""

    CAPABILITY = "capability"
    CONNECTOR = "connector"
    HEALTH = "health"
    DISCOVERY = "discovery"
    CALL = "call"
    POLICY = "policy"
    TRANSPORT = "transport"
    HANDLER = "handler"
    IMPLEMENTATION = "implementation"
    SCHEMA = "schema"
    EVENT = "event"
    RECEIPT = "receipt"
    BEHAVIOR = "behavior"


class RuntimePackageTarget(str, Enum):
    """Native supervisor plus the three package MCP targets."""

    IPFS_KIT_PY = "ipfs_kit_py"
    IPFS_DATASETS_PY = "ipfs_datasets_py"
    IPFS_ACCELERATE_PY = "ipfs_accelerate_py"
    AGENT_SUPERVISOR = "agent_supervisor"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RuntimeMcpInvocationTraceError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise RuntimeMcpInvocationTraceError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise RuntimeMcpInvocationTraceError(f"{name} is required")
    if len(value.encode("utf-8")) > 16_384:
        raise RuntimeMcpInvocationTraceError(f"{name} is oversized")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeMcpInvocationTraceError(f"{name} must be a sequence")
    return tuple(sorted({_text(str(item), name) for item in value}))


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise RuntimeMcpInvocationTraceError(
            "mediation value exceeds nesting bound"
        )
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise RuntimeMcpInvocationTraceError(
            "floating values are not canonical mediation data"
        )
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise RuntimeMcpInvocationTraceError(
                "mediation objects require at most 1024 string keys"
            )
        return {
            key: _plain(value[key], depth=depth + 1) for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 65_536:
            raise RuntimeMcpInvocationTraceError(
                "mediation sequence is oversized"
            )
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise RuntimeMcpInvocationTraceError(
        f"unsupported mediation value: {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeMcpInvocationTraceError(f"{name} must be an object")
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
        raise RuntimeMcpInvocationTraceError(
            f"unknown {name}: {value!r}"
        ) from exc


def _normalize_stage(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, DispatchPipelineStage):
        return value.value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if text in _STAGE_ALIASES:
        return _STAGE_ALIASES[text]
    if text in {item.value for item in DispatchPipelineStage}:
        return text
    # Allow dotted forms like pipeline.call
    if "." in text:
        return _normalize_stage(text.rsplit(".", 1)[-1])
    return None


def _dynamic_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _DYNAMIC_WORDS
    return False


def _contains_marker(value: Any, markers: Sequence[str]) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in markers)


@dataclass(frozen=True, slots=True)
class InterfaceDescriptor:
    """Exact route / schema / function identity binding for one operation.

    Display names are retained for audit only.  Mediation proof uses the
    identity fields (route_id, schema_id, function_id, behavior_id, event_id,
    receipt_id).  A name-only match never proves mediation.
    """

    route_id: str
    schema_id: str
    function_id: str
    behavior_id: str = ""
    event_id: str = ""
    receipt_id: str = ""
    package_id: str = ""
    descriptor_id: str = ""
    display_name: str = ""
    descriptor_content_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "route_id",
            "schema_id",
            "function_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        for name in (
            "behavior_id",
            "event_id",
            "receipt_id",
            "package_id",
            "descriptor_id",
            "display_name",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        expected = _cid(self._identity_payload())
        claimed = str(self.descriptor_content_id or "")
        if claimed and claimed != expected:
            raise RuntimeMcpInvocationTraceError(
                "interface descriptor identity mismatch"
            )
        object.__setattr__(self, "descriptor_content_id", expected)

    def matches(self, other: "InterfaceDescriptor") -> bool:
        """Exact identity equality on route, schema, and function."""

        return (
            self.route_id == other.route_id
            and self.schema_id == other.schema_id
            and self.function_id == other.function_id
            and (
                not self.behavior_id
                or not other.behavior_id
                or self.behavior_id == other.behavior_id
            )
            and (
                not self.package_id
                or not other.package_id
                or self.package_id == other.package_id
            )
        )

    def name_only_match(self, other: "InterfaceDescriptor") -> bool:
        """True when only display names agree — never proof-eligible."""

        if not self.display_name or not other.display_name:
            return False
        return (
            self.display_name == other.display_name
            and not self.matches(other)
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_DESCRIPTOR_SCHEMA,
            "route_id": self.route_id,
            "schema_id": self.schema_id,
            "function_id": self.function_id,
            "behavior_id": self.behavior_id,
            "event_id": self.event_id,
            "receipt_id": self.receipt_id,
            "package_id": self.package_id,
            "descriptor_id": self.descriptor_id,
            # display_name intentionally excluded from identity
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_content_id": self.descriptor_content_id,
            "display_name": self.display_name,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterfaceDescriptor":
        if value.get("schema") not in (None, INTERFACE_DESCRIPTOR_SCHEMA):
            raise RuntimeMcpInvocationTraceError(
                "unsupported interface descriptor schema"
            )
        return cls(
            route_id=str(value.get("route_id") or ""),
            schema_id=str(value.get("schema_id") or ""),
            function_id=str(value.get("function_id") or ""),
            behavior_id=str(value.get("behavior_id") or ""),
            event_id=str(value.get("event_id") or ""),
            receipt_id=str(value.get("receipt_id") or ""),
            package_id=str(value.get("package_id") or ""),
            descriptor_id=str(value.get("descriptor_id") or ""),
            display_name=str(value.get("display_name") or ""),
            descriptor_content_id=str(
                value.get("descriptor_content_id") or ""
            ),
        )


@dataclass(frozen=True, slots=True)
class DispatchPipeline:
    """Reviewed ordered stage set that may mediate a primary path."""

    pipeline_id: str
    stages: tuple[DispatchPipelineStage, ...]
    required_stages: tuple[DispatchPipelineStage, ...] = ()
    version: str = "1"
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pipeline_id", _text(self.pipeline_id, "pipeline_id")
        )
        object.__setattr__(
            self, "version", _text(self.version, "version")
        )
        stages = tuple(
            _enum(item, DispatchPipelineStage, "pipeline stage")
            for item in self.stages
        )
        if not stages:
            raise RuntimeMcpInvocationTraceError(
                "dispatch pipeline requires at least one stage"
            )
        # Preserve declared order while rejecting empty duplicates only as
        # consecutive no-ops; identity uses the ordered list.
        object.__setattr__(self, "stages", stages)
        if self.required_stages:
            required = tuple(
                _enum(item, DispatchPipelineStage, "required stage")
                for item in self.required_stages
            )
        else:
            # Default required set is the reviewed mediation stages, independent
            # of optional health/discovery surfaces.
            required = tuple(
                DispatchPipelineStage(name)
                for name in REQUIRED_MEDIATION_STAGES
            )
        object.__setattr__(self, "required_stages", required)
        expected = _cid(self._identity_payload())
        claimed = str(self.content_id or "")
        if claimed and claimed != expected:
            raise RuntimeMcpInvocationTraceError(
                "dispatch pipeline identity mismatch"
            )
        object.__setattr__(self, "content_id", expected)

    @property
    def required_stage_values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.required_stages)

    def covers(self, observed: Sequence[str | DispatchPipelineStage]) -> bool:
        observed_set = {
            _normalize_stage(item) or ""
            for item in observed
        }
        observed_set.discard("")
        return all(
            stage.value in observed_set for stage in self.required_stages
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": DISPATCH_PIPELINE_SCHEMA,
            "pipeline_id": self.pipeline_id,
            "version": self.version,
            "stages": [item.value for item in self.stages],
            "required_stages": [item.value for item in self.required_stages],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"content_id": self.content_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DispatchPipeline":
        if value.get("schema") not in (None, DISPATCH_PIPELINE_SCHEMA):
            raise RuntimeMcpInvocationTraceError(
                "unsupported dispatch pipeline schema"
            )
        return cls(
            pipeline_id=str(value.get("pipeline_id") or ""),
            stages=tuple(value.get("stages") or ()),
            required_stages=tuple(value.get("required_stages") or ()),
            version=str(value.get("version") or "1"),
            content_id=str(value.get("content_id") or ""),
        )

    @classmethod
    def reviewed_default(
        cls, pipeline_id: str = "mcp-plus-plus-reviewed@1"
    ) -> "DispatchPipeline":
        """Canonical reviewed mediation pipeline used by primary paths."""

        stages = tuple(
            DispatchPipelineStage(name)
            for name in (
                "capability",
                "connector",
                "health",
                "discovery",
                "call",
                "policy",
                "transport",
                "handler",
                "implementation",
                "schema",
                "event",
                "receipt",
                "behavior",
            )
        )
        return cls(
            pipeline_id=pipeline_id,
            stages=stages,
            required_stages=tuple(
                DispatchPipelineStage(name)
                for name in REQUIRED_MEDIATION_STAGES
            ),
        )


@dataclass(frozen=True, slots=True)
class MediationPathSegment:
    """One typed pipeline hop with exact source provenance."""

    stage: DispatchPipelineStage
    source_node_id: str
    target_node_id: str
    edge_id: str = ""
    edge_kind: str = ""
    source_ids: tuple[str, ...] = ()
    source_spans: tuple[Mapping[str, Any], ...] = ()
    dynamic: bool = False
    bypass: bool = False
    segment_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage",
            _enum(self.stage, DispatchPipelineStage, "pipeline stage"),
        )
        for name in ("source_node_id", "target_node_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "edge_id", _text(self.edge_id, "edge_id", required=False)
        )
        object.__setattr__(
            self,
            "edge_kind",
            _text(self.edge_kind, "edge_kind", required=False),
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
        object.__setattr__(self, "bypass", bool(self.bypass))
        expected = _cid(self._identity_payload())
        claimed = str(self.segment_id or "")
        if claimed and claimed != expected:
            raise RuntimeMcpInvocationTraceError("segment identity mismatch")
        object.__setattr__(self, "segment_id", expected)

    @property
    def has_exact_source(self) -> bool:
        return bool(self.source_ids and self.source_spans)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_MCP_MEDIATION_SEGMENT_SCHEMA,
            "stage": self.stage.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_id": self.edge_id,
            "edge_kind": self.edge_kind,
            "source_ids": list(self.source_ids),
            "source_spans": [dict(item) for item in self.source_spans],
            "dynamic": self.dynamic,
            "bypass": self.bypass,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MediationPathSegment":
        if value.get("schema") not in (
            None,
            RUNTIME_MCP_MEDIATION_SEGMENT_SCHEMA,
        ):
            raise RuntimeMcpInvocationTraceError(
                "unsupported mediation segment schema"
            )
        return cls(
            stage=value.get("stage", ""),
            source_node_id=str(value.get("source_node_id") or ""),
            target_node_id=str(value.get("target_node_id") or ""),
            edge_id=str(value.get("edge_id") or ""),
            edge_kind=str(value.get("edge_kind") or ""),
            source_ids=tuple(value.get("source_ids") or ()),
            source_spans=tuple(value.get("source_spans") or ()),
            dynamic=value.get("dynamic", False),
            bypass=value.get("bypass", False),
            segment_id=str(value.get("segment_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class MediationPath:
    """One classified mediation or bypass path with stage coverage."""

    path_class: MediationPathClass
    segments: tuple[MediationPathSegment, ...]
    expected_descriptor: InterfaceDescriptor | None = None
    observed_descriptor: InterfaceDescriptor | None = None
    structural_path_id: str = ""
    path_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path_class",
            _enum(self.path_class, MediationPathClass, "path class"),
        )
        segments = tuple(
            item
            if isinstance(item, MediationPathSegment)
            else MediationPathSegment.from_dict(item)
            for item in self.segments
        )
        object.__setattr__(self, "segments", segments)
        if self.expected_descriptor is not None and not isinstance(
            self.expected_descriptor, InterfaceDescriptor
        ):
            object.__setattr__(
                self,
                "expected_descriptor",
                InterfaceDescriptor.from_dict(self.expected_descriptor),
            )
        if self.observed_descriptor is not None and not isinstance(
            self.observed_descriptor, InterfaceDescriptor
        ):
            object.__setattr__(
                self,
                "observed_descriptor",
                InterfaceDescriptor.from_dict(self.observed_descriptor),
            )
        object.__setattr__(
            self,
            "structural_path_id",
            _text(
                self.structural_path_id,
                "structural_path_id",
                required=False,
            ),
        )
        expected = _cid(self._identity_payload())
        claimed = str(self.path_id or "")
        if claimed and claimed != expected:
            raise RuntimeMcpInvocationTraceError("path identity mismatch")
        object.__setattr__(self, "path_id", expected)

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(item.stage.value for item in self.segments)

    @property
    def dynamic(self) -> bool:
        return any(item.dynamic for item in self.segments)

    @property
    def is_bypass(self) -> bool:
        return self.path_class in {
            MediationPathClass.DIRECT_FETCH,
            MediationPathClass.DIRECT_IMPORT,
            MediationPathClass.COMPATIBILITY,
        } or any(item.bypass for item in self.segments)

    @property
    def is_primary(self) -> bool:
        return self.path_class in {
            MediationPathClass.TOOLS_DISPATCH,
            MediationPathClass.HTTP,
        }

    @property
    def identities_match(self) -> bool:
        if (
            self.expected_descriptor is None
            or self.observed_descriptor is None
        ):
            return False
        return self.expected_descriptor.matches(self.observed_descriptor)

    @property
    def name_only_match(self) -> bool:
        if (
            self.expected_descriptor is None
            or self.observed_descriptor is None
        ):
            return False
        return self.expected_descriptor.name_only_match(
            self.observed_descriptor
        )

    def pipeline_covers(self, pipeline: DispatchPipeline) -> bool:
        return pipeline.covers(self.stages)

    @property
    def proof_eligible(self) -> bool:
        """Primary path with full pipeline, matching identities, no dynamics."""

        return bool(
            self.is_primary
            and not self.is_bypass
            and not self.dynamic
            and self.segments
            and all(item.has_exact_source for item in self.segments)
            and self.identities_match
            and not self.name_only_match
        )

    def mediation_eligible(self, pipeline: DispatchPipeline) -> bool:
        return self.proof_eligible and self.pipeline_covers(pipeline)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_MCP_MEDIATION_PATH_SCHEMA,
            "path_class": self.path_class.value,
            "segments": [item.to_dict() for item in self.segments],
            "expected_descriptor": (
                self.expected_descriptor.to_dict()
                if self.expected_descriptor is not None
                else None
            ),
            "observed_descriptor": (
                self.observed_descriptor.to_dict()
                if self.observed_descriptor is not None
                else None
            ),
            "structural_path_id": self.structural_path_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "stages": list(self.stages),
            "dynamic": self.dynamic,
            "is_bypass": self.is_bypass,
            "is_primary": self.is_primary,
            "identities_match": self.identities_match,
            "name_only_match": self.name_only_match,
            "proof_eligible": self.proof_eligible,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MediationPath":
        if value.get("schema") not in (
            None,
            RUNTIME_MCP_MEDIATION_PATH_SCHEMA,
        ):
            raise RuntimeMcpInvocationTraceError(
                "unsupported mediation path schema"
            )
        expected = value.get("expected_descriptor")
        observed = value.get("observed_descriptor")
        result = cls(
            path_class=value.get("path_class", ""),
            segments=tuple(
                MediationPathSegment.from_dict(item)
                for item in value.get("segments") or ()
            ),
            expected_descriptor=(
                InterfaceDescriptor.from_dict(expected)
                if isinstance(expected, Mapping)
                else None
            ),
            observed_descriptor=(
                InterfaceDescriptor.from_dict(observed)
                if isinstance(observed, Mapping)
                else None
            ),
            structural_path_id=str(value.get("structural_path_id") or ""),
            path_id=str(value.get("path_id") or ""),
        )
        if (
            "proof_eligible" in value
            and bool(value["proof_eligible"]) != result.proof_eligible
        ):
            raise RuntimeMcpInvocationTraceError(
                "path proof_eligible claim mismatch"
            )
        return result


@dataclass(frozen=True, slots=True)
class RuntimeMcpInvocationTrace:
    """One operation with exactly one closed mediation terminal state."""

    operation_id: str
    package_id: str
    pipeline: DispatchPipeline
    terminal_state: MediationTerminalState
    reason_code: str
    snapshot_id: str = ""
    graph_root: str = ""
    expected_descriptor: InterfaceDescriptor | None = None
    observed_descriptor: InterfaceDescriptor | None = None
    mediated_paths: tuple[MediationPath, ...] = ()
    bypass_paths: tuple[MediationPath, ...] = ()
    unresolved_paths: tuple[MediationPath, ...] = ()
    structural_trace_id: str = ""
    complete: bool = True
    version: str = RUNTIME_MCP_INVOCATION_TRACE_VERSION
    trace_id: str = ""

    def __post_init__(self) -> None:
        for name in ("operation_id", "package_id", "reason_code", "version"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        for name in ("snapshot_id", "graph_root", "structural_trace_id"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        if not isinstance(self.pipeline, DispatchPipeline):
            object.__setattr__(
                self, "pipeline", DispatchPipeline.from_dict(self.pipeline)
            )
        object.__setattr__(
            self,
            "terminal_state",
            _enum(
                self.terminal_state,
                MediationTerminalState,
                "terminal state",
            ),
        )
        if self.expected_descriptor is not None and not isinstance(
            self.expected_descriptor, InterfaceDescriptor
        ):
            object.__setattr__(
                self,
                "expected_descriptor",
                InterfaceDescriptor.from_dict(self.expected_descriptor),
            )
        if self.observed_descriptor is not None and not isinstance(
            self.observed_descriptor, InterfaceDescriptor
        ):
            object.__setattr__(
                self,
                "observed_descriptor",
                InterfaceDescriptor.from_dict(self.observed_descriptor),
            )
        for name in (
            "mediated_paths",
            "bypass_paths",
            "unresolved_paths",
        ):
            paths = tuple(
                item
                if isinstance(item, MediationPath)
                else MediationPath.from_dict(item)
                for item in getattr(self, name)
            )
            unique = {item.path_id: item for item in paths}
            object.__setattr__(
                self, name, tuple(unique[key] for key in sorted(unique))
            )
        for item in self.bypass_paths:
            if item.path_class not in {
                MediationPathClass.DIRECT_FETCH,
                MediationPathClass.DIRECT_IMPORT,
                MediationPathClass.COMPATIBILITY,
                MediationPathClass.OTHER,
            } and not item.is_bypass:
                raise RuntimeMcpInvocationTraceError(
                    "bypass_paths must contain bypass path classes"
                )
        for item in self.unresolved_paths:
            if not item.dynamic:
                raise RuntimeMcpInvocationTraceError(
                    "unresolved_paths must contain a dynamic segment"
                )
        object.__setattr__(self, "complete", bool(self.complete))
        mediated = self.proof_paths
        if (
            self.terminal_state is MediationTerminalState.MEDIATED
            and not mediated
        ):
            raise RuntimeMcpInvocationTraceError(
                "mediated trace requires a pipeline-eligible path"
            )
        if (
            self.terminal_state is not MediationTerminalState.MEDIATED
            and mediated
            and self.terminal_state
            not in {
                MediationTerminalState.AMBIGUOUS,
                MediationTerminalState.NOT_MEASURED,
            }
        ):
            raise RuntimeMcpInvocationTraceError(
                "non-mediation terminal state contradicts a proved path"
            )
        expected = _cid(self._identity_payload())
        claimed = str(self.trace_id or "")
        if claimed and claimed != expected:
            raise RuntimeMcpInvocationTraceError("trace identity mismatch")
        object.__setattr__(self, "trace_id", expected)

    @property
    def state(self) -> MediationTerminalState:
        return self.terminal_state

    @property
    def all_paths(self) -> tuple[MediationPath, ...]:
        values = {
            item.path_id: item
            for item in (
                *self.mediated_paths,
                *self.bypass_paths,
                *self.unresolved_paths,
            )
        }
        return tuple(values[key] for key in sorted(values))

    @property
    def proof_paths(self) -> tuple[MediationPath, ...]:
        return tuple(
            item
            for item in self.mediated_paths
            if item.mediation_eligible(self.pipeline)
        )

    @property
    def visible_bypasses(self) -> tuple[MediationPath, ...]:
        return self.bypass_paths

    @property
    def identities_match(self) -> bool:
        if (
            self.expected_descriptor is None
            or self.observed_descriptor is None
        ):
            return False
        return self.expected_descriptor.matches(self.observed_descriptor)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_MCP_INVOCATION_TRACE_SCHEMA,
            "interface": RUNTIME_MCP_INVOCATION_TRACE_INTERFACE,
            "version": self.version,
            "operation_id": self.operation_id,
            "package_id": self.package_id,
            "pipeline": self.pipeline.to_dict(),
            "terminal_state": self.terminal_state.value,
            "reason_code": self.reason_code,
            "snapshot_id": self.snapshot_id,
            "graph_root": self.graph_root,
            "expected_descriptor": (
                self.expected_descriptor.to_dict()
                if self.expected_descriptor is not None
                else None
            ),
            "observed_descriptor": (
                self.observed_descriptor.to_dict()
                if self.observed_descriptor is not None
                else None
            ),
            "mediated_paths": [
                item.to_dict() for item in self.mediated_paths
            ],
            "bypass_paths": [item.to_dict() for item in self.bypass_paths],
            "unresolved_paths": [
                item.to_dict() for item in self.unresolved_paths
            ],
            "structural_trace_id": self.structural_trace_id,
            "complete": self.complete,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path_count": len(self.all_paths),
            "proved_path_ids": [item.path_id for item in self.proof_paths],
            "identities_match": self.identities_match,
            "visible_bypass_classes": sorted(
                {item.path_class.value for item in self.bypass_paths}
            ),
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
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "RuntimeMcpInvocationTrace":
        if value.get("schema") not in (
            None,
            RUNTIME_MCP_INVOCATION_TRACE_SCHEMA,
        ):
            raise RuntimeMcpInvocationTraceError(
                "unsupported runtime mediation trace schema"
            )
        if value.get("interface") not in (
            None,
            RUNTIME_MCP_INVOCATION_TRACE_INTERFACE,
        ):
            raise RuntimeMcpInvocationTraceError(
                "unsupported runtime mediation trace interface"
            )
        expected = value.get("expected_descriptor")
        observed = value.get("observed_descriptor")
        pipeline = value.get("pipeline")
        result = cls(
            operation_id=str(value.get("operation_id") or ""),
            package_id=str(value.get("package_id") or ""),
            pipeline=(
                DispatchPipeline.from_dict(pipeline)
                if isinstance(pipeline, Mapping)
                else pipeline
            ),
            terminal_state=value.get("terminal_state", ""),
            reason_code=str(value.get("reason_code") or ""),
            snapshot_id=str(value.get("snapshot_id") or ""),
            graph_root=str(value.get("graph_root") or ""),
            expected_descriptor=(
                InterfaceDescriptor.from_dict(expected)
                if isinstance(expected, Mapping)
                else None
            ),
            observed_descriptor=(
                InterfaceDescriptor.from_dict(observed)
                if isinstance(observed, Mapping)
                else None
            ),
            mediated_paths=tuple(
                MediationPath.from_dict(item)
                for item in value.get("mediated_paths") or ()
            ),
            bypass_paths=tuple(
                MediationPath.from_dict(item)
                for item in value.get("bypass_paths") or ()
            ),
            unresolved_paths=tuple(
                MediationPath.from_dict(item)
                for item in value.get("unresolved_paths") or ()
            ),
            structural_trace_id=str(value.get("structural_trace_id") or ""),
            complete=value.get("complete", False),
            version=str(
                value.get("version") or RUNTIME_MCP_INVOCATION_TRACE_VERSION
            ),
            trace_id=str(value.get("trace_id") or ""),
        )
        if "path_count" in value and int(value["path_count"]) != len(
            result.all_paths
        ):
            raise RuntimeMcpInvocationTraceError("trace path_count mismatch")
        if "proved_path_ids" in value and tuple(
            value["proved_path_ids"]
        ) != tuple(item.path_id for item in result.proof_paths):
            raise RuntimeMcpInvocationTraceError(
                "trace proved_path_ids mismatch"
            )
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "RuntimeMcpInvocationTrace":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeMcpInvocationTraceError(
                "trace JSON is malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeMcpInvocationTraceError(
                "trace JSON must contain an object"
            )
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class RuntimeMcpInvocationTraceBatch:
    """Deterministic multi-target mediation report."""

    traces: tuple[RuntimeMcpInvocationTrace, ...]
    batch_id: str = ""

    def __post_init__(self) -> None:
        items = tuple(
            item
            if isinstance(item, RuntimeMcpInvocationTrace)
            else RuntimeMcpInvocationTrace.from_dict(item)
            for item in self.traces
        )
        by_key: dict[tuple[str, str], RuntimeMcpInvocationTrace] = {}
        for item in items:
            key = (item.package_id, item.operation_id)
            if key in by_key:
                raise RuntimeMcpInvocationTraceError(
                    f"duplicate package/operation: {key[0]}:{key[1]}"
                )
            by_key[key] = item
        ordered = tuple(
            by_key[key]
            for key in sorted(by_key, key=lambda pair: (pair[0], pair[1]))
        )
        object.__setattr__(self, "traces", ordered)
        expected = _cid(self._identity_payload())
        claimed = str(self.batch_id or "")
        if claimed and claimed != expected:
            raise RuntimeMcpInvocationTraceError("batch identity mismatch")
        object.__setattr__(self, "batch_id", expected)

    def terminal_states_by_package(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                item.package_id: item.terminal_state.value
                for item in self.traces
            }
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_MCP_BATCH_SCHEMA,
            "traces": [item.to_dict() for item in self.traces],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "terminal_states_by_package": dict(
                self.terminal_states_by_package()
            ),
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
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "RuntimeMcpInvocationTraceBatch":
        if value.get("schema") not in (None, RUNTIME_MCP_BATCH_SCHEMA):
            raise RuntimeMcpInvocationTraceError(
                "unsupported mediation batch schema"
            )
        return cls(
            traces=tuple(
                RuntimeMcpInvocationTrace.from_dict(item)
                for item in value.get("traces") or ()
            ),
            batch_id=str(value.get("batch_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class MediationTraceRequest:
    """One package-scoped mediation query over explicit paths or a graph."""

    operation_id: str
    package_id: str
    expected_descriptor: InterfaceDescriptor
    paths: tuple[MediationPath, ...] = ()
    supported: bool = True
    measured: bool = True
    observed_descriptor: InterfaceDescriptor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self, "package_id", _text(self.package_id, "package_id")
        )
        if not isinstance(self.expected_descriptor, InterfaceDescriptor):
            object.__setattr__(
                self,
                "expected_descriptor",
                InterfaceDescriptor.from_dict(self.expected_descriptor),
            )
        if self.observed_descriptor is not None and not isinstance(
            self.observed_descriptor, InterfaceDescriptor
        ):
            object.__setattr__(
                self,
                "observed_descriptor",
                InterfaceDescriptor.from_dict(self.observed_descriptor),
            )
        raw_paths = self.paths
        if isinstance(raw_paths, MediationPath):
            raw_paths = (raw_paths,)
        elif isinstance(raw_paths, Mapping):
            raw_paths = (raw_paths,)
        elif raw_paths is None:
            raw_paths = ()
        elif not isinstance(raw_paths, Sequence) or isinstance(
            raw_paths, (str, bytes)
        ):
            raise RuntimeMcpInvocationTraceError("paths must be a sequence")
        paths = tuple(
            item
            if isinstance(item, MediationPath)
            else MediationPath.from_dict(item)
            for item in raw_paths
        )
        unique = {item.path_id: item for item in paths}
        object.__setattr__(
            self, "paths", tuple(unique[key] for key in sorted(unique))
        )
        object.__setattr__(self, "supported", bool(self.supported))
        object.__setattr__(self, "measured", bool(self.measured))


def _span_candidates(value: Any) -> tuple[Mapping[str, Any], ...]:
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
    unique: dict[bytes, Mapping[str, Any]] = {}
    for span in raw:
        path = span.get("path")
        if not isinstance(path, str) or not path:
            continue
        normalized = _mapping(span, "source span")
        unique[_canonical(normalized)] = normalized
    return tuple(unique[key] for key in sorted(unique))


def _source_evidence(
    edge: ContractGraphEdge,
    source: ContractGraphNode,
    target: ContractGraphNode,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    ids = set(edge.source_refs)
    spans = list(_span_candidates(edge.payload))
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
    )


def _edge_markers(edge: ContractGraphEdge) -> str:
    payload = edge.payload
    parts = [
        edge.kind.value,
        str(payload.get("pipeline_stage") or ""),
        str(payload.get("stage") or ""),
        str(payload.get("path_class") or ""),
        str(payload.get("path_kind") or ""),
        str(payload.get("route_kind") or ""),
        str(payload.get("kind") or ""),
        str(payload.get("mediation_kind") or ""),
        str(payload.get("target") or ""),
        str(payload.get("operation") or ""),
        str(payload.get("transport") or ""),
    ]
    return " ".join(parts).lower()


def _classify_path_class(
    edge_markers: Sequence[str],
    *,
    compatibility: bool = False,
) -> MediationPathClass:
    joined = " ".join(edge_markers).lower()
    if compatibility or any(
        marker in joined
        for marker in ("compat", "legacy", "shim", "/api/v0/", "compatibility")
    ):
        if any(marker in joined for marker in _DIRECT_FETCH_MARKERS):
            return MediationPathClass.DIRECT_FETCH
        if any(marker in joined for marker in _DIRECT_IMPORT_MARKERS):
            return MediationPathClass.DIRECT_IMPORT
        return MediationPathClass.COMPATIBILITY
    if any(marker in joined for marker in _DIRECT_FETCH_MARKERS):
        return MediationPathClass.DIRECT_FETCH
    if any(marker in joined for marker in _DIRECT_IMPORT_MARKERS):
        return MediationPathClass.DIRECT_IMPORT
    if "tools_dispatch" in joined or "hierarchical_dispatch" in joined:
        return MediationPathClass.TOOLS_DISPATCH
    if any(
        token in joined
        for token in (
            "tools/call",
            "tools_call",
            "http",
            "streamable_http",
            "sse",
        )
    ):
        return MediationPathClass.HTTP
    return MediationPathClass.OTHER


def _infer_stage(
    edge: ContractGraphEdge,
    source: ContractGraphNode,
    target: ContractGraphNode,
) -> DispatchPipelineStage | None:
    for payload in (edge.payload, source.payload, target.payload):
        for key in (
            "pipeline_stage",
            "stage",
            "mediation_stage",
            "dispatch_stage",
        ):
            stage = _normalize_stage(payload.get(key))
            if stage is not None:
                try:
                    return DispatchPipelineStage(stage)
                except ValueError:
                    continue
    # Kind / role inference — never upgrades bypass to mediation.
    markers = _edge_markers(edge)
    if edge.kind is ContractEdgeKind.ENFORCED_BY or "policy" in markers:
        return DispatchPipelineStage.POLICY
    if edge.kind is ContractEdgeKind.TRANSPORTED_BY or "transport" in markers:
        return DispatchPipelineStage.TRANSPORT
    if edge.kind is ContractEdgeKind.HANDLED_BY or target.kind is ContractNodeKind.HANDLER:
        return DispatchPipelineStage.HANDLER
    if edge.kind is ContractEdgeKind.IMPLEMENTS:
        return DispatchPipelineStage.IMPLEMENTATION
    if edge.kind is ContractEdgeKind.USES_SCHEMA or target.kind is ContractNodeKind.SCHEMA:
        return DispatchPipelineStage.SCHEMA
    if "health" in markers:
        return DispatchPipelineStage.HEALTH
    if any(
        token in markers
        for token in ("tools_list", "tools/list", "discovery", "list_tools")
    ):
        return DispatchPipelineStage.DISCOVERY
    if any(
        token in markers
        for token in (
            "tools_call",
            "tools/call",
            "tools_dispatch",
            "dispatch",
            "call",
        )
    ):
        return DispatchPipelineStage.CALL
    if edge.kind is ContractEdgeKind.DISPATCHES_TO:
        return DispatchPipelineStage.CALL
    if edge.kind is ContractEdgeKind.REGISTERS:
        return DispatchPipelineStage.HANDLER
    if edge.kind is ContractEdgeKind.DECLARES:
        return DispatchPipelineStage.CAPABILITY
    if source.kind is ContractNodeKind.INTERFACE or "connector" in markers:
        return DispatchPipelineStage.CONNECTOR
    if target.kind is ContractNodeKind.TOOL and edge.kind is ContractEdgeKind.CALLS:
        return DispatchPipelineStage.CALL
    if target.kind is ContractNodeKind.SYMBOL and edge.kind is ContractEdgeKind.IMPLEMENTS:
        return DispatchPipelineStage.IMPLEMENTATION
    return None


def _edge_is_bypass(edge: ContractGraphEdge) -> bool:
    payload = edge.payload
    if payload.get("compatibility") is True or payload.get(
        "bypass_candidate"
    ) is True or payload.get("bypass") is True:
        return True
    return _contains_marker(_edge_markers(edge), _BYPASS_MARKERS)


def _descriptor_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    defaults: InterfaceDescriptor | None = None,
) -> InterfaceDescriptor | None:
    if not isinstance(payload, Mapping):
        return defaults
    route_id = str(
        payload.get("route_id")
        or payload.get("route")
        or (defaults.route_id if defaults else "")
        or ""
    )
    schema_id = str(
        payload.get("schema_id")
        or payload.get("input_schema_id")
        or (defaults.schema_id if defaults else "")
        or ""
    )
    function_id = str(
        payload.get("function_id")
        or payload.get("handler_id")
        or payload.get("callable")
        or (defaults.function_id if defaults else "")
        or ""
    )
    if not (route_id and schema_id and function_id):
        return defaults
    return InterfaceDescriptor(
        route_id=route_id,
        schema_id=schema_id,
        function_id=function_id,
        behavior_id=str(
            payload.get("behavior_id")
            or (defaults.behavior_id if defaults else "")
            or ""
        ),
        event_id=str(
            payload.get("event_id")
            or (defaults.event_id if defaults else "")
            or ""
        ),
        receipt_id=str(
            payload.get("receipt_id")
            or (defaults.receipt_id if defaults else "")
            or ""
        ),
        package_id=str(
            payload.get("package_id")
            or (defaults.package_id if defaults else "")
            or ""
        ),
        descriptor_id=str(
            payload.get("descriptor_id")
            or (defaults.descriptor_id if defaults else "")
            or ""
        ),
        display_name=str(
            payload.get("display_name")
            or payload.get("name")
            or (defaults.display_name if defaults else "")
            or ""
        ),
    )


def _segment_from_edge(
    edge: ContractGraphEdge,
    nodes: Mapping[str, ContractGraphNode],
    stage: DispatchPipelineStage,
) -> MediationPathSegment:
    source = nodes[edge.source]
    target = nodes[edge.target]
    source_ids, spans = _source_evidence(edge, source, target)
    return MediationPathSegment(
        stage=stage,
        source_node_id=edge.source,
        target_node_id=edge.target,
        edge_id=edge.edge_id,
        edge_kind=edge.kind.value,
        source_ids=source_ids,
        source_spans=spans,
        dynamic=(
            _edge_is_dynamic(edge)
            or _node_is_dynamic(source)
            or _node_is_dynamic(target)
        ),
        bypass=_edge_is_bypass(edge),
    )


def _mediation_path_from_structural(
    path: InvocationPath,
    graph: SymbolicContractGraph,
    *,
    expected: InterfaceDescriptor,
    observed: InterfaceDescriptor | None,
    forced_class: MediationPathClass | None = None,
) -> MediationPath | None:
    nodes = {item.node_id: item for item in graph.nodes}
    edges = {item.edge_id: item for item in graph.edges}
    segments: list[MediationPathSegment] = []
    markers: list[str] = []
    compatibility = path.path_class is InvocationPathClass.COMPATIBILITY
    for segment in path.segments:
        edge = edges.get(segment.edge_id)
        if edge is None:
            continue
        source = nodes.get(edge.source)
        target = nodes.get(edge.target)
        if source is None or target is None:
            continue
        stage = _infer_stage(edge, source, target)
        if stage is None:
            # Unstaged hops remain visible as call/transport fallback only when
            # they carry exact provenance; otherwise skip without inventing.
            if edge.kind is ContractEdgeKind.DISPATCHES_TO:
                stage = DispatchPipelineStage.CALL
            elif edge.kind is ContractEdgeKind.TRANSPORTED_BY:
                stage = DispatchPipelineStage.TRANSPORT
            else:
                continue
        markers.append(_edge_markers(edge))
        segments.append(_segment_from_edge(edge, nodes, stage))
    if not segments:
        return None
    path_class = forced_class or _classify_path_class(
        markers, compatibility=compatibility
    )
    observed_descriptor = observed
    if observed_descriptor is None:
        # Prefer implementation-edge payload identities when present.
        for segment in reversed(path.segments):
            edge = edges.get(segment.edge_id)
            if edge is None:
                continue
            observed_descriptor = _descriptor_from_payload(
                edge.payload, defaults=expected
            )
            if observed_descriptor is not None and (
                observed_descriptor.route_id != expected.route_id
                or observed_descriptor.schema_id != expected.schema_id
                or observed_descriptor.function_id != expected.function_id
                or observed_descriptor.display_name
            ):
                break
        else:
            observed_descriptor = expected
    return MediationPath(
        path_class=path_class,
        segments=tuple(segments),
        expected_descriptor=expected,
        observed_descriptor=observed_descriptor,
        structural_path_id=path.path_id,
    )


class RuntimeMcpInvocationTracer:
    """Prove or refute reviewed MCP++ mediation for package operations."""

    interface = RUNTIME_MCP_INVOCATION_TRACE_INTERFACE

    def __init__(
        self,
        *,
        pipeline: DispatchPipeline | Mapping[str, Any] | None = None,
        graph: SymbolicContractGraph | None = None,
        bounds: TraceBounds | Mapping[str, Any] | None = None,
    ) -> None:
        if pipeline is None:
            self.pipeline = DispatchPipeline.reviewed_default()
        elif isinstance(pipeline, DispatchPipeline):
            self.pipeline = pipeline
        else:
            self.pipeline = DispatchPipeline.from_dict(pipeline)
        if graph is not None and not isinstance(graph, SymbolicContractGraph):
            raise RuntimeMcpInvocationTraceError(
                "graph must implement SymbolicContractGraph@1"
            )
        if graph is not None and graph.version != GRAPH_VERSION:
            raise RuntimeMcpInvocationTraceError(
                "unsupported SymbolicContractGraph version"
            )
        self.graph = graph
        self.bounds = (
            TraceBounds.from_value(bounds) if bounds is not None else None
        )

    def evaluate_paths(
        self,
        request: MediationTraceRequest,
        *,
        snapshot_id: str = "",
        graph_root: str = "",
        structural_trace_id: str = "",
    ) -> RuntimeMcpInvocationTrace:
        """Assign one terminal state from explicit mediation paths."""

        if not request.supported:
            return RuntimeMcpInvocationTrace(
                operation_id=request.operation_id,
                package_id=request.package_id,
                pipeline=self.pipeline,
                terminal_state=MediationTerminalState.UNSUPPORTED,
                reason_code="operation_family_unsupported",
                snapshot_id=snapshot_id,
                graph_root=graph_root,
                expected_descriptor=request.expected_descriptor,
                observed_descriptor=request.observed_descriptor,
                structural_trace_id=structural_trace_id,
                complete=True,
            )
        if not request.measured:
            return RuntimeMcpInvocationTrace(
                operation_id=request.operation_id,
                package_id=request.package_id,
                pipeline=self.pipeline,
                terminal_state=MediationTerminalState.NOT_MEASURED,
                reason_code="measurement_not_requested",
                snapshot_id=snapshot_id,
                graph_root=graph_root,
                expected_descriptor=request.expected_descriptor,
                observed_descriptor=request.observed_descriptor,
                structural_trace_id=structural_trace_id,
                complete=True,
            )

        primary: list[MediationPath] = []
        bypasses: list[MediationPath] = []
        unresolved: list[MediationPath] = []
        for path in request.paths:
            # Attach expected descriptor if a path omitted it.
            if path.expected_descriptor is None:
                path = MediationPath(
                    path_class=path.path_class,
                    segments=path.segments,
                    expected_descriptor=request.expected_descriptor,
                    observed_descriptor=(
                        path.observed_descriptor
                        or request.observed_descriptor
                    ),
                    structural_path_id=path.structural_path_id,
                )
            if path.dynamic:
                unresolved.append(path)
            elif path.is_bypass:
                bypasses.append(path)
            elif path.is_primary:
                primary.append(path)
            else:
                # Non-primary non-bypass routes are visible but cannot mediate.
                bypasses.append(
                    MediationPath(
                        path_class=path.path_class,
                        segments=tuple(
                            MediationPathSegment(
                                stage=item.stage,
                                source_node_id=item.source_node_id,
                                target_node_id=item.target_node_id,
                                edge_id=item.edge_id,
                                edge_kind=item.edge_kind,
                                source_ids=item.source_ids,
                                source_spans=item.source_spans,
                                dynamic=item.dynamic,
                                bypass=True,
                            )
                            for item in path.segments
                        ),
                        expected_descriptor=path.expected_descriptor,
                        observed_descriptor=path.observed_descriptor,
                        structural_path_id=path.structural_path_id,
                    )
                )

        proof = tuple(
            item
            for item in primary
            if item.mediation_eligible(self.pipeline)
        )
        # Name-only matches never prove mediation.
        name_only = tuple(
            item for item in primary if item.name_only_match
        )
        incomplete_provenance = tuple(
            item
            for item in primary
            if not item.dynamic
            and item.is_primary
            and not all(seg.has_exact_source for seg in item.segments)
        )
        missing_pipeline = tuple(
            item
            for item in primary
            if item.proof_eligible and not item.pipeline_covers(self.pipeline)
        )
        identity_mismatch = tuple(
            item
            for item in primary
            if item.is_primary
            and not item.dynamic
            and not item.identities_match
            and not item.name_only_match
        )

        observed = request.observed_descriptor
        if observed is None and proof:
            observed = proof[0].observed_descriptor
        elif observed is None and primary:
            observed = primary[0].observed_descriptor

        if unresolved and not proof:
            state = MediationTerminalState.AMBIGUOUS
            reason = "unresolved_dynamic_segment"
        elif len(proof) > 1:
            # Multiple mediated endpoints remain exact but ambiguous when they
            # disagree on observed function identity.
            functions = {
                (
                    item.observed_descriptor.function_id
                    if item.observed_descriptor
                    else ""
                )
                for item in proof
            }
            if len(functions) > 1:
                state = MediationTerminalState.AMBIGUOUS
                reason = "multiple_mediated_handlers"
            else:
                state = MediationTerminalState.MEDIATED
                reason = "exact_reviewed_pipeline_path"
        elif proof:
            state = MediationTerminalState.MEDIATED
            reason = "exact_reviewed_pipeline_path"
        elif name_only:
            state = MediationTerminalState.REFUTED
            reason = "descriptor_name_match_not_identity"
        elif incomplete_provenance:
            state = MediationTerminalState.NOT_MEASURED
            reason = "path_source_provenance_incomplete"
        elif missing_pipeline:
            state = MediationTerminalState.REFUTED
            reason = "primary_path_missing_reviewed_pipeline_stage"
        elif identity_mismatch:
            state = MediationTerminalState.REFUTED
            reason = "route_schema_function_identity_mismatch"
        elif primary:
            # Primary path present but not mediation-eligible for other reasons
            # (e.g. bypass flag on a tools_dispatch segment).
            state = MediationTerminalState.REFUTED
            reason = "primary_path_not_pipeline_mediated"
        elif bypasses:
            # Only bypasses: mediation is refuted, bypasses remain visible.
            state = MediationTerminalState.REFUTED
            reason = "only_bypass_paths_visible"
        else:
            state = MediationTerminalState.REFUTED
            reason = "no_reviewed_mediation_path"

        return RuntimeMcpInvocationTrace(
            operation_id=request.operation_id,
            package_id=request.package_id,
            pipeline=self.pipeline,
            terminal_state=state,
            reason_code=reason,
            snapshot_id=snapshot_id,
            graph_root=graph_root,
            expected_descriptor=request.expected_descriptor,
            observed_descriptor=observed,
            mediated_paths=tuple(primary),
            bypass_paths=tuple(bypasses),
            unresolved_paths=tuple(unresolved),
            structural_trace_id=structural_trace_id,
            complete=True,
        )

    def trace_request(
        self, request: MediationTraceRequest
    ) -> RuntimeMcpInvocationTrace:
        """Trace one request, optionally projecting from a pinned graph."""

        if request.paths:
            snapshot_id = self.graph.snapshot_id if self.graph else ""
            graph_root = self.graph.graph_root if self.graph else ""
            return self.evaluate_paths(
                request,
                snapshot_id=snapshot_id,
                graph_root=graph_root,
            )
        if self.graph is None:
            return self.evaluate_paths(request)
        raise RuntimeMcpInvocationTraceError(
            "graph-backed mediation requires explicit paths or "
            "trace_from_structural_graph arguments"
        )

    def trace_from_structural_graph(
        self,
        *,
        operation_id: str,
        package_id: str,
        source_node_id: str,
        target_node_ids: Sequence[str],
        expected_descriptor: InterfaceDescriptor | Mapping[str, Any],
        observed_descriptor: (
            InterfaceDescriptor | Mapping[str, Any] | None
        ) = None,
        path_class_hint: MediationPathClass | str | None = None,
        supported: bool = True,
        measured: bool = True,
    ) -> RuntimeMcpInvocationTrace:
        """Lift an SCA-050 structural trace into a mediation decision."""

        if self.graph is None:
            raise RuntimeMcpInvocationTraceError(
                "trace_from_structural_graph requires a pinned graph"
            )
        if not isinstance(expected_descriptor, InterfaceDescriptor):
            expected_descriptor = InterfaceDescriptor.from_dict(
                expected_descriptor
            )
        if observed_descriptor is not None and not isinstance(
            observed_descriptor, InterfaceDescriptor
        ):
            observed_descriptor = InterfaceDescriptor.from_dict(
                observed_descriptor
            )
        if not self.graph.complete:
            return RuntimeMcpInvocationTrace(
                operation_id=operation_id,
                package_id=package_id,
                pipeline=self.pipeline,
                terminal_state=MediationTerminalState.NOT_MEASURED,
                reason_code="incomplete_symbolic_contract_graph",
                snapshot_id=self.graph.snapshot_id,
                graph_root=self.graph.graph_root,
                expected_descriptor=expected_descriptor,
                observed_descriptor=observed_descriptor,
                complete=False,
            )
        structural = compute_mcp_invocation_trace(
            self.graph,
            operation_id,
            source_node_id,
            target_node_ids,
            bounds=self.bounds,
            supported=supported,
            measured=measured,
        )
        if structural.terminal_state is InvocationTerminalState.UNSUPPORTED:
            return RuntimeMcpInvocationTrace(
                operation_id=operation_id,
                package_id=package_id,
                pipeline=self.pipeline,
                terminal_state=MediationTerminalState.UNSUPPORTED,
                reason_code=structural.reason_code,
                snapshot_id=self.graph.snapshot_id,
                graph_root=self.graph.graph_root,
                expected_descriptor=expected_descriptor,
                observed_descriptor=observed_descriptor,
                structural_trace_id=structural.trace_id,
                complete=structural.complete,
            )
        if structural.terminal_state is InvocationTerminalState.NOT_MEASURED:
            return RuntimeMcpInvocationTrace(
                operation_id=operation_id,
                package_id=package_id,
                pipeline=self.pipeline,
                terminal_state=MediationTerminalState.NOT_MEASURED,
                reason_code=structural.reason_code,
                snapshot_id=self.graph.snapshot_id,
                graph_root=self.graph.graph_root,
                expected_descriptor=expected_descriptor,
                observed_descriptor=observed_descriptor,
                structural_trace_id=structural.trace_id,
                complete=structural.complete,
            )

        forced = (
            None
            if path_class_hint is None
            else _enum(path_class_hint, MediationPathClass, "path class hint")
        )
        paths: list[MediationPath] = []
        for structural_path in structural.all_paths:
            projected = _mediation_path_from_structural(
                structural_path,
                self.graph,
                expected=expected_descriptor,
                observed=observed_descriptor,
                forced_class=forced,
            )
            if projected is not None:
                paths.append(projected)
        request = MediationTraceRequest(
            operation_id=operation_id,
            package_id=package_id,
            expected_descriptor=expected_descriptor,
            observed_descriptor=observed_descriptor,
            paths=tuple(paths),
            supported=supported,
            measured=measured,
        )
        return self.evaluate_paths(
            request,
            snapshot_id=self.graph.snapshot_id,
            graph_root=self.graph.graph_root,
            structural_trace_id=structural.trace_id,
        )

    def trace_many(
        self, requests: Iterable[MediationTraceRequest]
    ) -> RuntimeMcpInvocationTraceBatch:
        """Trace package operations deterministically, rejecting duplicates."""

        items: list[RuntimeMcpInvocationTrace] = []
        seen: set[tuple[str, str]] = set()
        for request in requests:
            if not isinstance(request, MediationTraceRequest):
                raise RuntimeMcpInvocationTraceError(
                    "trace_many requires MediationTraceRequest values"
                )
            key = (request.package_id, request.operation_id)
            if key in seen:
                raise RuntimeMcpInvocationTraceError(
                    f"duplicate package/operation: {key[0]}:{key[1]}"
                )
            seen.add(key)
            items.append(self.trace_request(request))
        return RuntimeMcpInvocationTraceBatch(traces=tuple(items))


def build_mediation_path(
    *,
    path_class: MediationPathClass | str,
    stages: Sequence[DispatchPipelineStage | str],
    expected: InterfaceDescriptor | Mapping[str, Any],
    observed: InterfaceDescriptor | Mapping[str, Any] | None = None,
    source_prefix: str = "node",
    path: str = "fixture/mediation.py",
    start_line: int = 1,
    dynamic_stages: Sequence[str] = (),
    bypass: bool = False,
) -> MediationPath:
    """Fixture helper: build an exact-provenance mediation path."""

    if not isinstance(expected, InterfaceDescriptor):
        expected = InterfaceDescriptor.from_dict(expected)
    if observed is None:
        observed = expected
    elif not isinstance(observed, InterfaceDescriptor):
        observed = InterfaceDescriptor.from_dict(observed)
    dynamic_set = {str(item) for item in dynamic_stages}
    segments: list[MediationPathSegment] = []
    stage_values = [
        _enum(item, DispatchPipelineStage, "stage") for item in stages
    ]
    for index, stage in enumerate(stage_values):
        source = f"{source_prefix}:{index}"
        target = f"{source_prefix}:{index + 1}"
        line = start_line + index
        span = {
            "path": path,
            "source_sha256": "sha256:" + f"{line:064x}",
            "start_line": line,
            "start_column": 0,
            "end_line": line,
            "end_column": 40,
        }
        segments.append(
            MediationPathSegment(
                stage=stage,
                source_node_id=source,
                target_node_id=target,
                edge_id=f"edge:{source_prefix}:{index}",
                edge_kind="dispatches_to",
                source_ids=(f"source:{source_prefix}:{index}", span["source_sha256"]),
                source_spans=(span,),
                dynamic=stage.value in dynamic_set,
                bypass=bypass
                or path_class
                in {
                    MediationPathClass.DIRECT_FETCH,
                    MediationPathClass.DIRECT_IMPORT,
                    MediationPathClass.COMPATIBILITY,
                    "direct_fetch",
                    "direct_import",
                    "compatibility",
                },
            )
        )
    return MediationPath(
        path_class=path_class,
        segments=tuple(segments),
        expected_descriptor=expected,
        observed_descriptor=observed,
    )


def compute_runtime_mcp_invocation_trace(
    request: MediationTraceRequest,
    *,
    pipeline: DispatchPipeline | Mapping[str, Any] | None = None,
    graph: SymbolicContractGraph | None = None,
) -> RuntimeMcpInvocationTrace:
    """Convenience entry for a single mediation decision."""

    return RuntimeMcpInvocationTracer(
        pipeline=pipeline, graph=graph
    ).trace_request(request)


def compute_runtime_mcp_invocation_traces(
    requests: Iterable[MediationTraceRequest],
    *,
    pipeline: DispatchPipeline | Mapping[str, Any] | None = None,
    graph: SymbolicContractGraph | None = None,
) -> RuntimeMcpInvocationTraceBatch:
    """Convenience entry for a deterministic multi-package batch."""

    return RuntimeMcpInvocationTracer(
        pipeline=pipeline, graph=graph
    ).trace_many(requests)


# Compact aliases for obligation compilers and catalogs.
RuntimeMcpTrace = RuntimeMcpInvocationTrace
RuntimeMcpTracer = RuntimeMcpInvocationTracer
DispatchPipelineSpec = DispatchPipeline
InterfaceDescriptorSpec = InterfaceDescriptor


__all__ = [
    "DISPATCH_PIPELINE_SCHEMA",
    "INTERFACE_DESCRIPTOR_SCHEMA",
    "KNOWN_PACKAGE_TARGETS",
    "PIPELINE_SURFACE_STAGES",
    "REQUIRED_MEDIATION_STAGES",
    "RUNTIME_MCP_BATCH_SCHEMA",
    "RUNTIME_MCP_INVOCATION_TRACE_INTERFACE",
    "RUNTIME_MCP_INVOCATION_TRACE_SCHEMA",
    "RUNTIME_MCP_INVOCATION_TRACE_VERSION",
    "RUNTIME_MCP_MEDIATION_PATH_SCHEMA",
    "RUNTIME_MCP_MEDIATION_SEGMENT_SCHEMA",
    "DispatchPipeline",
    "DispatchPipelineSpec",
    "DispatchPipelineStage",
    "InterfaceDescriptor",
    "InterfaceDescriptorSpec",
    "MediationPath",
    "MediationPathClass",
    "MediationPathSegment",
    "MediationTerminalState",
    "MediationTraceRequest",
    "RuntimeMcpInvocationTrace",
    "RuntimeMcpInvocationTraceBatch",
    "RuntimeMcpInvocationTraceError",
    "RuntimeMcpInvocationTracer",
    "RuntimeMcpTrace",
    "RuntimeMcpTracer",
    "RuntimePackageTarget",
    "build_mediation_path",
    "compute_runtime_mcp_invocation_trace",
    "compute_runtime_mcp_invocation_traces",
]
