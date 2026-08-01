"""Resolve SwissKnife MCP++ calls to package registrations (VFS-017 / VFS-G060).

This module performs **static** inventory-bound call-path resolution only:

```text
caller
  -> connector method
  -> negotiated profile / transport
  -> tools/list or declared interface
  -> tools/call name and schema
  -> registered server adapter
  -> package implementation
  -> result / error mapping back to the caller
```

Static resolution is deliberately split from hermetic runtime conformance
(VFS-G061 / ``mcplusplus_runtime_witness``):

* A ``proved`` path means every hop is ``resolved_static`` under inventory
  evidence. It is **not** a hermetic runtime witness and never grants
  ``runtime_witnessed`` authority.
* Runtime request/result/error/capability/transport observations are deferred
  to the child goal ``VFS-G061`` and evidence ``vfs/mcplusplus-runtime-witness@1``.
* This module never opens network, never dispatches adapters, and never emits
  runtime receipts.

Resolution is fail-closed and evidence-bound:

* TypeScript and Python names, JSON Schema, manifests, aliases, version and
  profile negotiation, HTTP, and ``mcp+p2p`` edges may bind a hop when
  inventory evidence proves the edge.
* Same-name local helpers, mocks, test servers, copied manifests, static
  dashboards, legacy fallbacks, and imports without call edges never prove
  invocation.
* Ambiguous multi-candidate hops and external packages remain explicit
  frontiers. Manifest drift emits minimal witnesses rather than silent
  merges.

GraphRAG / model enrichment is out of scope. Confidence and reason codes are
deterministic pure functions of status and rule identity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .program_assurance_contracts import ClaimLevel
from .program_graph import (
    ProgramEdgeKind,
    ProgramGraph,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    canonical_program_json,
)
from .proof.formal_verification_contracts import content_identity


MCPLUSPLUS_CONTRACT_RESOLVER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-contract-resolver@1"
)
MCPLUSPLUS_CALL_PATH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-call-path@1"
)
MCPLUSPLUS_PATH_HOP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-path-hop@1"
)
MCPLUSPLUS_PATH_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-path-evidence@1"
)
MCPLUSPLUS_RESOLUTION_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-resolution-result@1"
)
MCPLUSPLUS_MANIFEST_DRIFT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-manifest-drift@1"
)
MCPLUSPLUS_FRONTIER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-frontier@1"
)
MCPLUSPLUS_INVENTORY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-inventory@1"
)
MCPLUSPLUS_ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-inventory-artifact@1"
)

# Evidence kinds produced by this static resolver (VFS-G060 / packet G152+G153).
EVIDENCE_CALL_PATH = "vfs/mcplusplus-call-path@1"
EVIDENCE_MANIFEST_PARITY = "vfs/mcplusplus-manifest-parity@1"
# Runtime evidence is owned by the hermetic child goal — never emitted here.
EVIDENCE_RUNTIME_WITNESS = "vfs/mcplusplus-runtime-witness@1"
STATIC_EVIDENCE_KINDS: tuple[str, ...] = (
    EVIDENCE_CALL_PATH,
    EVIDENCE_MANIFEST_PARITY,
)
EXCLUDED_RUNTIME_EVIDENCE_KINDS: tuple[str, ...] = (EVIDENCE_RUNTIME_WITNESS,)

# Objective-heap alignment: static parent goal vs hermetic runtime child.
STATIC_RESOLUTION_GOAL_ID = "VFS-G060"
HERMETIC_RUNTIME_CHILD_GOAL_ID = "VFS-G061"
STATIC_RESOLUTION_CLAIM_LEVEL = ClaimLevel.RESOLVED_STATIC
HERMETIC_RUNTIME_CLAIM_LEVEL = ClaimLevel.RUNTIME_WITNESSED

# Leaf goals for the mcp_interop goal packet (VFS-G152 call-path, VFS-G153 parity).
# Labels are discovery metadata only — never enter path_id / result_id identity.
OBJECTIVE_PARENT_GOAL_ID = STATIC_RESOLUTION_GOAL_ID
OBJECTIVE_CALL_PATH_GOAL_ID = "VFS-G152"
OBJECTIVE_MANIFEST_PARITY_GOAL_ID = "VFS-G153"
OBJECTIVE_CALL_PATH_TASK_ID = "VFS-072"
OBJECTIVE_MANIFEST_PARITY_TASK_ID = "VFS-075"
# Primary objective for this packet-anchor task surface (call-path leaf).
OBJECTIVE_GOAL_ID = OBJECTIVE_CALL_PATH_GOAL_ID
OBJECTIVE_TASK_ID = OBJECTIVE_CALL_PATH_TASK_ID
OBJECTIVE_GOAL_PACKET_ID = (
    "goal_packet/mcp_interop/ipfs_accelerate_py/9f2828fd2adb"
)
OBJECTIVE_DOMAIN_EVIDENCE_TERMS: tuple[str, ...] = STATIC_EVIDENCE_KINDS
OBJECTIVE_PACKET_GOAL_IDS: tuple[str, ...] = (
    OBJECTIVE_CALL_PATH_GOAL_ID,
    OBJECTIVE_MANIFEST_PARITY_GOAL_ID,
)
OBJECTIVE_PACKET_TASK_IDS: tuple[str, ...] = (
    OBJECTIVE_CALL_PATH_TASK_ID,
    OBJECTIVE_MANIFEST_PARITY_TASK_ID,
)

MCPLUSPLUS_CALL_PATH_CLAIM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-call-path-claim@1"
)
MCPLUSPLUS_MANIFEST_PARITY_CLAIM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-manifest-parity-claim@1"
)
MCPLUSPLUS_STATIC_PACKET_CLAIM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-static-packet-claim@1"
)

CALL_PATH_INVARIANTS: tuple[str, ...] = (
    "full stage chain caller->connector->transport->list->call->registry->"
    "adapter->implementation->result/error is inventory-bound",
    "same-name helpers, mocks, static payloads, copied manifests, and "
    "fallbacks never prove invocation",
    "ambiguous and dynamic registrations remain explicit frontiers",
    "proved paths are resolved_static only and never claim runtime_witnessed",
)
MANIFEST_PARITY_INVARIANTS: tuple[str, ...] = (
    "Python, TypeScript, schema, and error-map names are checked for parity",
    "manifest drift emits minimal witnesses rather than silent merges",
    "schema, version, language-name, and error-map mismatches fail closed",
    "static parity never grants hermetic runtime or completion authority",
)
MANIFEST_PARITY_REQUIRED_ASPECTS: tuple[str, ...] = (
    "python_name",
    "typescript_name",
    "input_schema",
    "output_schema",
    "version",
    "error_map",
)

# Keep exact-text discovery anchors aligned with the objective heap.
assert EVIDENCE_CALL_PATH == "vfs/mcplusplus-call-path@1"
assert EVIDENCE_MANIFEST_PARITY == "vfs/mcplusplus-manifest-parity@1"
assert EVIDENCE_RUNTIME_WITNESS == "vfs/mcplusplus-runtime-witness@1"
assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G060"
assert OBJECTIVE_CALL_PATH_GOAL_ID == "VFS-G152"
assert OBJECTIVE_MANIFEST_PARITY_GOAL_ID == "VFS-G153"
assert OBJECTIVE_CALL_PATH_TASK_ID == "VFS-072"
assert OBJECTIVE_MANIFEST_PARITY_TASK_ID == "VFS-075"
assert OBJECTIVE_GOAL_ID == "VFS-G152"
assert OBJECTIVE_TASK_ID == "VFS-072"
assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
    "vfs/mcplusplus-call-path@1",
    "vfs/mcplusplus-manifest-parity@1",
)
assert OBJECTIVE_PACKET_GOAL_IDS == ("VFS-G152", "VFS-G153")

RESOLVER_VERSION = "mcplusplus-contract-resolver@1"
RESOLVER_PRODUCER = "mcplusplus-contract-resolver@1"
RESOLUTION_LAYER_STATIC = "static"
RESOLUTION_LAYER_RUNTIME = "runtime"

DEFAULT_MAX_PATHS = 50_000
DEFAULT_MAX_HOPS = 32
DEFAULT_MAX_ARTIFACTS = 250_000
DEFAULT_MAX_DRIFT_WITNESSES = 50_000
DEFAULT_MAX_FRONTIER_ITEMS = 50_000
DEFAULT_MAX_LABEL_BYTES = 4_096
DEFAULT_MAX_NOTES_BYTES = 8_192
DEFAULT_MAX_SCHEMA_BYTES = 262_144

# Ordered stages that must be walked for a proved invocation path.
PATH_STAGE_ORDER: tuple[str, ...] = (
    "caller",
    "connector",
    "profile_transport",
    "tools_list",
    "tools_call",
    "server_registry",
    "adapter",
    "package_implementation",
    "result_error_mapping",
)

# Artifact roles that can never authorize a resolved_static hop.
_NON_INVOCATION_ROLES: frozenset[str] = frozenset(
    {
        "mock",
        "test_server",
        "copied_manifest",
        "static_dashboard",
        "legacy_fallback",
        "local_helper",
    }
)

# Path / name markers that classify an artifact as non-authoritative.
_MOCK_MARKERS: frozenset[str] = frozenset(
    {
        "mock",
        "mocks",
        "fake",
        "stub",
        "dummy",
        "unittest.mock",
        "MagicMock",
        "AsyncMock",
    }
)
_TEST_MARKERS: frozenset[str] = frozenset(
    {
        "test",
        "tests",
        "testing",
        "fixture",
        "fixtures",
        "conftest",
        "__tests__",
        "spec.",
        ".spec.",
        ".test.",
    }
)
_DASHBOARD_MARKERS: frozenset[str] = frozenset(
    {
        "dashboard",
        "static_dashboard",
        "dashboard_data",
        "static payload",
        "hardcoded_tools",
    }
)
_FALLBACK_MARKERS: frozenset[str] = frozenset(
    {
        "legacy",
        "fallback",
        "compat_shim",
        "deprecated",
        "shadow",
        ".fixed",
        ".full",
        ".broken",
        ".new",
        ".clean",
        ".optimized",
    }
)

_KNOWN_PROFILES: frozenset[str] = frozenset(
    {
        "mcp++/basic",
        "mcp++/mcp-idl",
        "mcp++/idl",
        "mcp++/cid-envelope",
        "mcp++/ucan",
        "mcp++/deontic-policy",
        "mcp++/event-dag",
        "mcp++/p2p-transport",
        "mcp++/risk-scheduling",
        "mcp++/x402-payments",
    }
)

_HIERARCHICAL_ALIAS_RE = re.compile(
    r"^(?P<category>[A-Za-z0-9_]+)[./](?P<tool>[A-Za-z0-9_]+)$"
)


class MCPlusPlusResolverError(ValueError):
    """An MCP++ resolution input or record violates the fail-closed contract."""


class MCPlusPlusResolverBoundsError(MCPlusPlusResolverError):
    """A resolution result exceeded a hard deterministic bound."""


class MissingPathEvidenceError(MCPlusPlusResolverError):
    """A hop or path was emitted without required evidence."""


class ManufacturedInvocationError(MCPlusPlusResolverError):
    """A caller attempted to prove invocation without a call edge."""


class PathStage(str, Enum):
    """Closed vocabulary of MCP++ call-path stages."""

    CALLER = "caller"
    CONNECTOR = "connector"
    PROFILE_TRANSPORT = "profile_transport"
    TOOLS_LIST = "tools_list"
    TOOLS_CALL = "tools_call"
    SERVER_REGISTRY = "server_registry"
    ADAPTER = "adapter"
    PACKAGE_IMPLEMENTATION = "package_implementation"
    RESULT_ERROR_MAPPING = "result_error_mapping"


class ArtifactRole(str, Enum):
    """Closed vocabulary of inventory artifact roles."""

    CALLER = "caller"
    CONNECTOR = "connector"
    TRANSPORT = "transport"
    INTERFACE_DESCRIPTOR = "interface_descriptor"
    TOOL_LIST_ENTRY = "tool_list_entry"
    TOOL_CALL_SITE = "tool_call_site"
    REGISTRATION = "registration"
    ADAPTER = "adapter"
    IMPLEMENTATION = "implementation"
    JSON_SCHEMA = "json_schema"
    MANIFEST = "manifest"
    ERROR_MAP = "error_map"
    RESULT_MAP = "result_map"
    ALIAS = "alias"
    MOCK = "mock"
    TEST_SERVER = "test_server"
    COPIED_MANIFEST = "copied_manifest"
    STATIC_DASHBOARD = "static_dashboard"
    LEGACY_FALLBACK = "legacy_fallback"
    LOCAL_HELPER = "local_helper"


class TransportKind(str, Enum):
    """Transport used for MCP++ JSON-RPC."""

    HTTP = "http"
    MCP_P2P = "mcp+p2p"
    UNKNOWN = "unknown"


class DriftKind(str, Enum):
    """Closed vocabulary of manifest / parity drift kinds."""

    NAME_MISMATCH = "name_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    ALIAS_MISMATCH = "alias_mismatch"
    ERROR_MAP_MISMATCH = "error_map_mismatch"
    RESULT_MAP_MISMATCH = "result_map_mismatch"
    MISSING_REGISTRATION = "missing_registration"
    EXTRA_UNREACHABLE = "extra_unreachable"
    STALE_MANIFEST = "stale_manifest"
    COPIED_WITHOUT_BINDING = "copied_without_binding"
    TRANSPORT_MISMATCH = "transport_mismatch"
    LANGUAGE_NAME_MISMATCH = "language_name_mismatch"


class ReasonCode(str, Enum):
    """Closed vocabulary of deterministic MCP++ resolution reason codes."""

    PROVED_INVOCATION_CHAIN = "proved_invocation_chain"
    CONNECTOR_BINDING = "connector_binding"
    PROFILE_NEGOTIATED = "profile_negotiated"
    TRANSPORT_HTTP = "transport_http"
    TRANSPORT_MCP_P2P = "transport_mcp_p2p"
    TOOLS_LIST_BINDING = "tools_list_binding"
    INTERFACE_DESCRIPTOR_BINDING = "interface_descriptor_binding"
    TOOLS_CALL_BINDING = "tools_call_binding"
    REGISTRATION_MATCH = "registration_match"
    ADAPTER_BINDING = "adapter_binding"
    IMPLEMENTATION_MATCH = "implementation_match"
    RESULT_MAP_MATCH = "result_map_match"
    ERROR_MAP_MATCH = "error_map_match"
    SCHEMA_PARITY = "schema_parity"
    NAME_ALIAS = "name_alias"
    MANIFEST_PARITY = "manifest_parity"
    CALL_EDGE_PRESENT = "call_edge_present"

    SAME_NAME_HELPER = "same_name_helper"
    MOCK_IMPLEMENTATION = "mock_implementation"
    TEST_SERVER = "test_server"
    COPIED_MANIFEST = "copied_manifest"
    STATIC_DASHBOARD = "static_dashboard"
    LEGACY_FALLBACK = "legacy_fallback"
    IMPORT_WITHOUT_CALL = "import_without_call"
    NAME_ONLY_MATCH = "name_only_match"

    AMBIGUOUS_REGISTRATION = "ambiguous_registration"
    AMBIGUOUS_ADAPTER = "ambiguous_adapter"
    AMBIGUOUS_IMPLEMENTATION = "ambiguous_implementation"
    AMBIGUOUS_ALIAS = "ambiguous_alias"
    EXTERNAL_PACKAGE = "external_package"
    EXTERNAL_TRANSPORT = "external_transport"
    PROFILE_MISMATCH = "profile_mismatch"
    TRANSPORT_MISMATCH = "transport_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    MANIFEST_DRIFT = "manifest_drift"
    MISSING_HOP = "missing_hop"
    MISSING_REGISTRATION = "missing_registration"
    MISSING_CALL_EDGE = "missing_call_edge"
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"
    EVIDENCE_REQUIRED = "evidence_required"
    ALREADY_RESOLVED = "already_resolved"
    NO_TARGET = "no_target"
    DYNAMIC_DISPATCH = "dynamic_dispatch"


class PathVerdict(str, Enum):
    """Overall verdict for one traced MCP++ call path.

    ``PROVED`` means statically proved under inventory evidence only. It never
    means hermetic runtime conformance (that is VFS-G061).
    """

    PROVED = "proved"
    CANDIDATE = "candidate"
    AMBIGUOUS = "ambiguous"
    EXTERNAL = "external"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ResolutionLayer(str, Enum):
    """Assurance layer of an MCP++ resolution product.

    Static resolution (this module) and hermetic runtime conformance
    (``mcplusplus_runtime_witness``) are closed, non-interchangeable layers.
    """

    STATIC = RESOLUTION_LAYER_STATIC
    # Runtime is named only so forgeries can be rejected fail-closed; this
    # module never constructs runtime-layer results.
    RUNTIME = RESOLUTION_LAYER_RUNTIME


# Deterministic confidence table. Values are fixed; never learned.
_CONFIDENCE_BY_STATUS: Mapping[ResolverStatus, int] = MappingProxyType(
    {
        ResolverStatus.RESOLVED_STATIC: 100,
        ResolverStatus.CANDIDATE: 50,
        ResolverStatus.EXTERNAL: 40,
        ResolverStatus.AMBIGUOUS: 25,
        ResolverStatus.UNSUPPORTED: 5,
        ResolverStatus.UNKNOWN: 10,
        ResolverStatus.UNRESOLVED: 0,
    }
)

_CONFIDENCE_BY_REASON: Mapping[ReasonCode, int] = MappingProxyType(
    {
        ReasonCode.PROVED_INVOCATION_CHAIN: 100,
        ReasonCode.CONNECTOR_BINDING: 100,
        ReasonCode.PROFILE_NEGOTIATED: 100,
        ReasonCode.TRANSPORT_HTTP: 100,
        ReasonCode.TRANSPORT_MCP_P2P: 100,
        ReasonCode.TOOLS_LIST_BINDING: 100,
        ReasonCode.INTERFACE_DESCRIPTOR_BINDING: 100,
        ReasonCode.TOOLS_CALL_BINDING: 100,
        ReasonCode.REGISTRATION_MATCH: 100,
        ReasonCode.ADAPTER_BINDING: 100,
        ReasonCode.IMPLEMENTATION_MATCH: 100,
        ReasonCode.RESULT_MAP_MATCH: 100,
        ReasonCode.ERROR_MAP_MATCH: 100,
        ReasonCode.SCHEMA_PARITY: 100,
        ReasonCode.NAME_ALIAS: 100,
        ReasonCode.MANIFEST_PARITY: 100,
        ReasonCode.CALL_EDGE_PRESENT: 100,
        ReasonCode.SAME_NAME_HELPER: 0,
        ReasonCode.MOCK_IMPLEMENTATION: 0,
        ReasonCode.TEST_SERVER: 0,
        ReasonCode.COPIED_MANIFEST: 0,
        ReasonCode.STATIC_DASHBOARD: 0,
        ReasonCode.LEGACY_FALLBACK: 0,
        ReasonCode.IMPORT_WITHOUT_CALL: 0,
        ReasonCode.NAME_ONLY_MATCH: 25,
        ReasonCode.AMBIGUOUS_REGISTRATION: 25,
        ReasonCode.AMBIGUOUS_ADAPTER: 25,
        ReasonCode.AMBIGUOUS_IMPLEMENTATION: 25,
        ReasonCode.AMBIGUOUS_ALIAS: 25,
        ReasonCode.EXTERNAL_PACKAGE: 40,
        ReasonCode.EXTERNAL_TRANSPORT: 40,
        ReasonCode.PROFILE_MISMATCH: 10,
        ReasonCode.TRANSPORT_MISMATCH: 10,
        ReasonCode.SCHEMA_MISMATCH: 10,
        ReasonCode.VERSION_MISMATCH: 10,
        ReasonCode.MANIFEST_DRIFT: 10,
        ReasonCode.MISSING_HOP: 0,
        ReasonCode.MISSING_REGISTRATION: 0,
        ReasonCode.MISSING_CALL_EDGE: 0,
        ReasonCode.UNSUPPORTED_CONSTRUCT: 5,
        ReasonCode.EVIDENCE_REQUIRED: 0,
        ReasonCode.ALREADY_RESOLVED: 100,
        ReasonCode.NO_TARGET: 0,
        ReasonCode.DYNAMIC_DISPATCH: 25,
    }
)

_STAGE_TO_REASON: Mapping[PathStage, ReasonCode] = MappingProxyType(
    {
        PathStage.CALLER: ReasonCode.CALL_EDGE_PRESENT,
        PathStage.CONNECTOR: ReasonCode.CONNECTOR_BINDING,
        PathStage.PROFILE_TRANSPORT: ReasonCode.PROFILE_NEGOTIATED,
        PathStage.TOOLS_LIST: ReasonCode.TOOLS_LIST_BINDING,
        PathStage.TOOLS_CALL: ReasonCode.TOOLS_CALL_BINDING,
        PathStage.SERVER_REGISTRY: ReasonCode.REGISTRATION_MATCH,
        PathStage.ADAPTER: ReasonCode.ADAPTER_BINDING,
        PathStage.PACKAGE_IMPLEMENTATION: ReasonCode.IMPLEMENTATION_MATCH,
        PathStage.RESULT_ERROR_MAPPING: ReasonCode.RESULT_MAP_MATCH,
    }
)


def static_resolution_boundary() -> Mapping[str, Any]:
    """Machine-readable split between static resolution and hermetic runtime.

    Static call-path resolution (VFS-G060) and hermetic runtime conformance
    (VFS-G061) share inventory vocabulary but never share claim authority.
    Runtime witnesses may *supplement* static resolution; static results never
    claim runtime or replace hermetic observations.
    """

    return MappingProxyType(
        {
            "resolution_layer": ResolutionLayer.STATIC.value,
            "claim_level": STATIC_RESOLUTION_CLAIM_LEVEL.value,
            "claims_runtime_conformance": False,
            "claims_hermetic_runtime": False,
            "static_goal_id": STATIC_RESOLUTION_GOAL_ID,
            "defers_runtime_conformance_to_goal": HERMETIC_RUNTIME_CHILD_GOAL_ID,
            "defers_runtime_claim_level": HERMETIC_RUNTIME_CLAIM_LEVEL.value,
            "defers_runtime_evidence": EVIDENCE_RUNTIME_WITNESS,
            "evidence_kinds": list(STATIC_EVIDENCE_KINDS),
            "excluded_evidence_kinds": list(EXCLUDED_RUNTIME_EVIDENCE_KINDS),
            "resolver_version": RESOLVER_VERSION,
            "opens_network": False,
            "dispatches_adapters": False,
            "emits_runtime_receipts": False,
        }
    )


def _reject_runtime_layer_claim(
    payload: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    """Fail closed when a static artifact forges runtime authority."""

    layer = payload.get("resolution_layer")
    if layer is not None and str(layer).strip() not in ("", ResolutionLayer.STATIC.value):
        raise MCPlusPlusResolverError(
            f"{artifact_name} resolution_layer must be "
            f"{ResolutionLayer.STATIC.value!r} (got {layer!r}); "
            "hermetic runtime conformance is VFS-G061"
        )

    claims_runtime = payload.get("claims_runtime_conformance")
    if claims_runtime is True or (
        isinstance(claims_runtime, str)
        and claims_runtime.strip().lower() in {"true", "1", "yes"}
    ):
        raise MCPlusPlusResolverError(
            f"{artifact_name} cannot claim runtime conformance; "
            "use mcplusplus_runtime_witness / VFS-G061"
        )

    is_runtime = payload.get("is_runtime_witnessed")
    if is_runtime is True or (
        isinstance(is_runtime, str)
        and is_runtime.strip().lower() in {"true", "1", "yes"}
    ):
        raise MCPlusPlusResolverError(
            f"{artifact_name} cannot set is_runtime_witnessed; "
            "static resolution never carries runtime authority"
        )

    claim_level = payload.get("claim_level")
    if claim_level is not None:
        text = str(claim_level).strip()
        if text == HERMETIC_RUNTIME_CLAIM_LEVEL.value:
            raise MCPlusPlusResolverError(
                f"{artifact_name} cannot assert claim_level "
                f"{HERMETIC_RUNTIME_CLAIM_LEVEL.value!r}"
            )
        if text and text not in {
            ClaimLevel.OBSERVED_SYNTAX.value,
            STATIC_RESOLUTION_CLAIM_LEVEL.value,
        }:
            # Allow only static-compatible claim levels on this product surface.
            raise MCPlusPlusResolverError(
                f"{artifact_name} claim_level {text!r} is not admitted for "
                "static MCP++ resolution"
            )

    kinds = payload.get("evidence_kinds")
    if kinds is not None:
        for item in kinds:
            if str(item) in EXCLUDED_RUNTIME_EVIDENCE_KINDS:
                raise MCPlusPlusResolverError(
                    f"{artifact_name} cannot include runtime evidence kind "
                    f"{item!r}"
                )

    evidence_kind = payload.get("evidence_kind")
    if evidence_kind is not None and str(evidence_kind) in EXCLUDED_RUNTIME_EVIDENCE_KINDS:
        raise MCPlusPlusResolverError(
            f"{artifact_name} cannot use runtime evidence kind "
            f"{evidence_kind!r}"
        )


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise MCPlusPlusResolverError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise MCPlusPlusResolverError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
        raise MCPlusPlusResolverBoundsError(f"{name} exceeds label bound")
    return text


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise MCPlusPlusResolverError(f"unsupported {label}: {text!r}") from exc


def _mapping(value: Any, name: str, *, max_bytes: int = DEFAULT_MAX_NOTES_BYTES) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise MCPlusPlusResolverError(f"{name} must be a mapping")
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise MCPlusPlusResolverError(f"{name} keys must be strings")
        if isinstance(item, (str, bool, int)) or item is None:
            plain[key] = item
        elif isinstance(item, Enum):
            plain[key] = item.value
        elif isinstance(item, Mapping):
            plain[key] = dict(_mapping(item, f"{name}.{key}", max_bytes=max_bytes))
        elif isinstance(item, (list, tuple)):
            plain[key] = [
                (
                    dict(_mapping(entry, f"{name}.{key}[]", max_bytes=max_bytes))
                    if isinstance(entry, Mapping)
                    else (
                        entry.value
                        if isinstance(entry, Enum)
                        else entry
                    )
                )
                for entry in item
            ]
        else:
            raise MCPlusPlusResolverError(
                f"{name}.{key} has unsupported type {type(item).__name__}"
            )
    encoded = canonical_program_json(plain).encode("utf-8")
    if len(encoded) > max_bytes:
        raise MCPlusPlusResolverBoundsError(f"{name} exceeds notes bound")
    return MappingProxyType(dict(sorted(plain.items())))


def _schema_payload(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return MappingProxyType({})
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPlusPlusResolverError(f"{name} is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise MCPlusPlusResolverError(f"{name} JSON must be an object")
        return _mapping(decoded, name, max_bytes=DEFAULT_MAX_SCHEMA_BYTES)
    return _mapping(value, name, max_bytes=DEFAULT_MAX_SCHEMA_BYTES)


def confidence_for(
    status: ResolverStatus | str,
    reason_code: ReasonCode | str,
) -> int:
    """Return deterministic confidence for a status/reason pair.

    Confidence is never learned or tuned for coverage. The status baseline is
    an upper bound; a reason code may only lower confidence.
    """

    status_enum = _enum(status, ResolverStatus, "resolver_status")
    reason_enum = _enum(reason_code, ReasonCode, "reason_code")
    status_conf = int(_CONFIDENCE_BY_STATUS[status_enum])
    reason_conf = int(_CONFIDENCE_BY_REASON.get(reason_enum, status_conf))
    return min(status_conf, reason_conf)


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name for cross-language comparison."""

    text = _text(name, "tool_name", required=False)
    if not text:
        return ""
    # Hierarchical aliases: category.tool and category/tool are equivalent.
    text = text.replace("/", ".")
    return text.strip().lower()


def split_hierarchical_alias(name: str) -> tuple[str, str] | None:
    """Split ``category.tool`` / ``category/tool`` into parts, or ``None``."""

    text = _text(name, "alias", required=False)
    if not text:
        return None
    match = _HIERARCHICAL_ALIAS_RE.match(text.replace("/", "."))
    if match is None:
        return None
    return match.group("category"), match.group("tool")


def tool_name_aliases(name: str) -> tuple[str, ...]:
    """Return the closed alias set for one tool name (deterministic order)."""

    text = _text(name, "tool_name", required=False)
    if not text:
        return ()
    aliases: list[str] = [text]
    normalized = normalize_tool_name(text)
    if normalized and normalized != text:
        aliases.append(normalized)
    parts = split_hierarchical_alias(text)
    if parts is not None:
        category, tool = parts
        for form in (f"{category}.{tool}", f"{category}/{tool}", tool):
            if form not in aliases:
                aliases.append(form)
            low = form.lower()
            if low not in aliases:
                aliases.append(low)
    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in aliases:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def schema_fingerprint(schema: Mapping[str, Any] | None) -> str:
    """Content identity of a JSON Schema payload (empty for missing)."""

    if not schema:
        return ""
    return content_identity(
        {
            "schema": "mcplusplus-json-schema-fingerprint@1",
            "payload": dict(_schema_payload(schema, "schema")),
        }
    )


def classify_non_invocation(
    *,
    role: ArtifactRole | str,
    path: str = "",
    qualified_name: str = "",
    markers: Sequence[str] = (),
    record: Mapping[str, Any] | None = None,
) -> ReasonCode | None:
    """Return a rejection reason when an artifact cannot prove invocation."""

    role_enum = _enum(role, ArtifactRole, "role")
    if role_enum.value in _NON_INVOCATION_ROLES:
        return {
            ArtifactRole.MOCK: ReasonCode.MOCK_IMPLEMENTATION,
            ArtifactRole.TEST_SERVER: ReasonCode.TEST_SERVER,
            ArtifactRole.COPIED_MANIFEST: ReasonCode.COPIED_MANIFEST,
            ArtifactRole.STATIC_DASHBOARD: ReasonCode.STATIC_DASHBOARD,
            ArtifactRole.LEGACY_FALLBACK: ReasonCode.LEGACY_FALLBACK,
            ArtifactRole.LOCAL_HELPER: ReasonCode.SAME_NAME_HELPER,
        }[role_enum]

    path_low = path.lower().replace("\\", "/")
    q_low = qualified_name.lower()
    marker_text = " ".join(str(item).lower() for item in markers)
    record = record or {}
    record_bits: list[str] = []
    for key in ("kind", "source", "backend", "implementation_kind", "dispatch"):
        value = record.get(key)
        if isinstance(value, str):
            record_bits.append(value.lower())
    joined = " ".join([path_low, q_low, marker_text, *record_bits])

    # Test-tree paths outrank generic mock tokens such as "fake" inside fixtures.
    under_test_tree = (
        "/test/" in f"/{path_low}/"
        or "/tests/" in f"/{path_low}/"
        or path_low.startswith("test/")
        or path_low.startswith("tests/")
        or "/__tests__/" in f"/{path_low}/"
    )
    if role_enum in {
        ArtifactRole.IMPLEMENTATION,
        ArtifactRole.ADAPTER,
        ArtifactRole.REGISTRATION,
        ArtifactRole.TEST_SERVER,
        ArtifactRole.TOOL_CALL_SITE,
    }:
        if under_test_tree or any(
            token in joined for token in ("fixture", "conftest")
        ):
            return ReasonCode.TEST_SERVER

    if any(marker in joined for marker in _MOCK_MARKERS):
        return ReasonCode.MOCK_IMPLEMENTATION
    if any(marker in joined for marker in _DASHBOARD_MARKERS):
        return ReasonCode.STATIC_DASHBOARD
    if any(marker in joined for marker in _FALLBACK_MARKERS):
        return ReasonCode.LEGACY_FALLBACK
    if "local_helper" in joined or "same_name_helper" in joined:
        return ReasonCode.SAME_NAME_HELPER
    return None


def _transport_reason(transport: TransportKind) -> ReasonCode:
    if transport is TransportKind.HTTP:
        return ReasonCode.TRANSPORT_HTTP
    if transport is TransportKind.MCP_P2P:
        return ReasonCode.TRANSPORT_MCP_P2P
    return ReasonCode.EXTERNAL_TRANSPORT


@dataclass(frozen=True)
class PathEvidence:
    """Required provenance for one hop or path decision."""

    rule_id: str
    producer: str
    blob_cid: str
    forest_id: str
    span: SourceSpan = field(default_factory=SourceSpan)
    source_record_key: str = ""
    target_record_key: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "evidence.rule_id"))
        object.__setattr__(self, "producer", _text(self.producer, "evidence.producer"))
        object.__setattr__(self, "blob_cid", _text(self.blob_cid, "evidence.blob_cid"))
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "evidence.forest_id")
        )
        span = (
            self.span
            if isinstance(self.span, SourceSpan)
            else SourceSpan.from_dict(self.span)
        )
        object.__setattr__(self, "span", span)
        object.__setattr__(
            self,
            "source_record_key",
            _text(self.source_record_key, "evidence.source_record_key", required=False),
        )
        object.__setattr__(
            self,
            "target_record_key",
            _text(self.target_record_key, "evidence.target_record_key", required=False),
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "evidence.notes"))

    @property
    def evidence_id(self) -> str:
        return "mpevid-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_PATH_EVIDENCE_SCHEMA,
            "rule_id": self.rule_id,
            "producer": self.producer,
            "blob_cid": self.blob_cid,
            "forest_id": self.forest_id,
            "span": self.span.to_dict(),
            "source_record_key": self.source_record_key,
            "target_record_key": self.target_record_key,
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathEvidence":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("evidence payload must be a mapping")
        return cls(
            rule_id=str(payload.get("rule_id") or ""),
            producer=str(payload.get("producer") or ""),
            blob_cid=str(payload.get("blob_cid") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            span=SourceSpan.from_dict(payload.get("span")),
            source_record_key=str(payload.get("source_record_key") or ""),
            target_record_key=str(payload.get("target_record_key") or ""),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class InventoryArtifact:
    """One closed inventory record used during MCP++ path resolution."""

    artifact_id: str
    role: ArtifactRole
    name: str
    language: str = ""
    package: str = ""
    module_path: str = ""
    qualified_name: str = ""
    server_name: str = ""
    transport: TransportKind = TransportKind.UNKNOWN
    profiles: tuple[str, ...] = ()
    tool_name: str = ""
    alias_of: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    error_codes: tuple[str, ...] = ()
    version: str = ""
    path: str = ""
    blob_cid: str = ""
    forest_id: str = ""
    has_call_edge: bool = False
    has_import_edge: bool = False
    is_external: bool = False
    markers: tuple[str, ...] = ()
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _text(self.artifact_id, "artifact_id")
        )
        object.__setattr__(self, "role", _enum(self.role, ArtifactRole, "role"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(
            self, "language", _text(self.language, "language", required=False)
        )
        object.__setattr__(
            self, "package", _text(self.package, "package", required=False)
        )
        object.__setattr__(
            self, "module_path", _text(self.module_path, "module_path", required=False)
        )
        object.__setattr__(
            self,
            "qualified_name",
            _text(self.qualified_name, "qualified_name", required=False)
            or self.name,
        )
        object.__setattr__(
            self, "server_name", _text(self.server_name, "server_name", required=False)
        )
        object.__setattr__(
            self,
            "transport",
            _enum(self.transport, TransportKind, "transport"),
        )
        profiles = tuple(
            sorted(
                {
                    _text(item, "profile")
                    for item in (self.profiles or ())
                    if str(item or "").strip()
                }
            )
        )
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(
            self, "alias_of", _text(self.alias_of, "alias_of", required=False)
        )
        object.__setattr__(
            self,
            "input_schema",
            _schema_payload(self.input_schema, "input_schema"),
        )
        object.__setattr__(
            self,
            "output_schema",
            _schema_payload(self.output_schema, "output_schema"),
        )
        object.__setattr__(
            self,
            "error_codes",
            tuple(
                sorted(
                    {
                        _text(item, "error_code")
                        for item in (self.error_codes or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self, "version", _text(self.version, "version", required=False)
        )
        object.__setattr__(self, "path", _text(self.path, "path", required=False))
        object.__setattr__(
            self, "blob_cid", _text(self.blob_cid, "blob_cid", required=False)
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", required=False)
        )
        if not isinstance(self.has_call_edge, bool):
            raise MCPlusPlusResolverError("has_call_edge must be a boolean")
        if not isinstance(self.has_import_edge, bool):
            raise MCPlusPlusResolverError("has_import_edge must be a boolean")
        if not isinstance(self.is_external, bool):
            raise MCPlusPlusResolverError("is_external must be a boolean")
        object.__setattr__(
            self,
            "markers",
            tuple(
                sorted(
                    {
                        _text(item, "marker")
                        for item in (self.markers or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        object.__setattr__(self, "record", _mapping(self.record, "artifact.record"))

    @property
    def content_id(self) -> str:
        return "mpart-" + content_identity(self._identity_payload())

    @property
    def effective_tool_name(self) -> str:
        return self.tool_name or self.name

    @property
    def non_invocation_reason(self) -> ReasonCode | None:
        return classify_non_invocation(
            role=self.role,
            path=self.path,
            qualified_name=self.qualified_name,
            markers=self.markers,
            record=self.record,
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_ARTIFACT_SCHEMA,
            "artifact_id": self.artifact_id,
            "role": self.role.value,
            "name": self.name,
            "language": self.language,
            "package": self.package,
            "module_path": self.module_path,
            "qualified_name": self.qualified_name,
            "server_name": self.server_name,
            "transport": self.transport.value,
            "profiles": list(self.profiles),
            "tool_name": self.tool_name,
            "alias_of": self.alias_of,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "error_codes": list(self.error_codes),
            "version": self.version,
            "path": self.path,
            "blob_cid": self.blob_cid,
            "forest_id": self.forest_id,
            "has_call_edge": self.has_call_edge,
            "has_import_edge": self.has_import_edge,
            "is_external": self.is_external,
            "markers": list(self.markers),
            "record": dict(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "content_id": self.content_id,
            "effective_tool_name": self.effective_tool_name,
            "input_schema_fingerprint": schema_fingerprint(self.input_schema),
            "output_schema_fingerprint": schema_fingerprint(self.output_schema),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InventoryArtifact":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("artifact payload must be a mapping")
        return cls(
            artifact_id=str(payload.get("artifact_id") or ""),
            role=payload.get("role", ArtifactRole.IMPLEMENTATION.value),
            name=str(payload.get("name") or ""),
            language=str(payload.get("language") or ""),
            package=str(payload.get("package") or ""),
            module_path=str(payload.get("module_path") or ""),
            qualified_name=str(payload.get("qualified_name") or ""),
            server_name=str(payload.get("server_name") or ""),
            transport=payload.get("transport", TransportKind.UNKNOWN.value),
            profiles=tuple(payload.get("profiles") or ()),
            tool_name=str(payload.get("tool_name") or ""),
            alias_of=str(payload.get("alias_of") or ""),
            input_schema=payload.get("input_schema") or {},
            output_schema=payload.get("output_schema") or {},
            error_codes=tuple(payload.get("error_codes") or ()),
            version=str(payload.get("version") or ""),
            path=str(payload.get("path") or ""),
            blob_cid=str(payload.get("blob_cid") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            has_call_edge=bool(payload.get("has_call_edge", False)),
            has_import_edge=bool(payload.get("has_import_edge", False)),
            is_external=bool(payload.get("is_external", False)),
            markers=tuple(payload.get("markers") or ()),
            record=payload.get("record") or {},
        )


@dataclass(frozen=True)
class MCPlusPlusInventory:
    """Closed catalogs of MCP++ artifacts available for resolution."""

    forest_id: str
    artifacts: tuple[InventoryArtifact, ...] = ()
    required_profiles: tuple[str, ...] = ()
    admitted_transports: tuple[TransportKind, ...] = (
        TransportKind.HTTP,
        TransportKind.MCP_P2P,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "forest_id", _text(self.forest_id, "forest_id"))
        artifacts = tuple(
            item
            if isinstance(item, InventoryArtifact)
            else InventoryArtifact.from_dict(item)
            for item in (self.artifacts or ())
        )
        if len(artifacts) > DEFAULT_MAX_ARTIFACTS:
            raise MCPlusPlusResolverBoundsError("too many inventory artifacts")
        # Stable order by artifact_id then role then content_id.
        ordered = tuple(
            sorted(
                artifacts,
                key=lambda item: (
                    item.artifact_id,
                    item.role.value,
                    item.content_id,
                ),
            )
        )
        # Reject duplicate artifact_id values.
        seen: set[str] = set()
        for item in ordered:
            if item.artifact_id in seen:
                raise MCPlusPlusResolverError(
                    f"duplicate inventory artifact_id: {item.artifact_id!r}"
                )
            seen.add(item.artifact_id)
        object.__setattr__(self, "artifacts", ordered)
        object.__setattr__(
            self,
            "required_profiles",
            tuple(
                sorted(
                    {
                        _text(item, "required_profile")
                        for item in (self.required_profiles or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        transports = tuple(
            _enum(item, TransportKind, "admitted_transport")
            for item in (self.admitted_transports or ())
        )
        object.__setattr__(
            self,
            "admitted_transports",
            tuple(sorted(set(transports), key=lambda item: item.value)),
        )

    @property
    def inventory_id(self) -> str:
        return "mpinv-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_INVENTORY_SCHEMA,
            "forest_id": self.forest_id,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "required_profiles": list(self.required_profiles),
            "admitted_transports": [item.value for item in self.admitted_transports],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "inventory_id": self.inventory_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCPlusPlusInventory":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("inventory payload must be a mapping")
        return cls(
            forest_id=str(payload.get("forest_id") or ""),
            artifacts=tuple(payload.get("artifacts") or ()),
            required_profiles=tuple(payload.get("required_profiles") or ()),
            admitted_transports=tuple(
                payload.get("admitted_transports")
                or (TransportKind.HTTP, TransportKind.MCP_P2P)
            ),
        )

    def by_role(self, role: ArtifactRole | str) -> tuple[InventoryArtifact, ...]:
        role_enum = _enum(role, ArtifactRole, "role")
        return tuple(item for item in self.artifacts if item.role is role_enum)

    def find(
        self,
        *,
        role: ArtifactRole | str | None = None,
        name: str = "",
        tool_name: str = "",
        server_name: str = "",
        package: str = "",
    ) -> tuple[InventoryArtifact, ...]:
        role_enum = _enum(role, ArtifactRole, "role") if role is not None else None
        name_aliases = set(tool_name_aliases(name)) if name else set()
        tool_aliases = set(tool_name_aliases(tool_name)) if tool_name else set()
        results: list[InventoryArtifact] = []
        for item in self.artifacts:
            if role_enum is not None and item.role is not role_enum:
                continue
            if server_name and item.server_name != server_name:
                continue
            if package and item.package != package:
                continue
            if name:
                candidates = {
                    item.name,
                    item.qualified_name,
                    item.effective_tool_name,
                    *tool_name_aliases(item.effective_tool_name),
                    *tool_name_aliases(item.name),
                }
                if not (name_aliases & candidates):
                    continue
            if tool_name:
                candidates = {
                    item.effective_tool_name,
                    item.name,
                    item.alias_of,
                    *tool_name_aliases(item.effective_tool_name),
                    *tool_name_aliases(item.name),
                    *tool_name_aliases(item.alias_of),
                }
                if not (tool_aliases & candidates):
                    continue
            results.append(item)
        return tuple(results)


@dataclass(frozen=True)
class PathHop:
    """One resolved hop in an MCP++ call path."""

    stage: PathStage
    status: ResolverStatus
    reason_code: ReasonCode
    confidence: int
    source_ref: str = ""
    target_ref: str = ""
    artifact_ids: tuple[str, ...] = ()
    transport: TransportKind = TransportKind.UNKNOWN
    profiles: tuple[str, ...] = ()
    evidence: tuple[PathEvidence, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(self.stage, PathStage, "stage"))
        object.__setattr__(
            self, "status", _enum(self.status, ResolverStatus, "status")
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, ReasonCode, "reason_code"),
        )
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, int):
            raise MCPlusPlusResolverError("confidence must be an integer")
        expected = confidence_for(self.status, self.reason_code)
        if self.confidence != expected:
            raise MCPlusPlusResolverError(
                f"confidence {self.confidence} is not deterministic for "
                f"{self.status.value}/{self.reason_code.value} (expected {expected})"
            )
        object.__setattr__(
            self, "source_ref", _text(self.source_ref, "source_ref", required=False)
        )
        object.__setattr__(
            self, "target_ref", _text(self.target_ref, "target_ref", required=False)
        )
        object.__setattr__(
            self,
            "artifact_ids",
            tuple(
                _text(item, "artifact_id")
                for item in (self.artifact_ids or ())
                if str(item or "").strip()
            ),
        )
        object.__setattr__(
            self,
            "transport",
            _enum(self.transport, TransportKind, "transport"),
        )
        object.__setattr__(
            self,
            "profiles",
            tuple(
                sorted(
                    {
                        _text(item, "profile")
                        for item in (self.profiles or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        evidence = tuple(
            item if isinstance(item, PathEvidence) else PathEvidence.from_dict(item)
            for item in (self.evidence or ())
        )
        if not evidence:
            raise MissingPathEvidenceError(
                f"hop {self.stage.value} requires evidence "
                f"(status={self.status.value}, reason={self.reason_code.value})"
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "notes", _mapping(self.notes, "hop.notes"))
        if (
            self.status is ResolverStatus.RESOLVED_STATIC
            and not self.target_ref
            and self.stage is not PathStage.CALLER
        ):
            # Caller may be the origin; every other static hop needs a target.
            if self.stage is not PathStage.PROFILE_TRANSPORT:
                raise MCPlusPlusResolverError(
                    f"resolved_static hop {self.stage.value} requires target_ref"
                )

    @property
    def hop_id(self) -> str:
        return "mphop-" + content_identity(self._identity_payload())

    @property
    def is_frontier(self) -> bool:
        return self.status.frontier

    @property
    def proves_invocation(self) -> bool:
        return self.status is ResolverStatus.RESOLVED_STATIC

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_PATH_HOP_SCHEMA,
            "stage": self.stage.value,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "artifact_ids": list(self.artifact_ids),
            "transport": self.transport.value,
            "profiles": list(self.profiles),
            "evidence": [item.to_dict() for item in self.evidence],
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "hop_id": self.hop_id,
            "is_frontier": self.is_frontier,
            "proves_invocation": self.proves_invocation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathHop":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("hop payload must be a mapping")
        return cls(
            stage=payload.get("stage", PathStage.CALLER.value),
            status=payload.get("status", ResolverStatus.UNRESOLVED.value),
            reason_code=payload.get("reason_code", ReasonCode.NO_TARGET.value),
            confidence=int(payload.get("confidence") or 0),
            source_ref=str(payload.get("source_ref") or ""),
            target_ref=str(payload.get("target_ref") or ""),
            artifact_ids=tuple(payload.get("artifact_ids") or ()),
            transport=payload.get("transport", TransportKind.UNKNOWN.value),
            profiles=tuple(payload.get("profiles") or ()),
            evidence=tuple(payload.get("evidence") or ()),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class ManifestDriftWitness:
    """Minimal witness that manifests / names / schemas disagree."""

    drift_kind: DriftKind
    tool_name: str
    left_ref: str
    right_ref: str
    left_value: str = ""
    right_value: str = ""
    evidence: tuple[PathEvidence, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "drift_kind", _enum(self.drift_kind, DriftKind, "drift_kind")
        )
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(self, "left_ref", _text(self.left_ref, "left_ref"))
        object.__setattr__(self, "right_ref", _text(self.right_ref, "right_ref"))
        object.__setattr__(
            self, "left_value", _text(self.left_value, "left_value", required=False)
        )
        object.__setattr__(
            self, "right_value", _text(self.right_value, "right_value", required=False)
        )
        evidence = tuple(
            item if isinstance(item, PathEvidence) else PathEvidence.from_dict(item)
            for item in (self.evidence or ())
        )
        if not evidence:
            raise MissingPathEvidenceError("manifest drift witness requires evidence")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "notes", _mapping(self.notes, "drift.notes"))

    @property
    def witness_id(self) -> str:
        return "mpdrift-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_MANIFEST_DRIFT_SCHEMA,
            "drift_kind": self.drift_kind.value,
            "tool_name": self.tool_name,
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "evidence": [item.to_dict() for item in self.evidence],
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "witness_id": self.witness_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestDriftWitness":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("drift payload must be a mapping")
        return cls(
            drift_kind=payload.get("drift_kind", DriftKind.NAME_MISMATCH.value),
            tool_name=str(payload.get("tool_name") or ""),
            left_ref=str(payload.get("left_ref") or ""),
            right_ref=str(payload.get("right_ref") or ""),
            left_value=str(payload.get("left_value") or ""),
            right_value=str(payload.get("right_value") or ""),
            evidence=tuple(payload.get("evidence") or ()),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class FrontierItem:
    """Explicit ambiguous / external / unknown frontier element."""

    element_id: str
    element_kind: str
    status: ResolverStatus
    reason_code: ReasonCode
    stage: PathStage | None = None
    qualified_name: str = ""
    tool_name: str = ""
    candidates: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "element_id", _text(self.element_id, "element_id")
        )
        object.__setattr__(
            self, "element_kind", _text(self.element_kind, "element_kind")
        )
        object.__setattr__(
            self, "status", _enum(self.status, ResolverStatus, "status")
        )
        if not self.status.frontier:
            raise MCPlusPlusResolverError(
                f"frontier item status must be frontier, got {self.status.value}"
            )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, ReasonCode, "reason_code"),
        )
        if self.stage is None:
            object.__setattr__(self, "stage", None)
        else:
            object.__setattr__(
                self, "stage", _enum(self.stage, PathStage, "stage")
            )
        object.__setattr__(
            self,
            "qualified_name",
            _text(self.qualified_name, "qualified_name", required=False),
        )
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(
            self,
            "candidates",
            tuple(
                _text(item, "candidate")
                for item in (self.candidates or ())
                if str(item or "").strip()
            ),
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "frontier.notes"))

    @property
    def frontier_id(self) -> str:
        return "mpfr-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_FRONTIER_SCHEMA,
            "element_id": self.element_id,
            "element_kind": self.element_kind,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "stage": self.stage.value if self.stage is not None else "",
            "qualified_name": self.qualified_name,
            "tool_name": self.tool_name,
            "candidates": list(self.candidates),
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "frontier_id": self.frontier_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrontierItem":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("frontier payload must be a mapping")
        stage_raw = payload.get("stage")
        stage = stage_raw if stage_raw else None
        return cls(
            element_id=str(payload.get("element_id") or ""),
            element_kind=str(payload.get("element_kind") or ""),
            status=payload.get("status", ResolverStatus.UNKNOWN.value),
            reason_code=payload.get("reason_code", ReasonCode.NO_TARGET.value),
            stage=stage,
            qualified_name=str(payload.get("qualified_name") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            candidates=tuple(payload.get("candidates") or ()),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class MCPlusPlusCallPath:
    """One fully traced MCP++ call path with hop-level evidence."""

    path_name: str
    forest_id: str
    tool_name: str
    hops: tuple[PathHop, ...]
    verdict: PathVerdict
    caller_ref: str = ""
    connector_ref: str = ""
    server_name: str = ""
    transport: TransportKind = TransportKind.UNKNOWN
    profiles: tuple[str, ...] = ()
    implementation_ref: str = ""
    language_names: Mapping[str, str] = field(default_factory=dict)
    drift_witnesses: tuple[ManifestDriftWitness, ...] = ()
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_name", _text(self.path_name, "path_name"))
        object.__setattr__(self, "forest_id", _text(self.forest_id, "forest_id"))
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        hops = tuple(
            item if isinstance(item, PathHop) else PathHop.from_dict(item)
            for item in (self.hops or ())
        )
        if not hops:
            raise MCPlusPlusResolverError("call path requires at least one hop")
        if len(hops) > DEFAULT_MAX_HOPS:
            raise MCPlusPlusResolverBoundsError("too many hops")
        object.__setattr__(self, "hops", hops)
        object.__setattr__(
            self, "verdict", _enum(self.verdict, PathVerdict, "verdict")
        )
        object.__setattr__(
            self, "caller_ref", _text(self.caller_ref, "caller_ref", required=False)
        )
        object.__setattr__(
            self,
            "connector_ref",
            _text(self.connector_ref, "connector_ref", required=False),
        )
        object.__setattr__(
            self, "server_name", _text(self.server_name, "server_name", required=False)
        )
        object.__setattr__(
            self,
            "transport",
            _enum(self.transport, TransportKind, "transport"),
        )
        object.__setattr__(
            self,
            "profiles",
            tuple(
                sorted(
                    {
                        _text(item, "profile")
                        for item in (self.profiles or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "implementation_ref",
            _text(self.implementation_ref, "implementation_ref", required=False),
        )
        object.__setattr__(
            self,
            "language_names",
            _mapping(self.language_names, "language_names"),
        )
        drift = tuple(
            item
            if isinstance(item, ManifestDriftWitness)
            else ManifestDriftWitness.from_dict(item)
            for item in (self.drift_witnesses or ())
        )
        object.__setattr__(
            self,
            "drift_witnesses",
            tuple(sorted(drift, key=lambda item: item.witness_id)),
        )
        object.__setattr__(self, "record", _mapping(self.record, "path.record"))
        expected = _verdict_from_hops(hops, drift_count=len(self.drift_witnesses))
        if self.verdict is not expected:
            raise MCPlusPlusResolverError(
                f"verdict {self.verdict.value} is not consistent with hops "
                f"(expected {expected.value})"
            )

    @property
    def path_id(self) -> str:
        return "mppath-" + content_identity(self._identity_payload())

    @property
    def is_proved(self) -> bool:
        """True when every hop is statically resolved (not runtime-witnessed)."""

        return self.verdict is PathVerdict.PROVED

    @property
    def is_statically_proved(self) -> bool:
        """Alias emphasizing that proof is inventory-static only."""

        return self.is_proved

    @property
    def is_runtime_witnessed(self) -> bool:
        """Static paths never carry hermetic runtime authority."""

        return False

    @property
    def claim_level(self) -> ClaimLevel:
        """Claim level for a proved static path; never ``runtime_witnessed``."""

        if self.is_proved:
            return STATIC_RESOLUTION_CLAIM_LEVEL
        return ClaimLevel.OBSERVED_SYNTAX

    @property
    def resolution_layer(self) -> ResolutionLayer:
        return ResolutionLayer.STATIC

    @property
    def has_frontier(self) -> bool:
        return any(hop.is_frontier for hop in self.hops)

    def hop_for(self, stage: PathStage | str) -> PathHop | None:
        stage_enum = _enum(stage, PathStage, "stage")
        for hop in self.hops:
            if hop.stage is stage_enum:
                return hop
        return None

    def frontier(self) -> tuple[FrontierItem, ...]:
        items: list[FrontierItem] = []
        for hop in self.hops:
            if not hop.is_frontier:
                continue
            items.append(
                FrontierItem(
                    element_id=hop.hop_id,
                    element_kind=f"hop:{hop.stage.value}",
                    status=hop.status,
                    reason_code=hop.reason_code,
                    stage=hop.stage,
                    qualified_name=hop.target_ref or hop.source_ref,
                    tool_name=self.tool_name,
                    candidates=tuple(hop.artifact_ids),
                    notes=dict(hop.notes),
                )
            )
        return tuple(sorted(items, key=lambda item: item.frontier_id))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_CALL_PATH_SCHEMA,
            "path_name": self.path_name,
            "forest_id": self.forest_id,
            "tool_name": self.tool_name,
            "hops": [item.to_dict() for item in self.hops],
            "verdict": self.verdict.value,
            "caller_ref": self.caller_ref,
            "connector_ref": self.connector_ref,
            "server_name": self.server_name,
            "transport": self.transport.value,
            "profiles": list(self.profiles),
            "implementation_ref": self.implementation_ref,
            "language_names": dict(self.language_names),
            "drift_witnesses": [item.to_dict() for item in self.drift_witnesses],
            "record": dict(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "path_id": self.path_id,
            "is_proved": self.is_proved,
            "is_statically_proved": self.is_statically_proved,
            "is_runtime_witnessed": self.is_runtime_witnessed,
            "has_frontier": self.has_frontier,
            "evidence_kind": EVIDENCE_CALL_PATH,
            "claim_level": self.claim_level.value,
            "resolution_layer": self.resolution_layer.value,
            "claims_runtime_conformance": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCPlusPlusCallPath":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("call path payload must be a mapping")
        _reject_runtime_layer_claim(payload, artifact_name="call path")
        return cls(
            path_name=str(payload.get("path_name") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            hops=tuple(payload.get("hops") or ()),
            verdict=payload.get("verdict", PathVerdict.UNKNOWN.value),
            caller_ref=str(payload.get("caller_ref") or ""),
            connector_ref=str(payload.get("connector_ref") or ""),
            server_name=str(payload.get("server_name") or ""),
            transport=payload.get("transport", TransportKind.UNKNOWN.value),
            profiles=tuple(payload.get("profiles") or ()),
            implementation_ref=str(payload.get("implementation_ref") or ""),
            language_names=payload.get("language_names") or {},
            drift_witnesses=tuple(payload.get("drift_witnesses") or ()),
            record=payload.get("record") or {},
        )


@dataclass(frozen=True)
class MCPlusPlusResolutionResult:
    """Deterministic batch of static MCP++ path resolutions and witnesses.

    Results are always on the static resolution layer. Hermetic runtime
    conformance is never claimed here; it is deferred to VFS-G061.
    """

    forest_id: str
    resolver_version: str
    inventory_id: str
    paths: tuple[MCPlusPlusCallPath, ...] = ()
    drift_witnesses: tuple[ManifestDriftWitness, ...] = ()
    frontiers: tuple[FrontierItem, ...] = ()
    truncated: bool = False
    truncation_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "forest_id", _text(self.forest_id, "forest_id"))
        object.__setattr__(
            self,
            "resolver_version",
            _text(self.resolver_version, "resolver_version"),
        )
        object.__setattr__(
            self, "inventory_id", _text(self.inventory_id, "inventory_id")
        )
        paths = tuple(
            item
            if isinstance(item, MCPlusPlusCallPath)
            else MCPlusPlusCallPath.from_dict(item)
            for item in (self.paths or ())
        )
        object.__setattr__(
            self,
            "paths",
            tuple(sorted(paths, key=lambda item: (item.path_name, item.path_id))),
        )
        if len(self.paths) > DEFAULT_MAX_PATHS:
            raise MCPlusPlusResolverBoundsError("too many paths")
        for path in self.paths:
            if path.is_runtime_witnessed:
                raise MCPlusPlusResolverError(
                    "static resolution result cannot include runtime-witnessed "
                    f"path {path.path_name!r}"
                )
            if path.resolution_layer is not ResolutionLayer.STATIC:
                raise MCPlusPlusResolverError(
                    "static resolution result requires static-layer paths"
                )
        drift = tuple(
            item
            if isinstance(item, ManifestDriftWitness)
            else ManifestDriftWitness.from_dict(item)
            for item in (self.drift_witnesses or ())
        )
        object.__setattr__(
            self,
            "drift_witnesses",
            tuple(sorted(drift, key=lambda item: item.witness_id)),
        )
        if len(self.drift_witnesses) > DEFAULT_MAX_DRIFT_WITNESSES:
            raise MCPlusPlusResolverBoundsError("too many drift witnesses")
        frontiers = tuple(
            item if isinstance(item, FrontierItem) else FrontierItem.from_dict(item)
            for item in (self.frontiers or ())
        )
        object.__setattr__(
            self,
            "frontiers",
            tuple(sorted(frontiers, key=lambda item: item.frontier_id)),
        )
        if len(self.frontiers) > DEFAULT_MAX_FRONTIER_ITEMS:
            raise MCPlusPlusResolverBoundsError("too many frontier items")
        if not isinstance(self.truncated, bool):
            raise MCPlusPlusResolverError("truncated must be a boolean")
        object.__setattr__(
            self,
            "truncation_reason",
            _text(self.truncation_reason, "truncation_reason", required=False),
        )

    @property
    def result_id(self) -> str:
        return "mpres-" + content_identity(self._identity_payload())

    @property
    def resolution_layer(self) -> ResolutionLayer:
        return ResolutionLayer.STATIC

    @property
    def claim_level(self) -> ClaimLevel:
        """Highest claim level this result may assert (static only)."""

        return STATIC_RESOLUTION_CLAIM_LEVEL

    @property
    def claims_runtime_conformance(self) -> bool:
        return False

    @property
    def defers_runtime_to_goal(self) -> str:
        return HERMETIC_RUNTIME_CHILD_GOAL_ID

    def paths_for_tool(self, tool_name: str) -> tuple[MCPlusPlusCallPath, ...]:
        aliases = set(tool_name_aliases(tool_name))
        return tuple(
            path
            for path in self.paths
            if set(tool_name_aliases(path.tool_name)) & aliases
            or path.tool_name == tool_name
        )

    def proved_paths(self) -> tuple[MCPlusPlusCallPath, ...]:
        return tuple(path for path in self.paths if path.is_proved)

    def statically_proved_paths(self) -> tuple[MCPlusPlusCallPath, ...]:
        """Proved under inventory-static evidence only (never runtime)."""

        return self.proved_paths()

    def stats(self) -> Mapping[str, Any]:
        by_verdict: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        for path in self.paths:
            by_verdict[path.verdict.value] = by_verdict.get(path.verdict.value, 0) + 1
            for hop in path.hops:
                by_reason[hop.reason_code.value] = (
                    by_reason.get(hop.reason_code.value, 0) + 1
                )
        return MappingProxyType(
            {
                "path_count": len(self.paths),
                "proved_count": len(self.proved_paths()),
                "statically_proved_count": len(self.statically_proved_paths()),
                "runtime_witnessed_count": 0,
                "drift_count": len(self.drift_witnesses),
                "frontier_count": len(self.frontiers),
                "by_verdict": dict(sorted(by_verdict.items())),
                "by_reason": dict(sorted(by_reason.items())),
                "truncated": self.truncated,
                "resolution_layer": self.resolution_layer.value,
            }
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCPLUSPLUS_RESOLUTION_RESULT_SCHEMA,
            "forest_id": self.forest_id,
            "resolver_version": self.resolver_version,
            "inventory_id": self.inventory_id,
            "paths": [item.to_dict() for item in self.paths],
            "drift_witnesses": [item.to_dict() for item in self.drift_witnesses],
            "frontiers": [item.to_dict() for item in self.frontiers],
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        boundary = static_resolution_boundary()
        return {
            **self._identity_payload(),
            "result_id": self.result_id,
            "stats": dict(self.stats()),
            "evidence_kinds": list(STATIC_EVIDENCE_KINDS),
            "resolution_layer": self.resolution_layer.value,
            "claim_level": self.claim_level.value,
            "claims_runtime_conformance": self.claims_runtime_conformance,
            "defers_runtime_to_goal": self.defers_runtime_to_goal,
            "static_runtime_boundary": dict(boundary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCPlusPlusResolutionResult":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("result payload must be a mapping")
        _reject_runtime_layer_claim(payload, artifact_name="resolution result")
        kinds = payload.get("evidence_kinds")
        if kinds is not None:
            kind_list = [str(item) for item in kinds]
            for excluded in EXCLUDED_RUNTIME_EVIDENCE_KINDS:
                if excluded in kind_list:
                    raise MCPlusPlusResolverError(
                        "static resolution result cannot claim runtime evidence "
                        f"{excluded}"
                    )
            for required in STATIC_EVIDENCE_KINDS:
                if required not in kind_list:
                    raise MCPlusPlusResolverError(
                        f"static resolution result missing evidence kind {required}"
                    )
        return cls(
            forest_id=str(payload.get("forest_id") or ""),
            resolver_version=str(payload.get("resolver_version") or ""),
            inventory_id=str(payload.get("inventory_id") or ""),
            paths=tuple(payload.get("paths") or ()),
            drift_witnesses=tuple(payload.get("drift_witnesses") or ()),
            frontiers=tuple(payload.get("frontiers") or ()),
            truncated=bool(payload.get("truncated", False)),
            truncation_reason=str(payload.get("truncation_reason") or ""),
        )


def _verdict_from_hops(
    hops: Sequence[PathHop],
    *,
    drift_count: int = 0,
) -> PathVerdict:
    if not hops:
        return PathVerdict.UNKNOWN
    statuses = [hop.status for hop in hops]
    reasons = [hop.reason_code for hop in hops]
    if any(
        reason
        in {
            ReasonCode.SAME_NAME_HELPER,
            ReasonCode.MOCK_IMPLEMENTATION,
            ReasonCode.TEST_SERVER,
            ReasonCode.COPIED_MANIFEST,
            ReasonCode.STATIC_DASHBOARD,
            ReasonCode.LEGACY_FALLBACK,
            ReasonCode.IMPORT_WITHOUT_CALL,
        }
        for reason in reasons
    ):
        return PathVerdict.REJECTED
    if any(status is ResolverStatus.UNSUPPORTED for status in statuses):
        return PathVerdict.UNSUPPORTED
    if any(status is ResolverStatus.AMBIGUOUS for status in statuses):
        return PathVerdict.AMBIGUOUS
    if any(status is ResolverStatus.EXTERNAL for status in statuses):
        return PathVerdict.EXTERNAL
    if any(
        status
        in {
            ResolverStatus.UNRESOLVED,
            ResolverStatus.UNKNOWN,
            ResolverStatus.CANDIDATE,
        }
        for status in statuses
    ):
        if all(
            status
            in {
                ResolverStatus.RESOLVED_STATIC,
                ResolverStatus.CANDIDATE,
            }
            for status in statuses
        ) and any(status is ResolverStatus.CANDIDATE for status in statuses):
            return PathVerdict.CANDIDATE
        return PathVerdict.UNKNOWN
    if drift_count > 0:
        # Fully resolved hops with residual manifest drift stay candidate:
        # name binding may be fine while schema/version still drifts.
        return PathVerdict.CANDIDATE
    if all(status is ResolverStatus.RESOLVED_STATIC for status in statuses):
        return PathVerdict.PROVED
    return PathVerdict.UNKNOWN


def make_evidence(
    *,
    rule_id: str,
    blob_cid: str,
    forest_id: str,
    producer: str = RESOLVER_PRODUCER,
    span: SourceSpan | Mapping[str, Any] | None = None,
    source_record_key: str = "",
    target_record_key: str = "",
    notes: Mapping[str, Any] | None = None,
) -> PathEvidence:
    """Construct path evidence with deterministic defaults."""

    return PathEvidence(
        rule_id=rule_id,
        producer=producer,
        blob_cid=blob_cid,
        forest_id=forest_id,
        span=span if isinstance(span, SourceSpan) else SourceSpan.from_dict(span),
        source_record_key=source_record_key,
        target_record_key=target_record_key,
        notes=notes or {},
    )


def make_hop(
    *,
    stage: PathStage | str,
    status: ResolverStatus | str,
    reason_code: ReasonCode | str,
    evidence: Sequence[PathEvidence | Mapping[str, Any]],
    source_ref: str = "",
    target_ref: str = "",
    artifact_ids: Sequence[str] = (),
    transport: TransportKind | str = TransportKind.UNKNOWN,
    profiles: Sequence[str] = (),
    notes: Mapping[str, Any] | None = None,
) -> PathHop:
    """Construct a hop with deterministic confidence."""

    status_enum = _enum(status, ResolverStatus, "status")
    reason_enum = _enum(reason_code, ReasonCode, "reason_code")
    return PathHop(
        stage=stage,
        status=status_enum,
        reason_code=reason_enum,
        confidence=confidence_for(status_enum, reason_enum),
        source_ref=source_ref,
        target_ref=target_ref,
        artifact_ids=tuple(artifact_ids),
        transport=transport,
        profiles=tuple(profiles),
        evidence=tuple(evidence),
        notes=notes or {},
    )


def make_artifact(
    *,
    artifact_id: str,
    role: ArtifactRole | str,
    name: str,
    **kwargs: Any,
) -> InventoryArtifact:
    """Convenience constructor for inventory artifacts."""

    return InventoryArtifact(
        artifact_id=artifact_id,
        role=role,
        name=name,
        **kwargs,
    )


@dataclass(frozen=True)
class CallPathClaim:
    """A claimed MCP++ path the resolver must try to prove or reject."""

    path_name: str
    tool_name: str
    caller_name: str = ""
    connector_name: str = ""
    server_name: str = ""
    transport: TransportKind = TransportKind.UNKNOWN
    profiles: tuple[str, ...] = ()
    language_names: Mapping[str, str] = field(default_factory=dict)
    require_interface: bool = False
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_name", _text(self.path_name, "path_name"))
        object.__setattr__(self, "tool_name", _text(self.tool_name, "tool_name"))
        object.__setattr__(
            self, "caller_name", _text(self.caller_name, "caller_name", required=False)
        )
        object.__setattr__(
            self,
            "connector_name",
            _text(self.connector_name, "connector_name", required=False),
        )
        object.__setattr__(
            self, "server_name", _text(self.server_name, "server_name", required=False)
        )
        object.__setattr__(
            self,
            "transport",
            _enum(self.transport, TransportKind, "transport"),
        )
        object.__setattr__(
            self,
            "profiles",
            tuple(
                sorted(
                    {
                        _text(item, "profile")
                        for item in (self.profiles or ())
                        if str(item or "").strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "language_names",
            _mapping(self.language_names, "language_names"),
        )
        if not isinstance(self.require_interface, bool):
            raise MCPlusPlusResolverError("require_interface must be a boolean")
        object.__setattr__(self, "record", _mapping(self.record, "claim.record"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_name": self.path_name,
            "tool_name": self.tool_name,
            "caller_name": self.caller_name,
            "connector_name": self.connector_name,
            "server_name": self.server_name,
            "transport": self.transport.value,
            "profiles": list(self.profiles),
            "language_names": dict(self.language_names),
            "require_interface": self.require_interface,
            "record": dict(self.record),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallPathClaim":
        if not isinstance(payload, Mapping):
            raise MCPlusPlusResolverError("claim payload must be a mapping")
        return cls(
            path_name=str(payload.get("path_name") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            caller_name=str(payload.get("caller_name") or ""),
            connector_name=str(payload.get("connector_name") or ""),
            server_name=str(payload.get("server_name") or ""),
            transport=payload.get("transport", TransportKind.UNKNOWN.value),
            profiles=tuple(payload.get("profiles") or ()),
            language_names=payload.get("language_names") or {},
            require_interface=bool(payload.get("require_interface", False)),
            record=payload.get("record") or {},
        )


def _evidence_for_artifact(
    artifact: InventoryArtifact,
    *,
    rule_id: str,
    forest_id: str,
    notes: Mapping[str, Any] | None = None,
) -> PathEvidence:
    return make_evidence(
        rule_id=rule_id,
        blob_cid=artifact.blob_cid or "blob:inventory",
        forest_id=artifact.forest_id or forest_id,
        source_record_key=artifact.artifact_id,
        target_record_key=artifact.qualified_name,
        notes={
            "role": artifact.role.value,
            "path": artifact.path,
            **dict(notes or {}),
        },
    )


def _reject_hop(
    stage: PathStage,
    reason: ReasonCode,
    *,
    forest_id: str,
    source_ref: str = "",
    target_ref: str = "",
    artifact_ids: Sequence[str] = (),
    notes: Mapping[str, Any] | None = None,
) -> PathHop:
    status = ResolverStatus.UNRESOLVED
    if reason in {
        ReasonCode.SAME_NAME_HELPER,
        ReasonCode.MOCK_IMPLEMENTATION,
        ReasonCode.TEST_SERVER,
        ReasonCode.COPIED_MANIFEST,
        ReasonCode.STATIC_DASHBOARD,
        ReasonCode.LEGACY_FALLBACK,
        ReasonCode.IMPORT_WITHOUT_CALL,
    }:
        status = ResolverStatus.UNRESOLVED
    elif reason in {
        ReasonCode.AMBIGUOUS_REGISTRATION,
        ReasonCode.AMBIGUOUS_ADAPTER,
        ReasonCode.AMBIGUOUS_IMPLEMENTATION,
        ReasonCode.AMBIGUOUS_ALIAS,
        ReasonCode.DYNAMIC_DISPATCH,
        ReasonCode.NAME_ONLY_MATCH,
    }:
        status = ResolverStatus.AMBIGUOUS
    elif reason in {
        ReasonCode.EXTERNAL_PACKAGE,
        ReasonCode.EXTERNAL_TRANSPORT,
    }:
        status = ResolverStatus.EXTERNAL
    elif reason in {
        ReasonCode.UNSUPPORTED_CONSTRUCT,
    }:
        status = ResolverStatus.UNSUPPORTED
    elif reason in {
        ReasonCode.PROFILE_MISMATCH,
        ReasonCode.TRANSPORT_MISMATCH,
        ReasonCode.SCHEMA_MISMATCH,
        ReasonCode.VERSION_MISMATCH,
        ReasonCode.MANIFEST_DRIFT,
        ReasonCode.MISSING_HOP,
        ReasonCode.MISSING_REGISTRATION,
        ReasonCode.MISSING_CALL_EDGE,
        ReasonCode.NO_TARGET,
        ReasonCode.EVIDENCE_REQUIRED,
    }:
        status = ResolverStatus.UNKNOWN
    return make_hop(
        stage=stage,
        status=status,
        reason_code=reason,
        evidence=(
            make_evidence(
                rule_id=f"rule:reject:{reason.value}",
                blob_cid="blob:resolver",
                forest_id=forest_id,
                source_record_key=source_ref,
                target_record_key=target_ref,
                notes=notes or {},
            ),
        ),
        source_ref=source_ref,
        target_ref=target_ref,
        artifact_ids=tuple(artifact_ids),
        notes=notes or {},
    )


def _select_unique(
    candidates: Sequence[InventoryArtifact],
    *,
    stage: PathStage,
    forest_id: str,
    source_ref: str,
    ambiguous_reason: ReasonCode,
    missing_reason: ReasonCode,
    require_call_edge: bool = False,
) -> tuple[InventoryArtifact | None, PathHop | None]:
    """Return (artifact, None) on unique bind, or (None, reject_hop)."""

    if not candidates:
        return None, _reject_hop(
            stage,
            missing_reason,
            forest_id=forest_id,
            source_ref=source_ref,
            notes={"candidate_count": 0},
        )

    # Filter non-invocation artifacts first; they never win a unique bind.
    usable: list[InventoryArtifact] = []
    rejected: list[tuple[InventoryArtifact, ReasonCode]] = []
    for item in candidates:
        reason = item.non_invocation_reason
        if reason is not None:
            rejected.append((item, reason))
            continue
        if require_call_edge and not item.has_call_edge:
            if item.has_import_edge:
                rejected.append((item, ReasonCode.IMPORT_WITHOUT_CALL))
            else:
                rejected.append((item, ReasonCode.MISSING_CALL_EDGE))
            continue
        if item.is_external:
            rejected.append((item, ReasonCode.EXTERNAL_PACKAGE))
            continue
        usable.append(item)

    if len(usable) == 1:
        return usable[0], None

    if len(usable) > 1:
        return None, _reject_hop(
            stage,
            ambiguous_reason,
            forest_id=forest_id,
            source_ref=source_ref,
            target_ref=",".join(item.qualified_name for item in usable),
            artifact_ids=[item.artifact_id for item in usable],
            notes={
                "candidate_count": len(usable),
                "candidates": [item.qualified_name for item in usable],
            },
        )

    # Only non-invocation or external candidates remained.
    if rejected:
        # Prefer the strongest rejection reason deterministically.
        priority = [
            ReasonCode.MOCK_IMPLEMENTATION,
            ReasonCode.TEST_SERVER,
            ReasonCode.STATIC_DASHBOARD,
            ReasonCode.COPIED_MANIFEST,
            ReasonCode.LEGACY_FALLBACK,
            ReasonCode.SAME_NAME_HELPER,
            ReasonCode.IMPORT_WITHOUT_CALL,
            ReasonCode.MISSING_CALL_EDGE,
            ReasonCode.EXTERNAL_PACKAGE,
        ]
        reason = rejected[0][1]
        for preferred in priority:
            if any(item[1] is preferred for item in rejected):
                reason = preferred
                break
        return None, _reject_hop(
            stage,
            reason,
            forest_id=forest_id,
            source_ref=source_ref,
            target_ref=rejected[0][0].qualified_name,
            artifact_ids=[item.artifact_id for item, _ in rejected],
            notes={
                "rejected": [
                    {
                        "artifact_id": item.artifact_id,
                        "reason": why.value,
                        "qualified_name": item.qualified_name,
                    }
                    for item, why in rejected
                ]
            },
        )

    return None, _reject_hop(
        stage,
        missing_reason,
        forest_id=forest_id,
        source_ref=source_ref,
    )


class MCPlusPlusContractResolver:
    """Resolve SwissKnife MCP++ call paths against a closed inventory."""

    def __init__(
        self,
        inventory: MCPlusPlusInventory,
        *,
        max_paths: int = DEFAULT_MAX_PATHS,
    ) -> None:
        if not isinstance(inventory, MCPlusPlusInventory):
            raise MCPlusPlusResolverError("inventory must be MCPlusPlusInventory")
        if (
            isinstance(max_paths, bool)
            or not isinstance(max_paths, int)
            or max_paths < 1
            or max_paths > DEFAULT_MAX_PATHS
        ):
            raise MCPlusPlusResolverBoundsError("max_paths out of bounds")
        self._inventory = inventory
        self._max_paths = max_paths
        self._by_id = {item.artifact_id: item for item in inventory.artifacts}

    @property
    def inventory(self) -> MCPlusPlusInventory:
        return self._inventory

    def resolve(
        self,
        claims: Sequence[CallPathClaim | Mapping[str, Any]],
    ) -> MCPlusPlusResolutionResult:
        """Resolve every claim; emit paths, frontiers, and drift witnesses."""

        parsed = tuple(
            item if isinstance(item, CallPathClaim) else CallPathClaim.from_dict(item)
            for item in claims
        )
        paths: list[MCPlusPlusCallPath] = []
        drift: list[ManifestDriftWitness] = []
        frontiers: list[FrontierItem] = []
        truncated = False
        truncation_reason = ""

        for claim in sorted(parsed, key=lambda item: (item.path_name, item.tool_name)):
            if len(paths) >= self._max_paths:
                truncated = True
                truncation_reason = "max_paths"
                break
            path = self.resolve_claim(claim)
            paths.append(path)
            drift.extend(path.drift_witnesses)
            frontiers.extend(path.frontier())

        # Global manifest parity scan across inventory (not only claimed tools).
        global_drift = self.compare_manifests()
        for witness in global_drift:
            if all(witness.witness_id != existing.witness_id for existing in drift):
                drift.append(witness)

        # Explicit frontiers for external-only registrations not claimed.
        for artifact in self._inventory.artifacts:
            if artifact.is_external:
                frontiers.append(
                    FrontierItem(
                        element_id=artifact.artifact_id,
                        element_kind=f"artifact:{artifact.role.value}",
                        status=ResolverStatus.EXTERNAL,
                        reason_code=ReasonCode.EXTERNAL_PACKAGE,
                        qualified_name=artifact.qualified_name,
                        tool_name=artifact.effective_tool_name,
                        notes={"package": artifact.package, "path": artifact.path},
                    )
                )

        # Deduplicate frontiers by frontier_id.
        unique_frontiers: dict[str, FrontierItem] = {}
        for item in frontiers:
            unique_frontiers[item.frontier_id] = item

        return MCPlusPlusResolutionResult(
            forest_id=self._inventory.forest_id,
            resolver_version=RESOLVER_VERSION,
            inventory_id=self._inventory.inventory_id,
            paths=tuple(paths),
            drift_witnesses=tuple(drift),
            frontiers=tuple(unique_frontiers.values()),
            truncated=truncated,
            truncation_reason=truncation_reason,
        )

    def resolve_claim(self, claim: CallPathClaim | Mapping[str, Any]) -> MCPlusPlusCallPath:
        """Trace one claim through every MCP++ path stage."""

        if not isinstance(claim, CallPathClaim):
            claim = CallPathClaim.from_dict(claim)
        forest_id = self._inventory.forest_id
        hops: list[PathHop] = []
        drift: list[ManifestDriftWitness] = []

        # --- caller ---
        caller_hop, caller = self._resolve_caller(claim)
        hops.append(caller_hop)

        # --- connector ---
        connector_hop, connector = self._resolve_connector(claim, caller)
        hops.append(connector_hop)

        # --- profile / transport ---
        transport_hop, transport, profiles = self._resolve_profile_transport(
            claim, connector
        )
        hops.append(transport_hop)

        # --- tools/list or interface ---
        list_hop, listed = self._resolve_tools_list(claim, require_interface=claim.require_interface)
        hops.append(list_hop)

        # --- tools/call ---
        call_hop, call_site = self._resolve_tools_call(claim, connector)
        hops.append(call_hop)

        # --- server registry ---
        reg_hop, registration = self._resolve_registration(claim)
        hops.append(reg_hop)

        # --- adapter ---
        adapter_hop, adapter = self._resolve_adapter(claim, registration)
        hops.append(adapter_hop)

        # --- package implementation ---
        impl_hop, implementation = self._resolve_implementation(claim, adapter, registration)
        hops.append(impl_hop)

        # --- result / error mapping ---
        map_hop, map_drift = self._resolve_result_error_mapping(
            claim,
            listed=listed,
            registration=registration,
            implementation=implementation,
            call_site=call_site,
        )
        hops.append(map_hop)
        drift.extend(map_drift)

        # Cross-language name parity for this tool.
        drift.extend(self._language_name_drift(claim, registration))

        # Manifest parity for this tool.
        drift.extend(self._manifest_drift_for_tool(claim.tool_name, claim.server_name))

        language_names = dict(claim.language_names)
        if registration is not None and "python" not in language_names:
            language_names["python"] = registration.effective_tool_name
        if call_site is not None and call_site.language:
            language_names.setdefault(call_site.language, call_site.effective_tool_name)
        if connector is not None and connector.language:
            language_names.setdefault(connector.language, claim.tool_name)

        verdict = _verdict_from_hops(hops, drift_count=len(drift))
        return MCPlusPlusCallPath(
            path_name=claim.path_name,
            forest_id=forest_id,
            tool_name=claim.tool_name,
            hops=tuple(hops),
            verdict=verdict,
            caller_ref=(
                caller.qualified_name
                if caller is not None
                else claim.caller_name
            ),
            connector_ref=(
                connector.qualified_name
                if connector is not None
                else claim.connector_name
            ),
            server_name=claim.server_name
            or (
                registration.server_name
                if registration is not None
                else ""
            ),
            transport=transport,
            profiles=profiles,
            implementation_ref=(
                implementation.qualified_name if implementation is not None else ""
            ),
            language_names=language_names,
            drift_witnesses=tuple(drift),
            record={
                "require_interface": claim.require_interface,
                **dict(claim.record),
            },
        )

    def compare_manifests(
        self,
        *,
        tool_name: str = "",
        server_name: str = "",
    ) -> tuple[ManifestDriftWitness, ...]:
        """Compare tools/list, registrations, manifests, and SDKs."""

        tools: set[str] = set()
        if tool_name:
            tools.add(tool_name)
        else:
            for role in (
                ArtifactRole.TOOL_LIST_ENTRY,
                ArtifactRole.REGISTRATION,
                ArtifactRole.MANIFEST,
                ArtifactRole.ALIAS,
            ):
                for item in self._inventory.by_role(role):
                    if server_name and item.server_name and item.server_name != server_name:
                        continue
                    tools.add(item.effective_tool_name)

        witnesses: list[ManifestDriftWitness] = []
        for name in sorted(tools, key=lambda item: (normalize_tool_name(item), item)):
            witnesses.extend(self._manifest_drift_for_tool(name, server_name))
        # Deduplicate.
        unique: dict[str, ManifestDriftWitness] = {}
        for item in witnesses:
            unique[item.witness_id] = item
        return tuple(sorted(unique.values(), key=lambda item: item.witness_id))

    # ------------------------------------------------------------------
    # Stage resolvers
    # ------------------------------------------------------------------

    def _resolve_caller(
        self, claim: CallPathClaim
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        if not claim.caller_name:
            # Caller optional when claim starts at connector; mark candidate.
            return (
                make_hop(
                    stage=PathStage.CALLER,
                    status=ResolverStatus.CANDIDATE,
                    reason_code=ReasonCode.EVIDENCE_REQUIRED,
                    evidence=(
                        make_evidence(
                            rule_id="rule:caller:optional",
                            blob_cid="blob:resolver",
                            forest_id=forest_id,
                            notes={"optional_caller": True},
                        ),
                    ),
                    source_ref="",
                    target_ref=claim.connector_name,
                    notes={"optional_caller": True},
                ),
                None,
            )
        candidates = self._inventory.find(
            role=ArtifactRole.CALLER, name=claim.caller_name
        )
        if not candidates:
            # Allow connector-role artifacts to also act as caller origins when
            # the claim names a service method that is itself the entry.
            candidates = self._inventory.find(name=claim.caller_name)
            candidates = tuple(
                item
                for item in candidates
                if item.role in {ArtifactRole.CALLER, ArtifactRole.CONNECTOR}
            )
        artifact, reject = _select_unique(
            candidates,
            stage=PathStage.CALLER,
            forest_id=forest_id,
            source_ref=claim.caller_name,
            ambiguous_reason=ReasonCode.AMBIGUOUS_ALIAS,
            missing_reason=ReasonCode.MISSING_HOP,
            require_call_edge=False,
        )
        if reject is not None:
            return reject, None
        assert artifact is not None
        # Caller hop is the origin; call edge is required toward connector.
        if claim.connector_name and not artifact.has_call_edge:
            if artifact.has_import_edge:
                return (
                    _reject_hop(
                        PathStage.CALLER,
                        ReasonCode.IMPORT_WITHOUT_CALL,
                        forest_id=forest_id,
                        source_ref=artifact.qualified_name,
                        target_ref=claim.connector_name,
                        artifact_ids=(artifact.artifact_id,),
                    ),
                    artifact,
                )
            return (
                _reject_hop(
                    PathStage.CALLER,
                    ReasonCode.MISSING_CALL_EDGE,
                    forest_id=forest_id,
                    source_ref=artifact.qualified_name,
                    target_ref=claim.connector_name,
                    artifact_ids=(artifact.artifact_id,),
                ),
                artifact,
            )
        return (
            make_hop(
                stage=PathStage.CALLER,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.CALL_EDGE_PRESENT
                if artifact.has_call_edge
                else ReasonCode.CONNECTOR_BINDING,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:caller:bind",
                        forest_id=forest_id,
                    ),
                ),
                source_ref=artifact.qualified_name,
                target_ref=claim.connector_name or artifact.qualified_name,
                artifact_ids=(artifact.artifact_id,),
            ),
            artifact,
        )

    def _resolve_connector(
        self,
        claim: CallPathClaim,
        caller: InventoryArtifact | None,
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        name = claim.connector_name or (
            caller.record.get("connector") if caller is not None else ""
        )
        name = str(name or "").strip()
        if not name:
            return (
                _reject_hop(
                    PathStage.CONNECTOR,
                    ReasonCode.MISSING_HOP,
                    forest_id=forest_id,
                    source_ref=claim.caller_name,
                    notes={"missing": "connector_name"},
                ),
                None,
            )
        candidates = self._inventory.find(role=ArtifactRole.CONNECTOR, name=name)
        artifact, reject = _select_unique(
            candidates,
            stage=PathStage.CONNECTOR,
            forest_id=forest_id,
            source_ref=name,
            ambiguous_reason=ReasonCode.AMBIGUOUS_ALIAS,
            missing_reason=ReasonCode.MISSING_HOP,
            require_call_edge=True,
        )
        if reject is not None:
            # Name-only connector without call edge cannot prove tools/call.
            return reject, None
        assert artifact is not None
        return (
            make_hop(
                stage=PathStage.CONNECTOR,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.CONNECTOR_BINDING,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:connector:bind",
                        forest_id=forest_id,
                    ),
                ),
                source_ref=claim.caller_name or name,
                target_ref=artifact.qualified_name,
                artifact_ids=(artifact.artifact_id,),
                transport=artifact.transport,
                profiles=artifact.profiles,
            ),
            artifact,
        )

    def _resolve_profile_transport(
        self,
        claim: CallPathClaim,
        connector: InventoryArtifact | None,
    ) -> tuple[PathHop, TransportKind, tuple[str, ...]]:
        forest_id = self._inventory.forest_id
        transport = claim.transport
        profiles = claim.profiles
        if connector is not None:
            if transport is TransportKind.UNKNOWN:
                transport = connector.transport
            if not profiles:
                profiles = connector.profiles
        # Inventory transport artifacts may refine.
        transports = self._inventory.by_role(ArtifactRole.TRANSPORT)
        if claim.server_name:
            transports = tuple(
                item
                for item in transports
                if not item.server_name or item.server_name == claim.server_name
            )
        if transport is TransportKind.UNKNOWN and len(transports) == 1:
            transport = transports[0].transport
            if not profiles:
                profiles = transports[0].profiles

        if transport is TransportKind.UNKNOWN:
            hop = _reject_hop(
                PathStage.PROFILE_TRANSPORT,
                ReasonCode.EXTERNAL_TRANSPORT,
                forest_id=forest_id,
                source_ref=claim.connector_name,
                notes={"transport": "unknown"},
            )
            return hop, transport, profiles

        if transport not in self._inventory.admitted_transports:
            hop = _reject_hop(
                PathStage.PROFILE_TRANSPORT,
                ReasonCode.TRANSPORT_MISMATCH,
                forest_id=forest_id,
                source_ref=claim.connector_name,
                notes={
                    "transport": transport.value,
                    "admitted": [item.value for item in self._inventory.admitted_transports],
                },
            )
            hop = make_hop(
                stage=PathStage.PROFILE_TRANSPORT,
                status=ResolverStatus.UNKNOWN,
                reason_code=ReasonCode.TRANSPORT_MISMATCH,
                evidence=hop.evidence,
                source_ref=claim.connector_name,
                target_ref=transport.value,
                transport=transport,
                profiles=profiles,
                notes=dict(hop.notes),
            )
            return hop, transport, profiles

        # Claim profiles and inventory-required profiles must be covered by the
        # negotiated set when negotiation evidence exists. Known-profile names
        # alone never waive a missing negotiated capability.
        required = set(self._inventory.required_profiles) | set(claim.profiles)
        negotiated = set(profiles)
        if required and negotiated and not required.issubset(negotiated):
            missing = sorted(required - negotiated)
            hop = make_hop(
                stage=PathStage.PROFILE_TRANSPORT,
                status=ResolverStatus.UNKNOWN,
                reason_code=ReasonCode.PROFILE_MISMATCH,
                evidence=(
                    make_evidence(
                        rule_id="rule:profile:mismatch",
                        blob_cid="blob:resolver",
                        forest_id=forest_id,
                        notes={"missing_profiles": missing},
                    ),
                ),
                source_ref=claim.connector_name,
                target_ref=transport.value,
                transport=transport,
                profiles=profiles,
                notes={"missing_profiles": missing},
            )
            return hop, transport, profiles

        reason = _transport_reason(transport)
        hop = make_hop(
            stage=PathStage.PROFILE_TRANSPORT,
            status=ResolverStatus.RESOLVED_STATIC,
            reason_code=reason if not profiles else ReasonCode.PROFILE_NEGOTIATED,
            evidence=(
                make_evidence(
                    rule_id="rule:profile_transport:bind",
                    blob_cid=(
                        connector.blob_cid
                        if connector is not None
                        else "blob:resolver"
                    ),
                    forest_id=forest_id,
                    source_record_key=(
                        connector.artifact_id if connector is not None else ""
                    ),
                    notes={
                        "transport": transport.value,
                        "profiles": list(profiles),
                    },
                ),
            ),
            source_ref=claim.connector_name,
            target_ref=transport.value,
            transport=transport,
            profiles=profiles,
            artifact_ids=(
                (connector.artifact_id,) if connector is not None else ()
            ),
        )
        return hop, transport, profiles

    def _resolve_tools_list(
        self,
        claim: CallPathClaim,
        *,
        require_interface: bool,
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        listed = self._inventory.find(
            role=ArtifactRole.TOOL_LIST_ENTRY,
            tool_name=claim.tool_name,
            server_name=claim.server_name,
        )
        interfaces = self._inventory.find(
            role=ArtifactRole.INTERFACE_DESCRIPTOR,
            tool_name=claim.tool_name,
            server_name=claim.server_name,
        )
        if require_interface:
            artifact, reject = _select_unique(
                interfaces,
                stage=PathStage.TOOLS_LIST,
                forest_id=forest_id,
                source_ref=claim.tool_name,
                ambiguous_reason=ReasonCode.AMBIGUOUS_ALIAS,
                missing_reason=ReasonCode.MISSING_HOP,
            )
            if reject is not None:
                return reject, None
            assert artifact is not None
            return (
                make_hop(
                    stage=PathStage.TOOLS_LIST,
                    status=ResolverStatus.RESOLVED_STATIC,
                    reason_code=ReasonCode.INTERFACE_DESCRIPTOR_BINDING,
                    evidence=(
                        _evidence_for_artifact(
                            artifact,
                            rule_id="rule:interface:bind",
                            forest_id=forest_id,
                        ),
                    ),
                    source_ref=claim.tool_name,
                    target_ref=artifact.qualified_name,
                    artifact_ids=(artifact.artifact_id,),
                ),
                artifact,
            )

        # Prefer tools/list; fall back to interface descriptor.
        pool = listed or interfaces
        artifact, reject = _select_unique(
            pool,
            stage=PathStage.TOOLS_LIST,
            forest_id=forest_id,
            source_ref=claim.tool_name,
            ambiguous_reason=ReasonCode.AMBIGUOUS_ALIAS,
            missing_reason=ReasonCode.MISSING_HOP,
        )
        if reject is not None:
            return reject, None
        assert artifact is not None
        reason = (
            ReasonCode.TOOLS_LIST_BINDING
            if artifact.role is ArtifactRole.TOOL_LIST_ENTRY
            else ReasonCode.INTERFACE_DESCRIPTOR_BINDING
        )
        return (
            make_hop(
                stage=PathStage.TOOLS_LIST,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=reason,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:tools_list:bind",
                        forest_id=forest_id,
                    ),
                ),
                source_ref=claim.tool_name,
                target_ref=artifact.qualified_name,
                artifact_ids=(artifact.artifact_id,),
            ),
            artifact,
        )

    def _resolve_tools_call(
        self,
        claim: CallPathClaim,
        connector: InventoryArtifact | None,
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        candidates = self._inventory.find(
            role=ArtifactRole.TOOL_CALL_SITE,
            tool_name=claim.tool_name,
        )
        if not candidates and connector is not None:
            # Connector that itself performs tools/call may stand in when it
            # declares the tool name and a call edge.
            if (
                connector.has_call_edge
                and (
                    not connector.tool_name
                    or normalize_tool_name(connector.tool_name)
                    == normalize_tool_name(claim.tool_name)
                    or claim.tool_name in tool_name_aliases(connector.tool_name)
                )
            ):
                candidates = (connector,)
        artifact, reject = _select_unique(
            candidates,
            stage=PathStage.TOOLS_CALL,
            forest_id=forest_id,
            source_ref=claim.tool_name,
            ambiguous_reason=ReasonCode.DYNAMIC_DISPATCH,
            missing_reason=ReasonCode.MISSING_CALL_EDGE,
            require_call_edge=True,
        )
        if reject is not None:
            return reject, None
        assert artifact is not None
        return (
            make_hop(
                stage=PathStage.TOOLS_CALL,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.TOOLS_CALL_BINDING,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:tools_call:bind",
                        forest_id=forest_id,
                        notes={"rpc": "tools/call"},
                    ),
                ),
                source_ref=artifact.qualified_name,
                target_ref=claim.tool_name,
                artifact_ids=(artifact.artifact_id,),
                transport=artifact.transport,
            ),
            artifact,
        )

    def _resolve_registration(
        self, claim: CallPathClaim
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        candidates = self._inventory.find(
            role=ArtifactRole.REGISTRATION,
            tool_name=claim.tool_name,
            server_name=claim.server_name,
        )
        # Alias expansion: hierarchical alias may register under category.tool.
        if not candidates:
            for alias in tool_name_aliases(claim.tool_name):
                candidates = self._inventory.find(
                    role=ArtifactRole.REGISTRATION,
                    tool_name=alias,
                    server_name=claim.server_name,
                )
                if candidates:
                    break
        artifact, reject = _select_unique(
            candidates,
            stage=PathStage.SERVER_REGISTRY,
            forest_id=forest_id,
            source_ref=claim.tool_name,
            ambiguous_reason=ReasonCode.AMBIGUOUS_REGISTRATION,
            missing_reason=ReasonCode.MISSING_REGISTRATION,
        )
        if reject is not None:
            return reject, None
        assert artifact is not None
        return (
            make_hop(
                stage=PathStage.SERVER_REGISTRY,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.REGISTRATION_MATCH,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:registration:bind",
                        forest_id=forest_id,
                    ),
                ),
                source_ref=claim.tool_name,
                target_ref=artifact.qualified_name,
                artifact_ids=(artifact.artifact_id,),
            ),
            artifact,
        )

    def _resolve_adapter(
        self,
        claim: CallPathClaim,
        registration: InventoryArtifact | None,
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        candidates = self._inventory.find(
            role=ArtifactRole.ADAPTER,
            tool_name=claim.tool_name,
            server_name=claim.server_name,
        )
        if registration is not None:
            adapter_name = str(
                registration.record.get("adapter")
                or registration.record.get("adapter_name")
                or ""
            ).strip()
            if adapter_name:
                named = self._inventory.find(role=ArtifactRole.ADAPTER, name=adapter_name)
                if named:
                    candidates = named
            # Registration may directly point at adapter qualified name.
            if not candidates and registration.qualified_name:
                named = tuple(
                    item
                    for item in self._inventory.by_role(ArtifactRole.ADAPTER)
                    if item.qualified_name == registration.qualified_name
                    or item.artifact_id
                    == str(registration.record.get("adapter_id") or "")
                )
                if named:
                    candidates = named
        artifact, reject = _select_unique(
            candidates,
            stage=PathStage.ADAPTER,
            forest_id=forest_id,
            source_ref=claim.tool_name,
            ambiguous_reason=ReasonCode.AMBIGUOUS_ADAPTER,
            missing_reason=ReasonCode.MISSING_HOP,
            require_call_edge=True,
        )
        if reject is not None:
            return reject, None
        assert artifact is not None
        return (
            make_hop(
                stage=PathStage.ADAPTER,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.ADAPTER_BINDING,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:adapter:bind",
                        forest_id=forest_id,
                    ),
                ),
                source_ref=(
                    registration.qualified_name
                    if registration is not None
                    else claim.tool_name
                ),
                target_ref=artifact.qualified_name,
                artifact_ids=(artifact.artifact_id,),
            ),
            artifact,
        )

    def _implementation_pool(
        self,
        claim: CallPathClaim,
        *,
        preferred_name: str = "",
    ) -> tuple[InventoryArtifact, ...]:
        """Collect implementation-like artifacts, including non-invocation roles.

        Non-invocation roles (mock, helper, fallback, test server) are retained
        so ``_select_unique`` can reject them with an explicit reason instead of
        reporting a bare missing hop.
        """

        roles = (
            ArtifactRole.IMPLEMENTATION,
            ArtifactRole.MOCK,
            ArtifactRole.LOCAL_HELPER,
            ArtifactRole.LEGACY_FALLBACK,
            ArtifactRole.TEST_SERVER,
        )
        pool: list[InventoryArtifact] = []
        for role in roles:
            pool.extend(
                self._inventory.find(
                    role=role,
                    tool_name=claim.tool_name,
                    server_name=claim.server_name,
                )
            )
        if preferred_name:
            named: list[InventoryArtifact] = []
            aliases = set(tool_name_aliases(preferred_name))
            aliases.add(preferred_name)
            for role in roles:
                for item in self._inventory.by_role(role):
                    if (
                        item.name == preferred_name
                        or item.qualified_name == preferred_name
                        or preferred_name in tool_name_aliases(item.name)
                        or preferred_name in tool_name_aliases(item.qualified_name)
                        or bool(aliases & set(tool_name_aliases(item.effective_tool_name)))
                    ):
                        named.append(item)
            if named:
                return tuple(named)
        # Stable unique by artifact_id.
        unique: dict[str, InventoryArtifact] = {
            item.artifact_id: item for item in pool
        }
        return tuple(unique.values())

    def _resolve_implementation(
        self,
        claim: CallPathClaim,
        adapter: InventoryArtifact | None,
        registration: InventoryArtifact | None,
    ) -> tuple[PathHop, InventoryArtifact | None]:
        forest_id = self._inventory.forest_id
        preferred = ""
        if adapter is not None:
            preferred = str(
                adapter.record.get("implementation")
                or adapter.record.get("implementation_name")
                or adapter.record.get("target")
                or ""
            ).strip()
        if not preferred and registration is not None:
            preferred = str(
                registration.record.get("implementation")
                or registration.record.get("handler")
                or ""
            ).strip()
        candidates = self._implementation_pool(claim, preferred_name=preferred)

        artifact, reject = _select_unique(
            candidates,
            stage=PathStage.PACKAGE_IMPLEMENTATION,
            forest_id=forest_id,
            source_ref=claim.tool_name,
            ambiguous_reason=ReasonCode.AMBIGUOUS_IMPLEMENTATION,
            missing_reason=ReasonCode.MISSING_HOP,
            require_call_edge=True,
        )
        if reject is not None:
            return reject, None
        assert artifact is not None
        return (
            make_hop(
                stage=PathStage.PACKAGE_IMPLEMENTATION,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.IMPLEMENTATION_MATCH,
                evidence=(
                    _evidence_for_artifact(
                        artifact,
                        rule_id="rule:implementation:bind",
                        forest_id=forest_id,
                    ),
                ),
                source_ref=(
                    adapter.qualified_name if adapter is not None else claim.tool_name
                ),
                target_ref=artifact.qualified_name,
                artifact_ids=(artifact.artifact_id,),
            ),
            artifact,
        )

    def _resolve_result_error_mapping(
        self,
        claim: CallPathClaim,
        *,
        listed: InventoryArtifact | None,
        registration: InventoryArtifact | None,
        implementation: InventoryArtifact | None,
        call_site: InventoryArtifact | None,
    ) -> tuple[PathHop, tuple[ManifestDriftWitness, ...]]:
        forest_id = self._inventory.forest_id
        result_maps = self._inventory.find(
            role=ArtifactRole.RESULT_MAP,
            tool_name=claim.tool_name,
            server_name=claim.server_name,
        )
        error_maps = self._inventory.find(
            role=ArtifactRole.ERROR_MAP,
            tool_name=claim.tool_name,
            server_name=claim.server_name,
        )
        drift: list[ManifestDriftWitness] = []

        # If explicit maps exist, require unique non-invocation bind.
        result_art: InventoryArtifact | None = None
        error_art: InventoryArtifact | None = None
        if result_maps:
            result_art, reject = _select_unique(
                result_maps,
                stage=PathStage.RESULT_ERROR_MAPPING,
                forest_id=forest_id,
                source_ref=claim.tool_name,
                ambiguous_reason=ReasonCode.AMBIGUOUS_ALIAS,
                missing_reason=ReasonCode.MISSING_HOP,
            )
            if reject is not None and reject.reason_code in {
                ReasonCode.MOCK_IMPLEMENTATION,
                ReasonCode.STATIC_DASHBOARD,
                ReasonCode.COPIED_MANIFEST,
                ReasonCode.LEGACY_FALLBACK,
                ReasonCode.SAME_NAME_HELPER,
                ReasonCode.IMPORT_WITHOUT_CALL,
            }:
                return reject, ()
        if error_maps:
            error_art, reject = _select_unique(
                error_maps,
                stage=PathStage.RESULT_ERROR_MAPPING,
                forest_id=forest_id,
                source_ref=claim.tool_name,
                ambiguous_reason=ReasonCode.AMBIGUOUS_ALIAS,
                missing_reason=ReasonCode.MISSING_HOP,
            )
            if reject is not None and reject.reason_code in {
                ReasonCode.MOCK_IMPLEMENTATION,
                ReasonCode.STATIC_DASHBOARD,
                ReasonCode.COPIED_MANIFEST,
                ReasonCode.LEGACY_FALLBACK,
                ReasonCode.SAME_NAME_HELPER,
                ReasonCode.IMPORT_WITHOUT_CALL,
            }:
                return reject, ()

        # Compare every contract-bearing hop, not merely the two endpoints.
        # A list and implementation that happen to agree must not conceal a
        # drifting Python registration between them. Missing values stay
        # unverified in hop notes and therefore cannot satisfy VFS-G153.
        contract_artifacts = tuple(
            item
            for item in (listed, registration, implementation)
            if item is not None
        )
        checked_aspects: list[str] = []
        matched_aspects: list[str] = []

        def _append_mismatch(
            *,
            aspect: str,
            drift_kind: DriftKind,
            values: Sequence[tuple[InventoryArtifact, str]],
            rule_id: str,
        ) -> None:
            baseline_artifact, baseline_value = values[0]
            divergent = next(
                (
                    (artifact, value)
                    for artifact, value in values[1:]
                    if value != baseline_value
                ),
                None,
            )
            if divergent is None:
                matched_aspects.append(aspect)
                return
            divergent_artifact, divergent_value = divergent
            drift.append(
                ManifestDriftWitness(
                    drift_kind=drift_kind,
                    tool_name=claim.tool_name,
                    left_ref=baseline_artifact.artifact_id,
                    right_ref=divergent_artifact.artifact_id,
                    left_value=baseline_value,
                    right_value=divergent_value,
                    evidence=(
                        _evidence_for_artifact(
                            baseline_artifact,
                            rule_id=rule_id,
                            forest_id=forest_id,
                        ),
                        _evidence_for_artifact(
                            divergent_artifact,
                            rule_id=rule_id,
                            forest_id=forest_id,
                        ),
                    ),
                    notes={"aspect": aspect},
                )
            )

        aspect_values: tuple[
            tuple[str, DriftKind, str, tuple[tuple[InventoryArtifact, str], ...]],
            ...,
        ] = (
            (
                "input_schema",
                DriftKind.SCHEMA_MISMATCH,
                "rule:schema:input_mismatch",
                tuple(
                    (artifact, schema_fingerprint(artifact.input_schema))
                    for artifact in contract_artifacts
                ),
            ),
            (
                "output_schema",
                DriftKind.SCHEMA_MISMATCH,
                "rule:schema:output_mismatch",
                tuple(
                    (artifact, schema_fingerprint(artifact.output_schema))
                    for artifact in contract_artifacts
                ),
            ),
            (
                "version",
                DriftKind.VERSION_MISMATCH,
                "rule:version:mismatch",
                tuple(
                    (artifact, artifact.version)
                    for artifact in contract_artifacts
                ),
            ),
            (
                "error_map",
                DriftKind.ERROR_MAP_MISMATCH,
                "rule:error_map:mismatch",
                tuple(
                    (artifact, ",".join(sorted(set(artifact.error_codes))))
                    for artifact in contract_artifacts
                ),
            ),
        )
        for aspect, drift_kind, rule_id, values in aspect_values:
            # VFS-G153 requires the tools/list, Python registration, and
            # package implementation contract surfaces to all participate.
            if len(values) != 3 or any(not value for _, value in values):
                continue
            checked_aspects.append(aspect)
            _append_mismatch(
                aspect=aspect,
                drift_kind=drift_kind,
                values=values,
                rule_id=rule_id,
            )

        artifact_ids = tuple(
            dict.fromkeys(
                item.artifact_id
                for item in (
                    result_art,
                    error_art,
                    *contract_artifacts,
                    call_site,
                )
                if item is not None
            )
        )
        parity_notes = {
            "manifest_parity_checked_aspects": sorted(checked_aspects),
            "manifest_parity_matched_aspects": sorted(matched_aspects),
            "manifest_parity_contract_artifacts": [
                item.artifact_id for item in contract_artifacts
            ],
        }
        if drift:
            hop = make_hop(
                stage=PathStage.RESULT_ERROR_MAPPING,
                status=ResolverStatus.CANDIDATE,
                reason_code=ReasonCode.MANIFEST_DRIFT,
                evidence=(
                    make_evidence(
                        rule_id="rule:result_error:drift",
                        blob_cid="blob:resolver",
                        forest_id=forest_id,
                        notes={"drift_count": len(drift)},
                    ),
                ),
                source_ref=claim.tool_name,
                target_ref=(
                    implementation.qualified_name
                    if implementation is not None
                    else claim.tool_name
                ),
                artifact_ids=artifact_ids,
                notes={"drift_count": len(drift), **parity_notes},
            )
            return hop, tuple(drift)

        # This hop may still resolve for VFS-G152 when an explicit result/error
        # mapping exists, but absent parity aspects remain unverified and keep
        # the separate VFS-G153 claim fail-closed.
        if result_art is not None or error_art is not None or (
            len(contract_artifacts) >= 2
        ):
            reason = ReasonCode.RESULT_MAP_MATCH
            if error_art is not None and result_art is None:
                reason = ReasonCode.ERROR_MAP_MATCH
            elif {"input_schema", "output_schema"}.issubset(matched_aspects):
                reason = ReasonCode.SCHEMA_PARITY
            hop = make_hop(
                stage=PathStage.RESULT_ERROR_MAPPING,
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=reason,
                evidence=(
                    make_evidence(
                        rule_id="rule:result_error:bind",
                        blob_cid="blob:resolver",
                        forest_id=forest_id,
                        source_record_key=claim.tool_name,
                        target_record_key=(
                            implementation.qualified_name
                            if implementation is not None
                            else ""
                        ),
                        notes={
                            "result_map": result_art.artifact_id if result_art else "",
                            "error_map": error_art.artifact_id if error_art else "",
                        },
                    ),
                ),
                source_ref=claim.tool_name,
                target_ref=(
                    implementation.qualified_name
                    if implementation is not None
                    else claim.tool_name
                ),
                artifact_ids=artifact_ids,
                notes=parity_notes,
            )
            return hop, ()

        # No mapping evidence at all: candidate, not proved.
        hop = make_hop(
            stage=PathStage.RESULT_ERROR_MAPPING,
            status=ResolverStatus.CANDIDATE,
            reason_code=ReasonCode.EVIDENCE_REQUIRED,
            evidence=(
                make_evidence(
                    rule_id="rule:result_error:missing",
                    blob_cid="blob:resolver",
                    forest_id=forest_id,
                    notes={"missing": "result_error_mapping"},
                ),
            ),
            source_ref=claim.tool_name,
            notes={"missing": "result_error_mapping"},
        )
        return hop, ()

    def _language_name_drift(
        self,
        claim: CallPathClaim,
        registration: InventoryArtifact | None,
    ) -> tuple[ManifestDriftWitness, ...]:
        forest_id = self._inventory.forest_id
        witnesses: list[ManifestDriftWitness] = []
        names = dict(claim.language_names)
        if not names:
            return ()
        # Compare each declared language name against registration / list.
        canonical = normalize_tool_name(claim.tool_name)
        for language, lang_name in sorted(names.items()):
            if normalize_tool_name(str(lang_name)) != canonical:
                # Alias-equivalent hierarchical forms are OK.
                if not (
                    set(tool_name_aliases(str(lang_name)))
                    & set(tool_name_aliases(claim.tool_name))
                ):
                    ref = (
                        registration.artifact_id
                        if registration is not None
                        else claim.tool_name
                    )
                    witnesses.append(
                        ManifestDriftWitness(
                            drift_kind=DriftKind.LANGUAGE_NAME_MISMATCH,
                            tool_name=claim.tool_name,
                            left_ref=f"language:{language}",
                            right_ref=ref,
                            left_value=str(lang_name),
                            right_value=claim.tool_name,
                            evidence=(
                                make_evidence(
                                    rule_id="rule:language:name_mismatch",
                                    blob_cid="blob:resolver",
                                    forest_id=forest_id,
                                    notes={
                                        "language": language,
                                        "declared": str(lang_name),
                                    },
                                ),
                            ),
                        )
                    )
        return tuple(witnesses)

    def _manifest_drift_for_tool(
        self,
        tool_name: str,
        server_name: str = "",
    ) -> tuple[ManifestDriftWitness, ...]:
        forest_id = self._inventory.forest_id
        witnesses: list[ManifestDriftWitness] = []
        listed = self._inventory.find(
            role=ArtifactRole.TOOL_LIST_ENTRY,
            tool_name=tool_name,
            server_name=server_name,
        )
        registrations = self._inventory.find(
            role=ArtifactRole.REGISTRATION,
            tool_name=tool_name,
            server_name=server_name,
        )
        manifests = self._inventory.find(
            role=ArtifactRole.MANIFEST,
            tool_name=tool_name,
            server_name=server_name,
        )
        copied = self._inventory.find(
            role=ArtifactRole.COPIED_MANIFEST,
            tool_name=tool_name,
            server_name=server_name,
        )
        dashboards = self._inventory.find(
            role=ArtifactRole.STATIC_DASHBOARD,
            tool_name=tool_name,
            server_name=server_name,
        )

        if copied:
            for item in copied:
                witnesses.append(
                    ManifestDriftWitness(
                        drift_kind=DriftKind.COPIED_WITHOUT_BINDING,
                        tool_name=tool_name,
                        left_ref=item.artifact_id,
                        right_ref="registration",
                        left_value=item.qualified_name,
                        right_value="",
                        evidence=(
                            _evidence_for_artifact(
                                item,
                                rule_id="rule:manifest:copied",
                                forest_id=forest_id,
                            ),
                        ),
                        notes={"cannot_prove_invocation": True},
                    )
                )

        if dashboards and not registrations:
            for item in dashboards:
                witnesses.append(
                    ManifestDriftWitness(
                        drift_kind=DriftKind.COPIED_WITHOUT_BINDING,
                        tool_name=tool_name,
                        left_ref=item.artifact_id,
                        right_ref="registration",
                        left_value="static_dashboard",
                        right_value="missing",
                        evidence=(
                            _evidence_for_artifact(
                                item,
                                rule_id="rule:manifest:dashboard",
                                forest_id=forest_id,
                            ),
                        ),
                    )
                )

        interfaces = self._inventory.find(
            role=ArtifactRole.INTERFACE_DESCRIPTOR,
            tool_name=tool_name,
            server_name=server_name,
        )

        if listed and not registrations:
            for item in listed:
                witnesses.append(
                    ManifestDriftWitness(
                        drift_kind=DriftKind.MISSING_REGISTRATION,
                        tool_name=tool_name,
                        left_ref=item.artifact_id,
                        right_ref="server_registry",
                        left_value=item.effective_tool_name,
                        right_value="",
                        evidence=(
                            _evidence_for_artifact(
                                item,
                                rule_id="rule:manifest:missing_registration",
                                forest_id=forest_id,
                            ),
                        ),
                    )
                )

        # Interface descriptors also satisfy tools/list discovery for parity.
        discovery = listed or interfaces
        if registrations and not discovery and not manifests:
            for item in registrations:
                # Registered but never listed: extra/unreachable from connector.
                witnesses.append(
                    ManifestDriftWitness(
                        drift_kind=DriftKind.EXTRA_UNREACHABLE,
                        tool_name=tool_name,
                        left_ref=item.artifact_id,
                        right_ref="tools_list",
                        left_value=item.effective_tool_name,
                        right_value="missing",
                        evidence=(
                            _evidence_for_artifact(
                                item,
                                rule_id="rule:manifest:extra_unreachable",
                                forest_id=forest_id,
                            ),
                        ),
                    )
                )

        # Manifest vs registration schema/name drift.
        for manifest in manifests:
            if manifest.non_invocation_reason is not None:
                witnesses.append(
                    ManifestDriftWitness(
                        drift_kind=DriftKind.STALE_MANIFEST
                        if manifest.non_invocation_reason
                        is ReasonCode.COPIED_MANIFEST
                        else DriftKind.COPIED_WITHOUT_BINDING,
                        tool_name=tool_name,
                        left_ref=manifest.artifact_id,
                        right_ref="authoritative_inventory",
                        left_value=manifest.non_invocation_reason.value,
                        right_value="",
                        evidence=(
                            _evidence_for_artifact(
                                manifest,
                                rule_id="rule:manifest:non_invocation",
                                forest_id=forest_id,
                            ),
                        ),
                    )
                )
                continue
            if not registrations:
                continue
            # Compare against each registration (usually one).
            for reg in registrations:
                if (
                    normalize_tool_name(manifest.effective_tool_name)
                    != normalize_tool_name(reg.effective_tool_name)
                    and not (
                        set(tool_name_aliases(manifest.effective_tool_name))
                        & set(tool_name_aliases(reg.effective_tool_name))
                    )
                ):
                    witnesses.append(
                        ManifestDriftWitness(
                            drift_kind=DriftKind.NAME_MISMATCH,
                            tool_name=tool_name,
                            left_ref=manifest.artifact_id,
                            right_ref=reg.artifact_id,
                            left_value=manifest.effective_tool_name,
                            right_value=reg.effective_tool_name,
                            evidence=(
                                _evidence_for_artifact(
                                    manifest,
                                    rule_id="rule:manifest:name_mismatch",
                                    forest_id=forest_id,
                                ),
                                _evidence_for_artifact(
                                    reg,
                                    rule_id="rule:manifest:name_mismatch",
                                    forest_id=forest_id,
                                ),
                            ),
                        )
                    )
                m_in = schema_fingerprint(manifest.input_schema)
                r_in = schema_fingerprint(reg.input_schema)
                if m_in and r_in and m_in != r_in:
                    witnesses.append(
                        ManifestDriftWitness(
                            drift_kind=DriftKind.SCHEMA_MISMATCH,
                            tool_name=tool_name,
                            left_ref=manifest.artifact_id,
                            right_ref=reg.artifact_id,
                            left_value=m_in,
                            right_value=r_in,
                            evidence=(
                                _evidence_for_artifact(
                                    manifest,
                                    rule_id="rule:manifest:schema_mismatch",
                                    forest_id=forest_id,
                                ),
                                _evidence_for_artifact(
                                    reg,
                                    rule_id="rule:manifest:schema_mismatch",
                                    forest_id=forest_id,
                                ),
                            ),
                            notes={"aspect": "input_schema"},
                        )
                    )
                if (
                    manifest.version
                    and reg.version
                    and manifest.version != reg.version
                ):
                    witnesses.append(
                        ManifestDriftWitness(
                            drift_kind=DriftKind.STALE_MANIFEST,
                            tool_name=tool_name,
                            left_ref=manifest.artifact_id,
                            right_ref=reg.artifact_id,
                            left_value=manifest.version,
                            right_value=reg.version,
                            evidence=(
                                _evidence_for_artifact(
                                    manifest,
                                    rule_id="rule:manifest:stale",
                                    forest_id=forest_id,
                                ),
                                _evidence_for_artifact(
                                    reg,
                                    rule_id="rule:manifest:stale",
                                    forest_id=forest_id,
                                ),
                            ),
                        )
                    )
        return tuple(witnesses)


def inventory_from_program_graph(
    graph: ProgramGraph,
    *,
    forest_id: str = "",
    default_server: str = "",
    default_transport: TransportKind | str = TransportKind.UNKNOWN,
) -> MCPlusPlusInventory:
    """Project MCP-related program-graph nodes into a closed inventory.

    Only nodes already present in the graph are admitted. This helper never
    invents registrations, adapters, or call edges.
    """

    if not isinstance(graph, ProgramGraph):
        raise MCPlusPlusResolverError("graph must be a ProgramGraph")
    forest = forest_id or graph.forest_id
    transport = _enum(default_transport, TransportKind, "default_transport")
    artifacts: list[InventoryArtifact] = []

    # Call edges: source has call edge to target.
    call_sources: set[str] = set()
    call_targets: set[str] = set()
    import_sources: set[str] = set()
    for edge in graph.edges:
        if edge.kind is ProgramEdgeKind.CALLS:
            call_sources.add(edge.source)
            call_targets.add(edge.target)
        elif edge.kind is ProgramEdgeKind.IMPORTS:
            import_sources.add(edge.source)

    nodes_by_id = {node.node_id: node for node in graph.nodes}

    for node in graph.nodes:
        record = dict(node.record or {})
        qname = node.qualified_name or node.record_key
        path = node.path or str(record.get("path") or "")
        tool_name = str(
            record.get("tool_name")
            or record.get("name")
            or (qname.rsplit(".", 1)[-1] if qname else "")
        )
        role: ArtifactRole | None = None
        if node.kind is ProgramNodeKind.MCP_TOOL:
            role = ArtifactRole.TOOL_LIST_ENTRY
        elif node.kind is ProgramNodeKind.MCP_REGISTRATION:
            role = ArtifactRole.REGISTRATION
        elif node.kind is ProgramNodeKind.TRANSPORT:
            role = ArtifactRole.TRANSPORT
        elif node.kind is ProgramNodeKind.CALL:
            callee = str(record.get("callee") or qname)
            if any(
                marker in callee
                for marker in (
                    "tools/call",
                    "tools.call",
                    "callTool",
                    "call_tool",
                    "tools_call",
                )
            ):
                role = ArtifactRole.TOOL_CALL_SITE
            elif "MCPPPServerConnector" in callee or "connector" in callee.lower():
                role = ArtifactRole.CONNECTOR
            else:
                role = ArtifactRole.CALLER
        elif node.kind is ProgramNodeKind.SCHEMA:
            role = ArtifactRole.JSON_SCHEMA
        elif node.kind in {ProgramNodeKind.DEFINITION, ProgramNodeKind.SYMBOL}:
            kind_hint = str(record.get("mcp_role") or record.get("role") or "").lower()
            if kind_hint in {item.value for item in ArtifactRole}:
                role = ArtifactRole(kind_hint)
            elif "adapter" in kind_hint or "adapter" in path.lower():
                role = ArtifactRole.ADAPTER
            elif "mock" in kind_hint or "mock" in path.lower():
                role = ArtifactRole.MOCK
            else:
                # Generic definitions are only admitted when explicitly tagged.
                continue
        else:
            continue

        has_call = node.node_id in call_sources or bool(record.get("has_call_edge"))
        has_import = node.node_id in import_sources or bool(
            record.get("has_import_edge")
        )
        # If this node is a call site, it inherently participates in a call edge.
        if node.kind is ProgramNodeKind.CALL:
            has_call = True

        profiles = tuple(record.get("profiles") or ())
        transport_value = str(record.get("transport") or transport.value)
        try:
            node_transport = TransportKind(transport_value)
        except ValueError:
            node_transport = transport

        artifacts.append(
            InventoryArtifact(
                artifact_id=f"graph:{node.node_id}",
                role=role,
                name=tool_name or qname,
                language=node.language or str(record.get("language") or ""),
                package=str(record.get("package") or ""),
                module_path=str(record.get("module_path") or path),
                qualified_name=qname,
                server_name=str(record.get("server_name") or default_server),
                transport=node_transport,
                profiles=profiles,
                tool_name=tool_name,
                alias_of=str(record.get("alias_of") or ""),
                input_schema=record.get("input_schema")
                or record.get("inputSchema")
                or {},
                output_schema=record.get("output_schema")
                or record.get("outputSchema")
                or {},
                error_codes=tuple(record.get("error_codes") or record.get("errors") or ()),
                version=str(record.get("version") or ""),
                path=path,
                blob_cid=node.binding.blob_cid,
                forest_id=node.binding.forest_id or forest,
                has_call_edge=has_call,
                has_import_edge=has_import,
                is_external=bool(record.get("is_external", False)),
                markers=tuple(record.get("markers") or ()),
                record={
                    **record,
                    "node_kind": node.kind.value,
                    "node_id": node.node_id,
                },
            )
        )

        # Materialize adapter/implementation targets declared on registration
        # records only when those targets already exist as graph nodes.
        if role is ArtifactRole.REGISTRATION:
            for key, target_role in (
                ("adapter", ArtifactRole.ADAPTER),
                ("implementation", ArtifactRole.IMPLEMENTATION),
                ("handler", ArtifactRole.IMPLEMENTATION),
            ):
                target_name = str(record.get(key) or "").strip()
                if not target_name:
                    continue
                # Prefer matching existing graph nodes by qualified name.
                matched = [
                    other
                    for other in nodes_by_id.values()
                    if other.qualified_name == target_name
                ]
                if not matched:
                    continue
                for other in matched:
                    other_has_call = other.node_id in call_sources or other.node_id in call_targets
                    artifacts.append(
                        InventoryArtifact(
                            artifact_id=f"graph:{other.node_id}:{target_role.value}",
                            role=target_role,
                            name=other.qualified_name,
                            language=other.language,
                            package=str(other.record.get("package") or ""),
                            module_path=other.path,
                            qualified_name=other.qualified_name,
                            server_name=str(
                                other.record.get("server_name") or default_server
                            ),
                            transport=node_transport,
                            tool_name=tool_name,
                            path=other.path,
                            blob_cid=other.binding.blob_cid,
                            forest_id=other.binding.forest_id or forest,
                            has_call_edge=other_has_call
                            or bool(other.record.get("has_call_edge")),
                            has_import_edge=other.node_id in import_sources,
                            is_external=bool(other.record.get("is_external", False)),
                            record=dict(other.record or {}),
                        )
                    )

    # Deduplicate by artifact_id (last write wins deterministically via sort later).
    unique: dict[str, InventoryArtifact] = {}
    for item in artifacts:
        unique[item.artifact_id] = item

    return MCPlusPlusInventory(
        forest_id=forest,
        artifacts=tuple(unique.values()),
    )


def resolve_mcplusplus_paths(
    inventory: MCPlusPlusInventory,
    claims: Sequence[CallPathClaim | Mapping[str, Any]],
    *,
    max_paths: int = DEFAULT_MAX_PATHS,
) -> MCPlusPlusResolutionResult:
    """Resolve claims against inventory (module-level entry point)."""

    return MCPlusPlusContractResolver(inventory, max_paths=max_paths).resolve(claims)


def resolve_mcplusplus_from_graph(
    graph: ProgramGraph,
    claims: Sequence[CallPathClaim | Mapping[str, Any]],
    *,
    default_server: str = "",
    default_transport: TransportKind | str = TransportKind.UNKNOWN,
    max_paths: int = DEFAULT_MAX_PATHS,
) -> MCPlusPlusResolutionResult:
    """Project a program graph into inventory and resolve claims."""

    inventory = inventory_from_program_graph(
        graph,
        default_server=default_server,
        default_transport=default_transport,
    )
    return resolve_mcplusplus_paths(inventory, claims, max_paths=max_paths)


# ---------------------------------------------------------------------------
# Objective evidence discovery + prove claims (VFS-G152 / VFS-G153)
# ---------------------------------------------------------------------------

_NON_INVOCATION_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.SAME_NAME_HELPER,
        ReasonCode.MOCK_IMPLEMENTATION,
        ReasonCode.TEST_SERVER,
        ReasonCode.COPIED_MANIFEST,
        ReasonCode.STATIC_DASHBOARD,
        ReasonCode.LEGACY_FALLBACK,
        ReasonCode.IMPORT_WITHOUT_CALL,
    }
)

_FRONTIER_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.AMBIGUOUS_REGISTRATION,
        ReasonCode.AMBIGUOUS_ADAPTER,
        ReasonCode.AMBIGUOUS_IMPLEMENTATION,
        ReasonCode.AMBIGUOUS_ALIAS,
        ReasonCode.DYNAMIC_DISPATCH,
        ReasonCode.EXTERNAL_PACKAGE,
        ReasonCode.EXTERNAL_TRANSPORT,
    }
)


def mcplusplus_call_path_evidence() -> str:
    """Return the closed ``vfs/mcplusplus-call-path@1`` evidence term."""

    return EVIDENCE_CALL_PATH


def mcplusplus_manifest_parity_evidence() -> str:
    """Return the closed ``vfs/mcplusplus-manifest-parity@1`` evidence term."""

    return EVIDENCE_MANIFEST_PARITY


def mcplusplus_call_path_evidence_terms() -> tuple[str, ...]:
    """Return the call-path evidence surface for discovery scanners (VFS-G152)."""

    return (EVIDENCE_CALL_PATH,)


def mcplusplus_manifest_parity_evidence_terms() -> tuple[str, ...]:
    """Return the manifest-parity evidence surface for discovery (VFS-G153)."""

    return (EVIDENCE_MANIFEST_PARITY,)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return domain objective evidence terms this static resolver proves.

    Covers ``vfs/mcplusplus-call-path@1`` (VFS-G152) and
    ``vfs/mcplusplus-manifest-parity@1`` (VFS-G153) for the mcp_interop goal
    packet.  Hermetic runtime witnesses (``vfs/mcplusplus-runtime-witness@1``)
    are owned by :data:`HERMETIC_RUNTIME_CHILD_GOAL_ID` and are never mixed
    into this static surface.  Goal/task labels stay metadata and never enter
    path or result content identities.
    """

    return OBJECTIVE_DOMAIN_EVIDENCE_TERMS


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Alias of :func:`covered_evidence_terms` for cross-module discovery."""

    return covered_evidence_terms()


def packet_evidence_terms() -> tuple[str, ...]:
    """Return both static packet evidence terms (call-path + manifest parity)."""

    return OBJECTIVE_DOMAIN_EVIDENCE_TERMS


def path_satisfies_mcplusplus_call_path(
    path: MCPlusPlusCallPath | Mapping[str, Any],
) -> bool:
    """Machine-check VFS-G152 call-path acceptance on one resolved path.

    * Every stage in :data:`PATH_STAGE_ORDER` is present and ``resolved_static``.
    * Non-invocation roles (helpers, mocks, dashboards, fallbacks) never prove.
    * Ambiguous/dynamic hops keep the path non-proved (frontier explicit).
    * Static proof never claims runtime_witnessed authority.
    """

    if isinstance(path, Mapping):
        try:
            path = MCPlusPlusCallPath.from_dict(path)
        except (MCPlusPlusResolverError, TypeError, ValueError):
            return False
    if not isinstance(path, MCPlusPlusCallPath):
        return False
    if path.verdict is not PathVerdict.PROVED:
        return False
    if not path.is_proved or not path.is_statically_proved:
        return False
    if path.is_runtime_witnessed:
        return False
    if path.claim_level is not STATIC_RESOLUTION_CLAIM_LEVEL:
        return False
    if path.resolution_layer is not ResolutionLayer.STATIC:
        return False
    if not path.implementation_ref:
        return False
    stages = [hop.stage.value for hop in path.hops]
    if stages != list(PATH_STAGE_ORDER):
        return False
    for hop in path.hops:
        if hop.status is not ResolverStatus.RESOLVED_STATIC:
            return False
        if hop.reason_code in _NON_INVOCATION_REASON_CODES:
            return False
        if hop.reason_code in _FRONTIER_REASON_CODES:
            return False
        if not hop.evidence:
            return False
    payload = path.to_dict()
    if payload.get("evidence_kind") != EVIDENCE_CALL_PATH:
        return False
    if payload.get("claims_runtime_conformance") is True:
        return False
    if payload.get("is_runtime_witnessed") is True:
        return False
    return True


def manifest_parity_path_report(
    path: MCPlusPlusCallPath | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the evidence-bound VFS-G153 parity matrix for one call path.

    A comparison is ``checked`` only when all required inventory surfaces
    participated. A checked comparison is ``matched`` only when their values
    agree. This distinction prevents absent schemas or error declarations from
    being mistaken for parity merely because no mismatch witness was emitted.
    """

    if isinstance(path, Mapping):
        try:
            path = MCPlusPlusCallPath.from_dict(path)
        except (MCPlusPlusResolverError, TypeError, ValueError):
            return {
                "status": "invalid",
                "satisfied": False,
                "required_aspects": list(MANIFEST_PARITY_REQUIRED_ASPECTS),
                "checked_aspects": [],
                "matched_aspects": [],
                "mismatch_aspects": [],
                "unverified_aspects": list(MANIFEST_PARITY_REQUIRED_ASPECTS),
                "drift_witness_ids": [],
                "drift_kinds": [],
            }
    if not isinstance(path, MCPlusPlusCallPath):
        raise TypeError("path must be an MCPlusPlusCallPath or mapping")

    required = set(MANIFEST_PARITY_REQUIRED_ASPECTS)
    checked: set[str] = set()
    matched: set[str] = set()

    language_names = {
        str(language).strip().lower(): str(name).strip()
        for language, name in path.language_names.items()
        if str(language).strip() and str(name).strip()
    }
    tool_aliases = set(tool_name_aliases(path.tool_name))
    for language, aspect, binding_stage in (
        ("python", "python_name", PathStage.SERVER_REGISTRY),
        ("typescript", "typescript_name", PathStage.TOOLS_CALL),
    ):
        name = language_names.get(language, "")
        binding_hop = path.hop_for(binding_stage)
        if (
            not name
            or binding_hop is None
            or binding_hop.status is not ResolverStatus.RESOLVED_STATIC
        ):
            continue
        checked.add(aspect)
        if tool_aliases & set(tool_name_aliases(name)):
            matched.add(aspect)

    mapping_hop = path.hop_for(PathStage.RESULT_ERROR_MAPPING)
    if mapping_hop is not None:
        note_checked = mapping_hop.notes.get(
            "manifest_parity_checked_aspects", ()
        )
        note_matched = mapping_hop.notes.get(
            "manifest_parity_matched_aspects", ()
        )
        note_contracts = mapping_hop.notes.get(
            "manifest_parity_contract_artifacts", ()
        )
        contract_refs = (
            {str(item) for item in note_contracts}
            if isinstance(note_contracts, (list, tuple))
            else set()
        )
        contract_binding_valid = (
            mapping_hop.reason_code
            in {
                ReasonCode.SCHEMA_PARITY,
                ReasonCode.RESULT_MAP_MATCH,
                ReasonCode.ERROR_MAP_MATCH,
                ReasonCode.MANIFEST_DRIFT,
            }
            and isinstance(note_contracts, (list, tuple))
            and len(note_contracts) == 3
            and len(contract_refs) == 3
            and contract_refs.issubset(mapping_hop.artifact_ids)
        )
        if contract_binding_valid and isinstance(note_checked, (list, tuple)):
            checked.update(
                str(item) for item in note_checked if str(item) in required
            )
        if contract_binding_valid and isinstance(note_matched, (list, tuple)):
            matched.update(
                str(item) for item in note_matched if str(item) in required
            )

    drift_kinds = sorted(
        {witness.drift_kind.value for witness in path.drift_witnesses}
    )
    mismatch_aspects = checked - matched
    for witness in path.drift_witnesses:
        aspect = str(witness.notes.get("aspect") or "")
        if aspect in required:
            mismatch_aspects.add(aspect)
        elif witness.drift_kind is DriftKind.LANGUAGE_NAME_MISMATCH:
            language = str(witness.left_ref).removeprefix("language:")
            if language in {"python", "typescript"}:
                mismatch_aspects.add(f"{language}_name")
        elif witness.drift_kind is DriftKind.SCHEMA_MISMATCH:
            # Older serialized results may not carry the aspect note.
            mismatch_aspects.update({"input_schema", "output_schema"} - matched)
        elif witness.drift_kind is DriftKind.ERROR_MAP_MISMATCH:
            mismatch_aspects.add("error_map")
        elif witness.drift_kind in {
            DriftKind.VERSION_MISMATCH,
            DriftKind.STALE_MANIFEST,
        }:
            mismatch_aspects.add("version")

    unverified = required - checked
    coverage_complete = not unverified
    satisfied = (
        coverage_complete
        and not mismatch_aspects
        and not path.drift_witnesses
    )
    status = "matched" if satisfied else (
        "mismatch" if mismatch_aspects or path.drift_witnesses else "unverified"
    )
    return {
        "path_id": path.path_id,
        "path_name": path.path_name,
        "tool_name": path.tool_name,
        "status": status,
        "satisfied": satisfied,
        "coverage_complete": coverage_complete,
        "required_aspects": list(MANIFEST_PARITY_REQUIRED_ASPECTS),
        "checked_aspects": sorted(checked),
        "matched_aspects": sorted(matched),
        "mismatch_aspects": sorted(mismatch_aspects),
        "unverified_aspects": sorted(unverified),
        "language_names": dict(sorted(language_names.items())),
        "drift_witness_ids": [
            witness.witness_id for witness in path.drift_witnesses
        ],
        "drift_kinds": drift_kinds,
    }


def result_satisfies_mcplusplus_manifest_parity(
    result: MCPlusPlusResolutionResult | Mapping[str, Any],
    *,
    require_proved_path: bool = False,
) -> bool:
    """Machine-check VFS-G153 manifest-parity acceptance on a resolution batch.

    * Static result envelopes declare both packet evidence kinds.
    * Runtime evidence is excluded; conformance is deferred to VFS-G061.
    * Every path has explicit, matched Python/TypeScript names plus input
      schema, output schema, version, and error-map comparisons.
    * Drift (schema, version, language name, error map, missing registration)
      is witnessed and makes parity unsatisfied rather than silently merging.
    * Optionally require at least one fully proved call path.
    """

    if isinstance(result, Mapping):
        try:
            result = MCPlusPlusResolutionResult.from_dict(result)
        except (MCPlusPlusResolverError, TypeError, ValueError):
            return False
    if not isinstance(result, MCPlusPlusResolutionResult):
        return False
    if result.resolution_layer is not ResolutionLayer.STATIC:
        return False
    if result.claims_runtime_conformance:
        return False
    if result.claim_level is not STATIC_RESOLUTION_CLAIM_LEVEL:
        return False
    if result.defers_runtime_to_goal != HERMETIC_RUNTIME_CHILD_GOAL_ID:
        return False
    payload = result.to_dict()
    kinds = list(payload.get("evidence_kinds") or ())
    if EVIDENCE_CALL_PATH not in kinds:
        return False
    if EVIDENCE_MANIFEST_PARITY not in kinds:
        return False
    if EVIDENCE_RUNTIME_WITNESS in kinds:
        return False
    if kinds != list(STATIC_EVIDENCE_KINDS):
        return False
    boundary = payload.get("static_runtime_boundary") or {}
    if not isinstance(boundary, Mapping):
        return False
    if boundary.get("claims_runtime_conformance") is True:
        return False
    if boundary.get("emits_runtime_receipts") is True:
        return False
    if boundary.get("opens_network") is True:
        return False
    if result.truncated or not result.paths:
        return False
    # Drift witnesses must remain self-consistent (no silent empty reason), and
    # any drift keeps the batch parity claim fail-closed.
    for witness in result.drift_witnesses:
        if not witness.drift_kind:
            return False
        if not witness.evidence:
            return False
    if result.drift_witnesses:
        return False
    for path in result.paths:
        for witness in path.drift_witnesses:
            if not witness.drift_kind:
                return False
            if not witness.evidence:
                return False
        # A proved path cannot carry non-invocation reason codes.
        if path.is_proved:
            for hop in path.hops:
                if hop.reason_code in _NON_INVOCATION_REASON_CODES:
                    return False
        if not manifest_parity_path_report(path)["satisfied"]:
            return False
    if require_proved_path and not result.proved_paths():
        return False
    return True


def prove_mcplusplus_call_path(
    path: MCPlusPlusCallPath | Mapping[str, Any],
    *,
    goal_id: str = OBJECTIVE_CALL_PATH_GOAL_ID,
    task_id: str = OBJECTIVE_CALL_PATH_TASK_ID,
) -> dict[str, Any]:
    """Emit a portable ``vfs/mcplusplus-call-path@1`` evidence claim (VFS-G152).

    Goal/task labels are metadata only and never enter :attr:`path_id`.
    """

    if isinstance(path, Mapping):
        path_obj = MCPlusPlusCallPath.from_dict(path)
    else:
        path_obj = path
    if not isinstance(path_obj, MCPlusPlusCallPath):
        raise TypeError("path must be an MCPlusPlusCallPath")

    satisfied = path_satisfies_mcplusplus_call_path(path_obj)
    hop_summaries = [
        {
            "stage": hop.stage.value,
            "status": hop.status.value,
            "reason_code": hop.reason_code.value,
            "artifact_ids": list(hop.artifact_ids),
        }
        for hop in path_obj.hops
    ]
    non_invocation_rejected = any(
        hop.reason_code in _NON_INVOCATION_REASON_CODES for hop in path_obj.hops
    )
    frontier_explicit = path_obj.has_frontier or any(
        hop.reason_code in _FRONTIER_REASON_CODES for hop in path_obj.hops
    )
    return {
        "schema": MCPLUSPLUS_CALL_PATH_CLAIM_SCHEMA,
        "evidence": EVIDENCE_CALL_PATH,
        "evidence_terms": list(mcplusplus_call_path_evidence_terms()),
        "requirement_id": EVIDENCE_CALL_PATH,
        "goal_id": str(goal_id or OBJECTIVE_CALL_PATH_GOAL_ID),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": str(task_id or OBJECTIVE_CALL_PATH_TASK_ID),
        "goal_packet_id": OBJECTIVE_GOAL_PACKET_ID,
        "path_id": path_obj.path_id,
        "path_name": path_obj.path_name,
        "forest_id": path_obj.forest_id,
        "tool_name": path_obj.tool_name,
        "server_name": path_obj.server_name,
        "transport": path_obj.transport.value,
        "profiles": list(path_obj.profiles),
        "implementation_ref": path_obj.implementation_ref,
        "language_names": dict(path_obj.language_names),
        "verdict": path_obj.verdict.value,
        "claim_level": path_obj.claim_level.value,
        "resolution_layer": path_obj.resolution_layer.value,
        "is_proved": path_obj.is_proved,
        "is_statically_proved": path_obj.is_statically_proved,
        "is_runtime_witnessed": path_obj.is_runtime_witnessed,
        "claims_runtime_conformance": False,
        "stage_order": list(PATH_STAGE_ORDER),
        "hops": hop_summaries,
        "non_invocation_rejected": non_invocation_rejected,
        "frontier_explicit": frontier_explicit,
        "drift_count": len(path_obj.drift_witnesses),
        "satisfied": satisfied,
        "invariants": list(CALL_PATH_INVARIANTS),
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_mcplusplus_manifest_parity(
    result: MCPlusPlusResolutionResult | Mapping[str, Any],
    *,
    goal_id: str = OBJECTIVE_MANIFEST_PARITY_GOAL_ID,
    task_id: str = OBJECTIVE_MANIFEST_PARITY_TASK_ID,
    require_proved_path: bool = False,
) -> dict[str, Any]:
    """Emit a portable ``vfs/mcplusplus-manifest-parity@1`` claim (VFS-G153).

    Goal/task labels are metadata only and never enter :attr:`result_id`.
    """

    if isinstance(result, Mapping):
        result_obj = MCPlusPlusResolutionResult.from_dict(result)
    else:
        result_obj = result
    if not isinstance(result_obj, MCPlusPlusResolutionResult):
        raise TypeError("result must be an MCPlusPlusResolutionResult")

    satisfied = result_satisfies_mcplusplus_manifest_parity(
        result_obj, require_proved_path=require_proved_path
    )
    witnesses = {
        item.witness_id: item
        for item in (
            *result_obj.drift_witnesses,
            *(
                item
                for path in result_obj.paths
                for item in path.drift_witnesses
            ),
        )
    }
    drift_kinds = sorted(
        {item.drift_kind.value for item in witnesses.values()}
    )
    path_checks = [
        manifest_parity_path_report(path) for path in result_obj.paths
    ]
    checked_aspects = sorted(
        {
            aspect
            for report in path_checks
            for aspect in report["checked_aspects"]
        }
    )
    matched_aspects = sorted(
        {
            aspect
            for report in path_checks
            for aspect in report["matched_aspects"]
        }
    )
    mismatch_aspects = sorted(
        {
            aspect
            for report in path_checks
            for aspect in report["mismatch_aspects"]
        }
    )
    unverified_aspects = sorted(
        {
            aspect
            for report in path_checks
            for aspect in report["unverified_aspects"]
        }
    )
    parity_status = "matched" if satisfied else (
        "mismatch" if drift_kinds or mismatch_aspects else "unverified"
    )
    return {
        "schema": MCPLUSPLUS_MANIFEST_PARITY_CLAIM_SCHEMA,
        "evidence": EVIDENCE_MANIFEST_PARITY,
        "evidence_terms": list(mcplusplus_manifest_parity_evidence_terms()),
        "requirement_id": EVIDENCE_MANIFEST_PARITY,
        "goal_id": str(goal_id or OBJECTIVE_MANIFEST_PARITY_GOAL_ID),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": str(task_id or OBJECTIVE_MANIFEST_PARITY_TASK_ID),
        "goal_packet_id": OBJECTIVE_GOAL_PACKET_ID,
        "result_id": result_obj.result_id,
        "forest_id": result_obj.forest_id,
        "inventory_id": result_obj.inventory_id,
        "resolver_version": result_obj.resolver_version,
        "claim_level": result_obj.claim_level.value,
        "resolution_layer": result_obj.resolution_layer.value,
        "claims_runtime_conformance": result_obj.claims_runtime_conformance,
        "defers_runtime_to_goal": result_obj.defers_runtime_to_goal,
        "evidence_kinds": list(STATIC_EVIDENCE_KINDS),
        "excluded_evidence_kinds": list(EXCLUDED_RUNTIME_EVIDENCE_KINDS),
        "path_count": len(result_obj.paths),
        "proved_count": len(result_obj.proved_paths()),
        "statically_proved_count": len(result_obj.statically_proved_paths()),
        "runtime_witnessed_count": 0,
        "required_aspects": list(MANIFEST_PARITY_REQUIRED_ASPECTS),
        "checked_aspects": checked_aspects,
        "matched_aspects": matched_aspects,
        "mismatch_aspects": mismatch_aspects,
        "unverified_aspects": unverified_aspects,
        "coverage_complete": bool(path_checks)
        and all(report["coverage_complete"] for report in path_checks),
        "parity_status": parity_status,
        "path_checks": path_checks,
        "drift_count": len(witnesses),
        "drift_kinds": drift_kinds,
        "drift_witnesses": [
            {
                "witness_id": witness.witness_id,
                "drift_kind": witness.drift_kind.value,
                "tool_name": witness.tool_name,
                "left_ref": witness.left_ref,
                "right_ref": witness.right_ref,
                "aspect": str(witness.notes.get("aspect") or ""),
            }
            for witness in sorted(
                witnesses.values(), key=lambda item: item.witness_id
            )
        ],
        "frontier_count": len(result_obj.frontiers),
        "language_names_by_path": {
            path.path_id: dict(sorted(path.language_names.items()))
            for path in result_obj.paths
        },
        "require_proved_path": bool(require_proved_path),
        "satisfied": satisfied,
        "invariants": list(MANIFEST_PARITY_INVARIANTS),
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_mcplusplus_static_packet(
    result: MCPlusPlusResolutionResult | Mapping[str, Any],
    *,
    path: MCPlusPlusCallPath | Mapping[str, Any] | None = None,
    require_proved_path: bool = True,
) -> dict[str, Any]:
    """Emit the full mcp_interop static packet (call-path + manifest parity).

    Covers goal packet ``goal_packet/mcp_interop/ipfs_accelerate_py/9f2828fd2adb``
    leaf goals VFS-G152 and VFS-G153 in one claim.  When *path* is omitted the
    first statically proved path is selected; if none exist and
    *require_proved_path* is true, the call-path subclaim is unsatisfied.
    """

    if isinstance(result, Mapping):
        result_obj = MCPlusPlusResolutionResult.from_dict(result)
    else:
        result_obj = result
    if not isinstance(result_obj, MCPlusPlusResolutionResult):
        raise TypeError("result must be an MCPlusPlusResolutionResult")

    path_obj: MCPlusPlusCallPath | None
    if path is not None:
        if isinstance(path, Mapping):
            path_obj = MCPlusPlusCallPath.from_dict(path)
        else:
            path_obj = path
        if not isinstance(path_obj, MCPlusPlusCallPath):
            raise TypeError("path must be an MCPlusPlusCallPath")
    else:
        proved = result_obj.proved_paths()
        path_obj = proved[0] if proved else (
            result_obj.paths[0] if result_obj.paths else None
        )

    call_path_claim: dict[str, Any] | None = None
    if path_obj is not None:
        call_path_claim = prove_mcplusplus_call_path(path_obj)
    parity_claim = prove_mcplusplus_manifest_parity(
        result_obj, require_proved_path=require_proved_path
    )
    call_path_satisfied = bool(
        call_path_claim and call_path_claim.get("satisfied")
    )
    if path_obj is None and require_proved_path:
        call_path_satisfied = False
    parity_satisfied = bool(parity_claim.get("satisfied"))
    satisfied = call_path_satisfied and parity_satisfied
    return {
        "schema": MCPLUSPLUS_STATIC_PACKET_CLAIM_SCHEMA,
        "evidence_terms": list(packet_evidence_terms()),
        "requirement_ids": list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS),
        "goal_packet_id": OBJECTIVE_GOAL_PACKET_ID,
        "goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
        "task_ids": list(OBJECTIVE_PACKET_TASK_IDS),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "result_id": result_obj.result_id,
        "forest_id": result_obj.forest_id,
        "call_path_claim": call_path_claim,
        "manifest_parity_claim": parity_claim,
        "call_path_satisfied": call_path_satisfied,
        "manifest_parity_satisfied": parity_satisfied,
        "satisfied": satisfied,
        "resolution_layer": ResolutionLayer.STATIC.value,
        "claims_runtime_conformance": False,
        "defers_runtime_to_goal": HERMETIC_RUNTIME_CHILD_GOAL_ID,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


__all__ = [
    "ArtifactRole",
    "CALL_PATH_INVARIANTS",
    "CallPathClaim",
    "DEFAULT_MAX_ARTIFACTS",
    "DEFAULT_MAX_DRIFT_WITNESSES",
    "DEFAULT_MAX_FRONTIER_ITEMS",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_MAX_PATHS",
    "DriftKind",
    "EVIDENCE_CALL_PATH",
    "EVIDENCE_MANIFEST_PARITY",
    "EVIDENCE_RUNTIME_WITNESS",
    "EXCLUDED_RUNTIME_EVIDENCE_KINDS",
    "FrontierItem",
    "HERMETIC_RUNTIME_CHILD_GOAL_ID",
    "HERMETIC_RUNTIME_CLAIM_LEVEL",
    "InventoryArtifact",
    "MANIFEST_PARITY_INVARIANTS",
    "MANIFEST_PARITY_REQUIRED_ASPECTS",
    "MCPLUSPLUS_ARTIFACT_SCHEMA",
    "MCPLUSPLUS_CALL_PATH_CLAIM_SCHEMA",
    "MCPLUSPLUS_CALL_PATH_SCHEMA",
    "MCPLUSPLUS_CONTRACT_RESOLVER_SCHEMA",
    "MCPLUSPLUS_FRONTIER_SCHEMA",
    "MCPLUSPLUS_INVENTORY_SCHEMA",
    "MCPLUSPLUS_MANIFEST_DRIFT_SCHEMA",
    "MCPLUSPLUS_MANIFEST_PARITY_CLAIM_SCHEMA",
    "MCPLUSPLUS_PATH_EVIDENCE_SCHEMA",
    "MCPLUSPLUS_PATH_HOP_SCHEMA",
    "MCPLUSPLUS_RESOLUTION_RESULT_SCHEMA",
    "MCPLUSPLUS_STATIC_PACKET_CLAIM_SCHEMA",
    "MCPlusPlusCallPath",
    "MCPlusPlusContractResolver",
    "MCPlusPlusInventory",
    "MCPlusPlusResolutionResult",
    "MCPlusPlusResolverBoundsError",
    "MCPlusPlusResolverError",
    "ManufacturedInvocationError",
    "ManifestDriftWitness",
    "MissingPathEvidenceError",
    "OBJECTIVE_CALL_PATH_GOAL_ID",
    "OBJECTIVE_CALL_PATH_TASK_ID",
    "OBJECTIVE_DOMAIN_EVIDENCE_TERMS",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_GOAL_PACKET_ID",
    "OBJECTIVE_MANIFEST_PARITY_GOAL_ID",
    "OBJECTIVE_MANIFEST_PARITY_TASK_ID",
    "OBJECTIVE_PACKET_GOAL_IDS",
    "OBJECTIVE_PACKET_TASK_IDS",
    "OBJECTIVE_PARENT_GOAL_ID",
    "OBJECTIVE_TASK_ID",
    "PATH_STAGE_ORDER",
    "PathEvidence",
    "PathHop",
    "PathStage",
    "PathVerdict",
    "RESOLUTION_LAYER_RUNTIME",
    "RESOLUTION_LAYER_STATIC",
    "RESOLVER_PRODUCER",
    "RESOLVER_VERSION",
    "ReasonCode",
    "ResolutionLayer",
    "STATIC_EVIDENCE_KINDS",
    "STATIC_RESOLUTION_CLAIM_LEVEL",
    "STATIC_RESOLUTION_GOAL_ID",
    "TransportKind",
    "all_covered_evidence_terms",
    "classify_non_invocation",
    "confidence_for",
    "covered_evidence_terms",
    "inventory_from_program_graph",
    "make_artifact",
    "make_evidence",
    "make_hop",
    "mcplusplus_call_path_evidence",
    "mcplusplus_call_path_evidence_terms",
    "mcplusplus_manifest_parity_evidence",
    "mcplusplus_manifest_parity_evidence_terms",
    "manifest_parity_path_report",
    "normalize_tool_name",
    "packet_evidence_terms",
    "path_satisfies_mcplusplus_call_path",
    "prove_mcplusplus_call_path",
    "prove_mcplusplus_manifest_parity",
    "prove_mcplusplus_static_packet",
    "resolve_mcplusplus_from_graph",
    "resolve_mcplusplus_paths",
    "result_satisfies_mcplusplus_manifest_parity",
    "schema_fingerprint",
    "split_hierarchical_alias",
    "static_resolution_boundary",
    "tool_name_aliases",
]
