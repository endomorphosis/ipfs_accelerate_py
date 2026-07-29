"""Conservative cross-language call and import resolver (VFS-009).

This module consumes a canonical :class:`~.program_graph.ProgramGraph` and
emits typed resolution records for imports, re-exports, and call sites.
It never mutates source AST records and never manufactures a direct
``calls`` / ``resolves_to`` edge to improve coverage.

Static resolution is fail-closed:

* relative/package imports, aliases, re-exports, class/member calls, known
  registrations, generated SDK methods, and explicit cross-package interfaces
  may become ``resolved_static`` when exactly one target is proven by
  evidence;
* same-name collisions, re-export loops, namespace packages, optional
  imports, uninstalled dependencies, and multi-candidate sites remain
  ``candidate`` / ``ambiguous`` / ``external`` / ``unknown``;
* dependency injection, callbacks, monkey patches, dynamic imports,
  subprocess, HTTP, RPC, libp2p, and MCP sites are typed frontier edges and
  require evidence — they never upgrade to ``resolved_static`` by heuristic.

Confidence values and reason codes are deterministic pure functions of the
matched rule and status.  GraphRAG / model enrichment is out of scope.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .program_graph import (
    GraphFrontierItem,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    build_program_graph,
    canonical_program_json,
    make_edge,
)
from .proof.formal_verification_contracts import content_identity


PROGRAM_CALL_RESOLVER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-call-resolver@1"
)
PROGRAM_CALL_RESOLUTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-call-resolution@1"
)
PROGRAM_CALL_RESOLUTION_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-call-resolution-result@1"
)
PROGRAM_CALL_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-call-evidence@1"
)

RESOLVER_VERSION = "program-call-resolver@1"
RESOLVER_PRODUCER = "program-call-resolver@1"

DEFAULT_MAX_RESOLUTIONS = 250_000
DEFAULT_MAX_REEXPORT_DEPTH = 64
DEFAULT_MAX_EVIDENCE_NOTES_BYTES = 4_096
DEFAULT_MAX_LABEL_BYTES = 4_096

# Mechanisms that must never become resolved_static without an explicit
# binding already present as a known registration / interface edge.
_DYNAMIC_MECHANISMS: frozenset[str] = frozenset(
    {
        "dependency_injection",
        "callback",
        "monkey_patch",
        "dynamic_import",
        "subprocess",
        "http",
        "rpc",
        "libp2p",
        "mcp",
    }
)

# Callee roots / patterns that imply a typed external transport boundary.
_SUBPROCESS_CALLEES: frozenset[str] = frozenset(
    {
        "subprocess",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "os.system",
        "os.popen",
        "os.execv",
        "os.execve",
        "os.execl",
        "os.execle",
        "os.execvp",
        "os.execvpe",
    }
)
_HTTP_CALLEES: frozenset[str] = frozenset(
    {
        "requests",
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
        "requests.request",
        "httpx",
        "httpx.get",
        "httpx.post",
        "httpx.request",
        "urllib",
        "urllib.request",
        "urllib.request.urlopen",
        "aiohttp",
        "fetch",
        "axios",
        "axios.get",
        "axios.post",
    }
)
_RPC_CALLEES: frozenset[str] = frozenset(
    {
        "grpc",
        "jsonrpc",
        "xmlrpc",
        "xmlrpc.client",
        "rpc",
        "RPCClient",
        "call_rpc",
    }
)
_LIBP2P_CALLEES: frozenset[str] = frozenset(
    {
        "libp2p",
        "libp2p.dial",
        "libp2p.hangUp",
        "libp2p.handle",
        "libp2p.pubsub",
        "ipfs.pubsub",
        "ipfs.libp2p",
    }
)
_MCP_CALLEES: frozenset[str] = frozenset(
    {
        "tools/call",
        "tools.call",
        "call_tool",
        "mcp.call",
        "mcp.tools.call",
        "client.call_tool",
        "session.call_tool",
    }
)
_DYNAMIC_IMPORT_CALLEES: frozenset[str] = frozenset(
    {
        "__import__",
        "importlib.import_module",
        "importlib.__import__",
        "import_module",
    }
)
_MONKEY_PATCH_CALLEES: frozenset[str] = frozenset(
    {
        "setattr",
        "builtins.setattr",
        "unittest.mock.patch",
        "mock.patch",
        "patch",
    }
)

_RELATIVE_IMPORT_RE = re.compile(r"^(\.+)(.*)$")


class CallResolverError(ValueError):
    """A call-resolution input or record violates the fail-closed contract."""


class CallResolverBoundsError(CallResolverError):
    """A resolution result exceeded a hard deterministic bound."""


class MissingEvidenceError(CallResolverError):
    """A frontier or static resolution was emitted without required evidence."""


class ManufacturedEdgeError(CallResolverError):
    """A caller attempted to mint a direct edge without a site and rule."""


class ReasonCode(str, Enum):
    """Closed vocabulary of deterministic resolution reason codes."""

    RELATIVE_IMPORT = "relative_import"
    PACKAGE_IMPORT = "package_import"
    ALIAS_BINDING = "alias_binding"
    REEXPORT = "reexport"
    CLASS_MEMBER = "class_member"
    KNOWN_REGISTRATION = "known_registration"
    GENERATED_SDK_METHOD = "generated_sdk_method"
    CROSS_PACKAGE_INTERFACE = "cross_package_interface"
    SAME_MODULE_DEFINITION = "same_module_definition"

    SAME_NAME_COLLISION = "same_name_collision"
    REEXPORT_LOOP = "reexport_loop"
    NAMESPACE_PACKAGE = "namespace_package"
    OPTIONAL_IMPORT = "optional_import"
    UNINSTALLED_DEPENDENCY = "uninstalled_dependency"
    GENERATED_CLIENT = "generated_client"

    DEPENDENCY_INJECTION = "dependency_injection"
    CALLBACK = "callback"
    MONKEY_PATCH = "monkey_patch"
    DYNAMIC_IMPORT = "dynamic_import"
    SUBPROCESS = "subprocess"
    HTTP = "http"
    RPC = "rpc"
    LIBP2P = "libp2p"
    MCP = "mcp"

    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    EXTERNAL_MODULE = "external_module"
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"
    NO_TARGET = "no_target"
    UNRESOLVED_NAME = "unresolved_name"
    EVIDENCE_REQUIRED = "evidence_required"
    ALREADY_RESOLVED = "already_resolved"
    NO_SITE = "no_site"


# Deterministic confidence table.  Values are fixed; never inferred from
# heuristics or coverage goals.
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

# Reason-code adjustments applied after the status baseline (still deterministic).
_CONFIDENCE_BY_REASON: Mapping[ReasonCode, int] = MappingProxyType(
    {
        ReasonCode.RELATIVE_IMPORT: 100,
        ReasonCode.PACKAGE_IMPORT: 100,
        ReasonCode.ALIAS_BINDING: 100,
        ReasonCode.REEXPORT: 100,
        ReasonCode.CLASS_MEMBER: 100,
        ReasonCode.KNOWN_REGISTRATION: 50,
        ReasonCode.GENERATED_SDK_METHOD: 100,
        ReasonCode.CROSS_PACKAGE_INTERFACE: 100,
        ReasonCode.SAME_MODULE_DEFINITION: 100,
        ReasonCode.SAME_NAME_COLLISION: 25,
        ReasonCode.REEXPORT_LOOP: 25,
        ReasonCode.NAMESPACE_PACKAGE: 25,
        ReasonCode.OPTIONAL_IMPORT: 40,
        ReasonCode.UNINSTALLED_DEPENDENCY: 40,
        ReasonCode.GENERATED_CLIENT: 50,
        ReasonCode.DEPENDENCY_INJECTION: 25,
        ReasonCode.CALLBACK: 25,
        ReasonCode.MONKEY_PATCH: 25,
        ReasonCode.DYNAMIC_IMPORT: 40,
        ReasonCode.SUBPROCESS: 40,
        ReasonCode.HTTP: 40,
        ReasonCode.RPC: 40,
        ReasonCode.LIBP2P: 40,
        ReasonCode.MCP: 40,
        ReasonCode.AMBIGUOUS_CANDIDATES: 25,
        ReasonCode.EXTERNAL_MODULE: 40,
        ReasonCode.UNSUPPORTED_CONSTRUCT: 5,
        ReasonCode.NO_TARGET: 0,
        ReasonCode.UNRESOLVED_NAME: 0,
        ReasonCode.EVIDENCE_REQUIRED: 0,
        ReasonCode.ALREADY_RESOLVED: 100,
        ReasonCode.NO_SITE: 0,
    }
)

_DYNAMIC_REASON_BY_MECHANISM: Mapping[str, ReasonCode] = MappingProxyType(
    {
        "dependency_injection": ReasonCode.DEPENDENCY_INJECTION,
        "callback": ReasonCode.CALLBACK,
        "monkey_patch": ReasonCode.MONKEY_PATCH,
        "dynamic_import": ReasonCode.DYNAMIC_IMPORT,
        "subprocess": ReasonCode.SUBPROCESS,
        "http": ReasonCode.HTTP,
        "rpc": ReasonCode.RPC,
        "libp2p": ReasonCode.LIBP2P,
        "mcp": ReasonCode.MCP,
    }
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise CallResolverError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise CallResolverError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
        raise CallResolverBoundsError(f"{name} exceeds label bound")
    return text


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise CallResolverError(f"unsupported {label}: {text!r}") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise CallResolverError(f"{name} must be a mapping")
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CallResolverError(f"{name} keys must be strings")
        if isinstance(item, (str, bool, int)) or item is None:
            plain[key] = item
        elif isinstance(item, Enum):
            plain[key] = item.value
        elif isinstance(item, Mapping):
            plain[key] = dict(_mapping(item, f"{name}.{key}"))
        elif isinstance(item, (list, tuple)):
            plain[key] = [
                (
                    dict(_mapping(entry, f"{name}.{key}[]"))
                    if isinstance(entry, Mapping)
                    else entry
                )
                for entry in item
            ]
        else:
            raise CallResolverError(
                f"{name}.{key} has unsupported type {type(item).__name__}"
            )
    encoded = canonical_program_json(plain).encode("utf-8")
    if len(encoded) > DEFAULT_MAX_EVIDENCE_NOTES_BYTES:
        raise CallResolverBoundsError(f"{name} exceeds evidence notes bound")
    return MappingProxyType(dict(sorted(plain.items())))


def confidence_for(
    status: ResolverStatus | str,
    reason_code: ReasonCode | str,
) -> int:
    """Return the deterministic confidence for a status/reason pair.

    Confidence is never learned or tuned for coverage.  The status baseline is
    an upper bound; a reason code may only lower confidence, never raise it
    above what the resolver status permits.
    """

    status_enum = _enum(status, ResolverStatus, "resolver_status")
    reason_enum = _enum(reason_code, ReasonCode, "reason_code")
    status_conf = int(_CONFIDENCE_BY_STATUS[status_enum])
    reason_conf = int(_CONFIDENCE_BY_REASON.get(reason_enum, status_conf))
    return min(status_conf, reason_conf)


def _simple_name(qualified: str) -> str:
    text = (qualified or "").strip()
    if not text:
        return ""
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _package_of_module(module_qname: str) -> str:
    text = (module_qname or "").strip()
    if not text or "." not in text:
        return ""
    return text.rsplit(".", 1)[0]


def _join_module(base: str, relative: str) -> str:
    base = (base or "").strip().strip(".")
    relative = (relative or "").strip()
    if not relative:
        return base
    if not base:
        return relative.lstrip(".")
    return f"{base}.{relative.lstrip('.')}"


def resolve_relative_module(
    current_module: str,
    import_target: str,
    *,
    is_package: bool = False,
) -> str:
    """Resolve a Python-style relative import target against ``current_module``.

    ``import_target`` may start with one or more dots.  Empty residual after
    dots refers to the package itself.
    """

    target = _text(import_target, "import_target")
    match = _RELATIVE_IMPORT_RE.match(target)
    if match is None:
        return target

    dots, remainder = match.group(1), match.group(2)
    level = len(dots)
    module = _text(current_module, "current_module", required=False)
    parts = [part for part in module.split(".") if part] if module else []

    # from .x import y  inside package.mod → package.x (pop one for module)
    # from .x import y  inside package (is_package) → package.x
    pops = level if is_package else level
    if not is_package and parts:
        # Drop the module leaf so relative imports are package-relative.
        parts = parts[:-1]
    if pops > 0:
        # level=1 means "current package"; additional dots walk up.
        up = max(0, pops - 1)
        if up > len(parts):
            raise CallResolverError(
                f"relative import {target!r} escapes package of {current_module!r}"
            )
        if up:
            parts = parts[: len(parts) - up]
    residual = remainder.strip(".")
    if residual:
        parts.extend(part for part in residual.split(".") if part)
    return ".".join(parts)


@dataclass(frozen=True)
class ResolutionEvidence:
    """Required provenance for one resolution decision."""

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
        return "pevid-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_CALL_EVIDENCE_SCHEMA,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolutionEvidence":
        if not isinstance(payload, Mapping):
            raise CallResolverError("evidence payload must be a mapping")
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
class CallResolution:
    """One fail-closed resolution for a call site, import, or re-export."""

    site_id: str
    site_kind: str
    status: ResolverStatus
    reason_code: ReasonCode
    confidence: int
    targets: tuple[str, ...] = ()
    evidence: tuple[ResolutionEvidence, ...] = ()
    edge_kind: ProgramEdgeKind = ProgramEdgeKind.RESOLVES_TO
    mechanism: str = "static"
    component_id: str = ""
    site_qualified_name: str = ""
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_id", _text(self.site_id, "site_id"))
        object.__setattr__(self, "site_kind", _text(self.site_kind, "site_kind"))
        object.__setattr__(
            self, "status", _enum(self.status, ResolverStatus, "status")
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, ReasonCode, "reason_code"),
        )
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, int):
            raise CallResolverError("confidence must be an integer")
        if self.confidence < 0 or self.confidence > 100:
            raise CallResolverError("confidence must be in 0..100")
        expected = confidence_for(self.status, self.reason_code)
        if self.confidence != expected:
            raise CallResolverError(
                f"confidence {self.confidence} is not deterministic for "
                f"{self.status.value}/{self.reason_code.value} (expected {expected})"
            )
        targets = tuple(
            _text(item, "target", required=True) for item in (self.targets or ())
        )
        object.__setattr__(self, "targets", targets)
        evidence = tuple(
            item
            if isinstance(item, ResolutionEvidence)
            else ResolutionEvidence.from_dict(item)
            for item in (self.evidence or ())
        )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "edge_kind",
            _enum(self.edge_kind, ProgramEdgeKind, "edge_kind"),
        )
        object.__setattr__(
            self,
            "mechanism",
            _text(self.mechanism or "static", "mechanism", required=False) or "static",
        )
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "component_id", required=False),
        )
        object.__setattr__(
            self,
            "site_qualified_name",
            _text(self.site_qualified_name, "site_qualified_name", required=False),
        )
        object.__setattr__(self, "record", _mapping(self.record, "resolution.record"))
        self._validate_evidence_contract()

    def _validate_evidence_contract(self) -> None:
        # Every emitted resolution requires evidence: no fabricated edges.
        if not self.evidence:
            raise MissingEvidenceError(
                f"resolution for {self.site_id!r} requires evidence "
                f"(status={self.status.value}, reason={self.reason_code.value})"
            )
        if self.status is ResolverStatus.RESOLVED_STATIC:
            if len(self.targets) != 1:
                raise CallResolverError(
                    "resolved_static requires exactly one target"
                )
            if self.mechanism in _DYNAMIC_MECHANISMS:
                raise CallResolverError(
                    f"mechanism {self.mechanism!r} cannot be resolved_static"
                )
        if self.status is ResolverStatus.AMBIGUOUS and len(self.targets) < 2:
            # Ambiguity may be structural (loop/namespace) without multiple
            # concrete targets; only same-name collisions require >= 2.
            if self.reason_code is ReasonCode.SAME_NAME_COLLISION:
                raise CallResolverError(
                    "same_name_collision requires at least two targets"
                )
        if self.status is ResolverStatus.CANDIDATE and not self.targets:
            if self.reason_code not in {
                ReasonCode.OPTIONAL_IMPORT,
                ReasonCode.GENERATED_CLIENT,
                ReasonCode.DEPENDENCY_INJECTION,
                ReasonCode.CALLBACK,
                ReasonCode.MONKEY_PATCH,
                ReasonCode.DYNAMIC_IMPORT,
                ReasonCode.MCP,
                ReasonCode.KNOWN_REGISTRATION,
            }:
                # Candidate with no target still needs evidence (checked above)
                # but is allowed for open frontiers.
                pass

    @property
    def resolution_id(self) -> str:
        return "pres-" + content_identity(self._identity_payload())

    @property
    def is_frontier(self) -> bool:
        return self.status.frontier

    @property
    def is_direct_edge_allowed(self) -> bool:
        """True only for single-target static resolutions."""

        return (
            self.status is ResolverStatus.RESOLVED_STATIC
            and len(self.targets) == 1
            and self.mechanism not in _DYNAMIC_MECHANISMS
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_CALL_RESOLUTION_SCHEMA,
            "site_id": self.site_id,
            "site_kind": self.site_kind,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "confidence": self.confidence,
            "targets": list(self.targets),
            "evidence": [item.to_dict() for item in self.evidence],
            "edge_kind": self.edge_kind.value,
            "mechanism": self.mechanism,
            "component_id": self.component_id,
            "site_qualified_name": self.site_qualified_name,
            "record": dict(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "resolution_id": self.resolution_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallResolution":
        if not isinstance(payload, Mapping):
            raise CallResolverError("resolution payload must be a mapping")
        evidence_raw = payload.get("evidence") or ()
        return cls(
            site_id=str(payload.get("site_id") or ""),
            site_kind=str(payload.get("site_kind") or ""),
            status=payload.get("status", ResolverStatus.UNRESOLVED.value),
            reason_code=payload.get("reason_code", ReasonCode.NO_TARGET.value),
            confidence=int(payload.get("confidence") or 0),
            targets=tuple(payload.get("targets") or ()),
            evidence=tuple(evidence_raw),
            edge_kind=payload.get("edge_kind", ProgramEdgeKind.RESOLVES_TO.value),
            mechanism=str(payload.get("mechanism") or "static"),
            component_id=str(payload.get("component_id") or ""),
            site_qualified_name=str(payload.get("site_qualified_name") or ""),
            record=payload.get("record") or {},
        )


@dataclass(frozen=True)
class CallResolutionResult:
    """Deterministic batch of resolutions over one program graph."""

    forest_id: str
    resolver_version: str
    source_graph_id: str
    resolutions: tuple[CallResolution, ...] = ()
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
            self,
            "source_graph_id",
            _text(self.source_graph_id, "source_graph_id"),
        )
        resolutions = tuple(
            item
            if isinstance(item, CallResolution)
            else CallResolution.from_dict(item)
            for item in (self.resolutions or ())
        )
        # Stable order by site_id then reason then resolution identity.
        ordered = tuple(
            sorted(
                resolutions,
                key=lambda item: (
                    item.site_id,
                    item.reason_code.value,
                    item.resolution_id,
                ),
            )
        )
        object.__setattr__(self, "resolutions", ordered)
        if not isinstance(self.truncated, bool):
            raise CallResolverError("truncated must be a boolean")
        object.__setattr__(
            self,
            "truncation_reason",
            _text(self.truncation_reason, "truncation_reason", required=False),
        )
        if len(ordered) > DEFAULT_MAX_RESOLUTIONS:
            raise CallResolverBoundsError("too many resolutions")

    @property
    def result_id(self) -> str:
        return "presr-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_CALL_RESOLUTION_RESULT_SCHEMA,
            "forest_id": self.forest_id,
            "resolver_version": self.resolver_version,
            "source_graph_id": self.source_graph_id,
            "resolutions": [item.to_dict() for item in self.resolutions],
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "result_id": self.result_id}

    def resolutions_for_site(self, site_id: str) -> tuple[CallResolution, ...]:
        key = _text(site_id, "site_id")
        return tuple(item for item in self.resolutions if item.site_id == key)

    def frontier(self) -> tuple[GraphFrontierItem, ...]:
        items: list[GraphFrontierItem] = []
        for resolution in self.resolutions:
            if not resolution.is_frontier:
                continue
            items.append(
                GraphFrontierItem(
                    element_id=resolution.resolution_id,
                    element_kind=f"resolution:{resolution.site_kind}",
                    resolver_status=resolution.status,
                    reason=resolution.reason_code.value,
                    component_id=resolution.component_id,
                    qualified_name=resolution.site_qualified_name,
                )
            )
        return tuple(
            sorted(items, key=lambda item: (item.element_id, item.element_kind))
        )

    def stats(self) -> Mapping[str, Any]:
        by_status: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        by_mechanism: dict[str, int] = {}
        direct = 0
        for item in self.resolutions:
            by_status[item.status.value] = by_status.get(item.status.value, 0) + 1
            by_reason[item.reason_code.value] = (
                by_reason.get(item.reason_code.value, 0) + 1
            )
            by_mechanism[item.mechanism] = by_mechanism.get(item.mechanism, 0) + 1
            if item.is_direct_edge_allowed:
                direct += 1
        return MappingProxyType(
            {
                "resolution_count": len(self.resolutions),
                "frontier_count": len(self.frontier()),
                "direct_edge_count": direct,
                "by_status": dict(sorted(by_status.items())),
                "by_reason": dict(sorted(by_reason.items())),
                "by_mechanism": dict(sorted(by_mechanism.items())),
                "truncated": self.truncated,
            }
        )

    def resolution_edges(
        self,
        graph: ProgramGraph,
        *,
        node_id_by_qualified_name: Mapping[str, str] | None = None,
    ) -> tuple[ProgramGraphEdge, ...]:
        """Materialize resolution edges for sites that allow a direct edge.

        Candidate/ambiguous/external resolutions produce edges only when a
        concrete target node is known; their status remains non-terminal.
        No edge is minted without a site node and evidence.
        """

        if not isinstance(graph, ProgramGraph):
            raise CallResolverError("graph must be a ProgramGraph")
        if graph.forest_id != self.forest_id:
            raise CallResolverError("result forest_id does not match graph")

        qname_index: dict[str, str] = {}
        if node_id_by_qualified_name:
            for key, value in node_id_by_qualified_name.items():
                qname_index[_text(key, "qname")] = _text(value, "node_id")
        else:
            for node in graph.nodes:
                if node.qualified_name:
                    # First deterministic node wins for multi-definition names;
                    # ambiguous sites will still carry multi-target records.
                    qname_index.setdefault(node.qualified_name, node.node_id)
                    simple = _simple_name(node.qualified_name)
                    if simple and node.kind in {
                        ProgramNodeKind.SYMBOL,
                        ProgramNodeKind.DEFINITION,
                    }:
                        qname_index.setdefault(simple, node.node_id)

        node_ids = {node.node_id for node in graph.nodes}
        edges: list[ProgramGraphEdge] = []
        for resolution in self.resolutions:
            if resolution.site_id not in node_ids:
                # Never manufacture an edge for a missing site.
                continue
            if not resolution.evidence:
                raise MissingEvidenceError(
                    f"cannot materialize edge for {resolution.site_id!r}"
                )
            targets = list(resolution.targets)
            if resolution.status is ResolverStatus.RESOLVED_STATIC:
                if not resolution.is_direct_edge_allowed:
                    raise ManufacturedEdgeError(
                        f"refusing direct edge for {resolution.site_id!r}"
                    )
            for target_ref in targets:
                target_id = target_ref if target_ref in node_ids else qname_index.get(
                    target_ref, ""
                )
                if not target_id or target_id not in node_ids:
                    # Unknown target stays on the frontier via the resolution
                    # record; we do not invent a placeholder node or edge.
                    continue
                evidence0 = resolution.evidence[0]
                edges.append(
                    make_edge(
                        source=resolution.site_id,
                        target=target_id,
                        kind=resolution.edge_kind,
                        producer=RESOLVER_PRODUCER,
                        blob_cid=evidence0.blob_cid,
                        forest_id=self.forest_id,
                        component_id=resolution.component_id or resolution.site_id,
                        span=evidence0.span,
                        resolver_status=resolution.status,
                        record={
                            "reason": resolution.reason_code.value,
                            "reason_code": resolution.reason_code.value,
                            "confidence": resolution.confidence,
                            "mechanism": resolution.mechanism,
                            "rule_id": evidence0.rule_id,
                            "resolution_id": resolution.resolution_id,
                            "resolver_version": self.resolver_version,
                        },
                    )
                )
        return tuple(
            sorted(edges, key=lambda edge: (edge.source, edge.target, edge.edge_id))
        )

    def apply_to_graph(self, graph: ProgramGraph) -> ProgramGraph:
        """Return a new graph with resolution edges appended.

        Existing nodes and AST-derived edges are preserved unchanged.
        """

        if not isinstance(graph, ProgramGraph):
            raise CallResolverError("graph must be a ProgramGraph")
        new_edges = self.resolution_edges(graph)
        if not new_edges:
            return graph
        return build_program_graph(
            forest_id=graph.forest_id,
            nodes=graph.nodes,
            edges=tuple(graph.edges) + new_edges,
            producer=RESOLVER_PRODUCER,
            unexplained_gap_count=graph.unexplained_gap_count,
            truncated=graph.truncated or self.truncated,
            truncation_reason=graph.truncation_reason or self.truncation_reason,
        )


@dataclass(frozen=True)
class ResolverCatalog:
    """Optional closed catalogs that extend pure graph evidence.

    Catalogs never invent targets that are not declared here or in the graph.
    """

    known_registrations: Mapping[str, str] = field(default_factory=dict)
    generated_sdk_methods: Mapping[str, str] = field(default_factory=dict)
    cross_package_interfaces: Mapping[str, str] = field(default_factory=dict)
    installed_packages: frozenset[str] = field(default_factory=frozenset)
    namespace_packages: frozenset[str] = field(default_factory=frozenset)
    external_packages: frozenset[str] = field(default_factory=frozenset)
    module_is_package: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "known_registrations",
            MappingProxyType(
                {
                    _text(key, "registration"): _text(value, "registration_target")
                    for key, value in dict(self.known_registrations or {}).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "generated_sdk_methods",
            MappingProxyType(
                {
                    _text(key, "sdk_method"): _text(value, "sdk_target")
                    for key, value in dict(self.generated_sdk_methods or {}).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "cross_package_interfaces",
            MappingProxyType(
                {
                    _text(key, "interface"): _text(value, "interface_target")
                    for key, value in dict(self.cross_package_interfaces or {}).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "installed_packages",
            frozenset(
                _text(item, "installed_package")
                for item in (self.installed_packages or ())
            ),
        )
        object.__setattr__(
            self,
            "namespace_packages",
            frozenset(
                _text(item, "namespace_package")
                for item in (self.namespace_packages or ())
            ),
        )
        object.__setattr__(
            self,
            "external_packages",
            frozenset(
                _text(item, "external_package")
                for item in (self.external_packages or ())
            ),
        )
        object.__setattr__(
            self,
            "module_is_package",
            frozenset(
                _text(item, "package_module")
                for item in (self.module_is_package or ())
            ),
        )


def _record_flag(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _record_str(record: Mapping[str, Any], *keys: str) -> str:
    value = _record_flag(record, *keys)
    if value is None:
        return ""
    return str(value).strip()


def _evidence_from_node(
    node: ProgramGraphNode,
    *,
    rule_id: str,
    target_record_key: str = "",
    notes: Mapping[str, Any] | None = None,
) -> ResolutionEvidence:
    return ResolutionEvidence(
        rule_id=rule_id,
        producer=node.binding.producer or RESOLVER_PRODUCER,
        blob_cid=node.binding.blob_cid,
        forest_id=node.binding.forest_id,
        span=node.binding.span,
        source_record_key=node.record_key,
        target_record_key=target_record_key,
        notes=notes or {},
    )


def _make_resolution(
    *,
    site: ProgramGraphNode,
    site_kind: str,
    status: ResolverStatus,
    reason_code: ReasonCode,
    targets: Sequence[str] = (),
    evidence: Sequence[ResolutionEvidence],
    edge_kind: ProgramEdgeKind = ProgramEdgeKind.RESOLVES_TO,
    mechanism: str = "static",
    record: Mapping[str, Any] | None = None,
) -> CallResolution:
    return CallResolution(
        site_id=site.node_id,
        site_kind=site_kind,
        status=status,
        reason_code=reason_code,
        confidence=confidence_for(status, reason_code),
        targets=tuple(targets),
        evidence=tuple(evidence),
        edge_kind=edge_kind,
        mechanism=mechanism,
        component_id=site.component_id,
        site_qualified_name=site.qualified_name,
        record=record or {},
    )


class ProgramCallResolver:
    """Resolve imports and calls over a program graph without mutating it."""

    def __init__(
        self,
        graph: ProgramGraph,
        *,
        catalog: ResolverCatalog | None = None,
        max_reexport_depth: int = DEFAULT_MAX_REEXPORT_DEPTH,
        max_resolutions: int = DEFAULT_MAX_RESOLUTIONS,
    ) -> None:
        if not isinstance(graph, ProgramGraph):
            raise CallResolverError("graph must be a ProgramGraph")
        self._graph = graph
        self._catalog = catalog or ResolverCatalog()
        if (
            isinstance(max_reexport_depth, bool)
            or not isinstance(max_reexport_depth, int)
            or max_reexport_depth < 1
        ):
            raise CallResolverBoundsError("max_reexport_depth must be a positive int")
        if (
            isinstance(max_resolutions, bool)
            or not isinstance(max_resolutions, int)
            or max_resolutions < 1
            or max_resolutions > DEFAULT_MAX_RESOLUTIONS
        ):
            raise CallResolverBoundsError("max_resolutions out of bounds")
        self._max_reexport_depth = max_reexport_depth
        self._max_resolutions = max_resolutions
        self._nodes_by_id = {node.node_id: node for node in graph.nodes}
        self._modules_by_qname: dict[str, list[ProgramGraphNode]] = defaultdict(list)
        self._symbols_by_qname: dict[str, list[ProgramGraphNode]] = defaultdict(list)
        self._symbols_by_simple: dict[str, list[ProgramGraphNode]] = defaultdict(list)
        self._exports_by_module: dict[str, list[ProgramGraphNode]] = defaultdict(list)
        self._exports_by_qname: dict[str, list[ProgramGraphNode]] = defaultdict(list)
        self._imports: list[ProgramGraphNode] = []
        self._calls: list[ProgramGraphNode] = []
        self._reexports: list[ProgramGraphNode] = []
        self._definitions_by_module: dict[str, list[ProgramGraphNode]] = defaultdict(
            list
        )
        self._alias_by_component: dict[str, dict[str, str]] = defaultdict(dict)
        self._member_index: dict[str, list[str]] = defaultdict(list)
        self._mcp_tools_by_name: dict[str, list[ProgramGraphNode]] = defaultdict(list)
        self._build_indexes()

    @property
    def graph(self) -> ProgramGraph:
        return self._graph

    @property
    def catalog(self) -> ResolverCatalog:
        return self._catalog

    def _build_indexes(self) -> None:
        for node in self._graph.nodes:
            if node.kind is ProgramNodeKind.MODULE and node.qualified_name:
                self._modules_by_qname[node.qualified_name].append(node)
            elif node.kind in {ProgramNodeKind.SYMBOL, ProgramNodeKind.DEFINITION}:
                if node.qualified_name:
                    self._symbols_by_qname[node.qualified_name].append(node)
                    simple = _simple_name(node.qualified_name)
                    if simple:
                        self._symbols_by_simple[simple].append(node)
                    module_q = _package_of_module(node.qualified_name)
                    if module_q:
                        self._definitions_by_module[module_q].append(node)
                    # Class member: owner.member when record declares owner.
                    owner = _record_str(node.record, "owner", "class_name", "parent")
                    member = _record_str(node.record, "member", "name") or simple
                    if owner and member:
                        self._member_index[f"{owner}.{member}"].append(
                            node.qualified_name
                        )
                        self._member_index[member].append(node.qualified_name)
            elif node.kind is ProgramNodeKind.IMPORT:
                self._imports.append(node)
                local = _record_str(node.record, "alias", "local_name") or _simple_name(
                    node.qualified_name
                )
                target = (
                    _record_str(node.record, "target", "module", "import_target")
                    or node.qualified_name
                )
                if local and target:
                    self._alias_by_component[node.component_id][local] = target
            elif node.kind is ProgramNodeKind.EXPORT:
                self._exports_by_module[node.component_id].append(node)
                if node.qualified_name:
                    self._exports_by_qname[node.qualified_name].append(node)
                if (
                    _record_str(node.record, "kind", "export_kind") == "re_export"
                    or _record_str(node.record, "relationship") == "re_exports"
                    or _record_flag(node.record, "reexport", "re_export")
                    or _record_str(node.record, "from_module", "source_module")
                ):
                    self._reexports.append(node)
            elif node.kind is ProgramNodeKind.CALL:
                self._calls.append(node)
            elif node.kind is ProgramNodeKind.MCP_TOOL and node.qualified_name:
                self._mcp_tools_by_name[node.qualified_name].append(node)
                simple = _simple_name(node.qualified_name)
                if simple:
                    self._mcp_tools_by_name[simple].append(node)

        # Existing import edges may also establish aliases.
        for edge in self._graph.edges:
            if edge.kind is not ProgramEdgeKind.IMPORTS:
                continue
            source = self._nodes_by_id.get(edge.source)
            target = self._nodes_by_id.get(edge.target)
            if source is None or target is None:
                continue
            if source.kind is ProgramNodeKind.MODULE and target.kind is ProgramNodeKind.IMPORT:
                local = (
                    _record_str(target.record, "alias", "local_name")
                    or _simple_name(target.qualified_name)
                )
                imported = (
                    _record_str(target.record, "target", "module", "import_target")
                    or target.qualified_name
                )
                if local and imported:
                    self._alias_by_component[source.component_id][local] = imported

    def resolve(self) -> CallResolutionResult:
        """Resolve every import, re-export, and call site conservatively."""

        resolutions: list[CallResolution] = []
        truncated = False
        truncation_reason = ""

        for node in sorted(self._imports, key=lambda item: item.node_id):
            resolutions.append(self.resolve_import(node))
            if len(resolutions) >= self._max_resolutions:
                truncated = True
                truncation_reason = "max_resolutions"
                break

        if not truncated:
            for node in sorted(self._reexports, key=lambda item: item.node_id):
                resolutions.append(self.resolve_reexport(node))
                if len(resolutions) >= self._max_resolutions:
                    truncated = True
                    truncation_reason = "max_resolutions"
                    break

        if not truncated:
            for node in sorted(self._calls, key=lambda item: item.node_id):
                resolutions.append(self.resolve_call(node))
                if len(resolutions) >= self._max_resolutions:
                    truncated = True
                    truncation_reason = "max_resolutions"
                    break

        return CallResolutionResult(
            forest_id=self._graph.forest_id,
            resolver_version=RESOLVER_VERSION,
            source_graph_id=self._graph.graph_id,
            resolutions=tuple(resolutions),
            truncated=truncated,
            truncation_reason=truncation_reason,
        )

    def resolve_import(self, node: ProgramGraphNode | str) -> CallResolution:
        """Resolve one import node to a module/symbol target or frontier."""

        site = self._site(node, ProgramNodeKind.IMPORT)
        if site.binding.resolver_status is ResolverStatus.RESOLVED_STATIC:
            target = (
                _record_str(site.record, "resolved_target", "target")
                or site.qualified_name
            )
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.ALREADY_RESOLVED,
                targets=(target,) if target else (),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:already_resolved",
                        target_record_key=target,
                    ),
                ),
                edge_kind=ProgramEdgeKind.RESOLVES_TO,
                record={"prior_status": "resolved_static"},
            )

        target = (
            _record_str(site.record, "target", "module", "import_target")
            or site.qualified_name
        )
        optional = bool(
            _record_flag(site.record, "optional", "is_optional", "optional_import")
        )
        relative_level = int(_record_flag(site.record, "relative_level") or 0)
        if target.startswith(".") or relative_level > 0:
            return self._resolve_relative_import(
                site, target=target, relative_level=relative_level, optional=optional
            )
        return self._resolve_package_import(site, target=target, optional=optional)

    def _module_for_component(self, component_id: str) -> ProgramGraphNode | None:
        for node in self._graph.nodes:
            if (
                node.kind is ProgramNodeKind.MODULE
                and node.component_id == component_id
            ):
                return node
        return None

    def _resolve_relative_import(
        self,
        site: ProgramGraphNode,
        *,
        target: str,
        relative_level: int,
        optional: bool,
    ) -> CallResolution:
        module_node = self._module_for_component(site.component_id)
        current = module_node.qualified_name if module_node else ""
        is_package = bool(
            current
            and (
                current in self._catalog.module_is_package
                or _record_flag(
                    module_node.record if module_node else {}, "is_package"
                )
            )
        )
        synthetic = target
        if relative_level > 0 and not target.startswith("."):
            synthetic = ("." * relative_level) + target
        try:
            resolved_module = resolve_relative_module(
                current, synthetic or target, is_package=is_package
            )
        except CallResolverError:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.UNRESOLVED,
                reason_code=ReasonCode.NO_TARGET,
                targets=(),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:relative_import_escape",
                        notes={"target": target, "current_module": current},
                    ),
                ),
                record={"target": target, "current_module": current},
            )

        return self._finish_module_import(
            site,
            resolved_module=resolved_module,
            reason_code=ReasonCode.RELATIVE_IMPORT,
            rule_id="rule:relative_import",
            optional=optional,
            notes={"current_module": current, "raw_target": target},
        )

    def _resolve_package_import(
        self,
        site: ProgramGraphNode,
        *,
        target: str,
        optional: bool,
    ) -> CallResolution:
        # Alias binding: import target may already be a local alias expansion.
        alias_map = self._alias_by_component.get(site.component_id, {})
        local = _record_str(site.record, "alias", "local_name") or _simple_name(
            site.qualified_name
        )
        if local and local in alias_map and alias_map[local] != target:
            # Prefer explicit record target; alias map is diagnostic.
            pass

        root = target.split(".", 1)[0] if target else ""
        if root and root in self._catalog.namespace_packages:
            # Namespace packages are ambiguous unless a concrete module exists.
            modules = self._modules_by_qname.get(target, [])
            if len(modules) == 1:
                return self._finish_module_import(
                    site,
                    resolved_module=target,
                    reason_code=ReasonCode.PACKAGE_IMPORT,
                    rule_id="rule:namespace_concrete_module",
                    optional=optional,
                    notes={"namespace": root},
                )
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.AMBIGUOUS,
                reason_code=ReasonCode.NAMESPACE_PACKAGE,
                targets=tuple(sorted({node.qualified_name for node in modules if node.qualified_name})),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:namespace_package",
                        notes={"namespace": root, "target": target},
                    ),
                ),
                record={"target": target, "namespace": root},
            )

        if root and root in self._catalog.external_packages:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.EXTERNAL,
                reason_code=ReasonCode.EXTERNAL_MODULE,
                targets=(target,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:external_package",
                        target_record_key=target,
                        notes={"package": root},
                    ),
                ),
                record={"target": target, "package": root},
            )

        if (
            root
            and self._catalog.installed_packages
            and root not in self._catalog.installed_packages
            and target not in self._modules_by_qname
            and not any(
                qname == target or qname.startswith(target + ".")
                for qname in self._modules_by_qname
            )
        ):
            status = (
                ResolverStatus.CANDIDATE if optional else ResolverStatus.EXTERNAL
            )
            reason = (
                ReasonCode.OPTIONAL_IMPORT
                if optional
                else ReasonCode.UNINSTALLED_DEPENDENCY
            )
            return _make_resolution(
                site=site,
                site_kind="import",
                status=status,
                reason_code=reason,
                targets=(target,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:uninstalled_dependency",
                        target_record_key=target,
                        notes={"package": root, "optional": optional},
                    ),
                ),
                record={"target": target, "package": root, "optional": optional},
            )

        if optional and target not in self._modules_by_qname:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.CANDIDATE,
                reason_code=ReasonCode.OPTIONAL_IMPORT,
                targets=(target,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:optional_import",
                        target_record_key=target,
                        notes={"target": target},
                    ),
                ),
                record={"target": target, "optional": True},
            )

        return self._finish_module_import(
            site,
            resolved_module=target,
            reason_code=ReasonCode.PACKAGE_IMPORT,
            rule_id="rule:package_import",
            optional=optional,
            notes={},
        )

    def _finish_module_import(
        self,
        site: ProgramGraphNode,
        *,
        resolved_module: str,
        reason_code: ReasonCode,
        rule_id: str,
        optional: bool,
        notes: Mapping[str, Any],
    ) -> CallResolution:
        modules = self._modules_by_qname.get(resolved_module, [])
        # Also allow import of symbol path package.mod.symbol
        symbols = self._symbols_by_qname.get(resolved_module, [])
        if len(modules) == 1 and not symbols:
            target_q = modules[0].qualified_name
            alias = _record_str(site.record, "alias", "local_name")
            final_reason = (
                ReasonCode.ALIAS_BINDING if alias else reason_code
            )
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=final_reason,
                targets=(target_q,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id=rule_id,
                        target_record_key=modules[0].record_key,
                        notes={**dict(notes), "resolved_module": resolved_module},
                    ),
                ),
                edge_kind=ProgramEdgeKind.RESOLVES_TO,
                record={
                    "target": resolved_module,
                    "resolved": target_q,
                    "optional": optional,
                },
            )
        if len(symbols) == 1 and not modules:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.ALIAS_BINDING
                if _record_str(site.record, "alias")
                else reason_code,
                targets=(symbols[0].qualified_name,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id=rule_id + ":symbol",
                        target_record_key=symbols[0].record_key,
                        notes={**dict(notes), "resolved_symbol": symbols[0].qualified_name},
                    ),
                ),
                record={"target": resolved_module, "optional": optional},
            )
        candidates = sorted(
            {
                *(node.qualified_name for node in modules if node.qualified_name),
                *(node.qualified_name for node in symbols if node.qualified_name),
            }
        )
        if len(candidates) > 1:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.AMBIGUOUS,
                reason_code=ReasonCode.AMBIGUOUS_CANDIDATES,
                targets=tuple(candidates),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:ambiguous_import",
                        notes={**dict(notes), "candidates": list(candidates)},
                    ),
                ),
                record={"target": resolved_module, "candidates": list(candidates)},
            )
        if len(candidates) == 1:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=reason_code,
                targets=(candidates[0],),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id=rule_id,
                        target_record_key=candidates[0],
                        notes={**dict(notes), "resolved": candidates[0]},
                    ),
                ),
                record={"target": resolved_module, "resolved": candidates[0]},
            )
        # Module not in graph: external unless optional.
        if optional:
            return _make_resolution(
                site=site,
                site_kind="import",
                status=ResolverStatus.CANDIDATE,
                reason_code=ReasonCode.OPTIONAL_IMPORT,
                targets=(resolved_module,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:optional_missing_module",
                        target_record_key=resolved_module,
                        notes=dict(notes),
                    ),
                ),
                record={"target": resolved_module, "optional": True},
            )
        return _make_resolution(
            site=site,
            site_kind="import",
            status=ResolverStatus.EXTERNAL,
            reason_code=ReasonCode.EXTERNAL_MODULE,
            targets=(resolved_module,),
            evidence=(
                _evidence_from_node(
                    site,
                    rule_id="rule:external_module",
                    target_record_key=resolved_module,
                    notes=dict(notes),
                ),
            ),
            record={"target": resolved_module},
        )

    def resolve_reexport(self, node: ProgramGraphNode | str) -> CallResolution:
        """Follow a re-export chain with loop detection."""

        site = self._site(node, ProgramNodeKind.EXPORT)
        source_module = _record_str(
            site.record, "from_module", "source_module", "module"
        )
        export_name = (
            _record_str(site.record, "export_name", "name", "local_name")
            or _simple_name(site.qualified_name)
        )
        if not source_module:
            # Not a re-export after all: treat as local export binding.
            if site.qualified_name and site.qualified_name in self._symbols_by_qname:
                symbols = self._symbols_by_qname[site.qualified_name]
                if len(symbols) == 1:
                    return _make_resolution(
                        site=site,
                        site_kind="export",
                        status=ResolverStatus.RESOLVED_STATIC,
                        reason_code=ReasonCode.ALIAS_BINDING,
                        targets=(symbols[0].qualified_name,),
                        evidence=(
                            _evidence_from_node(
                                site,
                                rule_id="rule:local_export",
                                target_record_key=symbols[0].record_key,
                            ),
                        ),
                    )
            return _make_resolution(
                site=site,
                site_kind="export",
                status=ResolverStatus.UNRESOLVED,
                reason_code=ReasonCode.NO_TARGET,
                targets=(),
                evidence=(
                    _evidence_from_node(site, rule_id="rule:export_no_source"),
                ),
            )

        seen: list[str] = []
        current_module = source_module
        current_name = export_name
        depth = 0
        while depth < self._max_reexport_depth:
            depth += 1
            chain_key = f"{current_module}:{current_name}"
            if chain_key in seen:
                return _make_resolution(
                    site=site,
                    site_kind="reexport",
                    status=ResolverStatus.AMBIGUOUS,
                    reason_code=ReasonCode.REEXPORT_LOOP,
                    targets=tuple(seen + [chain_key]),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:reexport_loop",
                            notes={"cycle": seen + [chain_key]},
                        ),
                    ),
                    record={"cycle": seen + [chain_key]},
                )
            seen.append(chain_key)

            # Prefer a concrete symbol at module.name.
            qname = (
                f"{current_module}.{current_name}"
                if current_name
                else current_module
            )
            symbols = self._symbols_by_qname.get(qname, [])
            if len(symbols) == 1:
                return _make_resolution(
                    site=site,
                    site_kind="reexport",
                    status=ResolverStatus.RESOLVED_STATIC,
                    reason_code=ReasonCode.REEXPORT,
                    targets=(symbols[0].qualified_name,),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:reexport",
                            target_record_key=symbols[0].record_key,
                            notes={"chain": list(seen)},
                        ),
                    ),
                    record={"chain": list(seen), "resolved": symbols[0].qualified_name},
                )
            if len(symbols) > 1:
                return _make_resolution(
                    site=site,
                    site_kind="reexport",
                    status=ResolverStatus.AMBIGUOUS,
                    reason_code=ReasonCode.SAME_NAME_COLLISION,
                    targets=tuple(
                        sorted({node.qualified_name for node in symbols})
                    ),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:reexport_collision",
                            notes={"qname": qname},
                        ),
                    ),
                )

            # Follow another re-export from current_module.
            next_export = self._find_reexport(current_module, current_name)
            if next_export is None:
                modules = self._modules_by_qname.get(current_module, [])
                if len(modules) == 1 and not current_name:
                    return _make_resolution(
                        site=site,
                        site_kind="reexport",
                        status=ResolverStatus.RESOLVED_STATIC,
                        reason_code=ReasonCode.REEXPORT,
                        targets=(modules[0].qualified_name,),
                        evidence=(
                            _evidence_from_node(
                                site,
                                rule_id="rule:reexport_module",
                                target_record_key=modules[0].record_key,
                                notes={"chain": list(seen)},
                            ),
                        ),
                    )
                return _make_resolution(
                    site=site,
                    site_kind="reexport",
                    status=ResolverStatus.EXTERNAL
                    if current_module not in self._modules_by_qname
                    else ResolverStatus.UNRESOLVED,
                    reason_code=(
                        ReasonCode.EXTERNAL_MODULE
                        if current_module not in self._modules_by_qname
                        else ReasonCode.NO_TARGET
                    ),
                    targets=(qname,),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:reexport_unresolved",
                            target_record_key=qname,
                            notes={"chain": list(seen)},
                        ),
                    ),
                    record={"chain": list(seen), "qname": qname},
                )
            current_module = _record_str(
                next_export.record, "from_module", "source_module", "module"
            ) or current_module
            current_name = (
                _record_str(
                    next_export.record, "export_name", "name", "local_name"
                )
                or current_name
            )

        return _make_resolution(
            site=site,
            site_kind="reexport",
            status=ResolverStatus.UNSUPPORTED,
            reason_code=ReasonCode.UNSUPPORTED_CONSTRUCT,
            targets=tuple(seen),
            evidence=(
                _evidence_from_node(
                    site,
                    rule_id="rule:reexport_depth",
                    notes={"depth": self._max_reexport_depth, "chain": list(seen)},
                ),
            ),
            record={"chain": list(seen)},
        )

    def _find_reexport(
        self, module_qname: str, export_name: str
    ) -> ProgramGraphNode | None:
        for node in self._reexports:
            module_node = self._module_for_component(node.component_id)
            owner = module_node.qualified_name if module_node else ""
            if owner != module_qname:
                # Also allow qualified_name prefix match.
                if not (
                    node.qualified_name == f"{module_qname}.{export_name}"
                    or _record_str(node.record, "owner_module") == module_qname
                ):
                    continue
            name = (
                _record_str(node.record, "export_name", "name", "local_name")
                or _simple_name(node.qualified_name)
            )
            if name == export_name or not export_name:
                return node
        return None

    def resolve_call(self, node: ProgramGraphNode | str) -> CallResolution:
        """Resolve one call site conservatively."""

        site = self._site(node, ProgramNodeKind.CALL)
        callee = (
            _record_str(site.record, "callee", "target", "name")
            or site.qualified_name
        )
        mechanism = self._detect_mechanism(site, callee)
        if mechanism in _DYNAMIC_MECHANISMS:
            return self._resolve_dynamic_call(site, callee=callee, mechanism=mechanism)

        # Generated SDK / client methods (closed catalog).
        if callee in self._catalog.generated_sdk_methods:
            target = self._catalog.generated_sdk_methods[callee]
            is_client = bool(
                _record_flag(site.record, "generated_client", "is_generated_client")
            )
            if is_client:
                return _make_resolution(
                    site=site,
                    site_kind="call",
                    status=ResolverStatus.CANDIDATE,
                    reason_code=ReasonCode.GENERATED_CLIENT,
                    targets=(target,),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:generated_client",
                            target_record_key=target,
                            notes={"callee": callee},
                        ),
                    ),
                    edge_kind=ProgramEdgeKind.CALLS,
                    mechanism="generated_client",
                    record={"callee": callee, "generated_client": True},
                )
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.GENERATED_SDK_METHOD,
                targets=(target,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:generated_sdk_method",
                        target_record_key=target,
                        notes={"callee": callee},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="generated_sdk",
                record={"callee": callee},
            )

        # Explicit cross-package interfaces.
        if callee in self._catalog.cross_package_interfaces:
            target = self._catalog.cross_package_interfaces[callee]
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.CROSS_PACKAGE_INTERFACE,
                targets=(target,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:cross_package_interface",
                        target_record_key=target,
                        notes={"callee": callee},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="cross_package_interface",
                record={"callee": callee},
            )

        # Known MCP / tool registrations (candidate: registration ≠ static body).
        if callee in self._catalog.known_registrations:
            target = self._catalog.known_registrations[callee]
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.CANDIDATE,
                reason_code=ReasonCode.KNOWN_REGISTRATION,
                targets=(target,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:known_registration",
                        target_record_key=target,
                        notes={"callee": callee},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="registration",
                record={"callee": callee, "registration": True},
            )
        if callee in self._mcp_tools_by_name:
            tools = self._mcp_tools_by_name[callee]
            targets = tuple(sorted({tool.qualified_name for tool in tools}))
            status = (
                ResolverStatus.CANDIDATE
                if len(targets) == 1
                else ResolverStatus.AMBIGUOUS
            )
            reason = (
                ReasonCode.KNOWN_REGISTRATION
                if len(targets) == 1
                else ReasonCode.AMBIGUOUS_CANDIDATES
            )
            return _make_resolution(
                site=site,
                site_kind="call",
                status=status,
                reason_code=reason,
                targets=targets,
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:mcp_tool_name",
                        notes={"callee": callee, "tools": list(targets)},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="mcp",
                record={"callee": callee},
            )

        # Import-alias expansion: root of callee mapped via component aliases.
        root = callee.split(".", 1)[0] if callee else ""
        alias_map = self._alias_by_component.get(site.component_id, {})
        if root and root in alias_map:
            expanded = alias_map[root] + callee[len(root) :]
            return self._resolve_expanded_callee(
                site, callee=expanded, via_alias=root
            )

        # Class / member calls: owner.member with unique member index hit.
        if "." in callee:
            member_hits = self._member_index.get(callee, [])
            # Also index by Class.method from symbol qnames.
            q_hits = self._symbols_by_qname.get(callee, [])
            if len(q_hits) == 1:
                return _make_resolution(
                    site=site,
                    site_kind="call",
                    status=ResolverStatus.RESOLVED_STATIC,
                    reason_code=ReasonCode.CLASS_MEMBER,
                    targets=(q_hits[0].qualified_name,),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:class_member",
                            target_record_key=q_hits[0].record_key,
                            notes={"callee": callee},
                        ),
                    ),
                    edge_kind=ProgramEdgeKind.CALLS,
                    mechanism="class_member",
                    record={"callee": callee},
                )
            unique_members = sorted(set(member_hits))
            if len(unique_members) == 1:
                return _make_resolution(
                    site=site,
                    site_kind="call",
                    status=ResolverStatus.RESOLVED_STATIC,
                    reason_code=ReasonCode.CLASS_MEMBER,
                    targets=(unique_members[0],),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:class_member_index",
                            target_record_key=unique_members[0],
                            notes={"callee": callee},
                        ),
                    ),
                    edge_kind=ProgramEdgeKind.CALLS,
                    mechanism="class_member",
                    record={"callee": callee},
                )
            if len(unique_members) > 1 or len(q_hits) > 1:
                targets = tuple(
                    sorted(
                        {
                            *unique_members,
                            *(node.qualified_name for node in q_hits),
                        }
                    )
                )
                return _make_resolution(
                    site=site,
                    site_kind="call",
                    status=ResolverStatus.AMBIGUOUS,
                    reason_code=ReasonCode.SAME_NAME_COLLISION,
                    targets=targets,
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:member_collision",
                            notes={"callee": callee, "targets": list(targets)},
                        ),
                    ),
                    edge_kind=ProgramEdgeKind.CALLS,
                    record={"callee": callee},
                )

        # Same-module unique definition.
        module_node = self._module_for_component(site.component_id)
        module_q = module_node.qualified_name if module_node else ""
        if module_q and callee:
            same_module = [
                node
                for node in self._definitions_by_module.get(module_q, [])
                if _simple_name(node.qualified_name) == _simple_name(callee)
                or node.qualified_name == callee
                or node.qualified_name == f"{module_q}.{callee}"
            ]
            # Prefer exact qname matches inside this module.
            exact = [
                node
                for node in same_module
                if node.qualified_name
                in {callee, f"{module_q}.{_simple_name(callee)}"}
            ]
            pool = exact or same_module
            if len(pool) == 1:
                return _make_resolution(
                    site=site,
                    site_kind="call",
                    status=ResolverStatus.RESOLVED_STATIC,
                    reason_code=ReasonCode.SAME_MODULE_DEFINITION,
                    targets=(pool[0].qualified_name,),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:same_module_definition",
                            target_record_key=pool[0].record_key,
                            notes={"callee": callee, "module": module_q},
                        ),
                    ),
                    edge_kind=ProgramEdgeKind.CALLS,
                    mechanism="same_module",
                    record={"callee": callee, "module": module_q},
                )

        # Global simple-name lookup — never collapse collisions.
        simple = _simple_name(callee)
        hits = self._symbols_by_simple.get(simple, []) if simple else []
        # Deduplicate by qualified name.
        by_qname = {
            node.qualified_name: node
            for node in hits
            if node.qualified_name
        }
        if len(by_qname) > 1:
            targets = tuple(sorted(by_qname))
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.AMBIGUOUS,
                reason_code=ReasonCode.SAME_NAME_COLLISION,
                targets=targets,
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:same_name_collision",
                        notes={"callee": callee, "targets": list(targets)},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                record={"callee": callee},
            )
        if len(by_qname) == 1:
            only = next(iter(by_qname.values()))
            # Single global hit is only a candidate unless it is same-module
            # (handled above) or fully qualified exact match.
            if only.qualified_name == callee:
                return _make_resolution(
                    site=site,
                    site_kind="call",
                    status=ResolverStatus.RESOLVED_STATIC,
                    reason_code=ReasonCode.SAME_MODULE_DEFINITION,
                    targets=(only.qualified_name,),
                    evidence=(
                        _evidence_from_node(
                            site,
                            rule_id="rule:exact_qname",
                            target_record_key=only.record_key,
                            notes={"callee": callee},
                        ),
                    ),
                    edge_kind=ProgramEdgeKind.CALLS,
                    record={"callee": callee},
                )
            if module_q and only.qualified_name.startswith(module_q + "."):
                reason = ReasonCode.SAME_MODULE_DEFINITION
            else:
                # One foreign definition is a candidate, not a forged direct call.
                reason = ReasonCode.AMBIGUOUS_CANDIDATES
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.CANDIDATE,
                reason_code=reason,
                targets=(only.qualified_name,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:single_global_candidate",
                        target_record_key=only.record_key,
                        notes={"callee": callee},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                record={"callee": callee},
            )

        # Import candidate recorded by the AST adapter.
        import_candidate = _record_str(site.record, "import_candidate")
        if import_candidate:
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.CANDIDATE,
                reason_code=ReasonCode.ALIAS_BINDING,
                targets=(import_candidate,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:import_candidate",
                        target_record_key=import_candidate,
                        notes={"callee": callee},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="import_candidate",
                record={"callee": callee, "import_candidate": import_candidate},
            )

        if not callee or callee == "<dynamic>":
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.UNKNOWN,
                reason_code=ReasonCode.UNSUPPORTED_CONSTRUCT,
                targets=(),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:dynamic_expression",
                        notes={"callee": callee},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="dynamic_expression",
                record={"callee": callee},
            )

        return _make_resolution(
            site=site,
            site_kind="call",
            status=ResolverStatus.UNRESOLVED,
            reason_code=ReasonCode.UNRESOLVED_NAME,
            targets=(callee,) if callee else (),
            evidence=(
                _evidence_from_node(
                    site,
                    rule_id="rule:unresolved_name",
                    notes={"callee": callee},
                ),
            ),
            edge_kind=ProgramEdgeKind.CALLS,
            record={"callee": callee},
        )

    def _resolve_expanded_callee(
        self,
        site: ProgramGraphNode,
        *,
        callee: str,
        via_alias: str,
    ) -> CallResolution:
        symbols = self._symbols_by_qname.get(callee, [])
        modules = self._modules_by_qname.get(callee, [])
        if len(symbols) == 1:
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.RESOLVED_STATIC,
                reason_code=ReasonCode.ALIAS_BINDING,
                targets=(symbols[0].qualified_name,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:alias_call",
                        target_record_key=symbols[0].record_key,
                        notes={"callee": callee, "alias": via_alias},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="alias",
                record={"callee": callee, "alias": via_alias},
            )
        if len(modules) == 1 and not symbols:
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.CANDIDATE,
                reason_code=ReasonCode.ALIAS_BINDING,
                targets=(modules[0].qualified_name,),
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:alias_module_call",
                        target_record_key=modules[0].record_key,
                        notes={"callee": callee, "alias": via_alias},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                mechanism="alias",
                record={"callee": callee, "alias": via_alias},
            )
        if len(symbols) > 1:
            targets = tuple(sorted({node.qualified_name for node in symbols}))
            return _make_resolution(
                site=site,
                site_kind="call",
                status=ResolverStatus.AMBIGUOUS,
                reason_code=ReasonCode.SAME_NAME_COLLISION,
                targets=targets,
                evidence=(
                    _evidence_from_node(
                        site,
                        rule_id="rule:alias_collision",
                        notes={"callee": callee, "targets": list(targets)},
                    ),
                ),
                edge_kind=ProgramEdgeKind.CALLS,
                record={"callee": callee},
            )
        return _make_resolution(
            site=site,
            site_kind="call",
            status=ResolverStatus.CANDIDATE,
            reason_code=ReasonCode.ALIAS_BINDING,
            targets=(callee,),
            evidence=(
                _evidence_from_node(
                    site,
                    rule_id="rule:alias_external",
                    target_record_key=callee,
                    notes={"callee": callee, "alias": via_alias},
                ),
            ),
            edge_kind=ProgramEdgeKind.CALLS,
            mechanism="alias",
            record={"callee": callee, "alias": via_alias},
        )

    def _detect_mechanism(self, site: ProgramGraphNode, callee: str) -> str:
        explicit = _record_str(
            site.record, "mechanism", "dispatch", "transport", "kind"
        ).lower()
        if explicit in _DYNAMIC_MECHANISMS:
            return explicit
        relationship = _record_str(site.record, "relationship").lower()
        if relationship in {"mutates_member", "monkey_patch"}:
            return "monkey_patch"
        if relationship in {"injects", "dependency_injection"}:
            return "dependency_injection"
        if relationship in {"callback", "registers_callback"}:
            return "callback"
        if relationship in {"dynamic_import"} or explicit == "dynamic_import":
            return "dynamic_import"
        if _record_flag(site.record, "dynamic_import"):
            return "dynamic_import"
        if _record_flag(site.record, "monkey_patch"):
            return "monkey_patch"
        if _record_flag(site.record, "dependency_injection", "injected"):
            return "dependency_injection"
        if _record_flag(site.record, "callback"):
            return "callback"

        if callee in _DYNAMIC_IMPORT_CALLEES or callee.startswith("importlib."):
            return "dynamic_import"
        if callee in _MONKEY_PATCH_CALLEES:
            return "monkey_patch"
        if callee in _SUBPROCESS_CALLEES or callee.startswith("subprocess."):
            return "subprocess"
        if callee in _HTTP_CALLEES or callee.startswith(
            ("requests.", "httpx.", "urllib.", "aiohttp.", "axios.")
        ):
            return "http"
        if callee in _RPC_CALLEES or callee.startswith(
            ("grpc.", "xmlrpc.", "jsonrpc.")
        ):
            return "rpc"
        if callee in _LIBP2P_CALLEES or "libp2p" in callee:
            return "libp2p"
        if callee in _MCP_CALLEES or callee.startswith("mcp.") or "tools/call" in callee:
            return "mcp"
        return "static"

    def _resolve_dynamic_call(
        self,
        site: ProgramGraphNode,
        *,
        callee: str,
        mechanism: str,
    ) -> CallResolution:
        reason = _DYNAMIC_REASON_BY_MECHANISM.get(
            mechanism, ReasonCode.UNSUPPORTED_CONSTRUCT
        )
        # Dynamic sites are never resolved_static.  Prefer EXTERNAL for
        # process/network/transport boundaries; AMBIGUOUS for DI/callback/
        # monkey patch; CANDIDATE for dynamic import.
        if mechanism in {"subprocess", "http", "rpc", "libp2p", "mcp"}:
            status = ResolverStatus.EXTERNAL
        elif mechanism in {"dependency_injection", "callback", "monkey_patch"}:
            status = ResolverStatus.AMBIGUOUS
        elif mechanism == "dynamic_import":
            status = ResolverStatus.CANDIDATE
        else:
            status = ResolverStatus.UNSUPPORTED

        target_hint = _record_str(
            site.record, "import_candidate", "target", "tool_name", "endpoint"
        )
        targets = (target_hint,) if target_hint else ((callee,) if callee else ())
        # MCP registration catalog may attach a candidate implementation.
        if mechanism == "mcp" and callee in self._catalog.known_registrations:
            targets = (self._catalog.known_registrations[callee],)
            status = ResolverStatus.CANDIDATE
            reason = ReasonCode.MCP

        return _make_resolution(
            site=site,
            site_kind="call",
            status=status,
            reason_code=reason,
            targets=targets,
            evidence=(
                _evidence_from_node(
                    site,
                    rule_id=f"rule:dynamic:{mechanism}",
                    target_record_key=targets[0] if targets else "",
                    notes={
                        "callee": callee,
                        "mechanism": mechanism,
                        "dynamic": True,
                    },
                ),
            ),
            edge_kind=ProgramEdgeKind.CALLS
            if mechanism not in {"dynamic_import"}
            else ProgramEdgeKind.RESOLVES_TO,
            mechanism=mechanism,
            record={
                "callee": callee,
                "mechanism": mechanism,
                "dynamic": True,
            },
        )

    def _site(
        self, node: ProgramGraphNode | str, expected: ProgramNodeKind
    ) -> ProgramGraphNode:
        if isinstance(node, ProgramGraphNode):
            site = node
        else:
            site = self._nodes_by_id.get(_text(node, "node_id"))
            if site is None:
                raise CallResolverError(f"unknown site node_id: {node!r}")
        if site.kind is not expected:
            raise CallResolverError(
                f"site {site.node_id!r} has kind {site.kind.value}, "
                f"expected {expected.value}"
            )
        return site


def resolve_program_calls(
    graph: ProgramGraph,
    *,
    catalog: ResolverCatalog | None = None,
    max_reexport_depth: int = DEFAULT_MAX_REEXPORT_DEPTH,
    max_resolutions: int = DEFAULT_MAX_RESOLUTIONS,
) -> CallResolutionResult:
    """Resolve all call/import sites in ``graph`` and return a result record."""

    return ProgramCallResolver(
        graph,
        catalog=catalog,
        max_reexport_depth=max_reexport_depth,
        max_resolutions=max_resolutions,
    ).resolve()


def make_resolution(
    *,
    site_id: str,
    site_kind: str,
    status: ResolverStatus | str,
    reason_code: ReasonCode | str,
    evidence: Sequence[ResolutionEvidence | Mapping[str, Any]],
    targets: Sequence[str] = (),
    edge_kind: ProgramEdgeKind | str = ProgramEdgeKind.RESOLVES_TO,
    mechanism: str = "static",
    component_id: str = "",
    site_qualified_name: str = "",
    record: Mapping[str, Any] | None = None,
    confidence: int | None = None,
) -> CallResolution:
    """Construct a validated resolution with deterministic confidence."""

    status_enum = _enum(status, ResolverStatus, "status")
    reason_enum = _enum(reason_code, ReasonCode, "reason_code")
    conf = (
        int(confidence)
        if confidence is not None
        else confidence_for(status_enum, reason_enum)
    )
    return CallResolution(
        site_id=site_id,
        site_kind=site_kind,
        status=status_enum,
        reason_code=reason_enum,
        confidence=conf,
        targets=tuple(targets),
        evidence=tuple(evidence),
        edge_kind=edge_kind,
        mechanism=mechanism,
        component_id=component_id,
        site_qualified_name=site_qualified_name,
        record=record or {},
    )


__all__ = [
    "DEFAULT_MAX_REEXPORT_DEPTH",
    "DEFAULT_MAX_RESOLUTIONS",
    "CallResolution",
    "CallResolutionResult",
    "CallResolverBoundsError",
    "CallResolverError",
    "ManufacturedEdgeError",
    "MissingEvidenceError",
    "PROGRAM_CALL_EVIDENCE_SCHEMA",
    "PROGRAM_CALL_RESOLUTION_RESULT_SCHEMA",
    "PROGRAM_CALL_RESOLUTION_SCHEMA",
    "PROGRAM_CALL_RESOLVER_SCHEMA",
    "ProgramCallResolver",
    "RESOLVER_PRODUCER",
    "RESOLVER_VERSION",
    "ReasonCode",
    "ResolutionEvidence",
    "ResolverCatalog",
    "confidence_for",
    "make_resolution",
    "resolve_program_calls",
    "resolve_relative_module",
]
