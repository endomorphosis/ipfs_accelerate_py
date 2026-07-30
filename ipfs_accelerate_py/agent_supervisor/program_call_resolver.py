"""Conservative, bounded program call resolver.

Resolves call sites against a snapshot-bound :class:`ProgramGraph` and returns
one of ``resolved``, ``ambiguous``, ``dynamic``, ``external``, or
``unsupported`` together with an explicit unknown frontier.  Same-name and
vector nominations never authorize a direct edge on their own.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .program_graph import (
    Completeness,
    ProgramAuthority,
    ProgramEdge,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphError,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
    ProgramProvenance,
    ProgramTrust,
)

PROGRAM_CALL_RESOLVER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-call-resolver@1"
)
CALL_RESOLUTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/call-resolution@1"
)
PROGRAM_CALL_RESOLVER_VERSION = "program-call-resolver@1"

DEFAULT_MAX_CANDIDATES = 64
DEFAULT_MAX_FRONTIER = 256
DEFAULT_MAX_REASON_BYTES = 4_096

_DYNAMIC_MARKERS = frozenset(
    {
        "<dynamic>",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "eval",
        "exec",
        "__import__",
        "importlib.import_module",
        "import_module",
    }
)
_DYNAMIC_PREFIXES = (
    "getattr(",
    "globals()[",
    "locals()[",
    "eval(",
    "exec(",
)
_EXTERNAL_PREFIXES = (
    "subprocess.",
    "os.system",
    "requests.",
    "httpx.",
    "urllib.",
    "aiohttp.",
    "grpc.",
    "socket.",
)
_BUILTIN_NAMES = frozenset(
    {
        "print",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "open",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "super",
        "type",
        "object",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
        "AttributeError",
        "NotImplementedError",
    }
)


class CallResolverError(ProgramGraphError):
    """Raised when resolver input is malformed."""


class CallResolutionStatus(str, Enum):
    """Closed resolution vocabulary (capability contract)."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    DYNAMIC = "dynamic"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise CallResolverError(f"{name} must be a string")
    if text != text.strip() or "\x00" in text:
        raise CallResolverError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not text:
        raise CallResolverError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_REASON_BYTES:
        raise CallResolverError(f"{name} exceeds its byte bound")
    return text


def _string_tuple(
    value: Any, name: str, *, limit: int = DEFAULT_MAX_FRONTIER
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise CallResolverError(f"{name} must be a sequence of strings")
    if len(value) > limit:
        raise CallResolverError(f"{name} exceeds its item bound")
    return tuple(
        sorted(
            {
                _text(item, name, required=False)
                for item in value
                if str(item).strip()
            }
        )
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    return str(value)


def _callee_leaf(reference: str) -> str:
    text = str(reference or "").strip()
    if not text:
        return ""
    # Owner->callee form from ASTBlobRecord.calls
    if "->" in text:
        text = text.split("->", 1)[1]
    # Drop call-argument suffixes if present.
    text = text.split("(", 1)[0].strip()
    return text


def _simple_name(reference: str) -> str:
    leaf = _callee_leaf(reference)
    if not leaf:
        return ""
    return leaf.rsplit(".", 1)[-1]


def _looks_dynamic(reference: str) -> bool:
    leaf = _callee_leaf(reference)
    if not leaf:
        return True
    lower = leaf.lower()
    if leaf in _DYNAMIC_MARKERS or lower in _DYNAMIC_MARKERS:
        return True
    if any(lower.startswith(prefix) for prefix in _DYNAMIC_PREFIXES):
        return True
    if leaf == "<dynamic>" or ".__" in leaf and leaf.endswith("__"):
        # dunder attribute access is still statically named; only unknown
        # markers and reflection helpers count as dynamic.
        pass
    if re.search(r"\[.*\]", leaf) or leaf.endswith("]"):
        return True
    return False


def _looks_external(reference: str) -> bool:
    leaf = _callee_leaf(reference)
    if not leaf:
        return False
    if leaf in _BUILTIN_NAMES or _simple_name(leaf) in _BUILTIN_NAMES:
        return True
    return any(leaf.startswith(prefix) for prefix in _EXTERNAL_PREFIXES)


@dataclass(frozen=True)
class CallSite:
    """A compact call observation used by the rich resolver API."""

    caller_id: str
    callee_reference: str
    path: str = ""
    language: str = "python"
    call_form: str = "call"
    awaited: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "caller_id", _text(self.caller_id, "caller_id", required=False)
        )
        object.__setattr__(
            self,
            "callee_reference",
            _text(self.callee_reference, "callee_reference"),
        )
        object.__setattr__(
            self, "path", _text(self.path, "path", required=False)
        )
        object.__setattr__(
            self,
            "language",
            _text(self.language or "python", "language", required=False)
            or "python",
        )
        object.__setattr__(
            self,
            "call_form",
            _text(self.call_form or "call", "call_form", required=False)
            or "call",
        )
        if not isinstance(self.awaited, bool):
            raise CallResolverError("awaited must be a boolean")
        attrs = self.attributes or {}
        if not isinstance(attrs, Mapping):
            raise CallResolverError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType({str(key): _plain(value) for key, value in attrs.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller_id": self.caller_id,
            "callee_reference": self.callee_reference,
            "path": self.path,
            "language": self.language,
            "call_form": self.call_form,
            "awaited": self.awaited,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallSite":
        return cls(
            caller_id=str(payload.get("caller_id") or ""),
            callee_reference=str(
                payload.get("callee_reference")
                or payload.get("receiver_reference")
                or ""
            ),
            path=str(payload.get("path") or ""),
            language=str(payload.get("language") or "python"),
            call_form=str(payload.get("call_form") or "call"),
            awaited=bool(payload.get("awaited", False)),
            attributes=payload.get("attributes") or {},
        )


@dataclass(frozen=True)
class CallResolution:
    """One conservative resolution with an explicit incomplete frontier."""

    status: CallResolutionStatus
    call_site: CallSite
    graph_id: str
    target_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    local_scope_complete: bool = False
    route_closed: bool = False
    confidence: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CALL_RESOLUTION_SCHEMA
    resolution_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, CallResolutionStatus):
            status = self.status
        else:
            try:
                status = CallResolutionStatus(str(self.status))
            except ValueError as exc:
                raise CallResolverError(
                    f"invalid resolution status: {self.status!r}"
                ) from exc
        object.__setattr__(self, "status", status)
        if not isinstance(self.call_site, CallSite):
            if isinstance(self.call_site, Mapping):
                object.__setattr__(
                    self, "call_site", CallSite.from_dict(self.call_site)
                )
            else:
                raise CallResolverError("call_site must be CallSite")
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id")
        )
        object.__setattr__(
            self,
            "target_ids",
            _string_tuple(self.target_ids, "target_ids", limit=DEFAULT_MAX_CANDIDATES),
        )
        object.__setattr__(
            self,
            "candidate_ids",
            _string_tuple(
                self.candidate_ids, "candidate_ids", limit=DEFAULT_MAX_CANDIDATES
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(self.reason_codes, "reason_codes"),
        )
        object.__setattr__(
            self,
            "frontier_refs",
            _string_tuple(self.frontier_refs, "frontier_refs"),
        )
        object.__setattr__(
            self,
            "exclusion_refs",
            _string_tuple(self.exclusion_refs, "exclusion_refs"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _string_tuple(self.evidence_ids, "evidence_ids"),
        )
        if not isinstance(self.local_scope_complete, bool):
            raise CallResolverError("local_scope_complete must be a boolean")
        if not isinstance(self.route_closed, bool):
            raise CallResolverError("route_closed must be a boolean")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, int):
            raise CallResolverError("confidence must be an integer in 0..100")
        if self.confidence < 0 or self.confidence > 100:
            raise CallResolverError("confidence must be an integer in 0..100")
        attrs = self.attributes or {}
        if not isinstance(attrs, Mapping):
            raise CallResolverError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType({str(key): _plain(value) for key, value in attrs.items()}),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or CALL_RESOLUTION_SCHEMA, "schema"),
        )
        if self.schema != CALL_RESOLUTION_SCHEMA:
            raise CallResolverError(
                f"unsupported call resolution schema: {self.schema}"
            )

        # Status-specific invariants.
        if self.status is CallResolutionStatus.RESOLVED:
            if len(self.target_ids) != 1:
                raise CallResolverError(
                    "resolved status requires exactly one target"
                )
            if not self.route_closed:
                raise CallResolverError("resolved status requires a closed route")
        if self.status is CallResolutionStatus.AMBIGUOUS:
            if len(self.candidate_ids) < 2 and len(self.target_ids) < 2:
                raise CallResolverError(
                    "ambiguous status requires multiple candidates"
                )
            object.__setattr__(self, "route_closed", False)
        if self.status in {
            CallResolutionStatus.DYNAMIC,
            CallResolutionStatus.EXTERNAL,
            CallResolutionStatus.UNSUPPORTED,
        }:
            object.__setattr__(self, "route_closed", False)
            if not self.frontier_refs:
                object.__setattr__(
                    self,
                    "frontier_refs",
                    (f"status:{self.status.value}",),
                )

        claimed = str(self.resolution_id or "").strip()
        object.__setattr__(self, "resolution_id", "")
        actual = _identity("call-resolution", self._identity_payload())
        if claimed and claimed != actual:
            raise CallResolverError(
                "call resolution identity does not match payload"
            )
        object.__setattr__(self, "resolution_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status.value,
            "call_site": self.call_site.to_dict(),
            "graph_id": self.graph_id,
            "target_ids": list(self.target_ids),
            "candidate_ids": list(self.candidate_ids),
            "reason_codes": list(self.reason_codes),
            "frontier_refs": list(self.frontier_refs),
            "exclusion_refs": list(self.exclusion_refs),
            "evidence_ids": list(self.evidence_ids),
            "local_scope_complete": self.local_scope_complete,
            "route_closed": self.route_closed,
            "confidence": self.confidence,
            "attributes": dict(self.attributes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "resolution_id": self.resolution_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallResolution":
        return cls(
            status=payload.get("status", CallResolutionStatus.UNSUPPORTED),
            call_site=CallSite.from_dict(payload.get("call_site") or {}),
            graph_id=str(payload.get("graph_id") or ""),
            target_ids=tuple(payload.get("target_ids") or ()),
            candidate_ids=tuple(payload.get("candidate_ids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            frontier_refs=tuple(payload.get("frontier_refs") or ()),
            exclusion_refs=tuple(payload.get("exclusion_refs") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            local_scope_complete=bool(payload.get("local_scope_complete", False)),
            route_closed=bool(payload.get("route_closed", False)),
            confidence=int(payload.get("confidence") or 0),
            attributes=payload.get("attributes") or {},
            schema=str(payload.get("schema") or CALL_RESOLUTION_SCHEMA),
            resolution_id=str(payload.get("resolution_id") or ""),
        )


class ProgramCallResolver:
    """Conservative call resolver over a snapshot-bound program graph.

    Satisfies the ``vfs.program_call_resolver`` capability probe and the
    narrow broken-trace protocol via :meth:`resolve_call`.
    """

    def __init__(
        self,
        graph: ProgramGraph | ProgramGraphSnapshot | Mapping[str, Any] | None = None,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        if max_candidates < 1 or max_candidates > DEFAULT_MAX_CANDIDATES:
            raise CallResolverError("max_candidates is outside the hard bound")
        self._max_candidates = int(max_candidates)
        self._graph: ProgramGraph | None = None
        if graph is not None:
            self.bind(graph)

    @property
    def graph(self) -> ProgramGraph | None:
        return self._graph

    @property
    def version(self) -> str:
        return PROGRAM_CALL_RESOLVER_VERSION

    def bind(
        self, graph: ProgramGraph | ProgramGraphSnapshot | Mapping[str, Any]
    ) -> "ProgramCallResolver":
        if isinstance(graph, ProgramGraph):
            self._graph = graph
        elif isinstance(graph, ProgramGraphSnapshot):
            self._graph = ProgramGraph(graph)
        elif isinstance(graph, Mapping):
            self._graph = ProgramGraph.from_dict(graph)
        else:
            raise CallResolverError("graph must be a ProgramGraph or snapshot")
        return self

    def resolve(self, call_site: CallSite | Mapping[str, Any]) -> CallResolution:
        """Resolve one call site against the bound graph."""

        if self._graph is None:
            site = (
                call_site
                if isinstance(call_site, CallSite)
                else CallSite.from_dict(call_site)
            )
            return CallResolution(
                status=CallResolutionStatus.UNSUPPORTED,
                call_site=site,
                graph_id="unbound",
                reason_codes=("graph_unbound",),
                frontier_refs=("graph_unbound",),
                local_scope_complete=False,
                route_closed=False,
                confidence=0,
            )
        site = (
            call_site
            if isinstance(call_site, CallSite)
            else CallSite.from_dict(call_site)
        )
        return self._resolve_against(self._graph, site)

    def resolve_many(
        self, call_sites: Iterable[CallSite | Mapping[str, Any]]
    ) -> tuple[CallResolution, ...]:
        return tuple(self.resolve(item) for item in call_sites)

    def resolve_reference(
        self,
        callee_reference: str,
        *,
        caller_id: str = "",
        path: str = "",
    ) -> CallResolution:
        return self.resolve(
            CallSite(
                caller_id=caller_id,
                callee_reference=callee_reference,
                path=path,
            )
        )

    def resolve_call(self, call_site: Any, graph: Any = None) -> Any:
        """Protocol-compatible entry used by BrokenContractTraceBuilder.

        Accepts either the rich :class:`CallSite` or a broken-trace
        :class:`BrokenCallSite`, plus optional :class:`GraphEvidence`.
        """

        if graph is not None:
            bound = self._graph_from_evidence(graph)
            if bound is not None:
                previous = self._graph
                try:
                    self._graph = bound
                    return self._resolve_protocol(call_site)
                finally:
                    self._graph = previous
        return self._resolve_protocol(call_site)

    def _resolve_protocol(self, call_site: Any) -> Any:
        site = self._coerce_call_site(call_site)
        resolution = self.resolve(site)
        return self._to_resolver_evidence(resolution, call_site)

    def _graph_from_evidence(self, graph: Any) -> ProgramGraph | None:
        if isinstance(graph, ProgramGraph):
            return graph
        if isinstance(graph, ProgramGraphSnapshot):
            return ProgramGraph(graph)
        if isinstance(graph, Mapping) and (
            "snapshot" in graph or "nodes" in graph
        ):
            try:
                return ProgramGraph.from_dict(graph)
            except Exception:
                return self._graph
        # GraphEvidence only carries coverage; keep the currently bound graph.
        return self._graph

    def _coerce_call_site(self, call_site: Any) -> CallSite:
        if isinstance(call_site, CallSite):
            return call_site
        if isinstance(call_site, Mapping):
            return CallSite.from_dict(call_site)
        # BrokenCallSite-like object
        receiver = getattr(call_site, "receiver_reference", None)
        caller = getattr(call_site, "caller_symbol_id", "")
        span = getattr(call_site, "caller_span", None)
        path = getattr(span, "path", "") if span is not None else ""
        language = getattr(call_site, "language", "python")
        call_form = getattr(call_site, "call_form", "call")
        awaited = bool(getattr(call_site, "awaited", False))
        if receiver is None:
            raise CallResolverError("call_site is not recognized")
        return CallSite(
            caller_id=str(caller or ""),
            callee_reference=str(receiver),
            path=str(path or ""),
            language=str(language or "python"),
            call_form=str(call_form or "call"),
            awaited=awaited,
        )

    def _to_resolver_evidence(
        self, resolution: CallResolution, original: Any
    ) -> Any:
        """Project into broken-trace ResolverEvidence when available."""

        try:
            from .analysis.broken_contract_trace import ResolverEvidence
            from .analysis.contract_repair_contracts import (
                EvidenceReference,
                SourceSpan,
                TraceDisposition,
            )
        except Exception:
            return resolution.to_dict()

        disposition_map = {
            CallResolutionStatus.RESOLVED: TraceDisposition.RESOLVED_MISMATCH
            if resolution.attributes.get("mismatch")
            else TraceDisposition.MISSING_LOCAL
            if not resolution.target_ids
            else TraceDisposition.RESOLVED_MISMATCH,
            CallResolutionStatus.AMBIGUOUS: TraceDisposition.AMBIGUOUS,
            CallResolutionStatus.DYNAMIC: TraceDisposition.DYNAMIC,
            CallResolutionStatus.EXTERNAL: TraceDisposition.EXTERNAL,
            CallResolutionStatus.UNSUPPORTED: TraceDisposition.UNSUPPORTED,
        }
        # For a clean resolved target use RESOLVED_MISMATCH only when the
        # broken-trace pipeline is classifying a broken call; otherwise map to
        # a closed disposition that preserves target span when present.
        if resolution.status is CallResolutionStatus.RESOLVED:
            disposition = TraceDisposition.RESOLVED_MISMATCH
        else:
            disposition = disposition_map[resolution.status]

        target_span = None
        target_symbol_id = ""
        if resolution.target_ids and self._graph is not None:
            node = self._graph.node(resolution.target_ids[0])
            if node is not None:
                target_symbol_id = node.qualified_name or node.name
                if node.path:
                    start = int(node.span.get("line_start") or 0)
                    end = int(node.span.get("line_end") or start)
                    # Broken-trace SourceSpan uses byte offsets; synthesize a
                    # stable path-bound span from line numbers when available.
                    try:
                        target_span = SourceSpan(
                            path=node.path,
                            start=max(0, start),
                            end=max(max(0, start), end),
                            artifact_id=node.blob_identity or node.content_id,
                        )
                    except Exception:
                        target_span = None

        evidence_refs = (
            EvidenceReference(
                "call_resolution",
                resolution.resolution_id,
                resolution.status.value,
                "ipfs_accelerate_py.agent_supervisor.program_call_resolver",
            ),
        )
        return ResolverEvidence(
            disposition=disposition.value,
            target_span=target_span,
            target_symbol_id=target_symbol_id,
            local_scope_complete=resolution.local_scope_complete,
            route_closed=resolution.route_closed,
            evidence_refs=evidence_refs,
            frontier_refs=resolution.frontier_refs,
            exclusion_refs=resolution.exclusion_refs,
        )

    def _resolve_against(
        self, graph: ProgramGraph, site: CallSite
    ) -> CallResolution:
        reference = site.callee_reference
        leaf = _callee_leaf(reference)
        simple = _simple_name(reference)
        frontier = list(graph.frontier_refs)
        exclusions = list(graph.exclusion_refs)
        evidence: list[str] = [graph.graph_id]

        # Existing authoritative CALLS edges from the caller win when unique.
        if site.caller_id:
            edge_targets = self._authoritative_call_targets(graph, site.caller_id, leaf)
            if len(edge_targets) == 1:
                return CallResolution(
                    status=CallResolutionStatus.RESOLVED,
                    call_site=site,
                    graph_id=graph.graph_id,
                    target_ids=(edge_targets[0],),
                    reason_codes=("authoritative_call_edge",),
                    frontier_refs=tuple(frontier),
                    exclusion_refs=tuple(exclusions),
                    evidence_ids=tuple(sorted(set(evidence + [edge_targets[0]]))),
                    local_scope_complete=True,
                    route_closed=True,
                    confidence=100,
                )
            if len(edge_targets) > 1:
                return CallResolution(
                    status=CallResolutionStatus.AMBIGUOUS,
                    call_site=site,
                    graph_id=graph.graph_id,
                    candidate_ids=tuple(edge_targets[: self._max_candidates]),
                    reason_codes=("multiple_authoritative_call_edges",),
                    frontier_refs=tuple(
                        sorted(set(frontier + ["multiple_call_edges"]))
                    ),
                    exclusion_refs=tuple(exclusions),
                    evidence_ids=tuple(sorted(set(evidence + list(edge_targets)))),
                    local_scope_complete=True,
                    route_closed=False,
                    confidence=40,
                )

        if _looks_dynamic(reference):
            return CallResolution(
                status=CallResolutionStatus.DYNAMIC,
                call_site=site,
                graph_id=graph.graph_id,
                reason_codes=("dynamic_dispatch",),
                frontier_refs=tuple(
                    sorted(set(frontier + [f"dynamic:{leaf or reference}"]))
                ),
                exclusion_refs=tuple(exclusions),
                evidence_ids=tuple(evidence),
                local_scope_complete=False,
                route_closed=False,
                confidence=0,
            )

        if _looks_external(reference):
            return CallResolution(
                status=CallResolutionStatus.EXTERNAL,
                call_site=site,
                graph_id=graph.graph_id,
                reason_codes=("external_or_builtin",),
                frontier_refs=tuple(
                    sorted(set(frontier + [f"external:{leaf or reference}"]))
                ),
                exclusion_refs=tuple(exclusions),
                evidence_ids=tuple(evidence),
                local_scope_complete=True,
                route_closed=False,
                confidence=20,
            )

        candidates = self._collect_candidates(graph, leaf, simple, site.path)
        if not candidates:
            # Import alias / re-export trail.
            aliased = self._follow_aliases(graph, leaf, simple)
            if len(aliased) == 1:
                return CallResolution(
                    status=CallResolutionStatus.RESOLVED,
                    call_site=site,
                    graph_id=graph.graph_id,
                    target_ids=(aliased[0],),
                    reason_codes=("alias_or_reexport",),
                    frontier_refs=tuple(frontier),
                    exclusion_refs=tuple(exclusions),
                    evidence_ids=tuple(sorted(set(evidence + [aliased[0]]))),
                    local_scope_complete=True,
                    route_closed=True,
                    confidence=90,
                )
            if len(aliased) > 1:
                return CallResolution(
                    status=CallResolutionStatus.AMBIGUOUS,
                    call_site=site,
                    graph_id=graph.graph_id,
                    candidate_ids=tuple(aliased[: self._max_candidates]),
                    reason_codes=("ambiguous_alias",),
                    frontier_refs=tuple(
                        sorted(set(frontier + ["ambiguous_alias"]))
                    ),
                    exclusion_refs=tuple(exclusions),
                    evidence_ids=tuple(sorted(set(evidence + list(aliased)))),
                    local_scope_complete=True,
                    route_closed=False,
                    confidence=35,
                )
            # Nominated-only edges never close the route.
            nominated = self._nominated_targets(graph, leaf, simple)
            if nominated:
                return CallResolution(
                    status=CallResolutionStatus.UNSUPPORTED,
                    call_site=site,
                    graph_id=graph.graph_id,
                    candidate_ids=tuple(nominated[: self._max_candidates]),
                    reason_codes=("nominated_only_no_authority",),
                    frontier_refs=tuple(
                        sorted(
                            set(
                                frontier
                                + ["nominated_only", f"unresolved:{leaf or simple}"]
                            )
                        )
                    ),
                    exclusion_refs=tuple(exclusions),
                    evidence_ids=tuple(sorted(set(evidence + list(nominated)))),
                    local_scope_complete=False,
                    route_closed=False,
                    confidence=10,
                    attributes={"nominated": True},
                )
            return CallResolution(
                status=CallResolutionStatus.UNSUPPORTED,
                call_site=site,
                graph_id=graph.graph_id,
                reason_codes=("no_target",),
                frontier_refs=tuple(
                    sorted(set(frontier + [f"unresolved:{leaf or simple or reference}"]))
                ),
                exclusion_refs=tuple(exclusions),
                evidence_ids=tuple(evidence),
                local_scope_complete=bool(site.path),
                route_closed=False,
                confidence=0,
            )

        if len(candidates) == 1:
            return CallResolution(
                status=CallResolutionStatus.RESOLVED,
                call_site=site,
                graph_id=graph.graph_id,
                target_ids=(candidates[0],),
                reason_codes=("unique_symbol_match",),
                frontier_refs=tuple(frontier),
                exclusion_refs=tuple(exclusions),
                evidence_ids=tuple(sorted(set(evidence + [candidates[0]]))),
                local_scope_complete=True,
                route_closed=True,
                confidence=95,
            )

        # Prefer same-path / same-module candidates when multiple match.
        if site.path:
            same_path = [
                node_id
                for node_id in candidates
                if (node := graph.node(node_id)) is not None
                and node.path == site.path
            ]
            if len(same_path) == 1:
                return CallResolution(
                    status=CallResolutionStatus.RESOLVED,
                    call_site=site,
                    graph_id=graph.graph_id,
                    target_ids=(same_path[0],),
                    reason_codes=("unique_same_path_match",),
                    frontier_refs=tuple(frontier),
                    exclusion_refs=tuple(exclusions),
                    evidence_ids=tuple(sorted(set(evidence + [same_path[0]]))),
                    local_scope_complete=True,
                    route_closed=True,
                    confidence=92,
                )

        return CallResolution(
            status=CallResolutionStatus.AMBIGUOUS,
            call_site=site,
            graph_id=graph.graph_id,
            candidate_ids=tuple(candidates[: self._max_candidates]),
            reason_codes=("multiple_symbol_matches",),
            frontier_refs=tuple(
                sorted(set(frontier + ["ambiguous_symbol", f"name:{simple}"]))
            ),
            exclusion_refs=tuple(exclusions),
            evidence_ids=tuple(sorted(set(evidence + list(candidates)))),
            local_scope_complete=True,
            route_closed=False,
            confidence=30,
        )

    def _authoritative_call_targets(
        self, graph: ProgramGraph, caller_id: str, leaf: str
    ) -> list[str]:
        targets: list[str] = []
        for edge in graph.edges_from(caller_id):
            if edge.kind is not ProgramEdgeKind.CALLS:
                continue
            if not edge.authoritative:
                continue
            target = graph.node(edge.target)
            if target is None:
                continue
            if leaf and leaf not in {
                target.name,
                target.qualified_name,
                _simple_name(target.qualified_name),
            }:
                # Allow owner-qualified leaf match via attributes.
                callee_attr = str(edge.attributes.get("callee") or "")
                if callee_attr and leaf not in {
                    callee_attr,
                    _callee_leaf(callee_attr),
                    _simple_name(callee_attr),
                }:
                    continue
            targets.append(edge.target)
        return sorted(set(targets))

    def _collect_candidates(
        self,
        graph: ProgramGraph,
        leaf: str,
        simple: str,
        path: str,
    ) -> list[str]:
        if not leaf and not simple:
            return []
        candidates: list[str] = []
        callable_kinds = {
            ProgramNodeKind.FUNCTION,
            ProgramNodeKind.METHOD,
            ProgramNodeKind.CLASS,
            ProgramNodeKind.CONSTRUCTOR,
            ProgramNodeKind.FACTORY,
            ProgramNodeKind.BUILDER,
            ProgramNodeKind.SYMBOL,
            ProgramNodeKind.CALLBACK,
        }
        for node in graph.nodes:
            if node.kind not in callable_kinds:
                continue
            if not node.authoritative:
                continue
            names = {
                node.name,
                node.qualified_name,
                _simple_name(node.qualified_name),
            }
            if leaf and leaf in names:
                candidates.append(node.node_id)
                continue
            if simple and simple in names:
                candidates.append(node.node_id)
                continue
            # Attribute form: receiver.method where method is the leaf suffix.
            if leaf and "." in leaf:
                suffix = leaf.rsplit(".", 1)[-1]
                if suffix == node.name or suffix == _simple_name(node.qualified_name):
                    candidates.append(node.node_id)
        return sorted(set(candidates))

    def _follow_aliases(
        self, graph: ProgramGraph, leaf: str, simple: str
    ) -> list[str]:
        names = {leaf, simple} - {""}
        if not names:
            return []
        targets: list[str] = []
        for edge in graph.edges:
            if edge.kind not in {
                ProgramEdgeKind.ALIASES,
                ProgramEdgeKind.RE_EXPORTS,
                ProgramEdgeKind.IMPORTS,
                ProgramEdgeKind.EXPORTS,
            }:
                continue
            if not edge.authoritative:
                continue
            source = graph.node(edge.source)
            target = graph.node(edge.target)
            if source is None or target is None:
                continue
            source_names = {
                source.name,
                source.qualified_name,
                _simple_name(source.qualified_name),
                str(edge.attributes.get("alias") or ""),
                str(edge.attributes.get("imported_name") or ""),
            }
            if names & source_names:
                if target.kind in {
                    ProgramNodeKind.FUNCTION,
                    ProgramNodeKind.METHOD,
                    ProgramNodeKind.CLASS,
                    ProgramNodeKind.SYMBOL,
                    ProgramNodeKind.FACTORY,
                    ProgramNodeKind.CONSTRUCTOR,
                }:
                    targets.append(target.node_id)
        return sorted(set(targets))

    def _nominated_targets(
        self, graph: ProgramGraph, leaf: str, simple: str
    ) -> list[str]:
        names = {leaf, simple} - {""}
        if not names:
            return []
        targets: list[str] = []
        for edge in graph.edges:
            if edge.authoritative:
                continue
            if edge.kind not in {
                ProgramEdgeKind.RELATED_TO,
                ProgramEdgeKind.CALLS,
                ProgramEdgeKind.DEPENDS_ON,
            }:
                continue
            source = graph.node(edge.source)
            target = graph.node(edge.target)
            for node in (source, target):
                if node is None:
                    continue
                node_names = {
                    node.name,
                    node.qualified_name,
                    _simple_name(node.qualified_name),
                }
                if names & node_names and target is not None:
                    targets.append(target.node_id)
        return sorted(set(targets))


__all__ = [
    "CALL_RESOLUTION_SCHEMA",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_FRONTIER",
    "PROGRAM_CALL_RESOLVER_SCHEMA",
    "PROGRAM_CALL_RESOLVER_VERSION",
    "CallResolution",
    "CallResolutionStatus",
    "CallResolverError",
    "CallSite",
    "ProgramCallResolver",
]
