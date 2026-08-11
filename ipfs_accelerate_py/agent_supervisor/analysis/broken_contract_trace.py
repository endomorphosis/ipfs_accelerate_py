"""Conservative, bounded classification of one broken program call.

This adapter is intentionally narrower than candidate retrieval.  It turns a
resolver/graph observation into a :class:`BrokenContractTrace` without using a
name match, a filename, or vector similarity as call-resolution evidence.
The returned :class:`BrokenTraceAnalysis` keeps the compact call facts required
by the later sender-contract stage; the persisted contract itself contains
only content-addressed references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from .contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    ContractRepairBoundsError,
    ContractRepairError,
    EvidenceReference,
    SourceSpan,
    TraceDisposition,
)
from .program_ast_adapters import ProgramEvidenceFact


TRACE_ANALYSIS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/broken-contract-trace-analysis@1"
)
MAX_CALL_ARGUMENTS: Final[int] = 256
MAX_RESULT_USES: Final[int] = 128
MAX_CONTEXT_VALUES: Final[int] = 128
MAX_FRONTIER_REFS: Final[int] = 256

_RESOLUTION_VALUES = frozenset(item.value for item in TraceDisposition)
_IDENTITY_KINDS = frozenset({
    "history_lineage", "content_identity", "structural_identity",
    "reviewed_rename", "reviewed_move",
})
_ADAPTER_KINDS = frozenset({"adapter_mapping", "reviewed_adapter_mapping"})
_DYNAMIC_KINDS = frozenset({"dynamic_dispatch", "reflection", "ffi", "monkey_patch"})


class BrokenTraceError(ContractRepairError):
    """Base error for an invalid trace-analysis input."""


class BrokenTraceEvidenceError(BrokenTraceError):
    """Evidence was insufficient, malformed, stale, or non-authoritative."""


def _compact_text(value: Any, name: str, *, required: bool = False, limit: int = 4096) -> str:
    if not isinstance(value, str):
        raise BrokenTraceError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise BrokenTraceError(f"{name} is required")
    if len(value.encode("utf-8")) > limit:
        raise ContractRepairBoundsError(f"{name} exceeds its byte bound")
    return value


def _identifier(value: Any, name: str) -> str:
    result = _compact_text(value, name, required=True)
    if any(character.isspace() for character in result):
        raise BrokenTraceError(f"{name} must be a compact identifier")
    return result


def _bounded_strings(value: Sequence[str] | None, name: str, *, limit: int = MAX_CONTEXT_VALUES) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BrokenTraceError(f"{name} must be a sequence of strings")
    if len(value) > limit:
        raise ContractRepairBoundsError(f"{name} exceeds its item bound")
    return tuple(sorted({_identifier(item, name) for item in value}))


def _evidence_ref(value: EvidenceReference | ProgramEvidenceFact | Mapping[str, Any]) -> EvidenceReference:
    if isinstance(value, EvidenceReference):
        return value
    if isinstance(value, ProgramEvidenceFact):
        return EvidenceReference(
            "program_evidence_fact", value.fact_id, value.kind,
            "ipfs_accelerate_py.agent_supervisor.program_ast_adapters",
        )
    if isinstance(value, Mapping):
        allowed = {"kind", "artifact_id", "locator", "producer_id", "schema", "contract_version", "content_id", "cid"}
        if set(value).difference(allowed):
            raise BrokenTraceEvidenceError("evidence reference contains unsupported fields")
        return EvidenceReference.from_dict(value) if "schema" in value else EvidenceReference(**dict(value))
    raise BrokenTraceEvidenceError("evidence must be an evidence reference or program evidence fact")


def _evidence_refs(values: Sequence[EvidenceReference | ProgramEvidenceFact | Mapping[str, Any]], name: str, *, required: bool = False) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise BrokenTraceError(f"{name} must be a sequence")
    if len(values) > MAX_FRONTIER_REFS:
        raise ContractRepairBoundsError(f"{name} exceeds its item bound")
    result = tuple(sorted({_evidence_ref(value) for value in values}, key=lambda item: item.content_id))
    if required and not result:
        raise BrokenTraceEvidenceError(f"{name} must contain bounded evidence")
    return result


@dataclass(frozen=True)
class CallArgumentFact:
    """One actual argument as observed at the call site, never source text."""

    position: int
    name: str = ""
    type_ref: str = ""
    value_range: str = ""
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 0:
            raise BrokenTraceError("argument position must be a non-negative integer")
        object.__setattr__(self, "name", _compact_text(self.name, "argument name"))
        object.__setattr__(self, "type_ref", _compact_text(self.type_ref, "argument type_ref"))
        object.__setattr__(self, "value_range", _compact_text(self.value_range, "argument value_range"))
        object.__setattr__(self, "evidence_id", _compact_text(self.evidence_id, "argument evidence_id"))


@dataclass(frozen=True)
class CallPolicyContext:
    """The caller policy envelope; omissions remain explicit unknowns."""

    permitted_effects: tuple[str, ...] = ()
    authorized_capabilities: tuple[str, ...] = ()
    authorization_context_refs: tuple[str, ...] = ()
    resource_budget_refs: tuple[str, ...] = ()
    cancellation_behavior: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "permitted_effects", _bounded_strings(self.permitted_effects, "permitted_effects"))
        object.__setattr__(self, "authorized_capabilities", _bounded_strings(self.authorized_capabilities, "authorized_capabilities"))
        object.__setattr__(self, "authorization_context_refs", _bounded_strings(self.authorization_context_refs, "authorization_context_refs"))
        object.__setattr__(self, "resource_budget_refs", _bounded_strings(self.resource_budget_refs, "resource_budget_refs"))
        object.__setattr__(self, "cancellation_behavior", _compact_text(self.cancellation_behavior, "cancellation_behavior", required=True))


@dataclass(frozen=True)
class BrokenCallSite:
    """Exact bounded observation of the sender side of one call."""

    caller_span: SourceSpan
    caller_symbol_id: str
    receiver_reference: str
    call_form: str
    language: str
    runtime: str
    actual_arguments: tuple[CallArgumentFact, ...] = ()
    awaited: bool = False
    result_uses: tuple[str, ...] = ()
    handled_error_refs: tuple[str, ...] = ()
    policy_context: CallPolicyContext = field(default_factory=CallPolicyContext)
    evidence_refs: tuple[EvidenceReference | ProgramEvidenceFact | Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.caller_span, SourceSpan):
            raise BrokenTraceError("caller_span must be a contract source span")
        object.__setattr__(self, "caller_symbol_id", _identifier(self.caller_symbol_id, "caller_symbol_id"))
        object.__setattr__(self, "receiver_reference", _compact_text(self.receiver_reference, "receiver_reference", required=True))
        object.__setattr__(self, "call_form", _compact_text(self.call_form, "call_form", required=True))
        object.__setattr__(self, "language", _compact_text(self.language, "language", required=True))
        object.__setattr__(self, "runtime", _compact_text(self.runtime, "runtime", required=True))
        if not isinstance(self.awaited, bool):
            raise BrokenTraceError("awaited must be a boolean")
        if len(self.actual_arguments) > MAX_CALL_ARGUMENTS:
            raise ContractRepairBoundsError("actual_arguments exceeds its item bound")
        if not all(isinstance(item, CallArgumentFact) for item in self.actual_arguments):
            raise BrokenTraceError("actual_arguments must contain CallArgumentFact values")
        positions = [item.position for item in self.actual_arguments]
        if len(set(positions)) != len(positions):
            raise BrokenTraceError("actual argument positions must be unique")
        object.__setattr__(self, "actual_arguments", tuple(sorted(self.actual_arguments, key=lambda item: (item.position, item.name))))
        object.__setattr__(self, "result_uses", _bounded_strings(self.result_uses, "result_uses", limit=MAX_RESULT_USES))
        object.__setattr__(self, "handled_error_refs", _bounded_strings(self.handled_error_refs, "handled_error_refs"))
        if not isinstance(self.policy_context, CallPolicyContext):
            raise BrokenTraceError("policy_context must be CallPolicyContext")
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs, "call evidence_refs", required=True))

    @property
    def actual_argument_count(self) -> int:
        """The exact positional-plus-keyword count observed at this call."""
        return len(self.actual_arguments)


@dataclass(frozen=True)
class ResolverEvidence:
    """A normalized resolver answer with the evidence required for its claim.

    ``same_name`` and ``vector`` are retained for audit but deliberately never
    authorize a target or a refactor classification.
    """

    disposition: str
    target_span: SourceSpan | None = None
    target_symbol_id: str = ""
    local_scope_complete: bool = False
    route_closed: bool = False
    evidence_refs: tuple[EvidenceReference | ProgramEvidenceFact | Mapping[str, Any], ...] = ()
    identity_kinds: tuple[str, ...] = ()
    adapter_kinds: tuple[str, ...] = ()
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    same_name: bool = False
    vector_evidence: bool = False

    def __post_init__(self) -> None:
        disposition = _compact_text(self.disposition, "resolver disposition", required=True)
        if disposition not in _RESOLUTION_VALUES:
            raise BrokenTraceError("resolver disposition is not closed")
        object.__setattr__(self, "disposition", disposition)
        if self.target_span is not None and not isinstance(self.target_span, SourceSpan):
            raise BrokenTraceError("target_span must be a contract source span")
        object.__setattr__(self, "target_symbol_id", _compact_text(self.target_symbol_id, "target_symbol_id"))
        if not isinstance(self.local_scope_complete, bool) or not isinstance(self.route_closed, bool):
            raise BrokenTraceError("resolver completeness fields must be boolean")
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs, "resolver evidence_refs", required=True))
        identities = _bounded_strings(self.identity_kinds, "identity_kinds")
        adapters = _bounded_strings(self.adapter_kinds, "adapter_kinds")
        if set(identities).difference(_IDENTITY_KINDS):
            raise BrokenTraceEvidenceError("identity_kinds contains unsupported evidence")
        if set(adapters).difference(_ADAPTER_KINDS):
            raise BrokenTraceEvidenceError("adapter_kinds contains unsupported evidence")
        object.__setattr__(self, "identity_kinds", identities)
        object.__setattr__(self, "adapter_kinds", adapters)
        object.__setattr__(self, "frontier_refs", _bounded_strings(self.frontier_refs, "frontier_refs", limit=MAX_FRONTIER_REFS))
        object.__setattr__(self, "exclusion_refs", _bounded_strings(self.exclusion_refs, "exclusion_refs", limit=MAX_FRONTIER_REFS))
        if not isinstance(self.same_name, bool) or not isinstance(self.vector_evidence, bool):
            raise BrokenTraceError("same_name and vector_evidence must be boolean")


@dataclass(frozen=True)
class GraphEvidence:
    """Snapshot-bound graph coverage used to bound, not invent, a resolution."""

    graph_id: str
    complete: bool
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceReference | ProgramEvidenceFact | Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _identifier(self.graph_id, "graph_id"))
        if not isinstance(self.complete, bool):
            raise BrokenTraceError("graph complete must be a boolean")
        object.__setattr__(self, "frontier_refs", _bounded_strings(self.frontier_refs, "graph frontier_refs", limit=MAX_FRONTIER_REFS))
        object.__setattr__(self, "exclusion_refs", _bounded_strings(self.exclusion_refs, "graph exclusion_refs", limit=MAX_FRONTIER_REFS))
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs, "graph evidence_refs", required=True))


@runtime_checkable
class ProgramCallResolver(Protocol):
    """Minimal resolver protocol accepted by the adapter."""

    def resolve_call(self, call_site: BrokenCallSite, graph: GraphEvidence) -> ResolverEvidence: ...


@runtime_checkable
class ProgramGraph(Protocol):
    """Marker protocol for integrations that expose :class:`GraphEvidence`."""

    def trace_graph_evidence(self, roots: AuthorityRoots) -> GraphEvidence: ...


@dataclass(frozen=True)
class BrokenTraceAnalysis:
    """Trace plus the exact sender facts that cannot fit in the trace record."""

    trace: BrokenContractTrace
    call_site: BrokenCallSite
    resolver_evidence: ResolverEvidence | None
    graph_evidence: GraphEvidence | None
    unknown_frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    schema: str = TRACE_ANALYSIS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.trace, BrokenContractTrace) or not isinstance(self.call_site, BrokenCallSite):
            raise BrokenTraceError("trace and call_site must be typed values")
        if self.trace.caller_span != self.call_site.caller_span or self.trace.caller_symbol_id != self.call_site.caller_symbol_id:
            raise BrokenTraceEvidenceError("trace must bind the exact observed caller")
        if self.trace.receiver_reference != self.call_site.receiver_reference:
            raise BrokenTraceEvidenceError("trace must bind the exact observed receiver")
        if self.resolver_evidence is not None and not isinstance(self.resolver_evidence, ResolverEvidence):
            raise BrokenTraceError("resolver_evidence must be ResolverEvidence or None")
        if self.graph_evidence is not None and not isinstance(self.graph_evidence, GraphEvidence):
            raise BrokenTraceError("graph_evidence must be GraphEvidence or None")
        object.__setattr__(self, "unknown_frontier_refs", _bounded_strings(self.unknown_frontier_refs, "unknown_frontier_refs", limit=MAX_FRONTIER_REFS))
        object.__setattr__(self, "exclusion_refs", _bounded_strings(self.exclusion_refs, "exclusion_refs", limit=MAX_FRONTIER_REFS))


class BrokenTraceClassifier:
    """Classify only the resolver claim that bounded graph evidence supports."""

    def classify(self, roots: AuthorityRoots, call_site: BrokenCallSite, resolver: ResolverEvidence, graph: GraphEvidence) -> BrokenTraceAnalysis:
        if not isinstance(roots, AuthorityRoots):
            raise BrokenTraceError("roots must be AuthorityRoots")
        if not isinstance(call_site, BrokenCallSite) or not isinstance(resolver, ResolverEvidence) or not isinstance(graph, GraphEvidence):
            raise BrokenTraceError("classify requires typed call, resolver, and graph evidence")
        if graph.graph_id != roots.graph_id:
            return self._unsupported(roots, call_site, resolver, graph, "graph_root_mismatch")

        disposition = self._bounded_disposition(resolver, graph)
        target = (
            resolver.target_span
            if disposition in {
                TraceDisposition.RESOLVED_MISMATCH,
                TraceDisposition.LIKELY_REFACTOR,
                TraceDisposition.ADAPTER_REQUIRED,
            }
            else None
        )
        refs = tuple(sorted(set(call_site.evidence_refs + resolver.evidence_refs + graph.evidence_refs), key=lambda item: item.content_id))
        frontier = tuple(sorted(set(graph.frontier_refs + resolver.frontier_refs)))
        exclusions = tuple(sorted(set(graph.exclusion_refs + resolver.exclusion_refs)))
        trace = BrokenContractTrace(
            roots=roots, caller_span=call_site.caller_span,
            caller_symbol_id=call_site.caller_symbol_id,
            receiver_reference=call_site.receiver_reference,
            disposition=disposition, target_span=target, evidence_refs=refs,
            graph_frontier_refs=frontier, excluded_refs=exclusions,
        )
        return BrokenTraceAnalysis(trace, call_site, resolver, graph, frontier, exclusions)

    def _unsupported(self, roots: AuthorityRoots, call_site: BrokenCallSite, resolver: ResolverEvidence | None, graph: GraphEvidence | None, frontier: str) -> BrokenTraceAnalysis:
        refs = tuple(call_site.evidence_refs)
        if resolver is not None:
            refs = tuple(sorted(set(refs + resolver.evidence_refs), key=lambda item: item.content_id))
        if graph is not None:
            refs = tuple(sorted(set(refs + graph.evidence_refs), key=lambda item: item.content_id))
        trace = BrokenContractTrace(roots, call_site.caller_span, call_site.caller_symbol_id, call_site.receiver_reference, TraceDisposition.UNSUPPORTED, evidence_refs=refs, graph_frontier_refs=(frontier,))
        return BrokenTraceAnalysis(trace, call_site, resolver, graph, (frontier,), ())

    @staticmethod
    def _bounded_disposition(resolver: ResolverEvidence, graph: GraphEvidence) -> TraceDisposition:
        claimed = TraceDisposition(resolver.disposition)
        if claimed in {TraceDisposition.DYNAMIC, TraceDisposition.EXTERNAL, TraceDisposition.UNSUPPORTED}:
            return claimed
        if not graph.complete:
            return TraceDisposition.UNSUPPORTED
        if claimed is TraceDisposition.RESOLVED_MISMATCH:
            return claimed if resolver.target_span is not None and resolver.route_closed else TraceDisposition.UNSUPPORTED
        if claimed is TraceDisposition.MISSING_LOCAL:
            return claimed if resolver.local_scope_complete and resolver.target_span is None else TraceDisposition.UNSUPPORTED
        if claimed is TraceDisposition.LIKELY_REFACTOR:
            return claimed if resolver.route_closed and resolver.identity_kinds and resolver.target_span is not None else TraceDisposition.UNSUPPORTED
        if claimed is TraceDisposition.ADAPTER_REQUIRED:
            return claimed if resolver.route_closed and resolver.adapter_kinds and resolver.target_span is not None else TraceDisposition.UNSUPPORTED
        if claimed is TraceDisposition.AMBIGUOUS:
            return claimed
        return TraceDisposition.UNSUPPORTED


class BrokenContractTraceBuilder:
    """Adapter boundary that turns compatible integrations into a safe trace.

    An unavailable or incompatible resolver is represented as ``unsupported``;
    it is not an exception and therefore cannot block other analysis lanes.
    """

    def __init__(self, classifier: BrokenTraceClassifier | None = None) -> None:
        self._classifier = classifier or BrokenTraceClassifier()

    def build(self, roots: AuthorityRoots, call_site: BrokenCallSite, *, resolver: ProgramCallResolver | None, graph: GraphEvidence | ProgramGraph | None) -> BrokenTraceAnalysis:
        if not isinstance(roots, AuthorityRoots) or not isinstance(call_site, BrokenCallSite):
            raise BrokenTraceError("build requires AuthorityRoots and BrokenCallSite")
        graph_evidence = self._graph_evidence(roots, graph)
        if graph_evidence is None or resolver is None or not isinstance(resolver, ProgramCallResolver):
            return self._classifier._unsupported(roots, call_site, None, graph_evidence, "resolver_or_graph_unsupported")
        try:
            result = resolver.resolve_call(call_site, graph_evidence)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return self._classifier._unsupported(roots, call_site, None, graph_evidence, "resolver_incompatible")
        if not isinstance(result, ResolverEvidence):
            return self._classifier._unsupported(roots, call_site, None, graph_evidence, "resolver_incompatible")
        return self._classifier.classify(roots, call_site, result, graph_evidence)

    @staticmethod
    def _graph_evidence(roots: AuthorityRoots, graph: GraphEvidence | ProgramGraph | None) -> GraphEvidence | None:
        if isinstance(graph, GraphEvidence):
            return graph
        if graph is None or not isinstance(graph, ProgramGraph):
            return None
        try:
            evidence = graph.trace_graph_evidence(roots)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return None
        return evidence if isinstance(evidence, GraphEvidence) else None


__all__ = [
    "BrokenCallSite", "BrokenContractTraceBuilder", "BrokenTraceAnalysis",
    "BrokenTraceClassifier", "BrokenTraceError", "BrokenTraceEvidenceError",
    "CallArgumentFact", "CallPolicyContext", "GraphEvidence", "ProgramCallResolver",
    "ProgramGraph", "ResolverEvidence", "TRACE_ANALYSIS_SCHEMA", "TraceDisposition",
]
