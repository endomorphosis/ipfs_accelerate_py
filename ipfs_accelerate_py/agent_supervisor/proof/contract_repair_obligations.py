"""Fail-closed lowering of contract-repair evidence into proof obligations.

Candidate retrieval is deliberately non-authoritative.  This module is the
next boundary: it does not decide that a repair is safe, but turns the exact
facts needed to make that decision into immutable, candidate-specific claims.
There are no implicit defaults for a call slice, an adapter map, strategy
evidence, or a logic capability.  Consequently a later prover can distinguish
an open obligation from a claim that was never representable in the first
place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    EvidenceReference,
    MemorySafetyDisposition,
    MemorySafetyFacet,
    RepairCandidate,
    RepairStrategy,
)
from ..analysis.sender_receiver_contracts import (
    ProgramContractComparison,
)
from ..integrations.contract_repair_capabilities import (
    ContractRepairCapabilityReport,
    ContractRepairCapabilityStatus,
)
from ..program_contracts import SemanticAspect
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
)


CONTRACT_REPAIR_OBLIGATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-obligation@1"
)
CONTRACT_REPAIR_IR_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-ir-claim@1"
)
MAX_OBLIGATIONS: Final = 64
MAX_MAPPING_ENTRIES: Final = 64


class ContractRepairObligationError(ContractValidationError):
    """Evidence is insufficient or inconsistent for a sound lowering."""


class IncompleteCallSliceError(ContractRepairObligationError):
    """A partial call slice attempted to claim a closed proof obligation."""


class UnsupportedObligationError(ContractRepairObligationError):
    """The admitted LogicIR capability cannot express an obligation."""


class ObligationKind(str, Enum):
    CALLER_IMPLIES_RECEIVER_PRECONDITION = "caller_implies_receiver_precondition"
    RECEIVER_GUARANTEE_IMPLIES_CALLER_REQUIREMENT = "receiver_guarantee_implies_caller_requirement"
    ERROR_COMPATIBILITY = "error_compatibility"
    EFFECT_COMPATIBILITY = "effect_compatibility"
    CAPABILITY_COMPATIBILITY = "capability_compatibility"
    AUTHORIZATION_COMPATIBILITY = "authorization_compatibility"
    LIFECYCLE_COMPATIBILITY = "lifecycle_compatibility"
    RESOURCE_COMPATIBILITY = "resource_compatibility"
    MEMORY_COMPATIBILITY = "memory_compatibility"
    REVERSE_REFINEMENT = "reverse_refinement"
    EQUIVALENCE_IDENTITY_HISTORY = "equivalence_identity_history"
    ROUTE_WIRING = "route_wiring"
    ADAPTER_ARGUMENT_TOTALITY = "adapter_argument_totality"
    ADAPTER_RESULT_TOTALITY = "adapter_result_totality"
    ADAPTER_ERROR_TOTALITY = "adapter_error_totality"
    ADAPTER_EFFECT_CAPABILITY_PRESERVATION = "adapter_effect_capability_preservation"
    PLACEMENT_OWNERSHIP = "placement_ownership"
    PLACEMENT_NO_OMITTED_COMPATIBLE_IMPLEMENTATION = "placement_no_omitted_compatible_implementation"
    PLACEMENT_DEPENDENCY_DAG = "placement_dependency_dag"
    PLACEMENT_VISIBILITY_REGISTRATION = "placement_visibility_registration"
    PLACEMENT_EXACT_STUB_CONTRACT = "placement_exact_stub_contract"


class StrategyEvidenceKind(str, Enum):
    """Evidence which cannot be inferred from a local contract comparison."""

    CALL_SLICE = "call_slice"
    IDENTITY_HISTORY = "identity_history"
    ROUTE_WIRING = "route_wiring"
    ADAPTER_ARGUMENT = "adapter_argument"
    ADAPTER_RESULT = "adapter_result"
    ADAPTER_ERROR = "adapter_error"
    ADAPTER_EFFECT_CAPABILITY = "adapter_effect_capability"
    OWNERSHIP = "ownership"
    NO_OMITTED_COMPATIBLE_IMPLEMENTATION = "no_omitted_compatible_implementation"
    DEPENDENCY_DAG = "dependency_dag"
    VISIBILITY_REGISTRATION = "visibility_registration"
    EXACT_STUB_CONTRACT = "exact_stub_contract"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ContractRepairObligationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ContractRepairObligationError(f"{name} is required")
    if any(character.isspace() for character in result):
        raise ContractRepairObligationError(f"{name} must be an opaque identifier")
    return result


def _refs(value: Sequence[EvidenceReference], name: str, *, required: bool = True) -> tuple[EvidenceReference, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractRepairObligationError(f"{name} must be evidence references")
    if len(value) > 256:
        raise ContractRepairObligationError(f"{name} exceeds its bounded size")
    if not all(isinstance(item, EvidenceReference) for item in value):
        raise ContractRepairObligationError(f"{name} must contain EvidenceReference values")
    result = tuple(sorted(set(value), key=lambda item: item.content_id))
    if required and not result:
        raise ContractRepairObligationError(f"{name} must not be empty")
    return result


def _ids(value: Sequence[str], name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractRepairObligationError(f"{name} must be identifiers")
    result = tuple(sorted({_text(item, name) for item in value}))
    if required and not result:
        raise ContractRepairObligationError(f"{name} must not be empty")
    return result


@dataclass(frozen=True)
class CallSlice:
    """An explicit assertion of the finite caller/consumer slice used below."""

    evidence: EvidenceReference
    complete: bool
    frontier_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceReference):
            raise ContractRepairObligationError("call slice requires evidence")
        object.__setattr__(self, "frontier_refs", _ids(self.frontier_refs, "frontier_refs", required=False))
        if not isinstance(self.complete, bool):
            raise ContractRepairObligationError("call slice completeness must be boolean")
        if not self.complete:
            raise IncompleteCallSliceError(
                "partial call slices cannot be compiled as closed obligations"
            )


@dataclass(frozen=True)
class AssumptionBinding:
    """A reviewed assumption identity, not a free-form solver axiom."""

    evidence: EvidenceReference

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceReference):
            raise ContractRepairObligationError("assumptions require evidence references")
        kind = self.evidence.kind.casefold().replace("-", "_")
        if "assumption" not in kind or kind.startswith(("invented", "model", "llm")):
            raise ContractRepairObligationError(
                "assumptions must be reviewed assumption evidence, never free-form axioms"
            )

    @property
    def assumption_id(self) -> str:
        return self.evidence.content_id


@dataclass(frozen=True)
class LogicCapabilityBinding:
    """The exact capability-report fact authorizing LogicIR lowering."""

    capability_id: str
    capability_revision: str
    reconstruction_compatible: bool = True
    supported_semantics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _text(self.capability_id, "capability_id"))
        object.__setattr__(self, "capability_revision", _text(self.capability_revision, "capability_revision"))
        if not self.reconstruction_compatible:
            raise UnsupportedObligationError(
                "LogicIR lowering requires independently reconstructable capability semantics"
            )
        semantics = _ids(self.supported_semantics, "supported_semantics")
        if "ir" not in semantics:
            raise UnsupportedObligationError(
                "capability report must explicitly admit LogicIR semantics"
            )
        object.__setattr__(self, "supported_semantics", semantics)

    @classmethod
    def from_report(cls, report: ContractRepairCapabilityReport) -> "LogicCapabilityBinding":
        if not isinstance(report, ContractRepairCapabilityReport):
            raise ContractRepairObligationError("capability_report must be ContractRepairCapabilityReport")
        try:
            capability = report.capability("datasets.logic_ir")
        except KeyError as exc:
            raise UnsupportedObligationError("capability report lacks datasets.logic_ir") from exc
        if capability.status is not ContractRepairCapabilityStatus.AVAILABLE:
            raise UnsupportedObligationError("datasets.logic_ir is not available in this report")
        if not capability.reconstruction_compatible:
            raise UnsupportedObligationError("datasets.logic_ir is not reconstruction compatible")
        if "ir" not in capability.supported_semantics:
            raise UnsupportedObligationError("datasets.logic_ir does not declare LogicIR semantics")
        revision = str(capability.details.get("capability_revision") or capability.interface_version or capability.schema_version)
        if not revision:
            raise UnsupportedObligationError("datasets.logic_ir has no pinned capability revision")
        return cls(
            capability.capability_id, revision, capability.reconstruction_compatible,
            capability.supported_semantics,
        )


@dataclass(frozen=True)
class StrategyEvidence:
    """Finite, source-bound evidence for a strategy-specific predicate."""

    kind: StrategyEvidenceKind
    references: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", StrategyEvidenceKind(self.kind))
        object.__setattr__(self, "references", _refs(self.references, "strategy evidence references"))


@dataclass(frozen=True)
class FiniteMapping:
    """One explicitly sourced adapter mapping; there is no coercion fallback."""

    source: str
    target: str
    evidence: EvidenceReference

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "mapping source"))
        object.__setattr__(self, "target", _text(self.target, "mapping target"))
        if not isinstance(self.evidence, EvidenceReference):
            raise ContractRepairObligationError("adapter mapping requires evidence")


@dataclass(frozen=True)
class AdapterMappings:
    """All finite maps required to adapt the sender's modeled domain."""

    arguments: tuple[FiniteMapping, ...]
    results: tuple[FiniteMapping, ...] = ()
    errors: tuple[FiniteMapping, ...] = ()

    def __post_init__(self) -> None:
        for name in ("arguments", "results", "errors"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
                raise ContractRepairObligationError(f"adapter {name} must be finite mappings")
            if len(values) > MAX_MAPPING_ENTRIES or not all(isinstance(item, FiniteMapping) for item in values):
                raise ContractRepairObligationError(f"adapter {name} must be bounded FiniteMapping values")
            if len({item.source for item in values}) != len(values):
                raise ContractRepairObligationError(f"adapter {name} contains duplicate source mappings")
            object.__setattr__(self, name, tuple(sorted(values, key=lambda item: item.source)))


@dataclass(frozen=True)
class ObligationContext:
    """Exact non-semantic bindings shared by every emitted claim."""

    call_slice: CallSlice
    assumptions: tuple[AssumptionBinding, ...]
    capability: LogicCapabilityBinding
    strategy_evidence: tuple[StrategyEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.call_slice, CallSlice):
            raise ContractRepairObligationError("obligation context requires a complete call slice")
        if not isinstance(self.capability, LogicCapabilityBinding):
            raise ContractRepairObligationError("obligation context requires a LogicIR capability binding")
        if not isinstance(self.assumptions, Sequence) or not self.assumptions:
            raise ContractRepairObligationError("obligation context requires reviewed assumption bindings")
        if not all(isinstance(item, AssumptionBinding) for item in self.assumptions):
            raise ContractRepairObligationError("assumptions must be AssumptionBinding values")
        if not all(isinstance(item, StrategyEvidence) for item in self.strategy_evidence):
            raise ContractRepairObligationError("strategy evidence must be StrategyEvidence values")
        kinds = [item.kind for item in self.strategy_evidence]
        if len(kinds) != len(set(kinds)):
            raise ContractRepairObligationError("strategy evidence must not repeat a kind")
        object.__setattr__(self, "assumptions", tuple(sorted(self.assumptions, key=lambda item: item.assumption_id)))
        object.__setattr__(self, "strategy_evidence", tuple(sorted(self.strategy_evidence, key=lambda item: item.kind.value)))

    def refs_for(self, kind: StrategyEvidenceKind) -> tuple[EvidenceReference, ...]:
        for item in self.strategy_evidence:
            if item.kind is kind:
                return item.references
        raise ContractRepairObligationError(f"missing required strategy evidence: {kind.value}")


@dataclass(frozen=True)
class IRClaim(CanonicalContract):
    """Small immutable LogicIR claim with every source of authority named."""

    SCHEMA: ClassVar[str] = CONTRACT_REPAIR_IR_CLAIM_SCHEMA

    predicate: str
    subject_id: str
    premise_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    repository_id: str
    tree_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str
    capability_id: str
    capability_revision: str

    def __post_init__(self) -> None:
        for name in (
            "predicate", "subject_id", "repository_id", "tree_id", "translator_id",
            "toolchain_id", "policy_id", "capability_id", "capability_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("premise_ids", "source_ids", "assumption_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))

    def _payload(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate, "subject_id": self.subject_id,
            "premise_ids": list(self.premise_ids), "source_ids": list(self.source_ids),
            "assumption_ids": list(self.assumption_ids), "repository_id": self.repository_id,
            "tree_id": self.tree_id, "translator_id": self.translator_id,
            "toolchain_id": self.toolchain_id, "policy_id": self.policy_id,
            "capability_id": self.capability_id, "capability_revision": self.capability_revision,
        }

    def to_logic_ir(self) -> dict[str, Any]:
        """The backend-neutral, fully bound lowering payload."""
        return self.to_dict()


@dataclass(frozen=True)
class ProofObligation(CanonicalContract):
    """One candidate-specific claim plus its existing code-proof envelope."""

    SCHEMA: ClassVar[str] = CONTRACT_REPAIR_OBLIGATION_SCHEMA

    kind: ObligationKind
    candidate_id: str
    claim: IRClaim
    code_obligation: CodeProofObligation
    source_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ObligationKind(self.kind))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not isinstance(self.claim, IRClaim) or not isinstance(self.code_obligation, CodeProofObligation):
            raise ContractRepairObligationError("proof obligation requires typed claim and CodeProofObligation")
        object.__setattr__(self, "source_refs", _refs(self.source_refs, "source_refs"))
        if self.code_obligation.repository_tree_id != self.claim.tree_id:
            raise ContractRepairObligationError("code obligation tree must match the LogicIR claim")
        if set(self.claim.source_ids).difference(ref.content_id for ref in self.source_refs):
            raise ContractRepairObligationError("claim source ids must be carried by source_refs")

    @property
    def obligation_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "candidate_id": self.candidate_id,
            "claim": self.claim.to_dict(), "code_obligation": self.code_obligation.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
        }


@dataclass(frozen=True)
class SubstitutionObligation:
    """Compatibility obligation group for a direct receiver substitution."""

    obligations: tuple[ProofObligation, ...]


@dataclass(frozen=True)
class EquivalenceObligation:
    """Additional evidence required before calling a substitution a rename."""

    obligations: tuple[ProofObligation, ...]


@dataclass(frozen=True)
class AdapterObligation:
    """Explicit total mapping obligations for a bounded adapter."""

    obligations: tuple[ProofObligation, ...]


@dataclass(frozen=True)
class PlacementObligation:
    """Evidence required for existing-declaration or new-site placement."""

    obligations: tuple[ProofObligation, ...]


@dataclass(frozen=True)
class ContractRepairObligationCompilation:
    """Complete, deterministic collection compiled for one candidate and tree."""

    roots: AuthorityRoots
    trace_id: str
    candidate_id: str
    obligations: tuple[ProofObligation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots):
            raise ContractRepairObligationError("compilation requires authority roots")
        object.__setattr__(self, "trace_id", _text(self.trace_id, "trace_id"))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not self.obligations or len(self.obligations) > MAX_OBLIGATIONS:
            raise ContractRepairObligationError("compilation requires a bounded nonempty obligation set")
        if not all(isinstance(item, ProofObligation) for item in self.obligations):
            raise ContractRepairObligationError("compilation obligations must be typed")
        if len({item.kind for item in self.obligations}) != len(self.obligations):
            raise ContractRepairObligationError("compilation contains duplicate obligation kinds")
        if any(item.candidate_id != self.candidate_id for item in self.obligations):
            raise ContractRepairObligationError("obligations must bind one exact candidate")
        object.__setattr__(self, "obligations", tuple(sorted(self.obligations, key=lambda item: item.kind.value)))

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(item.obligation_id for item in self.obligations)


_ASPECT_KIND: Final[dict[SemanticAspect, ObligationKind]] = {
    SemanticAspect.INPUTS: ObligationKind.CALLER_IMPLIES_RECEIVER_PRECONDITION,
    SemanticAspect.OUTPUTS: ObligationKind.RECEIVER_GUARANTEE_IMPLIES_CALLER_REQUIREMENT,
    SemanticAspect.ERRORS: ObligationKind.ERROR_COMPATIBILITY,
    SemanticAspect.SIDE_EFFECTS: ObligationKind.EFFECT_COMPATIBILITY,
    SemanticAspect.CAPABILITIES: ObligationKind.CAPABILITY_COMPATIBILITY,
    SemanticAspect.AUTHORIZATION: ObligationKind.AUTHORIZATION_COMPATIBILITY,
    SemanticAspect.SYNC_ASYNC: ObligationKind.LIFECYCLE_COMPATIBILITY,
    SemanticAspect.IDEMPOTENCE: ObligationKind.LIFECYCLE_COMPATIBILITY,
    SemanticAspect.ORDERING: ObligationKind.LIFECYCLE_COMPATIBILITY,
    SemanticAspect.ATOMICITY: ObligationKind.LIFECYCLE_COMPATIBILITY,
    SemanticAspect.CONSISTENCY: ObligationKind.LIFECYCLE_COMPATIBILITY,
    SemanticAspect.FALLBACK_DEGRADATION: ObligationKind.LIFECYCLE_COMPATIBILITY,
    SemanticAspect.RESOURCE_BOUNDS: ObligationKind.RESOURCE_COMPATIBILITY,
}


class ContractRepairObligationCompiler:
    """Compile all mandatory obligations without ranking or admitting a candidate."""

    def compile(
        self,
        trace: BrokenContractTrace,
        comparison: ProgramContractComparison,
        candidate: RepairCandidate,
        memory_facet: MemorySafetyFacet,
        context: ObligationContext,
        *,
        adapter_mappings: AdapterMappings | None = None,
    ) -> ContractRepairObligationCompilation:
        self._validate_bindings(trace, comparison, candidate, memory_facet, context)
        if not comparison.compatible:
            failed = ", ".join(item.aspect.value for item in comparison.failed_clauses)
            raise UnsupportedObligationError(f"contract comparison is not a compatible substitution: {failed}")
        if memory_facet.disposition is MemorySafetyDisposition.UNSUPPORTED:
            raise UnsupportedObligationError("memory-safety facet is unsupported and cannot be approximated")
        if memory_facet.disposition in {MemorySafetyDisposition.STALE, MemorySafetyDisposition.ERROR}:
            raise UnsupportedObligationError("memory-safety facet is stale or erroneous")

        obligations = list(self._substitution(trace, comparison, candidate, memory_facet, context))
        if candidate.strategy is RepairStrategy.RENAME_SUBSTITUTION:
            obligations.extend(self._equivalence(trace, comparison, candidate, context))
        elif candidate.strategy is RepairStrategy.ADAPTER:
            if adapter_mappings is None:
                raise ContractRepairObligationError("adapter candidates require explicit total finite mappings")
            obligations.extend(self._adapter(trace, comparison, candidate, context, adapter_mappings))
        elif candidate.strategy in {RepairStrategy.IMPLEMENT_EXISTING_DECLARATION, RepairStrategy.NEW_IMPLEMENTATION}:
            obligations.extend(self._placement(trace, comparison, candidate, context))
        else:
            raise ContractRepairObligationError("reject and ambiguous candidates cannot receive proof obligations")
        return ContractRepairObligationCompilation(trace.roots, trace.content_id, candidate.content_id, tuple(obligations))

    compile_candidate = compile

    def _validate_bindings(self, trace: BrokenContractTrace, comparison: ProgramContractComparison,
                           candidate: RepairCandidate, memory: MemorySafetyFacet, context: ObligationContext) -> None:
        if not all(isinstance(item, expected) for item, expected in (
            (trace, BrokenContractTrace), (comparison, ProgramContractComparison),
            (candidate, RepairCandidate), (memory, MemorySafetyFacet), (context, ObligationContext),
        )):
            raise ContractRepairObligationError("compiler inputs must use the typed repair contracts")
        roots = trace.roots
        if candidate.roots != roots or memory.roots != roots or comparison.call_requirement.roots != roots:
            raise ContractRepairObligationError("trace, contracts, candidate, and memory facet must bind identical authority roots")
        if candidate.trace_id != trace.content_id or comparison.call_requirement.trace_id != trace.content_id:
            raise ContractRepairObligationError("candidate and call requirement must bind the exact trace")
        if comparison.call_requirement.caller_span != trace.caller_span:
            raise ContractRepairObligationError("call requirement must bind the exact caller span")
        if context.call_slice.evidence not in comparison.call_requirement.evidence_refs and context.call_slice.evidence not in trace.evidence_refs:
            raise ContractRepairObligationError("call slice evidence must be attached to the trace or call requirement")

    def _make(self, kind: ObligationKind, trace: BrokenContractTrace, candidate: RepairCandidate,
              context: ObligationContext, sources: Sequence[EvidenceReference], premise_ids: Sequence[str]) -> ProofObligation:
        all_sources = _refs(sources, "obligation sources")
        roots = trace.roots
        claim = IRClaim(
            predicate=kind.value, subject_id=candidate.content_id,
            premise_ids=_ids(premise_ids, "premise_ids"),
            source_ids=tuple(item.content_id for item in all_sources),
            assumption_ids=tuple(item.assumption_id for item in context.assumptions),
            repository_id=roots.repository_id, tree_id=roots.tree_id,
            translator_id=roots.translator_id, toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id, capability_id=context.capability.capability_id,
            capability_revision=context.capability.capability_revision,
        )
        scopes = tuple(sorted({trace.caller_span.artifact_id, candidate.target_span.artifact_id}))
        envelope = CodeProofObligation(
            repository_id=roots.repository_id, repository_tree_id=roots.tree_id,
            ast_scope_ids=scopes, statement=f"{kind.value}:{candidate.content_id}",
            premise_ids=claim.premise_ids, template_id=f"contract-repair/{kind.value}",
            template_version="1", template_semantic_hash=claim.content_id,
            invariant_class="contract_repair", task_id="RPR-009",
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            metadata={"claim_id": claim.content_id, "translator_id": roots.translator_id,
                      "toolchain_id": roots.toolchain_id, "policy_id": roots.policy_id,
                      "capability_id": context.capability.capability_id,
                      "capability_revision": context.capability.capability_revision,
                      "supported_semantics": context.capability.supported_semantics},
        )
        return ProofObligation(kind, candidate.content_id, claim, envelope, all_sources)

    def _substitution(self, trace: BrokenContractTrace, comparison: ProgramContractComparison,
                      candidate: RepairCandidate, memory: MemorySafetyFacet,
                      context: ObligationContext) -> tuple[ProofObligation, ...]:
        sources = tuple(sorted({*comparison.call_requirement.evidence_refs, *candidate.evidence_refs}, key=lambda item: item.content_id))
        premise = (comparison.call_requirement.content_id, candidate.content_id, context.call_slice.evidence.content_id)
        items: dict[ObligationKind, ProofObligation] = {}
        for clause in comparison.clauses:
            kind = _ASPECT_KIND.get(clause.aspect)
            if kind is not None:
                # Several lifecycle clauses lower to a single conjunction and are
                # explicitly represented by all their clause identities.
                items[kind] = self._make(kind, trace, candidate, context, sources, (*premise, clause.aspect.value))
        memory_sources = tuple(sorted({*sources, *memory.evidence_refs, *memory.proof_refs}, key=lambda item: item.content_id))
        items[ObligationKind.MEMORY_COMPATIBILITY] = self._make(
            ObligationKind.MEMORY_COMPATIBILITY, trace, candidate, context, memory_sources,
            (*premise, memory.content_id),
        )
        return tuple(items.values())

    def _equivalence(self, trace: BrokenContractTrace, comparison: ProgramContractComparison,
                     candidate: RepairCandidate, context: ObligationContext) -> tuple[ProofObligation, ...]:
        identity = context.refs_for(StrategyEvidenceKind.IDENTITY_HISTORY)
        route = context.refs_for(StrategyEvidenceKind.ROUTE_WIRING)
        common = (comparison.call_requirement.content_id, candidate.content_id, context.call_slice.evidence.content_id)
        # The direct comparison is forward refinement.  This separate reverse
        # claim stops a compatible substitution from being mislabeled a rename.
        reverse = self._make(ObligationKind.REVERSE_REFINEMENT, trace, candidate, context,
                             (*comparison.call_requirement.evidence_refs, *identity), common)
        identity_claim = self._make(ObligationKind.EQUIVALENCE_IDENTITY_HISTORY, trace, candidate, context,
                                    identity, common)
        route_claim = self._make(ObligationKind.ROUTE_WIRING, trace, candidate, context, route, common)
        return (reverse, identity_claim, route_claim)

    def _adapter(self, trace: BrokenContractTrace, comparison: ProgramContractComparison,
                 candidate: RepairCandidate, context: ObligationContext,
                 mappings: AdapterMappings) -> tuple[ProofObligation, ...]:
        sender = comparison.sender.contract
        receiver = comparison.receiver.contract
        self._require_total({item.name for item in sender.inputs}, mappings.arguments, "argument")
        if sender.returns is not None:
            self._require_total({"result"}, mappings.results, "result")
        self._require_total({item.error_name for item in receiver.errors}, mappings.errors, "error")
        common = (comparison.call_requirement.content_id, candidate.content_id, context.call_slice.evidence.content_id)
        argument = self._make(ObligationKind.ADAPTER_ARGUMENT_TOTALITY, trace, candidate, context,
                              (*context.refs_for(StrategyEvidenceKind.ADAPTER_ARGUMENT), *(item.evidence for item in mappings.arguments)), common)
        result: list[ProofObligation] = [argument]
        if sender.returns is not None:
            result.append(self._make(ObligationKind.ADAPTER_RESULT_TOTALITY, trace, candidate, context,
                                     (*context.refs_for(StrategyEvidenceKind.ADAPTER_RESULT), *(item.evidence for item in mappings.results)), common))
        if receiver.errors:
            result.append(self._make(ObligationKind.ADAPTER_ERROR_TOTALITY, trace, candidate, context,
                                     (*context.refs_for(StrategyEvidenceKind.ADAPTER_ERROR), *(item.evidence for item in mappings.errors)), common))
        result.append(self._make(ObligationKind.ADAPTER_EFFECT_CAPABILITY_PRESERVATION, trace, candidate, context,
                                 context.refs_for(StrategyEvidenceKind.ADAPTER_EFFECT_CAPABILITY), common))
        return tuple(result)

    @staticmethod
    def _require_total(required: set[str], values: Sequence[FiniteMapping], label: str) -> None:
        actual = {item.source for item in values}
        if required != actual:
            missing, extra = sorted(required.difference(actual)), sorted(actual.difference(required))
            raise ContractRepairObligationError(
                f"adapter {label} mapping is not total (missing={missing}, extra={extra})"
            )

    def _placement(self, trace: BrokenContractTrace, comparison: ProgramContractComparison,
                   candidate: RepairCandidate, context: ObligationContext) -> tuple[ProofObligation, ...]:
        common = (comparison.call_requirement.content_id, candidate.content_id, context.call_slice.evidence.content_id)
        pairs = (
            (ObligationKind.PLACEMENT_OWNERSHIP, StrategyEvidenceKind.OWNERSHIP),
            (ObligationKind.PLACEMENT_NO_OMITTED_COMPATIBLE_IMPLEMENTATION, StrategyEvidenceKind.NO_OMITTED_COMPATIBLE_IMPLEMENTATION),
            (ObligationKind.PLACEMENT_DEPENDENCY_DAG, StrategyEvidenceKind.DEPENDENCY_DAG),
            (ObligationKind.PLACEMENT_VISIBILITY_REGISTRATION, StrategyEvidenceKind.VISIBILITY_REGISTRATION),
            (ObligationKind.PLACEMENT_EXACT_STUB_CONTRACT, StrategyEvidenceKind.EXACT_STUB_CONTRACT),
        )
        return tuple(self._make(kind, trace, candidate, context, context.refs_for(evidence), common)
                     for kind, evidence in pairs)


__all__ = [
    "AdapterMappings", "AdapterObligation", "AssumptionBinding", "CallSlice",
    "ContractRepairObligationCompilation", "ContractRepairObligationCompiler",
    "ContractRepairObligationError", "EquivalenceObligation", "FiniteMapping",
    "IRClaim", "IncompleteCallSliceError", "LogicCapabilityBinding", "MAX_MAPPING_ENTRIES",
    "MAX_OBLIGATIONS", "ObligationContext", "ObligationKind", "PlacementObligation",
    "ProofObligation", "StrategyEvidence", "StrategyEvidenceKind", "SubstitutionObligation",
    "UnsupportedObligationError",
]
