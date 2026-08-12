"""Independent sender requirements and receiver-guarantee comparison.

This adapter deliberately consumes :mod:`program_contracts` rather than
re-extracting contracts from implementation.  A candidate's observation is
useful diagnostic evidence, but is never a source of a sender expectation or
of a receiver guarantee.  Incomplete, unsupported, and conflicting clauses
remain first-class comparison outcomes so callers cannot mistake an absence of
evidence for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..proof.program_contracts import (
    CapabilityMode,
    CapabilitySpec,
    ContractSourceKind,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    ObservedProgramContract,
    Optionality,
    ParameterKind,
    ParameterSpec,
    ReturnSpec,
    SemanticAspect,
    SupportStatus,
)
from .contract_repair_contracts import (
    BrokenContractTrace,
    CallRequirementContract,
    EvidenceReference,
)


MAX_CLAUSES: Final[int] = 64


class SenderReceiverContractError(ValueError):
    """The synthesis input cannot support an independently sourced claim."""


class ClauseDisposition(str, Enum):
    """Closed outcome for one call-contract semantic aspect."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ClauseComparison:
    """One explicit, human-readable proof obligation before logic lowering."""

    aspect: SemanticAspect
    disposition: ClauseDisposition
    reason: str

    @property
    def compatible(self) -> bool:
        return self.disposition in {
            ClauseDisposition.SATISFIED,
            ClauseDisposition.NOT_APPLICABLE,
        }


@dataclass(frozen=True)
class SenderRequirement:
    """The independently sourced domain and postcondition required at a call."""

    contract: ExpectedProgramContract
    call_requirement: CallRequirementContract

    @property
    def provided_inputs(self) -> tuple[ParameterSpec, ...]:
        return self.contract.inputs

    @property
    def required_output(self) -> ReturnSpec | None:
        return self.contract.returns

    @property
    def allowed_or_handled_errors(self) -> tuple[ErrorSpec, ...]:
        return self.contract.errors

    @property
    def permitted_effects(self):
        return self.contract.side_effects

    @property
    def authorized_capabilities(self) -> tuple[CapabilitySpec, ...]:
        return self.contract.capabilities

    @property
    def authorization_context(self):
        return self.contract.authorization

    @property
    def lifecycle_requirements(self) -> tuple[object | None, ...]:
        return (
            self.contract.sync_async,
            self.contract.idempotence,
            self.contract.ordering,
            self.contract.atomicity,
            self.contract.consistency,
            self.contract.resource_bounds,
            self.contract.fallback,
        )


@dataclass(frozen=True)
class ReceiverGuarantee:
    """A candidate guarantee derived exclusively from expectation sources."""

    contract: ExpectedProgramContract
    observed: ObservedProgramContract | None = None

    def __post_init__(self) -> None:
        if self.observed is not None and not self.observed.binds_same_subject(self.contract):
            raise SenderReceiverContractError(
                "receiver observation must bind the same subject and interface"
            )

    @property
    def accepted_inputs(self) -> tuple[ParameterSpec, ...]:
        return self.contract.inputs

    @property
    def guaranteed_output(self) -> ReturnSpec | None:
        return self.contract.returns

    @property
    def errors(self) -> tuple[ErrorSpec, ...]:
        return self.contract.errors

    @property
    def guaranteed_effects(self):
        return self.contract.side_effects

    @property
    def required_capabilities(self) -> tuple[CapabilitySpec, ...]:
        return self.contract.capabilities

    @property
    def authorization_precondition(self):
        return self.contract.authorization

    @property
    def lifecycle_guarantees(self) -> tuple[object | None, ...]:
        return (
            self.contract.sync_async,
            self.contract.idempotence,
            self.contract.ordering,
            self.contract.atomicity,
            self.contract.consistency,
            self.contract.resource_bounds,
            self.contract.fallback,
        )


@dataclass(frozen=True)
class ProgramContractComparison:
    """Complete compatibility result; unsupported and conflict are failures."""

    sender: SenderRequirement
    receiver: ReceiverGuarantee
    clauses: tuple[ClauseComparison, ...]

    def __post_init__(self) -> None:
        if not self.clauses or len(self.clauses) > MAX_CLAUSES:
            raise SenderReceiverContractError("comparison requires bounded clauses")
        aspects = [item.aspect for item in self.clauses]
        if len(aspects) != len(set(aspects)):
            raise SenderReceiverContractError("comparison has duplicate aspect clauses")

    @property
    def compatible(self) -> bool:
        return all(clause.compatible for clause in self.clauses)

    @property
    def failed_clauses(self) -> tuple[ClauseComparison, ...]:
        return tuple(item for item in self.clauses if not item.compatible)

    @property
    def unsupported_clauses(self) -> tuple[ClauseComparison, ...]:
        return tuple(
            item for item in self.clauses
            if item.disposition is ClauseDisposition.UNSUPPORTED
        )

    @property
    def conflicts(self) -> tuple[ClauseComparison, ...]:
        return tuple(
            item for item in self.clauses
            if item.disposition is ClauseDisposition.CONFLICT
        )

    @property
    def call_requirement(self) -> CallRequirementContract:
        """Trace-bound join record with provenance from both contract halves."""

        sender_record = self.sender.call_requirement
        receiver_refs = _source_evidence(self.receiver.contract)
        evidence = tuple(sorted(
            {*sender_record.evidence_refs, *receiver_refs}, key=lambda item: item.content_id
        ))
        unsupported = tuple(sorted({
            *sender_record.unsupported_clause_refs,
            *(item.unsupported_id for item in self.receiver.contract.unsupported),
        }))
        return CallRequirementContract(
            roots=sender_record.roots,
            trace_id=sender_record.trace_id,
            caller_span=sender_record.caller_span,
            requirement_refs=sender_record.requirement_refs,
            receiver_contract_refs=receiver_refs,
            evidence_refs=evidence,
            proof_refs=sender_record.proof_refs,
            unsupported_clause_refs=unsupported,
        )


# A shorter compatibility spelling for consumers following ProgramContract@1.
ContractComparison = ProgramContractComparison


def _source_evidence(contract: ExpectedProgramContract) -> tuple[EvidenceReference, ...]:
    """Convert compact approved source provenance into repair evidence refs."""

    result = []
    for source in contract.sources:
        if not source.source_kind.may_define_expectation:
            raise SenderReceiverContractError(
                "implementation observations cannot define expectations"
            )
        result.append(
            EvidenceReference(
                kind=source.source_kind.value,
                artifact_id=source.artifact_id,
                locator=source.locator,
                producer_id=source.source_id,
            )
        )
    return tuple(result)


def _unsupported(contract: ExpectedProgramContract, aspect: SemanticAspect) -> bool:
    # ProgramContract@1 uses UNKNOWN for absent optional facets.  Absence is
    # handled by the individual comparison as an explicit not-applicable or
    # missing-guarantee clause; only a recorded UnsupportedSemantics marker is
    # an unsupported claim.
    return contract.aspect_support(aspect) is SupportStatus.UNSUPPORTED


def _clause(aspect: SemanticAspect, disposition: ClauseDisposition, reason: str) -> ClauseComparison:
    return ClauseComparison(aspect, disposition, reason)


def _matching_receiver_parameter(
    sender_parameter: ParameterSpec, receiver_parameters: tuple[ParameterSpec, ...]
) -> ParameterSpec | None:
    """Match by stable call name first, then positional slot only."""

    for candidate in receiver_parameters:
        if candidate.name == sender_parameter.name:
            return candidate
    if sender_parameter.kind is ParameterKind.POSITIONAL:
        return next(
            (
                candidate for candidate in receiver_parameters
                if candidate.kind is ParameterKind.POSITIONAL
                and candidate.position == sender_parameter.position
            ),
            None,
        )
    return None


def _inputs(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.INPUTS
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "input domain is not fully modeled")
    for provided in sender.inputs:
        accepted = _matching_receiver_parameter(provided, receiver.inputs)
        if accepted is None:
            return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver lacks input {provided.name!r}")
        if accepted.kind is not provided.kind:
            return _clause(aspect, ClauseDisposition.VIOLATED, f"input {provided.name!r} changes calling convention")
        if not accepted.is_input_compatible_with(provided):
            return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver narrows input {provided.name!r}")
        if provided.optionality is not Optionality.REQUIRED and accepted.optionality is Optionality.REQUIRED:
            return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver requires optional input {provided.name!r}")
    for accepted in receiver.inputs:
        provided = _matching_receiver_parameter(accepted, sender.inputs)
        if accepted.optionality is Optionality.REQUIRED and provided is None:
            return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver adds required input {accepted.name!r}")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver accepts the complete sender input domain contravariantly")


def _outputs(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.OUTPUTS
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "output domain is not fully modeled")
    if sender.returns is None:
        return _clause(aspect, ClauseDisposition.NOT_APPLICABLE, "caller has no modeled result use")
    if receiver.returns is None:
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver does not guarantee a required result")
    if receiver.returns.optionality is not Optionality.REQUIRED and sender.returns.optionality is Optionality.REQUIRED:
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver may omit a required result")
    if not receiver.returns.is_subtype_of(sender.returns):
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver result is not covariant with consumer requirement")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver output refines consumer-required output covariantly")


def _errors(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.ERRORS
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "error behavior is not fully modeled")
    allowed = {(item.error_name, item.code) for item in sender.errors}
    extra = [item.error_name for item in receiver.errors if (item.error_name, item.code) not in allowed]
    if extra:
        return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver can raise unhandled errors: {', '.join(sorted(extra))}")
    return _clause(aspect, ClauseDisposition.SATISFIED, "all receiver errors are allowed or handled")


def _effects(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.SIDE_EFFECTS
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "effect policy is not fully modeled")
    permitted = {item.effect_kind for item in sender.side_effects if item.polarity in {EffectPolarity.ALLOWED, EffectPolarity.REQUIRED}}
    forbidden = {item.effect_kind for item in sender.side_effects if item.polarity is EffectPolarity.FORBIDDEN}
    receiver_effects = {item.effect_kind for item in receiver.side_effects if item.polarity in {EffectPolarity.ALLOWED, EffectPolarity.REQUIRED}}
    illegal = receiver_effects.intersection(forbidden) | (receiver_effects - permitted - {EffectKind.NONE})
    if illegal:
        return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver exceeds permitted effects: {', '.join(sorted(item.value for item in illegal))}")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver effects fit caller policy")


def _capabilities(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.CAPABILITIES
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "capability policy is not fully modeled")
    authorized = {item.capability_name: item for item in sender.capabilities if item.mode is not CapabilityMode.FORBIDDEN}
    forbidden = {item.capability_name for item in sender.capabilities if item.mode is CapabilityMode.FORBIDDEN}
    required = [item.capability_name for item in receiver.capabilities if item.mode is CapabilityMode.REQUIRED]
    missing = sorted(name for name in required if name not in authorized or name in forbidden)
    if missing:
        return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver requires unauthorized capabilities: {', '.join(missing)}")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver capabilities fit caller authorization")


def _refinement(
    aspect: SemanticAspect, sender_value, receiver_value, label: str
) -> ClauseComparison:
    if sender_value is None:
        return _clause(aspect, ClauseDisposition.NOT_APPLICABLE, f"caller has no {label} requirement")
    if receiver_value is None:
        return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver omits required {label} guarantee")
    if not receiver_value.is_refinement_of(sender_value):
        return _clause(aspect, ClauseDisposition.VIOLATED, f"receiver does not refine required {label}")
    return _clause(aspect, ClauseDisposition.SATISFIED, f"receiver refines required {label}")


def _ordering(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.ORDERING
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "ordering is not fully modeled")
    if sender.ordering is None:
        return _clause(aspect, ClauseDisposition.NOT_APPLICABLE, "caller has no ordering requirement")
    if receiver.ordering is None:
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver omits required ordering guarantee")
    if sender.ordering != receiver.ordering:
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver ordering differs from caller requirement")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver preserves required ordering")


def _sync_async(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.SYNC_ASYNC
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "async/cancellation behavior is not fully modeled")
    if sender.sync_async is None:
        return _clause(aspect, ClauseDisposition.NOT_APPLICABLE, "caller has no sync/async requirement")
    if receiver.sync_async is None:
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver omits sync/async guarantee")
    if not receiver.sync_async.is_compatible_with(sender.sync_async):
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver sync/async lifecycle is incompatible")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver matches caller sync/async lifecycle")


def _authorization(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.AUTHORIZATION
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "authorization behavior is not fully modeled")
    if receiver.authorization is None:
        return _clause(aspect, ClauseDisposition.NOT_APPLICABLE, "receiver declares no authorization precondition")
    if sender.authorization is None:
        return _clause(aspect, ClauseDisposition.VIOLATED, "caller authorization context is absent")
    if not sender.authorization.is_refinement_of(receiver.authorization):
        return _clause(aspect, ClauseDisposition.VIOLATED, "caller authorization cannot satisfy receiver precondition")
    return _clause(aspect, ClauseDisposition.SATISFIED, "caller authorization satisfies receiver precondition")


def _fallback(sender: ExpectedProgramContract, receiver: ExpectedProgramContract) -> ClauseComparison:
    aspect = SemanticAspect.FALLBACK_DEGRADATION
    if _unsupported(sender, aspect) or _unsupported(receiver, aspect):
        return _clause(aspect, ClauseDisposition.UNSUPPORTED, "fallback behavior is not fully modeled")
    if sender.fallback is None:
        return _clause(aspect, ClauseDisposition.NOT_APPLICABLE, "caller has no fallback requirement")
    if receiver.fallback != sender.fallback:
        return _clause(aspect, ClauseDisposition.VIOLATED, "receiver fallback/degradation behavior differs")
    return _clause(aspect, ClauseDisposition.SATISFIED, "receiver preserves required fallback behavior")


class SenderRequirementCompiler:
    """Build the caller/consumer half of a trace-bound requirement contract."""

    def compile(
        self, trace: BrokenContractTrace, expected: ExpectedProgramContract
    ) -> SenderRequirement:
        if expected.symbol.repository_id != trace.roots.repository_id or expected.symbol.tree_id != trace.roots.tree_id:
            raise SenderReceiverContractError("sender expectation must bind the trace repository and tree")
        evidence = tuple(sorted({*trace.evidence_refs, *_source_evidence(expected)}, key=lambda item: item.content_id))
        requirement = CallRequirementContract(
            roots=trace.roots,
            trace_id=trace.content_id,
            caller_span=trace.caller_span,
            requirement_refs=_source_evidence(expected),
            evidence_refs=evidence,
            unsupported_clause_refs=tuple(item.unsupported_id for item in expected.unsupported),
        )
        return SenderRequirement(expected, requirement)

    compile_sender = compile


class ReceiverGuaranteeCompiler:
    """Build candidate guarantees without allowing observation-only authority."""

    def compile(
        self,
        expected: ExpectedProgramContract,
        observed: ObservedProgramContract | None = None,
    ) -> ReceiverGuarantee:
        # ExpectedProgramContract itself rejects implementation-observation
        # provenance.  Observations are retained only as non-authoritative
        # diagnostics on the result.
        if not expected.sources or any(
            source.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION
            for source in expected.sources
        ):
            raise SenderReceiverContractError("receiver guarantee requires reviewed expectation evidence")
        return ReceiverGuarantee(expected, observed)

    compile_receiver = compile


class SenderReceiverContractCompiler:
    """Compare a sender requirement with one independently sourced receiver."""

    def __init__(self) -> None:
        self.sender_compiler = SenderRequirementCompiler()
        self.receiver_compiler = ReceiverGuaranteeCompiler()

    def compare(self, sender: SenderRequirement, receiver: ReceiverGuarantee) -> ProgramContractComparison:
        sender_contract = sender.contract
        receiver_contract = receiver.contract
        clauses: list[ClauseComparison] = []
        if not sender_contract.interface.binds_same_surface(receiver_contract.interface):
            clauses.append(_clause(SemanticAspect.IDENTITY, ClauseDisposition.VIOLATED, "receiver does not expose the required interface surface"))
        else:
            clauses.append(_clause(SemanticAspect.IDENTITY, ClauseDisposition.SATISFIED, "receiver exposes the required interface surface"))
        if sender_contract.policy_revision != receiver_contract.policy_revision:
            clauses.append(_clause(SemanticAspect.SOURCE_PRECEDENCE, ClauseDisposition.VIOLATED, "sender and receiver use different policy revisions"))
        elif sender_contract.has_conflicts or receiver_contract.has_conflicts:
            clauses.append(_clause(SemanticAspect.SOURCE_PRECEDENCE, ClauseDisposition.CONFLICT, "source-precedence conflict requires explicit adjudication"))
        else:
            clauses.append(_clause(SemanticAspect.SOURCE_PRECEDENCE, ClauseDisposition.SATISFIED, "expectations use explicit reviewed-source precedence"))
        clauses.extend((
            _inputs(sender_contract, receiver_contract),
            _outputs(sender_contract, receiver_contract),
            _sync_async(sender_contract, receiver_contract),
            _errors(sender_contract, receiver_contract),
            _effects(sender_contract, receiver_contract),
            _capabilities(sender_contract, receiver_contract),
            _authorization(sender_contract, receiver_contract),
            _refinement(SemanticAspect.IDEMPOTENCE, sender_contract.idempotence, receiver_contract.idempotence, "idempotence"),
            _ordering(sender_contract, receiver_contract),
            _refinement(SemanticAspect.ATOMICITY, sender_contract.atomicity, receiver_contract.atomicity, "atomicity"),
            _refinement(SemanticAspect.CONSISTENCY, sender_contract.consistency, receiver_contract.consistency, "consistency"),
            _refinement(SemanticAspect.RESOURCE_BOUNDS, sender_contract.resource_bounds, receiver_contract.resource_bounds, "resource bounds"),
            _fallback(sender_contract, receiver_contract),
        ))
        return ProgramContractComparison(sender, receiver, tuple(clauses))

    def synthesize(
        self,
        trace: BrokenContractTrace,
        sender_expected: ExpectedProgramContract,
        receiver_expected: ExpectedProgramContract,
        receiver_observed: ObservedProgramContract | None = None,
    ) -> ProgramContractComparison:
        sender = self.sender_compiler.compile(trace, sender_expected)
        if (
            receiver_expected.symbol.repository_id != trace.roots.repository_id
            or receiver_expected.symbol.tree_id != trace.roots.tree_id
        ):
            raise SenderReceiverContractError(
                "receiver expectation must bind the trace repository and tree"
            )
        receiver = self.receiver_compiler.compile(receiver_expected, receiver_observed)
        return self.compare(sender, receiver)


__all__ = [
    "MAX_CLAUSES",
    "SenderReceiverContractError",
    "ClauseDisposition",
    "ClauseComparison",
    "SenderRequirement",
    "ReceiverGuarantee",
    "ProgramContractComparison",
    "ContractComparison",
    "CallRequirementContract",
    "SenderRequirementCompiler",
    "ReceiverGuaranteeCompiler",
    "SenderReceiverContractCompiler",
]
