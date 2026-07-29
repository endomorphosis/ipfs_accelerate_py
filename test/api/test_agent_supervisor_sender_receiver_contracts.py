"""Conformance tests for independently sourced sender/receiver synthesis."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    EvidenceReference,
    SourceSpan,
    TraceDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.sender_receiver_contracts import (
    ClauseDisposition,
    ReceiverGuaranteeCompiler,
    SenderReceiverContractCompiler,
    SenderReceiverContractError,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    AtomicityMode,
    AtomicitySpec,
    AuthorizationMode,
    AuthorizationSpec,
    CapabilityMode,
    CapabilitySpec,
    ConsistencyMode,
    ConsistencySpec,
    ContractConflict,
    ContractSourceKind,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    IdempotenceMode,
    IdempotenceSpec,
    InterfaceIdentity,
    Optionality,
    OrderingMode,
    OrderingSpec,
    ParameterKind,
    ParameterSpec,
    ResourceBounds,
    ReturnSpec,
    SemanticAspect,
    SideEffectSpec,
    SourceReference,
    SymbolIdentity,
    SyncAsyncSpec,
    SyncMode,
    TypeConstructor,
    TypeShape,
)


ROOTS = AuthorityRoots(
    repository_id="repository:test", forest_id="forest:test", tree_id="tree:test",
    graph_id="graph:test", index_id="index:test", model_id="model:test",
    config_id="config:test", translator_id="translator:test",
    toolchain_id="toolchain:test", policy_id="policy:test",
)


def source(kind: ContractSourceKind = ContractSourceKind.REVIEWED_INTERFACE) -> SourceReference:
    return SourceReference(kind, "expected", f"artifact:{kind.value}", locator="surface:read")


def shape(name: str, *, nullable: bool = False) -> TypeShape:
    constructor = TypeConstructor.STRING if name == "str" else TypeConstructor.BYTES
    return TypeShape(constructor, name=name, nullable=nullable)


def contract(
    *,
    symbol_name: str,
    input_shape: TypeShape | None = None,
    return_shape: TypeShape | None = None,
    errors: tuple[ErrorSpec, ...] = (),
    side_effects: tuple[SideEffectSpec, ...] = (),
    capabilities: tuple[CapabilitySpec, ...] = (),
    authorization: AuthorizationSpec | None = None,
    sync_async: SyncAsyncSpec | None = None,
    idempotence: IdempotenceSpec | None = None,
    ordering: OrderingSpec | None = None,
    atomicity: AtomicitySpec | None = None,
    consistency: ConsistencySpec | None = None,
    resource_bounds: ResourceBounds | None = None,
    conflicts: tuple[ContractConflict, ...] = (),
) -> ExpectedProgramContract:
    return ExpectedProgramContract(
        symbol=SymbolIdentity("repository:test", "tree:test", f"pkg/{symbol_name}.py", symbol_name),
        interface=InterfaceIdentity("vfs", "tool", method="read"),
        policy_revision="policy:test",
        sources=(source(),),
        inputs=(ParameterSpec("path", input_shape or shape("str"), ParameterKind.POSITIONAL, Optionality.REQUIRED, position=0),),
        returns=ReturnSpec(return_shape or shape("bytes")),
        errors=errors,
        side_effects=side_effects,
        capabilities=capabilities,
        authorization=authorization,
        sync_async=sync_async or SyncAsyncSpec(SyncMode.ASYNC, awaitable=True),
        idempotence=idempotence,
        ordering=ordering,
        atomicity=atomicity,
        consistency=consistency,
        resource_bounds=resource_bounds,
        conflicts=conflicts,
    )


def trace() -> BrokenContractTrace:
    return BrokenContractTrace(
        ROOTS, SourceSpan("pkg/caller.py", 0, 12, "blob:caller"), "symbol:caller",
        "old_read", TraceDisposition.LIKELY_REFACTOR,
        evidence_refs=(EvidenceReference("trace", "artifact:trace", "call:read"),),
    )


def complete_sender() -> ExpectedProgramContract:
    return contract(
        symbol_name="caller",
        errors=(ErrorSpec("NotFound"),),
        side_effects=(SideEffectSpec(EffectKind.FILESYSTEM, EffectPolarity.ALLOWED),),
        capabilities=(CapabilitySpec("vfs.read", CapabilityMode.REQUIRED),),
        authorization=AuthorizationSpec(AuthorizationMode.CAPABILITY, scopes=("vfs.read",)),
        idempotence=IdempotenceSpec(IdempotenceMode.IDEMPOTENT),
        ordering=OrderingSpec(OrderingMode.SEQUENTIAL),
        atomicity=AtomicitySpec(AtomicityMode.ATOMIC),
        consistency=ConsistencySpec(ConsistencyMode.STRONG),
        resource_bounds=ResourceBounds(max_wall_time_ms=100, max_memory_bytes=2048),
    )


def complete_receiver() -> ExpectedProgramContract:
    return contract(
        symbol_name="receiver",
        errors=(ErrorSpec("NotFound"),),
        side_effects=(SideEffectSpec(EffectKind.FILESYSTEM, EffectPolarity.ALLOWED),),
        capabilities=(CapabilitySpec("vfs.read", CapabilityMode.REQUIRED),),
        authorization=AuthorizationSpec(AuthorizationMode.CAPABILITY, scopes=("vfs.read",)),
        idempotence=IdempotenceSpec(IdempotenceMode.PURE),
        ordering=OrderingSpec(OrderingMode.SEQUENTIAL),
        atomicity=AtomicitySpec(AtomicityMode.TRANSACTIONAL),
        consistency=ConsistencySpec(ConsistencyMode.STRONG),
        resource_bounds=ResourceBounds(max_wall_time_ms=10, max_memory_bytes=1024),
    )


def test_synthesis_binds_reviewed_sender_evidence_and_all_modeled_facets() -> None:
    result = SenderReceiverContractCompiler().synthesize(trace(), complete_sender(), complete_receiver())

    assert result.compatible
    assert result.sender.call_requirement.trace_id == trace().content_id
    assert result.sender.call_requirement.requirement_refs[0].kind == "reviewed_interface"
    assert result.call_requirement.receiver_contract_refs[0].kind == "reviewed_interface"
    assert {item.aspect for item in result.clauses} >= {
        SemanticAspect.INPUTS, SemanticAspect.OUTPUTS, SemanticAspect.ERRORS,
        SemanticAspect.SIDE_EFFECTS, SemanticAspect.CAPABILITIES,
        SemanticAspect.AUTHORIZATION, SemanticAspect.ORDERING,
        SemanticAspect.ATOMICITY, SemanticAspect.CONSISTENCY,
        SemanticAspect.RESOURCE_BOUNDS,
    }


def test_inputs_are_contravariant_and_outputs_are_covariant() -> None:
    sender = complete_sender()
    wider_input = TypeShape(TypeConstructor.ANY, name="any")
    result = SenderReceiverContractCompiler().synthesize(
        trace(), sender,
        contract(symbol_name="receiver", input_shape=wider_input, return_shape=shape("bytes"),
                 errors=sender.errors, side_effects=sender.side_effects, capabilities=sender.capabilities,
                 authorization=sender.authorization, idempotence=sender.idempotence, ordering=sender.ordering,
                 atomicity=sender.atomicity, consistency=sender.consistency, resource_bounds=sender.resource_bounds),
    )
    assert result.compatible

    narrow_receiver = replace(complete_receiver(), inputs=(ParameterSpec("path", shape("bytes"), position=0),))
    result = SenderReceiverContractCompiler().synthesize(trace(), sender, narrow_receiver)
    input_clause = next(item for item in result.clauses if item.aspect is SemanticAspect.INPUTS)
    assert input_clause.disposition is ClauseDisposition.VIOLATED

    bad_output = replace(complete_receiver(), returns=ReturnSpec(shape("str")))
    result = SenderReceiverContractCompiler().synthesize(trace(), sender, bad_output)
    output_clause = next(item for item in result.clauses if item.aspect is SemanticAspect.OUTPUTS)
    assert output_clause.disposition is ClauseDisposition.VIOLATED


def test_unhandled_effect_capability_and_resource_drift_fail_closed() -> None:
    receiver = replace(
        complete_receiver(),
        errors=(ErrorSpec("PermissionDenied"),),
        side_effects=(SideEffectSpec(EffectKind.NETWORK, EffectPolarity.ALLOWED),),
        capabilities=(CapabilitySpec("network", CapabilityMode.REQUIRED),),
        resource_bounds=ResourceBounds(max_wall_time_ms=101, max_memory_bytes=4096),
    )
    result = SenderReceiverContractCompiler().synthesize(trace(), complete_sender(), receiver)
    failed = {item.aspect for item in result.failed_clauses}
    assert {SemanticAspect.ERRORS, SemanticAspect.SIDE_EFFECTS, SemanticAspect.CAPABILITIES, SemanticAspect.RESOURCE_BOUNDS} <= failed
    assert not result.compatible


def test_conflicts_and_unsupported_semantics_remain_explicit() -> None:
    conflict = ContractConflict(
        "precedence_collision", SemanticAspect.OUTPUTS, "source:left", "source:right", "same-rank schemas disagree"
    )
    sender = replace(complete_sender(), conflicts=(conflict,))
    receiver = replace(complete_receiver(), unsupported=())
    # The receiver has no fallback requirement; set output unsupported to model
    # a dynamic result rather than allowing it through as an omitted detail.
    from ipfs_accelerate_py.agent_supervisor.program_contracts import UnsupportedSemantics
    receiver = replace(receiver, unsupported=(UnsupportedSemantics(SemanticAspect.OUTPUTS, "reflection"),))
    result = SenderReceiverContractCompiler().synthesize(trace(), sender, receiver)
    outcomes = {item.aspect: item.disposition for item in result.clauses}
    assert outcomes[SemanticAspect.SOURCE_PRECEDENCE] is ClauseDisposition.CONFLICT
    assert outcomes[SemanticAspect.OUTPUTS] is ClauseDisposition.UNSUPPORTED
    assert not result.compatible


def test_observation_cannot_be_promoted_to_receiver_guarantee() -> None:
    # ProgramContract@1 prevents constructing an Expected contract from an
    # implementation observation.  This adapter also rejects an observation
    # attached to a different candidate instead of treating it as authority.
    expected = complete_receiver()
    from ipfs_accelerate_py.agent_supervisor.program_contracts import ObservedProgramContract
    observation = ObservedProgramContract(
        symbol=SymbolIdentity("repository:test", "tree:test", "pkg/other.py", "other"),
        interface=expected.interface,
        policy_revision="policy:test", repository_observation_id="observation:one",
        sources=(SourceReference(ContractSourceKind.IMPLEMENTATION_OBSERVATION, "observed", "artifact:impl"),),
    )
    with pytest.raises(SenderReceiverContractError, match="same subject"):
        ReceiverGuaranteeCompiler().compile(expected, observation)
