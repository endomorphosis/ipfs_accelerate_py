"""Focused fail-closed coverage for contract-repair obligation lowering."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots, BrokenContractTrace, EvidenceReference, MemorySafetyDisposition,
    MemorySafetyFacet, RepairCandidate, RepairStrategy, SourceSpan, TraceDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.sender_receiver_contracts import (
    SenderReceiverContractCompiler,
)
from ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_capabilities import (
    ContractRepairCapability, ContractRepairCapabilityReport, ContractRepairCapabilityStatus,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    CapabilityMode, CapabilitySpec, ContractSourceKind, EffectKind, EffectPolarity,
    ErrorSpec, ExpectedProgramContract, InterfaceIdentity, Optionality, ParameterKind,
    ParameterSpec, ReturnSpec, SideEffectSpec, SourceReference, SymbolIdentity,
    SyncAsyncSpec, SyncMode, TypeConstructor, TypeShape,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_obligations import (
    AdapterMappings, AssumptionBinding, CallSlice, ContractRepairObligationCompiler,
    ContractRepairObligationError, FiniteMapping, IncompleteCallSliceError,
    LogicCapabilityBinding, ObligationContext, ObligationKind, StrategyEvidence,
    StrategyEvidenceKind, UnsupportedObligationError,
)


ROOTS = AuthorityRoots(
    repository_id="repository:test", forest_id="forest:test", tree_id="tree:test",
    graph_id="graph:test", index_id="index:test", model_id="model:test",
    config_id="config:test", translator_id="translator:test", toolchain_id="toolchain:test",
    policy_id="policy:test",
)


def ref(kind: str, artifact: str) -> EvidenceReference:
    return EvidenceReference(kind, artifact, producer_id="test")


def expected(name: str) -> ExpectedProgramContract:
    shape = TypeShape(TypeConstructor.STRING, name="str")
    return ExpectedProgramContract(
        symbol=SymbolIdentity("repository:test", "tree:test", f"pkg/{name}.py", name),
        interface=InterfaceIdentity("vfs", "tool", method="read"), policy_revision="policy:test",
        sources=(SourceReference(ContractSourceKind.REVIEWED_INTERFACE, "expected", f"contract:{name}"),),
        inputs=(ParameterSpec("path", shape, ParameterKind.POSITIONAL, Optionality.REQUIRED, position=0),),
        returns=ReturnSpec(shape), errors=(ErrorSpec("NotFound"),),
        side_effects=(SideEffectSpec(EffectKind.FILESYSTEM, EffectPolarity.ALLOWED),),
        capabilities=(CapabilitySpec("vfs.read", CapabilityMode.REQUIRED),),
        sync_async=SyncAsyncSpec(SyncMode.ASYNC, awaitable=True),
    )


def trace() -> BrokenContractTrace:
    return BrokenContractTrace(
        ROOTS, SourceSpan("pkg/caller.py", 0, 10, "blob:caller"), "symbol:caller",
        "old_read", TraceDisposition.LIKELY_REFACTOR,
        evidence_refs=(ref("trace", "trace:one"), ref("call_slice", "slice:one")),
    )


def comparison(trace_value: BrokenContractTrace):
    return SenderReceiverContractCompiler().synthesize(trace_value, expected("caller"), expected("receiver"))


def candidate(trace_value: BrokenContractTrace, strategy: RepairStrategy) -> RepairCandidate:
    return RepairCandidate(
        ROOTS, trace_value.content_id, strategy,
        SourceSpan("pkg/receiver.py", 0, 10, "blob:receiver"), (ref("candidate", "candidate:one"),),
    )


def memory() -> MemorySafetyFacet:
    return MemorySafetyFacet(
        ROOTS, SourceSpan("pkg/receiver.py", 0, 10, "blob:receiver"), "python",
        MemorySafetyDisposition.PROVED, proof_refs=(ref("proof", "memory:one"),),
    )


def capability() -> LogicCapabilityBinding:
    report = ContractRepairCapabilityReport(
        capabilities=(ContractRepairCapability(
            "datasets.logic_ir", ContractRepairCapabilityStatus.AVAILABLE,
            module_paths=("/tmp/logic_ir.py",), interface_version="logic-ir@1",
            supported_semantics=("ir",), reconstruction_compatible=True,
            details={"capability_revision": "logic:one"},
        ),),
        accelerator_module_paths=(), datasets_module_paths=("/tmp/logic_ir.py",),
        datasets_gitlink_revision="gitlink:one",
    )
    return LogicCapabilityBinding.from_report(report)


def context(*extra: StrategyEvidence) -> ObligationContext:
    return ObligationContext(
        CallSlice(ref("call_slice", "slice:one"), True),
        (AssumptionBinding(ref("reviewed_assumption", "assumption:one")),), capability(), extra,
    )


def strategy_evidence(*kinds: StrategyEvidenceKind) -> tuple[StrategyEvidence, ...]:
    return tuple(StrategyEvidence(kind, (ref(kind.value, f"evidence:{kind.value}"),)) for kind in kinds)


def test_substitution_lowers_all_contract_and_memory_facets_with_full_binding() -> None:
    item = trace()
    result = ContractRepairObligationCompiler().compile(
        item, comparison(item), candidate(item, RepairStrategy.RENAME_SUBSTITUTION), memory(),
        context(*strategy_evidence(StrategyEvidenceKind.IDENTITY_HISTORY, StrategyEvidenceKind.ROUTE_WIRING)),
    )

    kinds = {obligation.kind for obligation in result.obligations}
    assert {
        ObligationKind.CALLER_IMPLIES_RECEIVER_PRECONDITION,
        ObligationKind.RECEIVER_GUARANTEE_IMPLIES_CALLER_REQUIREMENT,
        ObligationKind.ERROR_COMPATIBILITY, ObligationKind.EFFECT_COMPATIBILITY,
        ObligationKind.CAPABILITY_COMPATIBILITY, ObligationKind.MEMORY_COMPATIBILITY,
        ObligationKind.REVERSE_REFINEMENT, ObligationKind.EQUIVALENCE_IDENTITY_HISTORY,
        ObligationKind.ROUTE_WIRING,
    } <= kinds
    for obligation in result.obligations:
        claim = obligation.claim
        assert claim.premise_ids and claim.source_ids and claim.assumption_ids
        assert (claim.tree_id, claim.translator_id, claim.toolchain_id, claim.policy_id) == (
            "tree:test", "translator:test", "toolchain:test", "policy:test",
        )
        assert obligation.code_obligation.metadata["claim_id"] == claim.content_id


def test_rename_requires_reverse_identity_and_route_evidence() -> None:
    item = trace()
    with pytest.raises(ContractRepairObligationError, match="identity_history"):
        ContractRepairObligationCompiler().compile(
            item, comparison(item), candidate(item, RepairStrategy.RENAME_SUBSTITUTION), memory(), context()
        )


def test_adapter_requires_explicit_total_finite_argument_result_and_error_maps() -> None:
    item = trace()
    evidence = strategy_evidence(
        StrategyEvidenceKind.ADAPTER_ARGUMENT, StrategyEvidenceKind.ADAPTER_RESULT,
        StrategyEvidenceKind.ADAPTER_ERROR, StrategyEvidenceKind.ADAPTER_EFFECT_CAPABILITY,
    )
    mappings = AdapterMappings(
        arguments=(FiniteMapping("path", "request.path", ref("map", "map:argument")),),
        results=(FiniteMapping("result", "response.body", ref("map", "map:result")),),
        errors=(FiniteMapping("NotFound", "Missing", ref("map", "map:error")),),
    )
    result = ContractRepairObligationCompiler().compile(
        item, comparison(item), candidate(item, RepairStrategy.ADAPTER), memory(), context(*evidence),
        adapter_mappings=mappings,
    )
    assert ObligationKind.ADAPTER_ARGUMENT_TOTALITY in {value.kind for value in result.obligations}
    assert ObligationKind.ADAPTER_ERROR_TOTALITY in {value.kind for value in result.obligations}

    with pytest.raises(ContractRepairObligationError, match="not total"):
        ContractRepairObligationCompiler().compile(
            item, comparison(item), candidate(item, RepairStrategy.ADAPTER), memory(), context(*evidence),
            adapter_mappings=replace(mappings, arguments=()),
        )


def test_placement_requires_every_admissibility_fact_and_exact_stub_contract() -> None:
    item = trace()
    all_evidence = strategy_evidence(
        StrategyEvidenceKind.OWNERSHIP, StrategyEvidenceKind.NO_OMITTED_COMPATIBLE_IMPLEMENTATION,
        StrategyEvidenceKind.DEPENDENCY_DAG, StrategyEvidenceKind.VISIBILITY_REGISTRATION,
        StrategyEvidenceKind.EXACT_STUB_CONTRACT,
    )
    result = ContractRepairObligationCompiler().compile(
        item, comparison(item), candidate(item, RepairStrategy.NEW_IMPLEMENTATION), memory(), context(*all_evidence),
    )
    assert {ObligationKind.PLACEMENT_OWNERSHIP, ObligationKind.PLACEMENT_EXACT_STUB_CONTRACT} <= {
        value.kind for value in result.obligations
    }
    with pytest.raises(ContractRepairObligationError, match="dependency_dag"):
        ContractRepairObligationCompiler().compile(
            item, comparison(item), candidate(item, RepairStrategy.NEW_IMPLEMENTATION), memory(),
            context(*all_evidence[:-3]),
        )


def test_partial_slices_unsupported_memory_and_missing_logic_capability_fail_closed() -> None:
    with pytest.raises(IncompleteCallSliceError):
        CallSlice(ref("call_slice", "slice:one"), False)

    item = trace()
    unsupported_memory = MemorySafetyFacet(
        ROOTS, SourceSpan("pkg/receiver.py", 0, 10, "blob:receiver"), "python",
        MemorySafetyDisposition.UNSUPPORTED, unsupported_refs=("memory:unsupported",),
    )
    with pytest.raises(UnsupportedObligationError, match="memory-safety"):
        ContractRepairObligationCompiler().compile(
            item, comparison(item), candidate(item, RepairStrategy.ADAPTER), unsupported_memory,
            context(*strategy_evidence(
                StrategyEvidenceKind.ADAPTER_ARGUMENT, StrategyEvidenceKind.ADAPTER_RESULT,
                StrategyEvidenceKind.ADAPTER_ERROR, StrategyEvidenceKind.ADAPTER_EFFECT_CAPABILITY,
            )), adapter_mappings=AdapterMappings(
                (FiniteMapping("path", "request.path", ref("map", "map:argument")),),
                (FiniteMapping("result", "response.body", ref("map", "map:result")),),
                (FiniteMapping("NotFound", "Missing", ref("map", "map:error")),),
            ),
        )
