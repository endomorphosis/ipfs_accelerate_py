"""Focused contract tests for proof-gated repair task projection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    SourceSpan,
    TraceDisposition,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_reranker import (
    CandidateEligibilityDisposition,
    CandidateRank,
    RerankDisposition,
    RerankReceipt,
)
from ipfs_accelerate_py.agent_supervisor.analysis.sender_receiver_contracts import (
    SenderReceiverContractCompiler,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_repair_task_source import (
    CONTRACT_REPAIR_TASK_SOURCE_INTERFACE,
    ContractRepairTaskProjectionReason,
    ContractRepairTaskSource,
    deterministic_contract_repair_task_id,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_target_admission import (
    DecisionExpiry,
    RepairTargetAdmission,
    TargetRepositoryAuthority,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    ContractSourceKind,
    ExpectedProgramContract,
    InterfaceIdentity,
    Optionality,
    ParameterKind,
    ParameterSpec,
    ReturnSpec,
    SourceReference,
    SymbolIdentity,
    TypeConstructor,
    TypeShape,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_edit_packet import (
    materialize_contract_repair_edit_packet,
)


ROOTS = AuthorityRoots(
    repository_id="repository:test", forest_id="forest:test", tree_id="tree:test",
    graph_id="graph:test", index_id="index:test", model_id="model:test",
    config_id="config:test", translator_id="translator:test", toolchain_id="toolchain:test",
    policy_id="policy:test",
)


def ref(kind: str, artifact: str) -> EvidenceReference:
    return EvidenceReference(kind, artifact, producer_id="test")


def contract(name: str) -> ExpectedProgramContract:
    shape = TypeShape(TypeConstructor.STRING, name="str")
    return ExpectedProgramContract(
        symbol=SymbolIdentity("repository:test", "tree:test", f"pkg/{name}.py", name),
        interface=InterfaceIdentity("vfs", "tool", method="read"), policy_revision="policy:test",
        sources=(SourceReference(ContractSourceKind.REVIEWED_INTERFACE, "expected", f"contract:{name}"),),
        inputs=(ParameterSpec("path", shape, ParameterKind.POSITIONAL, Optionality.REQUIRED, position=0),),
        returns=ReturnSpec(shape),
    )


def packet():
    trace = BrokenContractTrace(
        ROOTS, SourceSpan("pkg/caller.py", 0, 10, "blob:caller"), "symbol:caller",
        "old_receiver", TraceDisposition.LIKELY_REFACTOR,
        evidence_refs=(ref("trace", "trace:one"),), proof_refs=(ref("proof", "trace-proof:one"),),
    )
    candidate = RepairCandidate(
        ROOTS, trace.content_id, RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan("pkg/receiver.py", 0, 10, "blob:receiver"),
        (ref("candidate", "candidate:one"),), proof_refs=(ref("proof", "candidate-proof:one"),),
    )
    candidates = (candidate,)
    receipt = RerankReceipt(
        ROOTS, candidate_set_identity(candidates), "rerank:test",
        (CandidateRank(candidate.content_id, CandidateEligibilityDisposition.ELIGIBLE,
                       (100, 0, 0, 0, 0, 0, 0), proof_receipt_ids=("proof:selected",)),),
        RerankDisposition.RANKED, selected_candidate_id=candidate.content_id,
    )
    authority = TargetRepositoryAuthority(
        ROOTS, candidate_set_identity(candidates), candidate.content_id, candidate.target_span,
        (candidate.target_span,), (candidate.target_span,), (ref("authority", "authority:one"),),
    )
    admission = RepairTargetAdmission().admit(
        candidates, receipt, (authority,), expiry=DecisionExpiry(100, 200)
    )
    compared = SenderReceiverContractCompiler().synthesize(trace, contract("caller"), contract("receiver"))
    return materialize_contract_repair_edit_packet(
        admission, trace, compared, roots=ROOTS, candidates=candidates, rerank_receipt=receipt,
        authorities=(authority,), now=150, post_edit_obligation_ids=("obligation:contract",),
        validation_commands=("python -m pytest -q test/api/test_receiver.py",),
        reproof_commands=("python -m repair_reproof obligation:contract",),
    )


def test_projection_is_deterministic_idempotent_and_exact_scope() -> None:
    edit = packet()
    source = ContractRepairTaskSource()
    first = source.project(edit, current_roots=ROOTS)
    second = source.project(edit, roots=ROOTS)

    assert first is second
    assert first.emitted
    assert first.packet_id == edit.packet_id
    assert first.decision_id == edit.decision_id
    assert first.tree_id == ROOTS.tree_id
    assert first.task_id == deterministic_contract_repair_task_id(edit.packet_id, edit.decision_id, ROOTS.tree_id)
    assert first.predicted_files == first.write_scope == edit.write_paths
    assert first.task_record is not None
    assert tuple(first.task_record.finding.outputs) == edit.write_paths
    assert tuple(first.task_record.finding.predicted_files) == edit.write_paths
    assert CONTRACT_REPAIR_TASK_SOURCE_INTERFACE in first.task_record.finding.interfaces
    assert first.projection_id == second.projection_id


def test_prompt_is_precise_and_provider_cannot_widen_outputs() -> None:
    edit = packet()
    projection = ContractRepairTaskSource().project(edit, provider_outputs=edit.write_paths)

    for value in (
        edit.sender_expected_contract_id, edit.receiver_expected_contract_id,
        edit.strategy.value, edit.target_span.path, edit.decision_id,
        edit.validation_commands[0], edit.reproof_commands[0],
    ):
        assert value in projection.prompt
    assert "Unsupported limits:" in projection.prompt
    assert "must not add, modify, rename" in projection.prompt

    rejected = ContractRepairTaskSource().project(edit, provider_outputs=("pkg/receiver.py", "pkg/widened.py"))
    assert rejected.reason is ContractRepairTaskProjectionReason.SCOPE_MISMATCH
    assert rejected.implementation_task is None


def test_stale_rejected_ambiguous_and_malformed_packets_emit_no_task() -> None:
    edit = packet()
    stale_roots = replace(ROOTS, tree_id="tree:changed")
    stale = ContractRepairTaskSource().project(edit, current_roots=stale_roots)
    assert stale.reason is ContractRepairTaskProjectionReason.STALE
    assert stale.implementation_task is None

    rejected_payload = deepcopy(edit.to_record())
    rejected_payload["strategy"] = "reject"
    rejected = ContractRepairTaskSource().project(rejected_payload)
    assert rejected.reason is ContractRepairTaskProjectionReason.REJECTED
    assert rejected.implementation_task is None

    ambiguous_payload = deepcopy(edit.to_record())
    ambiguous_payload["strategy"] = "ambiguous"
    ambiguous = ContractRepairTaskSource().project(ambiguous_payload)
    assert ambiguous.reason is ContractRepairTaskProjectionReason.AMBIGUOUS
    assert ambiguous.implementation_task is None


def test_duplicate_decision_or_finding_does_not_duplicate_the_task() -> None:
    edit = packet()
    source = ContractRepairTaskSource()
    emitted = source.project(edit)
    duplicate = source.project(edit)
    assert duplicate is emitted
    batch = source.project_many((edit, edit))
    assert batch == (emitted,)
