"""Focused fail-closed coverage for ``ContractRepairEditPacket@2``."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

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
    CONTRACT_REPAIR_EDIT_PACKET_INTERFACE,
    ContractRepairEditPacket,
    ContractRepairEditPacketError,
    ExpansionHandle,
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


def trace() -> BrokenContractTrace:
    return BrokenContractTrace(
        ROOTS, SourceSpan("pkg/caller.py", 0, 10, "blob:caller"), "symbol:caller",
        "old_receiver", TraceDisposition.LIKELY_REFACTOR,
        evidence_refs=(ref("trace", "trace:one"), ref("counterexample", "counterexample:one")),
        proof_refs=(ref("proof", "trace-proof:one"),),
    )


def contract(name: str) -> ExpectedProgramContract:
    shape = TypeShape(TypeConstructor.STRING, name="str")
    return ExpectedProgramContract(
        symbol=SymbolIdentity("repository:test", "tree:test", f"pkg/{name}.py", name),
        interface=InterfaceIdentity("vfs", "tool", method="read"), policy_revision="policy:test",
        sources=(SourceReference(ContractSourceKind.REVIEWED_INTERFACE, "expected", f"contract:{name}"),),
        inputs=(ParameterSpec("path", shape, ParameterKind.POSITIONAL, Optionality.REQUIRED, position=0),),
        returns=ReturnSpec(shape),
    )


def comparison(value: BrokenContractTrace):
    return SenderReceiverContractCompiler().synthesize(value, contract("caller"), contract("receiver"))


def candidate(value: BrokenContractTrace) -> RepairCandidate:
    return RepairCandidate(
        ROOTS, value.content_id, RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan("pkg/receiver.py", 0, 10, "blob:receiver"),
        (ref("candidate", "candidate:one"),), proof_refs=(ref("proof", "candidate-proof:one"),),
    )


def receipt(items: tuple[RepairCandidate, ...]) -> RerankReceipt:
    selected = items[0]
    return RerankReceipt(
        ROOTS, candidate_set_identity(items), "rerank:test",
        tuple(CandidateRank(item.content_id, CandidateEligibilityDisposition.ELIGIBLE,
                            (100 if item is selected else 0, 0, 0, 0, 0, 0, 0),
                            proof_receipt_ids=(f"proof:{item.content_id}",)) for item in items),
        RerankDisposition.RANKED, selected_candidate_id=selected.content_id,
    )


def authority(item: RepairCandidate, items: tuple[RepairCandidate, ...]) -> TargetRepositoryAuthority:
    return TargetRepositoryAuthority(
        ROOTS, candidate_set_identity(items), item.content_id, item.target_span,
        (item.target_span,), (item.target_span,), (ref("repository_authority", "authority:one"),),
    )


def admitted():
    value = trace()
    item = candidate(value)
    items = (item,)
    ranking = receipt(items)
    repository_authority = authority(item, items)
    result = RepairTargetAdmission().admit(items, ranking, (repository_authority,), expiry=DecisionExpiry(100, 200))
    return result, value, comparison(value), items, ranking, repository_authority


def packet(**changes: object) -> ContractRepairEditPacket:
    result, value, compared, items, ranking, repository_authority = admitted()
    arguments: dict[str, object] = {
        "roots": ROOTS, "candidates": items, "rerank_receipt": ranking,
        "authorities": (repository_authority,), "now": 150,
        "post_edit_obligation_ids": ("obligation:caller-implies-receiver",),
        "validation_commands": ("python -m pytest -q test/api/test_receiver.py",),
        "reproof_commands": ("python -m repair_reproof obligation:caller-implies-receiver",),
        "counterexample_refs": (value.evidence_refs[1],),
        "expansion_handles": (ExpansionHandle("counterexample", "counterexample_slice", value.evidence_refs[1].content_id),),
    }
    arguments.update(changes)
    return materialize_contract_repair_edit_packet(result, value, compared, **arguments)


def test_materializes_current_admitted_decision_with_exact_scope_and_compact_contract() -> None:
    result, value, compared, *_ = admitted()
    edit = packet()

    assert edit.interface == CONTRACT_REPAIR_EDIT_PACKET_INTERFACE
    assert edit.decision_id == result.decision.content_id
    assert edit.trace_id == value.content_id
    assert edit.write_paths == result.decision.permitted_write_paths == ("pkg/receiver.py",)
    assert edit.read_paths == result.decision.permitted_read_paths == ("pkg/receiver.py",)
    assert edit.target_span.path == "pkg/receiver.py"
    assert edit.sender_expected_contract_id == compared.sender.contract.content_id
    assert edit.receiver_expected_contract_id == compared.receiver.contract.content_id
    assert edit.index_refs == (ROOTS.index_id,)
    assert edit.proof_refs == result.decision.proof_refs
    assert edit.packet_id == edit.content_id
    assert ContractRepairEditPacket.from_dict(edit.to_record()) == edit


def test_stale_or_bare_decisions_do_not_materialize() -> None:
    result, value, compared, items, ranking, repository_authority = admitted()
    common = {
        "roots": ROOTS, "candidates": items, "rerank_receipt": ranking,
        "authorities": (repository_authority,), "post_edit_obligation_ids": ("obligation:x",),
        "validation_commands": ("pytest -q",), "reproof_commands": ("reproof obligation:x",),
    }
    with pytest.raises(ContractRepairEditPacketError, match="current"):
        materialize_contract_repair_edit_packet(result, value, compared, now=200, **common)
    with pytest.raises(ContractRepairEditPacketError, match="AdmissionResult"):
        materialize_contract_repair_edit_packet(result.decision, value, compared, now=150, **common)  # type: ignore[arg-type]

    ambiguous_ranking = replace(
        ranking, disposition=RerankDisposition.AMBIGUOUS, selected_candidate_id="",
        reason_codes=("rank_tie",),
    )
    ambiguous = RepairTargetAdmission().admit(
        items, ambiguous_ranking, (), expiry=DecisionExpiry(100, 200)
    )
    with pytest.raises(ContractRepairEditPacketError, match="current and admitted"):
        materialize_contract_repair_edit_packet(ambiguous, value, compared, now=150, **common)


def test_non_selected_evidence_and_handles_cannot_expand_packet_scope() -> None:
    with pytest.raises(ContractRepairEditPacketError, match="counterexample refs"):
        packet(counterexample_refs=(ref("counterexample", "other-candidate"),))
    with pytest.raises(ContractRepairEditPacketError, match="packet-bound evidence"):
        packet(expansion_handles=(ExpansionHandle("bad", "proof_receipt", "proof:unbound"),))
    with pytest.raises(ContractRepairEditPacketError, match="read scope"):
        packet(expansion_handles=(ExpansionHandle("bad", "trace_slice", "index:test", ("pkg/not-selected.py",)),))


def test_packet_rejects_forged_identity_and_embedded_bodies() -> None:
    edit = packet()
    forged = deepcopy(edit.to_record())
    forged["content_id"] = "baguqeerapiforged"
    with pytest.raises(ContractRepairEditPacketError, match="identity is forged"):
        ContractRepairEditPacket.from_dict(forged)

    body = deepcopy(edit.to_dict())
    body["proof_body"] = "by exact unsafe"
    with pytest.raises(ContractRepairEditPacketError, match="unsupported"):
        ContractRepairEditPacket.from_dict(body)

    handle = deepcopy(edit.to_dict())
    handle["expansion_handles"][0]["body_embedded"] = True
    with pytest.raises(ContractRepairEditPacketError, match="cannot embed"):
        ContractRepairEditPacket.from_dict(handle)


def test_packet_rejects_unsupported_limit_downgrade() -> None:
    edit = packet()
    with pytest.raises(ContractRepairEditPacketError, match="unsupported limits"):
        ContractRepairEditPacket(
            **{**edit.__dict__, "unsupported_clause_ids": ("outputs",)}
        )
