"""Adversarial coverage for proof-gated contract-repair reranking."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    SourceSpan,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_reranker import (
    CandidateEligibility,
    CandidateEligibilityDisposition,
    ContractRepairReranker,
    RANKING_ORDER,
    RankingEvidence,
    RankingSignal,
    RerankDisposition,
    RerankPolicy,
)
from ipfs_accelerate_py.agent_supervisor.planning.implementation_site_admissibility import (
    PlacementDecision,
    PlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_prover import (
    CandidateProofBundle,
    CandidateProofResult,
    ContractRepairProofDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


ROOTS = AuthorityRoots(
    repository_id="repository:test", forest_id="forest:test", tree_id="tree:test",
    graph_id="graph:test", index_id="index:test", model_id="model:test",
    config_id="config:test", translator_id="translator:test", toolchain_id="toolchain:test",
    policy_id="policy:test",
)


def ref(kind: str, artifact: str, producer: str = "reviewer:test") -> EvidenceReference:
    return EvidenceReference(kind, artifact, producer_id=producer)


def candidate(name: str) -> RepairCandidate:
    return RepairCandidate(
        ROOTS, f"trace:{name}", RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan(f"pkg/{name}.py", 0, 10, f"blob:{name}"),
        (ref("candidate", f"candidate:{name}", "retrieval:test"),),
    )


def result(item: RepairCandidate, name: str = "obligation:one") -> CandidateProofResult:
    code = CodeProofObligation(
        repository_id=ROOTS.repository_id, repository_tree_id=ROOTS.tree_id,
        ast_scope_ids=(item.target_span.artifact_id,), statement=name, premise_ids=("premise:one",),
        template_id="contract-repair/test", template_version="1", template_semantic_hash="hash:test",
        invariant_class="contract_repair", task_id="RPR-012",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    evidence = ProofEvidence(
        EvidenceKind.KERNEL_VERIFICATION, EvidenceAuthority.KERNEL, EvidenceVerdict.ACCEPTED,
        artifact_id=f"kernel:{item.content_id}", subject_id=code.obligation_id,
        verifier_id="kernel:test", independent=True,
    )
    receipt = ProofReceipt(
        obligation_id=code.obligation_id, plan_id="plan:test", attempt_id=f"attempt:{item.content_id}",
        repository_id=ROOTS.repository_id, repository_tree_id=ROOTS.tree_id,
        ast_scope_ids=code.ast_scope_ids, premise_ids=code.premise_ids,
        translator_id=ROOTS.translator_id, solver_id="hammer", kernel_id="kernel:test",
        toolchain_id=ROOTS.toolchain_id, policy_id=ROOTS.policy_id,
        resource_budget=ResourceBudget(), verdict=ProofVerdict.PROVED, evidence=(evidence,),
    )
    return CandidateProofResult(
        name, receipt, ContractRepairProofDisposition.PROVED,
        ("independent_reconstruction",), f"cache:{item.content_id}",
    )


def eligibility(item: RepairCandidate, candidates: tuple[RepairCandidate, ...], **changes: object) -> CandidateEligibility:
    proof = result(item)
    bundle = CandidateProofBundle(
        item.content_id, ROOTS.repository_id, ROOTS.tree_id, (proof,), "hammer", "test",
    )
    placement = PlacementDecision(
        PlacementDisposition.ADMITTED, candidate_set_identity(candidates),
        selected_candidate_id=item.content_id, target_path=item.target_span.path,
        evidence_refs=(ref("architecture", f"site:{item.content_id}"),),
        proof_receipt_ids=(proof.receipt.receipt_id,),
    )
    values: dict[str, object] = {
        "candidate": item,
        "proof_bundle": bundle,
        "placement_decision": placement,
        "expectation_roots": ROOTS,
        "expectation_refs": (ref("reviewed_spec", f"spec:{item.content_id}"),),
        "complete_supported_slice": True,
        "target_valid": True,
        "target_validity_refs": (ref("target_validity", f"target:{item.content_id}"),),
        "write_authorized": True,
        "write_authority_refs": (ref("write_authority", f"write:{item.content_id}"),),
        "mandatory_obligation_ids": (proof.obligation_id,),
        "ranking_evidence": tuple(
            RankingEvidence(signal, 100 if signal is RankingSignal.PROOF_COVERAGE else 0, (ref(signal.value, f"{signal.value}:{item.content_id}"),))
            for signal in RANKING_ORDER
        ),
    }
    values.update(changes)
    return CandidateEligibility(**values)


def test_proof_gates_precede_vector_and_lexical_scores() -> None:
    good, poisoned = candidate("good"), candidate("poisoned")
    candidates = (good, poisoned)
    bad_bundle = replace(
        eligibility(poisoned, candidates).proof_bundle,
        results=(replace(result(poisoned), disposition=ContractRepairProofDisposition.NON_CONCLUSIVE),),
    )
    poison = eligibility(
        poisoned, candidates, proof_bundle=bad_bundle,
        ranking_evidence=(
            RankingEvidence(RankingSignal.LEXICAL, 1_000_000, (ref("lexical", "lexical:poison"),)),
            RankingEvidence(RankingSignal.VECTOR, 1_000_000, (ref("vector", "vector:poison"),)),
        ),
    )
    receipt = ContractRepairReranker().rank((eligibility(good, candidates), poison), roots=ROOTS)

    ranks = {rank.candidate_id: rank for rank in receipt.ranks}
    assert receipt.disposition is RerankDisposition.RANKED
    assert receipt.selected_candidate_id == good.content_id
    assert ranks[poisoned.content_id].disposition is CandidateEligibilityDisposition.INELIGIBLE
    assert "mandatory_proof_not_reconstructed" in ranks[poisoned.content_id].reason_codes
    assert receipt.write_paths == ()


def test_every_hard_gate_rejects_and_missing_signals_stay_zero() -> None:
    item = candidate("item")
    cases = (
        {"complete_supported_slice": False},
        {"target_valid": False},
        {"write_authorized": False},
        {"expectation_refs": (ref("candidate", "self", "retrieval:test"),)},
        {"placement_decision": PlacementDecision(PlacementDisposition.ABSTAINED, candidate_set_identity((item,)))},
    )
    for changes in cases:
        receipt = ContractRepairReranker().rank((eligibility(item, (item,), **changes),), roots=ROOTS)
        assert receipt.disposition is RerankDisposition.ABSTAINED
        assert receipt.ranks[0].score_vector == (0,) * len(RANKING_ORDER)


def test_normative_order_margin_tie_and_policy_receipt_are_replayable() -> None:
    first, second = candidate("first"), candidate("second")
    candidates = (first, second)
    first_evidence = (
        RankingEvidence(RankingSignal.PROOF_COVERAGE, 100, (ref("proof_coverage", "proof:first"),)),
        RankingEvidence(RankingSignal.VECTOR, 1_000_000, (ref("vector", "vector:first"),)),
    )
    second_evidence = (
        RankingEvidence(RankingSignal.PROOF_COVERAGE, 101, (ref("proof_coverage", "proof:second"),)),
    )
    policy = RerankPolicy(ROOTS.policy_id, minimum_margin=2)
    receipt = ContractRepairReranker(policy).rerank(
        (eligibility(first, candidates, ranking_evidence=first_evidence), eligibility(second, candidates, ranking_evidence=second_evidence)),
        roots=ROOTS,
    )
    assert receipt.disposition is RerankDisposition.AMBIGUOUS
    assert receipt.reason_codes == ("insufficient_rank_margin",)
    assert receipt.policy_receipt_id == policy.receipt_id

    tied = ContractRepairReranker().rank(
        (eligibility(first, candidates, ranking_evidence=()), eligibility(second, candidates, ranking_evidence=())),
        roots=ROOTS,
    )
    assert tied.disposition is RerankDisposition.AMBIGUOUS
    assert tied.reason_codes == ("rank_tie",)
    assert tied.selected_candidate_id == ""
