"""Focused adversarial coverage for final repair-target admission."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    SourceSpan,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_reranker import (
    CandidateEligibilityDisposition,
    CandidateRank,
    RerankDisposition,
    RerankReceipt,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_target_admission import (
    AdmissionInvalidator,
    DecisionExpiry,
    RepairTargetAdmission,
    RepairTargetDecisionValidator,
    TargetRepositoryAuthority,
)

ROOTS = AuthorityRoots(
    repository_id="repository:test",
    forest_id="forest:test",
    tree_id="tree:test",
    graph_id="graph:test",
    index_id="index:test",
    model_id="model:test",
    config_id="config:test",
    translator_id="translator:test",
    toolchain_id="toolchain:test",
    policy_id="policy:test",
)


def ref(kind: str, artifact: str) -> EvidenceReference:
    return EvidenceReference(kind, artifact, producer_id="reviewer:test")


def candidate(
    name: str, strategy: RepairStrategy = RepairStrategy.NEW_IMPLEMENTATION
) -> RepairCandidate:
    return RepairCandidate(
        ROOTS,
        f"trace:{name}",
        strategy,
        SourceSpan(f"pkg/{name}.py", 4, 12, f"blob:{name}"),
        (ref("candidate", f"candidate:{name}"),),
    )


def receipt(
    candidates: tuple[RepairCandidate, ...],
    disposition: RerankDisposition = RerankDisposition.RANKED,
    selected: RepairCandidate | None = None,
    reasons: tuple[str, ...] = (),
) -> RerankReceipt:
    selected = selected or candidates[0]
    ranks = tuple(
        CandidateRank(
            item.content_id,
            CandidateEligibilityDisposition.ELIGIBLE,
            (100 if item is selected else 0, 0, 0, 0, 0, 0, 0),
            proof_receipt_ids=(f"proof:{item.content_id}",),
        )
        for item in candidates
    )
    return RerankReceipt(
        ROOTS,
        candidate_set_identity(candidates),
        "policy-receipt:test",
        ranks,
        disposition,
        selected_candidate_id=selected.content_id
        if disposition is RerankDisposition.RANKED
        else "",
        reason_codes=reasons,
    )


def authority(
    item: RepairCandidate, candidates: tuple[RepairCandidate, ...], **changes: object
) -> TargetRepositoryAuthority:
    values: dict[str, object] = {
        "roots": ROOTS,
        "candidate_set_id": candidate_set_identity(candidates),
        "candidate_id": item.content_id,
        "target_span": item.target_span,
        "permitted_read_spans": (item.target_span,),
        "permitted_write_spans": (item.target_span,),
        "evidence_refs": (ref("repository_authority", f"authority:{item.content_id}"),),
    }
    values.update(changes)
    return TargetRepositoryAuthority(**values)


def test_admits_one_exact_target_with_authority_derived_paths_spans_and_expiry() -> (
    None
):
    item = candidate("receiver")
    candidates = (item,)
    rerank = receipt(candidates)
    expiry = DecisionExpiry(100, 200)
    result = RepairTargetAdmission().admit(
        candidates, rerank, (authority(item, candidates),), expiry=expiry
    )

    assert result.decision.disposition is DecisionDisposition.ADMITTED
    assert result.decision.strategy is RepairStrategy.NEW_IMPLEMENTATION
    assert item.permitted_read_paths == item.candidate_write_paths == ()
    assert result.decision.permitted_read_paths == ("pkg/receiver.py",)
    assert result.decision.permitted_write_paths == ("pkg/receiver.py",)
    assert result.permitted_read_spans == (item.target_span,)
    assert result.permitted_write_spans == (item.target_span,)
    assert result.audit.decision_id == result.decision.content_id
    assert RepairTargetDecisionValidator().is_valid(
        result,
        roots=ROOTS,
        candidates=candidates,
        rerank_receipt=rerank,
        authorities=(authority(item, candidates),),
        now=150,
    )


def test_candidate_set_mutation_and_read_only_target_reject_without_writes() -> None:
    first, second = candidate("first"), candidate("second")
    candidates = (first, second)
    rerank = receipt(candidates, selected=first)
    expiry = DecisionExpiry(100, 200)

    stale = replace(rerank, candidate_set_id="candidate-set:mutated")
    decision = RepairTargetAdmission().decide(
        candidates, stale, (authority(first, candidates),), expiry=expiry
    )
    assert decision.disposition is DecisionDisposition.REJECTED
    assert decision.strategy is RepairStrategy.REJECT
    assert decision.write_paths == ()

    result = RepairTargetAdmission().admit(
        candidates,
        rerank,
        (authority(first, candidates, read_only=True),),
        expiry=expiry,
    )
    assert result.decision.disposition is DecisionDisposition.REJECTED
    assert result.decision.write_paths == ()

    result = RepairTargetAdmission().admit(
        candidates,
        rerank,
        (
            authority(
                first, candidates, target_exists=False, insertion_anchor_proved=True
            ),
        ),
        expiry=expiry,
    )
    assert result.decision.disposition is DecisionDisposition.REJECTED
    assert result.decision.write_paths == ()


def test_unrelated_target_authority_rejects_and_contextual_replay_detects_drift() -> None:
    item = candidate("receiver")
    candidates = (item,)
    rerank = receipt(candidates)
    expiry = DecisionExpiry(100, 200)
    repository_authority = authority(item, candidates)
    result = RepairTargetAdmission().admit(
        candidates, rerank, (repository_authority,), expiry=expiry
    )
    validator = RepairTargetDecisionValidator()

    unrelated = SourceSpan("pkg/unrelated.py", 1, 2, "blob:unrelated")
    unrelated_authority = TargetRepositoryAuthority(
        ROOTS,
        candidate_set_identity(candidates),
        item.content_id,
        unrelated,
        (unrelated,),
        (unrelated,),
        (ref("repository_authority", "authority:unrelated"),),
    )
    rejected = RepairTargetAdmission().admit(
        candidates, rerank, (unrelated_authority,), expiry=expiry
    )
    assert rejected.decision.disposition is DecisionDisposition.REJECTED
    assert rejected.decision.read_paths == rejected.decision.write_paths == ()
    invalid = validator.validate(
        result,
        roots=ROOTS,
        candidates=candidates,
        rerank_receipt=rerank,
        authorities=(unrelated_authority,),
        now=150,
    )
    assert AdmissionInvalidator.TARGET_MISSING in invalid
    assert AdmissionInvalidator.REPOSITORY_AUTHORITY_CHANGED in invalid

    changed_evidence = authority(
        item,
        candidates,
        evidence_refs=(ref("repository_authority", "authority:changed"),),
    )
    invalid = validator.validate(
        result,
        roots=ROOTS,
        candidates=candidates,
        rerank_receipt=rerank,
        authorities=(changed_evidence,),
        now=150,
    )
    assert AdmissionInvalidator.REPOSITORY_AUTHORITY_CHANGED in invalid


def test_tie_low_margin_and_runtime_drift_invalidate_or_abstain() -> None:
    first, second = candidate("first"), candidate("second")
    candidates = (first, second)
    expiry = DecisionExpiry(100, 200)
    tie = receipt(candidates, RerankDisposition.AMBIGUOUS, reasons=("rank_tie",))
    decision = RepairTargetAdmission().decide(candidates, tie, (), expiry=expiry)
    assert decision.disposition is DecisionDisposition.ABSTAINED
    assert decision.strategy is RepairStrategy.AMBIGUOUS
    assert decision.write_paths == ()

    rerank = receipt(candidates, selected=first)
    result = RepairTargetAdmission().admit(
        candidates, rerank, (authority(first, candidates),), expiry=expiry
    )
    validator = RepairTargetDecisionValidator()
    invalid = validator.validate(
        result,
        roots=replace(ROOTS, tree_id="tree:changed"),
        candidates=candidates,
        rerank_receipt=rerank,
        authorities=(authority(first, candidates),),
        now=200,
    )
    assert AdmissionInvalidator.ROOT_CHANGED in invalid
    assert AdmissionInvalidator.EXPIRED in invalid

    downgraded = replace(
        rerank,
        ranks=tuple(
            replace(row, proof_receipt_ids=("proof:downgraded",))
            for row in rerank.ranks
        ),
    )
    invalid = validator.validate(
        result,
        roots=ROOTS,
        candidates=candidates,
        rerank_receipt=downgraded,
        authorities=(authority(first, candidates),),
        now=150,
    )
    assert AdmissionInvalidator.PROOF_DOWNGRADE in invalid
