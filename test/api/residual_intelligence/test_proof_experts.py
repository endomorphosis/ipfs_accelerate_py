from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.local_experts import (
    IndependentValidationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.proof_experts import (
    REASON_STALE_OBLIGATION,
    REASON_SUGGESTION_NOT_PROOF,
    PremiseRanking,
    ProofExpertAdapter,
    TacticCandidate,
)


OBLIGATION = "obligation:lemma-1"
SOURCE = "cid:source:1"
ENV = "env:prover:lean"


def ranking() -> PremiseRanking:
    return PremiseRanking(
        obligation_id=OBLIGATION,
        source_cid=SOURCE,
        environment_id=ENV,
        premise_ids=("premise:a", "premise:b"),
        lemma_ids=("lemma:1",),
        branch_ids=("branch:left",),
        counterexample_class="missing_hypothesis",
    )


def adapter(*, prover=None) -> ProofExpertAdapter:
    return ProofExpertAdapter(
        current_obligation_id=OBLIGATION,
        current_source_cid=SOURCE,
        current_environment_id=ENV,
        prover=prover,
    )


def test_prover_check_binds_obligation_and_never_labels_proof() -> None:
    def prover(payload):
        assert payload["obligation_id"] == OBLIGATION
        assert payload["source_cid"] == SOURCE
        assert payload["environment_id"] == ENV
        return {"checked": True, "accepted": True}

    receipt = adapter(prover=prover).nominate(
        ranking(),
        (TacticCandidate(obligation_id=OBLIGATION, tactic_id="tactic:intro"),),
        validation=IndependentValidationReceipt(
            validator_identity="validator:prover@1",
            accepted=True,
        ),
    )
    assert receipt.prover_checked is True
    assert receipt.prover_accepted is True
    assert receipt.candidate_only is True
    assert receipt.privacy_class is PrivacyClass.PROOF_WITNESS
    assert REASON_SUGGESTION_NOT_PROOF in receipt.reason_codes
    assert receipt.disposition is ExpertDisposition.ACCEPT
    assert "proof_accepted" not in receipt.to_dict()
    assert receipt.tactics[0].failed is False


def test_stale_obligation_and_missing_prover_fail_closed() -> None:
    stale = ranking()
    stale = PremiseRanking(
        obligation_id="obligation:old",
        source_cid=SOURCE,
        environment_id=ENV,
        premise_ids=("premise:a",),
    )
    blocked = adapter(prover=lambda _p: {"checked": True, "accepted": True}).nominate(
        stale, ()
    )
    assert blocked.disposition is ExpertDisposition.REJECT_INPUT
    assert REASON_STALE_OBLIGATION in blocked.reason_codes
    missing = adapter(prover=None).nominate(ranking(), ())
    assert missing.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    omitted = adapter(prover=lambda _p: {"checked": False}).nominate(ranking(), ())
    assert omitted.prover_checked is False


def test_failed_tactic_lineage_is_retained() -> None:
    receipt = adapter(prover=lambda _p: {"checked": True, "accepted": False}).nominate(
        ranking(),
        (TacticCandidate(obligation_id=OBLIGATION, tactic_id="tactic:auto"),),
    )
    assert receipt.tactics[0].failed is True
    assert receipt.tactics[0].lineage_id
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        PremiseRanking(
            obligation_id=OBLIGATION,
            source_cid=SOURCE,
            environment_id=ENV,
            premise_ids=("premise:a",),
            candidate_only=False,
        )
