"""DCR-062: admit a uniquely selected candidate from a finite portfolio.

Acceptance:
* Selected candidate is uniquely admitted.
* Ties and unknowns abstain.
* All candidates bind current evidence and exact operator CIDs.
* Rejection/failure reasons are complete for non-selected members.
* Runtime model calls remain 0; write authority is never granted.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CANDIDATE_ADMISSION_INTERFACE,
    DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
    REPAIR_CANDIDATE_INTERFACE,
    AdmissionDisposition,
    ApplicabilityStatus,
    CandidateAdmission,
    CandidateAdmissionError,
    CandidateAdmissionReason,
    CandidateEligibility,
    CandidateFacts,
    CandidateScoreTerms,
    IrAttachmentStatus,
    RepairCandidate,
    admit_candidate_portfolio,
    build_deterministic_candidate_portfolio,
    evaluate_candidate_eligibility,
)
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    CANDIDATE_ADMISSION_INTERFACE as PLANNER_ADMISSION_INTERFACE,
    CandidateAdmission as PlannerCandidateAdmission,
    admit_candidate_portfolio as planner_admit,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _evidence(tag: str = "admit") -> str:
    return content_identity({"role": "current_evidence", "tag": tag})


def _nom(
    kind: str,
    *,
    applicability: int,
    risk: int = 100_000,
    edit_size: int = 1,
    resource_cost: int = 100,
    validation: int = 700_000,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "operator_kind": kind,
        "operator_args": {"symbol": kind, "path": "pkg/x.py"},
        "write_paths": ("pkg/x.py",),
        "proved_applicability": applicability,
        "risk": risk,
        "edit_size": edit_size,
        "resource_cost": resource_cost,
        "validation_strength": validation,
        "applicability_status": ApplicabilityStatus.PROVED.value,
        "ir_attachment_status": IrAttachmentStatus.ATTACHED.value,
    }
    payload.update(extra)
    return payload


def test_admission_interface_is_stable() -> None:
    assert CANDIDATE_ADMISSION_INTERFACE == "CandidateAdmission@1"
    assert PLANNER_ADMISSION_INTERFACE == "CandidateAdmission@1"
    assert REPAIR_CANDIDATE_INTERFACE == "RepairCandidate@1"
    assert PlannerCandidateAdmission is CandidateAdmission


def test_unique_winner_is_admitted() -> None:
    evidence = _evidence("unique")
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nom("add_import", applicability=600_000, risk=50_000, resource_cost=200),
            _nom(
                "add_registration",
                applicability=950_000,
                risk=20_000,
                edit_size=1,
                resource_cost=40,
                validation=900_000,
            ),
            _nom("add_export", applicability=700_000, risk=30_000, resource_cost=80),
        ),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:unique",
    )
    admission = admit_candidate_portfolio(portfolio)
    assert isinstance(admission, CandidateAdmission)
    assert admission.disposition is AdmissionDisposition.SELECTED
    assert admission.ok is True
    assert admission.selected_candidate_cid
    assert admission.runtime_model_calls == 0
    assert admission.grants_write_authority is False
    assert admission.proposal_only is True
    assert CandidateAdmissionReason.UNIQUE_WINNER.value in admission.reason_codes

    winner = next(
        c
        for c in portfolio.candidates
        if c.candidate_cid == admission.selected_candidate_cid
    )
    assert winner.facts.operator_kind == "add_registration"
    assert winner.facts.current_evidence_cid == evidence
    assert winner.operator_cid

    # Non-selected eligible candidates carry explicit rejection reasons.
    assert len(admission.ranked_eligible_cids) == 3
    for cid in admission.ranked_eligible_cids:
        if cid != admission.selected_candidate_cid:
            assert cid in admission.rejected
            assert admission.rejected[cid]

    subset = admission.evidence_subset()
    assert subset["evidence_id"] == DCR_CANDIDATE_PORTFOLIO_EVIDENCE
    assert subset["selected_candidate_cid"] == admission.selected_candidate_cid
    assert subset["disposition"] == "selected"


def test_score_tie_abstains() -> None:
    evidence = _evidence("tie")
    # Identical score terms → unique admission is impossible.
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nom(
                "add_registration",
                applicability=800_000,
                risk=50_000,
                edit_size=2,
                resource_cost=100,
                validation=700_000,
            ),
            _nom(
                "add_import",
                applicability=800_000,
                risk=50_000,
                edit_size=2,
                resource_cost=100,
                validation=700_000,
            ),
        ),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:tie",
    )
    # Confirm both eligible with identical rank keys.
    eligible = portfolio.eligible()
    assert len(eligible) == 2
    assert eligible[0].score_terms.rank_key() == eligible[1].score_terms.rank_key()

    admission = admit_candidate_portfolio(portfolio)
    assert admission.disposition is AdmissionDisposition.ABSTAIN
    assert admission.selected_candidate_cid == ""
    assert admission.ok is False
    assert CandidateAdmissionReason.TIE_ABSTAIN.value in admission.reason_codes
    # Both tied candidates recorded.
    assert len(admission.ranked_eligible_cids) == 2
    for cid in admission.ranked_eligible_cids:
        assert cid in admission.rejected


def test_unknown_applicability_abstains() -> None:
    evidence = _evidence("unknown")
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nom(
                "add_registration",
                applicability=0,
                applicability_status=ApplicabilityStatus.UNKNOWN.value,
            ),
        ),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:unknown",
    )
    candidate = portfolio.candidates[0]
    assert candidate.eligibility is CandidateEligibility.UNKNOWN
    admission = admit_candidate_portfolio(portfolio)
    assert admission.disposition is AdmissionDisposition.ABSTAIN
    assert admission.selected_candidate_cid == ""
    assert CandidateAdmissionReason.UNKNOWN_ABSTAIN.value in admission.reason_codes
    assert candidate.candidate_cid in admission.rejected


def test_all_refuted_rejects() -> None:
    evidence = _evidence("refute")
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nom(
                "add_registration",
                applicability=0,
                applicability_status=ApplicabilityStatus.REFUTED.value,
            ),
            _nom(
                "add_import",
                applicability=0,
                applicability_status=ApplicabilityStatus.REFUTED.value,
                ir_attachment_status=IrAttachmentStatus.FAILED.value,
            ),
        ),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:refute",
    )
    admission = admit_candidate_portfolio(portfolio)
    assert admission.disposition is AdmissionDisposition.REJECT
    assert admission.selected_candidate_cid == ""
    assert CandidateAdmissionReason.ALL_REFUTED.value in admission.reason_codes
    assert len(admission.rejected) == 2


def test_missing_proof_receipt_is_unknown_not_admitted() -> None:
    evidence = _evidence("proof")
    # Force empty proof via evaluate helper.
    facts = CandidateFacts(
        current_evidence_cid=evidence,
        operator_cid="bafyoperatorcidexample000000000000000000000000000000000001",
        operator_id="repair-operator:add_registration@2",
        operator_kind="add_registration",
        operator_args={"symbol": "x"},
        proof_receipt_cid="",
        write_paths=("pkg/x.py",),
        applicability_status=ApplicabilityStatus.PROVED,
        ir_attachment_status=IrAttachmentStatus.ATTACHED,
    )
    score = CandidateScoreTerms(
        proved_applicability=900_000,
        risk=10,
        edit_size=1,
        resource_cost=10,
        validation_strength=900_000,
    )
    eligibility, reasons = evaluate_candidate_eligibility(
        facts, score, current_evidence_cid=evidence
    )
    assert eligibility is CandidateEligibility.UNKNOWN
    assert CandidateAdmissionReason.MISSING_PROOF_RECEIPT.value in reasons


def test_stale_evidence_blocks_eligibility() -> None:
    facts = CandidateFacts(
        current_evidence_cid=_evidence("old"),
        operator_cid="bafyoperatorcidexample000000000000000000000000000000000002",
        operator_id="repair-operator:add_registration@2",
        operator_kind="add_registration",
        operator_args={"symbol": "x"},
        proof_receipt_cid="proof:ok",
        write_paths=("pkg/x.py",),
        applicability_status=ApplicabilityStatus.PROVED,
        ir_attachment_status=IrAttachmentStatus.ATTACHED,
    )
    score = CandidateScoreTerms(
        proved_applicability=900_000,
        risk=10,
        edit_size=1,
        resource_cost=10,
        validation_strength=900_000,
    )
    eligibility, reasons = evaluate_candidate_eligibility(
        facts, score, current_evidence_cid=_evidence("new")
    )
    assert eligibility is not CandidateEligibility.ELIGIBLE
    assert CandidateAdmissionReason.STALE_EVIDENCE.value in reasons


def test_admission_round_trip_dict() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (_nom("add_registration", applicability=900_000, risk=5),),
        current_evidence_cid=_evidence("roundtrip"),
    )
    admission = admit_candidate_portfolio(portfolio)
    rebuilt = CandidateAdmission.from_dict(admission.to_dict())
    assert rebuilt.admission_cid == admission.admission_cid
    assert rebuilt.selected_candidate_cid == admission.selected_candidate_cid
    assert rebuilt.disposition is AdmissionDisposition.SELECTED


def test_selected_without_unique_winner_reason_fails_closed() -> None:
    with pytest.raises(CandidateAdmissionError, match="unique_winner"):
        CandidateAdmission(
            portfolio_cid="bafyportfolio",
            disposition=AdmissionDisposition.SELECTED,
            selected_candidate_cid="bafycandidate",
            reason_codes=("something_else",),
        )


def test_non_selected_cannot_claim_winner() -> None:
    with pytest.raises(CandidateAdmissionError, match="cannot carry selected"):
        CandidateAdmission(
            portfolio_cid="bafyportfolio",
            disposition=AdmissionDisposition.ABSTAIN,
            selected_candidate_cid="bafycandidate",
            reason_codes=(CandidateAdmissionReason.TIE_ABSTAIN.value,),
        )


def test_planner_surface_shares_admission() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (_nom("add_registration", applicability=990_000, risk=1, resource_cost=1),),
        current_evidence_cid=_evidence("planner"),
    )
    a = admit_candidate_portfolio(portfolio)
    b = planner_admit(portfolio)
    assert a.admission_cid == b.admission_cid
    assert a.selected_candidate_cid == b.selected_candidate_cid


def test_only_one_selected_when_multiple_eligible() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nom("add_registration", applicability=900_000, risk=10),
            _nom("add_import", applicability=500_000, risk=10),
            _nom("add_export", applicability=400_000, risk=10),
        ),
        current_evidence_cid=_evidence("one"),
    )
    admission = admit_candidate_portfolio(portfolio)
    assert admission.disposition is AdmissionDisposition.SELECTED
    # Exactly one selected; others rejected with ranked reason.
    assert sum(
        1
        for c in portfolio.candidates
        if c.candidate_cid == admission.selected_candidate_cid
    ) == 1
    assert len(admission.ranked_eligible_cids) == 3
    assert len(admission.rejected) == 2
