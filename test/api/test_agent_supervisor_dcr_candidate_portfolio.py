"""DCR-062: generate a finite symbolic candidate portfolio.

Acceptance:
* Ranks finite candidates by proved applicability, risk, edit size, resource
  cost, and validation strength.
* All candidates bind current evidence and exact operator CIDs.
* Evidence subset includes candidate CID, operator args, score terms, proof
  receipt, and rejected reason.
* Enumerates registered operators and bounded arguments only; natural-language
  implementation bodies and silent IR attachment failures are rejected.
* Runtime model calls remain 0; write authority is never granted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CANDIDATE_PORTFOLIO_INTERFACE,
    DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
    REPAIR_CANDIDATE_INTERFACE,
    AdmissionDisposition,
    ApplicabilityStatus,
    CandidateEligibility,
    CandidateNomination,
    CandidatePortfolio,
    CandidatePortfolioError,
    CandidateScoreTerms,
    IrAttachmentStatus,
    RepairCandidate,
    admit_candidate_portfolio,
    build_and_admit_candidate_portfolio,
    build_deterministic_candidate_portfolio,
    materialize_candidate_portfolios,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_operator_registry import (
    build_default_repair_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    SymbolicCandidatePlanner,
    admit_operator_candidate_portfolio,
    build_deterministic_candidate_portfolio as planner_build_portfolio,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _evidence(tag: str = "fixture") -> str:
    return content_identity({"role": "current_evidence", "tag": tag})


def _nomination(
    kind: str = "add_registration",
    *,
    applicability: int = 800_000,
    risk: int = 100_000,
    edit_size: int = 2,
    resource_cost: int = 100,
    validation: int = 700_000,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "operator_kind": kind,
        "operator_args": {"symbol": f"sym_{kind}", "target": "pkg/module.py"},
        "write_paths": ("pkg/module.py",),
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


def test_interfaces_and_evidence_are_stable() -> None:
    assert REPAIR_CANDIDATE_INTERFACE == "RepairCandidate@1"
    assert CANDIDATE_PORTFOLIO_INTERFACE == "CandidatePortfolio@1"
    assert DCR_CANDIDATE_PORTFOLIO_EVIDENCE == "dcr/candidate-portfolio@1"
    assert isinstance(SymbolicCandidatePlanner, type)


def test_build_portfolio_ranks_by_score_terms() -> None:
    evidence = _evidence("rank")
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nomination(
                "add_import",
                applicability=700_000,
                risk=50_000,
                edit_size=3,
                resource_cost=200,
                validation=500_000,
            ),
            _nomination(
                "add_registration",
                applicability=900_000,
                risk=80_000,
                edit_size=1,
                resource_cost=90,
                validation=850_000,
            ),
            _nomination(
                "add_export",
                applicability=900_000,
                risk=40_000,
                edit_size=1,
                resource_cost=90,
                validation=850_000,
            ),
        ),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:rank",
    )
    assert isinstance(portfolio, CandidatePortfolio)
    assert portfolio.runtime_model_calls == 0
    assert portfolio.grants_write_authority is False
    assert portfolio.current_evidence_cid == evidence
    assert len(portfolio.candidates) == 3

    # Best: higher applicability, then lower risk among equals.
    best = portfolio.candidates[0]
    assert best.facts.operator_kind == "add_export"
    assert best.score_terms.proved_applicability == 900_000
    assert best.score_terms.risk == 40_000

    # Every candidate binds current evidence and an exact operator CID.
    registry = build_default_repair_operator_registry()
    for candidate in portfolio.candidates:
        assert isinstance(candidate, RepairCandidate)
        assert candidate.facts.current_evidence_cid == evidence
        assert candidate.operator_cid
        assert candidate.operator_cid.startswith("b")
        assert candidate.facts.operator_id.startswith("repair-operator:")
        assert candidate.proof_receipt_cid
        assert candidate.grants_write_authority is False
        assert candidate.runtime_model_calls == 0
        # Operator CID matches the registry spec content identity.
        spec = registry.get(candidate.facts.operator_kind)
        assert candidate.operator_cid == spec.spec_id


def test_evidence_subset_contains_required_fields() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (_nomination(),),
        current_evidence_cid=_evidence("subset"),
    )
    subset = portfolio.evidence_subset()
    assert subset["evidence_id"] == DCR_CANDIDATE_PORTFOLIO_EVIDENCE
    assert subset["runtime_model_calls"] == 0
    assert len(subset["candidates"]) == 1
    row = subset["candidates"][0]
    assert row["candidate_cid"]
    assert "operator_args" in row
    assert "score_terms" in row
    assert row["proof_receipt"]
    assert "rejected_reason" in row
    assert row["operator_cid"]
    assert row["current_evidence_cid"] == portfolio.current_evidence_cid


def test_score_terms_rank_key_is_lexicographic() -> None:
    high = CandidateScoreTerms(
        proved_applicability=900_000,
        risk=10,
        edit_size=1,
        resource_cost=10,
        validation_strength=800_000,
    )
    low = CandidateScoreTerms(
        proved_applicability=100_000,
        risk=1,
        edit_size=1,
        resource_cost=1,
        validation_strength=999_000,
    )
    assert high.rank_key() < low.rank_key()


def test_rejects_natural_language_implementation_body() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (
            {
                "operator_kind": "add_registration",
                "operator_args": {
                    "source_body": "def fix():\n    return 1\n",
                },
                "write_paths": ("pkg/module.py",),
            },
        ),
        current_evidence_cid=_evidence("nl"),
    )
    assert len(portfolio.candidates) == 1
    candidate = portfolio.candidates[0]
    assert candidate.eligibility is CandidateEligibility.REJECTED
    assert any("natural_language" in reason for reason in candidate.rejection_reasons)


def test_rejects_silent_ir_attachment_failure() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (
            _nomination(
                ir_attachment_status=IrAttachmentStatus.MISSING.value,
            ),
        ),
        current_evidence_cid=_evidence("ir"),
    )
    candidate = portfolio.candidates[0]
    assert candidate.eligibility is not CandidateEligibility.ELIGIBLE
    assert any("silent_ir" in reason for reason in candidate.rejection_reasons)


def test_unregistered_operator_is_rejected_not_admitted() -> None:
    portfolio = build_deterministic_candidate_portfolio(
        (
            {
                "operator_kind": "totally_invented_operator",
                "operator_args": {"x": "1"},
                "write_paths": ("pkg/module.py",),
                "proved_applicability": 999_000,
            },
        ),
        current_evidence_cid=_evidence("unknown-op"),
    )
    candidate = portfolio.candidates[0]
    assert candidate.eligibility is CandidateEligibility.REJECTED
    admission = admit_candidate_portfolio(portfolio)
    assert admission.disposition is not AdmissionDisposition.SELECTED
    assert admission.selected_candidate_cid == ""


def test_stale_evidence_cannot_enter_portfolio() -> None:
    evidence = _evidence("fresh")
    portfolio = build_deterministic_candidate_portfolio(
        (_nomination(),),
        current_evidence_cid=evidence,
    )
    # Manually construct a candidate with drifted evidence — portfolio rejects.
    stale = RepairCandidate(
        facts={
            **portfolio.candidates[0].facts.to_dict(),
            "schema": portfolio.candidates[0].facts.SCHEMA,
            "current_evidence_cid": _evidence("stale"),
        },
        score_terms=portfolio.candidates[0].score_terms,
    )
    with pytest.raises(CandidatePortfolioError, match="stale_evidence"):
        CandidatePortfolio(
            portfolio_id="portfolio:stale",
            current_evidence_cid=evidence,
            registry_cid=portfolio.registry_cid,
            candidates=(stale,),
        )


def test_symbolic_planner_reexports_and_bridge() -> None:
    evidence = _evidence("bridge")
    portfolio_a = planner_build_portfolio(
        (_nomination(applicability=950_000, risk=10, edit_size=1, resource_cost=10),),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:bridge",
    )
    portfolio_b, admission = admit_operator_candidate_portfolio(
        (_nomination(applicability=950_000, risk=10, edit_size=1, resource_cost=10),),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:bridge",
    )
    assert portfolio_a.portfolio_cid == portfolio_b.portfolio_cid
    assert admission.ok is True
    assert admission.selected_candidate_cid == portfolio_b.candidates[0].candidate_cid


def test_materialize_candidate_portfolios(tmp_path: Path) -> None:
    dest = tmp_path / "candidate-portfolios.json"
    payload = materialize_candidate_portfolios(destination=dest)
    assert dest.is_file()
    assert payload["evidence_id"] == DCR_CANDIDATE_PORTFOLIO_EVIDENCE
    assert payload["runtime_model_calls"] == 0
    assert payload["grants_write_authority"] is False
    assert payload["admission"]["disposition"] == AdmissionDisposition.SELECTED.value
    rebuilt = CandidatePortfolio.from_dict(payload["portfolio"])
    assert rebuilt.portfolio_id == "portfolio:dcr062-fixture"


def test_nomination_dataclass_and_build_and_admit() -> None:
    evidence = _evidence("dataclass")
    nomination = CandidateNomination(
        operator_kind="add_registration",
        operator_args={"symbol": "echo"},
        write_paths=("pkg/a.py",),
        proved_applicability=880_000,
        risk=20_000,
        edit_size=1,
        resource_cost=50,
        validation_strength=900_000,
    )
    portfolio, admission = build_and_admit_candidate_portfolio(
        (nomination,),
        current_evidence_cid=evidence,
    )
    assert admission.disposition is AdmissionDisposition.SELECTED
    assert len(portfolio.eligible()) == 1
