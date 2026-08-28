"""Acceptance tests for ASEH-020's one receipt-to-human routing ladder."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessError
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    DecisionStage,
    DeterministicEvidenceOutcome,
    DeterministicLadderRequest,
    DeterministicStageEvidence,
    LadderReason,
    RouteReceipt,
    StageDisposition,
    UnresolvedQuestion,
    route_decision_ladder,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    AnalysisKind,
    CounterexampleQuality,
    ModelRouteFacts,
    ModelRoutePolicy,
    RiskLevel,
    decide_model_route_with_receipt,
    default_inventory,
    policy_cid_for,
)


_DETERMINISTIC_STAGES = (
    DecisionStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT,
    DecisionStage.AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS,
    DecisionStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS,
    DecisionStage.SELECTED_TESTS,
    DecisionStage.INCREMENTAL_SMT_OR_THEOREM_PROVER,
)


def _evidence(*, resolved_at: int | None = None) -> tuple[DeterministicStageEvidence, ...]:
    result: list[DeterministicStageEvidence] = []
    for index, stage in enumerate(_DETERMINISTIC_STAGES):
        outcome = (
            DeterministicEvidenceOutcome.RESOLVED
            if resolved_at == index
            else DeterministicEvidenceOutcome.UNRESOLVED
        )
        result.append(DeterministicStageEvidence(stage, outcome, (f"evidence-{index}",)))
        if resolved_at == index:
            break
    return tuple(result)


def _question(*, capability: str = "small_local_model") -> UnresolvedQuestion:
    return UnresolvedQuestion(
        question_id="question-1",
        exact_question="Does the remaining counterexample permit the patch?",
        why_prior_deterministic_stages_could_not_resolve=("all deterministic evidence is inconclusive",),
        evidence_available=("counterexample-cid",),
        evidence_missing=("semantic-intent",),
        candidate_decisions_answer_could_change=("accept_patch", "reject_patch"),
        minimum_model_capability=capability,
        context_budget=2048,
        response_schema="answer@1",
        deadline="2026-08-24T00:00:00Z",
        cost_budget=3,
    )


def test_cache_receipt_resolves_before_model_and_receipts_all_nine_stages() -> None:
    receipt = route_decision_ladder(
        DeterministicLadderRequest(
            deterministic_evidence=_evidence(resolved_at=0),
            proposed_route="frontier_model",  # ignored: prior evidence is decisive
            actual_resource_use={"cache_reads": 1},
        )
    )

    stages = receipt.stages
    assert tuple(item.stage for item in stages) == tuple(DecisionStage)
    assert stages[0].disposition is StageDisposition.RUN
    assert stages[0].reason is LadderReason.DETERMINISTIC_EVIDENCE_RESOLVED
    assert all(item.disposition is StageDisposition.SKIP for item in stages[1:])
    assert receipt.selected_executor == "deterministic_only"
    assert receipt.final_outcome == "deterministic_resolved"
    assert receipt.to_dict()["actual_resource_use"] == {"cache_reads": 1}
    assert RouteReceipt.from_dict(receipt.to_dict()).to_dict() == receipt.to_dict()


def test_all_deterministic_stages_precede_smallest_adequate_model() -> None:
    receipt = route_decision_ladder(
        DeterministicLadderRequest(
            deterministic_evidence=_evidence(),
            proposed_route="medium_model",
            unresolved_question=_question(capability="medium_model"),
        )
    )

    assert tuple(item.stage for item in receipt.stages) == tuple(DecisionStage)
    assert all(item.disposition is StageDisposition.RUN for item in receipt.stages[:5])
    assert receipt.stages[5].reason is LadderReason.SMALLER_CAPABILITY_INSUFFICIENT
    assert receipt.stages[6].disposition is StageDisposition.RUN
    assert receipt.stages[6].reason is LadderReason.UNRESOLVED_QUESTION_REQUIRES_MODEL
    assert receipt.stages[7].reason is LadderReason.SMALLEST_ADEQUATE_EXECUTOR_SELECTED
    assert receipt.selected_executor == "medium_model"
    assert receipt.unresolved_question == _question(capability="medium_model")


def test_model_escalation_fails_closed_without_question_or_complete_deterministic_ladder() -> None:
    with pytest.raises(HarnessError, match="all deterministic stages"):
        DeterministicLadderRequest(
            deterministic_evidence=_evidence()[:2], proposed_route="small_local_model"
        )

    with pytest.raises(HarnessError, match="typed unresolved_question"):
        route_decision_ladder(
            DeterministicLadderRequest(
                deterministic_evidence=_evidence(), proposed_route="small_local_model"
            )
        )

    with pytest.raises(HarnessError, match="must equal"):
        route_decision_ladder(
            DeterministicLadderRequest(
                deterministic_evidence=_evidence(),
                proposed_route="small_local_model",
                unresolved_question=_question(capability="medium_model"),
            )
        )

    with pytest.raises(HarnessError, match="must equal"):
        route_decision_ladder(
            DeterministicLadderRequest(
                deterministic_evidence=_evidence(),
                proposed_route="frontier_model",
                unresolved_question=_question(capability="medium_model"),
            )
        )


def test_human_receipt_follows_all_deterministic_stages_and_skips_models() -> None:
    receipt = route_decision_ladder(
        DeterministicLadderRequest(
            deterministic_evidence=_evidence(),
            proposed_route="human_review_required",
        )
    )

    assert tuple(item.stage for item in receipt.stages) == tuple(DecisionStage)
    assert all(item.disposition is StageDisposition.RUN for item in receipt.stages[:5])
    assert all(item.disposition is StageDisposition.SKIP for item in receipt.stages[5:8])
    assert receipt.stages[8].disposition is StageDisposition.RUN
    assert receipt.stages[8].reason is LadderReason.HUMAN_REVIEW_SELECTED
    assert receipt.selected_executor == "human_review_required"


def test_verification_scorer_is_a_thin_adapter_to_canonical_ladder() -> None:
    facts = ModelRouteFacts(
        context_token_estimate=1024,
        analysis_kind=AnalysisKind.LOCALIZED_EXACT,
        risk_level=RiskLevel.LOW,
        dependency_cone_size=1,
        changed_file_count=1,
        counterexample_quality=CounterexampleQuality.MINIMIZED,
        exact_contract_available=True,
    )
    decision, receipt = decide_model_route_with_receipt(
        facts,
        deterministic_evidence=_evidence(),
        unresolved_question=_question(),
        available_models=default_inventory(),
        policy=ModelRoutePolicy(policy_cid=policy_cid_for("ladder")),
    )

    assert decision.route.value == "small_local_model"
    assert receipt.selected_executor == decision.route.value
    assert tuple(item.stage for item in receipt.stages) == tuple(DecisionStage)
