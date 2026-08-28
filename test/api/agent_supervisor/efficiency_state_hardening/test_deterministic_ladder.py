"""ASEH-020: canonical receipt-to-human deterministic ladder coverage."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    DecisionLadder,
    DecisionLadderInputs,
    DeterministicEvidence,
    EvidenceStatus,
    LadderStage,
    RoutingInputs,
    StageAction,
    StageReason,
    UnresolvedQuestionRecord,
    route_decision_ladder,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    route_decision_ladder as verification_route_decision_ladder,
)


def _routing_inputs(**overrides: object) -> RoutingInputs:
    payload: dict[str, object] = {
        "context_tokens": 10_000,
        "lowest_confidence": "heuristic",
        "risk": "medium",
        "dependency_cone_size": 5,
        "unresolved_obligations": 0,
        "prior_repair_failures": 0,
        "available_proofs": 0,
        "prior_route_failed": False,
    }
    payload.update(overrides)
    return RoutingInputs.from_dict(payload)


def _question(*decisions: str, changes: bool = True) -> UnresolvedQuestionRecord:
    return UnresolvedQuestionRecord(
        question_id="question-1",
        question="Which admissible repair decision is correct?",
        admissible_decisions=decisions,
        answer_can_change_decision=changes,
    )


def _inputs(**overrides: object) -> DecisionLadderInputs:
    payload: dict[str, object] = {
        "routing_inputs": _routing_inputs(),
        "deterministic_evidence": {},
        "unresolved_question": _question(
            ModelRoute.MEDIUM_MODEL.value,
            ModelRoute.FRONTIER_MODEL.value,
        ),
    }
    payload.update(overrides)
    return DecisionLadderInputs(**payload)


def test_every_decision_receipts_the_exact_nine_stages_in_order() -> None:
    ladder = route_decision_ladder(
        _inputs(
            routing_inputs=_routing_inputs(risk="high"),
            deterministic_evidence={
                LadderStage.AUTHORITATIVE_RECEIPT.value: DeterministicEvidence(
                    status=EvidenceStatus.RESOLVED.value,
                    reference="admitted-receipt-7",
                )
            },
        )
    )

    assert [item.stage for item in ladder.stage_receipts] == [
        item.value for item in LadderStage
    ]
    assert ladder.selected_stage == LadderStage.AUTHORITATIVE_RECEIPT.value
    assert ladder.route == ModelRoute.DETERMINISTIC_ONLY.value
    assert ladder.stage_receipts[0].action == StageAction.RUN.value
    assert ladder.stage_receipts[0].reason == StageReason.DECISIVE_EVIDENCE.value
    assert ladder.stage_receipts[0].decisive is True
    assert all(
        item.action == StageAction.SKIP.value
        and item.reason == StageReason.PRIOR_STAGE_RESOLVED.value
        for item in ladder.stage_receipts[1:]
    )
    assert DecisionLadder.from_dict(ladder.to_dict()).to_dict() == ladder.to_dict()


def test_inconclusive_deterministic_evidence_precedes_medium_model_selection() -> None:
    ladder = route_decision_ladder(
        _inputs(
            deterministic_evidence={
                LadderStage.AST_SYMBOL_DEPENDENCY_IMPACT.value: DeterministicEvidence(
                    status=EvidenceStatus.INCONCLUSIVE.value,
                    reference="impact-42",
                ),
                LadderStage.SELECTED_TESTS.value: DeterministicEvidence(
                    status=EvidenceStatus.UNAVAILABLE.value,
                    reference="test-runner-unavailable",
                ),
            },
        )
    )

    assert ladder.selected_stage == LadderStage.LOCAL_REMOTE_MEDIUM.value
    assert ladder.route == ModelRoute.MEDIUM_MODEL.value
    receipts = {item.stage: item for item in ladder.stage_receipts}
    assert receipts[LadderStage.AST_SYMBOL_DEPENDENCY_IMPACT.value].action == "run"
    assert receipts[LadderStage.AST_SYMBOL_DEPENDENCY_IMPACT.value].reason == (
        StageReason.INCONCLUSIVE_EVIDENCE.value
    )
    assert receipts[LadderStage.SELECTED_TESTS.value].action == "skip"
    assert receipts[LadderStage.SELECTED_TESTS.value].reason == (
        StageReason.EVIDENCE_UNAVAILABLE.value
    )
    assert receipts[LadderStage.LOCAL_SMALL_SPECIALIST.value].reason == (
        StageReason.ROUTE_NOT_ELIGIBLE.value
    )
    assert receipts[LadderStage.LOCAL_REMOTE_MEDIUM.value].action == "run"
    assert receipts[LadderStage.REMOTE_FRONTIER.value].reason == (
        StageReason.PRIOR_STAGE_RESOLVED.value
    )


@pytest.mark.parametrize(
    ("question", "expected_reason"),
    [
        (None, StageReason.UNRESOLVED_QUESTION_REQUIRED.value),
        (
            _question(
                ModelRoute.MEDIUM_MODEL.value,
                ModelRoute.FRONTIER_MODEL.value,
                changes=False,
            ),
            StageReason.QUESTION_NOT_DECISION_RELEVANT.value,
        ),
    ],
)
def test_model_call_fails_closed_without_decision_relevant_question(
    question: UnresolvedQuestionRecord | None, expected_reason: str
) -> None:
    ladder = route_decision_ladder(_inputs(unresolved_question=question))

    assert ladder.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert ladder.selected_stage == LadderStage.HUMAN_REVIEW.value
    for receipt in ladder.stage_receipts[5:8]:
        assert receipt.action == StageAction.SKIP.value
        assert receipt.reason == expected_reason
    assert ladder.stage_receipts[-1].action == StageAction.RUN.value
    assert ladder.stage_receipts[-1].reason == StageReason.HUMAN_REVIEW_REQUIRED.value


def test_question_record_and_receipts_are_closed() -> None:
    with pytest.raises(HarnessError):
        UnresolvedQuestionRecord.from_dict(
            {
                **_question(
                    ModelRoute.SMALL_LOCAL_MODEL.value,
                    ModelRoute.MEDIUM_MODEL.value,
                ).to_dict(),
                "provider_prompt": "not admitted",
            }
        )
    with pytest.raises(HarnessError, match="canonical ladder order"):
        DecisionLadder.from_dict(
            {
                **route_decision_ladder(_inputs()).to_dict(),
                "stage_receipts": list(
                    reversed(route_decision_ladder(_inputs()).to_dict()["stage_receipts"])
                ),
            }
        )


def test_verification_reexports_the_single_ladder_authority() -> None:
    assert verification_route_decision_ladder is route_decision_ladder
