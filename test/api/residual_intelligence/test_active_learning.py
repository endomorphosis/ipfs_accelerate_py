from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.active_learning import (
    REASON_IMPACT_REQUIRED,
    AcquisitionBudget,
    AcquisitionCandidate,
    ResidualActiveLearningPlanner,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
)


def candidate(qid: str, *, impact: int, uncertainty: int = 10, tokens: int = 1, humans: int = 0, action: str = "revalidate") -> AcquisitionCandidate:
    return AcquisitionCandidate(
        action=action,
        question_id=qid,
        uncertainty_ppm=uncertainty,
        abstention_ppm=0,
        disagreement_ppm=0,
        validation_failure_ppm=0,
        novelty_ppm=0,
        task_frequency=0,
        token_cost=tokens,
        human_cost=humans,
        expected_route_improvement_ppm=impact,
    )


def test_impact_aware_idempotent_budgeted_selection() -> None:
    planner = ResidualActiveLearningPlanner(AcquisitionBudget(token_budget=10, human_review_cap=2))
    first = planner.select(
        (
            candidate("q-low", impact=1, uncertainty=100),
            candidate("q-high", impact=500, tokens=3, action="request_human_review", humans=1),
            candidate("q-high", impact=500, tokens=3, action="request_human_review", humans=1),
        )
    )
    second = planner.select(
        (
            candidate("q-high", impact=500, tokens=3, action="request_human_review", humans=1),
            candidate("q-low", impact=1, uncertainty=100),
        )
    )
    assert [item.question_id for item in first] == ["q-high"]
    assert [item.question_id for item in second] == ["q-high"]
    assert planner.plan_id(first) == planner.plan_id(second)
    with pytest.raises(ResidualIntelligenceError, match="production"):
        AcquisitionCandidate(
            action="explore_production",
            question_id="q",
            uncertainty_ppm=1,
            abstention_ppm=0,
            disagreement_ppm=0,
            validation_failure_ppm=0,
            novelty_ppm=0,
            task_frequency=0,
            token_cost=0,
            human_cost=0,
            expected_route_improvement_ppm=0,
        )
    with pytest.raises(ResidualIntelligenceError, match=REASON_IMPACT_REQUIRED):
        planner.select((candidate("q-low", impact=1, uncertainty=100),))
