"""Bounded offline acquisition planning.

Selection does not create TrainingCorpusAdmission, explore production, or
invoke shell.  Impact must beat uncertainty-only ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, canonical_id, required_text

ACQUISITION_BUDGET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-acquisition-budget@1"
)
ACQUISITION_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-acquisition-candidate@1"
)
ALLOWED_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "label_held_out",
        "request_human_review",
        "synthesize_adversarial",
        "revalidate",
        "refresh_calibration",
    }
)
REASON_NO_PRODUCTION_EXPLORATION: Final = "no_production_exploration"
REASON_HUMAN_REVIEW_CAP: Final = "human_review_cap"
REASON_IMPACT_REQUIRED: Final = "impact_aware_selection"


def _nonneg(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0 or value > maximum:
        raise ResidualIntelligenceError(f"{name} must be an integer in [0, {maximum}]")
    return value


@dataclass(frozen=True)
class AcquisitionBudget:
    token_budget: int
    human_review_cap: int
    authority_class: str = "candidate_only"
    schema: str = ACQUISITION_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_budget", _nonneg(self.token_budget, "token_budget", 1_000_000))
        object.__setattr__(
            self, "human_review_cap", _nonneg(self.human_review_cap, "human_review_cap", 50)
        )
        if self.authority_class != "candidate_only":
            raise ResidualIntelligenceError("acquisition cannot gain mutation authority")


@dataclass(frozen=True)
class AcquisitionCandidate:
    action: str
    question_id: str
    uncertainty_ppm: int
    abstention_ppm: int
    disagreement_ppm: int
    validation_failure_ppm: int
    novelty_ppm: int
    task_frequency: int
    token_cost: int
    human_cost: int
    expected_route_improvement_ppm: int
    schema: str = ACQUISITION_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        action = required_text(self.action, "action")
        if action not in ALLOWED_ACTIONS:
            raise ResidualIntelligenceError(f"unknown acquisition action: {action}")
        if action == "explore_production":
            raise ResidualIntelligenceError(REASON_NO_PRODUCTION_EXPLORATION)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "question_id", required_text(self.question_id, "question_id"))
        for name in (
            "uncertainty_ppm",
            "abstention_ppm",
            "disagreement_ppm",
            "validation_failure_ppm",
            "novelty_ppm",
            "expected_route_improvement_ppm",
        ):
            object.__setattr__(self, name, _nonneg(getattr(self, name), name, 1_000_000))
        object.__setattr__(self, "task_frequency", _nonneg(self.task_frequency, "task_frequency", 1_000_000))
        object.__setattr__(self, "token_cost", _nonneg(self.token_cost, "token_cost", 1_000_000))
        object.__setattr__(self, "human_cost", _nonneg(self.human_cost, "human_cost", 50))

    @property
    def impact_score(self) -> int:
        return (
            self.expected_route_improvement_ppm
            + self.validation_failure_ppm
            + self.disagreement_ppm
            + self.novelty_ppm
            + self.abstention_ppm
            + self.task_frequency
            - self.token_cost
            - self.human_cost
        )

    @property
    def uncertainty_only_score(self) -> int:
        return self.uncertainty_ppm

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action": self.action,
            "question_id": self.question_id,
            "impact_score": self.impact_score,
            "candidate_only": True,
        }


@dataclass(frozen=True)
class ResidualActiveLearningPlanner:
    budget: AcquisitionBudget

    def select(
        self, candidates: Sequence[AcquisitionCandidate]
    ) -> tuple[AcquisitionCandidate, ...]:
        ranked = tuple(
            sorted(candidates, key=lambda item: (-item.impact_score, item.question_id))
        )
        selected: list[AcquisitionCandidate] = []
        seen: set[str] = set()
        tokens = 0
        humans = 0
        for item in ranked:
            if item.question_id in seen:
                continue
            if item.impact_score <= item.uncertainty_only_score:
                continue
            if tokens + item.token_cost > self.budget.token_budget:
                continue
            if humans + item.human_cost > self.budget.human_review_cap:
                continue
            selected.append(item)
            seen.add(item.question_id)
            tokens += item.token_cost
            humans += item.human_cost
        if not selected and ranked:
            raise ResidualIntelligenceError(REASON_IMPACT_REQUIRED)
        return tuple(selected)

    def plan_id(self, selected: Sequence[AcquisitionCandidate]) -> str:
        return canonical_id(
            {
                "budget": self.budget.token_budget,
                "human_review_cap": self.budget.human_review_cap,
                "selected": [item.question_id for item in selected],
            }
        )
