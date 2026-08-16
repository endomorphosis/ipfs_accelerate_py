"""DCR-081 pure, typed evidence-driven repair selection and refill.

Selection is a projection only.  It replays the DCR-061 through DCR-064
dependency chain and the DCR-080 daemon receipt before emitting a key; it
never mutates a queue or grants execution authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..autonomous_repair.contracts import (
    AuthorityStage,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
    repair_evidence_cid,
)
from ..planning.deterministic_candidate_portfolio import (
    CandidatePortfolio,
    CandidatePortfolioDisposition,
)
from ..planning.deterministic_failure_memory import (
    FailureAttempt,
    FailureMemoryReceipt,
    ReplanMemoryDecision,
    ReplanMemoryDisposition,
    decide_replan,
)
from ..planning.proof_carrying_repair_dag import (
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
)
from ..planning.repair_resource_scheduler import (
    RepairResourcePolicy,
    RepairResourceSchedule,
    schedule_repair_resources,
)
from ..todo_daemon.deterministic_repair_composition import (
    DCR080_COMPOSITION_SCHEMA,
    DeterministicRepairCompositionResult,
)

DCR_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-selection@1"
)
DCR_SELECTION_DEPENDENCIES_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-selection-dependencies@1"
)


class SelectionState(str, Enum):  # noqa: UP042 - Python 3.8
    DERIVED = "derived"
    ADMITTED = "admitted"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    REVIEW = "review"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    ANALYSIS_ONLY = "analysis_only"


def _zero_authority(value: Any) -> bool:
    return (
        getattr(value, "execution_authorized", None) is False
        and getattr(value, "completion_authorized", None) is False
        and all(
            type(getattr(value, field, None)) is int and getattr(value, field) == 0
            for field in (
                "model_call_count",
                "provider_call_count",
                "network_call_count",
            )
        )
    )


@dataclass(frozen=True)
class RepairSelectionDependencyBundle:
    """Exact DCR-061→064 objects whose identities are replayed together."""

    roots: RepairAuthorityRoots
    plan: ProofCarryingRepairPlan
    plan_result: RepairPlanDagResult
    portfolio: CandidatePortfolio
    attempt: FailureAttempt
    history: tuple[FailureMemoryReceipt, ...]
    decision: ReplanMemoryDecision
    policy: RepairResourcePolicy
    schedule: RepairResourceSchedule

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(self.history))

    def reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not isinstance(self.roots, RepairAuthorityRoots):
            reasons.append("typed_current_roots_required")
        if not isinstance(self.plan, ProofCarryingRepairPlan) or not isinstance(
            self.plan_result, RepairPlanDagResult
        ):
            reasons.append("typed_dcr061_plan_and_result_required")
        elif (
            self.plan.authority_roots != self.roots
            or self.plan_result.plan_cid != self.plan.content_id
            or self.plan_result.disposition is not RepairPlanDagDisposition.INTEGRATION_PENDING
            or not _zero_authority(self.plan_result)
        ):
            reasons.append("dcr061_plan_result_or_root_identity_invalid")
        if not isinstance(self.portfolio, CandidatePortfolio):
            reasons.append("typed_dcr062_portfolio_required")
        elif (
            self.portfolio.disposition is not CandidatePortfolioDisposition.INTEGRATION_PENDING
            or not self.portfolio.portfolio_cid
            or not self.portfolio.candidate_cids
            or not _zero_authority(self.portfolio)
        ):
            reasons.append("dcr062_portfolio_is_not_current_pending_evidence")
        if not isinstance(self.attempt, FailureAttempt):
            reasons.append("typed_dcr063_attempt_required")
        elif (
            isinstance(self.roots, RepairAuthorityRoots)
            and isinstance(self.plan_result, RepairPlanDagResult)
            and isinstance(self.portfolio, CandidatePortfolio)
        ):
            if (
                self.attempt.root_cid != self.roots.content_id
                or self.attempt.plan_cid != self.plan_result.plan_cid
                or self.attempt.portfolio_cid != self.portfolio.portfolio_cid
                or self.attempt.candidate_cid not in self.portfolio.candidate_cids
            ):
                reasons.append("dcr063_attempt_dependency_identity_invalid")
        if any(not isinstance(item, FailureMemoryReceipt) for item in self.history):
            reasons.append("typed_dcr063_history_required")
        if not isinstance(self.decision, ReplanMemoryDecision) or not _zero_authority(
            self.decision
        ):
            reasons.append("typed_zero_authority_dcr063_decision_required")
        if not isinstance(self.policy, RepairResourcePolicy) or not isinstance(
            self.schedule, RepairResourceSchedule
        ):
            reasons.append("typed_dcr064_policy_and_schedule_required")
        if reasons:
            return tuple(sorted(set(reasons)))
        try:
            expected_decision = decide_replan(
                self.portfolio,
                self.plan_result,
                self.roots,
                self.attempt,
                history=self.history,
            )
            if (
                expected_decision != self.decision
                or self.decision.disposition is not ReplanMemoryDisposition.RETRY_PENDING
                or self.decision.receipt_cid != self.attempt.content_id
            ):
                reasons.append("dcr063_decision_does_not_recompute")
            expected_schedule = schedule_repair_resources(
                self.plan,
                self.plan_result,
                self.decision,
                portfolio=self.portfolio,
                attempt=self.attempt,
                history=self.history,
                current_roots=self.roots,
                policy=self.policy,
            )
            if (
                expected_schedule != self.schedule
                or self.schedule.disposition != "integration_pending"
                or not self.schedule.schedule_cid
                or not self.schedule.nodes
                or not _zero_authority(self.schedule)
            ):
                reasons.append("dcr064_schedule_does_not_recompute")
        except (TypeError, ValueError):
            reasons.append("dcr061_through_dcr064_replay_failed")
        return tuple(sorted(set(reasons)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_SELECTION_DEPENDENCIES_SCHEMA,
            "roots_cid": self.roots.content_id,
            "dcr061_plan_cid": self.plan.content_id,
            "dcr061_result_cid": repair_evidence_cid(
                {
                    "disposition": self.plan_result.disposition.value,
                    "reason_codes": list(self.plan_result.reason_codes),
                    "plan_cid": self.plan_result.plan_cid,
                    "node_cids": list(self.plan_result.node_cids),
                }
            ),
            "dcr062_portfolio_cid": self.portfolio.portfolio_cid,
            "dcr063_attempt_cid": self.attempt.content_id,
            "dcr063_history_cids": [item.receipt_cid for item in self.history],
            "dcr063_decision_cid": repair_evidence_cid(
                {
                    "disposition": self.decision.disposition.value,
                    "reason_codes": list(self.decision.reason_codes),
                    "receipt_cid": self.decision.receipt_cid,
                }
            ),
            "dcr064_policy_cid": repair_evidence_cid(self.policy.to_dict()),
            "dcr064_schedule_cid": self.schedule.schedule_cid,
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


@dataclass(frozen=True)
class RepairSelectionEvidence:
    key: str
    state: SelectionState
    envelope: RepairEvidenceEnvelope
    dependencies: RepairSelectionDependencyBundle
    owner_root: str
    risk: int
    capability: str
    transition: DeterministicRepairCompositionResult

    def __post_init__(self) -> None:
        for name in ("key", "owner_root", "capability"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be exact non-empty text")
        if (
            not isinstance(self.state, SelectionState)
            or type(self.risk) is not int
            or self.risk < 0
        ):
            raise ValueError("state and risk must be closed")
        if not isinstance(self.envelope, RepairEvidenceEnvelope):
            raise ValueError("typed DCR-002 envelope required")
        if not isinstance(self.dependencies, RepairSelectionDependencyBundle):
            raise ValueError("typed DCR-061 through DCR-064 dependency bundle required")
        if self.envelope.authority_roots != self.dependencies.roots:
            raise ValueError("DCR-002 envelope roots do not match dependency roots")
        required_stage = {
            SelectionState.DERIVED: AuthorityStage.DERIVED,
            SelectionState.ADMITTED: AuthorityStage.ADMITTED,
            SelectionState.COMPLETED: AuthorityStage.PUBLISHED,
        }.get(self.state)
        if required_stage is not None and self.envelope.authority_stage is not required_stage:
            raise ValueError("selection state does not match DCR-002 authority stage")
        if not isinstance(self.transition, DeterministicRepairCompositionResult):
            raise ValueError("typed DCR-080 transition receipt required")
        transition = self.transition.to_dict()
        if (
            transition.get("schema") != DCR080_COMPOSITION_SCHEMA
            or self.transition.receipt_cid != repair_evidence_cid(transition)
            or self.transition.task_id != self.key
            or any(
                transition.get(name) != 0
                for name in ("model_call_count", "provider_call_count", "network_call_count")
            )
        ):
            raise ValueError("DCR-080 receipt, counters, or task identity invalid")

    @property
    def envelope_cid(self) -> str:
        return self.envelope.content_id

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "state": self.state.value,
            "envelope_cid": self.envelope_cid,
            "dependencies_cid": self.dependencies.content_id,
            **self.dependencies.to_dict(),
            "owner_root": self.owner_root,
            "risk": self.risk,
            "capability": self.capability,
            "transition_cid": self.transition.receipt_cid,
        }


@dataclass(frozen=True)
class RepairSelectionResult:
    disposition: str
    reason_codes: tuple[str, ...]
    selected_key: str = ""
    refill_keys: tuple[str, ...] = ()
    selection_cid: str = ""
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def select_and_refill_repairs(
    evidence: Sequence[RepairSelectionEvidence], *, existing_keys: Sequence[str] = ()
) -> RepairSelectionResult:
    """Select one unique minimum pending item without mutating a queue."""

    values = tuple(evidence)
    if any(not isinstance(item, RepairSelectionEvidence) for item in values):
        return RepairSelectionResult("rejected", ("typed_evidence_required",))
    if len({item.key for item in values}) != len(values) or len(
        {item.envelope_cid for item in values}
    ) != len(values):
        return RepairSelectionResult("rejected", ("duplicate_canonical_key_or_evidence",))
    dependency_errors = {
        item.key: item.dependencies.reason_codes()
        for item in values
        if item.dependencies.reason_codes()
    }
    if dependency_errors:
        return RepairSelectionResult(
            "rejected",
            tuple(
                sorted(
                    {
                        f"{key}:{reason}"
                        for key, reasons in dependency_errors.items()
                        for reason in reasons
                    }
                )
            ),
        )
    if any(not isinstance(key, str) or not key for key in existing_keys):
        return RepairSelectionResult("rejected", ("existing_keys_must_be_exact_text",))
    existing = set(existing_keys)
    eligible = [
        item
        for item in values
        if item.state is SelectionState.DERIVED
        and item.capability == "available"
        and item.key not in existing
    ]
    if not eligible:
        return RepairSelectionResult(
            "integration_pending", ("fixed_point_or_no_eligible_evidence",)
        )
    minimum = min(item.risk for item in eligible)
    best = [item for item in eligible if item.risk == minimum]
    if len(best) != 1:
        return RepairSelectionResult("abstained", ("equal_authority_risk_tie",))
    selected = best[0]
    refill = tuple(sorted({item.key for item in eligible}))
    body = {
        "schema": DCR_SELECTION_SCHEMA,
        "selected": selected.content_id,
        "refill_keys": list(refill),
    }
    return RepairSelectionResult(
        "integration_pending",
        ("integration_pending_live_dcr080",),
        selected.key,
        refill,
        repair_evidence_cid(body),
    )


__all__ = [
    "DCR_SELECTION_DEPENDENCIES_SCHEMA",
    "DCR_SELECTION_SCHEMA",
    "RepairSelectionDependencyBundle",
    "RepairSelectionEvidence",
    "RepairSelectionResult",
    "SelectionState",
    "select_and_refill_repairs",
]
