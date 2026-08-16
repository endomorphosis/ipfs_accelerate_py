"""Evidence-driven, bounded refill orchestration.

This entrypoint deliberately does not own a board or a scheduler.  It turns a
small, current-tree observation into a deterministic refill decision and asks
the existing objective/refinery path to append work through its revision CAS.
Keeping those effects behind callbacks makes queue emptiness and stale task
status insufficient to either generate work or close a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID = "prompt_v3_refill.BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID"
REFILL_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/bounded-residual-refill-receipt@1"


class RefillTrigger(str, Enum):
    LOW_WATER = "low_water"
    DRAINED_OPEN_GOAL = "drained_open_goal"
    COMPLETION_REJECTED = "completion_rejected"
    ACTIONABLE_DRIFT = "actionable_drift"
    RETRY_EXHAUSTED = "retry_exhausted"
    ROLLOUT_THRESHOLD_MISSED = "rollout_threshold_missed"


class RefillDisposition(str, Enum):
    NO_REFILL = "no_refill"
    REFILLED = "refilled"
    REOPEN_CONVERGENCE = "reopen_convergence"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"
    CAS_CONFLICT = "cas_conflict"


@dataclass(frozen=True)
class RefillPolicy:
    """Immutable caps; refill output can never enlarge these values."""

    low_water_mark: int = 1
    max_findings_per_scan: int = 5
    max_new_work_per_epoch: int = 5
    max_refinement_depth: int = 3
    max_epochs: int = 8
    cooldown_epochs: int = 0
    max_unchanged_epochs: int = 2

    def __post_init__(self) -> None:
        for name in (
            "low_water_mark", "max_findings_per_scan", "max_new_work_per_epoch",
            "max_refinement_depth", "max_epochs", "max_unchanged_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.cooldown_epochs, bool) or self.cooldown_epochs < 0:
            raise ValueError("cooldown_epochs must be a non-negative integer")


@dataclass(frozen=True)
class RefillEpochCAS:
    """Fence supplied to the canonical task-source append operation."""

    plan_root_cid: str
    expected_revision: int
    epoch: int

    def __post_init__(self) -> None:
        if not self.plan_root_cid or self.expected_revision < 1 or self.epoch < 1:
            raise ValueError("refill CAS requires a plan root and positive revision/epoch")


@dataclass(frozen=True)
class CompletionAuthorityDecision:
    """Current-tree completion proof, never inferred from a drained queue."""

    root_evidence_complete: bool
    evidence_current: bool = True
    accepted_commits_reachable: bool = True
    final_forced_scan_clean: bool = False

    @property
    def authorized(self) -> bool:
        return bool(
            self.root_evidence_complete
            and self.evidence_current
            and self.accepted_commits_reachable
            and self.final_forced_scan_clean
        )


@dataclass(frozen=True)
class ResidualGap:
    """Smallest actionable unit emitted by the evidence evaluator."""

    goal_cid: str
    evidence_cid: str
    scope_cid: str
    lineage_goal_cids: tuple[str, ...]
    depth: int
    scheduler_metadata: Mapping[str, Any]
    kind: str = "task"

    @property
    def identity(self) -> str:
        payload = [self.goal_cid, self.evidence_cid, self.scope_cid]
        return "residual:sha256:" + hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def validate(self) -> None:
        if not self.goal_cid or not self.evidence_cid or not self.scope_cid:
            raise ValueError("residual gaps require goal, evidence, and scope identities")
        if not self.lineage_goal_cids or self.lineage_goal_cids[0] != self.goal_cid:
            raise ValueError("residual gap lineage must begin with its goal")
        if self.depth < 0:
            raise ValueError("residual gap depth cannot be negative")
        required = ("priority", "track", "parallel_lane", "resource_class")
        if any(not self.scheduler_metadata.get(name) for name in required):
            raise ValueError("residual gap is missing scheduler metadata")


@dataclass(frozen=True)
class ResidualEvidence:
    repository_tree_id: str
    gaps: tuple[ResidualGap, ...] = ()
    completion: CompletionAuthorityDecision = field(
        default_factory=lambda: CompletionAuthorityDecision(False, False, False, False)
    )


class ResidualEvidenceEvaluator(Protocol):
    def __call__(self, observation: "RefillObservation", *, force_final_scan: bool) -> ResidualEvidence: ...


class ProductionSelfImprovementHook(Protocol):
    def __call__(self, observation: "RefillObservation", evidence: ResidualEvidence) -> None: ...


@dataclass(frozen=True)
class RefillObservation:
    plan_root_cid: str
    revision: int
    ready_tasks: int = 0
    active_tasks: int = 0
    open_goals: int = 0
    validation_rejected: bool = False
    review_rejected: bool = False
    merge_rejected: bool = False
    stale_evidence: bool = False
    branch_only_completion: bool = False
    actionable_drift: bool = False
    retry_exhausted_with_refinement: bool = False
    rollout_threshold_missed: bool = False

    def __post_init__(self) -> None:
        if not self.plan_root_cid or self.revision < 1:
            raise ValueError("observation requires plan root and positive revision")
        if any(value < 0 for value in (self.ready_tasks, self.active_tasks, self.open_goals)):
            raise ValueError("work counts cannot be negative")


@dataclass(frozen=True)
class RefillDecision:
    disposition: RefillDisposition
    triggers: tuple[RefillTrigger, ...]
    epoch: int
    gap_identities: tuple[str, ...] = ()
    appended_count: int = 0
    reason: str = ""
    cas: RefillEpochCAS | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REFILL_RECEIPT_SCHEMA,
            "disposition": self.disposition.value,
            "triggers": [item.value for item in self.triggers],
            "epoch": self.epoch,
            "gap_identities": list(self.gap_identities),
            "appended_count": self.appended_count,
            "reason": self.reason,
            "cas": None if self.cas is None else {
                "plan_root_cid": self.cas.plan_root_cid,
                "expected_revision": self.cas.expected_revision,
                "epoch": self.cas.epoch,
            },
        }


AppendRefillWork = Callable[[Sequence[ResidualGap], RefillEpochCAS], bool]


def refill_triggers(observation: RefillObservation, policy: RefillPolicy) -> tuple[RefillTrigger, ...]:
    """Return the complete, stable trigger set in enum declaration order."""
    selected: list[RefillTrigger] = []
    if observation.ready_tasks + observation.active_tasks < policy.low_water_mark:
        selected.append(RefillTrigger.LOW_WATER)
    if observation.ready_tasks + observation.active_tasks == 0 and observation.open_goals:
        selected.append(RefillTrigger.DRAINED_OPEN_GOAL)
    if any((observation.validation_rejected, observation.review_rejected, observation.merge_rejected,
            observation.stale_evidence, observation.branch_only_completion)):
        selected.append(RefillTrigger.COMPLETION_REJECTED)
    if observation.actionable_drift:
        selected.append(RefillTrigger.ACTIONABLE_DRIFT)
    if observation.retry_exhausted_with_refinement:
        selected.append(RefillTrigger.RETRY_EXHAUSTED)
    if observation.rollout_threshold_missed:
        selected.append(RefillTrigger.ROLLOUT_THRESHOLD_MISSED)
    return tuple(selected)


class RefillController:
    """Stateful circuit breaker around current-tree residual evaluation."""

    def __init__(self, evaluator: ResidualEvidenceEvaluator, append: AppendRefillWork, *, policy: RefillPolicy | None = None,
                 production_hooks: Iterable[ProductionSelfImprovementHook] = ()) -> None:
        self.evaluator = evaluator
        self.append = append
        self.policy = policy or RefillPolicy()
        self.production_hooks = tuple(production_hooks)
        self._epoch = 0
        self._trigger_ticks = 0
        self._last_refill_tick: int | None = None
        self._last_gap_set: tuple[str, ...] = ()
        self._unchanged_epochs = 0
        self._seen_gap_ids: set[str] = set()

    def decide(self, observation: RefillObservation) -> RefillDecision:
        triggers = refill_triggers(observation, self.policy)
        if not triggers:
            return RefillDecision(RefillDisposition.NO_REFILL, (), self._epoch, reason="no_refill_trigger")
        self._trigger_ticks += 1
        routine_triggers = {RefillTrigger.LOW_WATER, RefillTrigger.DRAINED_OPEN_GOAL}
        if (
            self._last_refill_tick is not None
            and set(triggers).issubset(routine_triggers)
            and self._trigger_ticks - self._last_refill_tick <= self.policy.cooldown_epochs
        ):
            return RefillDecision(RefillDisposition.NO_REFILL, triggers, self._epoch, reason="cooldown")
        if self._epoch >= self.policy.max_epochs:
            return RefillDecision(RefillDisposition.BLOCKED, triggers, self._epoch, reason="epoch_budget_exhausted")

        evidence = self.evaluator(observation, force_final_scan=True)
        if not evidence.repository_tree_id:
            return RefillDecision(RefillDisposition.BLOCKED, triggers, self._epoch, reason="missing_current_tree_evidence")
        for hook in self.production_hooks:
            hook(observation, evidence)

        # A current-tree rejection (including branch-only or stale evidence)
        # must reopen convergence even if there is no immediately derivable task.
        if observation.branch_only_completion or observation.stale_evidence or not evidence.completion.evidence_current or not evidence.completion.accepted_commits_reachable:
            return RefillDecision(RefillDisposition.REOPEN_CONVERGENCE, triggers, self._epoch, reason="completion_not_authorized")

        candidates: list[ResidualGap] = []
        selected_ids: set[str] = set()
        # Evaluators may aggregate independent scanners.  Canonical identity
        # ordering and local deduplication keep their arrival order from
        # changing the first bounded batch.
        for gap in sorted(evidence.gaps, key=lambda item: item.identity)[: self.policy.max_findings_per_scan]:
            gap.validate()
            if (
                gap.depth >= self.policy.max_refinement_depth
                or gap.identity in self._seen_gap_ids
                or gap.identity in selected_ids
            ):
                continue
            candidates.append(gap)
            selected_ids.add(gap.identity)
            if len(candidates) == self.policy.max_new_work_per_epoch:
                break
        identities = tuple(gap.identity for gap in candidates)
        if identities == self._last_gap_set:
            self._unchanged_epochs += 1
        else:
            self._unchanged_epochs = 0
        self._last_gap_set = identities
        if self._unchanged_epochs >= self.policy.max_unchanged_epochs:
            return RefillDecision(RefillDisposition.BLOCKED, triggers, self._epoch, identities, reason="unchanged_residual_circuit_breaker")
        if not candidates:
            disposition = RefillDisposition.NO_REFILL if evidence.completion.authorized else RefillDisposition.BLOCKED
            reason = "final_scan_evidence_complete" if disposition is RefillDisposition.NO_REFILL else "no_novel_actionable_residual"
            return RefillDecision(disposition, triggers, self._epoch, reason=reason)

        epoch = self._epoch + 1
        cas = RefillEpochCAS(observation.plan_root_cid, observation.revision, epoch)
        if not self.append(tuple(candidates), cas):
            return RefillDecision(RefillDisposition.CAS_CONFLICT, triggers, self._epoch, identities, reason="revision_cas_rejected", cas=cas)
        self._epoch = epoch
        self._last_refill_tick = self._trigger_ticks
        self._seen_gap_ids.update(identities)
        return RefillDecision(RefillDisposition.REFILLED, triggers, epoch, identities, len(candidates), cas=cas)


__all__ = (
    "BOUNDED_RESIDUAL_REFILL_REQUIREMENT_ID", "REFILL_RECEIPT_SCHEMA", "AppendRefillWork",
    "CompletionAuthorityDecision", "ProductionSelfImprovementHook", "RefillController",
    "RefillDecision", "RefillDisposition", "RefillEpochCAS", "RefillObservation", "RefillPolicy",
    "RefillTrigger", "ResidualEvidence", "ResidualEvidenceEvaluator", "ResidualGap", "refill_triggers",
)
