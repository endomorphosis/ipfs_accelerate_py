"""Decompose a blocked frontier into successor subgoals.

A proof ceiling is a parked task whose declared postcondition asks for
owner evidence the recorded observation does not contain. Retrying that
same task cannot finish the parent goal. This planner emits one bounded
successor subgoal per supported ceiling, plus at most one filtered
language-model refinement per task.

The plan is a proposal. It does not mark a parent complete, drop a
dependency, renew an exhausted clock, edit a shared fence, or grant
completion authority. Callers still have to admit any later task.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


PROOF_CEILING_PLAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-ceiling-plan@1"

MANDATORY_REFUSALS: tuple[str, ...] = (
    "mark_parent_completed",
    "grant_completion_authority",
    "invent_queue",
    "invent_checkpoint",
    "invent_content_id",
    "drop_dependency",
    "edit_shared_fence",
    "renew_exhausted_clock",
)

_EXHAUSTED_PARKS = frozenset({
    "paid_role_failed_no_refund",
    "supplemental_deadline_exhausted",
    "final_failure_fence_differs",
    "keyboard_deadline_exhausted",
    "resume_deadline_without_sealed_intent",
})
_COMPLETED_PARENT = frozenset({"completed", "complete", "done"})
_READY = frozenset({"ready", "pending", "in_progress"})
_FORBIDDEN_MODEL_TEXT = (
    "mark the parent",
    "mark parent",
    "completion authority",
    "completion_authority",
    "invent",
    "fabricat",
    "drop the dependency",
    "drop dependencies",
    "remove the dependency",
    "renew the clock",
    "renew clock",
)


class CeilingKind(str, Enum):
    OWNER_LACKS_DECLARED_QUEUE = "owner_lacks_declared_queue"
    FAILED_OUTCOME_NOT_SEMANTIC = "failed_outcome_not_semantic"
    EXHAUSTED_CHILD_CLOCK = "exhausted_child_clock"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    BOUNDED_REFINEMENT = "bounded_refinement"


@dataclass(frozen=True)
class BlockedObservation:
    """One task the caller has already classified from its own records."""

    task_id: str
    goal_id: str
    status: str
    park_reason: str = ""
    parent_task_id: str = ""
    parent_status: str = ""
    owner_queue_used: bool | None = None
    owner_checkpoint_present: bool | None = None
    failed_outcome_joined: bool | None = None
    semantic_join: str = ""
    unmet_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.goal_id:
            raise ValueError("a blocked observation needs a task id and a goal id")
        object.__setattr__(self, "unmet_dependencies", tuple(self.unmet_dependencies))


@dataclass(frozen=True)
class SuccessorSubgoal:
    subgoal_id: str
    parent_goal_id: str
    blocked_task_id: str
    ceiling_kind: str
    title: str
    acceptance: tuple[str, ...]
    non_goals: tuple[str, ...]
    successor_task_title: str
    refused_effects: tuple[str, ...]
    source: str
    completion_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "parent_goal_id": self.parent_goal_id,
            "blocked_task_id": self.blocked_task_id,
            "ceiling_kind": self.ceiling_kind,
            "title": self.title,
            "acceptance": list(self.acceptance),
            "non_goals": list(self.non_goals),
            "successor_task_title": self.successor_task_title,
            "refused_effects": list(self.refused_effects),
            "source": self.source,
            "completion_authority": False,
        }


@dataclass(frozen=True)
class FrontierPlan:
    schema: str
    subgoals: tuple[SuccessorSubgoal, ...]
    refused: tuple[str, ...]
    work_complete: bool
    completion_authority: bool
    content_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "subgoals": [item.to_dict() for item in self.subgoals],
            "refused": list(self.refused),
            "work_complete": False,
            "completion_authority": False,
            "content_id": self.content_id,
        }


Proposer = Callable[[Sequence[BlockedObservation]], Sequence[Mapping[str, Any]]]


def supported_kinds(observation: BlockedObservation, blocked_ids: frozenset[str]) -> tuple[str, ...]:
    """Ceiling kinds this observation can support, in stable order."""

    kinds: list[str] = []
    if (
        observation.status == "blocked"
        and observation.owner_queue_used is False
        and observation.owner_checkpoint_present is False
    ):
        kinds.append(CeilingKind.OWNER_LACKS_DECLARED_QUEUE.value)
    if (
        observation.status == "blocked"
        and observation.failed_outcome_joined is False
        and observation.semantic_join == "executed_only"
    ):
        kinds.append(CeilingKind.FAILED_OUTCOME_NOT_SEMANTIC.value)
    if observation.park_reason in _EXHAUSTED_PARKS and observation.parent_task_id:
        kinds.append(CeilingKind.EXHAUSTED_CHILD_CLOCK.value)
    unmet = tuple(item for item in observation.unmet_dependencies if item)
    if (
        observation.status in _READY
        and unmet
        and all(item in blocked_ids for item in unmet)
    ):
        kinds.append(CeilingKind.DEPENDENCY_BLOCKED.value)
    return tuple(kinds)


def _subgoal_id(task_id: str, kind: str) -> str:
    digest = hashlib.sha256(f"{task_id}\n{kind}".encode()).hexdigest()[:16]
    return f"ceiling-{digest}"


def _subgoal(observation: BlockedObservation, kind: str) -> SuccessorSubgoal:
    parent_done = observation.parent_status in _COMPLETED_PARENT
    if kind == CeilingKind.OWNER_LACKS_DECLARED_QUEUE.value:
        title = "Align the declared postcondition with the observed synchronous owner result"
        acceptance = (
            "The declared result names the observed owner fields: completed status, queue unused, and no checkpoint.",
            "A preview completion is labeled as a preview.",
            "The parent visual task stays blocked until a new review passes on those terms.",
        )
        task_title = "Record the synchronous owner result without adding a queue, checkpoint, or content id"
    elif kind == CeilingKind.FAILED_OUTCOME_NOT_SEMANTIC.value:
        title = "Keep the rejected owner call beside the executed semantic result"
        acceptance = (
            "The failed call stays a request artifact and is not treated as an executed semantic proof.",
            "The later executed observation remains the semantic result.",
            "No sample, receipt, or content id is added by the planner.",
        )
        task_title = "Record the failed call as a non-semantic request artifact"
    elif kind == CeilingKind.EXHAUSTED_CHILD_CLOCK.value:
        title = "Leave the exhausted remediation child parked"
        if parent_done:
            acceptance = (
                "The child clock is not renewed.",
                "The completed parent receipt remains the parent evidence.",
                "This plan does not mark the child completed.",
            )
        else:
            acceptance = (
                "The child clock is not renewed.",
                "The unfinished parent is not treated as satisfied.",
                "This plan does not mark the child completed.",
            )
        task_title = "Record the exhausted child without renewing its clock"
    elif kind == CeilingKind.DEPENDENCY_BLOCKED.value:
        title = "Keep the ready task unselected while parked dependencies remain"
        acceptance = (
            "The task stays unselected until every unmet dependency is completed by its own evidence.",
            "No dependency is removed.",
        )
        task_title = "Record the parked dependencies that keep this task unselected"
    else:
        raise ValueError(f"unsupported ceiling kind: {kind}")
    return SuccessorSubgoal(
        subgoal_id=_subgoal_id(observation.task_id, kind),
        parent_goal_id=observation.goal_id,
        blocked_task_id=observation.task_id,
        ceiling_kind=kind,
        title=title,
        acceptance=acceptance,
        non_goals=MANDATORY_REFUSALS,
        successor_task_title=task_title,
        refused_effects=MANDATORY_REFUSALS,
        source="deterministic",
        completion_authority=False,
    )


def _clean_lines(value: object, *, limit: int) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    lines: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if not text or len(text) > 240:
            return None
        folded = text.casefold()
        if any(token in folded for token in _FORBIDDEN_MODEL_TEXT):
            return None
        lines.append(text)
        if len(lines) > limit:
            return None
    if not lines:
        return None
    return tuple(lines)


def _model_subgoal(
    observation: BlockedObservation,
    candidate: Mapping[str, Any],
    existing: SuccessorSubgoal,
) -> SuccessorSubgoal | None:
    title = " ".join(str(candidate.get("title") or "").split())
    acceptance = _clean_lines(candidate.get("acceptance"), limit=4)
    if not title or len(title) > 180 or acceptance is None:
        return None
    if str(candidate.get("ceiling_kind") or "") != CeilingKind.BOUNDED_REFINEMENT.value:
        return None
    if str(candidate.get("task_id") or "") != observation.task_id:
        return None
    if candidate.get("completion_authority") not in (False, None):
        return None
    if candidate.get("work_complete") not in (False, None):
        return None
    return SuccessorSubgoal(
        subgoal_id=_subgoal_id(observation.task_id, CeilingKind.BOUNDED_REFINEMENT.value),
        parent_goal_id=existing.subgoal_id,
        blocked_task_id=observation.task_id,
        ceiling_kind=CeilingKind.BOUNDED_REFINEMENT.value,
        title=title,
        acceptance=acceptance,
        non_goals=MANDATORY_REFUSALS,
        successor_task_title=title,
        refused_effects=MANDATORY_REFUSALS,
        source="llm",
        completion_authority=False,
    )


def _content_id(subgoals: Sequence[SuccessorSubgoal], refused: Sequence[str]) -> str:
    payload = {
        "schema": PROOF_CEILING_PLAN_SCHEMA,
        "subgoals": [item.to_dict() for item in subgoals],
        "refused": list(refused),
        "work_complete": False,
        "completion_authority": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def plan_blocked_frontier(
    observations: Sequence[BlockedObservation],
    *,
    proposer: Proposer | None = None,
) -> FrontierPlan:
    """Build a deterministic plan, then keep only admissible model refinements.

    ``proposer`` is optional. A model suggestion that asks to complete a
    parent, invent owner evidence, or drop a dependency is refused. The
    deterministic subgoals remain.
    """

    ordered = tuple(sorted(observations, key=lambda item: item.task_id))
    blocked_ids = frozenset(
        item.task_id for item in ordered if item.status == "blocked"
    )
    by_task = {item.task_id: item for item in ordered}
    subgoals: list[SuccessorSubgoal] = []
    refused: list[str] = []
    for observation in ordered:
        kinds = supported_kinds(observation, blocked_ids)
        if not kinds:
            refused.append(f"{observation.task_id}:no_supported_ceiling")
            continue
        for kind in kinds:
            subgoals.append(_subgoal(observation, kind))
    if proposer is not None:
        try:
            suggestions = list(proposer(ordered))
        except Exception as exc:
            refused.append("proposer_failed:" + type(exc).__name__)
            suggestions = []
        used: set[str] = set()
        for candidate in suggestions:
            if not isinstance(candidate, Mapping):
                refused.append("proposer_candidate_rejected")
                continue
            task_id = str(candidate.get("task_id") or "")
            observation = by_task.get(task_id)
            existing = next(
                (item for item in subgoals if item.blocked_task_id == task_id and item.source == "deterministic"),
                None,
            )
            if observation is None or existing is None or task_id in used:
                refused.append(f"{task_id or 'unknown'}:proposer_candidate_rejected")
                continue
            refined = _model_subgoal(observation, candidate, existing)
            if refined is None:
                refused.append(f"{task_id}:proposer_candidate_rejected")
                continue
            subgoals.append(refined)
            used.add(task_id)
    frozen = tuple(subgoals)
    refusals = tuple(refused)
    return FrontierPlan(
        schema=PROOF_CEILING_PLAN_SCHEMA,
        subgoals=frozen,
        refused=refusals,
        work_complete=False,
        completion_authority=False,
        content_id=_content_id(frozen, refusals),
    )
