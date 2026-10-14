"""Immutable checker declarations for existing native acceptance criteria.

This operation binds a declared validation command and advisory evidence to
one producer and all of its mirrored goal criteria. It does not execute the
command, authenticate evidence, satisfy a criterion, or admit a plan. Existing
IR, proof, validation-result and prompt admission checks remain mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from .prompt_workflow import (
    EvidenceAuthority, PromptEvidenceRecord, PromptGoalGraph, PromptGoalRecord,
    PromptTaskRecord, PromptValidationRecord, prompt_workflow_cid,
)


class AcceptanceBindingError(ValueError):
    """Stale, ambiguous, conflicting or unsupported checker declaration."""


@dataclass(frozen=True)
class AcceptanceBindingResult:
    graph: PromptGoalGraph
    original_plan_root: str
    binding_id: str
    criterion_key: str
    producer_task_key: str
    validation_key: str
    goal_cid_map: Mapping[str, str]
    task_cid_map: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_cid_map", MappingProxyType(dict(self.goal_cid_map)))
        object.__setattr__(self, "task_cid_map", MappingProxyType(dict(self.task_cid_map)))

    def to_dict(self) -> dict:
        return {
            "schema": "prompt-acceptance-checker-binding/v1",
            "original_plan_root": self.original_plan_root,
            "bound_plan_root": self.graph.plan_root_cid,
            "binding_id": self.binding_id,
            "criterion_key": self.criterion_key,
            "producer_task_key": self.producer_task_key,
            "validation_key": self.validation_key,
            "goal_cid_map": dict(self.goal_cid_map),
            "task_cid_map": dict(self.task_cid_map),
            "declaration_only": True,
            "checker_executed": False,
            "evidence_authenticated": False,
            "criterion_satisfied": False,
            "admitted": False,
            "Lean_admitted": False,
        }


def bind_acceptance_checker(
    graph: PromptGoalGraph, *, expected_plan_root: str, criterion_key: str,
    producer_task_key: str, validation: PromptValidationRecord,
    evidence: PromptEvidenceRecord,
) -> AcceptanceBindingResult:
    """Bind one exact criterion without editing its text or other checks.

    Schema-v1 records are supported. V2 lifecycle/dependency/validation DAGs
    require their own revision owner and are explicitly refused here.
    The original graph remains immutable; changed hierarchy/dependency CIDs are
    recomputed together and returned as a complete identity crosswalk.
    """
    if not isinstance(graph, PromptGoalGraph):
        raise AcceptanceBindingError("a native PromptGoalGraph is required")
    if expected_plan_root != graph.plan_root_cid:
        raise AcceptanceBindingError("stale expected plan root")
    if (any(type(goal) is not PromptGoalRecord for goal in graph.goals)
            or any(type(task) is not PromptTaskRecord for task in graph.tasks)):
        raise AcceptanceBindingError("schema-v2 lifecycle/DAG binding requires its revision owner")
    if not isinstance(validation, PromptValidationRecord) or not isinstance(evidence, PromptEvidenceRecord):
        raise AcceptanceBindingError("native validation and evidence records are required")
    if evidence.authority is not EvidenceAuthority.SCAN_ADVISORY:
        raise AcceptanceBindingError("checker declaration evidence must remain advisory")
    if criterion_key not in evidence.claim_keys:
        raise AcceptanceBindingError("evidence omits the exact criterion claim key")
    if any(item.validation_key == validation.validation_key for task in graph.tasks for item in task.validations):
        raise AcceptanceBindingError("validation key already exists; use its explicit revision owner")
    if any(item.evidence_key == evidence.evidence_key for item in graph.evidence):
        raise AcceptanceBindingError("evidence key already exists")
    producers = [task for task in graph.tasks if task.task_key == producer_task_key]
    if len(producers) != 1:
        raise AcceptanceBindingError("exactly one producer task is required")
    producer = producers[0]
    task_matches = [(task, criterion) for task in graph.tasks for criterion in task.acceptance
                    if criterion.criterion_key == criterion_key]
    goal_matches = [(goal, criterion) for goal in graph.goals for criterion in goal.acceptance
                    if criterion.criterion_key == criterion_key]
    if len(task_matches) != 1 or task_matches[0][0] != producer or not goal_matches:
        raise AcceptanceBindingError("criterion must have one exact producer and mirrored goal coverage")
    criterion = task_matches[0][1]
    if any(item.criterion != criterion.criterion for _, item in goal_matches):
        raise AcceptanceBindingError("mirrored criterion text conflicts")
    by_goal = {goal.goal_cid: goal for goal in graph.goals}
    ancestors = set()
    ancestor = producer.goal_cid
    while ancestor:
        if ancestor in ancestors or ancestor not in by_goal:
            raise AcceptanceBindingError("unresolved producer goal ancestry")
        ancestors.add(ancestor)
        ancestor = by_goal[ancestor].parent_goal_cid
    if any(goal.goal_cid not in ancestors for goal, _ in goal_matches):
        raise AcceptanceBindingError("mirrored criterion is outside the producer goal ancestry")
    mirrored_goals = {goal.goal_cid for goal, _ in goal_matches}
    spec = {
        "schema": "prompt-acceptance-checker-declaration/v1",
        "original_plan_root": expected_plan_root,
        "criterion_key": criterion_key, "producer_task_key": producer_task_key,
        "criterion": criterion.criterion, "validation": validation.to_dict(),
        "evidence_cid": evidence.evidence_cid,
    }
    binding_id = prompt_workflow_cid(spec)

    def acceptance(items):
        return tuple(replace(item,
            validation_keys=tuple(dict.fromkeys((*item.validation_keys, validation.validation_key))),
            evidence_cids=tuple(dict.fromkeys((*item.evidence_cids, evidence.evidence_cid))))
            if item.criterion_key == criterion_key else item for item in items)

    goal_map, goals = {}, []
    pending = {goal.goal_cid: goal for goal in graph.goals}
    while pending:
        ready = [cid for cid, goal in pending.items() if all(dependency in goal_map
            for dependency in (goal.parent_goal_cid, *goal.dependency_goal_cids) if dependency)]
        if not ready:
            raise AcceptanceBindingError("unresolved goal hierarchy or dependency")
        for cid in sorted(ready):
            goal = pending.pop(cid)
            changed = replace(goal, parent_goal_cid=goal_map.get(goal.parent_goal_cid, ""),
                dependency_goal_cids=tuple(goal_map[key] for key in goal.dependency_goal_cids),
                acceptance=acceptance(goal.acceptance),
                evidence_cids=tuple(dict.fromkeys((*goal.evidence_cids, evidence.evidence_cid)))
                if cid in mirrored_goals else goal.evidence_cids)
            goals.append(changed)
            goal_map[cid] = changed.goal_cid

    task_map, tasks = {}, []
    pending = {task.task_cid: task for task in graph.tasks}
    while pending:
        ready = [cid for cid, task in pending.items()
                 if all(dependency in task_map for dependency in task.dependency_task_cids)]
        if not ready:
            raise AcceptanceBindingError("unresolved task dependency")
        for cid in sorted(ready):
            task = pending.pop(cid)
            changed = replace(task, goal_cid=goal_map[task.goal_cid],
                dependency_task_cids=tuple(task_map[key] for key in task.dependency_task_cids),
                acceptance=acceptance(task.acceptance),
                evidence_cids=tuple(dict.fromkeys((*task.evidence_cids, evidence.evidence_cid)))
                if task == producer else task.evidence_cids,
                validations=(*task.validations, validation) if task == producer else task.validations)
            tasks.append(changed)
            task_map[cid] = changed.task_cid
    bound = replace(graph, goals=tuple(goals), tasks=tuple(tasks), evidence=(*graph.evidence, evidence))
    if PromptGoalGraph.from_dict(bound.to_dict()).to_dict() != bound.to_dict():
        raise AcceptanceBindingError("native graph readback differs")
    return AcceptanceBindingResult(bound, expected_plan_root, binding_id, criterion_key,
        producer_task_key, validation.validation_key, goal_map, task_map)
