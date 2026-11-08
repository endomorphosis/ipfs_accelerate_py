"""Lossless native goal contracts for the worldwide legal search campaign.

These contracts declare validation obligations. They do not mint completion
authority or treat the initial Markdown status as execution evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
from collections.abc import Mapping
from typing import Any

from ..objectives.goal_quality import (
    AcceptanceCriterion, EvidenceAuthority, EvidenceProducer, FreshnessPolicy,
    FrozenRootIdentity, GoalQualityError, GoalScope, ObjectiveTypedGoals,
    RefinementBudget, ResourceEnvelope, TypedGoal, UncertaintyDisposition,
    UncertaintyItem, UnsupportedSemantic, ValidationRule,
    lint_objective_typed_goals, validate_objective_typed_goals,
)
from ..objectives.objective_graph import objective_heap_content_id, parse_goal_heap


def _plan_identity(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_typed_goals(
    plan: Mapping[str, Any],
    markdown: str,
    *,
    config_path: Path | str,
    python_executable: str,
    freshness_seconds: int = 3600,
    per_attempt_token_limit: int = 65_536,
) -> ObjectiveTypedGoals:
    """Bind every plan goal to exact task contracts and the native verifier.

    Goal scope consists of finite child-goal or task identities, rather than
    an unbounded repository wildcard. The runtime separately enforces each
    task's concrete write scope and provider budgets. Resource envelopes are
    conservative sums of the finite per-task attempt bounds in the plan.
    """
    if plan.get("schema") != "worldwide-legal-search.plan@1":
        raise GoalQualityError("unsupported worldwide campaign plan")
    if type(freshness_seconds) is not int or freshness_seconds < 1:
        raise GoalQualityError("goal evidence needs a finite freshness horizon")
    if type(per_attempt_token_limit) is not int or per_attempt_token_limit < 1:
        raise GoalQualityError("goal token envelope must be positive")
    goals = list(plan.get("goals", []))
    tasks = {str(task["id"]): task for task in plan.get("tasks", [])}
    goal_ids = [str(goal["id"]) for goal in goals]
    if len(set(goal_ids)) != len(goal_ids) or set(goal_ids) != {goal.goal_id for goal in parse_goal_heap(markdown)}:
        raise GoalQualityError("plan and objective heap goal identities differ")
    if "WWL-G000" not in goal_ids:
        raise GoalQualityError("worldwide campaign root goal is missing")
    root = FrozenRootIdentity("WWL-G000", _plan_identity(plan))
    children = {goal_id: [] for goal_id in goal_ids}
    for goal in goals:
        for parent in goal.get("parent_ids", []):
            if parent not in children:
                raise GoalQualityError("unknown parent goal: " + str(parent))
            children[parent].append(str(goal["id"]))
    contracts = []
    for goal in goals:
        goal_id = str(goal["id"])
        members = tuple(str(item) for item in goal.get("task_ids", []))
        if not members or any(task_id not in tasks for task_id in members):
            raise GoalQualityError("goal has missing or unknown required tasks: " + goal_id)
        finite_scope = tuple(sorted(children[goal_id] or members))
        if len(finite_scope) > 32:
            raise GoalQualityError("goal needs bounded decomposition before admission: " + goal_id)
        acceptance_id = goal_id + ":accepted-task-contracts"
        producer_id = goal_id + ":native-receipt-verifier"
        validation_id = goal_id + ":current-tree-validation"
        command = shlex.join([
            str(python_executable), "-m",
            "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime",
            "validate-goal", "--config", str(Path(config_path).absolute()),
            "--goal-id", goal_id,
        ])
        attempts = sum(int(tasks[task_id]["bounds"]["max_attempts"]) for task_id in members)
        wall = sum(int(tasks[task_id]["bounds"]["max_attempts"]) * int(tasks[task_id]["bounds"]["wall_seconds_per_attempt"]) for task_id in members)
        artifacts = sum(len(tasks[task_id]["outputs"]) + 1 for task_id in members)
        contracts.append(TypedGoal(
            goal_id=goal_id, root=root, outcome=str(goal["title"]),
            scope=GoalScope(include=finite_scope, exclude=("unrelated-campaign-state", "private-browser-credentials")),
            assumptions=("Dependencies and writes are admitted through the native campaign task source.", "Source availability and redistribution rights require separate verified evidence."),
            non_goals=("Infer complete worldwide legal coverage from dataset names.", "Grant task completion from model prose or the initial Markdown status."),
            acceptance_criteria=(AcceptanceCriterion(
                criterion_id=acceptance_id,
                statement=str(goal["acceptance"]) + " Required task contracts: " + ", ".join(members) + ".",
                evidence_producer_ids=(producer_id,), validation_rule_ids=(validation_id,),
                completion_signal="native_verified_required_task_receipts == required_task_ids and accepted_tree_matches and unknown_effects == 0",
            ),),
            evidence_producers=(EvidenceProducer(
                producer_id=producer_id, kind="native_task_receipt_verifier",
                output_schema="worldwide-legal-search.goal-validation@1",
                authority=EvidenceAuthority.VALIDATION,
                capability_id="native-current-tree-task-receipt-validation", independent=True,
            ),),
            validation_rules=(ValidationRule(
                rule_id=validation_id, command=command, producer_id=producer_id,
                criterion_ids=(acceptance_id,), hermetic=False,
            ),),
            freshness=FreshnessPolicy(max_age_seconds=freshness_seconds),
            resources=ResourceEnvelope(
                max_wall_seconds=max(1, wall), max_tokens=max(1, attempts) * per_attempt_token_limit,
                max_artifacts=max(1, artifacts), max_parallelism=1,
                max_scope_items=max(1, len(finite_scope)),
            ),
            uncertainties=(UncertaintyItem(
                uncertainty_id=goal_id + ":source-and-execution-prerequisites",
                statement="External corpus, rights and execution prerequisites may be unavailable.",
                disposition=UncertaintyDisposition.MITIGATED,
                impact="Unresolved prerequisites prevent completion of their dependent task contracts.",
                resolution="Retain typed blockers and verify independent current-tree receipts before completion.",
            ),),
            unsupported_semantics=(UnsupportedSemantic(
                semantic_id=goal_id + ":applicability-and-coverage-limits",
                statement="Geographic retrieval is not proof of universal legal applicability or complete collection.",
                fallback="Preserve source provenance, unknown status and explicit coverage gaps.",
            ),),
            refinement_budget=RefinementBudget(
                max_rounds=3, max_children=max(2, len(children[goal_id]) or len(members)),
                max_depth=6, max_debt_items=32, max_tokens=max(1, len(members)) * per_attempt_token_limit,
            ),
            authorized_completion_producer_ids=(),
        ))
    document = ObjectiveTypedGoals(objective_heap_id=objective_heap_content_id(markdown), goals=tuple(contracts))
    validate_objective_typed_goals(markdown, document)
    reports = lint_objective_typed_goals(document)
    errors = [(report.goal_id, tuple(code.value for code in report.debt_codes)) for report in reports if not report.accepted]
    if errors:
        raise GoalQualityError("worldwide goal quality admission failed: " + repr(errors))
    return document


def lint_typed_goals(document: ObjectiveTypedGoals):
    """Expose the native reports without inventing a parallel admission rule."""
    return lint_objective_typed_goals(document)


def admit_typed_goals(plan: Mapping[str, Any], markdown: str, *, config_path: Path | str, python_executable: str) -> dict[str, Any]:
    """Return native linted declaration evidence; does not claim execution."""
    document = build_typed_goals(plan, markdown, config_path=config_path, python_executable=python_executable)
    reports = lint_typed_goals(document)
    return {"schema": "worldwide-legal-search.typed-goal-admission@1", "document": document.to_dict(),
            "lint_reports": [report.to_dict() for report in reports], "all_accepted": all(report.accepted for report in reports),
            "execution_completed": False}
