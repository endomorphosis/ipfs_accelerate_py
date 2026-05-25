"""Objective tracking document helpers for autonomous agent supervisors."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .objective_graph import ObjectiveFinding, ObjectiveGoal, goal_graph, parse_goal_heap, split_terms, utc_now


DEFAULT_ULTIMATE_GOAL = (
    "Make this repository satisfy its stated objective with verifiable code, tests, docs, and runtime evidence."
)
DEFAULT_ROOT_EVIDENCE = (
    "objective goal graph",
    "bundle-local todo shards",
    "AST evidence dataset",
    "embedding evidence scan",
    "LLM merge conflict resolver",
)
DEFAULT_GOAL_PREFIX = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_GOAL_PREFIX", "OBJ-G")
DEFAULT_TRACKING_DOCUMENT_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_DOCUMENT_TITLE", "Objective Heap")
DEFAULT_ROOT_GOAL_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_ROOT_TITLE", "Objective outcome")


@dataclass(frozen=True)
class ObjectiveTrackingResult:
    """Summary of objective tracking document mutations."""

    objective_path: Path
    created: bool
    appended_goal_ids: list[str]
    graph_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        if self.graph_path is not None:
            payload["graph_path"] = str(self.graph_path)
        return payload


def fibonacci_number(index: int) -> int:
    """Return a one-indexed Fibonacci-ish priority bucket."""

    if index <= 1:
        return 1
    left, right = 1, 1
    for _ in range(2, index + 1):
        left, right = right, left + right
    return right


def fibonacci_priority(depth: int, sibling_index: int = 0) -> int:
    """Return a stable integer priority for a goal depth and sibling order."""

    return fibonacci_number(max(1, depth + 2)) * 1000 + max(0, sibling_index)


def infer_goal_prefix(goals: Sequence[ObjectiveGoal], *, fallback: str = DEFAULT_GOAL_PREFIX) -> str:
    """Infer a numeric goal-id prefix from existing goals."""

    prefixes: dict[str, int] = {}
    for goal in goals:
        match = re.match(r"^(.+?)(\d+)$", goal.goal_id)
        if not match:
            continue
        prefixes[match.group(1)] = prefixes.get(match.group(1), 0) + 1
    if not prefixes:
        return fallback
    return sorted(prefixes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def next_goal_id(goals: Sequence[ObjectiveGoal], *, prefix: str | None = None) -> str:
    prefix = prefix or infer_goal_prefix(goals)
    highest = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for goal in goals:
        match = pattern.match(goal.goal_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def render_goal_block(*, goal_id: str, title: str, fields: dict[str, str]) -> str:
    rows = [f"## {goal_id} {title.strip()}", ""]
    for key, value in fields.items():
        rows.append(f"- {key}: {value}")
    return "\n".join(rows).rstrip() + "\n"


def ensure_objective_tracking_document(
    objective_path: Path,
    *,
    ultimate_goal: str = DEFAULT_ULTIMATE_GOAL,
    root_evidence: Sequence[str] = DEFAULT_ROOT_EVIDENCE,
    root_goal_id: str | None = None,
    goal_prefix: str = DEFAULT_GOAL_PREFIX,
    document_title: str = DEFAULT_TRACKING_DOCUMENT_TITLE,
    root_goal_title: str = DEFAULT_ROOT_GOAL_TITLE,
) -> ObjectiveTrackingResult:
    """Create the objective tracking document if it does not exist."""

    if objective_path.exists():
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    root_goal_id = root_goal_id or f"{goal_prefix}000"
    objective_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {document_title}",
            "",
            "This document is the supervisor planning state. It is intentionally separate from markdown todo",
            "boards: todos represent executable work, while this heap represents the objective graph used to",
            "decide what work should exist.",
            "",
            render_goal_block(
                goal_id=root_goal_id,
                title=root_goal_title,
                fields={
                    "Status": "active",
                    "Parent": "",
                    "Fib priority": str(fibonacci_priority(0)),
                    "Track": "ops",
                    "Priority": "P0",
                    "Bundle": "objective/ops/root",
                    "Goal": ultimate_goal,
                    "Evidence": ", ".join(root_evidence),
                    "Outputs": "ipfs_accelerate_py/agent_supervisor, docs",
                    "Validation": f"test -f {objective_path.as_posix()}",
                    "Refinement depth": "0",
                    "Conflict policy": "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts",
                    "Gap task": "Refine this root objective into concrete child goals with code, tests, docs, and runtime evidence.",
                },
            ).rstrip(),
            "",
        ]
    )
    objective_path.write_text(text, encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=True, appended_goal_ids=[root_goal_id])


def existing_refinement_keys(goals: Sequence[ObjectiveGoal]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for goal in goals:
        for parent in goal.parent_goal_ids:
            for evidence in goal.required_evidence:
                keys.add((parent, normalize_evidence_key(evidence)))
    return keys


def normalize_evidence_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def refinement_title(parent_title: str, evidence: str) -> str:
    compact = " ".join(evidence.strip().split())
    if len(compact) > 72:
        compact = compact[:69].rstrip() + "..."
    return f"Prove {compact} for {parent_title}"


def refinement_fields(finding: ObjectiveFinding, *, evidence: str, depth: int, sibling_index: int) -> dict[str, str]:
    outputs = ", ".join(finding.outputs) if finding.outputs else "ipfs_accelerate_py/agent_supervisor, docs, tests"
    return {
        "Status": "active",
        "Parent": finding.goal_id,
        "Fib priority": str(fibonacci_priority(depth, sibling_index)),
        "Track": finding.track,
        "Priority": finding.priority,
        "Bundle": finding.bundle_key,
        "Goal": f"Create concrete implementation, tests, docs, or interface descriptors proving `{evidence}`.",
        "Evidence": evidence,
        "Outputs": outputs,
        "Validation": finding.validation,
        "Refinement depth": str(depth),
        "Embedding query": evidence,
        "AST query": evidence,
        "Parallel lane": finding.parallel_lane,
        "Conflict policy": finding.conflict_policy,
        "Gap task": f"Close the missing objective evidence `{evidence}` with a narrow, verifiable change.",
    }


def append_refinement_goals(
    objective_path: Path,
    findings: Sequence[ObjectiveFinding],
    *,
    max_children_per_finding: int = 3,
    max_depth: int = 4,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Append child goals for missing evidence terms that are still too broad."""

    if not objective_path.exists() or max_children_per_finding <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    graph = goal_graph(goals)
    refinement_keys = existing_refinement_keys(goals)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for finding in findings:
        parent_depth = int(graph.get("depths", {}).get(finding.goal_id, finding.graph_depth))
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            continue
        created_for_parent = 0
        for evidence in finding.missing_evidence:
            key = (finding.goal_id, normalize_evidence_key(evidence))
            if key in refinement_keys:
                continue
            goal_id = allocate_goal_id()
            fields = refinement_fields(
                finding,
                evidence=evidence,
                depth=child_depth,
                sibling_index=created_for_parent,
            )
            appended_blocks.append(
                render_goal_block(
                    goal_id=goal_id,
                    title=refinement_title(finding.title, evidence),
                    fields=fields,
                )
            )
            appended_goal_ids.append(goal_id)
            refinement_keys.add(key)
            created_for_parent += 1
            if created_for_parent >= max_children_per_finding:
                break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def write_objective_graph_artifact(
    *,
    objective_path: Path,
    graph_path: Path,
) -> dict[str, Any]:
    """Write a JSON graph artifact for the current objective heap."""

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8")) if objective_path.exists() else []
    graph = goal_graph(goals)
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_graph",
        "generated_at": utc_now(),
        "objective_path": str(objective_path),
        "goal_count": len(goals),
        "active_goal_count": sum(1 for goal in goals if goal.status in {"active", "todo", "open"}),
        "goals": [
            {
                "goal_id": goal.goal_id,
                "title": goal.title,
                "status": goal.status,
                "fib_priority": goal.priority[0],
                "parents": goal.parent_goal_ids,
                "evidence": goal.required_evidence,
                "track": goal.fields.get("track", "ops"),
                "bundle": goal.fields.get("bundle", ""),
                "refinement_depth": goal.fields.get("refinement_depth", str(graph["depths"].get(goal.goal_id, 0))),
            }
            for goal in sorted(goals, key=lambda item: item.priority)
        ],
        "graph": graph,
    }
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_root_evidence(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        terms.extend(split_terms(value))
    return terms or list(DEFAULT_ROOT_EVIDENCE)
