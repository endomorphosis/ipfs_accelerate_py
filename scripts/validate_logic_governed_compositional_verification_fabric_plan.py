#!/usr/bin/env python3
"""Fail-closed consistency validator for the LGCVF successor projections.

This validates plan structure, ancestry, and cross-projection agreement only.
It is not source verification, release qualification, or production authority.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)
from scripts.emit_logic_governed_compositional_verification_fabric_plan import (
    IMMEDIATE_PREDECESSOR_PLAN_CID,
    PHASES,
    PLAN_REVISION,
    PREDECESSOR_ARCHIVE,
    PROGRAM,
    PROGRAM_ANCESTOR_PLAN_CID,
    ROOT_GOAL,
    TASKS,
    build_plan,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md"
OBJECTIVES_PATH = (
    REPO_ROOT / "docs/architecture/logic_governed_compositional_verification_fabric.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/logic_governed_compositional_verification_fabric.todo.md"
FORMAL_PATH = (
    REPO_ROOT
    / "data/agent_supervisor/logic_governed_compositional_verification_fabric/formal_work_plan.json"
)
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json"
)
TASK_HEADING = re.compile(r"^## (LGCVF-\d{3}) (.+)$", re.MULTILINE)
GOAL_HEADING = re.compile(r"^## (LGCVF-G\d{3}) (.+)$", re.MULTILINE)


def _task_blocks(text: str) -> dict[str, dict[str, str]]:
    matches = list(TASK_HEADING.finditer(text))
    result: dict[str, dict[str, str]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[match.end() : end].splitlines():
            if line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                fields[key.strip().casefold()] = value.strip()
        result[match.group(1)] = fields
    return result


def _csv(value: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip() and item.strip().casefold() not in {"none", "n/a"}
    )


def _cycle(edges: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> tuple[str, ...]:
        if node in visited:
            return ()
        if node in visiting:
            start = visiting.index(node)
            return (*visiting[start:], node)
        visiting.append(node)
        for dependency in edges.get(node, ()):
            found = visit(dependency)
            if found:
                return found
        visiting.pop()
        visited.add(node)
        return ()

    for node in sorted(edges):
        found = visit(node)
        if found:
            return found
    return ()


def validate() -> dict[str, Any]:
    errors: list[str] = []
    for path in (
        PLAN_PATH,
        OBJECTIVES_PATH,
        TODO_PATH,
        FORMAL_PATH,
        PREDECESSOR_ARCHIVE,
        SCHEDULER_PATH,
    ):
        if not path.is_file():
            errors.append(f"missing required projection: {path.relative_to(REPO_ROOT)}")
    if errors:
        return {"errors": errors, "valid": False}

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    expected_tasks = {str(item["id"]): item for item in TASKS}
    blocks = _task_blocks(todo_text)
    if tuple(blocks) != tuple(expected_tasks):
        errors.append("task board order/identity differs from the canonical task table")

    required_fields = {
        "status",
        "completion",
        "is schedulable",
        "review only",
        "priority",
        "track",
        "goal id",
        "parent goal id",
        "subgoal id",
        "depends on",
        "owning repository",
        "outputs",
        "predicted files",
        "validation",
        "acceptance",
        "board namespace",
        "parallel lane",
        "conflict policy",
        "required evidence",
    }
    edges: dict[str, tuple[str, ...]] = {}
    for task_id, task in expected_tasks.items():
        fields = blocks.get(task_id, {})
        missing = sorted(required_fields.difference(fields))
        if missing:
            errors.append(f"{task_id}: missing fields {missing}")
            continue
        dependencies = _csv(fields["depends on"])
        expected_dependencies = tuple(str(item) for item in task["deps"])
        if dependencies != expected_dependencies:
            errors.append(f"{task_id}: dependency projection differs")
        edges[task_id] = dependencies
        if fields["goal id"] != ROOT_GOAL or fields["parent goal id"] != ROOT_GOAL:
            errors.append(f"{task_id}: root goal binding differs")
        if fields["subgoal id"] != task["phase"]:
            errors.append(f"{task_id}: subgoal binding differs")
        if fields["board namespace"] != PROGRAM:
            errors.append(f"{task_id}: board namespace differs")
        if fields["outputs"] != fields["predicted files"]:
            errors.append(f"{task_id}: Outputs and Predicted files differ")
        status = fields["status"]
        construction_status = str(task["status"])
        if construction_status.startswith("blocked_"):
            if status != "blocked" or construction_status not in fields.get("blocked reason", ""):
                errors.append(f"{task_id}: precise blocked status is not preserved")
            if fields["is schedulable"].casefold() != "false":
                errors.append(f"{task_id}: blocked task is schedulable")
        elif status != construction_status:
            errors.append(f"{task_id}: status differs from construction snapshot")
        if status == "todo" and fields["is schedulable"].casefold() != "true":
            errors.append(f"{task_id}: todo task must be schedulable")
        if status == "completed" and fields["is schedulable"].casefold() != "false":
            errors.append(f"{task_id}: completed construction task must not redispatch")
        for dependency in dependencies:
            if dependency not in expected_tasks:
                errors.append(f"{task_id}: unknown dependency {dependency}")
    found_cycle = _cycle(edges)
    if found_cycle:
        errors.append(f"task dependency cycle: {found_cycle}")

    objective_ids = tuple(match.group(1) for match in GOAL_HEADING.finditer(objective_text))
    expected_goal_ids = (ROOT_GOAL, *(str(item["id"]) for item in PHASES))
    if objective_ids != expected_goal_ids:
        errors.append("objective heading order/identity differs from phase table")

    for noun, text in (
        ("human plan", plan_text),
        ("objective heap", objective_text),
        ("task board", todo_text),
    ):
        if IMMEDIATE_PREDECESSOR_PLAN_CID not in text:
            errors.append(f"{noun}: immediate predecessor plan CID missing")
        if PROGRAM_ANCESTOR_PLAN_CID not in text:
            errors.append(f"{noun}: original program ancestor CID missing")
        if PROGRAM not in text:
            errors.append(f"{noun}: board namespace missing")
    if "not production-authorized" not in plan_text:
        errors.append("human plan lacks explicit non-production disposition")
    if "blocked_manual" not in todo_text or "blocked_external_authority" not in todo_text:
        errors.append("task board lacks precise external/manual blockers")

    try:
        payload = json.loads(FORMAL_PATH.read_text(encoding="utf-8"))
        persisted = FormalWorkPlan.from_dict(payload)
        generated = build_plan()
        scheduler = json.loads(SCHEDULER_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"formal plan cannot be reconstructed: {exc}")
    else:
        if persisted.content_id != generated.content_id:
            errors.append("formal plan differs from canonical generator output")
        if (
            persisted.metadata.get("predecessor_plan_cid")
            != IMMEDIATE_PREDECESSOR_PLAN_CID
        ):
            errors.append("formal plan predecessor binding differs")
        if persisted.metadata.get("plan_revision") != PLAN_REVISION:
            errors.append("formal plan revision differs")
        if (
            persisted.metadata.get("program_ancestor_plan_cid")
            != PROGRAM_ANCESTOR_PLAN_CID
        ):
            errors.append("formal plan original ancestor binding differs")
        if tuple(item.task_id for item in persisted.tasks) != tuple(expected_tasks):
            errors.append("formal plan task order/identity differs")
        if tuple(item.subgoal_id for item in persisted.subgoals) != expected_goal_ids[1:]:
            errors.append("formal plan subgoal order/identity differs")
        logical_task_prefix = persisted.metadata.get("task_prefix")
        if not isinstance(logical_task_prefix, str) or not re.fullmatch(
            r"[A-Z][A-Z0-9]*-", logical_task_prefix
        ):
            errors.append("formal plan logical task prefix is not canonical")
        elif not isinstance(scheduler, dict) or scheduler.get("task_prefix") != (
            "## " + logical_task_prefix
        ):
            errors.append(
                "scheduler Markdown task selector differs from the formal logical prefix"
            )

    try:
        archived_payload = json.loads(PREDECESSOR_ARCHIVE.read_text(encoding="utf-8"))
        archived = FormalWorkPlan.from_dict(archived_payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"archived predecessor cannot be reconstructed: {exc}")
    else:
        if archived.content_id != IMMEDIATE_PREDECESSOR_PLAN_CID:
            errors.append("archived predecessor identity differs from its filename")

    return {
        "board_namespace": PROGRAM,
        "errors": errors,
        "formal_plan_content_id": (persisted.content_id if "persisted" in locals() else ""),
        "goals": len(expected_goal_ids),
        "immediate_predecessor_plan_cid": IMMEDIATE_PREDECESSOR_PLAN_CID,
        "plan_revision": PLAN_REVISION,
        "program_ancestor_plan_cid": PROGRAM_ANCESTOR_PLAN_CID,
        "subgoals": len(PHASES),
        "tasks": len(TASKS),
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="validate every projection")
    parser.parse_args()
    report = validate()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
