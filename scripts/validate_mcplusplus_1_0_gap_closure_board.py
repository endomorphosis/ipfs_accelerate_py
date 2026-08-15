#!/usr/bin/env python3
"""Fail-closed validator for the MCP++ 1.0 gap-closure program."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/mcplusplus_1_0_gap_closure.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/mcplusplus_1_0_gap_closure.todo.md"
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json"
)

BOARD_NAMESPACE = "mcplusplus-1-0-gap-closure-v1"
TASK_IDS = tuple(f"MCPP-{index:03d}" for index in range(84))
GOAL_IDS = (
    "MCPP-G000",
    "MCPP-G010",
    "MCPP-G020",
    "MCPP-G030",
    "MCPP-G040",
    "MCPP-G050",
    "MCPP-G060",
    "MCPP-G070",
    "MCPP-G080",
    "MCPP-G090",
    "MCPP-G100",
    "MCPP-G110",
    "MCPP-G120",
    "MCPP-G130",
    "MCPP-G140",
    "MCPP-G150",
    "MCPP-G160",
    "MCPP-G170",
)
INITIAL_COMPLETED = ("MCPP-000",)
INITIAL_READY = ("MCPP-001",)
TERMINAL_TASK = "MCPP-083"

GOAL_STATES = frozenset(
    {
        "active",
        "provisionally_complete",
        "verified_complete",
        "analysis_inconclusive",
        "blocked",
        "reopened",
    }
)
TASK_STATES = frozenset({"todo", "in_progress", "blocked", "completed"})
REQUIRED_GOAL_FIELDS = (
    "status",
    "parent",
    "depends_on",
    "fib_priority",
    "track",
    "priority",
    "bundle",
    "parallel_lane",
    "resource_class",
    "goal",
    "evidence",
    "evidence_criteria",
    "evidence_source_policy",
    "outputs",
    "predicted_files",
    "interfaces",
    "validation",
    "acceptance",
    "gap_task",
    "refinement",
    "embedding_query",
    "ast_query",
    "conflict_policy",
)
REQUIRED_TASK_FIELDS = (
    "status",
    "completion",
    "is schedulable",
    "review only",
    "priority",
    "track",
    "depends on",
    "goal id",
    "outputs",
    "validation",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "estimated tokens",
    "implementation timeout seconds",
    "predicted files",
    "interfaces",
    "allow concurrent with",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "symbolic first",
    "llm context budget bytes",
    "acceptance",
    "embedding query",
)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _fail(errors: list[str]) -> int:
    if not errors:
        return 0
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


def validate() -> list[str]:
    errors: list[str] = []
    for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH, SCHEDULER_PATH):
        if not path.is_file():
            errors.append(f"missing {path.relative_to(REPO_ROOT)}")
    if errors:
        return errors

    try:
        board = load_configured_board(SCHEDULER_PATH, repo_root=REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - validator must surface config errors
        errors.append(f"scheduler load failed: {exc}")
        return errors

    if board.board_namespace != BOARD_NAMESPACE:
        errors.append(f"board_namespace {board.board_namespace!r}")
    if board.task_prefix not in {"MCPP-", "## MCPP-"}:
        errors.append(f"task_prefix {board.task_prefix!r}")
    if board.max_lanes != 6:
        errors.append(f"max_lanes {board.max_lanes}")
    if board.merge_target_branch != "codex/mcplusplus-1.0-gap-closure":
        errors.append(f"merge_target_branch {board.merge_target_branch!r}")

    payload = dict(board.payload)
    initial = payload.get("initial_projection") or {}
    if initial.get("task_count") != 84:
        errors.append(f"initial task_count {initial.get('task_count')}")
    if initial.get("goal_count") != 18:
        errors.append(f"initial goal_count {initial.get('goal_count')}")
    if tuple(initial.get("completed_task_ids") or ()) != INITIAL_COMPLETED:
        errors.append("initial completed_task_ids must be exactly MCPP-000")
    if tuple(initial.get("ready_task_ids") or ()) != INITIAL_READY:
        errors.append("initial ready_task_ids must be exactly MCPP-001")
    if initial.get("terminal_task_id") != TERMINAL_TASK:
        errors.append("terminal_task_id must be MCPP-083")
    if initial.get("root_goal_id") != "MCPP-G000":
        errors.append("root_goal_id must be MCPP-G000")
    if payload.get("objective_refill_enabled") is not False:
        errors.append("objective_refill_enabled must be false")
    if payload.get("codebase_refill_enabled") is not False:
        errors.append("codebase_refill_enabled must be false")

    goals = parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))
    if [goal.goal_id for goal in goals] != list(GOAL_IDS):
        errors.append(
            "goal ids drifted: "
            + ",".join(goal.goal_id for goal in goals)
        )
    goal_by_id = {goal.goal_id: goal for goal in goals}
    for goal in goals:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing:
            errors.append(f"{goal.goal_id} missing fields: {', '.join(missing)}")
        if goal.status not in GOAL_STATES:
            errors.append(f"{goal.goal_id} status {goal.status!r}")
        if goal.goal_id == "MCPP-G000" and goal.status != "blocked":
            errors.append("root goal must be blocked until children complete")
        if goal.goal_id == "MCPP-G000" and str(goal.fields.get("review_only", "")).lower() != "true":
            errors.append("root goal must be review only")

    tasks = parse_task_text(
        TODO_PATH.read_text(encoding="utf-8"),
        path=TODO_PATH,
        task_header_prefix="## MCPP-",
    )
    if [task.task_id for task in tasks] != list(TASK_IDS):
        errors.append(
            "task ids drifted count="
            + str(len(tasks))
            + " first="
            + (tasks[0].task_id if tasks else "?")
            + " last="
            + (tasks[-1].task_id if tasks else "?")
        )

    task_by_id = {task.task_id: task for task in tasks}
    adjacency: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, int] = {task_id: 0 for task_id in TASK_IDS}
    for task in tasks:
        metadata = {key.lower(): value for key, value in task.metadata.items()}
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task.task_id} missing fields: {', '.join(missing)}")
        status = metadata.get("status", "")
        if status not in TASK_STATES:
            errors.append(f"{task.task_id} status {status!r}")
        if metadata.get("board namespace") != BOARD_NAMESPACE:
            errors.append(f"{task.task_id} board namespace")
        goal_id = metadata.get("goal id", "")
        if goal_id not in goal_by_id:
            errors.append(f"{task.task_id} unknown goal {goal_id!r}")
        expected_lane = {
            0: "mcpp-lane-spec",
            1: "mcpp-lane-schema",
            2: "mcpp-lane-crypto",
            3: "mcpp-lane-state",
            4: "mcpp-lane-transport",
            5: "mcpp-lane-runtime",
        }[int(task.task_id.split("-")[1]) % 6]
        if metadata.get("parallel lane") != expected_lane:
            errors.append(
                f"{task.task_id} parallel lane "
                f"{metadata.get('parallel lane')!r} != {expected_lane!r}"
            )
        if not metadata.get("acceptance", "").strip():
            errors.append(f"{task.task_id} empty acceptance")
        if not metadata.get("validation", "").strip():
            errors.append(f"{task.task_id} empty validation")
        if not metadata.get("outputs", "").strip():
            errors.append(f"{task.task_id} empty outputs")
        deps = _csv(metadata.get("depends on", ""))
        for dep in deps:
            if dep not in task_by_id:
                errors.append(f"{task.task_id} depends on unknown {dep}")
            else:
                adjacency[dep].append(task.task_id)
                incoming[task.task_id] += 1
        for other in _csv(metadata.get("allow concurrent with", "")):
            if other not in task_by_id:
                errors.append(f"{task.task_id} concurrent with unknown {other}")

    if task_by_id.get("MCPP-000") is None:
        errors.append("missing MCPP-000")
    else:
        if task_by_id["MCPP-000"].status != "completed":
            errors.append("MCPP-000 must be completed")
        if str(task_by_id["MCPP-000"].metadata.get("review only", "")).lower() != "true":
            errors.append("MCPP-000 must be review only")

    ready = []
    for task_id, task in task_by_id.items():
        if task.status == "completed":
            continue
        deps = _csv(task.metadata.get("depends on", ""))
        if all(task_by_id[dep].status == "completed" for dep in deps if dep in task_by_id):
            ready.append(task_id)
    if tuple(ready) != INITIAL_READY:
        errors.append(f"ready set {ready!r} != {list(INITIAL_READY)!r}")

    queue = deque([task_id for task_id, count in incoming.items() if count == 0])
    seen = 0
    incoming_copy = dict(incoming)
    while queue:
        node = queue.popleft()
        seen += 1
        for child in adjacency[node]:
            incoming_copy[child] -= 1
            if incoming_copy[child] == 0:
                queue.append(child)
    if seen != len(TASK_IDS):
        errors.append(f"task dependency graph is cyclic or disconnected ({seen} visited)")

    groups = payload.get("task_groups") or {}
    grouped = []
    for goal_id in GOAL_IDS:
        grouped.extend(groups.get(goal_id) or [])
    if grouped != list(TASK_IDS):
        errors.append("scheduler task_groups do not cover MCPP-000 through MCPP-083 exactly once")

    digest = hashlib.sha256(TODO_PATH.read_bytes()).hexdigest()
    report = {
        "valid": not errors,
        "errors": errors,
        "taskboard_sha256": f"sha256:{digest}",
        "goals": len(goals) if "goals" in locals() else 0,
        "tasks": len(tasks) if "tasks" in locals() else 0,
        "ready": ready if "ready" in locals() else [],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()
    if not args.check_all:
        print(
            json.dumps({"valid": False, "errors": ["pass --check-all"]}),
            file=sys.stderr,
        )
        return 2
    errors = validate()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
