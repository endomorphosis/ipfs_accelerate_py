#!/usr/bin/env python3
"""Fail-closed validator for the DuckDB + Quack control-plane program."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    ConfiguredBoardError,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PLAN_PATH = (
    REPO_ROOT
    / "docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md"
)
OBJECTIVE_PATH = (
    REPO_ROOT
    / "docs/architecture/agent_supervisor_duckdb_quack_control_plane.objectives.md"
)
TODO_PATH = (
    REPO_ROOT
    / "docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md"
)
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_duckdb_quack_control_plane_scheduler.json"
)
VALIDATOR_PATH = Path(__file__).resolve()
PARSER_TEST_PATH = (
    REPO_ROOT
    / "test/api/test_agent_supervisor_duckdb_quack_control_plane_board.py"
)

BOARD_NAMESPACE = "agent-supervisor-duckdb-quack-control-plane-v1"
TARGET_BRANCH = "agent/duckdb-quack-control-plane-20260808"
TASK_IDS = tuple(f"DQP-{index:03d}" for index in range(40))
GOAL_IDS = (
    "DQP-G000",
    "DQP-G010",
    "DQP-G020",
    "DQP-G030",
    "DQP-G040",
    "DQP-G050",
    "DQP-G060",
    "DQP-G070",
    "DQP-G080",
    "DQP-G090",
)
INITIAL_COMPLETED = ("DQP-000",)
INITIAL_READY = ("DQP-001", "DQP-002", "DQP-003", "DQP-004", "DQP-009")
TERMINAL_TASK = "DQP-039"
LANE_COUNT = 4

TASK_STATES = frozenset({"todo", "completed"})
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
REQUIRED_GOAL_FIELDS = (
    "status",
    "review_only",
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
REQUIRED_PLAN_TERMS = (
    "versioned duckdb control plane",
    "quack",
    "beta/experimental",
    "one writer path",
    "schema_migrations",
    "daemon_sessions",
    "worktree_snapshots",
    "ast_nodes",
    "mutations",
    "impact_closures",
    "context_manifests",
    "llm churn reduction",
    "deterministic exports",
    "legacy retirement",
    "rollback",
    "loopback",
    "optimistic concurrency",
    "one failure domain",
)
CONTROL_RELATIVE_PATHS = (
    ".gitignore",
    PLAN_PATH.relative_to(REPO_ROOT).as_posix(),
    OBJECTIVE_PATH.relative_to(REPO_ROOT).as_posix(),
    TODO_PATH.relative_to(REPO_ROOT).as_posix(),
    SCHEDULER_PATH.relative_to(REPO_ROOT).as_posix(),
    VALIDATOR_PATH.relative_to(REPO_ROOT).as_posix(),
    PARSER_TEST_PATH.relative_to(REPO_ROOT).as_posix(),
)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _contained_relative(value: str) -> bool:
    path = PurePosixPath(value.strip().replace("\\", "/"))
    return bool(
        value.strip()
        and not path.is_absolute()
        and ".." not in path.parts
        and "\x00" not in value
    )


def _task_shard(task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % LANE_COUNT


def _acyclic(adjacency: Mapping[str, Iterable[str]]) -> tuple[bool, list[str]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: list[str] = []

    def visit(node: str, trail: list[str]) -> bool:
        if node in visiting:
            cycle.extend([*trail, node])
            return False
        if node in visited:
            return True
        visiting.add(node)
        for dependency in adjacency.get(node, ()):
            if not visit(dependency, [*trail, node]):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    passed = all(visit(node, []) for node in adjacency if node not in visited)
    return passed, cycle


def _append(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def validate_program() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (
            PLAN_PATH,
            OBJECTIVE_PATH,
            TODO_PATH,
            SCHEDULER_PATH,
            VALIDATOR_PATH,
            PARSER_TEST_PATH,
        )
        if not path.is_file()
    ]
    _append(checks, errors, name="control_files_present", passed=not missing, detail=missing)
    if missing:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/duckdb-quack-board-validation@1",
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVE_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    config = _load_json(SCHEDULER_PATH)

    task_heading_ids = re.findall(r"^## (DQP-\d{3})\b", todo_text, flags=re.MULTILINE)
    goal_heading_ids = re.findall(r"^## (DQP-G\d{3})\b", objective_text, flags=re.MULTILINE)
    tasks = parse_task_text(todo_text, path=TODO_PATH, task_header_prefix="## DQP-")
    goals = parse_goal_heap(objective_text)

    _append(
        checks,
        errors,
        name="task_population",
        passed=tuple(task_heading_ids) == TASK_IDS and tuple(task.task_id for task in tasks) == TASK_IDS,
        detail={"expected": list(TASK_IDS), "headings": task_heading_ids, "parsed": [task.task_id for task in tasks]},
    )
    _append(
        checks,
        errors,
        name="task_ids_unique",
        passed=len(set(task_heading_ids)) == len(task_heading_ids),
        detail=[item for item, count in Counter(task_heading_ids).items() if count != 1],
    )
    _append(
        checks,
        errors,
        name="goal_population",
        passed=tuple(goal_heading_ids) == GOAL_IDS and tuple(goal.goal_id for goal in goals) == GOAL_IDS,
        detail={"expected": list(GOAL_IDS), "headings": goal_heading_ids, "parsed": [goal.goal_id for goal in goals]},
    )

    missing_task_fields: dict[str, list[str]] = {}
    invalid_task_values: dict[str, list[str]] = {}
    task_by_id = {task.task_id: task for task in tasks}
    for task in tasks:
        missing_fields = [field for field in REQUIRED_TASK_FIELDS if field not in task.metadata]
        if missing_fields:
            missing_task_fields[task.task_id] = missing_fields
        invalid: list[str] = []
        if task.status not in TASK_STATES:
            invalid.append(f"status={task.status!r}")
        if task.metadata.get("board namespace") != BOARD_NAMESPACE:
            invalid.append("board namespace")
        if task.metadata.get("is schedulable", "").lower() != "true":
            invalid.append("is schedulable")
        if task.metadata.get("symbolic first", "").lower() != "true":
            invalid.append("symbolic first")
        if task.metadata.get("goal id") not in GOAL_IDS:
            invalid.append("goal id")
        if not task.outputs or not all(_contained_relative(path) for path in task.outputs):
            invalid.append("outputs")
        if not task.validation or not task.acceptance.strip():
            invalid.append("validation/acceptance")
        if invalid:
            invalid_task_values[task.task_id] = invalid
    _append(checks, errors, name="task_required_fields", passed=not missing_task_fields, detail=missing_task_fields)
    _append(checks, errors, name="task_field_values", passed=not invalid_task_values, detail=invalid_task_values)

    dependency_errors: dict[str, list[str]] = {}
    adjacency: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        unknown = [dependency for dependency in task.depends_on if dependency not in task_by_id]
        self_edges = [dependency for dependency in task.depends_on if dependency == task.task_id]
        duplicates = [item for item, count in Counter(task.depends_on).items() if count > 1]
        issues = [*(f"unknown:{item}" for item in unknown), *(f"self:{item}" for item in self_edges), *(f"duplicate:{item}" for item in duplicates)]
        if issues:
            dependency_errors[task.task_id] = issues
        adjacency[task.task_id] = tuple(task.depends_on)
    _append(checks, errors, name="task_dependencies_local_unique", passed=not dependency_errors, detail=dependency_errors)
    dag_ok, cycle = _acyclic(adjacency)
    graph = materialize_task_dependency_dag(tasks)
    _append(
        checks,
        errors,
        name="task_dependency_dag",
        passed=dag_ok and not graph.invalid_task_cids,
        detail={"cycle": cycle, "invalid_task_cids": list(graph.invalid_task_cids), "edge_count": len(graph.edges)},
    )

    seed_completed = set(INITIAL_COMPLETED)
    calculated_initial_ready = tuple(
        task.task_id
        for task in tasks
        if task.task_id not in seed_completed
        and set(task.depends_on).issubset(seed_completed)
    )
    current_completed = {task.task_id for task in tasks if task.status == "completed"}
    _append(
        checks,
        errors,
        name="initial_projection",
        passed=calculated_initial_ready == INITIAL_READY and set(INITIAL_COMPLETED).issubset(current_completed),
        detail={"expected_ready": list(INITIAL_READY), "calculated_ready": list(calculated_initial_ready), "current_completed": sorted(current_completed)},
    )
    ready_by_shard = {
        str(index): [task_id for task_id in INITIAL_READY if _task_shard(task_id) == index]
        for index in range(LANE_COUNT)
    }
    _append(
        checks,
        errors,
        name="initial_shard_coverage",
        passed=all(ready_by_shard[str(index)] for index in range(LANE_COUNT)),
        detail=ready_by_shard,
    )

    output_owners: dict[str, list[str]] = {}
    for task in tasks:
        for output in task.outputs:
            output_owners.setdefault(output, []).append(task.task_id)
    duplicate_outputs = {
        path: owners
        for path, owners in output_owners.items()
        if len(owners) > 1
    }
    protected_violations = {
        task.task_id: sorted(set(task.outputs).intersection(CONTROL_RELATIVE_PATHS))
        for task in tasks
        if task.task_id != "DQP-000" and set(task.outputs).intersection(CONTROL_RELATIVE_PATHS)
    }
    _append(checks, errors, name="exclusive_output_ownership", passed=not duplicate_outputs, detail=duplicate_outputs)
    _append(checks, errors, name="control_paths_protected", passed=not protected_violations, detail=protected_violations)

    goal_by_id = {goal.goal_id: goal for goal in goals}
    missing_goal_fields: dict[str, list[str]] = {}
    goal_dependency_errors: dict[str, list[str]] = {}
    goal_adjacency: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        missing_fields = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing_fields:
            missing_goal_fields[goal.goal_id] = missing_fields
        dependencies = _csv(goal.fields.get("depends_on", ""))
        parent = goal.fields.get("parent", "").strip()
        references = (*dependencies, *((parent,) if parent else ()))
        unknown = [item for item in references if item not in goal_by_id]
        if unknown:
            goal_dependency_errors[goal.goal_id] = unknown
        goal_adjacency[goal.goal_id] = tuple((*dependencies, *((parent,) if parent else ())))
    goal_dag_ok, goal_cycle = _acyclic(goal_adjacency)
    _append(checks, errors, name="goal_required_fields", passed=not missing_goal_fields, detail=missing_goal_fields)
    _append(checks, errors, name="goal_dependencies", passed=not goal_dependency_errors and goal_dag_ok, detail={"unknown": goal_dependency_errors, "cycle": goal_cycle})

    missing_terms = [term for term in REQUIRED_PLAN_TERMS if term.casefold() not in plan_text.casefold()]
    _append(checks, errors, name="plan_scope", passed=not missing_terms, detail=missing_terms)

    projection = config.get("initial_projection") if isinstance(config.get("initial_projection"), dict) else {}
    lane_initial = {
        str(lane.get("index")): list(lane.get("initial_task_ids") or [])
        for lane in config.get("lanes", [])
        if isinstance(lane, dict)
    }
    config_errors: list[str] = []
    expected_values = {
        "taskboard_path": TODO_PATH.relative_to(REPO_ROOT).as_posix(),
        "objectives_path": OBJECTIVE_PATH.relative_to(REPO_ROOT).as_posix(),
        "plan_path": PLAN_PATH.relative_to(REPO_ROOT).as_posix(),
        "validator_path": VALIDATOR_PATH.relative_to(REPO_ROOT).as_posix(),
        "task_prefix": "DQP-",
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": TARGET_BRANCH,
        "max_lanes": LANE_COUNT,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "worktree_submodule_paths": [],
    }
    for field, expected in expected_values.items():
        if config.get(field) != expected:
            config_errors.append(f"{field}={config.get(field)!r}, expected {expected!r}")
    expected_projection = {
        "task_count": len(TASK_IDS),
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": len(GOAL_IDS),
        "root_goal_id": GOAL_IDS[0],
    }
    if projection != expected_projection:
        config_errors.append("initial_projection mismatch")
    if lane_initial != ready_by_shard:
        config_errors.append(f"lane initial tasks {lane_initial!r} != shard map {ready_by_shard!r}")
    protected = config.get("protected_paths")
    if not isinstance(protected, list) or set(protected) != set(CONTROL_RELATIVE_PATHS):
        config_errors.append("protected_paths mismatch")
    runtime_paths = config.get("runtime_paths") if isinstance(config.get("runtime_paths"), dict) else {}
    if not str(runtime_paths.get("root") or "").startswith("state/"):
        config_errors.append("runtime root must be under ignored state/")
    provider = config.get("provider") if isinstance(config.get("provider"), dict) else {}
    if provider.get("fallback_trigger") != "primary_quota_exhausted":
        config_errors.append("provider fallback must require verified primary quota exhaustion")
    _append(checks, errors, name="scheduler_contract", passed=not config_errors, detail=config_errors)

    try:
        board = load_configured_board(SCHEDULER_PATH, repo_root=REPO_ROOT)
        structural_detail: Any = {
            "board_namespace": board.board_namespace,
            "max_lanes": board.max_lanes,
            "runtime_root": board.runtime_paths["root"],
        }
        structural_ok = True
    except ConfiguredBoardError as exc:
        structural_detail = str(exc)
        structural_ok = False
    _append(checks, errors, name="configured_board_loader", passed=structural_ok, detail=structural_detail)

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/duckdb-quack-board-validation@1",
        "valid": not errors,
        "board_namespace": BOARD_NAMESPACE,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "current_completed_task_ids": sorted(current_completed),
        "initial_ready_task_ids": list(INITIAL_READY),
        "initial_ready_by_shard": ready_by_shard,
        "terminal_task_id": TERMINAL_TASK,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="Validate all sealed program invariants.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if not args.check_all:
        _parser().error("--check-all is required")
    try:
        report = validate_program()
    except Exception as exc:
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/duckdb-quack-board-validation@1",
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "checks": [],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
