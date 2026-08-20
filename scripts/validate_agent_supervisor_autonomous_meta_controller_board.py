#!/usr/bin/env python3
"""Fail-closed structural validator for the APMC program and P0 tranche."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PROGRAM_ID = "agent-supervisor-autonomous-meta-controller-v1"
TASK_PREFIX = "APMC-"
ROOT_OBJECTIVE = "APMC-G000"
TASK_IDS = tuple(f"APMC-{index:03d}" for index in range(21))
GOAL_IDS = tuple(["APMC-G000", *(f"APMC-G{index:03d}" for index in range(10, 111, 10))])
EXPECTED_PRELAUNCH_QUALIFIED_TASK_IDS = (
    "APMC-000",
    "APMC-001",
    "APMC-018",
    "APMC-002",
    "APMC-003",
    "APMC-004",
    "APMC-005",
)
EXPECTED_PRELAUNCH_COMPLETED_TASK_IDS = tuple(f"APMC-{index:03d}" for index in range(6)) + (
    "APMC-018",
)
EXPECTED_PRELAUNCH_READY_TASK_IDS = ("APMC-006", "APMC-012", "APMC-014")

PLAN_PATH = REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md"
OBJECTIVES_PATH = (
    REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md"
INVENTORY_DIR = (
    REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller_inventory"
)
BASELINE_PATH = INVENTORY_DIR / "baseline.json"
AUTHORITY_MAP_PATH = INVENTORY_DIR / "authority_map.md"
DUCKLAKE_CAPABILITY_PATH = INVENTORY_DIR / "ducklake_projection_capability.json"
BASELINE_SEAL_PATH = INVENTORY_DIR / "current_tree_baseline_seal.md"
BENCHMARK_DIR = REPO_ROOT / "benchmarks/agent_supervisor/autonomous_meta_controller"
BENCHMARK_MANIFEST_PATH = BENCHMARK_DIR / "baseline_manifest.json"
BENCHMARK_CASES_PATH = BENCHMARK_DIR / "cases.json"
BENCHMARK_VALIDATOR_PATH = BENCHMARK_DIR / "validate.py"
MATERIALIZER_PATH = (
    REPO_ROOT / "scripts/materialize_agent_supervisor_autonomous_meta_controller_board.py"
)
SCHEDULER_CONFIG_PATH = (
    REPO_ROOT / "config/agent_supervisor_autonomous_meta_controller_scheduler.json"
)
MCP_CONTRACT_CATALOG_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py"
)
MCP_INVOCATION_TRACE_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/mcp_invocation_trace.py"
)
OBJECTIVE_DAEMON_IMPORT_TEST_PATH = (
    REPO_ROOT / "test/api/test_agent_supervisor_objective_daemon_import.py"
)
APMC_MATERIALIZER_TEST_PATH = REPO_ROOT / "test/api/test_agent_supervisor_apmc_materializer.py"
DATABASE_PORTAL_BRIDGE_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py"
)
DATABASE_PORTAL_BRIDGE_TEST_PATH = (
    REPO_ROOT / "test/api/test_agent_supervisor_database_portal_bridge.py"
)
DATABASE_COORDINATION_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py"
)
DATABASE_COORDINATION_TEST_PATH = (
    REPO_ROOT / "test/api/test_agent_supervisor_database_coordination.py"
)
DATABASE_IMPLEMENTATION_DAEMON_TEST_PATH = (
    REPO_ROOT / "test/api/test_agent_supervisor_database_implementation_daemon.py"
)

REQUIRED_CONTROL_FILES = (
    PLAN_PATH,
    OBJECTIVES_PATH,
    TODO_PATH,
    BASELINE_PATH,
    AUTHORITY_MAP_PATH,
    DUCKLAKE_CAPABILITY_PATH,
    BENCHMARK_DIR / "README.md",
    BENCHMARK_MANIFEST_PATH,
    BENCHMARK_CASES_PATH,
    BENCHMARK_VALIDATOR_PATH,
    MATERIALIZER_PATH,
    SCHEDULER_CONFIG_PATH,
    MCP_CONTRACT_CATALOG_PATH,
    MCP_INVOCATION_TRACE_PATH,
    OBJECTIVE_DAEMON_IMPORT_TEST_PATH,
    APMC_MATERIALIZER_TEST_PATH,
    DATABASE_COORDINATION_PATH,
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
    DATABASE_PORTAL_BRIDGE_PATH,
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py",
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
    REPO_ROOT / "scripts/ops/agent_supervisor/quack_state_server.py",
    REPO_ROOT / "scripts/lgswf_start_quack_control.py",
    REPO_ROOT / "test/api/test_agent_supervisor_intent_repository.py",
    DATABASE_COORDINATION_TEST_PATH,
    DATABASE_IMPLEMENTATION_DAEMON_TEST_PATH,
    DATABASE_PORTAL_BRIDGE_TEST_PATH,
    REPO_ROOT / "test/api/test_agent_supervisor_database_runner_propagation.py",
    REPO_ROOT / "test/api/test_agent_supervisor_duckdb_connection_policy.py",
    REPO_ROOT / "test/api/test_agent_supervisor_implementation_auto_rescue.py",
    REPO_ROOT / "test/api/test_agent_supervisor_implementation_progress.py",
    REPO_ROOT / "test/api/test_agent_supervisor_implementation_protected_paths.py",
    REPO_ROOT / "test/api/test_agent_supervisor_merge_resolver.py",
    REPO_ROOT / "test/api/test_agent_supervisor_quack_state_server.py",
    REPO_ROOT / "test/api/test_agent_supervisor_task_revision_reconciliation.py",
    REPO_ROOT / "test/api/test_agent_supervisor_todo_llm.py",
    REPO_ROOT / "test/api/test_implementation_daemon_stale_quarantined_merge.py",
    Path(__file__).resolve(),
)
P0_FILES = tuple(
    REPO_ROOT / relative
    for relative in (
        "ipfs_accelerate_py/agent_supervisor/autonomy/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/decision_graph.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/cognitive_budget.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/cognitive_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/runtime.py",
        "test/api/autonomy/test_contracts.py",
        "test/api/autonomy/test_decision_graph.py",
        "test/api/autonomy/test_cognitive_budget.py",
        "test/api/autonomy/test_cognitive_scheduler.py",
        "test/api/autonomy/test_runtime.py",
    )
)
REQUIRED_CONTRACTS = (
    "AutonomyPolicy",
    "AutonomyEnvelope",
    "AutonomyLevel",
    "RiskAssessment",
    "DecisionQuestion",
    "DecisionGraph",
    "BeliefFact",
    "BeliefState",
    "ResolutionAction",
    "ResolutionCandidate",
    "MetaDecision",
    "CognitiveBudget",
    "BudgetReservation",
    "BudgetLedger",
    "ExperienceEpisode",
    "CausalAttribution",
    "PolicyObservation",
    "RoutePolicyCandidate",
    "DistillationCandidate",
    "DistilledDecisionRule",
    "SupervisorSkill",
    "HumanEscalationPacket",
    "AutonomousRepairPlan",
    "AutonomousRepairReceipt",
    "AutonomyRunReceipt",
    "AutonomyPromotionReceipt",
)
REQUIRED_TASK_FIELDS = (
    "stable task id",
    "status",
    "completion",
    "is schedulable",
    "review only",
    "priority",
    "track",
    "depends on",
    "goal id",
    "parent goal id",
    "subgoal id",
    "owning repository",
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
    "risk class",
    "network policy",
    "write scope",
    "prohibited effects",
    "rollback or compensation",
    "acceptance",
    "embedding query",
)
REQUIRED_GOAL_FIELDS = (
    "status",
    "review_only",
    "parent",
    "depends_on",
    "priority",
    "track",
    "bundle",
    "parallel_lane",
    "resource_class",
    "goal",
    "evidence",
    "evidence_criteria",
    "evidence_source_policy",
    "outputs",
    "interfaces",
    "validation",
    "acceptance",
    "gap_task",
    "refinement",
    "conflict_policy",
)
REQUIRED_PLAN_TERMS = (
    "AutonomousMetaController",
    "DecisionRuntime",
    "FormalDeltaReplanner",
    "ContextCompiler",
    "autonomous-repair",
    "DuckDB",
    "Quack",
    "DuckLake",
    "non-authoritative",
    "validation reserve",
    "model call is forbidden",
    "insufficient_counterfactual_evidence",
    "shadow",
    "content-addressed",
    "idle",
    "zero false completions",
    "30% lower median model input tokens",
)
EXPECTED_ORIGIN_MAIN = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
EXPECTED_ORIGIN_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
EXPECTED_MCP_COMPATIBILITY_SHA256 = {
    MCP_CONTRACT_CATALOG_PATH: "958a7c9c6fb8e8810922a8dd80fae9e375a1917ec4e90910be19703eb698b00b",
    MCP_INVOCATION_TRACE_PATH: "c263ada779b3996774ef93311b992b914ecdb02e31f04c669e5928a54b584a43",
}


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
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
        value.strip() and not path.is_absolute() and ".." not in path.parts and "\x00" not in value
    )


def _acyclic(adjacency: Mapping[str, Iterable[str]]) -> tuple[bool, tuple[str, ...]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: list[str] = []

    def visit(node: str, trail: tuple[str, ...]) -> bool:
        if node in visiting:
            cycle.extend((*trail, node))
            return False
        if node in visited:
            return True
        visiting.add(node)
        for dependency in adjacency.get(node, ()):
            if not visit(dependency, (*trail, node)):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    passed = all(visit(node, ()) for node in adjacency if node not in visited)
    return passed, tuple(cycle)


def _depends_transitively(
    task: str, ancestor: str, adjacency: Mapping[str, tuple[str, ...]]
) -> bool:
    frontier = list(adjacency.get(task, ()))
    seen: set[str] = set()
    while frontier:
        candidate = frontier.pop()
        if candidate == ancestor:
            return True
        if candidate not in seen:
            seen.add(candidate)
            frontier.extend(adjacency.get(candidate, ()))
    return False


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


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


def _structural_checks(checks: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in REQUIRED_CONTROL_FILES
        if not path.is_file()
    ]
    _append(checks, errors, name="required_deliverables", passed=not missing, detail=missing)
    if missing:
        return {}

    required_tracked = tuple(dict.fromkeys((*REQUIRED_CONTROL_FILES, *P0_FILES)))
    tracked_output = _run_git(
        "ls-files",
        "--",
        *(path.relative_to(REPO_ROOT).as_posix() for path in required_tracked),
    )
    tracked = {line for line in tracked_output.splitlines() if line}
    untracked = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in required_tracked
        if path.relative_to(REPO_ROOT).as_posix() not in tracked
    ]
    _append(
        checks,
        errors,
        name="required_controls_tracked",
        passed=not untracked,
        detail=untracked,
    )

    scheduler = _load_json(SCHEDULER_CONFIG_PATH)
    database_program = (
        scheduler.get("database_program")
        if isinstance(scheduler.get("database_program"), dict)
        else {}
    )
    authority_policy = (
        scheduler.get("authority_policy")
        if isinstance(scheduler.get("authority_policy"), dict)
        else {}
    )
    protected_paths = {str(item) for item in scheduler.get("protected_paths") or ()}
    expected_initial_projection = {
        "task_count": len(TASK_IDS),
        "completed_task_ids": list(EXPECTED_PRELAUNCH_COMPLETED_TASK_IDS),
        "ready_task_ids": list(EXPECTED_PRELAUNCH_READY_TASK_IDS),
        "blocked_task_ids": [],
        "terminal_task_id": "APMC-020",
        "goal_count": len(GOAL_IDS),
        "root_goal_id": ROOT_OBJECTIVE,
    }
    expected_lane_frontier = {
        0: ["APMC-012"],
        1: [],
        2: ["APMC-006", "APMC-014"],
        3: [],
    }
    observed_lane_frontier = {
        int(lane.get("index", -1)): list(lane.get("initial_task_ids") or ())
        for lane in scheduler.get("lanes") or ()
        if isinstance(lane, Mapping) and type(lane.get("index")) is int
    }
    portal_bridge_relative = DATABASE_PORTAL_BRIDGE_PATH.relative_to(REPO_ROOT).as_posix()
    database_coordination_relative = DATABASE_COORDINATION_PATH.relative_to(
        REPO_ROOT
    ).as_posix()
    scheduler_ok = bool(
        scheduler.get("board_namespace") == PROGRAM_ID
        and scheduler.get("task_prefix") == TASK_PREFIX
        and scheduler.get("merge_target_branch")
        == "codex/agent-supervisor-autonomous-meta-controller-v1"
        and scheduler.get("max_lanes") == 4
        and scheduler.get("strict_task_sharding") is True
        and database_program.get("authority_mode") == "quack"
        and database_program.get("task_source_kind") == "duckdb"
        and database_program.get("quack_endpoint") == "quack:127.0.0.1:45231"
        and database_program.get("store_id")
        == "state/agent_supervisor_autonomous_meta_controller/control.duckdb"
        and database_program.get("failover_policy") == "fail_closed"
        and authority_policy.get("automatic_file_fallback_from_quack") is False
        and authority_policy.get("ducklake_projection_authoritative") is False
        and authority_policy.get("ducklake_projection_required_for_scheduling") is False
        and SCHEDULER_CONFIG_PATH.relative_to(REPO_ROOT).as_posix() in protected_paths
        and portal_bridge_relative in protected_paths
        and database_coordination_relative in protected_paths
    )
    _append(
        checks,
        errors,
        name="scheduler_database_authority",
        passed=scheduler_ok,
        detail={
            "max_lanes": scheduler.get("max_lanes"),
            "database_program": database_program,
            "authority_policy": authority_policy,
        },
    )
    _append(
        checks,
        errors,
        name="scheduler_prelaunch_projection",
        passed=(
            scheduler.get("initial_projection") == expected_initial_projection
            and observed_lane_frontier == expected_lane_frontier
        ),
        detail={
            "expected_projection": expected_initial_projection,
            "observed_projection": scheduler.get("initial_projection"),
            "expected_lane_frontier": expected_lane_frontier,
            "observed_lane_frontier": observed_lane_frontier,
        },
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            load_configured_board,
        )

        configured_board = load_configured_board(
            SCHEDULER_CONFIG_PATH,
            repo_root=REPO_ROOT,
        )
        configured_program = configured_board.resolved_database_program()
        _append(
            checks,
            errors,
            name="scheduler_contract_load",
            passed=(
                configured_board.board_namespace == PROGRAM_ID
                and configured_board.max_lanes == 4
                and configured_program.authority_mode == "quack"
                and configured_program.task_source_kind == "duckdb"
            ),
            detail={
                "configuration_root": configured_board.configuration_root,
                "database_program": configured_program.redacted_dict(),
            },
        )
    except Exception as exc:
        _append(
            checks,
            errors,
            name="scheduler_contract_load",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    tasks = parse_task_text(todo_text, path=TODO_PATH, task_header_prefix="## APMC-")
    goals = parse_goal_heap(objective_text)
    task_headings = re.findall(r"^## (APMC-\d{3})\b", todo_text, flags=re.MULTILINE)
    goal_headings = re.findall(r"^## (APMC-G\d{3})\b", objective_text, flags=re.MULTILINE)

    parsed_task_ids = tuple(task.task_id for task in tasks)
    parsed_goal_ids = tuple(goal.goal_id for goal in goals)
    _append(
        checks,
        errors,
        name="task_population",
        passed=tuple(task_headings) == TASK_IDS and parsed_task_ids == TASK_IDS,
        detail={"expected": TASK_IDS, "headings": task_headings, "parsed": parsed_task_ids},
    )
    _append(
        checks,
        errors,
        name="task_ids_unique",
        passed=len(set(task_headings)) == len(task_headings),
        detail=[item for item, count in Counter(task_headings).items() if count != 1],
    )
    _append(
        checks,
        errors,
        name="goal_population",
        passed=tuple(goal_headings) == GOAL_IDS and parsed_goal_ids == GOAL_IDS,
        detail={"expected": GOAL_IDS, "headings": goal_headings, "parsed": parsed_goal_ids},
    )

    task_by_id = {task.task_id: task for task in tasks}
    missing_fields: dict[str, list[str]] = {}
    invalid_values: dict[str, list[str]] = {}
    adjacency: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        absent = [field for field in REQUIRED_TASK_FIELDS if field not in task.metadata]
        if absent:
            missing_fields[task.task_id] = absent
        invalid: list[str] = []
        if task.metadata.get("stable task id") != task.task_id:
            invalid.append("stable task id")
        if task.status not in {"todo", "completed"}:
            invalid.append("status")
        if task.metadata.get("board namespace") != PROGRAM_ID:
            invalid.append("board namespace")
        if task.metadata.get("parent goal id") != ROOT_OBJECTIVE:
            invalid.append("parent goal id")
        if task.metadata.get("goal id") not in GOAL_IDS[1:]:
            invalid.append("goal id")
        if task.metadata.get("subgoal id") != task.metadata.get("goal id"):
            invalid.append("subgoal id")
        if task.metadata.get("is schedulable", "").casefold() != "true":
            invalid.append("is schedulable")
        terminal_review = task.task_id in {"APMC-019", "APMC-020"}
        if task.completion != ("manual" if terminal_review else "automatic"):
            invalid.append("completion")
        if task.metadata.get("review only", "").casefold() != (
            "true" if terminal_review else "false"
        ):
            invalid.append("review only")
        if task.metadata.get("symbolic first", "").casefold() != "true":
            invalid.append("symbolic first")
        if task.metadata.get("risk class") not in {
            "R0_PURE",
            "R1_READ_ONLY",
            "R2_REVERSIBLE_LOCAL",
            "R3_BOUNDED_REPOSITORY_MUTATION",
            "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
            "R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL",
        }:
            invalid.append("risk class")
        if not task.outputs or not all(_contained_relative(item) for item in task.outputs):
            invalid.append("outputs")
        predicted = _csv(task.metadata.get("predicted files", ""))
        if tuple(task.outputs) != predicted:
            invalid.append("predicted files != outputs")
        if not task.validation or not task.acceptance.strip():
            invalid.append("validation/acceptance")
        unknown = [dependency for dependency in task.depends_on if dependency not in task_by_id]
        duplicates = [item for item, count in Counter(task.depends_on).items() if count > 1]
        if unknown or task.task_id in task.depends_on or duplicates:
            invalid.append("dependencies")
        if invalid:
            invalid_values[task.task_id] = invalid
        adjacency[task.task_id] = tuple(task.depends_on)
    _append(
        checks,
        errors,
        name="task_required_fields",
        passed=not missing_fields,
        detail=missing_fields,
    )
    _append(checks, errors, name="task_values", passed=not invalid_values, detail=invalid_values)
    dag_ok, cycle = _acyclic(adjacency)
    _append(checks, errors, name="task_dependency_dag", passed=dag_ok, detail={"cycle": cycle})

    initial_ready = tuple(task.task_id for task in tasks if not task.depends_on)
    after_baseline = tuple(
        task.task_id
        for task in tasks
        if task.task_id != "APMC-000" and set(task.depends_on).issubset({"APMC-000"})
    )
    completed_prelaunch = set(EXPECTED_PRELAUNCH_COMPLETED_TASK_IDS)
    after_prelaunch = tuple(
        task.task_id
        for task in tasks
        if task.task_id not in completed_prelaunch
        and set(task.depends_on).issubset(completed_prelaunch)
    )
    _append(
        checks,
        errors,
        name="parallel_bootstrap_frontier",
        passed=(
            initial_ready == ("APMC-000",)
            and after_baseline == ("APMC-001", "APMC-018")
            and after_prelaunch == EXPECTED_PRELAUNCH_READY_TASK_IDS
        ),
        detail={
            "initial": initial_ready,
            "after_baseline": after_baseline,
            "after_prelaunch_qualification": after_prelaunch,
        },
    )

    output_owners: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for output in task.outputs:
            output_owners[output].append(task.task_id)
    unsafe_overlaps: dict[str, list[str]] = {}
    for path, owners in output_owners.items():
        if len(owners) < 2:
            continue
        ordered = sorted(owners)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if not _depends_transitively(later, earlier, adjacency):
                unsafe_overlaps[path] = owners
    _append(
        checks,
        errors,
        name="conflict_serialization",
        passed=not unsafe_overlaps,
        detail=unsafe_overlaps,
    )

    goal_by_id = {goal.goal_id: goal for goal in goals}
    missing_goal_fields: dict[str, list[str]] = {}
    goal_errors: dict[str, list[str]] = {}
    goal_adjacency: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        absent = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if absent:
            missing_goal_fields[goal.goal_id] = absent
        dependencies = _csv(goal.fields.get("depends_on", ""))
        parent = goal.fields.get("parent", "").strip()
        references = (*dependencies, *((parent,) if parent else ()))
        unknown = [item for item in references if item not in goal_by_id]
        if unknown or (goal.goal_id != ROOT_OBJECTIVE and parent != ROOT_OBJECTIVE):
            goal_errors[goal.goal_id] = [
                *unknown,
                *(
                    "parent"
                    for _ in [0]
                    if goal.goal_id != ROOT_OBJECTIVE and parent != ROOT_OBJECTIVE
                ),
            ]
        goal_adjacency[goal.goal_id] = references
    goal_dag_ok, goal_cycle = _acyclic(goal_adjacency)
    _append(
        checks,
        errors,
        name="goal_required_fields",
        passed=not missing_goal_fields,
        detail=missing_goal_fields,
    )
    _append(
        checks,
        errors,
        name="goal_tree",
        passed=not goal_errors and goal_dag_ok,
        detail={"errors": goal_errors, "cycle": goal_cycle},
    )

    grouped_tasks = defaultdict(list)
    for task in tasks:
        grouped_tasks[task.metadata.get("goal id", "")].append(task.task_id)
    unassigned = [goal for goal in GOAL_IDS[1:] if not grouped_tasks[goal]]
    _append(checks, errors, name="subgoal_task_coverage", passed=not unassigned, detail=unassigned)

    missing_terms = [
        term for term in REQUIRED_PLAN_TERMS if term.casefold() not in plan_text.casefold()
    ]
    _append(
        checks, errors, name="architecture_scope", passed=not missing_terms, detail=missing_terms
    )
    return {"task_count": len(tasks), "goal_count": len(goals), "initial_ready": initial_ready}


def _inventory_checks(checks: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    baseline = _load_json(BASELINE_PATH)
    ducklake_observation = _load_json(DUCKLAKE_CAPABILITY_PATH)
    repository = baseline.get("repository") if isinstance(baseline.get("repository"), dict) else {}
    observed_main = _run_git("rev-parse", "origin/main")
    observed_tree = _run_git("rev-parse", "origin/main^{tree}")
    identity_ok = (
        repository.get("origin_main_commit") == EXPECTED_ORIGIN_MAIN == observed_main
        and repository.get("origin_main_tree") == EXPECTED_ORIGIN_TREE == observed_tree
    )
    _append(
        checks,
        errors,
        name="baseline_source_identity",
        passed=identity_ok,
        detail={
            "recorded_commit": repository.get("origin_main_commit"),
            "observed_commit": observed_main,
            "recorded_tree": repository.get("origin_main_tree"),
            "observed_tree": observed_tree,
        },
    )

    dependency_state = (
        baseline.get("dependency_state")
        if isinstance(baseline.get("dependency_state"), dict)
        else {}
    )
    declarations = (
        dependency_state.get("declaration_files")
        if isinstance(dependency_state.get("declaration_files"), dict)
        else {}
    )
    mismatched_declarations: dict[str, Any] = {}
    for relative, expected in declarations.items():
        path = REPO_ROOT / str(relative)
        actual = (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else "missing"
        )
        if actual != expected:
            mismatched_declarations[str(relative)] = {"expected": expected, "actual": actual}
    _append(
        checks,
        errors,
        name="dependency_declarations",
        passed=not mismatched_declarations,
        detail=mismatched_declarations,
    )

    recorded_gitlinks = (
        baseline.get("gitlinks") if isinstance(baseline.get("gitlinks"), dict) else {}
    )
    raw_tree = _run_git("ls-tree", "-r", "origin/main")
    actual_gitlinks: dict[str, str] = {}
    for line in raw_tree.splitlines():
        match = re.fullmatch(r"160000 commit ([0-9a-f]{40})\t(.+)", line)
        if match:
            actual_gitlinks[match.group(2)] = match.group(1)
    _append(
        checks,
        errors,
        name="gitlink_pins",
        passed=recorded_gitlinks == actual_gitlinks,
        detail={"recorded": recorded_gitlinks, "observed": actual_gitlinks},
    )

    capabilities = (
        baseline.get("capabilities") if isinstance(baseline.get("capabilities"), dict) else {}
    )
    quack = capabilities.get("quack") if isinstance(capabilities.get("quack"), dict) else {}
    ducklake = (
        capabilities.get("ducklake") if isinstance(capabilities.get("ducklake"), dict) else {}
    )
    ducklake_capability = (
        ducklake_observation.get("capability")
        if isinstance(ducklake_observation.get("capability"), dict)
        else {}
    )
    ducklake_gate = (
        ducklake_observation.get("promotion_gate")
        if isinstance(ducklake_observation.get("promotion_gate"), dict)
        else {}
    )
    ducklake_disposition = (
        ducklake_observation.get("apmc_disposition")
        if isinstance(ducklake_observation.get("apmc_disposition"), dict)
        else {}
    )
    held_by = {str(item) for item in ducklake.get("held_by") or ()}
    observed_held_by = {str(item) for item in ducklake_gate.get("held_by") or ()}
    capability_policy_ok = (
        quack.get("network_install_attempted") is False
        and ducklake.get("authority") is False
        and ducklake.get("production_catalog_activation") == "held"
        and ducklake.get("production_mutation_permitted") is False
        and held_by == {"DQK-088", "DQK-094", "DQK-102"}
        and ducklake_observation.get("program_id") == PROGRAM_ID
        and ducklake_capability.get("production_endpoint_permitted") is False
        and ducklake_capability.get("production_mutation_permitted") is False
        and ducklake_capability.get("catalog_created_for_apmc") is False
        and ducklake_capability.get("network_install_attempted") is False
        and ducklake_gate.get("status") == "held"
        and ducklake_gate.get("self_authorization_permitted") is False
        and observed_held_by == held_by
        and ducklake_disposition.get("operational_authority") is False
        and ducklake_disposition.get("projection_enabled_at_launch") is False
        and ducklake_disposition.get("duckdb_quack_scheduling_may_continue") is True
        and baseline.get("authority_policy", {}).get("raw_model_sql_permitted") is False
        and baseline.get("authority_policy", {}).get("sibling_mutation_permitted") is False
    )
    _append(
        checks,
        errors,
        name="capability_authority_policy",
        passed=capability_policy_ok,
        detail={
            "quack": quack,
            "ducklake": ducklake,
            "ducklake_observation": ducklake_observation,
        },
    )

    compatibility_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): {
            "expected": expected,
            "observed": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path, expected in EXPECTED_MCP_COMPATIBILITY_SHA256.items()
    }
    exact_compatibility = all(
        item["expected"] == item["observed"] for item in compatibility_hashes.values()
    )
    _append(
        checks,
        errors,
        name="reviewed_mcp_compatibility_sources",
        passed=exact_compatibility,
        detail=compatibility_hashes,
    )

    try:
        from ipfs_accelerate_py.agent_supervisor.objectives import objective_daemon
        from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
            AUTHORITATIVE_DISPOSITION_SCHEMA,
            AUTHORITY_LATTICE_SCHEMA,
            CHECKER_TRACE_SCHEMA,
            COUNTEREXAMPLE_TRACE_SCHEMA,
            HAMMER_TRACE_SCHEMA,
        )

        compatibility_ok = bool(
            objective_daemon
            and AUTHORITY_LATTICE_SCHEMA
            and HAMMER_TRACE_SCHEMA
            and COUNTEREXAMPLE_TRACE_SCHEMA
            and CHECKER_TRACE_SCHEMA
            and AUTHORITATIVE_DISPOSITION_SCHEMA
        )
        _append(
            checks,
            errors,
            name="objective_daemon_compatibility",
            passed=compatibility_ok,
            detail={"imported": compatibility_ok, "mocks_used": False},
        )
    except Exception as exc:
        _append(
            checks,
            errors,
            name="objective_daemon_compatibility",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
            probe_quack_capabilities,
        )

        report = probe_quack_capabilities(allow_network_install=False, use_cache=False)
        live = report.to_dict()
        live_ok = (
            str(live.get("status", "")).casefold() == "compatible"
            and live.get("network_install_attempted") is False
        )
        _append(
            checks,
            errors,
            name="quack_read_only_capability",
            passed=live_ok,
            detail={
                "status": live.get("status"),
                "network_install_attempted": live.get("network_install_attempted"),
            },
        )
    except Exception as exc:
        _append(
            checks,
            errors,
            name="quack_read_only_capability",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _p0_and_benchmark_checks(checks: list[dict[str, Any]], errors: list[str]) -> None:
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in P0_FILES if not path.is_file()]
    _append(checks, errors, name="p0_files_present", passed=not missing, detail=missing)
    try:
        from ipfs_accelerate_py.agent_supervisor.autonomy import contracts

        absent = [name for name in REQUIRED_CONTRACTS if not hasattr(contracts, name)]
        _append(checks, errors, name="closed_contract_population", passed=not absent, detail=absent)
    except Exception as exc:
        _append(
            checks,
            errors,
            name="closed_contract_population",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    try:
        materializer_tree = ast.parse(
            MATERIALIZER_PATH.read_text(encoding="utf-8"),
            filename=str(MATERIALIZER_PATH),
        )
        build_function = next(
            node
            for node in materializer_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "build_population"
        )
        policy_node = next(
            value.elts[0]
            for node in ast.walk(build_function)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant)
            and key.value == "acceptance"
            and isinstance(value, ast.List)
            and len(value.elts) == 1
            and isinstance(value.elts[0], ast.Dict)
        )
        literal_policy = {
            str(key.value): value.value
            for key, value in zip(
                policy_node.keys,
                policy_node.values,
                strict=True,
            )
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Constant)
        }
        evidence_unpacks = [
            value
            for key, value in zip(
                policy_node.keys,
                policy_node.values,
                strict=True,
            )
            if key is None
            and {
                "BASELINE_TASK_ID",
                "BASELINE_VALIDATION_SET_EVIDENCE_KIND",
                "PRELAUNCH_COMPLETED_TASK_IDS",
                "PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND",
            }.issubset(
                {candidate.id for candidate in ast.walk(value) if isinstance(candidate, ast.Name)}
            )
        ]
        assignments = {
            target.id: node.value
            for node in materializer_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        baseline_evidence_kind = ast.literal_eval(
            assignments["BASELINE_VALIDATION_SET_EVIDENCE_KIND"]
        )
        prelaunch_evidence_kind = ast.literal_eval(
            assignments["PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND"]
        )
        qualified_task_ids = tuple(ast.literal_eval(assignments["PRELAUNCH_QUALIFIED_TASK_IDS"]))
        ready_task_ids = tuple(ast.literal_eval(assignments["PRELAUNCH_READY_TASK_IDS"]))
        expected_completed_expression = ast.parse(
            "tuple(f'APMC-{index:03d}' for index in range(6)) + ('APMC-018',)",
            mode="eval",
        ).body
        completed_expression_exact = ast.dump(
            assignments["PRELAUNCH_COMPLETED_TASK_IDS"],
            include_attributes=False,
        ) == ast.dump(expected_completed_expression, include_attributes=False)
        nested_policy = any(
            isinstance(key, ast.Constant) and key.value == "evidence_policy"
            for key in policy_node.keys
        )
        policy_ok = bool(
            literal_policy.get("current_tree_required") is True
            and literal_policy.get("declared_validation_required") is True
            and literal_policy.get("markdown_status_is_authority") is False
            and not nested_policy
            and len(evidence_unpacks) == 1
            and baseline_evidence_kind == "apmc_baseline_validation_set"
            and prelaunch_evidence_kind == "apmc_prelaunch_validation_set"
            and qualified_task_ids == EXPECTED_PRELAUNCH_QUALIFIED_TASK_IDS
            and completed_expression_exact
            and ready_task_ids == EXPECTED_PRELAUNCH_READY_TASK_IDS
        )
        _append(
            checks,
            errors,
            name="baseline_top_level_evidence_policy",
            passed=policy_ok,
            detail={
                "literal_policy": literal_policy,
                "baseline_dedicated_kind": baseline_evidence_kind,
                "prelaunch_dedicated_kind": prelaunch_evidence_kind,
                "top_level_evidence_unpack_count": len(evidence_unpacks),
                "qualified_task_ids": qualified_task_ids,
                "completed_task_expression_exact": completed_expression_exact,
                "ready_task_ids": ready_task_ids,
            },
        )

        tasks = parse_task_text(
            TODO_PATH.read_text(encoding="utf-8"),
            path=TODO_PATH,
            task_header_prefix="## APMC-",
        )
        baseline_task = next(task for task in tasks if task.task_id == "APMC-000")
        validation_argv = tuple(tuple(shlex.split(command)) for command in baseline_task.validation)
        required_completion_validations = (
            (
                "python3",
                "-m",
                "pytest",
                "-q",
                "test/api/test_agent_supervisor_apmc_materializer.py",
            ),
            (
                "python3",
                "-m",
                "pytest",
                "-q",
                "test/api/test_agent_supervisor_intent_repository.py",
            ),
            (
                "python3",
                "-m",
                "pytest",
                "-q",
                "test/api/test_agent_supervisor_database_portal_bridge.py",
            ),
            (
                "python3",
                "-m",
                "pytest",
                "-q",
                "test/api/test_agent_supervisor_database_coordination.py::test_authoritative_task_sync_is_idempotent_fail_closed_and_preserves_prepared",
                "test/api/test_agent_supervisor_database_implementation_daemon.py::test_apmc_bootstrap_completions_unlock_exact_frontier_across_lane_sidecars",
                "test/api/test_agent_supervisor_database_implementation_daemon.py::test_removed_authoritative_task_is_excluded_without_idle_growth",
                "test/api/test_agent_supervisor_database_implementation_daemon.py::test_authoritative_dependency_reopen_invalidates_stale_lane_readiness",
                "test/api/test_agent_supervisor_database_implementation_daemon.py::test_fenced_retry_cannot_bypass_dependency_reopen_after_local_claim",
                "test/api/test_agent_supervisor_database_implementation_daemon.py::test_restart_retires_prepared_absent_expired_attempt_then_refences_retry",
                "test/api/test_agent_supervisor_database_implementation_daemon.py::test_expired_preparation_without_control_cas_is_aborted_and_requeued",
            ),
            (
                "python3",
                "-m",
                "pytest",
                "-q",
                "test/api/test_agent_supervisor_quack_state_server.py::test_mutation_request_is_published_only_after_complete_fsync",
                "test/api/test_agent_supervisor_quack_state_server.py::test_mutation_request_publication_fails_closed_without_atomic_rename",
            ),
        )
        exact_completion_validations = all(
            validation_argv.count(expected) == 1 for expected in required_completion_validations
        )
        _append(
            checks,
            errors,
            name="baseline_completion_validation_commands",
            passed=exact_completion_validations,
            detail={
                "required": required_completion_validations,
                "observed": validation_argv,
            },
        )
    except Exception as exc:
        _append(
            checks,
            errors,
            name="baseline_top_level_evidence_policy",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    manifest = _load_json(BENCHMARK_MANIFEST_PATH)
    corpus = _load_json(BENCHMARK_CASES_PATH)
    manifest_corpus = manifest.get("corpus") if isinstance(manifest.get("corpus"), dict) else {}
    digest = hashlib.sha256(BENCHMARK_CASES_PATH.read_bytes()).hexdigest()
    cases = corpus.get("cases") if isinstance(corpus.get("cases"), list) else []
    benchmark_ok = (
        manifest.get("program_id") == PROGRAM_ID
        and corpus.get("program_id") == PROGRAM_ID
        and manifest_corpus.get("sha256") == digest
        and manifest_corpus.get("case_count") == len(cases) == 16
        and manifest.get("measurements", {}).get("status") == "not_run"
        and manifest.get("promotion_eligible") is False
    )
    _append(
        checks,
        errors,
        name="frozen_benchmark_inputs",
        passed=benchmark_ok,
        detail={
            "sha256": digest,
            "case_count": len(cases),
            "measurement_status": manifest.get("measurements", {}).get("status"),
        },
    )


def _baseline_seal_checks(checks: list[dict[str, Any]], errors: list[str]) -> None:
    if not BASELINE_SEAL_PATH.is_file():
        _append(
            checks,
            errors,
            name="operator_readable_baseline_seal",
            passed=False,
            detail="current_tree_baseline_seal.md is absent",
        )
        return
    seal_bytes = BASELINE_SEAL_PATH.read_bytes()
    seal_text = seal_bytes.decode("utf-8", errors="strict")
    required_terms = (
        PROGRAM_ID,
        EXPECTED_ORIGIN_MAIN,
        "Status: sealed",
        "DuckDB",
        "Quack",
        "DuckLake",
        "non-authoritative",
        "Production promotion eligible: no",
    )
    missing_terms = [item for item in required_terms if item not in seal_text]
    _append(
        checks,
        errors,
        name="operator_readable_baseline_seal",
        passed=(200 <= len(seal_bytes) <= 16_384 and not missing_terms),
        detail={"bytes": len(seal_bytes), "missing_terms": missing_terms},
    )


def validate(
    *,
    check_inventory: bool,
    check_all: bool,
    check_baseline_seal: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    summary = _structural_checks(checks, errors)
    if not errors and (check_inventory or check_all):
        _inventory_checks(checks, errors, warnings)
    if not errors and check_all:
        _p0_and_benchmark_checks(checks, errors)
    if not errors and check_baseline_seal:
        _baseline_seal_checks(checks, errors)
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-validation@1",
        "program_id": PROGRAM_ID,
        "root_objective_id": ROOT_OBJECTIVE,
        "valid": not errors,
        **summary,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-inventory",
        action="store_true",
        help="Also verify the sealed current-main inventory and local Quack capability.",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Verify inventory, P0 source/test population, and frozen benchmark inputs.",
    )
    parser.add_argument(
        "--check-baseline-seal",
        action="store_true",
        help="Also require the operator-readable APMC-000 current-tree seal.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = validate(
            check_inventory=args.check_inventory or args.check_baseline_seal,
            check_all=args.check_all,
            check_baseline_seal=args.check_baseline_seal,
        )
    except Exception as exc:
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-validation@1",
            "program_id": PROGRAM_ID,
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "checks": [],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
