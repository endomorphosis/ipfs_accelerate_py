#!/usr/bin/env python3
"""Validate the sealed Tactician/Hammer logic-repair program board.

This validator is intentionally deterministic and side-effect free.  It checks
the finite goal/task DAG and the scheduler's safety policy; live checkout and
provider checks belong to the launcher's ``doctor`` command.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_file,
)


PLAN_PATH = Path(
    "docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md"
)
OBJECTIVE_PATH = Path(
    "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md"
)
TODO_PATH = Path(
    "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md"
)
SCHEDULER_PATH = Path(
    "config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
)
VALIDATOR_PATH = Path("scripts/validate_tactician_hammer_logic_repair_board.py")
LAUNCHER_PATH = Path("scripts/tactician_hammer_logic_repair_supervisor.sh")
BOOTSTRAP_TEST_PATH = Path(
    "test/api/test_agent_supervisor_tactician_hammer_logic_repair_bootstrap.py"
)
RPR_TODO_PATH = Path(
    "docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md"
)

TASK_PREFIX = "LPR-"
BOARD_NAMESPACE = "agent-supervisor-tactician-hammer-logic-repair-v1"
TARGET_BRANCH = "agent/proof-gated-contract-repair"
EXPECTED_TASK_IDS = tuple(f"LPR-{number:03d}" for number in range(21))
EXPECTED_GOAL_IDS = (
    "LPR-G000",
    "LPR-G010",
    "LPR-G020",
    "LPR-G030",
    "LPR-G040",
    "LPR-G050",
    "LPR-G060",
)
POST_BOOTSTRAP_READY = ("LPR-001", "LPR-002", "LPR-003", "LPR-004")
CONTROL_ARTIFACTS = (
    PLAN_PATH,
    OBJECTIVE_PATH,
    TODO_PATH,
    SCHEDULER_PATH,
    VALIDATOR_PATH,
    LAUNCHER_PATH,
)
BOOTSTRAP_OUTPUTS = (*CONTROL_ARTIFACTS, BOOTSTRAP_TEST_PATH)
REQUIRED_TASK_METADATA = (
    "goal id",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "token class",
    "estimated tokens",
    "implementation timeout seconds",
    "predicted files",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "acceptance",
    "embedding query",
)
ZERO_SAFETY_FLOORS = (
    "missed_resolved_impacted_consumer_rate",
    "unreconstructed_logic_or_unvalidated_countermodel_admission_rate",
    "unauthorized_premise_or_axiom_admission_rate",
    "behavior_invented_without_independent_authority_rate",
    "wrong_value_source_or_placement_admission_rate",
    "stale_root_corpus_or_receipt_admission_rate",
    "failed_obligation_override_rate",
    "llm_scope_or_semantic_escape_rate",
    "partial_transaction_completion_rate",
    "false_fixed_point_completion_rate",
)
NON_AUTHORITY_FLAGS = (
    "tactician_semantic_authority",
    "vector_semantic_authority",
    "knowledge_graph_semantic_authority",
    "learned_ranking_semantic_authority",
    "hammer_candidate_semantic_authority",
    "raw_countermodel_semantic_authority",
    "ordinary_test_semantic_authority",
    "runtime_witness_semantic_authority",
    "llm_router_semantic_authority",
    "llm_router_write_authority",
)
REQUIRED_AUTHORITY_GATES = (
    "native_kernel_reconstruction_required_for_proof",
    "independent_countermodel_validation_required_for_refutation",
    "existing_rpr_plan_lease_and_transaction_authority_required",
)
ROLLOUT_OFF_FLAGS = (
    "logic_prediction_enabled",
    "learned_tactician_ranking_enabled",
    "hammer_execution_enabled",
    "counterexample_refinement_enabled",
    "llm_router_enabled",
    "narrow_autonomous_mutation_enabled",
)


class BoardValidationError(RuntimeError):
    """Raised when a sealed control-plane invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoardValidationError(message)


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{name} must be a JSON list")
    result = tuple(str(item).strip() for item in value)
    _require(all(result), f"{name} contains an empty value")
    _require(len(result) == len(set(result)), f"{name} contains duplicates")
    return result


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _safe_relative(path_text: str, *, field: str) -> None:
    path = PurePosixPath(path_text)
    _require(path_text == path_text.strip(), f"{field} has surrounding whitespace")
    _require(not path.is_absolute(), f"{field} must be repository-relative: {path_text}")
    _require(".." not in path.parts, f"{field} escapes the repository: {path_text}")
    _require("\x00" not in path_text, f"{field} contains NUL")


def _assert_acyclic(graph: Mapping[str, Iterable[str]], *, label: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: tuple[str, ...]) -> None:
        if node in visiting:
            raise BoardValidationError(
                f"{label} cycle: {' -> '.join((*trail, node))}"
            )
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, ())


def _load_scheduler() -> dict[str, object]:
    try:
        payload = json.loads((REPO_ROOT / SCHEDULER_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoardValidationError(f"cannot read scheduler: {exc}") from exc
    _require(isinstance(payload, dict), "scheduler root must be a JSON object")
    return payload


def _validate_control_artifacts() -> None:
    for path in BOOTSTRAP_OUTPUTS:
        _require((REPO_ROOT / path).is_file(), f"missing control artifact: {path}")


def _validate_goals() -> tuple[object, ...]:
    goals = tuple(
        parse_goal_heap((REPO_ROOT / OBJECTIVE_PATH).read_text(encoding="utf-8"))
    )
    ids = tuple(goal.goal_id for goal in goals)
    _require(len(ids) == len(set(ids)), "duplicate goal id")
    _require(set(ids) == set(EXPECTED_GOAL_IDS), f"unexpected goal ids: {sorted(ids)}")
    by_id = {goal.goal_id: goal for goal in goals}
    graph: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        _require(re.fullmatch(r"LPR-G\d{3}", goal.goal_id) is not None, f"bad goal id: {goal.goal_id}")
        dependencies = tuple(goal.dependencies)
        unknown = sorted(set(dependencies) - set(by_id))
        _require(not unknown, f"unknown goal dependencies for {goal.goal_id}: {unknown}")
        graph[goal.goal_id] = dependencies
        parent = goal.fields.get("parent", "").strip()
        if goal.goal_id == "LPR-G000":
            _require(not parent, "root goal must not have a parent")
            children = _csv(goal.fields.get("subgoals", ""))
            _require(set(children) == set(EXPECTED_GOAL_IDS[1:]), "root subgoal set mismatch")
        else:
            _require(parent == "LPR-G000", f"{goal.goal_id} must be parented by LPR-G000")
    _assert_acyclic(graph, label="goal dependency")
    return goals


def _validate_tasks(goal_ids: set[str]) -> tuple[object, ...]:
    tasks = tuple(
        parse_task_file(REPO_ROOT / TODO_PATH, task_header_prefix=TASK_PREFIX)
    )
    ids = tuple(task.task_id for task in tasks)
    _require(len(ids) == len(set(ids)), "duplicate task id")
    _require(tuple(sorted(ids)) == EXPECTED_TASK_IDS, f"unexpected task ids: {sorted(ids)}")
    by_id = {task.task_id: task for task in tasks}
    graph: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        _require(re.fullmatch(r"LPR-\d{3}", task.task_id) is not None, f"bad task id: {task.task_id}")
        unknown = sorted(set(task.depends_on) - set(by_id))
        _require(not unknown, f"unknown dependencies for {task.task_id}: {unknown}")
        graph[task.task_id] = tuple(task.depends_on)
        missing = [name for name in REQUIRED_TASK_METADATA if not task.metadata.get(name, "").strip()]
        _require(not missing, f"{task.task_id} missing metadata: {missing}")
        _require(task.metadata["goal id"] in goal_ids, f"{task.task_id} has unknown goal")
        _require(task.board_namespace == BOARD_NAMESPACE, f"{task.task_id} namespace mismatch")
        _require(task.completion in {"auto", "manual"}, f"{task.task_id} completion mismatch")
        _require(
            bool(task.validation)
            and all(str(command).strip() for command in task.validation),
            f"{task.task_id} has no validation command",
        )
        _require(task.acceptance.strip(), f"{task.task_id} has no acceptance criteria")
        for output in task.outputs:
            _safe_relative(output, field=f"{task.task_id} output")
        for predicted in _csv(task.metadata["predicted files"]):
            _safe_relative(predicted, field=f"{task.task_id} predicted file")
        try:
            estimated = int(task.metadata["estimated tokens"])
            timeout = int(task.metadata["implementation timeout seconds"])
        except ValueError as exc:
            raise BoardValidationError(f"{task.task_id} has a non-integer bound") from exc
        _require(0 < estimated <= 100_000, f"{task.task_id} token bound is unsafe")
        _require(0 < timeout <= 14_400, f"{task.task_id} timeout bound is unsafe")
    _assert_acyclic(graph, label="task dependency")
    roots = sorted(task_id for task_id, dependencies in graph.items() if not dependencies)
    _require(roots == ["LPR-000"], f"task roots mismatch: {roots}")
    _require(graph["LPR-020"] == ("LPR-019",), "LPR-020 must be terminal after LPR-019")

    simulated_completed = {"LPR-000"}
    ready = tuple(
        sorted(
            task.task_id
            for task in tasks
            if task.task_id != "LPR-000"
            and set(task.depends_on).issubset(simulated_completed)
        )
    )
    _require(ready == POST_BOOTSTRAP_READY, f"post-bootstrap ready set mismatch: {ready}")
    bootstrap = by_id["LPR-000"]
    _require(
        tuple(bootstrap.outputs) == tuple(str(path) for path in BOOTSTRAP_OUTPUTS),
        "LPR-000 bootstrap output list mismatch",
    )

    foundations = [by_id[task_id] for task_id in POST_BOOTSTRAP_READY]
    owned: list[tuple[PurePosixPath, str]] = []
    for task in foundations:
        for predicted in _csv(task.metadata["predicted files"]):
            path = PurePosixPath(predicted)
            for other, owner in owned:
                overlaps = path == other or path in other.parents or other in path.parents
                _require(
                    not overlaps or owner == task.task_id,
                    f"foundation path conflict: {predicted} ({owner}, {task.task_id})",
                )
            owned.append((path, task.task_id))
    _require(
        bootstrap.status == "completed",
        "LPR-000 must be completed before the sealed board is launched",
    )
    return tasks


def _validate_scheduler(scheduler: Mapping[str, object], tasks: Sequence[object]) -> None:
    expected_scalars = {
        "schema": "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.scheduler_config@1",
        "taskboard_path": str(TODO_PATH),
        "objectives_path": str(OBJECTIVE_PATH),
        "plan_path": str(PLAN_PATH),
        "validator_path": str(VALIDATOR_PATH),
        "launcher_path": str(LAUNCHER_PATH),
        "task_prefix": TASK_PREFIX,
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": TARGET_BRANCH,
        "max_lanes": 4,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for key, expected in expected_scalars.items():
        _require(scheduler.get(key) == expected, f"scheduler {key} mismatch")
    for key in (
        "poll_interval_seconds",
        "daemon_interval_seconds",
        "check_interval_seconds",
        "stale_seconds",
        "watchdog_startup_grace_seconds",
        "max_restarts",
        "max_task_attempts",
        "implementation_retry_budget",
        "validation_retry_budget",
        "merge_retry_budget",
        "implementation_timeout_seconds",
        "implementation_max_timeout_seconds",
        "implementation_log_stall_seconds",
    ):
        value = scheduler.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"scheduler {key} must be positive")
    _require(scheduler["implementation_max_timeout_seconds"] >= scheduler["implementation_timeout_seconds"], "max timeout is below default timeout")
    _require(_strings(scheduler.get("worktree_submodule_paths"), name="worktree_submodule_paths") == ("ipfs_datasets_py",), "datasets gitlink binding missing")
    protected = _strings(scheduler.get("protected_paths"), name="protected_paths")
    _require(protected == tuple(str(path) for path in CONTROL_ARTIFACTS), "protected control artifacts mismatch")
    for path in protected:
        _safe_relative(path, field="protected path")

    source = scheduler.get("source_binding")
    _require(isinstance(source, dict), "source_binding must be an object")
    for key in (
        "require_exact_accelerator_branch",
        "require_initialized_datasets_gitlink",
        "require_superproject_gitlink_equals_nested_head",
        "record_accelerator_and_datasets_revisions_at_launch",
    ):
        _require(source.get(key) is True, f"source binding disabled: {key}")
    _require(source.get("accelerator_branch") == TARGET_BRANCH, "source branch binding mismatch")
    _require(source.get("datasets_submodule_path") == "ipfs_datasets_py", "datasets source path mismatch")

    lane_rows = scheduler.get("lanes")
    _require(isinstance(lane_rows, list) and len(lane_rows) == 4, "scheduler must define four lanes")
    expected_initial = {0: ["LPR-004"], 1: ["LPR-001"], 2: ["LPR-002"], 3: ["LPR-003"]}
    observed_initial: dict[int, object] = {}
    for row in lane_rows:
        _require(isinstance(row, dict), "lane row must be an object")
        index = row.get("index")
        _require(isinstance(index, int) and not isinstance(index, bool), "lane index must be an integer")
        _require(index in range(4) and index not in observed_initial, f"invalid or duplicate lane index: {index}")
        _require(row.get("name") == f"lpr-lane-{index}", f"lane {index} name mismatch")
        _require(row.get("strict_shard_remainder") == index, f"lane {index} shard mismatch")
        observed_initial[index] = row.get("initial_task_ids")
    _require(observed_initial == expected_initial, f"initial lane assignment mismatch: {observed_initial}")
    for index, task_ids in observed_initial.items():
        for task_id in task_ids:
            _require(int(task_id.rsplit("-", 1)[1]) % 4 == index, f"{task_id} does not map to lane {index}")

    provider = scheduler.get("provider")
    _require(isinstance(provider, dict), "provider must be an object")
    _require(provider.get("max_concurrency") == 4, "provider concurrency must equal lane count")
    _require(provider.get("secrets_in_argv_or_logs") is False, "secrets must not enter argv/logs")

    rollout = scheduler.get("rollout")
    _require(isinstance(rollout, dict), "rollout must be an object")
    _require(rollout.get("mode") == "shadow", "initial rollout must be shadow")
    for key in ROLLOUT_OFF_FLAGS:
        _require(rollout.get(key) is False, f"initial feature flag must be off: {key}")

    authority = scheduler.get("authority_policy")
    _require(isinstance(authority, dict), "authority_policy must be an object")
    for key in NON_AUTHORITY_FLAGS:
        _require(authority.get(key) is False, f"advisory source promoted to authority: {key}")
    for key in REQUIRED_AUTHORITY_GATES:
        _require(authority.get(key) is True, f"authority gate disabled: {key}")
    _require(authority.get("unknown_or_unsupported_disposition") == "abstain", "unknown/unsupported work must abstain")

    repair = scheduler.get("repair_policy")
    _require(isinstance(repair, dict), "repair_policy must be an object")
    for key in (
        "impact_closure_required_before_mutation",
        "one_disposition_per_resolved_consumer",
        "logic_goal_and_premise_independence_required",
        "tactician_plan_gate_required",
        "native_goal_round_trip_required",
        "analytical_transform_precedes_llm_router",
        "llm_router_requires_admitted_semantics_and_exact_paths",
        "proposal_overlay_analysis_required_for_ordinary_model_edits",
        "atomic_scc_transaction_required",
        "logic_and_program_fixed_point_required",
    ):
        _require(repair.get(key) is True, f"repair gate disabled: {key}")
    _require(repair.get("partial_plan_completion_allowed") is False, "partial completion must be forbidden")
    _require(repair.get("open_required_frontier_disposition") == "abstain", "open required frontier must abstain")
    _require(repair.get("memory_resource_or_type_evidence_implies_memory_safety") is False, "memory safety must not be inferred")

    floors = scheduler.get("release_safety_floors")
    _require(isinstance(floors, dict), "release_safety_floors must be an object")
    _require(set(floors) == set(ZERO_SAFETY_FLOORS), "release safety floor set mismatch")
    for key in ZERO_SAFETY_FLOORS:
        value = floors.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value == 0, f"release safety floor must be integer zero: {key}")

    hints = scheduler.get("resource_hints")
    _require(isinstance(hints, dict), "resource_hints must be an object")
    lanes = {task.metadata["parallel lane"] for task in tasks}
    _require(lanes.issubset(hints), f"missing resource hints: {sorted(lanes - set(hints))}")
    for task in tasks:
        lane = task.metadata["parallel lane"]
        _require(
            hints.get(lane) == task.metadata["resource class"],
            f"resource hint mismatch for {task.task_id}: {lane}",
        )


def _validate_authority_language(scheduler: Mapping[str, object]) -> None:
    text = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH)
    ).lower()
    required = (
        "semantic_authority=false",
        "shadow mode remains the default",
        "independently validated countermodel",
        "analytical",
        "fixed point",
    )
    for phrase in required:
        _require(phrase in text, f"normative authority phrase is missing: {phrase}")
    encoded = json.dumps(scheduler, sort_keys=True).lower()
    for secret_word in ("api_key", "access_token", "bearer_token", "password"):
        _require(secret_word not in encoded, f"scheduler must not contain secret field: {secret_word}")


def _validate_predecessor() -> None:
    path = REPO_ROOT / RPR_TODO_PATH
    _require(path.is_file(), "completed RPR predecessor board is missing")
    tasks = parse_task_file(path, task_header_prefix="RPR-")
    _require(len(tasks) == 48, f"RPR predecessor task count mismatch: {len(tasks)}")
    incomplete = [task.task_id for task in tasks if task.status != "completed"]
    _require(not incomplete, f"RPR predecessor is incomplete: {incomplete}")


def validate_all() -> dict[str, object]:
    _validate_control_artifacts()
    goals = _validate_goals()
    tasks = _validate_tasks({goal.goal_id for goal in goals})
    scheduler = _load_scheduler()
    _validate_scheduler(scheduler, tasks)
    _validate_authority_language(scheduler)
    _validate_predecessor()
    completed = sorted(task.task_id for task in tasks if task.status == "completed")
    ready = sorted(
        task.task_id
        for task in tasks
        if task.status == "todo" and set(task.depends_on).issubset(completed)
    )
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.board_validation@1",
        "valid": True,
        "goal_count": len(goals),
        "task_count": len(tasks),
        "completed_count": len(completed),
        "ready_task_ids": ready,
        "post_bootstrap_ready_task_ids": list(POST_BOOTSTRAP_READY),
        "lane_count": 4,
        "rollout_mode": "shadow",
        "protected_artifact_count": len(CONTROL_ARTIFACTS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="validate the complete goal/task/scheduler control plane",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.check_all:
        parser.error("--check-all is required")
    try:
        payload = validate_all()
    except BoardValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
