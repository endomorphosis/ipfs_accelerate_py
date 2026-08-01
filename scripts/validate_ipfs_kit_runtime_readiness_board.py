#!/usr/bin/env python3
"""Fail-closed validator for the IPFS Kit runtime-readiness program."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/IPFS_KIT_RUNTIME_READINESS_PLAN.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/ipfs_kit_runtime_readiness.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_runtime_readiness.todo.md"
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json"
)

BOARD_NAMESPACE = "ipfs-kit-runtime-readiness-v1"
TASK_IDS = tuple(f"KITA-{index:03d}" for index in range(48))
GOAL_IDS = (
    "KITA-G000",
    "KITA-G010",
    "KITA-G020",
    "KITA-G030",
    "KITA-G040",
    "KITA-G050",
    "KITA-G060",
    "KITA-G070",
    "KITA-G080",
    "KITA-G090",
    "KITA-G100",
    "KITA-G110",
)
INITIAL_COMPLETED = ("KITA-000",)
INITIAL_READY = ("KITA-001", "KITA-002", "KITA-003", "KITA-004")
TERMINAL_TASK = "KITA-047"
SEALED_TASKBOARD_DEFINITION_SHA256 = (
    "sha256:51a55a9a900688a382788940eb3f62d6b5b101a4a6fbf0218c17bdd15972e524"
)

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
PERSISTED_PROGRESS_STATES = frozenset({"todo", "completed"})
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
REQUIRED_JOIN_DEPENDENCIES = {
    "KITA-009": {
        "KITA-007",
        "KITA-008",
        "KITA-021",
        "KITA-037",
        "KITA-040",
        "KITA-041",
    },
    "KITA-013": {
        "KITA-011",
        "KITA-012",
        "KITA-021",
        "KITA-029",
        "KITA-033",
        "KITA-037",
        "KITA-040",
        "KITA-041",
    },
    "KITA-017": {"KITA-015", "KITA-016", "KITA-037"},
    "KITA-021": {"KITA-007", "KITA-019", "KITA-020"},
    "KITA-025": {"KITA-022", "KITA-023", "KITA-024"},
    "KITA-029": {
        "KITA-012",
        "KITA-016",
        "KITA-020",
        "KITA-024",
        "KITA-028",
        "KITA-040",
        "KITA-041",
    },
    "KITA-033": {
        "KITA-011",
        "KITA-016",
        "KITA-021",
        "KITA-029",
        "KITA-031",
        "KITA-032",
        "KITA-037",
    },
    "KITA-037": {"KITA-032", "KITA-034", "KITA-035", "KITA-036"},
    "KITA-042": {
        "KITA-013",
        "KITA-017",
        "KITA-021",
        "KITA-025",
        "KITA-029",
        "KITA-033",
        "KITA-037",
        "KITA-040",
        "KITA-041",
    },
    "KITA-044": {
        "KITA-009",
        "KITA-013",
        "KITA-017",
        "KITA-021",
        "KITA-025",
        "KITA-029",
        "KITA-033",
        "KITA-037",
        "KITA-042",
        "KITA-043",
    },
    "KITA-046": {
        "KITA-009",
        "KITA-013",
        "KITA-017",
        "KITA-021",
        "KITA-025",
        "KITA-029",
        "KITA-033",
        "KITA-037",
        "KITA-042",
        "KITA-045",
    },
    "KITA-047": {"KITA-046"},
}
REQUIRED_PLAN_TERMS = (
    "virtual buckets",
    "graphrag",
    "write-ahead log",
    "adaptive replacement cache",
    "replica",
    "ucan",
    "profile d",
    "cli",
    "mcp++",
    "transactions",
    "throughput",
    "storage backend",
    "tactician",
    "hammer",
    "llm_router",
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _taskboard_definition_sha256(text: str) -> str:
    """Hash the sealed board while excluding its one mutable field.

    The implementation daemon is authorized to persist only ``todo`` to
    ``completed`` status transitions.  Normalizing those values back to the
    launch projection makes that progress hash-neutral while retaining every
    task heading, dependency, contract, ownership field, and byte of the
    surrounding control document under the original seal.
    """

    normalized: list[str] = []
    current_task_id = ""
    for line in text.splitlines(keepends=True):
        if line.startswith("## KITA-"):
            header = line[3:].strip()
            current_task_id = header.split(" ", 1)[0] if header else ""
        if current_task_id and line.startswith("- Status:"):
            newline = (
                "\r\n"
                if line.endswith("\r\n")
                else "\n"
                if line.endswith("\n")
                else ""
            )
            initial_status = (
                "completed" if current_task_id == "KITA-000" else "todo"
            )
            line = f"- Status: {initial_status}{newline}"
        normalized.append(line)
    return "sha256:" + hashlib.sha256(
        "".join(normalized).encode("utf-8")
    ).hexdigest()


def _safe_relative_paths(values: Iterable[str], *, field: str) -> list[str]:
    errors: list[str] = []
    for raw in values:
        value = str(raw).strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() in {".", ".."}
            or (path.parts and path.parts[0].endswith(":"))
        ):
            errors.append(f"{field} contains unsafe path {raw!r}")
    return errors


def _cycle_nodes(edges: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str, lineage: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            if node in lineage:
                cycle.update(lineage[lineage.index(node) :])
            cycle.add(node)
            return
        visiting.add(node)
        for parent in edges.get(node, ()):
            visit(parent, (*lineage, node))
        visiting.remove(node)
        visited.add(node)

    for item in sorted(edges):
        visit(item, ())
    return tuple(sorted(cycle))


def _parse_positive_int(
    value: object, *, noun: str, errors: list[str], allow_zero: bool = False
) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        errors.append(f"{noun} is not an integer")
        return -1
    if parsed < (0 if allow_zero else 1):
        errors.append(f"{noun} is outside its positive bound")
    return parsed


def _load_scheduler(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.is_file():
        errors.append(f"scheduler file is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"scheduler is not valid JSON: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append("scheduler root must be an object")
        return {}
    return value


def validate(
    *,
    plan_path: Path,
    objective_path: Path,
    todo_path: Path,
    scheduler_path: Path,
) -> dict[str, object]:
    errors: list[str] = []
    for noun, path in (
        ("plan", plan_path),
        ("objective heap", objective_path),
        ("taskboard", todo_path),
    ):
        if not path.is_file():
            errors.append(f"{noun} is missing: {path}")
    if errors:
        return {"valid": False, "errors": errors}

    plan_text = plan_path.read_text(encoding="utf-8")
    plan_lower = plan_text.lower()
    for term in REQUIRED_PLAN_TERMS:
        if term not in plan_lower:
            errors.append(f"plan is missing required coverage term {term!r}")

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    goal_ids = tuple(goal.goal_id for goal in goals)
    goal_id_set = set(goal_ids)
    if goal_ids != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {GOAL_IDS}, got {goal_ids}")
    if len(goal_ids) != len(goal_id_set):
        errors.append("objective heap contains duplicate goal IDs")

    goal_edges: dict[str, tuple[str, ...]] = {}
    goal_dependency_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        if not re.fullmatch(r"KITA-G\d{3}", goal.goal_id):
            errors.append(f"invalid goal ID {goal.goal_id!r}")
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing:
            errors.append(f"{goal.goal_id} missing goal fields: {missing}")
        if goal.status not in GOAL_STATES:
            errors.append(f"{goal.goal_id} has invalid status {goal.status!r}")
        parent = str(goal.fields.get("parent") or "").strip()
        parents = (parent,) if parent else ()
        goal_edges[goal.goal_id] = parents
        if parent and parent not in goal_id_set:
            errors.append(f"{goal.goal_id} has unknown parent {parent!r}")
        goal_dependencies = _csv(goal.fields.get("depends_on", ""))
        goal_dependency_edges[goal.goal_id] = goal_dependencies
        for dependency in goal_dependencies:
            if dependency not in goal_id_set:
                errors.append(
                    f"{goal.goal_id} has unknown goal dependency {dependency!r}"
                )
            elif dependency == goal.goal_id:
                errors.append(f"{goal.goal_id} depends on itself")
        _parse_positive_int(
            goal.fields.get("fib_priority"),
            noun=f"{goal.goal_id} Fib priority",
            errors=errors,
        )
        if str(goal.fields.get("priority") or "") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{goal.goal_id} has invalid priority")
        for field in (
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
        ):
            if not str(goal.fields.get(field) or "").strip():
                errors.append(f"{goal.goal_id} has empty {field}")
        for field in ("outputs", "predicted_files"):
            errors.extend(
                f"{goal.goal_id}: {error}"
                for error in _safe_relative_paths(
                    _csv(goal.fields.get(field, "")), field=field
                )
            )

    goal_roots = tuple(
        sorted(goal_id for goal_id, parents in goal_edges.items() if not parents)
    )
    if goal_roots != ("KITA-G000",):
        errors.append(f"expected only KITA-G000 as root, got {goal_roots}")
    for goal_id in GOAL_IDS[1:]:
        if goal_edges.get(goal_id) != ("KITA-G000",):
            errors.append(f"{goal_id} must be a direct subgoal of KITA-G000")
    for name, edges in (
        ("goal parent", goal_edges),
        ("goal dependency", goal_dependency_edges),
    ):
        cycle = _cycle_nodes(edges)
        if cycle:
            errors.append(f"{name} graph contains a cycle: {list(cycle)}")

    todo_text = todo_path.read_text(encoding="utf-8")
    tasks = parse_task_text(
        todo_text,
        path=todo_path,
        task_header_prefix="## KITA-",
    )
    task_ids = tuple(task.task_id for task in tasks)
    task_id_set = set(task_ids)
    if task_ids != TASK_IDS:
        errors.append(f"task IDs/order differ: expected {TASK_IDS}, got {task_ids}")
    if len(task_ids) != len(task_id_set):
        errors.append("taskboard contains duplicate task IDs")
    taskboard_definition_sha256 = _taskboard_definition_sha256(todo_text)
    if taskboard_definition_sha256 != SEALED_TASKBOARD_DEFINITION_SHA256:
        errors.append(
            "taskboard topology or metadata differs from the sealed projection"
        )

    scheduler = _load_scheduler(scheduler_path, errors)
    protected_paths = tuple(
        str(item) for item in scheduler.get("protected_paths", ()) if str(item)
    )
    expected_protected = {
        ".gitignore",
        plan_path.relative_to(REPO_ROOT).as_posix(),
        objective_path.relative_to(REPO_ROOT).as_posix(),
        todo_path.relative_to(REPO_ROOT).as_posix(),
        scheduler_path.relative_to(REPO_ROOT).as_posix(),
        "scripts/validate_ipfs_kit_runtime_readiness_board.py",
        "test/api/test_ipfs_kit_runtime_readiness_board.py",
    }
    if set(protected_paths) != expected_protected:
        errors.append("scheduler protected_paths do not exactly match control artifacts")

    task_edges: dict[str, tuple[str, ...]] = {}
    task_records: list[dict[str, object]] = []
    completed: set[str] = set()
    blocked: set[str] = set()
    task_goal_map: dict[str, str] = {}
    protected_owned_after_seal: dict[str, list[str]] = {}
    for task in tasks:
        if not re.fullmatch(r"KITA-\d{3}", task.task_id):
            errors.append(f"invalid task ID {task.task_id!r}")
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in task.metadata]
        if missing:
            errors.append(f"{task.task_id} missing task fields: {missing}")
        if task.status not in TASK_STATES:
            errors.append(f"{task.task_id} has invalid status {task.status!r}")
        if task.task_id == "KITA-000":
            if task.status != "completed":
                errors.append("KITA-000 must be completed")
            if not str(task.metadata.get("completion evidence") or "").strip():
                errors.append("KITA-000 requires completion evidence")
        elif task.status not in PERSISTED_PROGRESS_STATES:
            errors.append(
                f"{task.task_id} has non-persistent progress status "
                f"{task.status!r}; the sealed board permits only todo or completed"
            )
        if task.status == "completed":
            completed.add(task.task_id)
        if task.status == "blocked":
            blocked.add(task.task_id)
        if task.priority not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{task.task_id} has invalid priority {task.priority!r}")
        if task.metadata.get("is schedulable") != "true":
            errors.append(f"{task.task_id} must be schedulable")
        if task.metadata.get("symbolic first") != "true":
            errors.append(f"{task.task_id} must declare symbolic-first analysis")
        if task.metadata.get("review only") not in {"true", "false"}:
            errors.append(f"{task.task_id} has invalid review-only flag")
        _parse_positive_int(
            task.metadata.get("estimated tokens"),
            noun=f"{task.task_id} estimated tokens",
            errors=errors,
        )
        _parse_positive_int(
            task.metadata.get("implementation timeout seconds"),
            noun=f"{task.task_id} implementation timeout",
            errors=errors,
        )
        _parse_positive_int(
            task.metadata.get("llm context budget bytes"),
            noun=f"{task.task_id} LLM context budget",
            errors=errors,
        )
        goal_id = str(task.metadata.get("goal id") or "").strip()
        task_goal_map[task.task_id] = goal_id
        if goal_id not in goal_id_set:
            errors.append(f"{task.task_id} has unknown goal ID {goal_id!r}")
        dependencies = tuple(task.depends_on)
        task_edges[task.task_id] = dependencies
        for dependency in dependencies:
            if dependency not in task_id_set:
                errors.append(
                    f"{task.task_id} has unknown task dependency {dependency!r}"
                )
            elif dependency == task.task_id:
                errors.append(f"{task.task_id} depends on itself")
        if task.task_id != "KITA-000" and not dependencies:
            errors.append(f"{task.task_id} must have a dependency")
        if not task.outputs:
            errors.append(f"{task.task_id} has no outputs")
        if not task.validation:
            errors.append(f"{task.task_id} has no validation command")
        if not task.acceptance:
            errors.append(f"{task.task_id} has empty acceptance")
        if task.board_namespace != BOARD_NAMESPACE:
            errors.append(
                f"{task.task_id} has unexpected board namespace "
                f"{task.board_namespace!r}"
            )
        predicted_files = _csv(task.metadata.get("predicted files", ""))
        if not predicted_files:
            errors.append(f"{task.task_id} has no predicted files")
        for field, values in (
            ("outputs", task.outputs),
            ("predicted files", predicted_files),
        ):
            errors.extend(
                f"{task.task_id}: {error}"
                for error in _safe_relative_paths(values, field=field)
            )
        if task.task_id != "KITA-000":
            outside_kit = sorted(
                path
                for path in (*task.outputs, *predicted_files)
                if not path.startswith("ipfs_kit_py/")
            )
            if outside_kit:
                errors.append(
                    f"{task.task_id} owns paths outside ipfs_kit_py: {outside_kit}"
                )
            overlaps = sorted(
                path
                for path in (*task.outputs, *predicted_files)
                if path in expected_protected
            )
            if overlaps:
                protected_owned_after_seal[task.task_id] = overlaps
        task_records.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "goal_id": goal_id,
                "depends_on": list(dependencies),
                "outputs": list(task.outputs),
                "acceptance": task.acceptance,
                "board_namespace": task.board_namespace,
                "canonical_task_cid": task.canonical_task_cid,
            }
        )

    if protected_owned_after_seal:
        errors.append(
            "post-seal tasks own protected artifacts: "
            + json.dumps(protected_owned_after_seal, sort_keys=True)
        )
    task_cycle = _cycle_nodes(task_edges)
    if task_cycle:
        errors.append(f"task dependency graph contains a cycle: {list(task_cycle)}")
    for task_id, required in REQUIRED_JOIN_DEPENDENCIES.items():
        missing = sorted(required.difference(task_edges.get(task_id, ())))
        if missing:
            errors.append(f"{task_id} missing required join dependencies: {missing}")
    incomplete_dependencies = {
        task_id: sorted(
            dependency
            for dependency in task_edges.get(task_id, ())
            if dependency not in completed
        )
        for task_id in sorted(completed)
        if any(
            dependency not in completed
            for dependency in task_edges.get(task_id, ())
        )
    }
    if incomplete_dependencies:
        errors.append(
            "completed tasks are not dependency-closed: "
            + json.dumps(incomplete_dependencies, sort_keys=True)
        )

    ready = tuple(
        task_id
        for task_id in task_ids
        if task_id not in completed
        and task_id not in blocked
        and all(dependency in completed for dependency in task_edges.get(task_id, ()))
    )
    waiting = tuple(
        task_id
        for task_id in task_ids
        if task_id not in completed
        and task_id not in blocked
        and task_id not in ready
    )
    if blocked:
        errors.append(
            "persistent board progress must not contain blocked tasks: "
            f"{sorted(blocked)}"
        )

    dependency_graph = materialize_task_dependency_dag(task_records)
    if dependency_graph.invalid_task_cids:
        errors.append(
            "typed dependency graph has invalid task CIDs: "
            f"{list(dependency_graph.invalid_task_cids)}"
        )
    if dependency_graph.repair_evidence:
        errors.append(
            "typed dependency graph requires repair: "
            + json.dumps(
                [item.to_dict() for item in dependency_graph.repair_evidence],
                sort_keys=True,
            )
        )

    expected_scheduler_scalars = {
        "taskboard_path": todo_path.relative_to(REPO_ROOT).as_posix(),
        "objectives_path": objective_path.relative_to(REPO_ROOT).as_posix(),
        "plan_path": plan_path.relative_to(REPO_ROOT).as_posix(),
        "validator_path": "scripts/validate_ipfs_kit_runtime_readiness_board.py",
        "task_prefix": "KITA-",
        "goal_prefix": "KITA-G",
        "board_namespace": BOARD_NAMESPACE,
        "max_lanes": 4,
        "strict_task_sharding": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for key, expected in expected_scheduler_scalars.items():
        if scheduler.get(key) != expected:
            errors.append(
                f"scheduler {key} mismatch: expected {expected!r}, "
                f"got {scheduler.get(key)!r}"
            )
    projection = scheduler.get("initial_projection")
    if not isinstance(projection, dict):
        errors.append("scheduler initial_projection must be an object")
    else:
        expected_projection = {
            "task_count": 48,
            "completed_task_ids": list(INITIAL_COMPLETED),
            "ready_task_ids": list(INITIAL_READY),
            "blocked_task_ids": [],
            "terminal_task_id": TERMINAL_TASK,
            "goal_count": 12,
            "root_goal_id": "KITA-G000",
        }
        if projection != expected_projection:
            errors.append("scheduler initial_projection differs from the launch seal")

    lanes = scheduler.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 4:
        errors.append("scheduler must define exactly four lanes")
    else:
        lane_initial: list[str] = []
        for index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                errors.append(f"scheduler lane {index} is not an object")
                continue
            if lane.get("index") != index:
                errors.append(f"scheduler lane {index} has wrong index")
            if lane.get("strict_shard_remainder") != index:
                errors.append(f"scheduler lane {index} has wrong shard remainder")
            initial = lane.get("initial_task_ids")
            if not isinstance(initial, list) or len(initial) != 1:
                errors.append(f"scheduler lane {index} needs one initial task")
                continue
            task_id = str(initial[0])
            lane_initial.append(task_id)
            numeric = int(task_id.rsplit("-", 1)[1])
            if numeric % 4 != index:
                errors.append(
                    f"scheduler lane {index} initial task {task_id} violates sharding"
                )
        if set(lane_initial) != set(INITIAL_READY):
            errors.append("scheduler lane initial tasks differ from parsed ready set")

    task_groups = scheduler.get("task_groups")
    if not isinstance(task_groups, dict):
        errors.append("scheduler task_groups must be an object")
    else:
        grouped: list[str] = []
        for goal_id in GOAL_IDS[1:]:
            raw = task_groups.get(goal_id)
            if not isinstance(raw, list) or not raw:
                errors.append(f"scheduler task group {goal_id} is missing or empty")
                continue
            for task_id in raw:
                selected = str(task_id)
                grouped.append(selected)
                if task_goal_map.get(selected) != goal_id:
                    errors.append(
                        f"scheduler maps {selected} to {goal_id}, "
                        f"board maps it to {task_goal_map.get(selected)!r}"
                    )
        if tuple(grouped) != TASK_IDS[1:]:
            errors.append("scheduler task groups do not cover KITA-001..047 exactly")

    if scheduler.get("worktree_submodule_paths") != [
        "ipfs_kit_py",
        "ipfs_datasets_py",
    ]:
        errors.append("scheduler worktree_submodule_paths are not exact")
    source_binding = scheduler.get("source_binding")
    if not isinstance(source_binding, dict):
        errors.append("scheduler source_binding must be an object")
    else:
        expected_revisions = {
            "accelerator_required_ancestor": "f25e5719cb738a50fb96bac4bea3f66ebca9800b",
            "ipfs_kit_planning_revision": "f6a574375febbcf9a46fcd24bbc7bc5cfb551de5",
            "ipfs_datasets_planning_revision": "7415adc5100192ee35676778f1018f6b072378f9",
        }
        for key, expected in expected_revisions.items():
            if source_binding.get(key) != expected:
                errors.append(f"scheduler source binding {key} is not pinned")
        for key in (
            "require_initialized_gitlinks",
            "require_superproject_gitlink_equals_nested_head",
            "record_recursive_repository_forest_at_launch",
            "changed_revision_requires_fresh_inventory_and_baseline",
        ):
            if source_binding.get(key) is not True:
                errors.append(f"scheduler source binding must enable {key}")

    report = {
        "schema": "ipfs_accelerate_py/ipfs-kit-runtime-readiness-preflight@1",
        "valid": not errors,
        "errors": errors,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "objective_path": str(objective_path),
        "objective_sha256": _sha256(objective_path),
        "goal_count": len(goals),
        "root_goal_ids": list(goal_roots),
        "todo_path": str(todo_path),
        "todo_sha256": _sha256(todo_path),
        "taskboard_definition_sha256": taskboard_definition_sha256,
        "task_count": len(tasks),
        "completed_task_ids": sorted(completed),
        "ready_task_ids": list(ready),
        "ready_task_count": len(ready),
        "waiting_task_count": len(waiting),
        "blocked_task_ids": sorted(blocked),
        "terminal_task_id": TERMINAL_TASK,
        "scheduler_path": str(scheduler_path),
        "scheduler_sha256": _sha256(scheduler_path)
        if scheduler_path.is_file()
        else "",
        "dependency_graph_id": _canonical_sha256(dependency_graph.to_dict()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--plan-path", type=Path, default=PLAN_PATH)
    parser.add_argument("--objective-path", type=Path, default=OBJECTIVE_PATH)
    parser.add_argument("--todo-path", type=Path, default=TODO_PATH)
    parser.add_argument("--scheduler-path", type=Path, default=SCHEDULER_PATH)
    args = parser.parse_args()
    del args.check_all
    result = validate(
        plan_path=args.plan_path.resolve(),
        objective_path=args.objective_path.resolve(),
        todo_path=args.todo_path.resolve(),
        scheduler_path=args.scheduler_path.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
