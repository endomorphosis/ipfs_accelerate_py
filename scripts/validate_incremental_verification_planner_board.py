#!/usr/bin/env python3
"""Fail-closed validator for the incremental-verification planner board."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_text,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_PLAN.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/incremental_verification_planner.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/incremental_verification_planner.todo.md"
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_incremental_verification_planner_scheduler.json"
)

BOARD_NAMESPACE = "incremental-verification-planner-v1"
BRANCH = "integration/incremental-verification-planner-main-20260811"
BASE_REVISION = "c1e33e8f443253e106c464d7c5b5c341c3095876"
TASK_IDS = tuple(f"IVP-{index:03d}" for index in range(21))
GOAL_IDS = (
    "IVP-G000",
    "IVP-G010",
    "IVP-G020",
    "IVP-G030",
    "IVP-G040",
    "IVP-G050",
    "IVP-G060",
    "IVP-G070",
    "IVP-G080",
    "IVP-G090",
    "IVP-G100",
)
TERMINAL_TASK = "IVP-019"
INITIAL_COMPLETED = ("IVP-000",)
INITIAL_READY = ("IVP-001",)

EXPECTED_DEPENDENCIES = {
    "IVP-000": (),
    "IVP-001": ("IVP-000",),
    "IVP-002": ("IVP-001",),
    "IVP-003": ("IVP-001",),
    "IVP-004": ("IVP-001",),
    "IVP-005": ("IVP-001", "IVP-004"),
    "IVP-006": ("IVP-001", "IVP-004"),
    "IVP-007": ("IVP-001", "IVP-002", "IVP-004"),
    "IVP-008": ("IVP-001", "IVP-003"),
    "IVP-009": ("IVP-001", "IVP-002"),
    "IVP-010": ("IVP-002", "IVP-008", "IVP-009"),
    "IVP-011": ("IVP-005",),
    "IVP-012": ("IVP-001",),
    "IVP-013": ("IVP-001",),
    "IVP-014": (
        "IVP-005",
        "IVP-006",
        "IVP-007",
        "IVP-008",
        "IVP-010",
        "IVP-011",
        "IVP-012",
        "IVP-013",
    ),
    "IVP-015": ("IVP-009", "IVP-014"),
    "IVP-016": ("IVP-012", "IVP-013", "IVP-014", "IVP-015"),
    "IVP-017": ("IVP-015",),
    "IVP-018": ("IVP-017",),
    "IVP-019": ("IVP-016", "IVP-017", "IVP-018", "IVP-020"),
    "IVP-020": ("IVP-016", "IVP-017", "IVP-018"),
}

EXPECTED_GROUPS = {
    "IVP-G010": ("IVP-001",),
    "IVP-G020": ("IVP-002", "IVP-003", "IVP-004"),
    "IVP-G030": ("IVP-008",),
    "IVP-G040": ("IVP-005", "IVP-006", "IVP-007"),
    "IVP-G050": ("IVP-009", "IVP-010"),
    "IVP-G060": ("IVP-011", "IVP-013"),
    "IVP-G070": ("IVP-012",),
    "IVP-G080": ("IVP-014",),
    "IVP-G090": ("IVP-015", "IVP-016", "IVP-017"),
    "IVP-G100": ("IVP-018", "IVP-020", "IVP-019"),
}

EXPECTED_GOAL_DEPENDENCIES = {
    "IVP-G000": (),
    "IVP-G010": (),
    "IVP-G020": ("IVP-G010",),
    "IVP-G030": ("IVP-G010", "IVP-G020"),
    "IVP-G040": ("IVP-G010", "IVP-G020"),
    "IVP-G050": ("IVP-G010", "IVP-G020", "IVP-G030"),
    "IVP-G060": ("IVP-G010", "IVP-G020", "IVP-G040"),
    "IVP-G070": ("IVP-G010",),
    "IVP-G080": ("IVP-G030", "IVP-G040", "IVP-G050", "IVP-G060", "IVP-G070"),
    "IVP-G090": ("IVP-G080",),
    "IVP-G100": ("IVP-G090",),
}

TASK_GOALS = {
    "IVP-000": "IVP-G000",
    **{
        task_id: goal_id
        for goal_id, task_ids in EXPECTED_GROUPS.items()
        for task_id in task_ids
    },
}

TASK_STATES = frozenset({"todo", "completed"})
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
    "evidence_requirements_json",
    "evidence_criteria",
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

REQUIRED_PLAN_TERMS = (
    "incrementalverificationplanner",
    "verificationreceiptcache",
    "modelrouteplanner",
    "create_verification_plan",
    "choose_model_route",
    "build_verification_commitment",
    "repositorystate",
    "invalidationplan",
    "semanticcapsule",
    "contextpack",
    "repository tree cid",
    "semantic-state root cid",
    "affected symbol-version cids",
    "environment cid",
    "dependency-lock cid",
    "selector cid",
    "proof-obligation cid",
    "tool version",
    "configuration cid",
    "fixture-data cids",
    "network policy",
    "receipt-schema version",
    "staticanalysisreceipt",
    "typecheckreceipt",
    "testreceipt",
    "proofreceipt",
    "counterexamplereceipt",
    "verificationbundle",
    "verificationsummary",
    "cachereusedecision",
    "modelroutedecision",
    "passed, failed, proved, disproved, unknown, timeout, unavailable",
    "not_modeled, stale, invalid, cancelled, simulated",
    "pytest",
    "mypy",
    "z3",
    "proof assistant",
    "false-negative",
    "full-suite fallback",
    "deterministic_only",
    "small_local_model",
    "medium_model",
    "frontier_model",
    "human_review_required",
    "not a zero-knowledge proof",
    "signed receipts do not prove test execution",
    "structural validation is not cryptographic validation",
    "automatic dependency installation",
    "x402",
    "cache hit rate",
    "frontier escalation rate",
    "verification tokens saved",
)

REQUIRED_CONFORMANCE_TERMS = (
    "unchanged receipt",
    "relevant code change",
    "unrelated code change",
    "environment change",
    "dependency-lock change",
    "tool-version change",
    "stale receipt",
    "simulated receipt",
    "timeout remains timeout",
    "unavailable prover",
    "minimized counterexample",
    "uncertain test selection",
    "concurrent cache writers",
    "cancellation terminates child processes",
    "small-model route",
    "frontier route",
    "high-risk policy",
    "commitment changes whenever a required receipt changes",
)

EXPECTED_WAVES = (
    ("IVP-000",),
    ("IVP-001",),
    ("IVP-002", "IVP-003", "IVP-004", "IVP-012", "IVP-013"),
    ("IVP-005", "IVP-006", "IVP-007", "IVP-008", "IVP-009"),
    ("IVP-010", "IVP-011"),
    ("IVP-014",),
    ("IVP-015",),
    ("IVP-016", "IVP-017"),
    ("IVP-018",),
    ("IVP-020",),
    ("IVP-019",),
)

PROTECTED_PATHS = {
    ".gitignore",
    "docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_PLAN.md",
    "docs/architecture/incremental_verification_planner.objectives.md",
    "docs/architecture/incremental_verification_planner.todo.md",
    "config/agent_supervisor_incremental_verification_planner_scheduler.json",
    "scripts/validate_incremental_verification_planner_board.py",
    "scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py",
}


def _csv(value: object) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in re.split(r"[,;]", str(value or ""))
        if part.strip()
    )


def _positive_int(
    value: object,
    *,
    noun: str,
    errors: list[str],
    allow_zero: bool = False,
) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        errors.append(f"{noun} is not an integer")
        return -1
    if parsed < (0 if allow_zero else 1):
        errors.append(f"{noun} is outside its admitted range")
    return parsed


def _safe_paths(values: Iterable[str], *, noun: str, errors: list[str]) -> None:
    for raw in values:
        normalized = str(raw).strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or "\x00" in normalized
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() in {".", ".."}
            or (path.parts and path.parts[0].endswith(":"))
        ):
            errors.append(f"{noun} contains unsafe path {raw!r}")


def _cycle_nodes(edges: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str, lineage: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle.add(node)
            if node in lineage:
                cycle.update(lineage[lineage.index(node) :])
            return
        visiting.add(node)
        for dependency in edges.get(node, ()):
            visit(dependency, (*lineage, node))
        visiting.remove(node)
        visited.add(node)

    for node in sorted(edges):
        visit(node, ())
    return tuple(sorted(cycle))


def _transitive_dependencies(
    task_id: str, edges: Mapping[str, tuple[str, ...]]
) -> frozenset[str]:
    pending = list(edges.get(task_id, ()))
    reached: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency in reached:
            continue
        reached.add(dependency)
        pending.extend(edges.get(dependency, ()))
    return frozenset(reached)


def _load_json(path: Path, errors: list[str]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name} is not valid JSON: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} root must be an object")
        return {}
    return value


def _validate_one_line_task_metadata(text: str, errors: list[str]) -> None:
    matches = list(re.finditer(r"^## (IVP-\d{3})\b.*$", text, re.MULTILINE))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        fields: set[str] = set()
        for offset, line in enumerate(block.splitlines(), start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > 16_384:
                errors.append(f"{match.group(1)} metadata row exceeds 16 KiB")
            field_match = re.fullmatch(r"- ([A-Za-z][A-Za-z ]*):(?: .*)?", line)
            if field_match is None:
                errors.append(
                    f"{match.group(1)} contains non-one-line metadata at block line {offset}"
                )
                continue
            field = field_match.group(1).strip().lower()
            if field in fields:
                errors.append(f"{match.group(1)} repeats metadata field {field!r}")
            fields.add(field)


def _validate_plan(plan_text: str, todo_text: str, errors: list[str]) -> None:
    normalized = " ".join(plan_text.lower().split())
    for term in REQUIRED_PLAN_TERMS:
        if term not in normalized:
            errors.append(f"plan is missing required coverage term {term!r}")
    todo_normalized = " ".join(f"{plan_text}\n{todo_text}".lower().split())
    for term in REQUIRED_CONFORMANCE_TERMS:
        if term not in todo_normalized:
            errors.append(f"taskboard is missing conformance case {term!r}")
    for index, task_ids in enumerate(EXPECTED_WAVES):
        row_prefix = f"| {index} | {', '.join(task_ids)} |"
        if row_prefix not in plan_text:
            errors.append(f"plan wave {index} differs from task DAG")


def _validate_goals(text: str, errors: list[str]) -> None:
    goals = parse_goal_heap(text)
    observed = tuple(goal.goal_id for goal in goals)
    if observed != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {GOAL_IDS}, got {observed}")
    if len(observed) != len(set(observed)):
        errors.append("objective heap contains duplicate goal IDs")
    goal_set = set(observed)
    parent_edges: dict[str, tuple[str, ...]] = {}
    dependency_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing:
            errors.append(f"{goal.goal_id} is missing goal fields: {missing}")
        if goal.status not in GOAL_STATES:
            errors.append(f"{goal.goal_id} has invalid status {goal.status!r}")
        parent = str(goal.fields.get("parent") or "").strip()
        parent_edges[goal.goal_id] = (parent,) if parent else ()
        if parent and parent not in goal_set:
            errors.append(f"{goal.goal_id} has unknown parent {parent!r}")
        dependencies = _csv(goal.fields.get("depends_on"))
        dependency_edges[goal.goal_id] = dependencies
        if dependencies != EXPECTED_GOAL_DEPENDENCIES.get(goal.goal_id):
            errors.append(
                f"{goal.goal_id} dependencies differ: expected "
                f"{EXPECTED_GOAL_DEPENDENCIES.get(goal.goal_id)}, got {dependencies}"
            )
        for dependency in dependencies:
            if dependency not in goal_set:
                errors.append(f"{goal.goal_id} has unknown dependency {dependency!r}")
        _positive_int(
            goal.fields.get("fib_priority"),
            noun=f"{goal.goal_id} Fib priority",
            errors=errors,
        )
        if str(goal.fields.get("priority") or "") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{goal.goal_id} has invalid priority")
        for field in (
            "goal",
            "evidence",
            "evidence_requirements_json",
            "evidence_criteria",
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
        _safe_paths(
            _csv(goal.fields.get("outputs")), noun=f"{goal.goal_id} outputs", errors=errors
        )
        _safe_paths(
            _csv(goal.fields.get("predicted_files")),
            noun=f"{goal.goal_id} predicted files",
            errors=errors,
        )
        evidence_requirements = str(goal.fields.get("evidence_requirements_json") or "")
        try:
            evidence_value = json.loads(evidence_requirements)
        except json.JSONDecodeError:
            errors.append(f"{goal.goal_id} evidence requirements JSON is invalid")
        else:
            if (
                not isinstance(evidence_value, list)
                or not evidence_value
                or any(not isinstance(item, str) or not item for item in evidence_value)
            ):
                errors.append(f"{goal.goal_id} evidence requirements must be a string list")
        evidence_criteria = str(goal.fields.get("evidence_criteria") or "")
        try:
            criteria_value = json.loads(evidence_criteria)
        except json.JSONDecodeError:
            errors.append(f"{goal.goal_id} evidence criteria JSON is invalid")
        else:
            if not isinstance(criteria_value, dict) or not criteria_value:
                errors.append(f"{goal.goal_id} evidence criteria must be an object")

    roots = tuple(sorted(goal for goal, parents in parent_edges.items() if not parents))
    if roots != ("IVP-G000",):
        errors.append(f"expected IVP-G000 as the only root, got {roots}")
    for goal_id in GOAL_IDS[1:]:
        if parent_edges.get(goal_id) != ("IVP-G000",):
            errors.append(f"{goal_id} must be a direct IVP-G000 child")
    for noun, edges in (
        ("goal-parent", parent_edges),
        ("goal-dependency", dependency_edges),
    ):
        cycle = _cycle_nodes(edges)
        if cycle:
            errors.append(f"{noun} graph has a cycle: {cycle}")


def _validate_tasks(text: str, scheduler: Mapping[str, object], errors: list[str]) -> None:
    _validate_one_line_task_metadata(text, errors)
    tasks = parse_task_text(text, path=TODO_PATH, task_header_prefix="## IVP-")
    observed = tuple(task.task_id for task in tasks)
    if observed != TASK_IDS:
        errors.append(f"task IDs/order differ: expected {TASK_IDS}, got {observed}")
    if len(observed) != len(set(observed)):
        errors.append("taskboard contains duplicate task IDs")
    task_set = set(observed)
    edges: dict[str, tuple[str, ...]] = {}
    predicted_by_task: dict[str, set[str]] = {}
    max_timeout = int(scheduler.get("implementation_max_timeout_seconds") or 0)
    for task in tasks:
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in task.metadata]
        if missing:
            errors.append(f"{task.task_id} is missing task fields: {missing}")
        if task.status not in TASK_STATES:
            errors.append(f"{task.task_id} has inadmissible persisted status {task.status!r}")
        expected_completion = "manual" if task.task_id == "IVP-000" else "auto"
        if task.completion != expected_completion:
            errors.append(f"{task.task_id} completion must be {expected_completion}")
        if task.task_id == "IVP-000" and task.status != "completed":
            errors.append("IVP-000 must remain completed")
        if task.priority not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{task.task_id} has invalid priority {task.priority!r}")
        if task.metadata.get("is schedulable") != "true":
            errors.append(f"{task.task_id} must remain schedulable")
        if task.metadata.get("review only") != "false":
            errors.append(f"{task.task_id} must be implementation work, not review-only")
        if task.metadata.get("symbolic first") != "true":
            errors.append(f"{task.task_id} must require symbolic-first analysis")
        if task.board_namespace != BOARD_NAMESPACE:
            errors.append(f"{task.task_id} has wrong board namespace")
        _positive_int(
            task.metadata.get("estimated tokens"),
            noun=f"{task.task_id} estimated tokens",
            errors=errors,
            allow_zero=task.task_id == "IVP-000",
        )
        timeout = _positive_int(
            task.metadata.get("implementation timeout seconds"),
            noun=f"{task.task_id} implementation timeout",
            errors=errors,
        )
        if max_timeout > 0 and timeout > max_timeout:
            errors.append(f"{task.task_id} timeout exceeds scheduler maximum")
        _positive_int(
            task.metadata.get("llm context budget bytes"),
            noun=f"{task.task_id} LLM context budget",
            errors=errors,
        )
        dependencies = tuple(task.depends_on)
        edges[task.task_id] = dependencies
        if dependencies != EXPECTED_DEPENDENCIES.get(task.task_id):
            errors.append(
                f"{task.task_id} dependencies differ: expected "
                f"{EXPECTED_DEPENDENCIES.get(task.task_id)}, got {dependencies}"
            )
        for dependency in dependencies:
            if dependency not in task_set:
                errors.append(f"{task.task_id} has unknown dependency {dependency!r}")
        if not task.outputs or not task.validation or not task.acceptance:
            errors.append(f"{task.task_id} has empty output, validation, or acceptance")
        predicted = set(_csv(task.metadata.get("predicted files")))
        predicted_by_task[task.task_id] = predicted
        if not predicted:
            errors.append(f"{task.task_id} has no predicted files")
        _safe_paths(task.outputs, noun=f"{task.task_id} outputs", errors=errors)
        _safe_paths(predicted, noun=f"{task.task_id} predicted files", errors=errors)
        goal_id = str(task.metadata.get("goal id") or "")
        if task.task_id == "IVP-000":
            expected_goal = "IVP-G000"
        else:
            expected_goal = next(
                (goal for goal, members in EXPECTED_GROUPS.items() if task.task_id in members),
                "",
            )
        if goal_id != expected_goal:
            errors.append(
                f"{task.task_id} goal differs: expected {expected_goal!r}, got {goal_id!r}"
            )
        concurrent = _csv(task.metadata.get("allow concurrent with"))
        if concurrent:
            errors.append(f"{task.task_id} concurrency metadata must remain advisory-empty")
        for peer in concurrent:
            if peer not in task_set or peer == task.task_id:
                errors.append(f"{task.task_id} has invalid concurrent peer {peer!r}")
        if task.task_id != "IVP-000":
            forbidden = sorted(PROTECTED_PATHS.intersection((*task.outputs, *predicted)))
            if forbidden:
                errors.append(f"{task.task_id} owns protected controls: {forbidden}")
        if task.task_id != "IVP-000":
            validation_text = str(task.metadata.get("validation") or "")
            if "PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:." not in validation_text:
                errors.append(f"{task.task_id} validation lacks exact nested package roots")
            if "python3 -m pytest" in validation_text and "--timeout=" not in validation_text:
                errors.append(f"{task.task_id} pytest validation lacks a per-test timeout")

    for task_id, dependencies in edges.items():
        goal_id = TASK_GOALS.get(task_id, "")
        goal_closure = _transitive_dependencies(goal_id, EXPECTED_GOAL_DEPENDENCIES)
        for dependency in dependencies:
            if dependency == "IVP-000":
                continue
            dependency_goal = TASK_GOALS.get(dependency, "")
            if dependency_goal not in {goal_id, *goal_closure}:
                errors.append(
                    f"{task_id} cross-goal dependency {dependency} is absent from "
                    f"{goal_id} goal lineage"
                )

    cycle = _cycle_nodes(edges)
    if cycle:
        errors.append(f"task dependency graph has a cycle: {cycle}")
    completed = {task.task_id for task in tasks if task.status == "completed"}
    if not set(INITIAL_COMPLETED).issubset(completed):
        errors.append("planning control task is not completed")
    if completed == set(INITIAL_COMPLETED):
        ready = tuple(
            task_id
            for task_id in TASK_IDS
            if task_id not in completed
            and all(dependency in completed for dependency in edges.get(task_id, ()))
        )
        if ready != INITIAL_READY:
            errors.append(f"initial ready projection differs: expected {INITIAL_READY}, got {ready}")

    for left_index, left in enumerate(TASK_IDS):
        left_closure = _transitive_dependencies(left, edges)
        for right in TASK_IDS[left_index + 1 :]:
            overlap = predicted_by_task.get(left, set()).intersection(
                predicted_by_task.get(right, set())
            )
            if not overlap:
                continue
            right_closure = _transitive_dependencies(right, edges)
            if left not in right_closure and right not in left_closure:
                errors.append(
                    f"unordered tasks {left}/{right} overlap predicted files: {sorted(overlap)}"
                )


def _validate_scheduler(scheduler: Mapping[str, object], errors: list[str]) -> None:
    exact = {
        "schema": "ipfs_accelerate_py.agent_supervisor.incremental_verification_planner.scheduler_config@1",
        "taskboard_path": TODO_PATH.relative_to(REPO_ROOT).as_posix(),
        "objectives_path": OBJECTIVE_PATH.relative_to(REPO_ROOT).as_posix(),
        "plan_path": PLAN_PATH.relative_to(REPO_ROOT).as_posix(),
        "validator_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "task_prefix": "IVP-",
        "goal_prefix": "IVP-G",
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": BRANCH,
        "max_lanes": 3,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for field, expected in exact.items():
        if scheduler.get(field) != expected:
            errors.append(
                f"scheduler {field} differs: expected {expected!r}, got {scheduler.get(field)!r}"
            )
    source = scheduler.get("source_binding")
    if not isinstance(source, Mapping):
        errors.append("scheduler source_binding must be an object")
    else:
        if source.get("accelerator_required_ancestor") != BASE_REVISION:
            errors.append("scheduler required ancestor differs from reviewed base")
        if source.get("accelerator_required_branch") != BRANCH:
            errors.append("scheduler required branch differs")
        if source.get("bootstrap_task_source") != "legacy-markdown":
            errors.append("scheduler task source must be explicit legacy-markdown")
        expected_gitlinks = {
            "ipfs_kit": (
                "ipfs_kit_py",
                "5a7a2df8181cfdc33bc19be09989df7ff83f2d4e",
            ),
            "ipfs_datasets": (
                "ipfs_datasets_py",
                "6cd037c7738f44904add46391537588e67f6f238",
            ),
        }
        for prefix, (path, revision) in expected_gitlinks.items():
            if source.get(f"{prefix}_submodule_path") != path:
                errors.append(f"scheduler {prefix} submodule path differs")
            if source.get(f"{prefix}_planning_revision") != revision:
                errors.append(f"scheduler {prefix} planning revision differs")
        if source.get("require_initialized_gitlinks") is not True:
            errors.append("scheduler must require initialized gitlinks")
        if source.get("require_superproject_gitlink_equals_nested_head") is not True:
            errors.append("scheduler must bind each nested HEAD to its gitlink")
        if source.get("require_clean_nested_worktree_at_task_start") is not True:
            errors.append("scheduler must require clean nested worktrees")
    if scheduler.get("worktree_submodule_paths") != [
        "ipfs_kit_py",
        "ipfs_datasets_py",
    ]:
        errors.append("scheduler must expose the two exact read-only adapter gitlinks")
    protected = scheduler.get("protected_paths")
    if not isinstance(protected, list) or set(map(str, protected)) != PROTECTED_PATHS:
        errors.append("scheduler protected paths differ from the seven control artifacts")
    runtime = scheduler.get("runtime_paths")
    expected_root = "data/agent_supervisor/incremental_verification_planner"
    if not isinstance(runtime, Mapping) or runtime.get("root") != expected_root:
        errors.append("scheduler runtime root differs")
    else:
        for field in ("state", "worktrees", "merge_queue", "logs"):
            value = str(runtime.get(field) or "")
            if not value.startswith(expected_root + "/"):
                errors.append(f"scheduler runtime {field} escapes its root")
    ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    if f"/{expected_root}/" not in ignore_text:
        errors.append("scheduler runtime root is not ignored")
    if (
        "!artifacts/agent_supervisor/incremental_verification/benchmark.json"
        not in ignore_text
    ):
        errors.append("stable benchmark JSON lacks an exact Git ignore exception")

    lanes = scheduler.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 3:
        errors.append("scheduler must have three lanes")
    else:
        for index, lane in enumerate(lanes):
            if not isinstance(lane, Mapping):
                errors.append(f"scheduler lane {index} is not an object")
                continue
            if lane.get("index") != index or lane.get("strict_shard_remainder") != index:
                errors.append(f"scheduler lane {index} has a sharding mismatch")
            initial = tuple(str(item) for item in lane.get("initial_task_ids", ()))
            for task_id in initial:
                remainder = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 3
                if remainder != index:
                    errors.append(f"scheduler initial task {task_id} is assigned to wrong lane")
        all_initial = tuple(
            str(task_id)
            for lane in lanes
            if isinstance(lane, Mapping)
            for task_id in lane.get("initial_task_ids", ())
        )
        if all_initial != INITIAL_READY:
            errors.append(f"scheduler lane initial tasks differ: {all_initial}")
    groups = scheduler.get("task_groups")
    if not isinstance(groups, Mapping):
        errors.append("scheduler task_groups must be an object")
    else:
        normalized_groups = {
            str(goal): tuple(str(task) for task in members)
            for goal, members in groups.items()
            if isinstance(members, list)
        }
        if normalized_groups != EXPECTED_GROUPS:
            errors.append("scheduler task_groups differ from the reviewed goal projection")
    projection = scheduler.get("initial_projection")
    expected_projection = {
        "task_count": 21,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 11,
        "root_goal_id": "IVP-G000",
    }
    if projection != expected_projection:
        errors.append("scheduler initial projection differs")
    provider = scheduler.get("provider")
    if not isinstance(provider, Mapping):
        errors.append("scheduler provider must be an object")
    else:
        expected_provider_route = {
            "primary_provider_id": "grok_cli",
            "primary_model_id": "grok-4.5",
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_trigger": "primary_quota_exhausted",
            "fallback_reasoning_effort": "high",
        }
        observed_provider_route = {
            field: provider.get(field) for field in expected_provider_route
        }
        if observed_provider_route != expected_provider_route:
            errors.append("scheduler ordered implementation provider route differs")
        if "provider_id" in provider or "model_id" in provider:
            errors.append("scheduler must not mix legacy and ordered provider fields")
        if provider.get("max_concurrency") != 3:
            errors.append("scheduler provider concurrency must equal lane count")
        if provider.get("secrets_from_environment_only") is not True:
            errors.append("scheduler provider secrets must come only from environment")
        if provider.get("secrets_in_argv_prompts_logs_or_receipts") is not False:
            errors.append("scheduler must prohibit secrets in execution evidence")
    for field in (
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
        _positive_int(scheduler.get(field), noun=f"scheduler {field}", errors=errors)
    completion = scheduler.get("completion_policy")
    if not isinstance(completion, Mapping) or completion.get("terminal_task_id") != TERMINAL_TASK:
        errors.append("scheduler completion policy has wrong terminal task")
    elif (
        completion.get("all_task_dependencies_terminal_required") is not True
        or completion.get("goal_heap_is_planning_lineage_only") is not True
    ):
        errors.append("scheduler completion policy misstates task/goal authority")


def _validate_nested_capability_seams(errors: list[str]) -> None:
    expected_files = (
        REPO_ROOT
        / "ipfs_kit_py/ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
        REPO_ROOT
        / "ipfs_datasets_py/ipfs_datasets_py/knowledge_graphs/adapters/code_evidence.py",
        REPO_ROOT
        / "ipfs_datasets_py/ipfs_datasets_py/logic/backends/process.py",
    )
    for path in expected_files:
        if not path.is_file():
            errors.append(f"required pinned capability source is absent: {path}")
    if any(not path.is_file() for path in expected_files):
        return
    probe = (
        "from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import "
        "ArtifactIntegrityError, ArtifactNotFound, DurableCoordinationStore\n"
        "from ipfs_datasets_py.knowledge_graphs.adapters.code_evidence import "
        "CodeEvidenceCorpusAdapter, impact_from_index, normalize_impact_index\n"
        "from ipfs_datasets_py.logic.backends.process import BoundedToolRunner\n"
        "assert all((ArtifactIntegrityError, ArtifactNotFound, "
        "DurableCoordinationStore, CodeEvidenceCorpusAdapter, impact_from_index, "
        "normalize_impact_index, BoundedToolRunner))"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(
            (
                str(REPO_ROOT / "ipfs_kit_py"),
                str(REPO_ROOT / "ipfs_datasets_py"),
                str(REPO_ROOT),
            )
        ),
    }
    try:
        result = subprocess.run(
            (sys.executable, "-P", "-c", probe),
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"pinned capability leaf probe failed: {type(exc).__name__}")
        return
    if result.returncode != 0:
        errors.append(
            "pinned capability leaf probe rejected: "
            + (result.stderr.strip()[-1000:] or f"exit {result.returncode}")
        )


def validate() -> dict[str, object]:
    errors: list[str] = []
    for noun, path in (
        ("plan", PLAN_PATH),
        ("objective heap", OBJECTIVE_PATH),
        ("taskboard", TODO_PATH),
        ("scheduler", SCHEDULER_PATH),
    ):
        if not path.is_file():
            errors.append(f"{noun} is missing: {path}")
    if errors:
        return {"schema": "ivp-board-validation@1", "valid": False, "errors": errors}
    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVE_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    scheduler = _load_json(SCHEDULER_PATH, errors)
    _validate_plan(plan_text, todo_text, errors)
    _validate_goals(objective_text, errors)
    _validate_scheduler(scheduler, errors)
    _validate_nested_capability_seams(errors)
    _validate_tasks(todo_text, scheduler, errors)
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-board-validation@1",
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "task_count": len(TASK_IDS),
        "goal_count": len(GOAL_IDS),
        "initial_completed_task_ids": list(INITIAL_COMPLETED),
        "initial_ready_task_ids": list(INITIAL_READY),
        "terminal_task_id": TERMINAL_TASK,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()
    if not args.check_all:
        parser.error("--check-all is required")
    report = validate()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["valid"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
