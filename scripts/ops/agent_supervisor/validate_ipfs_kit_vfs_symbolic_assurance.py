#!/usr/bin/env python3
"""Fail-closed preflight for the VFS symbolic-assurance objective and board."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objective_graph import (  # noqa: E402
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_file,
)


OBJECTIVE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "ipfs_kit_vfs_symbolic_assurance.objectives.md"
)
TODO_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "ipfs_kit_vfs_symbolic_assurance.todo.md"
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
REQUIRED_GOAL_FIELDS = (
    "status",
    "parent",
    "fib_priority",
    "track",
    "priority",
    "bundle",
    "goal",
    "evidence",
    "outputs",
    "validation",
    "acceptance",
    "gap_task",
)
REQUIRED_TASK_FIELDS = (
    "status",
    "completion",
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
    "acceptance",
)


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


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


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


def validate(objective_path: Path, todo_path: Path) -> dict[str, object]:
    errors: list[str] = []
    if not objective_path.is_file():
        errors.append(f"objective file is missing: {objective_path}")
    if not todo_path.is_file():
        errors.append(f"task board is missing: {todo_path}")
    if errors:
        return {"valid": False, "errors": errors}

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    goal_ids = [goal.goal_id for goal in goals]
    goal_id_set = set(goal_ids)
    if len(goal_ids) != len(goal_id_set):
        duplicate_ids = sorted(
            item for item in goal_id_set if goal_ids.count(item) > 1
        )
        errors.append(f"duplicate goal ids: {duplicate_ids}")
    if not goals:
        errors.append("objective heap is empty")

    goal_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        if not re.fullmatch(r"VFS-G\d{3}", goal.goal_id):
            errors.append(f"invalid goal id: {goal.goal_id}")
        missing = [name for name in REQUIRED_GOAL_FIELDS if name not in goal.fields]
        if missing:
            errors.append(f"{goal.goal_id} missing fields: {missing}")
        status = str(goal.fields.get("status") or "").strip()
        if status not in GOAL_STATES:
            errors.append(f"{goal.goal_id} has noncanonical status {status!r}")
        parent = str(goal.fields.get("parent") or "").strip()
        parents = (parent,) if parent else ()
        goal_edges[goal.goal_id] = parents
        if parent and parent not in goal_id_set:
            errors.append(f"{goal.goal_id} has unknown parent {parent}")
        try:
            fib_priority = int(str(goal.fields.get("fib_priority") or ""))
            if fib_priority < 1:
                raise ValueError
        except ValueError:
            errors.append(f"{goal.goal_id} has invalid fib priority")
        outputs = _csv(goal.fields.get("outputs", ""))
        if not outputs:
            errors.append(f"{goal.goal_id} has no outputs")
        errors.extend(
            f"{goal.goal_id}: {item}"
            for item in _safe_relative_paths(outputs, field="outputs")
        )
        for name in ("goal", "evidence", "validation", "acceptance", "gap_task"):
            if not str(goal.fields.get(name) or "").strip():
                errors.append(f"{goal.goal_id} has empty {name}")
    goal_cycles = _cycle_nodes(goal_edges)
    if goal_cycles:
        errors.append(f"goal parent cycle: {list(goal_cycles)}")
    roots = sorted(goal_id for goal_id, parents in goal_edges.items() if not parents)
    if roots != ["VFS-G000"]:
        errors.append(f"expected only VFS-G000 as root, got {roots}")

    tasks = parse_task_file(todo_path, "## VFS-")
    task_ids = [task.task_id for task in tasks]
    task_id_set = set(task_ids)
    if len(task_ids) != len(task_id_set):
        duplicate_ids = sorted(
            item for item in task_id_set if task_ids.count(item) > 1
        )
        errors.append(f"duplicate task ids: {duplicate_ids}")
    if not tasks:
        errors.append("task board is empty")

    task_edges: dict[str, tuple[str, ...]] = {}
    task_records: list[dict[str, object]] = []
    for task in tasks:
        if not re.fullmatch(r"VFS-\d{3}", task.task_id):
            errors.append(f"invalid task id: {task.task_id}")
        missing = [name for name in REQUIRED_TASK_FIELDS if name not in task.metadata]
        if missing:
            errors.append(f"{task.task_id} missing fields: {missing}")
        if task.status not in TASK_STATES:
            errors.append(
                f"{task.task_id} has noncanonical normalized status {task.status!r}"
            )
        if task.priority not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{task.task_id} has invalid priority {task.priority!r}")
        goal_id = str(task.metadata.get("goal id") or "").strip()
        if goal_id not in goal_id_set:
            errors.append(f"{task.task_id} has unknown goal id {goal_id!r}")
        dependencies = tuple(task.depends_on)
        task_edges[task.task_id] = tuple(
            item for item in dependencies if item in task_id_set
        )
        for dependency in dependencies:
            if dependency == task.task_id:
                errors.append(f"{task.task_id} depends on itself")
            elif dependency not in task_id_set and dependency not in goal_id_set:
                errors.append(
                    f"{task.task_id} has unknown dependency {dependency!r}"
                )
        if not task.outputs:
            errors.append(f"{task.task_id} has no outputs")
        errors.extend(
            f"{task.task_id}: {item}"
            for item in _safe_relative_paths(task.outputs, field="outputs")
        )
        if not task.validation:
            errors.append(f"{task.task_id} has no validation command")
        if not task.acceptance:
            errors.append(f"{task.task_id} has empty acceptance")
        if task.board_namespace != "ipfs-kit-vfs-symbolic-assurance-v1":
            errors.append(
                f"{task.task_id} has unexpected board namespace "
                f"{task.board_namespace!r}"
            )
        task_records.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "goal_id": goal_id,
                "depends_on": list(task.depends_on),
                "outputs": list(task.outputs),
                "acceptance": task.acceptance,
                "board_namespace": task.board_namespace,
                "canonical_task_cid": task.canonical_task_cid,
            }
        )
    task_cycles = _cycle_nodes(task_edges)
    if task_cycles:
        errors.append(f"task dependency cycle: {list(task_cycles)}")

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

    return {
        "schema": "ipfs_accelerate_py/vfs-symbolic-assurance-preflight@1",
        "valid": not errors,
        "errors": errors,
        "objective_path": str(objective_path),
        "objective_sha256": _sha256(objective_path),
        "goal_count": len(goals),
        "root_goal_ids": roots,
        "todo_path": str(todo_path),
        "todo_sha256": _sha256(todo_path),
        "task_count": len(tasks),
        "ready_task_count": sum(task.status == "todo" for task in tasks),
        "dependency_graph_id": _canonical_sha256(dependency_graph.to_dict()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-path", type=Path, default=OBJECTIVE_PATH)
    parser.add_argument("--todo-path", type=Path, default=TODO_PATH)
    args = parser.parse_args()
    result = validate(args.objective_path.resolve(), args.todo_path.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
