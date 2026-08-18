#!/usr/bin/env python3
"""Materialize and verify the LGCVF plan in the existing DuckDB control plane.

The canonical ``FormalWorkPlan@1`` supplies semantic identities and dependency
structure.  The reviewed Markdown board supplies human-facing work metadata.
Both projections must agree before this trusted bootstrap writes anything.
After bootstrap, ``DatabaseTaskSource@1`` and ``DatabaseImplementationDaemon@1``
own operational state; this script never writes task status back to Markdown.

The configured profile is deliberately one-writer embedded DuckDB.  It does
not claim Quack qualification and does not install or probe network services.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    read_coordination_registry_projection,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
    parse_todo_blocks,
    split_csv,
)

CONFIG_PATH = (
    ROOT / "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json"
)
SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-materialization@1"
VERIFICATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-read-only-verification@1"
POPULATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgcvf-population@1"
EXPECTED_NAMESPACE = "logic-governed-compositional-verification-fabric-v1"
EXPECTED_SCHEMA_REVISION = "datasets-authoritative-operational-v1"
EXPECTED_SCHEMA_PROFILE = "datasets-authoritative-operational"
SCHEMA_REVISION_ENV = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"


class MaterializationError(RuntimeError):
    """Raised when bootstrap input or an operational store fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _safe_path(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise MaterializationError(f"{field} escapes the repository root") from exc
    return resolved


def load_config(
    config_path: Path = CONFIG_PATH,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load the closed embedded profile without importing any provider."""

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"scheduler config is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise MaterializationError("scheduler config root must be an object")
    if payload.get("board_namespace") != EXPECTED_NAMESPACE:
        raise MaterializationError("unexpected LGCVF board namespace")
    program = payload.get("database_program")
    if not isinstance(program, dict):
        raise MaterializationError("database_program is required")
    expected = {
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "schema_revision": EXPECTED_SCHEMA_REVISION,
        "schema_profile": EXPECTED_SCHEMA_PROFILE,
        "semantic_relations_permitted": False,
        "failover_policy": "fail_closed",
    }
    for field, value in expected.items():
        if program.get(field) != value:
            raise MaterializationError(f"database_program.{field} must equal {value!r}")
    writer = payload.get("bootstrap_writer_policy")
    if not isinstance(writer, dict):
        raise MaterializationError("bootstrap_writer_policy is required")
    if (
        writer.get("maximum_processes") != 1
        or writer.get("direct_multi_process_duckdb_permitted") is not False
        or writer.get("automatic_installation_permitted") is not False
    ):
        raise MaterializationError("LGCVF bootstrap must remain one-writer and offline")
    if int(payload.get("max_lanes") or 0) != 1:
        raise MaterializationError("embedded LGCVF authority permits exactly one lane")
    for field in (
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "formal_plan_path",
        "validator_path",
        "materializer_path",
    ):
        path = _safe_path(root, payload.get(field), field=field)
        if not path.is_file():
            raise MaterializationError(f"required LGCVF source is absent: {field}")
    _safe_path(root, program.get("store_id"), field="database_program.store_id")
    return payload


def _git(root: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *argv],
        cwd=root,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise MaterializationError(f"git {' '.join(argv)} failed: {detail}")
    return completed.stdout.strip()


def verify_source_binding(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Bind bootstrap to one clean branch, tree, and exact datasets gitlink."""

    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise MaterializationError("refusing to materialize from a dirty execution worktree")
    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("source_binding must be an object")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    expected_branch = str(binding.get("accelerator_required_branch") or "")
    if branch != expected_branch:
        raise MaterializationError(
            f"accelerator branch differs: expected {expected_branch!r}, observed {branch!r}"
        )
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    ancestor = str(binding.get("accelerator_required_ancestor") or "")
    _git(root, "merge-base", "--is-ancestor", ancestor, head)

    datasets = _safe_path(
        root,
        binding.get("ipfs_datasets_submodule_path"),
        field="source_binding.ipfs_datasets_submodule_path",
    )
    if _git(datasets, "status", "--porcelain=v1", "--untracked-files=all"):
        raise MaterializationError("ipfs_datasets_py nested worktree is dirty")
    datasets_head = _git(datasets, "rev-parse", "HEAD")
    datasets_tree = _git(datasets, "rev-parse", "HEAD^{tree}")
    expected_datasets = str(binding.get("ipfs_datasets_planning_revision") or "")
    if datasets_head != expected_datasets:
        raise MaterializationError("ipfs_datasets_py HEAD differs from the configured revision")
    relative = datasets.relative_to(root).as_posix()
    gitlink = _git(root, "ls-tree", head, "--", relative).split()
    if (
        len(gitlink) < 3
        or gitlink[0] != "160000"
        or gitlink[1] != "commit"
        or gitlink[2] != datasets_head
    ):
        raise MaterializationError("ipfs_datasets_py is not the exact configured gitlink")
    report = {
        "accelerator_branch": branch,
        "accelerator_head": head,
        "accelerator_tree": tree,
        "accelerator_required_ancestor": ancestor,
        "datasets_gitlink": datasets_head,
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "datasets_path": relative,
        "nested_repository_count": 1,
        "worktrees_clean": True,
    }
    report["source_forest_root"] = content_identity(report)
    return report


def _metadata_bool(fields: Mapping[str, str], key: str) -> bool:
    value = str(fields.get(key) or "").strip().casefold()
    if value not in {"true", "false"}:
        raise MaterializationError(f"Markdown field {key!r} must be true or false")
    return value == "true"


def project_population(
    config: Mapping[str, Any],
    *,
    formal_plan: FormalWorkPlan,
    todo_text: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a checked DatabaseTaskSource population from both plan views."""

    namespace = str(formal_plan.metadata.get("board_namespace") or "")
    if namespace != EXPECTED_NAMESPACE:
        raise MaterializationError("formal plan namespace differs from the scheduler")
    plan_binding = config.get("plan_binding")
    if not isinstance(plan_binding, Mapping):
        raise MaterializationError("plan_binding must be an object")
    if formal_plan.content_id != str(plan_binding.get("formal_plan_content_id") or ""):
        raise MaterializationError("formal plan content identity differs from the scheduler")
    if formal_plan.metadata.get("predecessor_plan_cid") != plan_binding.get("predecessor_plan_cid"):
        raise MaterializationError("formal plan predecessor identity differs")

    blocks = parse_todo_blocks(todo_text, task_header_prefix="## LGCVF-")
    block_map = {task_id: (title, line, fields) for task_id, title, line, fields in blocks}
    formal_ids = tuple(task.task_id for task in formal_plan.tasks)
    if tuple(block_map) != formal_ids:
        raise MaterializationError("Markdown task order/identity differs from FormalWorkPlan")

    root_goal = formal_plan.goals[0]
    goal_cids = {root_goal.goal_id: root_goal.content_id}
    goal_records: list[dict[str, Any]] = [
        {
            "goal_cid": root_goal.content_id,
            "goal_id": root_goal.goal_id,
            "goal_alias": root_goal.goal_id,
            "title": str(root_goal.metadata.get("title") or root_goal.goal_id),
            "ordinal": 1,
            "status": "open",
            "objective_id": "objective:lgcvf-root",
            "objective_alias": root_goal.goal_id,
            "priority": "P0",
            "formal_content_id": root_goal.content_id,
            "formal_record": root_goal.to_dict(),
        }
    ]
    for ordinal, subgoal in enumerate(formal_plan.subgoals, start=2):
        goal_cids[subgoal.subgoal_id] = subgoal.content_id
        goal_records.append(
            {
                "goal_cid": subgoal.content_id,
                "goal_id": subgoal.subgoal_id,
                "goal_alias": subgoal.subgoal_id,
                "title": str(subgoal.metadata.get("title") or subgoal.subgoal_id),
                "ordinal": ordinal,
                "status": "open",
                "parent_goal_cid": root_goal.content_id,
                "priority": "P0",
                "formal_content_id": subgoal.content_id,
                "formal_record": subgoal.to_dict(),
            }
        )

    goal_edges: list[dict[str, str]] = []
    for subgoal in formal_plan.subgoals:
        goal_edges.append(
            {
                "parent_goal_cid": root_goal.content_id,
                "child_goal_cid": subgoal.content_id,
                "edge_kind": "goal_parent",
            }
        )
        for dependency in subgoal.depends_on:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": subgoal.content_id,
                    "edge_kind": "goal_dependency",
                }
            )

    task_cids = {task.task_id: task.content_id for task in formal_plan.tasks}
    tasks: list[dict[str, Any]] = []
    for ordinal, task in enumerate(formal_plan.tasks, start=1):
        title, source_line, fields = block_map[task.task_id]
        dependencies = tuple(split_csv(fields.get("depends_on", "")))
        if dependencies != task.depends_on:
            raise MaterializationError(f"{task.task_id}: dependency projections differ")
        if fields.get("goal_id") != task.goal_id or fields.get("subgoal_id") != task.subgoal_id:
            raise MaterializationError(f"{task.task_id}: goal projection differs")
        if fields.get("board_namespace") != EXPECTED_NAMESPACE:
            raise MaterializationError(f"{task.task_id}: board namespace differs")
        construction_status = str(task.metadata.get("construction_status") or "")
        markdown_status = str(fields.get("status") or "")
        if construction_status.startswith("blocked_"):
            if markdown_status != "blocked" or construction_status not in fields.get(
                "blocked_reason", ""
            ):
                raise MaterializationError(f"{task.task_id}: blocked disposition differs")
            durable_status = "blocked"
        elif markdown_status == construction_status and construction_status in {
            "completed",
            "todo",
        }:
            durable_status = construction_status
        else:
            raise MaterializationError(f"{task.task_id}: construction status differs")
        schedulable = _metadata_bool(fields, "is_schedulable")
        review_only = _metadata_bool(fields, "review_only")
        if durable_status == "todo" and (not schedulable or review_only):
            raise MaterializationError(f"{task.task_id}: runnable task policy differs")
        if durable_status != "todo" and schedulable:
            raise MaterializationError(f"{task.task_id}: non-runnable task is schedulable")
        if construction_status.startswith("blocked_") and not review_only:
            raise MaterializationError(f"{task.task_id}: protected blocker is not review-only")
        outputs = tuple(split_csv(fields.get("outputs", "")))
        if outputs != tuple(split_csv(fields.get("predicted_files", ""))):
            raise MaterializationError(f"{task.task_id}: outputs and predicted files differ")
        markdown_metadata = dict(sorted(fields.items()))
        tasks.append(
            {
                "task_cid": task.content_id,
                "task_id": task.task_id,
                "task_alias": task.task_id,
                "goal_cid": goal_cids[task.subgoal_id],
                "plan_cid": formal_plan.content_id,
                "objective_id": "objective:lgcvf-root",
                "ordinal": ordinal,
                "status": durable_status,
                "priority": fields.get("priority", "P0"),
                "title": title,
                "dependencies": [task_cids[item] for item in task.depends_on],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"formal_task_content_id": task.content_id, "path": path}
                        ),
                    }
                    for path in outputs
                ],
                "acceptance": [fields.get("acceptance", "")],
                "validations": [fields.get("validation", "")],
                "completion": fields.get("completion", "auto"),
                "review_only": review_only,
                "is_schedulable": schedulable,
                "blocked_reason": fields.get("blocked_reason", ""),
                "construction_status": construction_status,
                "formal_task_content_id": task.content_id,
                "formal_record": task.to_dict(),
                "markdown_metadata": markdown_metadata,
                "markdown_metadata_cid": content_identity(markdown_metadata),
                "source_line": source_line,
                "owning_repository": fields.get("owning_repository", ""),
                "board_namespace": EXPECTED_NAMESPACE,
            }
        )

    projection = config.get("initial_projection")
    if not isinstance(projection, Mapping):
        raise MaterializationError("initial_projection is required")
    if len(tasks) != projection.get("task_count") or len(goal_records) != projection.get(
        "goal_count"
    ):
        raise MaterializationError("population count differs from initial_projection")
    observed_completed = [item["task_id"] for item in tasks if item["status"] == "completed"]
    observed_blocked = [item["task_id"] for item in tasks if item["status"] == "blocked"]
    if observed_completed != projection.get("completed_task_ids"):
        raise MaterializationError("completed task projection differs")
    if observed_blocked != projection.get("blocked_task_ids"):
        raise MaterializationError("blocked task projection differs")

    population = {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": "git-tree:" + str(source["accelerator_tree"]),
        "source_head": str(source["accelerator_head"]),
        "source_forest_root": str(source["source_forest_root"]),
        "formal_repository_tree_id": formal_plan.repository_tree_id,
        "plan_root_cid": formal_plan.content_id,
        "objectives": goal_records,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": formal_plan.content_id,
                "plan_alias": EXPECTED_NAMESPACE,
                "goal_cid": root_goal.content_id,
                "status": "active",
                "repository_tree_id": "git-tree:" + str(source["accelerator_tree"]),
                "formal_repository_tree_id": formal_plan.repository_tree_id,
                "predecessor_plan_cid": formal_plan.metadata["predecessor_plan_cid"],
                "source_head": str(source["accelerator_head"]),
            }
        ],
        "tasks": tasks,
        "goal_cids_by_alias": goal_cids,
        "task_cids_by_alias": task_cids,
    }
    population["population_root"] = content_identity(population)
    return population


def build_population(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load and bind the exact local formal and Markdown projections."""

    source = verify_source_binding(config, root=root)
    formal_path = _safe_path(root, config.get("formal_plan_path"), field="formal_plan_path")
    todo_path = _safe_path(root, config.get("taskboard_path"), field="taskboard_path")
    try:
        formal_payload = json.loads(formal_path.read_text(encoding="utf-8"))
        formal_plan = FormalWorkPlan.from_dict(formal_payload)
        todo_text = todo_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MaterializationError(f"LGCVF plan projection is unreadable: {exc}") from exc
    return project_population(
        config,
        formal_plan=formal_plan,
        todo_text=todo_text,
        source=source,
    )


def _paths(config: Mapping[str, Any], *, root: Path) -> dict[str, Path]:
    program = config.get("database_program")
    if not isinstance(program, Mapping):
        raise MaterializationError("database_program is required")
    control = _safe_path(root, program.get("store_id"), field="database_program.store_id")
    runtime = config.get("runtime_paths")
    if not isinstance(runtime, Mapping):
        raise MaterializationError("runtime_paths is required")
    evidence = _safe_path(root, runtime.get("evidence"), field="runtime_paths.evidence")
    return {
        "control": control,
        "coordination": control.with_name(f"{control.stem}.coordination.duckdb"),
        "execution": control.with_name(f"{control.stem}.execution.duckdb"),
        "receipt": evidence / "bootstrap" / "materialization.json",
    }


def _read_only_control(
    path: Path,
    population: Mapping[str, Any],
    *,
    expected_stage: str,
) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise MaterializationError("DuckDB is unavailable; refusing materialization") from exc
    try:
        connection = duckdb.connect(
            str(path),
            read_only=True,
            config={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError("control store cannot be opened read-only") from exc
    try:
        relation_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        required = {"goals", "goal_edges", "plans", "tasks", "task_dependencies"}
        if not required.issubset(relation_names):
            raise MaterializationError("control store lacks required operational relations")
        forbidden = {
            "semantic_nodes",
            "semantic_edges",
            "semantic_state_roots",
            "proof_obligations",
            "semantic_capsules",
        }
        if relation_names & forbidden:
            raise MaterializationError("control store contains datasets semantic authority")
        task_rows = connection.execute(
            "SELECT task_cid, task_alias, status, body_json FROM tasks ORDER BY ordinal, task_cid"
        ).fetchall()
        expected_tasks = list(population["tasks"])
        if len(task_rows) != len(expected_tasks):
            raise MaterializationError("control task count differs from population")
        runtime_progress = False
        statuses: dict[str, str] = {}
        for row, expected in zip(task_rows, expected_tasks, strict=True):
            task_cid, task_alias, status, body_json = map(str, row)
            if task_cid != expected["task_cid"] or task_alias != expected["task_id"]:
                raise MaterializationError("control task identity/order differs")
            try:
                body = json.loads(body_json)
            except json.JSONDecodeError as exc:
                raise MaterializationError(f"{task_alias}: task body is invalid JSON") from exc
            for field in (
                "formal_task_content_id",
                "construction_status",
                "completion",
                "review_only",
                "blocked_reason",
                "board_namespace",
            ):
                if body.get(field) != expected.get(field):
                    raise MaterializationError(
                        f"{task_alias}: immutable task body differs at {field}"
                    )
            expected_status = str(expected["status"])
            if expected_stage == "initial":
                if status != expected_status:
                    raise MaterializationError(f"{task_alias}: initial status differs")
            elif expected_status == "blocked":
                if status != "blocked":
                    raise MaterializationError(f"{task_alias}: protected blocker was transitioned")
            elif status != expected_status:
                runtime_progress = True
            statuses[task_alias] = status
        dep_rows = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT task_cid, dependency_task_cid FROM task_dependencies"
            ).fetchall()
        }
        expected_deps = {
            (str(task["task_cid"]), str(dependency))
            for task in expected_tasks
            for dependency in task["dependencies"]
        }
        if dep_rows != expected_deps:
            raise MaterializationError("control dependency graph differs")
        goal_cids = {
            str(row[0]) for row in connection.execute("SELECT goal_cid FROM goals").fetchall()
        }
        if goal_cids != {str(item["goal_cid"]) for item in population["objectives"]}:
            raise MaterializationError("control goal identities differ")
        plan_rows = connection.execute("SELECT plan_cid FROM plans").fetchall()
        if [str(row[0]) for row in plan_rows] != [str(population["plan_root_cid"])]:
            raise MaterializationError("control plan identity differs")
        return {
            "task_count": len(task_rows),
            "goal_count": len(goal_cids),
            "dependency_count": len(dep_rows),
            "statuses": statuses,
            "runtime_progress_observed": runtime_progress,
            "relation_count": len(relation_names),
        }
    finally:
        connection.close()


def _read_only_execution(path: Path, *, expected_stage: str) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise MaterializationError("DuckDB is unavailable") from exc
    try:
        connection = duckdb.connect(
            str(path),
            read_only=True,
            config={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError("execution store cannot be opened read-only") from exc
    tables = (
        "database_task_attempts",
        "attempt_phases",
        "provider_invocations",
        "effect_claims",
        "daemon_execution_events",
    )
    try:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    except Exception as exc:
        raise MaterializationError("execution store lacks the daemon schema") from exc
    finally:
        connection.close()
    if expected_stage == "initial" and any(counts.values()):
        raise MaterializationError("initial execution store already contains attempts/effects")
    return {"row_counts": counts, "runtime_progress_observed": any(counts.values())}


def verify_read_only(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    expected_stage: str = "live",
) -> dict[str, Any]:
    """Reconstruct immutable authority without creating locks or writing stores."""

    if expected_stage not in {"initial", "live"}:
        raise MaterializationError("expected_stage must be 'initial' or 'live'")
    paths = _paths(config, root=root)
    stores = {key: path for key, path in paths.items() if key != "receipt"}
    missing = [key for key, path in stores.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"operational stores are absent: {', '.join(missing)}")
    before = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    control = _read_only_control(paths["control"], population, expected_stage=expected_stage)
    try:
        coordination = read_coordination_registry_projection(paths["coordination"])
    except Exception as exc:
        raise MaterializationError("coordination registry fails read-only verification") from exc
    expected_registry = {
        str(item["task_cid"]): str(item["task_id"]) for item in population["tasks"]
    }
    registered = {
        str(item["task_cid"]): str(item["task_id"]) for item in coordination.get("tasks", ())
    }
    # The typed coordination projection is deliberately ordered by durable
    # CID, while the formal population is ordered by task alias/ordinal.
    if registered != expected_registry:
        raise MaterializationError("coordination task identities differ")
    expected_edges = {
        (str(task["task_cid"]), str(dep))
        for task in population["tasks"]
        for dep in task["dependencies"]
    }
    coordination_edges = {
        (str(item["task_cid"]), str(item["dependency_task_cid"]))
        for item in coordination.get("dependency_edges", ())
    }
    if coordination_edges != expected_edges:
        raise MaterializationError("coordination dependency graph differs")
    execution = _read_only_execution(paths["execution"], expected_stage=expected_stage)
    after = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    if before != after:
        raise MaterializationError("read-only verification changed an operational store")
    report = {
        "schema": VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only",
        "expected_stage": expected_stage,
        "population_root": population["population_root"],
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "control": control,
        "coordination": {
            "counts": coordination["counts"],
            "projection_root": coordination["projection_root"],
        },
        "execution": execution,
        "stores_unchanged": True,
        "maximum_writer_processes": 1,
        "quack_qualified": False,
    }
    report["verification_root"] = content_identity(report)
    return report


def materialize(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    recheck_source: bool = True,
) -> dict[str, Any]:
    """Install and populate one fresh embedded operational control plane."""

    paths = _paths(config, root=root)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        names = ", ".join(path.relative_to(root).as_posix() for path in existing)
        raise MaterializationError(f"refusing to overwrite an existing control plane: {names}")
    if recheck_source:
        current = verify_source_binding(config, root=root)
        if (
            current["accelerator_head"] != population.get("source_head")
            or "git-tree:" + current["accelerator_tree"] != population.get("repository_tree_id")
            or current["source_forest_root"] != population.get("source_forest_root")
        ):
            raise MaterializationError("source forest changed after population construction")

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        duckdb_version = importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        duckdb_version = "unavailable"
    schema_install = install_datasets_authoritative_operational_schema(
        paths["control"],
        application_version="lgcvf-v1",
        tool_version=duckdb_version,
        owner_id="lgcvf-materializer:operational-schema",
    )
    prior_revision = os.environ.get(SCHEMA_REVISION_ENV)
    os.environ[SCHEMA_REVISION_ENV] = EXPECTED_SCHEMA_REVISION
    daemon: DatabaseImplementationDaemon | None = None
    try:
        daemon = DatabaseImplementationDaemon(
            database_path=paths["control"],
            coordination_path=paths["coordination"],
            execution_path=paths["execution"],
            owner_session_id="lgcvf-materializer:single-writer",
            authority_mode="embedded",
            task_source_kind="duckdb",
            install_schema=False,
        )
        database_receipt = daemon.materialize_population(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
    finally:
        if daemon is not None:
            daemon.close()
        if prior_revision is None:
            os.environ.pop(SCHEMA_REVISION_ENV, None)
        else:
            os.environ[SCHEMA_REVISION_ENV] = prior_revision

    task_source_receipt = database_receipt.get("task_source")
    expected_task_cids = [str(item["task_cid"]) for item in population["tasks"]]
    if not isinstance(task_source_receipt, Mapping):
        raise MaterializationError("DatabaseTaskSource did not return a typed receipt")
    expected_counts = {
        "task_count": len(expected_task_cids),
        "goal_count": len(population["objectives"]),
        "goal_edge_count": len(population["goal_edges"]),
        "plan_count": 1,
        "task_cids": expected_task_cids,
    }
    if any(task_source_receipt.get(key) != value for key, value in expected_counts.items()):
        raise MaterializationError("DatabaseTaskSource receipt differs from the exact population")
    if list(database_receipt.get("registered_task_cids") or ()) != expected_task_cids:
        raise MaterializationError("DatabaseImplementationDaemon registration differs")
    expected_completed_cids = [
        str(item["task_cid"])
        for item in population["tasks"]
        if str(item.get("status") or "").strip().lower()
        in {"completed", "complete", "done"}
    ]
    if (
        list(database_receipt.get("bootstrap_completed_task_cids") or ())
        != expected_completed_cids
    ):
        raise MaterializationError(
            "DatabaseImplementationDaemon completion projection differs"
        )
    verification = verify_read_only(
        config,
        population,
        root=root,
        expected_stage="initial",
    )
    receipt = {
        "schema": SCHEMA,
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "maximum_writer_processes": 1,
        "quack_qualified": False,
        "schema_revision": EXPECTED_SCHEMA_REVISION,
        "schema_profile": EXPECTED_SCHEMA_PROFILE,
        "semantic_truth_authority": "ipfs_datasets_py",
        "operational_coordination_authority": "ipfs_accelerate_py",
        "population_root": population["population_root"],
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "source_head": population["source_head"],
        "database_paths": {
            key: path.relative_to(root).as_posix()
            for key, path in sorted(paths.items())
            if key != "receipt"
        },
        "schema_install": (
            schema_install.to_dict()
            if callable(getattr(schema_install, "to_dict", None))
            else dict(schema_install)
        ),
        "materialization": dict(database_receipt),
        "verification": verification,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    paths["receipt"].write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("materialization receipt is absent or unreadable") from exc
    if not isinstance(receipt, dict):
        raise MaterializationError("materialization receipt must be an object")
    claimed = str(receipt.pop("receipt_cid", ""))
    observed = content_identity(receipt)
    receipt["receipt_cid"] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError("materialization receipt content identity does not verify")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("population", "materialize", "verify"),
        help="render, materialize, or read-only verify the exact LGCVF population",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config()
        population = build_population(config)
        if args.command == "population":
            result: Mapping[str, Any] = population
        elif args.command == "materialize":
            result = materialize(config, population)
        else:
            paths = _paths(config, root=ROOT)
            receipt = _load_receipt(paths["receipt"])
            if receipt.get("population_root") != population["population_root"]:
                raise MaterializationError("materialization receipt is stale for this population")
            result = verify_read_only(config, population, expected_stage="live")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MaterializationError as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "valid": False, "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
