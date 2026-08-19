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
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanWorktreeContract,
    PopulationKind,
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
SUCCESSOR_PREVIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-preview@1"
)
SUCCESSOR_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-apply-receipt@1"
)
SUCCESSOR_VERIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-read-only-verification@1"
)
SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-composite-projection@1"
)
SUCCESSOR_RECOVERY_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-recovery-manifest@1"
)
EXPECTED_NAMESPACE = "logic-governed-compositional-verification-fabric-v1"
EXPECTED_SCHEMA_REVISION = "datasets-authoritative-operational-v1"
EXPECTED_SCHEMA_PROFILE = "datasets-authoritative-operational"
SCHEMA_REVISION_ENV = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
SUCCESSOR_ADDED_ALIAS = "LGCVF-113"
SUCCESSOR_AMENDED_ALIASES = frozenset(
    {
        "LGCVF-081",
        "LGCVF-111",
        "LGCVF-112",
        "LGCVF-120",
        "LGCVF-122",
        "LGCVF-124",
    }
)
SUCCESSOR_REPRIORITIZED_ALIASES = frozenset({"LGCVF-121", "LGCVF-123"})
SUCCESSOR_CHANGED_ALIASES = (
    SUCCESSOR_AMENDED_ALIASES | SUCCESSOR_REPRIORITIZED_ALIASES
)
SUCCESSOR_RUNTIME_DEPENDENCIES = {"LGCVF-120": (SUCCESSOR_ADDED_ALIAS,)}


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
    logical_task_prefix = formal_plan.metadata.get("task_prefix")
    scheduler_task_prefix = config.get("task_prefix")
    if (
        not isinstance(logical_task_prefix, str)
        or not isinstance(scheduler_task_prefix, str)
        or scheduler_task_prefix != "## " + logical_task_prefix
    ):
        raise MaterializationError(
            "scheduler Markdown task selector differs from the formal logical prefix"
        )
    plan_binding = config.get("plan_binding")
    if not isinstance(plan_binding, Mapping):
        raise MaterializationError("plan_binding must be an object")
    if formal_plan.content_id != str(plan_binding.get("formal_plan_content_id") or ""):
        raise MaterializationError("formal plan content identity differs from the scheduler")
    if formal_plan.metadata.get("predecessor_plan_cid") != plan_binding.get("predecessor_plan_cid"):
        raise MaterializationError("formal plan predecessor identity differs")

    blocks = parse_todo_blocks(todo_text, task_header_prefix=scheduler_task_prefix)
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


def _successor_paths(
    config: Mapping[str, Any], *, root: Path
) -> dict[str, Path]:
    """Resolve revision-store paths without changing the bootstrap contract."""

    paths = _paths(config, root=root)
    runtime = config.get("runtime_paths")
    if not isinstance(runtime, Mapping):
        raise MaterializationError("runtime_paths is required")
    state = _safe_path(root, runtime.get("state"), field="runtime_paths.state")
    evidence = _safe_path(
        root, runtime.get("evidence"), field="runtime_paths.evidence"
    )
    binding = config.get("plan_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("plan_binding is required")
    predecessor = str(binding.get("predecessor_plan_cid") or "")
    if not predecessor:
        raise MaterializationError("plan_binding.predecessor_plan_cid is required")
    formal_path = _safe_path(
        root, config.get("formal_plan_path"), field="formal_plan_path"
    )
    paths.update(
        {
            "revision_store": state / "plan-revision-store",
            "revision_receipts": evidence / "plan-revisions",
            "predecessor_archive": (
                formal_path.parent / "plan_revisions" / f"{predecessor}.json"
            ),
        }
    )
    return paths


def _plain_json(value: Any) -> Any:
    """Return a detached canonical JSON value."""

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return [thaw(child) for child in item]
        return item

    return json.loads(_canonical_bytes(thaw(value)))


def _load_predecessor_plan(
    config: Mapping[str, Any], *, root: Path
) -> FormalWorkPlan:
    binding = config.get("plan_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("plan_binding is required")
    predecessor_cid = str(binding.get("predecessor_plan_cid") or "")
    if int(binding.get("plan_revision") or 0) != 2:
        raise MaterializationError("successor continuation requires plan_revision 2")
    archive = _successor_paths(config, root=root)["predecessor_archive"]
    try:
        payload = json.loads(archive.read_text(encoding="utf-8"))
        predecessor = FormalWorkPlan.from_dict(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MaterializationError("immutable predecessor archive is unreadable") from exc
    if predecessor.content_id != predecessor_cid:
        raise MaterializationError("immutable predecessor archive identity differs")
    if predecessor.metadata.get("board_namespace") != EXPECTED_NAMESPACE:
        raise MaterializationError("predecessor archive namespace differs")
    return predecessor


def _task_map(
    projection: Mapping[str, Any], *, noun: str
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw = projection.get("tasks")
    if not isinstance(raw, list):
        raise MaterializationError(f"{noun} has no typed task population")
    by_cid: dict[str, Mapping[str, Any]] = {}
    by_alias: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise MaterializationError(f"{noun} contains a malformed task")
        cid = str(item.get("task_cid") or "")
        alias = str(item.get("task_alias") or item.get("task_id") or "")
        if not cid or not alias or cid in by_cid or alias in by_alias:
            raise MaterializationError(f"{noun} task identity is missing or duplicated")
        by_cid[cid] = item
        by_alias[alias] = item
    return by_cid, by_alias


def _task_lifecycle(status: Any) -> LifecycleState:
    normalized = str(status or "").strip().lower()
    if normalized in {"todo", "pending", "queued", "retrying"}:
        return LifecycleState.UNSTARTED
    if normalized == "ready":
        return LifecycleState.READY
    if normalized == "blocked":
        return LifecycleState.BLOCKED
    if normalized == "claimed":
        return LifecycleState.CLAIMED
    if normalized in {"in_progress", "running"}:
        return LifecycleState.RUNNING
    if normalized in {"completed", "complete", "done", "skipped"}:
        return LifecycleState.COMPLETED
    if normalized in {"failed", "cancelled", "quarantined", "rejected"}:
        return LifecycleState.FAILED
    raise MaterializationError(f"task status {normalized!r} has no revision lifecycle")


def _raw_task_from_projection(task: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly adapt an IntentRepository task back to materializer input."""

    body = task.get("body")
    identity = task.get("identity")
    if not isinstance(body, Mapping) or not isinstance(identity, Mapping):
        raise MaterializationError("live task lacks typed body or identity")
    forbidden_body = {
        "task_cid",
        "task_id",
        "task_alias",
        "goal_cid",
        "dependencies",
        "outputs",
        "acceptance",
        "validations",
        "status",
        "priority",
        "ordinal",
        "plan_cid",
        "objective_id",
    }
    if forbidden_body & set(body):
        raise MaterializationError("live task body collides with projection fields")
    dependencies: list[str] = []
    for dependency in task.get("dependencies") or ():
        if not isinstance(dependency, Mapping):
            raise MaterializationError("live task dependency is malformed")
        if str(dependency.get("kind") or "depends_on") != "depends_on":
            raise MaterializationError("non-default dependency kind is not losslessly adaptable")
        dependency_cid = str(dependency.get("dependency_task_cid") or "")
        if not dependency_cid:
            raise MaterializationError("live task dependency identity is empty")
        dependencies.append(dependency_cid)
    outputs: list[dict[str, Any]] = []
    for output in task.get("outputs") or ():
        if not isinstance(output, Mapping) or not isinstance(output.get("effect"), Mapping):
            raise MaterializationError("live task output is malformed")
        effect = _plain_json(output["effect"])
        if effect.get("path") != output.get("path"):
            raise MaterializationError("live output effect/path projection differs")
        outputs.append(effect)
    acceptance: list[dict[str, Any]] = []
    for entry in task.get("acceptance") or ():
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("evidence_policy"), Mapping
        ):
            raise MaterializationError("live task acceptance is malformed")
        policy = _plain_json(entry["evidence_policy"])
        if policy.get("criterion") != entry.get("criterion"):
            raise MaterializationError("acceptance policy cannot be losslessly adapted")
        acceptance.append(policy)
    validations: list[dict[str, Any]] = []
    for entry in task.get("validations") or ():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("policy"), Mapping):
            raise MaterializationError("live task validation is malformed")
        argv = entry.get("argv")
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            raise MaterializationError("live validation argv is malformed")
        validations.append({"argv": list(argv), **_plain_json(entry["policy"])})
    expected_identity = {
        "task_cid": str(task.get("task_cid") or ""),
        "task_alias": str(task.get("task_alias") or ""),
        "repository_tree_id": str(identity.get("repository_tree_id") or ""),
    }
    if dict(identity) != expected_identity or not expected_identity["repository_tree_id"]:
        raise MaterializationError("live task identity is not the canonical LGCVF shape")
    return {
        "task_cid": expected_identity["task_cid"],
        "task_id": expected_identity["task_alias"],
        "task_alias": expected_identity["task_alias"],
        "goal_cid": str(task.get("goal_cid") or ""),
        "plan_cid": str(task.get("plan_cid") or ""),
        "objective_id": str(task.get("objective_id") or ""),
        "ordinal": int(task.get("ordinal") or 0),
        "status": str(task.get("status") or ""),
        "priority": str(task.get("priority") or ""),
        "dependencies": dependencies,
        "outputs": outputs,
        "acceptance": acceptance,
        "validations": validations,
        **_plain_json(body),
    }


def _read_successor_state(
    config: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Read all continuation evidence and prove that the read changed no store."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    paths = _successor_paths(config, root=root)
    stores = {key: paths[key] for key in ("control", "coordination", "execution")}
    missing = [key for key, path in stores.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"operational stores are absent: {', '.join(missing)}")
    before = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgcvf-successor:read-only-observer",
        install_schema=False,
    )
    try:
        plan_projection = _plain_json(source.plan_projection())
        completion_projection = _plain_json(source.completion_evidence_projection())
    finally:
        source.close()
    try:
        coordination = read_coordination_registry_projection(paths["coordination"])
    except Exception as exc:
        raise MaterializationError("coordination registry fails typed verification") from exc
    execution = _read_only_execution(paths["execution"], expected_stage="live")
    bootstrap = _load_receipt(paths["receipt"])
    after = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    if before != after:
        raise MaterializationError("successor observation changed an operational store")
    composite = {
        "schema": SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA,
        "plan_projection_cid": str(plan_projection.get("projection_cid") or ""),
        "completion_projection_cid": str(
            completion_projection.get("projection_cid") or ""
        ),
        "coordination_projection_root": str(coordination.get("projection_root") or ""),
    }
    composite["projection_cid"] = content_identity(composite)
    return {
        "plan_projection": plan_projection,
        "completion_projection": completion_projection,
        "coordination_projection": coordination,
        "execution_projection": execution,
        "bootstrap_receipt": bootstrap,
        "store_observations": {
            key: {"size": value[0], "mtime_ns": value[1], "sha256": value[2]}
            for key, value in before.items()
        },
        "composite_projection": composite,
    }


def _retained_completion_binding(
    state: Mapping[str, Any], completed_cids: Sequence[str]
) -> dict[str, Any]:
    completed = set(completed_cids)
    completion = state.get("completion_projection")
    coordination = state.get("coordination_projection")
    if not isinstance(completion, Mapping) or not isinstance(coordination, Mapping):
        raise MaterializationError("completion binding projections are malformed")
    binding = {
        "task_states": sorted(
            [
                _plain_json(item)
                for item in completion.get("task_states") or ()
                if isinstance(item, Mapping) and item.get("task_cid") in completed
            ],
            key=lambda item: str(item.get("task_cid") or ""),
        ),
        "completion_receipts": sorted(
            [
                _plain_json(item)
                for item in completion.get("completion_receipts") or ()
                if isinstance(item, Mapping) and item.get("task_cid") in completed
            ],
            key=lambda item: (
                str(item.get("task_cid") or ""),
                str(item.get("receipt_cid") or ""),
            ),
        ),
        "logical_completions": sorted(
            [
                _plain_json(item)
                for item in coordination.get("logical_completions") or ()
                if isinstance(item, Mapping) and item.get("task_cid") in completed
            ],
            key=lambda item: str(item.get("task_cid") or ""),
        ),
    }
    if {str(item.get("task_cid") or "") for item in binding["task_states"]} != completed:
        raise MaterializationError("retained completion states are incomplete")
    if {
        str(item.get("task_cid") or "") for item in binding["logical_completions"]
    } != completed:
        raise MaterializationError("retained logical completions are incomplete")
    binding["binding_cid"] = content_identity(binding)
    return binding


def _protected_blocker_binding(
    task_by_alias: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for alias in sorted(SUCCESSOR_REPRIORITIZED_ALIASES):
        task = task_by_alias.get(alias)
        if not isinstance(task, Mapping):
            raise MaterializationError(f"{alias}: protected blocker is absent")
        tasks[alias] = {
            key: _plain_json(value)
            for key, value in task.items()
            if key not in {"ordinal", "revision", "plan_cid", "spec_cid"}
        }
    binding = {"tasks": tasks}
    binding["binding_cid"] = content_identity(binding)
    return binding


def _assert_quiescent_predecessor(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    predecessor: FormalWorkPlan,
    root: Path,
) -> dict[str, Any]:
    """Validate the exact revision-1 population and all operational fences."""

    plan = state["plan_projection"]
    completion = state["completion_projection"]
    coordination = state["coordination_projection"]
    if not all(isinstance(item, Mapping) for item in (plan, completion, coordination)):
        raise MaterializationError("successor state projections are malformed")
    binding = config.get("plan_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("plan_binding is required")
    predecessor_cid = str(binding.get("predecessor_plan_cid") or "")
    active_plans = [
        item
        for item in plan.get("plans") or ()
        if isinstance(item, Mapping) and item.get("status") == "active"
    ]
    if len(active_plans) != 1 or active_plans[0].get("plan_cid") != predecessor_cid:
        raise MaterializationError("control store is not at the exact active predecessor")
    if state["bootstrap_receipt"].get("plan_root_cid") != predecessor_cid:
        raise MaterializationError("bootstrap receipt does not bind the predecessor plan")

    live_by_cid, live_by_alias = _task_map(plan, noun="live predecessor projection")
    expected_by_alias = {task.task_id: task.content_id for task in predecessor.tasks}
    observed_by_alias = {
        alias: str(task.get("task_cid") or "") for alias, task in live_by_alias.items()
    }
    if observed_by_alias != expected_by_alias:
        raise MaterializationError("live predecessor logical task population differs")
    if SUCCESSOR_ADDED_ALIAS in live_by_alias:
        raise MaterializationError("successor task already exists under the predecessor")

    counts = coordination.get("counts")
    if not isinstance(counts, Mapping):
        raise MaterializationError("coordination counts are absent")
    active_fields = (
        "active_task_claims",
        "active_task_attempts",
        "active_fenced_leases",
        "active_resource_claims",
        "active_maintenance_leases",
    )
    active = {field: int(counts.get(field) or 0) for field in active_fields}
    if any(active.values()):
        raise MaterializationError(
            "active claims, attempts, leases, or writer reservations block successor apply"
        )
    prepared = [
        item
        for item in coordination.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") != "succeeded"
    ]
    if prepared:
        raise MaterializationError("prepared or non-success completion blocks successor apply")

    registered = {
        str(item.get("task_cid") or ""): str(item.get("task_id") or "")
        for item in coordination.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    expected_registered = {
        cid: alias for alias, cid in expected_by_alias.items()
    }
    if registered != expected_registered:
        raise MaterializationError("coordination predecessor task registry differs")
    expected_edges = {
        (cid, str(dependency.get("dependency_task_cid") or ""))
        for cid, task in live_by_cid.items()
        for dependency in task.get("dependencies") or ()
        if isinstance(dependency, Mapping)
    }
    observed_edges = {
        (str(item.get("task_cid") or ""), str(item.get("dependency_task_cid") or ""))
        for item in coordination.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    if observed_edges != expected_edges:
        raise MaterializationError("coordination predecessor dependency graph differs")

    task_states = {
        str(item.get("task_cid") or ""): item
        for item in completion.get("task_states") or ()
        if isinstance(item, Mapping)
    }
    if set(task_states) != set(live_by_cid):
        raise MaterializationError("completion evidence task population differs")
    for cid, task in live_by_cid.items():
        if task_states[cid].get("status") != task.get("status"):
            raise MaterializationError("control and completion task states differ")
    completed_cids = {
        cid
        for cid, task in live_by_cid.items()
        if _task_lifecycle(task.get("status")) is LifecycleState.COMPLETED
    }
    coordination_completed = {
        str(item.get("task_cid") or "")
        for item in coordination.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") == "succeeded"
    }
    if coordination_completed != completed_cids:
        raise MaterializationError("completion evidence disagrees across control stores")
    receipt_task_cids = {
        str(item.get("task_cid") or "")
        for item in completion.get("completion_receipts") or ()
        if isinstance(item, Mapping)
    }
    live_alias_by_cid = {
        cid: str(task.get("task_alias") or "") for cid, task in live_by_cid.items()
    }
    live_revision_by_cid = {
        cid: int(task.get("revision") or 0) for cid, task in live_by_cid.items()
    }
    for logical in coordination.get("logical_completions") or ():
        if not isinstance(logical, Mapping):
            raise MaterializationError("logical completion record is malformed")
        cid = str(logical.get("task_cid") or "")
        body = logical.get("body")
        if cid not in receipt_task_cids and (
            not isinstance(body, Mapping)
            or body
            != {
                "authority": "database_population",
                "source_status": "completed",
                "task_alias": live_alias_by_cid.get(cid),
                "task_revision": live_revision_by_cid.get(cid),
            }
        ):
            raise MaterializationError("bootstrap completion evidence is stale or rewritten")
    for receipt in completion.get("completion_receipts") or ():
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("task_cid") not in completed_cids
            or not receipt.get("receipt_cid")
            or not receipt.get("evidence_digest")
        ):
            raise MaterializationError("runtime completion receipt is stale or malformed")

    for alias, authority, completion_mode in (
        ("LGCVF-121", "blocked_external_authority", "external-authority"),
        ("LGCVF-123", "blocked_manual", "manual"),
    ):
        task = live_by_alias.get(alias)
        body = task.get("body") if isinstance(task, Mapping) else None
        if (
            not isinstance(task, Mapping)
            or task.get("status") != "blocked"
            or not isinstance(body, Mapping)
            or body.get("construction_status") != authority
            or body.get("completion") != completion_mode
            or body.get("review_only") is not True
        ):
            raise MaterializationError(f"{alias}: protected blocker semantics differ")
    tree_ids = {
        str(task.get("identity", {}).get("repository_tree_id") or "")
        for task in live_by_cid.values()
        if isinstance(task.get("identity"), Mapping)
    }
    if len(tree_ids) != 1 or not next(iter(tree_ids), ""):
        raise MaterializationError("predecessor task identities have inconsistent tree roots")
    evidence = {
        "predecessor_plan_cid": predecessor.content_id,
        "predecessor_archive_sha256": _sha256_file(
            _successor_paths(config, root=root)["predecessor_archive"]
        ),
        "plan_projection_cid": str(plan.get("projection_cid") or ""),
        "completion_projection_cid": str(completion.get("projection_cid") or ""),
        "coordination_projection_root": str(coordination.get("projection_root") or ""),
        "bootstrap_receipt_cid": str(
            state["bootstrap_receipt"].get("receipt_cid") or ""
        ),
        "completed_task_cids": sorted(completed_cids),
        "blocked_task_cids": sorted(
            cid
            for cid, task in live_by_cid.items()
            if _task_lifecycle(task.get("status")) is LifecycleState.BLOCKED
        ),
        "task_spec_cids": {
            alias: str(task.get("spec_cid") or "")
            for alias, task in sorted(live_by_alias.items())
        },
        "base_repository_tree_id": next(iter(tree_ids)),
        "active_counts": active,
        "retained_completion_binding": _retained_completion_binding(
            state, sorted(completed_cids)
        ),
        "protected_blocker_binding": _protected_blocker_binding(live_by_alias),
    }
    evidence["evidence_root"] = content_identity(evidence)
    return evidence


def _successor_candidate_population(
    population: Mapping[str, Any],
    live_plan: Mapping[str, Any],
    *,
    predecessor: FormalWorkPlan,
    base_repository_tree_id: str,
) -> dict[str, Any]:
    """Build the narrow operational revision without rewriting narrative history."""

    _live_by_cid, live_by_alias = _task_map(
        live_plan, noun="live predecessor projection"
    )
    desired_by_alias = {
        str(task.get("task_id") or ""): task for task in population.get("tasks") or ()
    }
    desired_alias_by_cid = {
        str(task.get("task_cid") or ""): alias
        for alias, task in desired_by_alias.items()
    }
    predecessor_aliases = {task.task_id for task in predecessor.tasks}
    expected_aliases = predecessor_aliases | {SUCCESSOR_ADDED_ALIAS}
    if set(desired_by_alias) != expected_aliases:
        raise MaterializationError("revision-2 desired task population differs")
    candidate_tasks: list[dict[str, Any]] = []
    for desired in population.get("tasks") or ():
        if not isinstance(desired, Mapping):
            raise MaterializationError("revision-2 task projection is malformed")
        alias = str(desired.get("task_id") or "")
        if alias == SUCCESSOR_ADDED_ALIAS:
            candidate = _plain_json(desired)
        elif alias in SUCCESSOR_AMENDED_ALIASES:
            live = live_by_alias[alias]
            lifecycle = _task_lifecycle(live.get("status"))
            if lifecycle not in {
                LifecycleState.UNSTARTED,
                LifecycleState.READY,
            }:
                raise MaterializationError(f"{alias}: started history cannot be amended")
            candidate = _plain_json(desired)
            candidate["status"] = str(live.get("status") or "")
        elif alias in SUCCESSOR_REPRIORITIZED_ALIASES:
            live = live_by_alias[alias]
            if _task_lifecycle(live.get("status")) is not LifecycleState.BLOCKED:
                raise MaterializationError(f"{alias}: protected blocker is not blocked")
            candidate = _raw_task_from_projection(live)
            candidate["ordinal"] = int(desired.get("ordinal") or 0)
        else:
            candidate = _raw_task_from_projection(live_by_alias[alias])
        if alias in live_by_alias:
            candidate["task_cid"] = str(live_by_alias[alias]["task_cid"])
            candidate["task_id"] = alias
            candidate["task_alias"] = alias
        translated_dependencies: list[str] = []
        for dependency_cid in candidate.get("dependencies") or ():
            dependency_alias = desired_alias_by_cid.get(str(dependency_cid))
            if dependency_alias is None:
                translated_dependencies.append(str(dependency_cid))
            elif dependency_alias in live_by_alias:
                translated_dependencies.append(
                    str(live_by_alias[dependency_alias]["task_cid"])
                )
            else:
                translated_dependencies.append(str(dependency_cid))
        for dependency_alias in SUCCESSOR_RUNTIME_DEPENDENCIES.get(alias, ()):
            dependency = desired_by_alias.get(dependency_alias)
            if not isinstance(dependency, Mapping):
                raise MaterializationError(
                    f"{alias}: runtime dependency {dependency_alias} is absent"
                )
            dependency_cid = str(dependency.get("task_cid") or "")
            if not dependency_cid:
                raise MaterializationError(
                    f"{alias}: runtime dependency {dependency_alias} has no identity"
                )
            if dependency_cid not in translated_dependencies:
                translated_dependencies.append(dependency_cid)
        candidate["dependencies"] = translated_dependencies
        candidate_tasks.append(candidate)
    candidate_population = _plain_json(population)
    candidate_population["repository_tree_id"] = base_repository_tree_id
    candidate_population["tasks"] = candidate_tasks
    candidate_population["task_cids_by_alias"] = {
        str(task["task_id"]): str(task["task_cid"]) for task in candidate_tasks
    }
    candidate_population.pop("population_root", None)
    candidate_population["population_root"] = content_identity(candidate_population)
    return candidate_population


def _project_candidate(
    candidate_population: Mapping[str, Any],
    *,
    base_repository_tree_id: str,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(prefix="lgcvf-successor-preview-") as temporary:
        source = DatabaseTaskSource(Path(temporary) / "candidate.duckdb")
        try:
            source.materialize(
                candidate_population,
                repository_tree_id=base_repository_tree_id,
                plan_root_cid=str(candidate_population.get("plan_root_cid") or ""),
            )
            return _plain_json(source.plan_projection())
        finally:
            source.close()


def _population_digest(
    kind: PopulationKind, members: Sequence[str] = ()
) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=tuple(members))


def _revision_roots(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    program_root: str,
) -> PlanAuthorityRoots:
    provider = config.get("provider")
    if not isinstance(provider, Mapping):
        provider = {}
    return PlanAuthorityRoots(
        repository_id="repository:ipfs-accelerate-py",
        repository_root_cid=str(population.get("repository_tree_id") or ""),
        dirty_worktree_root=str(population.get("source_forest_root") or ""),
        task_source_id=content_identity(
            {
                "interface": "DatabaseTaskSource@1",
                "board_namespace": EXPECTED_NAMESPACE,
            }
        ),
        task_source_revision=str(
            state["composite_projection"].get("projection_cid") or ""
        ),
        policy_root=content_identity(
            {
                "protected_paths": list(config.get("protected_paths") or ()),
                "writer_policy": config.get("bootstrap_writer_policy"),
            }
        ),
        intent_ir_root=str(state["plan_projection"].get("projection_cid") or ""),
        legal_ir_root=content_identity(
            {"authority": "datasets", "status": "not_extended_by_successor"}
        ),
        security_ir_root=content_identity(
            {"authority": "datasets", "status": "not_extended_by_successor"}
        ),
        program_root=program_root,
        capability_catalog_root=content_identity(
            {
                "schema_revision": EXPECTED_SCHEMA_REVISION,
                "schema_profile": EXPECTED_SCHEMA_PROFILE,
                "quack_qualified": False,
            }
        ),
        provider_catalog_root=content_identity(
            {
                "provider": provider.get("name"),
                "route": provider.get("route"),
            }
        ),
        usage_policy_root=content_identity(
            {
                "maximum_writer_processes": 1,
                "automatic_installation_permitted": False,
                "production_authorized": False,
            }
        ),
        configuration_root=content_identity(_plain_json(config)),
    )


def _revision_contract(
    *,
    plan_root_cid: str,
    semantic_revision: int,
    parent_plan_root: str,
    origin: PlanOrigin,
    roots: PlanAuthorityRoots,
    request_cid: str,
    delta_cid: str,
    evidence_root: str,
    admission_receipt_cid: str,
    goal_cids: Sequence[str],
    task_cids: Sequence[str],
    added_cids: Sequence[str],
    retained_cids: Sequence[str],
    claimed_cids: Sequence[str],
    completed_cids: Sequence[str],
    blocked_cids: Sequence[str],
    control_path: str,
    coordination_path: str,
) -> PlanRevision:
    return PlanRevision(
        plan_root_cid=plan_root_cid,
        semantic_revision=semantic_revision,
        parent_plan_root=parent_plan_root,
        origin=origin,
        roots=roots,
        request_cid=request_cid,
        delta_cid=delta_cid,
        scan_receipt_cid=evidence_root,
        query_plan_cid="",
        evidence_bundle_cid=evidence_root,
        admission_receipt_cid=admission_receipt_cid,
        execution_plan_cid="",
        goal_population=_population_digest(PopulationKind.RETAINED, goal_cids),
        task_population=_population_digest(PopulationKind.RETAINED, task_cids),
        added_population=_population_digest(PopulationKind.ADDED, added_cids),
        superseded_population=_population_digest(PopulationKind.SUPERSEDED),
        retained_population=_population_digest(PopulationKind.RETAINED, retained_cids),
        deferred_population=_population_digest(PopulationKind.DEFERRED),
        claimed_population=_population_digest(PopulationKind.CLAIMED, claimed_cids),
        completed_population=_population_digest(
            PopulationKind.COMPLETED, completed_cids
        ),
        blocked_population=_population_digest(PopulationKind.BLOCKED, blocked_cids),
        resource_contract=PlanResourceContract(
            resource_class="cpu-small",
            resource_stage="plan-steer",
            cpu_slots=1,
            process_slots=1,
        ),
        provider_contract=PlanProviderContract(
            provider_requirement="",
            endpoint_policy_class="none",
        ),
        lease_contract=PlanLeaseContract(
            lease_scope="task-source",
            owner_identity_rule="single-writer-materializer",
        ),
        retry_contract=PlanRetryContract(
            max_retries=0,
            compensation_policy="exact-byte-restore",
        ),
        worktree_contract=PlanWorktreeContract(
            policy="require-clean",
            isolation_required=False,
        ),
        merge_strategy=PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        conflict_contract=PlanConflictContract(
            predicted_files=(control_path, coordination_path),
            exclusive_paths=(control_path, coordination_path),
            max_files=2,
        ),
        completion_rule=PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
            required_evidence_kinds=("typed-store-projection", "exact-byte-rollback"),
        ),
        validation_dag=(),
        rollback_ref=evidence_root,
        event_cursor=evidence_root,
    )


def preview_successor(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Construct a body-free, read-only revision-2 continuation proposal."""

    predecessor = _load_predecessor_plan(config, root=root)
    state = _read_successor_state(config, root=root)
    evidence = _assert_quiescent_predecessor(
        config,
        state,
        predecessor=predecessor,
        root=root,
    )
    candidate_population = _successor_candidate_population(
        population,
        state["plan_projection"],
        predecessor=predecessor,
        base_repository_tree_id=str(evidence["base_repository_tree_id"]),
    )
    candidate_projection = _project_candidate(
        candidate_population,
        base_repository_tree_id=str(evidence["base_repository_tree_id"]),
    )
    live_by_cid, live_by_alias = _task_map(
        state["plan_projection"], noun="live predecessor projection"
    )
    candidate_by_cid, candidate_by_alias = _task_map(
        candidate_projection, noun="candidate successor projection"
    )
    added_cids = set(candidate_by_cid) - set(live_by_cid)
    expected_added_cid = str(candidate_by_alias[SUCCESSOR_ADDED_ALIAS]["task_cid"])
    if added_cids != {expected_added_cid}:
        raise MaterializationError("successor adds a task other than LGCVF-113")
    changed_aliases = {
        alias
        for alias in live_by_alias
        if live_by_alias[alias].get("spec_cid")
        != candidate_by_alias[alias].get("spec_cid")
    }
    if changed_aliases != SUCCESSOR_CHANGED_ALIASES:
        raise MaterializationError(
            "successor changed task specifications outside its closed amendment set: "
            + ", ".join(sorted(changed_aliases ^ SUCCESSOR_CHANGED_ALIASES))
        )
    if _protected_blocker_binding(live_by_alias)["binding_cid"] != (
        _protected_blocker_binding(candidate_by_alias)["binding_cid"]
    ):
        raise MaterializationError(
            "protected blocker amendment changed more than its ordinal"
        )
    for alias in set(live_by_alias) - SUCCESSOR_CHANGED_ALIASES:
        if live_by_alias[alias].get("task_cid") != candidate_by_alias[alias].get(
            "task_cid"
        ):
            raise MaterializationError(f"{alias}: retained logical identity changed")

    paths = _successor_paths(config, root=root)
    control_relative = paths["control"].relative_to(root).as_posix()
    coordination_relative = paths["coordination"].relative_to(root).as_posix()
    roots = _revision_roots(
        config,
        population,
        state,
        program_root=str(population["plan_root_cid"]),
    )
    request_body = {
        "operation": "continue-lgcvf-revision-2",
        "base_plan_root": predecessor.content_id,
        "candidate_plan_root": str(population["plan_root_cid"]),
        "evidence_root": str(evidence["evidence_root"]),
    }
    request_cid = content_identity(request_body)
    completed_cids = tuple(str(item) for item in evidence["completed_task_cids"])
    blocked_cids = tuple(str(item) for item in evidence["blocked_task_cids"])
    claimed_population = _population_digest(PopulationKind.CLAIMED)
    completed_population = _population_digest(
        PopulationKind.COMPLETED, completed_cids
    )
    items: list[PlanDeltaItem] = []
    for alias in sorted(SUCCESSOR_CHANGED_ALIASES):
        live = live_by_alias[alias]
        candidate = candidate_by_alias[alias]
        operation = (
            PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK
            if alias in SUCCESSOR_REPRIORITIZED_ALIASES
            else PlanDeltaOperation.AMEND_UNSTARTED_TASK
        )
        effect = f"{operation.value}:{alias}"
        items.append(
            PlanDeltaItem(
                item_key=f"revision-2-{alias.lower()}",
                operation=operation,
                target_cid=str(live["task_cid"]),
                expected_target_lifecycle=_task_lifecycle(live.get("status")),
                expected_target_spec_revision=str(live.get("spec_cid") or ""),
                before_digest=str(live.get("spec_cid") or ""),
                after_record_cid=str(candidate.get("spec_cid") or ""),
                effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
                rationale=(
                    "Shift the protected blocker ordinal without changing its authority."
                    if operation is PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK
                    else (
                        "Apply reviewed metadata and gate launch on LGCVF-113."
                        if alias == "LGCVF-120"
                        else "Apply the reviewed revision-2 operational metadata amendment."
                    )
                ),
                provenance={
                    "predecessor_plan_cid": predecessor.content_id,
                    "candidate_plan_cid": population["plan_root_cid"],
                    "evidence_root": evidence["evidence_root"],
                },
                expected_effects=(effect,),
                rollback_refs=(str(evidence["evidence_root"]),),
                affected_task_cids=(str(live["task_cid"]),),
            )
        )
    add_effect = f"add_task:{SUCCESSOR_ADDED_ALIAS}"
    items.append(
        PlanDeltaItem(
            item_key="revision-2-add-lgcvf-113",
            operation=PlanDeltaOperation.ADD_TASK,
            target_cid="",
            expected_target_lifecycle=LifecycleState.PROPOSED,
            expected_target_spec_revision="",
            before_digest="",
            after_record_cid=expected_added_cid,
            effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
            rationale="Add the independently judged hermetic qualification task.",
            provenance={
                "predecessor_plan_cid": predecessor.content_id,
                "candidate_plan_cid": population["plan_root_cid"],
                "evidence_root": evidence["evidence_root"],
            },
            expected_effects=(add_effect,),
            rollback_refs=(str(evidence["evidence_root"]),),
            affected_task_cids=(expected_added_cid,),
        )
    )
    expected_effects = tuple(
        effect for item in items for effect in item.expected_effects
    )
    admission_body = {
        "request_cid": request_cid,
        "predecessor_archive": predecessor.content_id,
        "evidence_root": evidence["evidence_root"],
        "changed_aliases": sorted(SUCCESSOR_CHANGED_ALIASES),
        "added_aliases": [SUCCESSOR_ADDED_ALIAS],
        "runtime_dependency_edges": [
            [task_alias, dependency_alias]
            for task_alias, dependency_aliases in sorted(
                SUCCESSOR_RUNTIME_DEPENDENCIES.items()
            )
            for dependency_alias in dependency_aliases
        ],
        "active_authority_count": 0,
        "history_rewritten": False,
    }
    admission_cid = content_identity(admission_body)
    delta = PlanDelta(
        base_plan_root=predecessor.content_id,
        base_plan_revision=1,
        request_cid=request_cid,
        roots=roots,
        items=tuple(items),
        expected_effects=expected_effects,
        deferred_item_keys=(),
        claimed_population_digest=claimed_population.digest,
        accepted_population_digest=completed_population.digest,
        scan_receipt_cid=str(evidence["evidence_root"]),
        evidence_bundle_cid=str(evidence["evidence_root"]),
        admission_receipt_cid=admission_cid,
    )
    predecessor_roots = _revision_roots(
        config,
        population,
        state,
        program_root=predecessor.content_id,
    )
    predecessor_admission_cid = content_identity(
        {
            "operation": "adopt-live-predecessor",
            "predecessor_plan_cid": predecessor.content_id,
            "evidence_root": evidence["evidence_root"],
        }
    )
    goal_cids = tuple(
        [predecessor.goals[0].content_id]
        + [subgoal.content_id for subgoal in predecessor.subgoals]
    )
    retained_cids = tuple(sorted(live_by_cid))
    predecessor_revision = _revision_contract(
        plan_root_cid=predecessor.content_id,
        semantic_revision=1,
        parent_plan_root="",
        origin=PlanOrigin.CREATE,
        roots=predecessor_roots,
        request_cid=predecessor_admission_cid,
        delta_cid="",
        evidence_root=str(evidence["evidence_root"]),
        admission_receipt_cid=predecessor_admission_cid,
        goal_cids=goal_cids,
        task_cids=retained_cids,
        added_cids=retained_cids,
        retained_cids=(),
        claimed_cids=(),
        completed_cids=completed_cids,
        blocked_cids=blocked_cids,
        control_path=control_relative,
        coordination_path=coordination_relative,
    )
    successor_revision = _revision_contract(
        plan_root_cid=str(population["plan_root_cid"]),
        semantic_revision=2,
        parent_plan_root=predecessor.content_id,
        origin=PlanOrigin.STEER,
        roots=roots,
        request_cid=request_cid,
        delta_cid=delta.delta_cid,
        evidence_root=str(evidence["evidence_root"]),
        admission_receipt_cid=admission_cid,
        goal_cids=goal_cids,
        task_cids=tuple(sorted(candidate_by_cid)),
        added_cids=(expected_added_cid,),
        retained_cids=retained_cids,
        claimed_cids=(),
        completed_cids=completed_cids,
        blocked_cids=blocked_cids,
        control_path=control_relative,
        coordination_path=coordination_relative,
    )
    preview = {
        "schema": SUCCESSOR_PREVIEW_SCHEMA,
        "disposition": "admitted",
        "write_performed": False,
        "predecessor_plan_cid": predecessor.content_id,
        "candidate_plan_cid": population["plan_root_cid"],
        "predecessor_revision": predecessor_revision.to_dict(),
        "successor_revision": successor_revision.to_dict(),
        "delta": delta.to_dict(),
        "evidence": evidence,
        "admission": admission_body,
        "candidate_population": candidate_population,
        "candidate_task_spec_cids": {
            alias: str(task.get("spec_cid") or "")
            for alias, task in sorted(candidate_by_alias.items())
        },
        "retained_task_cids": list(retained_cids),
        "completed_task_cids": list(completed_cids),
        "blocked_task_cids": list(blocked_cids),
        "added_task_cids": [expected_added_cid],
        "amended_aliases": sorted(SUCCESSOR_AMENDED_ALIASES),
        "reprioritized_aliases": sorted(SUCCESSOR_REPRIORITIZED_ALIASES),
        "runtime_dependency_edges": admission_body["runtime_dependency_edges"],
        "runtime_dependency_expected_cids": {
            alias: sorted(
                str(dependency.get("dependency_task_cid") or "")
                for dependency in live_by_alias[alias].get("dependencies") or ()
                if isinstance(dependency, Mapping)
            )
            for alias in sorted(SUCCESSOR_RUNTIME_DEPENDENCIES)
        },
        "expected_effects": list(expected_effects),
        "execution_store_sha256": state["store_observations"]["execution"]["sha256"],
    }
    preview["preview_cid"] = content_identity(preview)
    return preview


def _composite_projection(
    control_path: Path, coordination_path: Path
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        control_path,
        owner_id="lgcvf-successor:projection",
        install_schema=False,
    )
    try:
        plan = _plain_json(source.plan_projection())
        completion = _plain_json(source.completion_evidence_projection())
    finally:
        source.close()
    coordination = read_coordination_registry_projection(coordination_path)
    projection = {
        "schema": SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA,
        "plan_projection_cid": str(plan.get("projection_cid") or ""),
        "completion_projection_cid": str(completion.get("projection_cid") or ""),
        "coordination_projection_root": str(coordination.get("projection_root") or ""),
    }
    projection["projection_cid"] = content_identity(projection)
    return projection


def _validate_applied_successor_projection(
    control_path: Path,
    coordination_path: Path,
    *,
    expected_task_spec_cids: Mapping[str, str],
    completed_task_cids: Sequence[str],
    expected_completion_binding: Mapping[str, Any],
    expected_blocker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Check protected postconditions while the revision backup is live."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        control_path,
        owner_id="lgcvf-successor:postcondition",
        install_schema=False,
    )
    try:
        plan = _plain_json(source.plan_projection())
        completion = _plain_json(source.completion_evidence_projection())
    finally:
        source.close()
    coordination = read_coordination_registry_projection(coordination_path)
    by_cid, by_alias = _task_map(plan, noun="applied successor projection")
    if set(by_alias) != set(expected_task_spec_cids):
        raise MaterializationError("applied successor task population differs")
    for alias, expected_spec_cid in expected_task_spec_cids.items():
        if by_alias[alias].get("spec_cid") != expected_spec_cid:
            raise MaterializationError(f"{alias}: applied successor spec differs")
    registered = {
        str(item.get("task_cid") or ""): str(item.get("task_id") or "")
        for item in coordination.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    expected_registered = {
        cid: str(task.get("task_alias") or "") for cid, task in by_cid.items()
    }
    if registered != expected_registered:
        raise MaterializationError("applied successor coordination registry differs")
    expected_edges = {
        (cid, str(dependency.get("dependency_task_cid") or ""))
        for cid, task in by_cid.items()
        for dependency in task.get("dependencies") or ()
        if isinstance(dependency, Mapping)
    }
    observed_edges = {
        (str(item.get("task_cid") or ""), str(item.get("dependency_task_cid") or ""))
        for item in coordination.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    if observed_edges != expected_edges:
        raise MaterializationError("applied successor coordination dependencies differ")
    state = {
        "completion_projection": completion,
        "coordination_projection": coordination,
    }
    if _retained_completion_binding(state, completed_task_cids) != dict(
        expected_completion_binding
    ):
        raise MaterializationError(
            "accepted completion evidence changed during successor apply"
        )
    if _protected_blocker_binding(by_alias) != dict(expected_blocker_binding):
        raise MaterializationError(
            "protected blocker authority changed during successor apply"
        )
    projection = {
        "schema": SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA,
        "plan_projection_cid": str(plan.get("projection_cid") or ""),
        "completion_projection_cid": str(completion.get("projection_cid") or ""),
        "coordination_projection_root": str(
            coordination.get("projection_root") or ""
        ),
    }
    projection["projection_cid"] = content_identity(projection)
    return projection


@dataclass
class _SuccessorProjectionAdapter:
    """Narrow adapter joining control and coordination under one rollback set."""

    database_path: Path
    coordination_path: Path
    candidate_population: Mapping[str, Any]
    predecessor_plan_cid: str
    expected_task_spec_cids: Mapping[str, str]
    completed_task_cids: tuple[str, ...]
    expected_completion_binding: Mapping[str, Any]
    expected_blocker_binding: Mapping[str, Any]
    runtime_dependency_expected_cids: Mapping[str, Sequence[str]]

    def plan_revision_projection_paths(self) -> Mapping[str, Path]:
        return {
            "control": self.database_path,
            "coordination": self.coordination_path,
        }

    def plan_revision_projection_cid(self) -> str:
        return str(
            _composite_projection(
                self.database_path, self.coordination_path
            )["projection_cid"]
        )

    def apply_plan_revision(self, **kwargs: Any) -> Mapping[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
            DatabaseCoordinator,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )

        source = DatabaseTaskSource(
            self.database_path,
            owner_id="lgcvf-successor:single-writer",
            plan_root_cid=self.predecessor_plan_cid,
            install_schema=False,
        )
        try:
            result = source.apply_plan_revision(
                **{
                    **kwargs,
                    "goal_graph": self.candidate_population,
                    "admission": None,
                }
            )
            task = source.get_task(SUCCESSOR_ADDED_ALIAS)
            if task is None:
                raise MaterializationError("LGCVF-113 was not added to the control store")
            added_task = task.to_dict()
        finally:
            source.close()
        candidate_by_alias = {
            str(task.get("task_id") or ""): task
            for task in self.candidate_population.get("tasks") or ()
            if isinstance(task, Mapping)
        }
        coordinator = DatabaseCoordinator(self.coordination_path)
        try:
            coordinator.open()
            registration = coordinator.register_task(
                task_cid=str(added_task["task_cid"]),
                task_id=str(added_task["task_alias"]),
                dependency_task_cids=tuple(str(item) for item in added_task["dependencies"]),
                body={
                    "task_alias": added_task["task_alias"],
                    "status": added_task["status"],
                    "priority": added_task["priority"],
                    "authority": "lgcvf-plan-revision-2",
                },
            )
            dependency_amendments: list[dict[str, Any]] = []
            for task_alias, dependency_aliases in sorted(
                SUCCESSOR_RUNTIME_DEPENDENCIES.items()
            ):
                target = candidate_by_alias.get(task_alias)
                if not isinstance(target, Mapping):
                    raise MaterializationError(
                        f"{task_alias}: runtime dependency target is absent"
                    )
                current_expected = [
                    str(item)
                    for item in self.runtime_dependency_expected_cids.get(
                        task_alias, ()
                    )
                ]
                for dependency_alias in dependency_aliases:
                    dependency = candidate_by_alias.get(dependency_alias)
                    if not isinstance(dependency, Mapping):
                        raise MaterializationError(
                            f"{dependency_alias}: runtime dependency task is absent"
                        )
                    dependency_cid = str(dependency.get("task_cid") or "")
                    dependency_amendments.append(
                        coordinator.add_unstarted_task_dependency(
                            task_cid=str(target.get("task_cid") or ""),
                            dependency_task_cid=dependency_cid,
                            expected_dependency_task_cids=tuple(current_expected),
                            operation_id=(
                                "lgcvf-revision-2-dependency:"
                                f"{task_alias}:{dependency_alias}"
                            ),
                        )
                    )
                    current_expected.append(dependency_cid)
        finally:
            coordinator.close()
        projection = _validate_applied_successor_projection(
            self.database_path,
            self.coordination_path,
            expected_task_spec_cids=self.expected_task_spec_cids,
            completed_task_cids=self.completed_task_cids,
            expected_completion_binding=self.expected_completion_binding,
            expected_blocker_binding=self.expected_blocker_binding,
        )
        return {
            **dict(result),
            "coordination_registration": registration,
            "coordination_dependency_amendments": dependency_amendments,
            "projection_cid": projection["projection_cid"],
        }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_successor_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("successor receipt is absent or unreadable") from exc
    if not isinstance(value, dict) or value.get("schema") != SUCCESSOR_RECEIPT_SCHEMA:
        raise MaterializationError("successor receipt schema differs")
    claimed = str(value.pop("receipt_cid", ""))
    observed = content_identity(value)
    value["receipt_cid"] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError("successor receipt content identity differs")
    return value


def _successor_receipt_path(paths: Mapping[str, Path], revision_cid: str) -> Path:
    if not revision_cid or "/" in revision_cid or ".." in revision_cid:
        raise MaterializationError("successor revision CID is unsafe")
    return paths["revision_receipts"] / f"{revision_cid}.json"


def _successor_manifest_path(paths: Mapping[str, Path], revision_cid: str) -> Path:
    if not revision_cid or "/" in revision_cid or ".." in revision_cid:
        raise MaterializationError("successor revision CID is unsafe")
    return (
        paths["revision_store"]
        / "lgcvf-successor-manifests"
        / f"{revision_cid}.json"
    )


def _load_successor_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("successor recovery manifest is absent or unreadable") from exc
    if not isinstance(value, dict) or value.get("schema") != SUCCESSOR_RECOVERY_MANIFEST_SCHEMA:
        raise MaterializationError("successor recovery manifest schema differs")
    claimed = str(value.pop("manifest_cid", ""))
    observed = content_identity(value)
    value["manifest_cid"] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError("successor recovery manifest identity differs")
    return value


def _prepare_successor_manifest(
    paths: Mapping[str, Path],
    preview: Mapping[str, Any],
    before: Mapping[str, str],
) -> dict[str, Any]:
    candidate_tasks = {
        str(task.get("task_id") or ""): str(task.get("task_cid") or "")
        for task in preview["candidate_population"].get("tasks") or ()
        if isinstance(task, Mapping)
    }
    manifest = {
        "schema": SUCCESSOR_RECOVERY_MANIFEST_SCHEMA,
        "predecessor_plan_cid": preview["predecessor_plan_cid"],
        "candidate_plan_cid": preview["candidate_plan_cid"],
        "predecessor_revision_cid": content_identity(preview["predecessor_revision"]),
        "successor_revision_cid": content_identity(preview["successor_revision"]),
        "delta_cid": content_identity(preview["delta"]),
        "preview_cid": preview["preview_cid"],
        "predecessor_archive_sha256": preview["evidence"][
            "predecessor_archive_sha256"
        ],
        "bootstrap_receipt_cid": preview["evidence"]["bootstrap_receipt_cid"],
        "predecessor_evidence_root": preview["evidence"]["evidence_root"],
        "retained_task_cids": list(preview["retained_task_cids"]),
        "completed_task_cids": list(preview["completed_task_cids"]),
        "blocked_task_cids": list(preview["blocked_task_cids"]),
        "added_task_cids": list(preview["added_task_cids"]),
        "candidate_task_cids": candidate_tasks,
        "candidate_task_spec_cids": dict(preview["candidate_task_spec_cids"]),
        "retained_completion_binding": preview["evidence"][
            "retained_completion_binding"
        ],
        "protected_blocker_binding": preview["evidence"][
            "protected_blocker_binding"
        ],
        "database_sha256_before": dict(before),
    }
    manifest["manifest_cid"] = content_identity(manifest)
    path = _successor_manifest_path(
        paths, str(manifest["successor_revision_cid"])
    )
    if path.exists():
        existing = _load_successor_manifest(path)
        if existing != manifest:
            raise MaterializationError(
                "existing successor recovery manifest differs from current evidence"
            )
        return existing
    _atomic_write_json(path, manifest)
    return manifest


def _committed_apply_receipt(
    store: Any,
    *,
    revision_cid: str,
    candidate_plan_cid: str,
    delta_cid: str,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionStoreError,
    )

    continuation_paths = sorted(store.continuations_dir.glob("*.json"))
    if len(continuation_paths) > 64:
        raise MaterializationError("plan revision continuation population exceeds bound")
    committed: list[Mapping[str, Any]] = []
    for path in continuation_paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MaterializationError("plan revision continuation is unreadable") from exc
        if not isinstance(record, Mapping):
            raise MaterializationError("plan revision continuation is malformed")
        key = str(record.get("idempotency_key") or "")
        payload = store.load_continuation(key) if key else None
        if (
            isinstance(payload, Mapping)
            and payload.get("phase") == "committed"
            and payload.get("revision_cid") == revision_cid
        ):
            committed.append(payload)
    if len(committed) != 1:
        raise MaterializationError(
            "exactly one committed successor continuation is required"
        )
    receipt_cid = str(committed[0].get("receipt_cid") or "")
    try:
        receipt = store.get_cas(receipt_cid)
    except (OSError, PlanRevisionStoreError) as exc:
        raise MaterializationError("committed plan-revision receipt is unavailable") from exc
    if (
        receipt.get("receipt_cid") != receipt_cid
        or receipt.get("state") != "committed"
        or receipt.get("revision_cid") != revision_cid
        or receipt.get("plan_root_cid") != candidate_plan_cid
        or receipt.get("delta_cid") != delta_cid
    ):
        raise MaterializationError("committed plan-revision receipt identity differs")
    return _plain_json(receipt)


def _build_successor_receipt(
    manifest: Mapping[str, Any],
    apply_receipt: Mapping[str, Any],
    post_state: Mapping[str, Any],
    after: Mapping[str, str],
) -> dict[str, Any]:
    receipt = {
        "schema": SUCCESSOR_RECEIPT_SCHEMA,
        "authority_mode": "embedded-single-writer",
        "production_authorized": False,
        "predecessor_plan_cid": manifest["predecessor_plan_cid"],
        "candidate_plan_cid": manifest["candidate_plan_cid"],
        "predecessor_revision_cid": manifest["predecessor_revision_cid"],
        "successor_revision_cid": manifest["successor_revision_cid"],
        "delta_cid": manifest["delta_cid"],
        "preview_cid": manifest["preview_cid"],
        "recovery_manifest_cid": manifest["manifest_cid"],
        "plan_revision_apply_receipt": dict(apply_receipt),
        "predecessor_archive_sha256": manifest["predecessor_archive_sha256"],
        "bootstrap_receipt_cid": manifest["bootstrap_receipt_cid"],
        "bootstrap_receipt_sha256": manifest["database_sha256_before"]["receipt"],
        "predecessor_evidence_root": manifest["predecessor_evidence_root"],
        "retained_task_cids": list(manifest["retained_task_cids"]),
        "completed_task_cids": list(manifest["completed_task_cids"]),
        "blocked_task_cids": list(manifest["blocked_task_cids"]),
        "added_task_cids": list(manifest["added_task_cids"]),
        "candidate_task_spec_cids": dict(manifest["candidate_task_spec_cids"]),
        "retained_completion_binding": manifest["retained_completion_binding"],
        "protected_blocker_binding": manifest["protected_blocker_binding"],
        "post_composite_projection": post_state["composite_projection"],
        "database_sha256_before": dict(manifest["database_sha256_before"]),
        "database_sha256_after": dict(after),
        "execution_store_mutated": False,
        "bootstrap_receipt_mutated": False,
        "historical_status_rewritten": False,
        "manual_or_external_task_completed": False,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    return receipt


def _finalize_committed_successor(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path,
    paths: Mapping[str, Path],
    store: Any,
) -> dict[str, Any]:
    """Reconstruct the external receipt after a committed crash window."""

    predecessor = _load_predecessor_plan(config, root=root)
    active = store.get_active()
    candidate_plan_cid = str(population.get("plan_root_cid") or "")
    if (
        active is None
        or active.plan_root_cid != candidate_plan_cid
        or active.semantic_revision != 2
        or active.quarantined
    ):
        raise MaterializationError("committed successor recovery has no valid active head")
    revision = store.load_revision(active.revision_cid)
    if revision.parent_plan_root != predecessor.content_id:
        raise MaterializationError("committed successor ancestry differs")
    manifest = _load_successor_manifest(
        _successor_manifest_path(paths, active.revision_cid)
    )
    if (
        manifest.get("predecessor_plan_cid") != predecessor.content_id
        or manifest.get("candidate_plan_cid") != candidate_plan_cid
        or manifest.get("successor_revision_cid") != active.revision_cid
        or manifest.get("delta_cid") != revision.delta_cid
        or manifest.get("predecessor_archive_sha256")
        != _sha256_file(paths["predecessor_archive"])
    ):
        raise MaterializationError("committed successor recovery manifest is stale")
    apply_receipt = _committed_apply_receipt(
        store,
        revision_cid=active.revision_cid,
        candidate_plan_cid=candidate_plan_cid,
        delta_cid=revision.delta_cid,
    )
    post_state = _read_successor_state(config, root=root)
    live_by_cid, live_by_alias = _task_map(
        post_state["plan_projection"], noun="committed successor projection"
    )
    expected_task_cids = manifest.get("candidate_task_cids")
    if not isinstance(expected_task_cids, Mapping) or {
        alias: str(task.get("task_cid") or "")
        for alias, task in live_by_alias.items()
    } != dict(expected_task_cids):
        raise MaterializationError("committed successor logical task identities differ")
    expected_specs = manifest.get("candidate_task_spec_cids")
    if not isinstance(expected_specs, Mapping):
        raise MaterializationError("committed successor spec manifest is malformed")
    projection = _validate_applied_successor_projection(
        paths["control"],
        paths["coordination"],
        expected_task_spec_cids={str(k): str(v) for k, v in expected_specs.items()},
        completed_task_cids=tuple(
            str(item) for item in manifest.get("completed_task_cids") or ()
        ),
        expected_completion_binding=manifest["retained_completion_binding"],
        expected_blocker_binding=manifest["protected_blocker_binding"],
    )
    if (
        projection != post_state["composite_projection"]
        or projection["projection_cid"] != apply_receipt["duckdb_projection_cid"]
        or len(live_by_cid) != len(expected_task_cids)
    ):
        raise MaterializationError("committed successor projection receipt differs")
    before = manifest.get("database_sha256_before")
    if not isinstance(before, Mapping):
        raise MaterializationError("committed successor pre-state hashes are absent")
    after = {
        key: _sha256_file(paths[key])
        for key in ("control", "coordination", "execution", "receipt")
    }
    if after["execution"] != before.get("execution"):
        raise MaterializationError("committed successor changed the execution store")
    if after["receipt"] != before.get("receipt"):
        raise MaterializationError("committed successor changed the bootstrap receipt")
    receipt = _build_successor_receipt(manifest, apply_receipt, post_state, after)
    receipt_path = _successor_receipt_path(paths, active.revision_cid)
    _atomic_write_json(receipt_path, receipt)
    verify_successor_read_only(config, population, root=root)
    return receipt


def steer_successor(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    fault_injector: Any | None = None,
) -> dict[str, Any]:
    """Adopt revision 1 and atomically steer control+coordination to revision 2."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionApplyRequest,
        PlanRevisionStore,
        PlanRevisionStoreError,
    )

    paths = _successor_paths(config, root=root)
    try:
        store = PlanRevisionStore(paths["revision_store"], recover=True)
    except PlanRevisionStoreError as exc:
        raise MaterializationError(
            f"plan revision recovery failed closed: {exc}"
        ) from exc
    existing_receipts = sorted(paths["revision_receipts"].glob("*.json"))
    if existing_receipts:
        if len(existing_receipts) != 1:
            raise MaterializationError("multiple successor receipts fail idempotent replay")
        receipt = _load_successor_receipt(existing_receipts[0])
        verify_successor_read_only(config, population, root=root)
        return receipt
    active = store.get_active()
    candidate_plan_cid = str(population.get("plan_root_cid") or "")
    if (
        active is not None
        and active.plan_root_cid == candidate_plan_cid
        and active.semantic_revision == 2
    ):
        return _finalize_committed_successor(
            config,
            population,
            root=root,
            paths=paths,
            store=store,
        )
    preview = preview_successor(config, population, root=root)
    predecessor_revision = PlanRevision.from_dict(preview["predecessor_revision"])
    successor_revision = PlanRevision.from_dict(preview["successor_revision"])
    delta = PlanDelta.from_dict(preview["delta"])
    receipt_path = _successor_receipt_path(paths, successor_revision.revision_cid)
    if receipt_path.exists():
        raise MaterializationError("successor receipt path collided after preview")

    before = {
        key: _sha256_file(paths[key])
        for key in ("control", "coordination", "execution", "receipt")
    }
    active = store.get_active()
    if active is None:
        adoption_base = (
            "lgcvf-adopt-revision-1:"
            + str(preview["evidence"]["evidence_root"])
        )
        adoption_key = ""
        for retry_index in range(9):
            candidate_key = (
                adoption_base
                if retry_index == 0
                else f"{adoption_base}:retry-{retry_index}"
            )
            continuation = store.load_continuation(candidate_key)
            if continuation is None:
                adoption_key = candidate_key
                break
            if str(continuation.get("phase") or "") not in {"restored", "blocked"}:
                raise MaterializationError(
                    "prior predecessor adoption remains non-retryable"
                )
        if not adoption_key:
            raise MaterializationError("predecessor adoption retry key budget is exhausted")
        try:
            adoption = store.apply(
                PlanRevisionApplyRequest(
                    revision=predecessor_revision,
                    observed_roots=predecessor_revision.roots,
                    idempotency_key=adoption_key,
                    expected_effects=("adopt-live-predecessor",),
                    records={
                        "predecessor-formal-plan": {
                            "plan_cid": preview["predecessor_plan_cid"],
                            "archive_sha256": preview["evidence"][
                                "predecessor_archive_sha256"
                            ],
                        },
                        "predecessor-evidence": preview["evidence"],
                    },
                )
            )
        except PlanRevisionStoreError as exc:
            raise MaterializationError(
                f"predecessor revision adoption failed closed: {exc}"
            ) from exc
        active = store.get_active()
        if active is None or not adoption.committed:
            raise MaterializationError("predecessor adoption did not commit")
    if (
        active.plan_root_cid != predecessor_revision.plan_root_cid
        or active.revision_cid != predecessor_revision.revision_cid
        or active.semantic_revision != 1
    ):
        raise MaterializationError("revision store active pointer is not the predecessor")
    manifest = _prepare_successor_manifest(paths, preview, before)

    adapter = _SuccessorProjectionAdapter(
        database_path=paths["control"],
        coordination_path=paths["coordination"],
        candidate_population=preview["candidate_population"],
        predecessor_plan_cid=predecessor_revision.plan_root_cid,
        expected_task_spec_cids=preview["candidate_task_spec_cids"],
        completed_task_cids=tuple(preview["completed_task_cids"]),
        expected_completion_binding=preview["evidence"][
            "retained_completion_binding"
        ],
        expected_blocker_binding=preview["evidence"]["protected_blocker_binding"],
        runtime_dependency_expected_cids=preview[
            "runtime_dependency_expected_cids"
        ],
    )
    idempotency_base = f"lgcvf-steer-revision-2:{preview['preview_cid']}"
    idempotency_key = ""
    for retry_index in range(9):
        candidate_key = (
            idempotency_base
            if retry_index == 0
            else f"{idempotency_base}:retry-{retry_index}"
        )
        continuation = store.load_continuation(candidate_key)
        if continuation is None:
            idempotency_key = candidate_key
            break
        phase = str(continuation.get("phase") or "")
        if phase not in {"restored", "blocked"}:
            raise MaterializationError(
                f"prior successor continuation remains non-retryable at {phase!r}"
            )
    if not idempotency_key:
        raise MaterializationError("successor retry key budget is exhausted")
    try:
        apply_receipt = store.apply(
            PlanRevisionApplyRequest(
                revision=successor_revision,
                observed_roots=successor_revision.roots,
                idempotency_key=idempotency_key,
                expected_effects=tuple(preview["expected_effects"]),
                delta=delta,
                goal_graph=preview["candidate_population"],
                duckdb_source=adapter,
                repository_tree_id=str(
                    preview["evidence"]["base_repository_tree_id"]
                ),
                fencing_token=1,
                base_event_cursor=active.event_cursor,
                expected_active_plan_root=active.plan_root_cid,
                expected_active_revision_cid=active.revision_cid,
                fault_injector=fault_injector,
                records={
                    "successor-formal-plan": {
                        "plan_cid": preview["candidate_plan_cid"],
                        "population_root": population["population_root"],
                    },
                    "successor-admission": preview["admission"],
                    "successor-recovery-manifest": manifest,
                },
            )
        )
    except PlanRevisionStoreError as exc:
        restored = {
            key: _sha256_file(paths[key])
            for key in ("control", "coordination", "execution", "receipt")
        }
        if restored != before:
            raise MaterializationError(
                "successor apply failed and exact operational rollback did not verify"
            ) from exc
        raise MaterializationError(
            f"successor apply failed after exact operational rollback: {exc}"
        ) from exc
    if not apply_receipt.committed:
        raise MaterializationError("successor plan revision did not commit")
    if callable(fault_injector):
        fault_injector("after_revision_commit_before_external_receipt")
    after = {
        key: _sha256_file(paths[key])
        for key in ("control", "coordination", "execution", "receipt")
    }
    if after["execution"] != before["execution"]:
        raise MaterializationError("successor apply changed the execution store")
    if after["receipt"] != before["receipt"]:
        raise MaterializationError("successor apply changed the bootstrap receipt")
    post_state = _read_successor_state(config, root=root)
    post_plan_by_cid, post_plan_by_alias = _task_map(
        post_state["plan_projection"], noun="applied successor projection"
    )
    if set(post_plan_by_alias) != set(preview["candidate_task_spec_cids"]):
        raise MaterializationError("applied successor task population differs")
    for alias, expected_spec in preview["candidate_task_spec_cids"].items():
        if post_plan_by_alias[alias].get("spec_cid") != expected_spec:
            raise MaterializationError(f"{alias}: applied successor spec differs")
    predecessor_completed = set(preview["completed_task_cids"])
    post_completion_binding = _retained_completion_binding(
        post_state, sorted(predecessor_completed)
    )
    if post_completion_binding != preview["evidence"]["retained_completion_binding"]:
        raise MaterializationError("accepted completion evidence changed during successor apply")
    post_blocker_binding = _protected_blocker_binding(post_plan_by_alias)
    if post_blocker_binding != preview["evidence"]["protected_blocker_binding"]:
        raise MaterializationError("protected blocker authority changed during successor apply")
    receipt = _build_successor_receipt(
        manifest, apply_receipt.to_dict(), post_state, after
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _directory_fingerprint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        raise MaterializationError(f"required directory is absent: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if len(files) > 10_000:
        raise MaterializationError("revision store file population exceeds bound")
    return {
        item.relative_to(path).as_posix(): {
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
            "sha256": _sha256_file(item),
        }
        for item in files
    }


def verify_successor_read_only(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Reconstruct revision ancestry and immutable task specs without writes."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionStore,
    )

    predecessor = _load_predecessor_plan(config, root=root)
    paths = _successor_paths(config, root=root)
    if not paths["revision_store"].is_dir():
        raise MaterializationError("plan revision store is absent")
    receipt_files = sorted(paths["revision_receipts"].glob("*.json"))
    if len(receipt_files) != 1:
        raise MaterializationError("exactly one successor revision receipt is required")
    receipt = _load_successor_receipt(receipt_files[0])
    manifest = _load_successor_manifest(
        _successor_manifest_path(
            paths, str(receipt.get("successor_revision_cid") or "")
        )
    )
    if (
        receipt.get("recovery_manifest_cid") != manifest.get("manifest_cid")
        or receipt.get("preview_cid") != manifest.get("preview_cid")
        or receipt.get("delta_cid") != manifest.get("delta_cid")
        or receipt.get("database_sha256_before")
        != manifest.get("database_sha256_before")
        or receipt.get("retained_completion_binding")
        != manifest.get("retained_completion_binding")
        or receipt.get("protected_blocker_binding")
        != manifest.get("protected_blocker_binding")
    ):
        raise MaterializationError("successor receipt/recovery manifest binding differs")
    before_revision_store = _directory_fingerprint(paths["revision_store"])
    before_receipt = (
        receipt_files[0].stat().st_size,
        receipt_files[0].stat().st_mtime_ns,
        _sha256_file(receipt_files[0]),
    )
    state = _read_successor_state(config, root=root)
    plan = state["plan_projection"]
    coordination = state["coordination_projection"]
    active_plans = [
        item
        for item in plan.get("plans") or ()
        if isinstance(item, Mapping) and item.get("status") == "active"
    ]
    candidate_plan_cid = str(population.get("plan_root_cid") or "")
    if len(active_plans) != 1 or active_plans[0].get("plan_cid") != candidate_plan_cid:
        raise MaterializationError("revision-2 plan is not the exact active control head")
    plan_rows = {
        str(item.get("plan_cid") or ""): item
        for item in plan.get("plans") or ()
        if isinstance(item, Mapping)
    }
    if (
        predecessor.content_id not in plan_rows
        or plan_rows[predecessor.content_id].get("status") == "active"
    ):
        raise MaterializationError("predecessor plan was lost or remains active")
    live_by_cid, live_by_alias = _task_map(plan, noun="live successor projection")
    desired_tasks = {
        str(item.get("task_id") or ""): str(item.get("task_cid") or "")
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    expected_tasks = {task.task_id: task.content_id for task in predecessor.tasks}
    if SUCCESSOR_ADDED_ALIAS not in desired_tasks:
        raise MaterializationError("revision-2 formal plan omits LGCVF-113")
    expected_tasks[SUCCESSOR_ADDED_ALIAS] = desired_tasks[SUCCESSOR_ADDED_ALIAS]
    observed_tasks = {
        alias: str(item.get("task_cid") or "") for alias, item in live_by_alias.items()
    }
    if observed_tasks != expected_tasks:
        raise MaterializationError("successor logical task identities differ")
    expected_specs = receipt.get("candidate_task_spec_cids")
    if not isinstance(expected_specs, Mapping) or set(expected_specs) != set(live_by_alias):
        raise MaterializationError("successor receipt task spec population differs")
    for alias, task in live_by_alias.items():
        if task.get("spec_cid") != expected_specs.get(alias):
            raise MaterializationError(f"{alias}: current task specification is stale")

    registered = {
        str(item.get("task_cid") or ""): str(item.get("task_id") or "")
        for item in coordination.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    if registered != {cid: alias for alias, cid in expected_tasks.items()}:
        raise MaterializationError("successor coordination task registry differs")
    expected_edges = {
        (cid, str(dependency.get("dependency_task_cid") or ""))
        for cid, task in live_by_cid.items()
        for dependency in task.get("dependencies") or ()
        if isinstance(dependency, Mapping)
    }
    observed_edges = {
        (str(item.get("task_cid") or ""), str(item.get("dependency_task_cid") or ""))
        for item in coordination.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    if observed_edges != expected_edges:
        raise MaterializationError("successor coordination dependencies differ")
    for alias, authority, completion_mode in (
        ("LGCVF-121", "blocked_external_authority", "external-authority"),
        ("LGCVF-123", "blocked_manual", "manual"),
    ):
        task = live_by_alias.get(alias)
        body = task.get("body") if isinstance(task, Mapping) else None
        if (
            not isinstance(task, Mapping)
            or task.get("status") != "blocked"
            or not isinstance(body, Mapping)
            or body.get("construction_status") != authority
            or body.get("completion") != completion_mode
            or body.get("review_only") is not True
        ):
            raise MaterializationError(f"{alias}: protected authority was rewritten")
    retained_completion = receipt.get("retained_completion_binding")
    if not isinstance(retained_completion, Mapping) or (
        _retained_completion_binding(
            state, tuple(str(item) for item in receipt.get("completed_task_cids") or ())
        )
        != retained_completion
    ):
        raise MaterializationError("accepted predecessor completion evidence changed")
    protected_blockers = receipt.get("protected_blocker_binding")
    if not isinstance(protected_blockers, Mapping) or (
        _protected_blocker_binding(live_by_alias) != protected_blockers
    ):
        raise MaterializationError("protected blocker authority binding changed")

    if _sha256_file(paths["receipt"]) != receipt.get("bootstrap_receipt_sha256"):
        raise MaterializationError("bootstrap receipt bytes changed after successor apply")
    bootstrap = state["bootstrap_receipt"]
    if bootstrap.get("receipt_cid") != receipt.get("bootstrap_receipt_cid"):
        raise MaterializationError("bootstrap receipt identity changed")
    store = PlanRevisionStore(paths["revision_store"], recover=False)
    active = store.get_active()
    if (
        active is None
        or active.plan_root_cid != candidate_plan_cid
        or active.revision_cid != receipt.get("successor_revision_cid")
        or active.semantic_revision != 2
        or active.quarantined
    ):
        raise MaterializationError("plan revision store active pointer differs")
    revision = store.load_revision(active.revision_cid)
    if (
        revision.parent_plan_root != predecessor.content_id
        or revision.delta_cid != receipt.get("delta_cid")
        or set(revision.retained_population.member_cids)
        != set(receipt.get("retained_task_cids") or ())
        or set(revision.added_population.member_cids)
        != set(receipt.get("added_task_cids") or ())
    ):
        raise MaterializationError("stored successor ancestry/population differs")
    apply_receipt = receipt.get("plan_revision_apply_receipt")
    if (
        not isinstance(apply_receipt, Mapping)
        or apply_receipt.get("committed") is not True
        or apply_receipt.get("revision_cid") != active.revision_cid
        or apply_receipt.get("plan_root_cid") != active.plan_root_cid
        or apply_receipt.get("delta_cid") != revision.delta_cid
    ):
        raise MaterializationError("plan revision apply receipt differs")
    after_revision_store = _directory_fingerprint(paths["revision_store"])
    after_receipt = (
        receipt_files[0].stat().st_size,
        receipt_files[0].stat().st_mtime_ns,
        _sha256_file(receipt_files[0]),
    )
    if before_revision_store != after_revision_store or before_receipt != after_receipt:
        raise MaterializationError("read-only successor verification changed evidence")
    result = {
        "schema": SUCCESSOR_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only",
        "predecessor_plan_cid": predecessor.content_id,
        "candidate_plan_cid": candidate_plan_cid,
        "successor_revision_cid": active.revision_cid,
        "delta_cid": revision.delta_cid,
        "successor_receipt_cid": receipt["receipt_cid"],
        "task_count": len(live_by_cid),
        "retained_task_count": len(revision.retained_population.member_cids),
        "added_task_count": len(revision.added_population.member_cids),
        "accepted_history_preserved": True,
        "protected_blockers_preserved": True,
        "execution_store_mutated": False,
        "stores_unchanged": True,
        "active_coordination_counts": {
            key: value
            for key, value in dict(coordination.get("counts") or {}).items()
            if key.startswith("active_")
        },
    }
    result["verification_root"] = content_identity(result)
    return result


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
        choices=(
            "population",
            "materialize",
            "verify",
            "successor-preview",
            "successor-steer",
            "successor-verify",
        ),
        help=(
            "render/bootstrap/verify the population, or preview, atomically steer, "
            "and read-only verify the immutable revision-2 continuation"
        ),
    )
    args = parser.parse_args(argv)
    try:
        config = load_config()
        population = build_population(config)
        if args.command == "population":
            result: Mapping[str, Any] = population
        elif args.command == "materialize":
            result = materialize(config, population)
        elif args.command == "successor-preview":
            result = preview_successor(config, population)
        elif args.command == "successor-steer":
            result = steer_successor(config, population)
        elif args.command == "successor-verify":
            result = verify_successor_read_only(config, population)
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
