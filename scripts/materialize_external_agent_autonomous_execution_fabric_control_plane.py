#!/usr/bin/env python3
"""Materialize and verify the EAAEF embedded bootstrap control plane.

This script intentionally materializes only the reconciliation bootstrap
population declared by the reviewed board. Future tasks stay outside the
database until the board's terminal bootstrap task emits a current
semantic-root-bound Plan R2. The bootstrap uses one embedded DuckDB writer; it
neither enables continuous Quack operation nor DuckLake authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct ``python scripts/...py`` execution otherwise exposes only the
    # scripts directory, making the reviewed local package unimportable after
    # the immutable namespace claim has already been published.
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "config/external_agent_autonomous_execution_fabric_scheduler.json"
RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-materialization@1"
)
POPULATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-population@1"
)
NAMESPACE_CLAIM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-namespace-claim@1"
)
SCHEDULER_CONFIG_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "external_agent_autonomous_execution_fabric.scheduler_config@1"
)


class MaterializationError(RuntimeError):
    """Fail-closed bootstrap materialization error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _cid(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid JSON {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{path.relative_to(ROOT)} must contain an object")
    return value


def _relative_path(value: Any, *, field: str) -> Path:
    raw = str(value or "")
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved = (ROOT / path).resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise MaterializationError(f"{field} escapes the repository") from exc
    return resolved


def _git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise MaterializationError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _assert_clean() -> None:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MaterializationError(
            "materialization requires a clean exact source tree; commit the board, "
            "manifests, policy and source bindings first"
        )


def _file_cid(path: Path) -> str:
    try:
        return _cid(path.read_bytes())
    except OSError as exc:
        raise MaterializationError(f"unable to read {path.relative_to(ROOT)}: {exc}") from exc


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    if str(config.get("schema") or "") != SCHEDULER_CONFIG_SCHEMA:
        raise MaterializationError("scheduler config schema identity is not canonical")
    program = config.get("database_program")
    if not isinstance(program, Mapping):
        raise MaterializationError("database_program is missing")
    if str(program.get("authority_mode") or "") != "embedded":
        raise MaterializationError("bootstrap database authority must be embedded")
    if str(program.get("task_source_kind") or "") != "duckdb":
        raise MaterializationError("bootstrap task source must be duckdb")
    if int(program.get("maximum_writer_processes") or 0) != 1:
        raise MaterializationError("bootstrap permits exactly one writer process")
    for field in (
        "store_generation",
        "schema_revision",
        "event_store_path",
        "runtime_registry_path",
        "worktree_root",
        "export_profile",
        "failover_policy",
    ):
        if not str(program.get(field) or "").strip():
            raise MaterializationError(f"database_program.{field} is required")
    if str(program.get("failover_policy")) != "fail_closed":
        raise MaterializationError("bootstrap database failover policy must be fail_closed")
    result = {
        "control": _relative_path(program.get("store_id"), field="database_program.store_id"),
        "coordination": _relative_path(
            program.get("coordination_store_id"),
            field="database_program.coordination_store_id",
        ),
        "execution": _relative_path(
            program.get("execution_store_id"),
            field="database_program.execution_store_id",
        ),
    }
    if len(set(result.values())) != 3:
        raise MaterializationError("control, coordination and execution stores must be distinct")
    control = result["control"]
    if control.suffix.lower() not in {".duckdb", ".ddb"}:
        raise MaterializationError("database_program.store_id must identify a DuckDB file")
    expected = {
        "coordination": control.with_name(f"{control.stem}.coordination.duckdb"),
        "execution": control.with_name(f"{control.stem}.execution.duckdb"),
    }
    for name, path in expected.items():
        if result[name] != path:
            raise MaterializationError(
                f"database_program.{name}_store_id must equal the deterministic "
                f"DatabaseImplementationDaemon sidecar {path.relative_to(ROOT)}"
            )
    return result


def _receipt_path(config: Mapping[str, Any]) -> Path:
    registry = _relative_path(
        (config.get("database_program") or {}).get("runtime_registry_path"),
        field="database_program.runtime_registry_path",
    )
    return registry / "bootstrap-materialization.json"


def _claim_path(config: Mapping[str, Any]) -> Path:
    registry = _relative_path(
        (config.get("database_program") or {}).get("runtime_registry_path"),
        field="database_program.runtime_registry_path",
    )
    return registry / "bootstrap-materialization-claim.json"


def _namespace_artifacts(config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return every known file whose presence makes the namespace non-fresh."""

    paths = _paths(config)
    # The complete run directory is one immutable generation.  Checking only
    # the three databases would admit a stale PID, event cursor, merge queue,
    # worktree, or registry artifact from an earlier partial attempt.
    members: list[Path] = [
        paths["control"].parent,
        *paths.values(),
        _claim_path(config),
        _receipt_path(config),
    ]
    program = config.get("database_program") or {}
    for field in (
        "event_store_path",
        "runtime_registry_path",
        "worktree_root",
        "merge_queue_dir",
        "state_dir",
    ):
        members.append(
            _relative_path(program.get(field), field=f"database_program.{field}")
        )
    for path in paths.values():
        members.append(Path(f"{path}.wal"))
    members.append(
        paths["execution"].with_name(f".{paths['execution'].name}.writer.lock")
    )
    return tuple(dict.fromkeys(members))


def _source_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("source_binding is missing")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    required_accelerator = str(binding.get("ipfs_accelerate_planning_revision") or "")
    required_accelerator_tree = str(binding.get("ipfs_accelerate_planning_tree") or "")
    if not required_accelerator or not required_accelerator_tree:
        raise MaterializationError("reviewed accelerator commit/tree binding is incomplete")
    if _git("rev-parse", f"{required_accelerator}^{{tree}}") != required_accelerator_tree:
        raise MaterializationError("reviewed accelerator integration tree differs from config")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", required_accelerator, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0
    ):
        raise MaterializationError("reviewed accelerator integration root is not an ancestor")
    repositories = {
        "ipfs_accelerate_py": {
            "path": ROOT,
            "head": head,
            "tree": tree,
            "required_head": required_accelerator,
            "required_tree": required_accelerator_tree,
        },
        "ipfs_datasets_py": {
            "path": _relative_path(
                binding.get("ipfs_datasets_submodule_path"),
                field="source_binding.ipfs_datasets_submodule_path",
            ),
            "required_head": str(binding.get("ipfs_datasets_planning_revision") or ""),
            "required_tree": str(binding.get("ipfs_datasets_planning_tree") or ""),
        },
        "ipfs_kit_py": {
            "path": _relative_path(
                binding.get("ipfs_kit_submodule_path"),
                field="source_binding.ipfs_kit_submodule_path",
            ),
            "required_head": str(binding.get("ipfs_kit_planning_revision") or ""),
            "required_tree": str(binding.get("ipfs_kit_planning_tree") or ""),
        },
        "Mcp-Plus-Plus": {
            "path": _relative_path(
                binding.get("mcp_plus_plus_submodule_path"),
                field="source_binding.mcp_plus_plus_submodule_path",
            ),
            "required_head": str(binding.get("mcp_plus_plus_planning_revision") or ""),
            "required_tree": str(binding.get("mcp_plus_plus_planning_tree") or ""),
        },
    }
    projection: dict[str, Any] = {}
    planning_repositories: dict[str, dict[str, str]] = {}
    for name, record in repositories.items():
        path = Path(record["path"])
        nested_head = record.get("head") or _git("rev-parse", "HEAD", cwd=path)
        nested_tree = record.get("tree") or _git("rev-parse", "HEAD^{tree}", cwd=path)
        required_head = str(record.get("required_head") or "")
        required_tree = str(record.get("required_tree") or nested_tree)
        if not required_head or not required_tree:
            raise MaterializationError(f"{name} reviewed commit/tree binding is incomplete")
        if _git("rev-parse", f"{required_head}^{{tree}}", cwd=path) != required_tree:
            raise MaterializationError(f"{name} reviewed commit/tree binding is invalid")
        if name != "ipfs_accelerate_py" and (
            nested_head != required_head or nested_tree != required_tree
        ):
            raise MaterializationError(f"{name} nested checkout differs from its reviewed root")
        if name != "ipfs_accelerate_py":
            gitlink = _git("rev-parse", f"HEAD:{path.relative_to(ROOT).as_posix()}")
            if gitlink != nested_head:
                raise MaterializationError(f"{name} superproject gitlink differs from nested HEAD")
        nested_status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=path)
        if nested_status:
            raise MaterializationError(f"{name} nested checkout is dirty")
        projection[name] = {
            "head": nested_head,
            "tree": nested_tree,
            "required_integration_head": required_head,
            "required_integration_tree": required_tree,
        }
        planning_repositories[name] = {"commit": required_head, "tree": required_tree}
    planning_forest = {
        "schema": "ExternalAgentSourceForest@1",
        "repositories": planning_repositories,
    }
    configured_forest_root = str(binding.get("source_forest_root") or "")
    if _cid(planning_forest) != configured_forest_root:
        raise MaterializationError("source_binding.source_forest_root differs from exact roots")
    projection["planning_source_forest_root"] = configured_forest_root
    projection["source_generation_cid"] = _cid(projection)
    return projection


def _validate_board() -> dict[str, Any]:
    validator = ROOT / "scripts/validate_external_agent_autonomous_execution_fabric_board.py"
    result = subprocess.run(
        [sys.executable, str(validator), "--check-all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise MaterializationError(
            "board validation failed: " + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MaterializationError("board validator did not emit JSON") from exc
    if not isinstance(report, dict) or report.get("valid") is not True:
        raise MaterializationError("board validator did not report valid=true")
    return report


def build_population(config: Mapping[str, Any]) -> dict[str, Any]:
    board_path = _relative_path(config.get("taskboard_json_path"), field="taskboard_json_path")
    source_path = _relative_path(
        config.get("source_reconciliation_manifest_path"),
        field="source_reconciliation_manifest_path",
    )
    stack_path = _relative_path(
        config.get("stack_compatibility_manifest_path"),
        field="stack_compatibility_manifest_path",
    )
    board = _load_object(board_path)
    source_generation = _source_generation(config)
    head = str(source_generation["ipfs_accelerate_py"]["head"])
    tree = str(source_generation["ipfs_accelerate_py"]["tree"])
    controls = {
        "board": _file_cid(board_path),
        "taskboard_markdown": _file_cid(_relative_path(config.get("taskboard_path"), field="taskboard_path")),
        "objectives": _file_cid(_relative_path(config.get("objectives_path"), field="objectives_path")),
        "plan": _file_cid(_relative_path(config.get("plan_path"), field="plan_path")),
        "source_manifest": _file_cid(source_path),
        "stack_manifest": _file_cid(stack_path),
        "config": _file_cid(CONFIG_PATH),
        "validator": _file_cid(_relative_path(config.get("validator_path"), field="validator_path")),
        "materializer": _file_cid(_relative_path(config.get("materializer_path"), field="materializer_path")),
        "materialization_attempt_history": _file_cid(
            _relative_path(
                config.get("materialization_attempt_history_path"),
                field="materialization_attempt_history_path",
            )
        ),
    }
    plan_root_cid = _cid(
        {
            "schema": "ExternalAgentFormalWorkPlanRoot@1",
            "plan_revision": board.get("plan_revision"),
            "board_cid": board.get("board_cid"),
            "controls": controls,
            "source_head": head,
            "repository_tree_id": tree,
            "source_generation_cid": source_generation["source_generation_cid"],
        }
    )
    raw_goals = board.get("goals")
    raw_tasks = board.get("tasks")
    initial_ids = board.get("initial_population_task_ids")
    if not isinstance(raw_goals, list) or not isinstance(raw_tasks, list) or not isinstance(initial_ids, list):
        raise MaterializationError("board goals/tasks/initial population are malformed")
    goal_cids = {
        str(goal["goal_id"]): _cid(
            {"schema": "EAAEFGoalIdentity@1", "goal": goal, "plan_root_cid": plan_root_cid}
        )
        for goal in raw_goals
        if isinstance(goal, Mapping)
    }
    goals: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    for ordinal, goal in enumerate(raw_goals, start=1):
        if not isinstance(goal, Mapping):
            raise MaterializationError("goal is not an object")
        goal_id = str(goal["goal_id"])
        parent = str(goal.get("parent_goal_id") or "")
        goals.append(
            {
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "goal_alias": goal_id,
                "title": str(goal.get("title") or goal_id),
                "ordinal": ordinal,
                "status": "open",
                "objective_id": "objective:eaaef-root" if goal_id == "EAAEF-G000" else "",
                "objective_alias": "EAAEF-G000",
                "parent_goal_cid": goal_cids[parent] if parent else "",
                "priority": "P0",
                "body": dict(goal),
            }
        )
        if parent:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in goal.get("dependencies") or ():
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[str(dependency)],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
    task_by_id = {
        str(task.get("stable_task_id") or ""): task
        for task in raw_tasks
        if isinstance(task, Mapping) and str(task.get("stable_task_id") or "")
    }
    normalized_initial_ids = [str(item) for item in initial_ids]
    if len(normalized_initial_ids) != len(set(normalized_initial_ids)):
        raise MaterializationError("initial task population contains duplicate identities")
    missing_initial = [task_id for task_id in normalized_initial_ids if task_id not in task_by_id]
    if missing_initial:
        raise MaterializationError(f"initial task population is missing tasks: {missing_initial}")
    selected = [task_by_id[task_id] for task_id in normalized_initial_ids]
    task_cids = {
        str(task["stable_task_id"]): _cid(
            {
                "schema": "EAAEFTaskIdentity@1",
                "task_spec_cid": task.get("task_spec_cid"),
                "plan_root_cid": plan_root_cid,
                "source_head": head,
                "repository_tree_id": tree,
            }
        )
        for task in selected
    }
    tasks: list[dict[str, Any]] = []
    for ordinal, task in enumerate(selected, start=1):
        task_id = str(task["stable_task_id"])
        dependencies = [str(item) for item in task.get("dependencies") or ()]
        if any(item not in task_cids for item in dependencies):
            raise MaterializationError(f"{task_id} has a dependency outside the initial population")
        execution_owned_files = task.get("execution_owned_files")
        if (
            not isinstance(execution_owned_files, list)
            or not execution_owned_files
            or any(not isinstance(item, str) or not item for item in execution_owned_files)
        ):
            raise MaterializationError(
                f"{task_id} has no canonical accelerator-root execution_owned_files"
            )
        raw_execution_validation = task.get("execution_validation")
        if not isinstance(raw_execution_validation, list) or not raw_execution_validation:
            raise MaterializationError(
                f"{task_id} has no canonical accelerator-root execution_validation"
            )
        execution_validation: list[dict[str, Any]] = []
        for validation_index, item in enumerate(raw_execution_validation):
            if not isinstance(item, Mapping):
                raise MaterializationError(
                    f"{task_id} execution_validation[{validation_index}] is not an object"
                )
            working_directory = str(item.get("working_directory") or "")
            raw_argv = item.get("argv")
            if (
                not working_directory
                or Path(working_directory).is_absolute()
                or ".." in Path(working_directory).parts
                or not isinstance(raw_argv, list)
                or not raw_argv
                or any(not isinstance(part, str) or not part for part in raw_argv)
            ):
                raise MaterializationError(
                    f"{task_id} execution_validation[{validation_index}] is not bounded cwd/argv"
                )
            execution_validation.append(
                {"working_directory": working_directory, "argv": list(raw_argv)}
            )
        body = dict(task)
        body.update(
            {
                "task_id": task_id,
                "task_alias": task_id,
                "base_revision": str(
                    ((task.get("source_revisions") or {}).get(task["owning_repository"]) or {}).get("commit")
                    or ""
                ),
                "base_repository_tree_id": str(
                    ((task.get("source_revisions") or {}).get(task["owning_repository"]) or {}).get("tree")
                    or ""
                ),
                "accepted_plan_root_cid": plan_root_cid,
                "completion": task.get("completion_mode"),
                "review_only": False,
                "predicted_files": list(execution_owned_files),
                "depends_on": dependencies,
            }
        )
        tasks.append(
            {
                **body,
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "goal_cid": goal_cids[str(task["subgoal_id"])],
                "plan_cid": plan_root_cid,
                "objective_id": "objective:eaaef-root",
                "ordinal": ordinal,
                "status": "todo",
                "priority": str(task.get("priority") or "P0"),
                "title": str(task.get("title") or task_id),
                "dependencies": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": str(path),
                        "effect_id": _cid({"task": task_id, "path": str(path)}),
                    }
                    for path in execution_owned_files
                ],
                "acceptance": [str(task.get("acceptance") or "")],
                "validations": execution_validation,
            }
        )
    ready_task_aliases = [
        str(task["task_alias"])
        for task in tasks
        if not list(task.get("dependencies") or ())
    ]
    initial_projection = config.get("initial_projection")
    expected_initial_projection = {
        "task_count": len(tasks),
        "goal_count": len(goals),
        "completed_task_ids": [],
        "ready_task_ids": ready_task_aliases,
        "terminal_bootstrap_task_id": "EAAEF-009",
        "future_task_count": len(raw_tasks) - len(tasks),
        "future_tasks_materialized": False,
    }
    if initial_projection != expected_initial_projection:
        raise MaterializationError(
            "scheduler initial_projection differs from the exact board bootstrap population"
        )
    population = {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "source_generation": source_generation,
        "plan_root_cid": plan_root_cid,
        "controls": controls,
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root_cid,
                "plan_alias": str(board["plan_revision"]),
                "goal_cid": goal_cids["EAAEF-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
                "body": {
                    "board_cid": board["board_cid"],
                    "future_population_rule": board["future_population_rule"],
                    "future_task_count": len(raw_tasks) - len(tasks),
                },
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
        "initial_task_aliases": normalized_initial_ids,
        "ready_task_aliases": ready_task_aliases,
        "initial_task_count": len(tasks),
        "goal_count": len(goals),
        "future_task_count": len(raw_tasks) - len(tasks),
    }
    population["population_cid"] = _cid(population)
    return population


def _read_only_connection(path: Path) -> Any:
    if not path.is_file():
        raise MaterializationError(f"authority database does not exist: {path}")
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise MaterializationError("DuckDB is unavailable") from exc
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        connect_duckdb_with_policy,
    )

    try:
        return connect_duckdb_with_policy(
            duckdb,
            path,
            read_only=True,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError(f"unable to open authority read-only: {path}") from exc


def _control_schema_projection(path: Path) -> dict[str, Any]:
    """Project the installed operational profile through a read-only handle."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        compute_schema_fingerprint,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        BOOKKEEPING_TABLES,
        DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
        DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION,
        DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
        DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
        DATASETS_SEMANTIC_TRUTH_RELATIONS,
        DIAGNOSTIC_VIEWS,
        LEASE_IDENTITY_COLUMNS,
        TASK_IDENTITY_COLUMNS,
        load_datasets_authoritative_operational_catalog,
    )

    catalog = load_datasets_authoritative_operational_catalog()
    expected_migration = catalog.get(DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION)
    connection = _read_only_connection(path)
    try:
        relations = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        required = set(BOOKKEEPING_TABLES).union(
            DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
            DIAGNOSTIC_VIEWS,
        )
        missing = sorted(required - relations)
        forbidden = sorted(relations.intersection(DATASETS_SEMANTIC_TRUTH_RELATIONS))
        if missing or forbidden:
            raise MaterializationError(
                "datasets-authoritative operational profile relation mismatch: "
                f"missing={missing}, forbidden={forbidden}"
            )
        migration = connection.execute(
            "SELECT migration_id, checksum FROM schema_migrations WHERE version = ?",
            [DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION],
        ).fetchone()
        if migration is None or tuple(str(value) for value in migration) != (
            DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
            expected_migration.checksum,
        ):
            raise MaterializationError("operational-profile migration identity/checksum mismatch")
        contract = connection.execute(
            "SELECT payload_schema, description FROM schema_contracts "
            "WHERE contract_id = "
            "'contract:DatasetsAuthoritativeOperationalControlPlane@1'"
        ).fetchone()
        if contract is None:
            raise MaterializationError("operational-profile authority contract is missing")
        payload_schema, description = (str(contract[0]), str(contract[1]))
        if (
            payload_schema != DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA
            or "operational" not in description.lower()
            or "ipfs_datasets_py" not in description
        ):
            raise MaterializationError("operational-profile authority contract drifted")

        def columns(table: str) -> set[str]:
            return {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            }

        missing_task_columns = sorted(set(TASK_IDENTITY_COLUMNS) - columns("tasks"))
        missing_lease_columns = sorted(set(LEASE_IDENTITY_COLUMNS) - columns("leases"))
        if missing_task_columns or missing_lease_columns:
            raise MaterializationError(
                "operational-profile identity columns are missing: "
                f"tasks={missing_task_columns}, leases={missing_lease_columns}"
            )
        projection = {
            "valid": True,
            "database_path": str(path),
            "profile_id": DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
            "profile_schema": DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
            "catalog_fingerprint": catalog.fingerprint(),
            "migration_id": DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
            "migration_checksum": expected_migration.checksum,
            "schema_fingerprint": compute_schema_fingerprint(connection),
            "required_relations": sorted(required),
            "forbidden_relations": forbidden,
            "task_identity_columns": sorted(TASK_IDENTITY_COLUMNS),
            "lease_identity_columns": sorted(LEASE_IDENTITY_COLUMNS),
            "authority_contract": {
                "payload_schema": payload_schema,
                "operational_authority": "ipfs_accelerate_py",
                "semantic_and_proof_authority": "ipfs_datasets_py",
            },
            "connection_mode": "read_only",
        }
    finally:
        connection.close()
    projection["projection_root"] = _cid(projection)
    return projection


def _control_projection(path: Path) -> dict[str, Any]:
    connection = _read_only_connection(path)
    try:
        objectives = [
            {
                "objective_id": str(row[0]),
                "objective_alias": str(row[1]),
                "parent_objective_id": str(row[2]),
                "title": str(row[3]),
                "status": str(row[4]),
                "priority": str(row[5]),
                "revision": int(row[6]),
                "body": json.loads(row[7]),
            }
            for row in connection.execute(
                "SELECT objective_id, objective_alias, parent_objective_id, title, "
                "status, priority, revision, body_json FROM objectives "
                "ORDER BY objective_id"
            ).fetchall()
        ]
        tasks = [
            {
                "task_cid": str(row[0]),
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "plan_cid": str(row[3]),
                "objective_id": str(row[4]),
                "ordinal": int(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "priority": str(row[8]),
                "identity": json.loads(row[9]),
                "body": json.loads(row[10]),
            }
            for row in connection.execute(
                "SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, ordinal, "
                "status, revision, priority, identity_json, body_json "
                "FROM tasks ORDER BY ordinal"
            ).fetchall()
        ]
        goals = [
            {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2]),
                "parent_goal_cid": str(row[3]),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "body": json.loads(row[8]),
            }
            for row in connection.execute(
                "SELECT goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal, "
                "title, status, revision, body_json "
                "FROM goals ORDER BY ordinal"
            ).fetchall()
        ]
        dependencies = [
            {"task_cid": str(row[0]), "dependency_task_cid": str(row[1]), "kind": str(row[2])}
            for row in connection.execute(
                "SELECT task_cid, dependency_task_cid, kind FROM task_dependencies "
                "ORDER BY task_cid, dependency_task_cid, kind"
            ).fetchall()
        ]
        goal_edges = [
            {
                "parent_goal_cid": str(row[0]),
                "child_goal_cid": str(row[1]),
                "edge_kind": str(row[2]),
            }
            for row in connection.execute(
                "SELECT parent_goal_cid, child_goal_cid, edge_kind FROM goal_edges "
                "ORDER BY parent_goal_cid, child_goal_cid, edge_kind"
            ).fetchall()
        ]
        task_outputs = [
            {
                "task_cid": str(row[0]),
                "ordinal": int(row[1]),
                "path": str(row[2]),
                "effect": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, ordinal, path, effect_json FROM task_outputs "
                "ORDER BY task_cid, ordinal"
            ).fetchall()
        ]
        task_acceptance = [
            {
                "task_cid": str(row[0]),
                "ordinal": int(row[1]),
                "criterion": str(row[2]),
                "evidence_policy": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, ordinal, criterion, evidence_policy_json "
                "FROM task_acceptance ORDER BY task_cid, ordinal"
            ).fetchall()
        ]
        task_validations = [
            {
                "task_cid": str(row[0]),
                "ordinal": int(row[1]),
                "argv": json.loads(row[2]),
                "policy": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, ordinal, argv_json, policy_json "
                "FROM task_validations ORDER BY task_cid, ordinal"
            ).fetchall()
        ]
        plans = [
            {
                "plan_cid": str(row[0]),
                "goal_cid": str(row[1]),
                "plan_alias": str(row[2]),
                "status": str(row[3]),
                "revision": int(row[4]),
                "body": json.loads(row[5]),
            }
            for row in connection.execute(
                "SELECT plan_cid, goal_cid, plan_alias, status, revision, body_json "
                "FROM plans ORDER BY plan_cid"
            ).fetchall()
        ]
        objective_revisions = [
            {
                "objective_id": str(row[0]),
                "revision": int(row[1]),
                "status": str(row[2]),
                "body": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT objective_id, revision, status, body_json "
                "FROM objective_revisions ORDER BY objective_id, revision"
            ).fetchall()
        ]
        plan_revisions = [
            {
                "plan_cid": str(row[0]),
                "revision": int(row[1]),
                "body": json.loads(row[2]),
            }
            for row in connection.execute(
                "SELECT plan_cid, revision, body_json "
                "FROM plan_revisions ORDER BY plan_cid, revision"
            ).fetchall()
        ]
        task_revisions = [
            {
                "task_cid": str(row[0]),
                "revision": int(row[1]),
                "status": str(row[2]),
                "body": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, revision, status, body_json "
                "FROM task_revisions ORDER BY task_cid, revision"
            ).fetchall()
        ]
        # Bind every bootstrap-owned control relation, including revision and
        # materialization history that the ergonomic projections above omit.
        # The table allowlist is closed and identifiers are not caller input.
        exact_relation_names = (
            "control_plane_metadata",
            "schema_migrations",
            "schema_migration_attempts",
            "schema_contracts",
            "store_generations",
            "objectives",
            "objective_revisions",
            "goals",
            "goal_edges",
            "plans",
            "plan_revisions",
            "planning_decisions",
            "plan_candidates",
            "tasks",
            "task_revisions",
            "task_dependencies",
            "task_outputs",
            "task_acceptance",
            "task_validations",
            "artifacts",
            "completion_receipts",
        )
        exact_relations: dict[str, Any] = {}
        for table_name in exact_relation_names:
            columns = [
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]
            rows = [
                [
                    value
                    if value is None or isinstance(value, (bool, int, float, str))
                    else str(value)
                    for value in row
                ]
                for row in connection.execute(
                    f'SELECT * FROM "{table_name}" ORDER BY ALL'
                ).fetchall()
            ]
            exact_relations[table_name] = {"columns": columns, "rows": rows}
    finally:
        connection.close()
    projection = {
        "objectives": objectives,
        "goals": goals,
        "goal_edges": goal_edges,
        "plans": plans,
        "tasks": tasks,
        "dependencies": dependencies,
        "task_outputs": task_outputs,
        "task_acceptance": task_acceptance,
        "task_validations": task_validations,
        "objective_revisions": objective_revisions,
        "plan_revisions": plan_revisions,
        "task_revisions": task_revisions,
        "exact_relations": exact_relations,
    }
    projection["projection_root"] = _cid(projection)
    return projection


def _expected_population_projection(population: Mapping[str, Any]) -> dict[str, Any]:
    """Project the admitted input through the canonical repository boundary.

    This is intentionally independent of the rows read back from DuckDB.  It
    mirrors the documented DatabaseTaskSource/IntentRepository normalization
    so a buggy or fault-injected initial write cannot become its own oracle and
    be sealed merely because the post-write projection is internally stable.
    """

    source_goals = [
        dict(item)
        for item in population.get("objectives") or ()
        if isinstance(item, Mapping)
    ]
    objectives: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    for index, item in enumerate(source_goals):
        objective_id = str(item.get("objective_id") or "")
        if objective_id:
            objectives.append(
                {
                    "objective_id": objective_id,
                    "objective_alias": str(item.get("objective_alias") or objective_id),
                    "parent_objective_id": "",
                    "title": str(item.get("title") or objective_id),
                    "status": str(item.get("status") or "open").lower(),
                    "priority": str(item.get("priority") or "P2"),
                    "revision": 1,
                    "body": {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "objective_id",
                            "objective_alias",
                            "title",
                            "status",
                            "priority",
                        }
                    },
                }
            )
        goal_cid = str(item.get("goal_cid") or item.get("goal_id") or f"goal:cid:{index + 1}")
        goals.append(
            {
                "goal_cid": goal_cid,
                "goal_alias": str(item.get("goal_alias") or item.get("goal_id") or goal_cid),
                "objective_id": objective_id,
                "parent_goal_cid": str(item.get("parent_goal_cid") or ""),
                "ordinal": int(item.get("ordinal") or index + 1),
                "title": str(item.get("title") or item.get("goal_alias") or goal_cid),
                "status": str(item.get("status") or "open").lower(),
                "revision": 1,
                "body": {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "goal_cid",
                        "goal_id",
                        "goal_alias",
                        "title",
                        "status",
                        "ordinal",
                        "objective_id",
                    }
                },
            }
        )

    goal_edges = sorted(
        (
            {
                "parent_goal_cid": str(item.get("parent_goal_cid") or item.get("parent") or ""),
                "child_goal_cid": str(item.get("child_goal_cid") or item.get("child") or ""),
                "edge_kind": str(item.get("edge_kind") or "goal_dependency"),
            }
            for item in population.get("goal_edges") or ()
            if isinstance(item, Mapping)
        ),
        key=lambda item: (
            item["parent_goal_cid"],
            item["child_goal_cid"],
            item["edge_kind"],
        ),
    )
    plans = sorted(
        (
            {
                "plan_cid": str(item.get("plan_cid") or item.get("plan_id") or ""),
                "goal_cid": str(item.get("goal_cid") or ""),
                "plan_alias": str(item.get("plan_alias") or item.get("alias") or item.get("plan_cid") or ""),
                "status": str(item.get("status") or "active").lower(),
                "revision": 1,
                "body": dict(item),
            }
            for item in population.get("plans") or ()
            if isinstance(item, Mapping)
        ),
        key=lambda item: item["plan_cid"],
    )

    tasks: list[dict[str, Any]] = []
    dependencies: list[dict[str, str]] = []
    task_outputs: list[dict[str, Any]] = []
    task_acceptance: list[dict[str, Any]] = []
    task_validations: list[dict[str, Any]] = []
    tree_id = str(population.get("repository_tree_id") or "tree:unknown")
    source_tasks = [
        dict(item)
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    ]
    task_cids_by_alias = {
        str(item.get("task_id") or item.get("task_alias") or item.get("alias") or task_cid): task_cid
        for index, item in enumerate(source_tasks)
        for task_cid in (
            str(item.get("task_cid") or item.get("cid") or f"task:cid:{index + 1}"),
        )
    }
    for index, raw_task in enumerate(source_tasks):
        if not isinstance(raw_task, Mapping):
            continue
        item = dict(raw_task)
        task_cid = str(item.get("task_cid") or item.get("cid") or f"task:cid:{index + 1}")
        task_alias = str(item.get("task_id") or item.get("task_alias") or item.get("alias") or task_cid)
        tasks.append(
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "goal_cid": str(item.get("goal_cid") or item.get("goal_id") or ""),
                "plan_cid": str(item.get("plan_cid") or population.get("plan_root_cid") or ""),
                "objective_id": str(item.get("objective_id") or ""),
                "ordinal": int(item.get("ordinal") or index + 1),
                "status": str(item.get("status") or "ready").lower(),
                "revision": 1,
                "priority": str(item.get("priority") or "P2"),
                "identity": {
                    "repository_tree_id": tree_id,
                    "task_alias": task_alias,
                    "task_cid": task_cid,
                },
                "body": {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "task_cid",
                        "task_id",
                        "task_alias",
                        "cid",
                        "goal_cid",
                        "goal_id",
                        "depends_on",
                        "dependencies",
                        "effects",
                        "outputs",
                        "acceptance_criteria",
                        "acceptance",
                        "validation_commands",
                        "validations",
                        "status",
                        "priority",
                        "ordinal",
                        "plan_cid",
                        "objective_id",
                    }
                },
            }
        )
        for dependency in item.get("depends_on") or item.get("dependencies") or ():
            dependency_text = str(dependency)
            dependencies.append(
                {
                    "task_cid": task_cid,
                    "dependency_task_cid": task_cids_by_alias.get(
                        dependency_text, dependency_text
                    ),
                    "kind": "depends_on",
                }
            )
        for ordinal, output in enumerate(item.get("effects") or item.get("outputs") or ()):
            if not isinstance(output, Mapping):
                continue
            effect = dict(output)
            task_outputs.append(
                {
                    "task_cid": task_cid,
                    "ordinal": ordinal,
                    "path": str(effect.get("path") or effect.get("effect_id") or f"output:{ordinal}"),
                    "effect": effect,
                }
            )
        for ordinal, acceptance in enumerate(
            item.get("acceptance_criteria") or item.get("acceptance") or ()
        ):
            if isinstance(acceptance, str):
                criterion = acceptance.strip()
                policy: dict[str, Any] = {"criterion": criterion}
            elif isinstance(acceptance, Mapping):
                policy = dict(acceptance)
                criterion = str(
                    policy.get("criterion")
                    or policy.get("statement")
                    or policy.get("criterion_key")
                    or f"criterion:{ordinal}"
                ).strip()
            else:
                continue
            task_acceptance.append(
                {
                    "task_cid": task_cid,
                    "ordinal": ordinal,
                    "criterion": criterion,
                    "evidence_policy": policy,
                }
            )
        for ordinal, validation in enumerate(
            item.get("validation_commands") or item.get("validations") or ()
        ):
            if isinstance(validation, str):
                argv = [validation]
                policy = {}
            elif isinstance(validation, Mapping):
                validation_map = dict(validation)
                raw_argv = validation_map.get("argv") or validation_map.get("validation_commands")
                if isinstance(raw_argv, str):
                    argv = [raw_argv]
                elif isinstance(raw_argv, list):
                    argv = [str(part) for part in raw_argv]
                else:
                    argv = [str(validation_map.get("command") or f"validation:{ordinal}")]
                policy = {
                    key: value
                    for key, value in validation_map.items()
                    if key not in {"argv", "validation_commands", "command"}
                }
            elif isinstance(validation, list):
                argv = [str(part) for part in validation]
                policy = {}
            else:
                continue
            task_validations.append(
                {
                    "task_cid": task_cid,
                    "ordinal": ordinal,
                    "argv": argv,
                    "policy": policy,
                }
            )

    sorted_objectives = sorted(objectives, key=lambda item: item["objective_id"])
    sorted_plans = plans
    sorted_tasks = sorted(tasks, key=lambda item: item["ordinal"])
    return {
        "objectives": sorted_objectives,
        "goals": sorted(goals, key=lambda item: item["ordinal"]),
        "goal_edges": goal_edges,
        "plans": sorted_plans,
        "tasks": sorted_tasks,
        "dependencies": sorted(
            dependencies,
            key=lambda item: (item["task_cid"], item["dependency_task_cid"], item["kind"]),
        ),
        "task_outputs": sorted(task_outputs, key=lambda item: (item["task_cid"], item["ordinal"])),
        "task_acceptance": sorted(task_acceptance, key=lambda item: (item["task_cid"], item["ordinal"])),
        "task_validations": sorted(task_validations, key=lambda item: (item["task_cid"], item["ordinal"])),
        "objective_revisions": [
            {
                "objective_id": item["objective_id"],
                "revision": item["revision"],
                "status": item["status"],
                "body": item["body"],
            }
            for item in sorted_objectives
        ],
        "plan_revisions": [
            {
                "plan_cid": item["plan_cid"],
                "revision": item["revision"],
                "body": item["body"],
            }
            for item in sorted_plans
        ],
        "task_revisions": sorted(
            (
                {
                    "task_cid": item["task_cid"],
                    "revision": item["revision"],
                    "status": item["status"],
                    "body": item["body"],
                }
                for item in sorted_tasks
            ),
            key=lambda item: (item["task_cid"], item["revision"]),
        ),
    }


def _assert_population_equivalent(
    population: Mapping[str, Any], control: Mapping[str, Any]
) -> None:
    expected = _expected_population_projection(population)
    observed = {key: control.get(key) for key in expected}
    if observed != expected:
        raise MaterializationError(
            "materialized control population differs from the admitted board projection"
        )


def _execution_projection(path: Path) -> dict[str, Any]:
    connection = _read_only_connection(path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        tracked = [
            name
            for name in (
                "database_task_attempts",
                "attempt_phases",
                "provider_call_intents",
                "effect_claims",
                "validation_intents",
            )
            if name in tables
        ]
        row_counts = {
            name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in tracked
        }
    finally:
        connection.close()
    projection = {"tracked_tables": tracked, "row_counts": row_counts}
    projection["projection_root"] = _cid(projection)
    return projection


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        os.fchmod(fd, 0o600)
        data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short receipt write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            # A same-directory hard link publishes the fully-fsynced bytes
            # atomically while retaining O_EXCL-style no-overwrite semantics.
            # Unlike os.replace(), a racing writer can never replace an
            # already-published immutable claim or receipt.
            os.link(temp, path)
        except FileExistsError as exc:
            raise MaterializationError(
                f"refusing to overwrite immutable record {path.relative_to(ROOT)}"
            ) from exc
        temp.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temp.unlink(missing_ok=True)


def materialize(config: Mapping[str, Any]) -> dict[str, Any]:
    _assert_clean()
    validation = _validate_board()
    population = build_population(config)
    paths = _paths(config)
    claim_path = _claim_path(config)
    receipt_path = _receipt_path(config)
    existing = [path for path in _namespace_artifacts(config) if path.exists()]
    if existing:
        raise MaterializationError(
            "refusing to overwrite existing bootstrap namespace: "
            + ", ".join(path.relative_to(ROOT).as_posix() for path in existing)
        )
    namespace_claim: dict[str, Any] = {
        "schema": NAMESPACE_CLAIM_SCHEMA,
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "source_head": population["source_head"],
        "source_tree": population["repository_tree_id"],
        "source_generation_cid": population["source_generation"]["source_generation_cid"],
        "store_generation": str(
            (config.get("database_program") or {}).get("store_generation") or ""
        ),
        "database_paths": {
            name: path.relative_to(ROOT).as_posix() for name, path in paths.items()
        },
        "maximum_writer_processes": 1,
        "partial_effect_policy": (
            "preserve claim and every created file; advance to a new explicit "
            "store generation after any failed attempt"
        ),
        "process_started": False,
    }
    namespace_claim["claim_cid"] = _cid(namespace_claim)
    _write_json_immutable(claim_path, namespace_claim)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
            install_datasets_authoritative_operational_schema,
            verify_datasets_authoritative_operational_schema,
        )
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            DatabaseImplementationDaemon,
        )

        schema_install = install_datasets_authoritative_operational_schema(
            paths["control"],
            application_version="0.0.45",
            tool_version="1.5.2",
            owner_id="eaaef-materializer:embedded-single-writer",
        )
        prior_revision = os.environ.get("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION")
        os.environ["IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"] = str(
            (config.get("database_program") or {}).get("schema_revision")
        )
        daemon = None
        try:
            daemon = DatabaseImplementationDaemon(
                database_path=paths["control"],
                coordination_path=paths["coordination"],
                execution_path=paths["execution"],
                owner_session_id="eaaef-materializer:embedded-single-writer",
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
                os.environ.pop("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION", None)
            else:
                os.environ["IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"] = prior_revision
        schema_verification = verify_datasets_authoritative_operational_schema(paths["control"])
        control_schema = _control_schema_projection(paths["control"])
        control = _control_projection(paths["control"])
        from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
            read_coordination_registry_projection,
        )

        coordination = read_coordination_registry_projection(paths["coordination"])
        execution = _execution_projection(paths["execution"])
        _assert_population_equivalent(population, control)
        expected_aliases = list(population["task_cids_by_alias"])
        if [item["task_alias"] for item in control["tasks"]] != expected_aliases:
            raise MaterializationError("control task aliases differ from initial population")
        if any(item["status"] != "todo" for item in control["tasks"]):
            raise MaterializationError("fresh control tasks are not all todo")
        if any(value != 0 for value in execution["row_counts"].values()):
            raise MaterializationError("fresh execution store contains attempt/effect history")
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "namespace_claim_cid": namespace_claim["claim_cid"],
            "authority_mode": "embedded",
            "maximum_writer_processes": 1,
            "continuous_quack_authority": False,
            "ducklake_authority": False,
            "board_validation": validation,
            "population_cid": population["population_cid"],
            "plan_root_cid": population["plan_root_cid"],
            "source_head": population["source_head"],
            "source_tree": population["repository_tree_id"],
            "source_generation": population["source_generation"],
            "controls": population["controls"],
            "database_paths": dict(namespace_claim["database_paths"]),
            "schema_install": schema_install.to_dict(),
            "schema_verification": dict(schema_verification),
            "control_schema_projection": control_schema,
            "database_materialization": dict(database_receipt),
            "control_projection": control,
            "coordination_projection": coordination,
            "execution_projection": execution,
            "ready_task_aliases": list(population["ready_task_aliases"]),
            "process_started": False,
        }
        receipt["receipt_cid"] = _cid(receipt)
        _write_json_immutable(receipt_path, receipt)
        return receipt
    except Exception as exc:
        if isinstance(exc, MaterializationError):
            detail = str(exc)
        else:
            detail = f"{type(exc).__name__}: {exc}"
        raise MaterializationError(
            "bootstrap namespace claim is immutable and partial effects are preserved; "
            "advance to a new explicit store generation after review; "
            f"claim={claim_path.relative_to(ROOT)}, failure={detail}"
        ) from exc


def verify(config: Mapping[str, Any]) -> dict[str, Any]:
    validation = _validate_board()
    population = build_population(config)
    paths = _paths(config)
    claim_path = _claim_path(config)
    receipt_path = _receipt_path(config)
    claim = _load_object(claim_path)
    claim_projection = dict(claim)
    claim_cid = str(claim_projection.pop("claim_cid", ""))
    if claim_cid != _cid(claim_projection):
        raise MaterializationError("bootstrap namespace claim self-address is invalid")
    receipt = _load_object(receipt_path)
    receipt_projection = dict(receipt)
    receipt_cid = str(receipt_projection.pop("receipt_cid", ""))
    if receipt_cid != _cid(receipt_projection):
        raise MaterializationError("bootstrap receipt self-address is invalid")
    for key in ("population_cid", "plan_root_cid", "source_head", "source_tree", "controls"):
        expected = population[
            "repository_tree_id" if key == "source_tree" else key
        ]
        if receipt.get(key) != expected:
            raise MaterializationError(f"bootstrap receipt {key} differs from current source")
    expected_paths = {
        name: path.relative_to(ROOT).as_posix() for name, path in paths.items()
    }
    claim_expectations = {
        "schema": NAMESPACE_CLAIM_SCHEMA,
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "source_head": population["source_head"],
        "source_tree": population["repository_tree_id"],
        "source_generation_cid": population["source_generation"]["source_generation_cid"],
        "store_generation": str(
            (config.get("database_program") or {}).get("store_generation") or ""
        ),
        "database_paths": expected_paths,
        "maximum_writer_processes": 1,
        "process_started": False,
    }
    for key, expected in claim_expectations.items():
        if claim.get(key) != expected:
            raise MaterializationError(f"bootstrap namespace claim {key} differs from current source")
    if receipt.get("namespace_claim_cid") != claim_cid:
        raise MaterializationError("bootstrap receipt is not bound to the namespace claim")
    if receipt.get("database_paths") != expected_paths:
        raise MaterializationError("bootstrap receipt database paths differ from config")
    if receipt.get("source_generation") != population["source_generation"]:
        raise MaterializationError("bootstrap receipt source generation differs from current source")
    receipt_expectations = {
        "schema": RECEIPT_SCHEMA,
        "authority_mode": "embedded",
        "maximum_writer_processes": 1,
        "continuous_quack_authority": False,
        "ducklake_authority": False,
        "ready_task_aliases": list(population["ready_task_aliases"]),
        "process_started": False,
    }
    for key, expected in receipt_expectations.items():
        if receipt.get(key) != expected:
            raise MaterializationError(f"bootstrap receipt {key} violates bootstrap policy")
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )

    control_schema = _control_schema_projection(paths["control"])
    control = _control_projection(paths["control"])
    coordination = read_coordination_registry_projection(paths["coordination"])
    execution = _execution_projection(paths["execution"])
    if control != receipt.get("control_projection"):
        raise MaterializationError("control authority differs from materialization receipt")
    if coordination != receipt.get("coordination_projection"):
        raise MaterializationError("coordination authority differs from materialization receipt")
    if execution != receipt.get("execution_projection"):
        raise MaterializationError("execution authority differs from materialization receipt")
    if control_schema != receipt.get("control_schema_projection"):
        raise MaterializationError("control schema differs from materialization receipt")
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-verification@1",
        "valid": True,
        "verification_mode": "read_only",
        "namespace_claim_cid": claim_cid,
        "receipt_cid": receipt_cid,
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "board_validation": validation,
        "control_projection_root": control["projection_root"],
        "coordination_projection_root": coordination["projection_root"],
        "execution_projection_root": execution["projection_root"],
        "process_started": False,
    }


def launch_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    report = verify(config)
    policy = config.get("launch_policy") or {}
    program = config.get("database_program") or {}
    _paths(config)
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        DatabaseProgramConfig,
        DatabaseProgramConfigError,
    )

    try:
        database_program = DatabaseProgramConfig.from_mapping(program)
    except DatabaseProgramConfigError as exc:
        raise MaterializationError(f"invalid DatabaseProgramConfig: {exc}") from exc
    if database_program.authority_mode != "embedded":
        raise MaterializationError("bootstrap launch plan must retain embedded authority")
    command = [
        sys.executable,
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        "--scheduler-config",
        CONFIG_PATH.relative_to(ROOT).as_posix(),
        "--todo-path",
        str(config["taskboard_path"]),
        "--task-prefix",
        str(config["task_prefix"]),
        *database_program.cli_args(),
        "--state-dir",
        str(program["state_dir"]),
        "--worktree-root",
        str(program["worktree_root"]),
        "--merge-queue-dir",
        str(program["merge_queue_dir"]),
        "--merge-target-branch",
        str(config["merge_target_branch"]),
        "--strict-task-sharding",
        "--task-shard-count",
        "1",
        "--task-shard-index",
        "0",
        "--implement",
    ]
    for protected in config.get("protected_paths") or ():
        protected_path = _relative_path(
            protected,
            field="protected_paths[]",
        )
        command.extend(
            ["--implementation-protected-path", protected_path.relative_to(ROOT).as_posix()]
        )
    for submodule in config.get("worktree_submodule_paths") or ():
        submodule_path = _relative_path(
            submodule,
            field="worktree_submodule_paths[]",
        )
        command.extend(
            ["--worktree-submodule-path", submodule_path.relative_to(ROOT).as_posix()]
        )
    blockers = [str(item) for item in policy.get("blockers") or () if str(item)]
    if report.get("board_validation", {}).get("live_launch_allowed") is not True:
        blockers.append("board validation has not admitted live launch")
    container = dict(config.get("container_policy") or {})
    if container.get("live_dispatch_allowed") is not True:
        blockers.append("container_policy.live_dispatch_allowed is not true")
    if str(container.get("bootstrap_image_status") or "") != "admitted":
        blockers.append("container_policy.bootstrap_image_status is not admitted")
    image_digest = str(container.get("bootstrap_image_digest") or "")
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        blockers.append("container_policy.bootstrap_image_digest is not a full sha256 identity")
    # EAAEF-000 is a manual, independently reviewed admission task.  This
    # checkpoint deliberately contains no positive production path: a later
    # immutable plan revision must add and verify the signed route/image/SBOM
    # receipt at the actual OCI process-birth boundary.
    blockers.append("trusted EAAEF-000 launch admission is not implemented")
    blockers = list(dict.fromkeys(blockers))
    requested = policy.get("live_single_supervisor_allowed") is True
    allowed = bool(requested and not blockers)
    # A no-go report must not double as a copy/paste executable command.  Keep
    # the reviewed candidate identity for diagnostics, but expose an argv only
    # after every launch gate has admitted it.
    executable_argv = command if allowed else []
    exposed_candidate_argv = command if allowed else []
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-launch-plan@1",
        "allowed": allowed,
        "blockers": blockers,
        "argv": executable_argv,
        "argv_cid": _cid(executable_argv),
        "candidate_argv": exposed_candidate_argv,
        "candidate_argv_cid": _cid(command),
        "candidate_argv_length": len(command),
        "candidate_executable_withheld": not allowed,
        "execution_prohibited": not allowed,
        "materialization_receipt_cid": report["receipt_cid"],
        "container_policy": container,
        "process_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "materialize", "verify", "launch-plan"))
    args = parser.parse_args(argv)
    try:
        config = _load_object(CONFIG_PATH)
        if args.command == "build":
            result = build_population(config)
        elif args.command == "materialize":
            result = materialize(config)
        elif args.command == "verify":
            result = verify(config)
        else:
            result = launch_plan(config)
    except MaterializationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
