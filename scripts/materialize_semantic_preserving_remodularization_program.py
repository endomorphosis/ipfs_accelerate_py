#!/usr/bin/env python3
"""Materialize and operate the SPAR DuckDB + Quack + DuckLake control plane.

The authority split is deliberately narrow:

* ``DatabaseTaskSource@1`` over DuckDB is transactional task/goal authority.
* one fenced loopback Quack process exclusively owns the DuckDB file while
  supervisors are running;
* DuckLake is an optional, rebuildable history projection and is never read by
  readiness, completion, promotion, or release gates.

The Markdown plan, objectives, and task board are immutable bootstrap inputs.
This operator never mutates their status fields and never publishes the raw
Quack authentication token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import stat as stat_module
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG: Final = Path(
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"
)
RUNTIME_RELATIVE: Final = Path(
    "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
)
BOOTSTRAP_RECEIPT_NAME: Final = "bootstrap-materialization.json"
SPAR000_QUALIFICATION_NAME: Final = "spar-000-qualification.json"
DUCKLAKE_RECEIPT_NAME: Final = "ducklake-history-projection.json"
OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "semantic-preserving-autonomous-remodularization-operator@1"
)
POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "semantic-preserving-autonomous-remodularization-population@1"
)
BOOTSTRAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "semantic-preserving-autonomous-remodularization-bootstrap@1"
)
DUCKLAKE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "semantic-preserving-autonomous-remodularization-ducklake-projection@1"
)
GOAL_RE: Final = re.compile(r"^## (SPAR-G\d{3}) (.+)$", re.MULTILINE)
QUACK_ENDPOINT_RE: Final = re.compile(
    r"^quack:(?://)?(127(?:\.\d{1,3}){3}|localhost):(\d{1,5})$",
    re.IGNORECASE,
)
READY_STATUSES: Final = (
    "proposed",
    "admitted",
    "pending",
    "ready",
    "todo",
    "queued",
    "retrying",
)
COMPLETED_STATUSES: Final = ("completed", "skipped", "complete", "done")
ACTIVE_STATUSES: Final = ("claimed", "in_progress", "running")
TERMINAL_STATUSES: Final = (
    *COMPLETED_STATUSES,
    "cancelled",
    "failed",
    "quarantined",
    "rejected",
)
OWNER_DML_PREFIXES: Final = (
    "UPDATE ",
    "DELETE ",
    "MERGE ",
    "INSERT OR REPLACE",
    "INSERT OR IGNORE",
)
INTERNAL_CLIENT_GRANT_TTL_SECONDS: Final = 86_400.0
INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS: Final = 43_200.0
BOOTSTRAP_READY_TIMEOUT_SECONDS: Final = 10.0
BOOTSTRAP_PROCESS_STOP_GRACE_SECONDS: Final = 35.0


class OperatorError(RuntimeError):
    """Fail-closed SPAR operator error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"JSON root must be an object: {path}")
    return value


def _safe_path(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise OperatorError(f"{field} must be a safe repository-relative path")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OperatorError(f"{field} escapes repository") from exc
    return resolved


def _git(*arguments: str, check: bool = True, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if check and completed.returncode != 0:
        error = completed.stderr or completed.stdout
        if isinstance(error, bytes):
            error = error.decode("utf-8", errors="replace")
        raise OperatorError(
            f"git {' '.join(arguments)} failed: {str(error).strip()}"
        )
    return completed.stdout


def _assert_clean_current_tree(config: Mapping[str, Any]) -> tuple[str, str]:
    status_output = str(
        _git("status", "--porcelain=v1", "--untracked-files=all")
    ).strip()
    if status_output:
        raise OperatorError(
            "refusing to materialize from a dirty worktree; commit the exact "
            "plan, board, configuration, validator, and operator first"
        )
    head = str(_git("rev-parse", "HEAD")).strip()
    tree = str(_git("rev-parse", "HEAD^{tree}")).strip()
    branch = str(_git("branch", "--show-current")).strip()
    required_branch = str(config.get("merge_target_branch") or "").strip()
    if required_branch and branch != required_branch:
        raise OperatorError(
            f"execution branch {branch!r} differs from configured branch "
            f"{required_branch!r}"
        )
    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    ancestor = str(binding.get("accelerator_required_ancestor") or "").strip()
    if ancestor:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise OperatorError("configured accelerator base is not an ancestor")
    return head, tree


def _tracked_bytes(path: Path, *, head: str) -> bytes:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise OperatorError(f"authority input escapes repository: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise OperatorError(f"authority input is not a regular file: {relative}")
    working = path.read_bytes()
    recorded = _git("show", f"{head}:{relative}", binary=True)
    if not isinstance(recorded, bytes) or working != recorded:
        raise OperatorError(f"authority input differs from current HEAD: {relative}")
    return working


def _source_forest(config: Mapping[str, Any], *, head: str) -> dict[str, Any]:
    """Verify configured sibling gitlinks without granting write authority."""

    binding = config.get("source_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    nested: list[dict[str, str]] = []
    configured_repositories = (
        (
            "ipfs_datasets",
            ("ipfs_datasets_submodule_path", "datasets_submodule_path"),
            ("ipfs_datasets_planning_revision", "datasets_planning_revision"),
        ),
        (
            "ipfs_kit",
            ("ipfs_kit_submodule_path", "kit_submodule_path"),
            ("ipfs_kit_planning_revision", "kit_planning_revision"),
        ),
        (
            "mcp_plus_plus",
            ("mcp_plus_plus_submodule_path",),
            ("mcp_plus_plus_planning_revision",),
        ),
    )
    for prefix, path_fields, revision_fields in configured_repositories:
        raw_path = next(
            (binding.get(field) for field in path_fields if binding.get(field)),
            None,
        )
        raw_revision = next(
            (binding.get(field) for field in revision_fields if binding.get(field)),
            None,
        )
        if raw_path in (None, "") and raw_revision in (None, ""):
            continue
        if raw_path in (None, "") or raw_revision in (None, ""):
            raise OperatorError(f"{prefix} source binding is incomplete")
        nested_path = _safe_path(
            ROOT,
            raw_path,
            field=f"source_binding.{prefix}_submodule_path",
        )
        if not nested_path.is_dir():
            raise OperatorError(f"{prefix} submodule is not initialized")
        nested_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if nested_status.returncode != 0 or nested_status.stdout.strip():
            raise OperatorError(f"{prefix} nested worktree is not clean")
        nested_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        nested_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        revision = nested_head.stdout.strip()
        tree = nested_tree.stdout.strip()
        if (
            nested_head.returncode != 0
            or nested_tree.returncode != 0
            or revision != str(raw_revision)
            or not tree
        ):
            raise OperatorError(f"{prefix} nested revision differs from its seal")
        relative = nested_path.relative_to(ROOT).as_posix()
        tree_row = str(_git("ls-tree", head, "--", relative)).strip().split()
        if (
            len(tree_row) < 3
            or tree_row[0] != "160000"
            or tree_row[1] != "commit"
            or tree_row[2] != revision
        ):
            raise OperatorError(f"{prefix} gitlink differs from its nested HEAD")
        nested.append(
            {
                "repository": prefix,
                "path": relative,
                "head": revision,
                "tree": tree,
                "access": "read_only_contract_audit",
            }
        )
    result: dict[str, Any] = {
        "source_head": head,
        "nested_repositories": nested,
        "cross_repository_writes": False,
    }
    result["source_forest_root"] = _identity(result)
    return result


def _load_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
    )

    board = load_configured_board(config_path, repo_root=ROOT)
    payload = dict(board.payload)
    if board.task_prefix.removeprefix("## ") != "SPAR-":
        raise OperatorError("SPAR operator requires task_prefix='SPAR-'")
    if board.board_namespace != "semantic-preserving-autonomous-remodularization-v1":
        raise OperatorError("scheduler board_namespace is not the SPAR v1 namespace")
    program = board.resolved_database_program()
    if program.authority_mode != "quack" or program.task_source_kind != "duckdb":
        raise OperatorError("SPAR requires DuckDB task authority served through Quack")
    if program.failover_policy != "fail_closed":
        raise OperatorError("SPAR Quack authority must fail closed")
    if QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint) is None:
        raise OperatorError("SPAR Quack endpoint must be a bounded loopback URI")
    return board, payload


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _goal_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        normalize_metadata_key,
    )

    matches = list(GOAL_RE.finditer(text))
    result: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[match.end() : end].splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", 1)
            normalized = normalize_metadata_key(key)
            if normalized in fields:
                raise OperatorError(
                    f"{match.group(1)} contains duplicate metadata field {normalized}"
                )
            fields[normalized] = value.strip()
        result.append((match.group(1), match.group(2).strip(), fields))
    return result


def _population(board: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
        split_validation_commands,
    )

    head, tree = _assert_clean_current_tree(config)
    source_forest = _source_forest(config, head=head)
    sources = {
        "config": _tracked_bytes(board.config_path, head=head),
        "taskboard": _tracked_bytes(board.path(board.taskboard_path), head=head),
        "objectives": _tracked_bytes(board.path(board.objectives_path), head=head),
        "plan": _tracked_bytes(board.path(board.plan_path), head=head),
        "validator": _tracked_bytes(board.path(board.validator_path), head=head),
    }
    plan_root = content_identity(
        {
            "schema": "spar-plan-root@1",
            "source_head": head,
            "repository_tree_id": tree,
            "sources": {
                name: _identity(value) for name, value in sorted(sources.items())
            },
        }
    )

    objective_text = sources["objectives"].decode("utf-8")
    parsed_goals = _goal_blocks(objective_text)
    if not parsed_goals or parsed_goals[0][0] != "SPAR-G000":
        raise OperatorError("objectives must begin with root SPAR-G000")
    if len({item[0] for item in parsed_goals}) != len(parsed_goals):
        raise OperatorError("objectives contain duplicate goal IDs")
    goal_cids = {
        goal_id: content_identity(
            {
                "goal_id": goal_id,
                "title": title,
                "metadata": fields,
                "plan_root_cid": plan_root,
            }
        )
        for goal_id, title, fields in parsed_goals
    }
    goals: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    observed_goals: set[str] = set()
    for ordinal, (goal_id, title, fields) in enumerate(parsed_goals, start=1):
        parent = str(fields.get("parent") or "").strip()
        if parent and parent not in observed_goals:
            raise OperatorError(f"{goal_id} parent must precede it: {parent}")
        dependencies = _split_csv(fields.get("depends_on"))
        unknown = [item for item in dependencies if item not in goal_cids]
        if unknown:
            raise OperatorError(f"{goal_id} has unknown goal dependencies: {unknown}")
        goal = {
            "goal_cid": goal_cids[goal_id],
            "goal_id": goal_id,
            "goal_alias": goal_id,
            "title": title,
            "ordinal": ordinal,
            "status": str(fields.get("status") or "open").lower(),
            "objective_id": "objective:spar-root" if goal_id == "SPAR-G000" else "",
            "objective_alias": "SPAR-G000",
            "priority": str(fields.get("priority") or "P0"),
            "body": dict(fields),
        }
        if parent:
            goal["parent_goal_cid"] = goal_cids[parent]
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in dependencies:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
        goals.append(goal)
        observed_goals.add(goal_id)

    task_text = sources["taskboard"].decode("utf-8")
    parsed_tasks = parse_todo_blocks(task_text, task_header_prefix="## SPAR-")
    if not parsed_tasks:
        raise OperatorError("task board contains no SPAR tasks")
    task_ids = [item[0] for item in parsed_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise OperatorError("task board contains duplicate SPAR task IDs")
    task_cids = {
        task_id: content_identity(
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "metadata": fields,
                "plan_root_cid": plan_root,
                "repository_tree_id": tree,
            }
        )
        for task_id, title, source_line, fields in parsed_tasks
    }
    tasks: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    for ordinal, (task_id, title, source_line, fields) in enumerate(
        parsed_tasks, start=1
    ):
        dependencies = _split_csv(fields.get("depends_on"))
        unknown = [item for item in dependencies if item not in task_cids]
        if unknown:
            raise OperatorError(f"{task_id} has unknown dependencies: {unknown}")
        future = [item for item in dependencies if item not in observed_tasks]
        if future:
            raise OperatorError(
                f"{task_id} dependencies must precede it for atomic ingestion: {future}"
            )
        goal_id = str(
            fields.get("subgoal_id")
            or fields.get("goal_id")
            or fields.get("goal")
            or "SPAR-G000"
        ).strip()
        if goal_id not in goal_cids:
            raise OperatorError(f"{task_id} refers to unknown goal {goal_id}")
        output_paths = _split_csv(fields.get("outputs") or fields.get("predicted_files"))
        task = dict(fields)
        task.update(
            {
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "title": title,
                "source_line": source_line,
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "plan_cid": plan_root,
                "objective_id": "objective:spar-root",
                "ordinal": ordinal,
                "status": str(fields.get("status") or "todo").lower(),
                "priority": str(fields.get("priority") or "P1"),
                "dependencies": [task_cids[item] for item in dependencies],
                "depends_on": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"task_cid": task_cids[task_id], "path": path}
                        ),
                    }
                    for path in output_paths
                ],
                "acceptance": [
                    str(fields.get("acceptance") or fields.get("acceptance_subset") or "")
                ],
                "validations": list(
                    split_validation_commands(str(fields.get("validation") or ""))
                ),
                "accepted_plan_root_cid": plan_root,
                "base_revision": head,
                "base_repository_tree_id": tree,
                "owning_repository": str(
                    fields.get("owning_repository") or "ipfs_accelerate_py"
                ),
            }
        )
        tasks.append(task)
        observed_tasks.add(task_id)

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    expected_tasks = projection.get("task_count")
    expected_goals = projection.get("goal_count")
    expected_dependencies = projection.get("task_dependency_count")
    if expected_tasks is not None and int(expected_tasks) != len(tasks):
        raise OperatorError("task count differs from configured initial projection")
    if expected_goals is not None and int(expected_goals) != len(goals):
        raise OperatorError("goal count differs from configured initial projection")
    dependency_count = sum(
        len(_split_csv(item[3].get("depends_on"))) for item in parsed_tasks
    )
    if expected_dependencies is not None and int(expected_dependencies) != dependency_count:
        raise OperatorError(
            "task dependency count differs from configured initial projection"
        )
    return {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "source_identities": {
            name: _identity(value) for name, value in sorted(sources.items())
        },
        "source_forest": source_forest,
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "SPAR-PLAN-R1",
                "goal_cid": goal_cids["SPAR-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _runtime_paths(board: Any) -> dict[str, Path]:
    program = board.resolved_database_program()
    database = _safe_path(ROOT, program.store_id, field="database_program.store_id")
    runtime = board.path(board.runtime_paths["root"])
    try:
        database.relative_to(runtime)
    except ValueError as exc:
        raise OperatorError("DuckDB authority store must be below runtime_paths.root") from exc
    raw_runtime = board.payload.get("runtime_paths")
    raw_runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    evidence = _safe_path(
        ROOT,
        raw_runtime.get("evidence") or runtime.relative_to(ROOT) / "evidence",
        field="runtime_paths.evidence",
    )
    owner = _safe_path(
        ROOT,
        raw_runtime.get("quack_owner") or runtime.relative_to(ROOT) / "quack-owner",
        field="runtime_paths.quack_owner",
    )
    raw_ducklake = board.payload.get("ducklake_projection_program")
    raw_ducklake = raw_ducklake if isinstance(raw_ducklake, Mapping) else {}
    ducklake_catalog = _safe_path(
        ROOT,
        raw_ducklake.get("catalog_path")
        or runtime.relative_to(ROOT) / "ducklake" / "catalog.duckdb",
        field="ducklake_projection_program.catalog_path",
    )
    ducklake_data = _safe_path(
        ROOT,
        raw_ducklake.get("data_path")
        or runtime.relative_to(ROOT) / "ducklake" / "data",
        field="ducklake_projection_program.data_path",
    )
    for label, path in (
        ("evidence", evidence),
        ("quack_owner", owner),
        ("ducklake_catalog", ducklake_catalog),
        ("ducklake_data", ducklake_data),
    ):
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError(f"{label} must be below runtime_paths.root") from exc
    return {
        "runtime": runtime,
        "database": database,
        "owner": owner,
        "bootstrap_receipt": evidence / "bootstrap" / BOOTSTRAP_RECEIPT_NAME,
        "spar000_qualification": evidence / "bootstrap" / SPAR000_QUALIFICATION_NAME,
        "ducklake_receipt": evidence / "bootstrap" / DUCKLAKE_RECEIPT_NAME,
        "ducklake_catalog": ducklake_catalog,
        "ducklake_data": ducklake_data,
    }


def _harden_runtime_directories(
    board: Any,
    paths: Mapping[str, Path],
) -> None:
    """Create exact SPAR runtime directories as private same-UID boundaries."""

    runtime = paths["runtime"].resolve(strict=False)
    raw_runtime = board.payload.get("runtime_paths")
    raw_runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    program = board.resolved_database_program()
    candidates = {
        runtime,
        paths["database"].parent,
        paths["owner"],
        paths["bootstrap_receipt"].parent,
        paths["ducklake_catalog"].parent,
        paths["ducklake_data"],
    }
    for key, value in raw_runtime.items():
        candidates.add(_safe_path(ROOT, value, field=f"runtime_paths.{key}"))
    for field, value in (
        ("event_store_path", program.event_store_path),
        ("runtime_registry_path", program.runtime_registry_path),
    ):
        if value:
            candidates.add(_safe_path(ROOT, value, field=field))
    state_root = _safe_path(
        ROOT,
        raw_runtime.get("state") or runtime.relative_to(ROOT) / "state",
        field="runtime_paths.state",
    )
    for index in range(board.max_lanes):
        candidates.add(state_root / f"lane-{index}")
    merge_root = _safe_path(
        ROOT,
        raw_runtime.get("merge_queue") or runtime.relative_to(ROOT) / "merge-queue",
        field="runtime_paths.merge_queue",
    )
    for name in ("pending", "processing", "completed", "failed", "cancelled", "quarantine"):
        candidates.add(merge_root / name)

    for path in sorted(candidates, key=lambda item: (len(item.parts), str(item))):
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError("SPAR private runtime directory escapes its root") from exc
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat_module.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
            ):
                raise OperatorError("SPAR runtime directory is not privately owned")
            os.fchmod(descriptor, 0o700)
            if stat_module.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
                raise OperatorError("SPAR runtime directory did not become private")
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _ducklake_projection(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one non-authoritative bootstrap observation to DuckLake."""

    projection: dict[str, Any] = {
        "schema": DUCKLAKE_SCHEMA,
        "authoritative": False,
        "scheduler_gate": False,
        "completion_gate": False,
        "status": "unavailable",
        "reason_code": "ducklake_projection_unavailable",
        "source_head": str(population["source_head"]),
        "repository_tree_id": str(population["repository_tree_id"]),
        "plan_root_cid": str(population["plan_root_cid"]),
    }
    try:
        import duckdb

        catalog = paths["ducklake_catalog"]
        data_path = paths["ducklake_data"]
        catalog.parent.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        memory = duckdb.connect(":memory:")
        try:
            memory.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data_path).replace("'", "''")
            memory.execute(
                f"ATTACH 'ducklake:{catalog_sql}' AS spar_history "
                f"(DATA_PATH '{data_sql}')"
            )
            memory.execute(
                """
                CREATE TABLE IF NOT EXISTS spar_history.bootstrap_history (
                    event_id VARCHAR,
                    observed_at_epoch DOUBLE,
                    source_head VARCHAR,
                    repository_tree_id VARCHAR,
                    plan_root_cid VARCHAR,
                    projection_cid VARCHAR,
                    task_count BIGINT,
                    goal_count BIGINT,
                    body_json VARCHAR
                )
                """
            )
            event_id = _identity(
                {
                    "source_head": population["source_head"],
                    "plan_root_cid": population["plan_root_cid"],
                    "projection_cid": control_receipt.get("projection_cid"),
                }
            )
            existing = memory.execute(
                "SELECT COUNT(*) FROM spar_history.bootstrap_history WHERE event_id = ?",
                [event_id],
            ).fetchone()
            if existing is None or int(existing[0]) == 0:
                memory.execute(
                    """
                    INSERT INTO spar_history.bootstrap_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        event_id,
                        time.time(),
                        population["source_head"],
                        population["repository_tree_id"],
                        population["plan_root_cid"],
                        str(control_receipt.get("projection_cid") or ""),
                        int(control_receipt.get("task_count") or 0),
                        int(control_receipt.get("goal_count") or 0),
                        json.dumps(
                            {
                                "authority": "DuckDB/DatabaseTaskSource@1",
                                "transport": "QuackStateServer@1",
                                "projection": "DuckLake/non-authoritative",
                            },
                            sort_keys=True,
                        ),
                    ],
                )
            row_count = int(
                memory.execute(
                    "SELECT COUNT(*) FROM spar_history.bootstrap_history"
                ).fetchone()[0]
            )
            memory.execute("DETACH spar_history")
        finally:
            memory.close()
        projection.update(
            {
                "status": "available",
                "reason_code": "",
                "event_id": event_id,
                "row_count": row_count,
                "catalog_path": str(catalog.relative_to(ROOT)),
                "data_path": str(data_path.relative_to(ROOT)),
            }
        )
    except Exception as exc:
        # This projection is optional by contract. Preserve a typed absence and
        # never use it to reject a valid DuckDB materialization.
        projection["error_class"] = type(exc).__name__
    projection["projection_receipt_id"] = _identity(projection)
    _atomic_json(paths["ducklake_receipt"], projection)
    return projection


def materialize(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    board, config = _load_config(config_path)
    paths = _runtime_paths(board)
    _harden_runtime_directories(board, paths)
    population = _population(board, config)
    receipt_path = paths["bootstrap_receipt"]
    if paths["database"].exists() or receipt_path.exists():
        if not paths["database"].is_file() or not receipt_path.is_file():
            raise OperatorError("partial bootstrap state exists; operator review required")
        prior = _json_object(receipt_path)
        exact = all(
            prior.get(key) == population.get(key)
            for key in ("source_head", "repository_tree_id", "plan_root_cid")
        )
        if not exact:
            raise OperatorError(
                "existing DuckDB authority is bound to a different source tree or plan"
            )
        with DatabaseTaskSource(
            paths["database"],
            owner_id="spar-bootstrap:verify-existing",
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        ) as source:
            snapshot = source.snapshot().to_dict()
        if int(snapshot["task_count"]) != len(population["tasks"]):
            raise OperatorError("existing DuckDB task population differs from sealed board")
        return {
            "schema": OPERATOR_SCHEMA,
            "command": "materialize",
            "idempotent_replay": True,
            "materialized": True,
            "bootstrap_receipt": prior,
            "snapshot": snapshot,
        }

    qualification_commands = (
        (sys.executable, "scripts/validate_semantic_preserving_remodularization_dependencies.py", "--check-all"),
        (sys.executable, "scripts/validate_semantic_preserving_remodularization_board.py", "--check-all"),
        (sys.executable, "-m", "pytest", "-q", "test/api/semantic_refactoring/test_bootstrap_controls.py"),
    )
    qualification_results: list[dict[str, Any]] = []
    for command in qualification_commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=False,
            capture_output=True,
            check=False,
            timeout=900,
        )
        result = {
            "argv": list(command),
            "returncode": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "stdout_bytes": len(completed.stdout),
            "stderr_bytes": len(completed.stderr),
        }
        qualification_results.append(result)
        if completed.returncode != 0:
            raise OperatorError(
                "SPAR-000 current-tree qualification failed: " + command[-1]
            )
    qualification = {
        "schema": "spar/operator-bootstrap-qualification@1",
        "task_id": "SPAR-000",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "results": qualification_results,
        "operator_only": True,
        "simulated": False,
    }
    qualification["qualification_cid"] = _identity(qualification)

    paths["runtime"].mkdir(parents=True, exist_ok=True)
    with DatabaseTaskSource(
        paths["database"],
        owner_id="spar-bootstrap:single-writer",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    ) as source:
        control_receipt = dict(source.materialize(population))
        bootstrap_task = source.get_task("SPAR-000")
        if bootstrap_task is None:
            raise OperatorError("SPAR-000 is absent after materialization")
        source.record_validation_result(
            task_cid=bootstrap_task.task_cid,
            outcome="passed",
            evidence_digest=str(qualification["qualification_cid"]),
            argv=[sys.executable, "operator:SPAR-000"],
            body={
                "repository_commit": population["source_head"],
                "repository_tree": population["repository_tree_id"],
                "plan_root_cid": population["plan_root_cid"],
                "operator_only": True,
                "simulated": False,
            },
        )
        current_bootstrap = source.get_task(bootstrap_task.task_cid)
        if current_bootstrap is None:
            raise OperatorError("SPAR-000 vanished before operator completion")
        source.compare_and_set_status(
            current_bootstrap.task_cid,
            current_bootstrap.revision,
            "completed",
            {
                "schema": "spar/operator-bootstrap-completion@1",
                "task_id": "SPAR-000",
                "qualification_cid": qualification["qualification_cid"],
                "repository_commit": population["source_head"],
                "repository_tree": population["repository_tree_id"],
                "operator_only": True,
            },
            evidence_digests=[str(qualification["qualification_cid"])],
        )
        snapshot = source.snapshot().to_dict()
        ready_ids = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    if int(snapshot["task_count"]) != len(population["tasks"]):
        raise OperatorError("DuckDB materialization task count is not exact")
    if int(snapshot["goal_count"]) != len(population["objectives"]):
        raise OperatorError("DuckDB materialization goal count is not exact")
    if ready_ids != ["SPAR-001"]:
        raise OperatorError(
            "post-bootstrap DuckDB readiness frontier must contain exactly SPAR-001"
        )
    _atomic_json(paths["spar000_qualification"], qualification)
    ducklake = _ducklake_projection(
        paths=paths,
        population=population,
        control_receipt=control_receipt,
    )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "source_identities": population["source_identities"],
        "source_forest": population["source_forest"],
        "database_task_source_receipt": control_receipt,
        "projection_cid": snapshot["projection_cid"],
        "task_count": snapshot["task_count"],
        "goal_count": snapshot["goal_count"],
        "dependency_count": snapshot["dependency_count"],
        "operator_completed_task_ids": ["SPAR-000"],
        "ready_task_ids": ready_ids,
        "spar000_qualification": qualification,
        "authority": {
            "semantic_state": "DuckDB/DatabaseTaskSource@1",
            "state_owner_transport": "QuackStateServer@1",
            "ducklake": "optional_non_authoritative_history_projection",
        },
        "ducklake_projection": ducklake,
    }
    receipt["bootstrap_receipt_id"] = _identity(receipt)
    _atomic_json(receipt_path, receipt)
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "materialize",
        "idempotent_replay": False,
        "materialized": True,
        "bootstrap_receipt": receipt,
        "snapshot": snapshot,
    }


def _verify_control_plane(path: Path) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        MigrationRunReport,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        CONTROL_PLANE_MIGRATION_VERSION,
        load_control_plane_catalog,
        verify_installed_schema,
    )

    # SPAR uses the canonical full control-plane schema revision ``1``.  The
    # smaller datasets-authoritative operational profile is deliberately not
    # selected: the generic multi-supervisor rejects that profile for live
    # Quack operation, and the SPAR board needs the full proof/evidence tables.
    verification = verify_installed_schema(path)
    fingerprint = str(verification.get("schema_fingerprint") or "")
    if not fingerprint:
        raise OperatorError("existing full control plane has no schema fingerprint")
    return MigrationRunReport(
        from_version=CONTROL_PLANE_MIGRATION_VERSION,
        to_version=CONTROL_PLANE_MIGRATION_VERSION,
        receipts=(),
        schema_fingerprint=fingerprint,
        catalog_fingerprint=load_control_plane_catalog().fingerprint(),
        changed=False,
    )


def _owner_connection(path: Path) -> Any:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )

    connection = duckdb.connect(str(path))
    try:
        connection.execute("LOAD quack")
    except BaseException:
        connection.close()
        raise
    return DuckDBConnection.wrap(connection)


def _normalized_owner_dml(sql: str) -> str:
    normalized = " ".join(str(sql or "").strip().upper().split())
    if not normalized.startswith(OWNER_DML_PREFIXES):
        raise OperatorError("mutation inbox accepts only the closed owner-DML vocabulary")
    if ";" in normalized.rstrip(";"):
        raise OperatorError("mutation inbox accepts exactly one SQL statement")
    return normalized


def _process_mutations(server: Any, mutation_dir: Path) -> None:
    mutation_dir.mkdir(parents=True, exist_ok=True)
    for request in sorted(mutation_dir.glob("*.request.json")):
        done = request.with_name(request.name.replace(".request.json", ".done.json"))
        try:
            try:
                payload = json.loads(request.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # The client creates a tiny same-filesystem request. A partial
                # read is retried rather than converted into a false failure.
                continue
            if not isinstance(payload, Mapping):
                raise OperatorError("mutation request must be an object")
            sql = str(payload.get("sql") or "")
            _normalized_owner_dml(sql)
            parameters = payload.get("parameters")
            if parameters is not None and (
                isinstance(parameters, (str, bytes, bytearray))
                or not isinstance(parameters, (Mapping, Sequence))
            ):
                raise OperatorError("mutation parameters must be a mapping or sequence")
            owner_connection = getattr(server, "_connection", None)
            if owner_connection is None:
                raise OperatorError("state-owner connection is unavailable")
            result = (
                owner_connection.execute(sql)
                if parameters is None
                else owner_connection.execute(sql, parameters)
            )
            rowcount = -1
            try:
                if getattr(result, "description", None):
                    result.fetchall()
                elif hasattr(result, "rowcount"):
                    rowcount = int(result.rowcount)
            except Exception:
                pass
            _atomic_json(done, {"ok": True, "rowcount": rowcount})
        except Exception as exc:
            _atomic_json(
                done,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: mutation rejected",
                },
            )
        try:
            request.unlink()
        except FileNotFoundError:
            pass


def _build_state_owner(config_path: Path) -> tuple[Any, dict[str, Path], Any]:
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        establish_state_authority_process_boundary,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )

    board, _config = _load_config(config_path)
    paths = _runtime_paths(board)
    _harden_runtime_directories(board, paths)
    if not paths["database"].is_file() or not paths["bootstrap_receipt"].is_file():
        raise OperatorError("materialize the sealed SPAR board before starting Quack")
    program = board.resolved_database_program()
    endpoint = QUACK_ENDPOINT_RE.fullmatch(program.quack_endpoint)
    if endpoint is None:
        raise OperatorError("configured Quack endpoint is not loopback")
    host = endpoint.group(1)
    port = int(endpoint.group(2))
    if not 1 <= port <= 65535:
        raise OperatorError("configured Quack port is out of range")
    # The owner mints raw credentials into its in-memory grant vault. Harden
    # this controller before constructing that vault so same-UID worker or
    # provider descendants cannot inspect it through procfs.
    establish_state_authority_process_boundary()
    server = build_server(
        database_path=paths["database"],
        state_dir=paths["owner"],
        host=host,
        port=port,
        repository_id="repository:ipfs_accelerate_py",
        store_id=program.store_id,
        secret_handle=program.endpoint_secret_handle,
        allow_experimental=False,
        migrate=_verify_control_plane,
        connection_factory=_owner_connection,
    )
    return server, paths, program


def _start_state_owner(config_path: Path) -> tuple[Any, dict[str, Path], Any, Any, dict[str, Any]]:
    server, paths, program = _build_state_owner(config_path)
    try:
        identity = server.start()
        ready = server.ready()
        # The generic SPAR controller issues exact birth-bound grants through
        # its inherited bootstrap listener. The server's reusable status
        # bootstrap credential is therefore unnecessary and must not remain
        # at rest.
        server.typed_command_token_path().unlink(missing_ok=True)
    except BaseException:
        server.stop()
        raise
    return server, paths, program, identity, ready


def _owner_task_projection(server: Any) -> dict[str, Any]:
    connection = getattr(server, "_connection", None)
    transaction_lock = getattr(server, "_owner_transaction_lock", None)
    if connection is None or transaction_lock is None:
        raise OperatorError("state-owner task projection connection is unavailable")
    with transaction_lock:
        return _task_status(connection)


def _publish_live_projection(server: Any, paths: Mapping[str, Path]) -> dict[str, Any]:
    ready = server.ready()
    task_projection = _owner_task_projection(server)
    unsigned = {
        "schema": "spar/live-owner-projection@1",
        "observed_at_unix_ns": time.time_ns(),
        "owner_process_birth_id": str(ready["process_birth_id"]),
        "server_id": str(ready["server_id"]),
        "store_id": str(ready["store_id"]),
        "generation": int(ready["generation"]),
        "schema_revision": int(ready["schema_revision"]),
        "quack_authenticated_live_query": bool(ready.get("live") is True),
        "task_projection": task_projection,
    }
    payload = {**unsigned, "projection_cid": _identity(unsigned)}
    _atomic_json(paths["owner"] / "spar-live-projection.json", payload)
    return payload


def _serve_state_owner(
    server: Any,
    paths: Mapping[str, Path],
    *,
    child: subprocess.Popen[Any] | None = None,
) -> tuple[dict[str, Any], int | None]:
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        ServerLifecycle,
    )

    stopped = {"value": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stopped["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    mutation_dir = paths["owner"] / "mutations"
    control_path = server.stop_control_path()
    next_projection = 0.0
    child_returncode: int | None = None
    try:
        while server.lifecycle is ServerLifecycle.READY and not stopped["value"]:
            if control_path.is_file():
                break
            if child is not None:
                child_returncode = child.poll()
                if child_returncode is not None:
                    break
            _process_mutations(server, mutation_dir)
            now = time.monotonic()
            if now >= next_projection:
                _publish_live_projection(server, paths)
                next_projection = now + 1.0
            time.sleep(0.05)
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 30.0
            while child.poll() is None and time.monotonic() < deadline:
                _process_mutations(server, mutation_dir)
                time.sleep(0.05)
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5.0)
        if child is not None:
            child_returncode = child.poll()
        result = server.stop()
    return result, child_returncode


def state_owner(config_path: Path) -> int:
    server, paths, _program, identity, ready = _start_state_owner(config_path)
    print(
        json.dumps(
            {
                "schema": OPERATOR_SCHEMA,
                "command": "state-owner",
                "ready": True,
                "identity": identity.to_dict(),
                "live": ready,
                "mutation_dir": str((paths["owner"] / "mutations").relative_to(ROOT)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result, _child_returncode = _serve_state_owner(server, paths)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _execution_route_policy(paths: Mapping[str, Path]) -> Any:
    """Seal one exact all-task route before the owner takes the DuckDB lease."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
        TaskExecutionRoutePolicy,
    )

    bootstrap = _json_object(paths["bootstrap_receipt"])
    with DatabaseTaskSource(
        paths["database"],
        owner_id="spar-route-policy:single-writer",
        repository_tree_id=str(bootstrap["repository_tree_id"]),
        plan_root_cid=str(bootstrap["plan_root_cid"]),
    ) as source:
        snapshot = source.snapshot()
        tasks: list[Any] = []
        cursor = ""
        while True:
            page = source.list_tasks(cursor=cursor, limit=500)
            tasks.extend(page.tasks)
            cursor = page.next_cursor
            if not cursor:
                break
    if len(tasks) != int(snapshot.task_count):
        raise OperatorError("execution-route task population is incomplete")
    return TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=tasks,
        execution_modes={
            task.task_alias: GROK_CODEX_EXECUTION_MODE for task in tasks
        },
    )


def _exact_argv_option(argv: Sequence[str], name: str) -> str:
    indexes = [index for index, value in enumerate(argv) if value == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        raise OperatorError(f"supervisor argv does not bind exactly one {name}")
    return str(argv[indexes[0] + 1])


class _SparStateOwnerBootstrapBroker:
    """Issue a distinct PID/birth-fenced typed grant for each lane daemon."""

    def __init__(
        self,
        *,
        channel: socket.socket,
        server: Any,
        board: Any,
        paths: Mapping[str, Path],
        execution_route_policy: Any,
    ) -> None:
        self.channel = channel
        self.server = server
        self.board = board
        self.paths = paths
        self.execution_route_policy = execution_route_policy
        self.allowed_sessions = tuple(
            f"{board.board_namespace}-{index}" for index in range(board.max_lanes)
        )
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.fail_fast_enabled = threading.Event()
        self.failure = ""
        self.last_rejection = ""
        self.rejection_count = 0
        self.current_by_session: dict[str, dict[str, Any]] = {}
        self.active_grants: dict[str, str] = {}
        self._accepted: socket.socket | None = None
        self._lock = threading.RLock()
        self._thread = threading.Thread(
            target=self._run,
            name="spar-state-owner-bootstrap",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            raise OperatorError("SPAR state-owner bootstrap broker was already started")
        self._started = True
        self._thread.start()
        if not self.ready.wait(BOOTSTRAP_READY_TIMEOUT_SECONDS):
            raise OperatorError("SPAR state-owner bootstrap broker did not become ready")
        if self.failure:
            raise OperatorError("SPAR state-owner bootstrap broker failed during startup")

    def enable_fail_fast(self) -> None:
        self.fail_fast_enabled.set()

    def _terminal_failure(self, exc: BaseException) -> None:
        with self._lock:
            self.failure = self.failure or type(exc).__name__
        self.ready.set()
        if self.fail_fast_enabled.is_set() and not self.stopping.is_set():
            os.kill(os.getpid(), signal.SIGTERM)

    def stop(self) -> None:
        self.stopping.set()
        with self._lock:
            accepted = self._accepted
        if accepted is not None:
            try:
                accepted.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            accepted.close()
        try:
            self.channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.channel.close()
        if self._started:
            self._thread.join(timeout=5.0)
        if self._started and self._thread.is_alive():
            raise OperatorError("SPAR state-owner bootstrap broker did not stop")
        self._fence_admitted_births()
        failures: list[str] = []
        with self._lock:
            grant_ids = tuple(self.active_grants.values())
        for grant_id in grant_ids:
            try:
                self.server.revoke_typed_client_grant(grant_id)
            except Exception as exc:
                failures.append(type(exc).__name__)
        with self._lock:
            self.active_grants.clear()
        if failures:
            raise OperatorError("SPAR bootstrap grant revocation failed")

    def _persist(self) -> None:
        with self._lock:
            unsigned = {
                "schema": "spar/state-owner-bootstrap-broker@1",
                "controller_pid": os.getpid(),
                "allowed_sessions": list(self.allowed_sessions),
                "active_sessions": sorted(self.active_grants),
                "current_births": {
                    key: dict(value)
                    for key, value in sorted(self.current_by_session.items())
                },
                "rejection_count": self.rejection_count,
                "last_rejection": self.last_rejection,
                "failure": self.failure,
                "credential_transport": "private_inherited_socket",
                "credential_persisted": False,
            }
        _atomic_json(
            self.paths["owner"] / "spar-bootstrap-broker.json",
            {**unsigned, "projection_cid": _identity(unsigned)},
        )

    @staticmethod
    def _require_dead(birth_payload: Mapping[str, Any], *, noun: str) -> None:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )

        birth = ProcessBirthIdentity.from_dict(birth_payload)
        if owner_liveness(birth) is not OwnerLiveness.DEAD:
            raise OperatorError(f"prior {noun} birth remains live")

    def _admitted_births(self) -> tuple[Any, ...]:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
        )

        with self._lock:
            records = tuple(dict(record) for record in self.current_by_session.values())
        births: list[Any] = []
        seen: set[tuple[int, int, str]] = set()
        for field in ("supervisor_process_birth", "daemon_process_birth"):
            for record in records:
                raw = record.get(field)
                if not isinstance(raw, Mapping):
                    raise OperatorError("SPAR admitted process birth is unavailable")
                try:
                    birth = ProcessBirthIdentity.from_dict(raw)
                except (KeyError, OverflowError, TypeError, ValueError) as exc:
                    raise OperatorError("SPAR admitted process birth is malformed") from exc
                if birth.pid <= 1 or birth.start_time_ticks <= 0:
                    raise OperatorError("SPAR admitted process birth is unsafe")
                key = (birth.pid, birth.start_time_ticks, birth.boot_id)
                if key not in seen:
                    seen.add(key)
                    births.append(birth)
        return tuple(births)

    @staticmethod
    def _signal_admitted_birth(birth: Any, signum: int) -> None:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            owner_liveness,
        )

        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise OperatorError("SPAR admitted-process fencing requires Linux pidfds")
        descriptor = -1
        try:
            descriptor = os.pidfd_open(birth.pid, 0)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise OperatorError("SPAR admitted-process pidfd is unavailable") from exc
        try:
            state = owner_liveness(birth)
            if state is OwnerLiveness.DEAD:
                return
            if state is not OwnerLiveness.ALIVE:
                raise OperatorError("SPAR admitted process is uninspectable")
            try:
                signal.pidfd_send_signal(descriptor, signum)
            except ProcessLookupError:
                return
            except OSError as exc:
                raise OperatorError("SPAR admitted process could not be signalled") from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _live_admitted_births(births: Sequence[Any]) -> tuple[Any, ...]:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            owner_liveness,
        )

        live: list[Any] = []
        for birth in births:
            state = owner_liveness(birth)
            if state is OwnerLiveness.UNKNOWN:
                raise OperatorError("SPAR admitted process became uninspectable")
            if state is OwnerLiveness.ALIVE:
                live.append(birth)
        return tuple(live)

    def _fence_admitted_births(self) -> None:
        births = self._admitted_births()
        live = self._live_admitted_births(births)
        for birth in live:
            self._signal_admitted_birth(birth, signal.SIGTERM)
        deadline = time.monotonic() + BOOTSTRAP_PROCESS_STOP_GRACE_SECONDS
        while live and time.monotonic() < deadline:
            time.sleep(0.02)
            live = self._live_admitted_births(live)
        for birth in live:
            self._signal_admitted_birth(birth, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while live and time.monotonic() < deadline:
            time.sleep(0.02)
            live = self._live_admitted_births(live)
        if live:
            raise OperatorError("SPAR admitted process births survived bounded stop")

    def _validate_supervisor_parent(self, daemon_birth: Any, session: str) -> Any:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            read_process_birth,
        )

        supervisor = read_process_birth(int(daemon_birth.parent_pid))
        if supervisor is None or int(supervisor.parent_pid) != os.getpid():
            raise OperatorError("daemon is not a child of this controller's supervisor")
        before = supervisor
        try:
            argv = [
                item.decode("utf-8")
                for item in Path(f"/proc/{supervisor.pid}/cmdline")
                .read_bytes()
                .split(b"\0")
                if item
            ]
        except (OSError, UnicodeDecodeError) as exc:
            raise OperatorError("cannot attest supervisor argv") from exc
        after = read_process_birth(supervisor.pid)
        if before != after:
            raise OperatorError("supervisor birth changed during argv attestation")
        lane_index = self.allowed_sessions.index(session)
        state_slug = re.sub(
            r"[^a-z0-9._-]+",
            "-",
            self.board.task_prefix.strip().lower(),
        ).strip("-") or "configured-board"
        exact = {
            "--board-namespace": self.board.board_namespace,
            "--task-shard-count": str(self.board.max_lanes),
            "--task-shard-index": str(lane_index),
            "--state-prefix": f"{state_slug}_lane_{lane_index}",
            "--database-owner-session-id": session,
            "--state-owner-bootstrap-store-id": (
                self.board.resolved_database_program().store_id
            ),
            "--state-owner-bootstrap-fd": str(self.channel.fileno()),
        }
        if any(_exact_argv_option(argv, name) != value for name, value in exact.items()):
            raise OperatorError("supervisor birth differs from the sealed lane profile")
        supervisor_id = process_birth_id(supervisor)
        for other_session, record in self.current_by_session.items():
            if (
                other_session != session
                and record.get("supervisor_process_birth_id") == supervisor_id
            ):
                raise OperatorError("one supervisor requested multiple owner sessions")
        return supervisor

    def _admit(
        self,
        request: Mapping[str, Any],
        *,
        peer_pid: int,
        peer_uid: int,
    ) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
            read_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
            STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
            daemon_required_owner_command_operations,
            daemon_required_owner_operations,
        )

        if set(request) != {
            "schema",
            "pid",
            "process_birth",
            "process_birth_id",
            "client_id",
            "store_id",
        } or request.get("schema") != STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA:
            raise OperatorError("bootstrap request differs from its closed schema")
        if self.stopping.is_set() or peer_uid != os.geteuid():
            raise OperatorError("bootstrap admission is closed or foreign")
        try:
            pid = int(request.get("pid") or 0)
            supplied = ProcessBirthIdentity.from_dict(
                dict(request.get("process_birth") or {})
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise OperatorError("bootstrap process birth is malformed") from exc
        observed = read_process_birth(pid)
        supplied_birth_id = str(request.get("process_birth_id") or "")
        if (
            pid <= 1
            or pid != peer_pid
            or observed is None
            or observed != supplied
            or process_birth_id(observed) != supplied_birth_id
        ):
            raise OperatorError("bootstrap process birth is stale or substituted")
        client_id = str(request.get("client_id") or "")
        prefix = "database-implementation-daemon:"
        session = client_id.removeprefix(prefix)
        program = self.board.resolved_database_program()
        if (
            not client_id.startswith(prefix)
            or session not in self.allowed_sessions
            or request.get("store_id") != program.store_id
        ):
            raise OperatorError("bootstrap request scope differs from the sealed board")
        with self._lock:
            supervisor = self._validate_supervisor_parent(supplied, session)
            prior = self.current_by_session.get(session)
            if prior is not None:
                prior_daemon = prior.get("daemon_process_birth")
                if not isinstance(prior_daemon, Mapping):
                    raise OperatorError("prior lane daemon birth is malformed")
                self._require_dead(prior_daemon, noun="lane daemon")
                prior_supervisor = prior.get("supervisor_process_birth")
                if (
                    isinstance(prior_supervisor, Mapping)
                    and dict(prior_supervisor) != supervisor.to_dict()
                ):
                    self._require_dead(prior_supervisor, noun="lane supervisor")
                grant_id = self.active_grants.pop(session, "")
                if grant_id:
                    self.server.revoke_typed_client_grant(grant_id)
            token, grant = self.server.issue_typed_client_grant_record(
                client_id=client_id,
                process_birth_id=supplied_birth_id,
                allowed_operations=daemon_required_owner_operations(),
                allowed_command_operations=daemon_required_owner_command_operations(),
                peer_pid=pid,
                ttl_seconds=INTERNAL_CLIENT_GRANT_TTL_SECONDS,
            )
            if self.stopping.is_set():
                self.server.revoke_typed_client_grant(grant.grant_id)
                raise OperatorError("bootstrap admission closed during grant issue")
            identity = self.server.identity
            if identity is None:
                self.server.revoke_typed_client_grant(grant.grant_id)
                raise OperatorError("state owner lost identity during grant issue")
            self.current_by_session[session] = {
                "session": session,
                "client_id": client_id,
                "daemon_process_birth": supplied.to_dict(),
                "daemon_process_birth_id": supplied_birth_id,
                "supervisor_process_birth": supervisor.to_dict(),
                "supervisor_process_birth_id": process_birth_id(supervisor),
                "grant_expires_at_ms": int(grant.expires_at),
                "grant_renew_after": (
                    time.monotonic() + INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS
                ),
            }
            self.active_grants[session] = grant.grant_id
            self._persist()
        return {
            "schema": STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
            "ok": True,
            "endpoint": program.quack_endpoint,
            "socket_path": str(self.server.typed_command_socket_path()),
            "store_id": program.store_id,
            "server_id": identity.server_id,
            "client_id": client_id,
            "process_birth_id": supplied_birth_id,
            "token": token,
            "execution_route_policy": self.execution_route_policy.to_dict(),
        }

    def _renew_due_grants(self) -> None:
        """Renew only grants whose exact credential-holding birth is live."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
            TypedStateOwnerAuthorizationError,
        )

        now = time.monotonic()
        with self._lock:
            due = tuple(
                (
                    session,
                    grant_id,
                    float(
                        (self.current_by_session.get(session) or {}).get(
                            "grant_renew_after",
                            0.0,
                        )
                    ),
                    dict(self.current_by_session.get(session) or {}),
                )
                for session, grant_id in self.active_grants.items()
            )
        changed = False
        for session, grant_id, renew_after, record in due:
            if now < renew_after:
                continue
            raw_birth = record.get("daemon_process_birth")
            if not isinstance(raw_birth, Mapping):
                raise OperatorError("SPAR grant-renewal daemon birth is unavailable")
            daemon_birth = ProcessBirthIdentity.from_dict(raw_birth)
            liveness = owner_liveness(daemon_birth)
            if liveness is OwnerLiveness.DEAD:
                continue
            if liveness is not OwnerLiveness.ALIVE:
                raise OperatorError("SPAR grant-renewal daemon birth is uninspectable")
            try:
                renewed = self.server.renew_typed_client_grant(
                    grant_id,
                    ttl_seconds=INTERNAL_CLIENT_GRANT_TTL_SECONDS,
                )
            except TypedStateOwnerAuthorizationError:
                if owner_liveness(daemon_birth) is OwnerLiveness.DEAD:
                    continue
                raise
            with self._lock:
                if self.active_grants.get(session) != grant_id:
                    raise OperatorError("SPAR state-owner grant rotated during renewal")
                current = self.current_by_session.get(session)
                if not isinstance(current, dict):
                    raise OperatorError("SPAR grant-renewal record is unavailable")
                current["grant_expires_at_ms"] = int(renewed.expires_at)
                current["grant_renew_after"] = (
                    time.monotonic() + INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS
                )
                changed = True
        if changed:
            self._persist()

    def _run(self) -> None:
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            StateOwnerBootstrapError,
            _receive_frame,
            _send_frame,
        )

        try:
            self.channel.settimeout(1.0)
            self._persist()
        except BaseException as exc:
            self._terminal_failure(exc)
            return
        self.ready.set()
        while not self.stopping.is_set():
            accepted: socket.socket | None = None
            try:
                self._renew_due_grants()
            except BaseException as exc:
                self._terminal_failure(exc)
                return
            try:
                accepted, _address = self.channel.accept()
                with self._lock:
                    if self.stopping.is_set():
                        accepted.close()
                        return
                    self._accepted = accepted
                accepted.settimeout(1.0)
                peer = accepted.getsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_PEERCRED,
                    struct.calcsize("3i"),
                )
                peer_pid, peer_uid, _peer_gid = struct.unpack("3i", peer)
                response = self._admit(
                    _receive_frame(accepted),
                    peer_pid=int(peer_pid),
                    peer_uid=int(peer_uid),
                )
                _send_frame(accepted, response)
            except TimeoutError:
                continue
            except (EOFError, OperatorError, StateOwnerBootstrapError) as exc:
                if not self.stopping.is_set():
                    with self._lock:
                        self.rejection_count += 1
                        self.last_rejection = type(exc).__name__
                    try:
                        self._persist()
                    except BaseException as persist_exc:
                        self._terminal_failure(persist_exc)
                        return
                continue
            except OSError as exc:
                if not self.stopping.is_set() and accepted is None:
                    self._terminal_failure(exc)
                    return
                continue
            except BaseException as exc:
                self._terminal_failure(exc)
                return
            finally:
                if accepted is not None:
                    with self._lock:
                        if self._accepted is accepted:
                            self._accepted = None
                    try:
                        accepted.close()
                    except OSError:
                        pass


class _OwnerProjectionMonitor:
    def __init__(
        self,
        server: Any,
        paths: Mapping[str, Path],
        *,
        on_failure: Callable[[BaseException], None],
    ) -> None:
        self.server = server
        self.paths = paths
        self.on_failure = on_failure
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.failure = ""
        self._thread = threading.Thread(
            target=self._run,
            name="spar-owner-projection",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        if not self.ready.wait(BOOTSTRAP_READY_TIMEOUT_SECONDS):
            raise OperatorError("SPAR owner projection monitor did not become ready")
        if self.failure:
            raise OperatorError("SPAR owner projection monitor failed during startup")

    def stop(self) -> None:
        self.stopping.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise OperatorError("SPAR owner projection monitor did not stop")

    def _run(self) -> None:
        initial = True
        while not self.stopping.is_set():
            try:
                _publish_live_projection(self.server, self.paths)
            except BaseException as exc:
                self.failure = type(exc).__name__
                self.ready.set()
                self.on_failure(exc)
                return
            if initial:
                initial = False
                self.ready.set()
            self.stopping.wait(1.0)


def _new_bootstrap_listener(*, lane_count: int) -> socket.socket:
    from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
        validate_state_owner_bootstrap_listener,
    )

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind("\0spar-bootstrap-" + secrets.token_hex(16))
        listener.listen(max(8, int(lane_count) * 2))
        validate_state_owner_bootstrap_listener(listener.fileno())
    except BaseException:
        listener.close()
        raise
    return listener


def _bind_bootstrap_launch_plan(
    plan: dict[str, Any],
    *,
    listener: socket.socket,
    store_id: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        _generic_state_owner_bootstrap_binding,
    )

    argv = list(plan.get("argv") or ())
    for value in (
        "--state-owner-bootstrap-fd",
        str(listener.fileno()),
        "--state-owner-bootstrap-store-id",
        store_id,
    ):
        argv.append(f"--common-arg={value}")
    common_args = tuple(
        item.split("=", 1)[1]
        for item in argv
        if item.startswith("--common-arg=")
    )
    if _generic_state_owner_bootstrap_binding(common_args) != listener.fileno():
        raise OperatorError("SPAR bootstrap launch-plan descriptor changed")
    if any("IPFS_ACCELERATE_AGENT_QUACK_TOKEN=" in item for item in argv):
        raise OperatorError("SPAR bootstrap launch plan contains a raw credential")
    plan["argv"] = argv
    plan["state_owner_bootstrap"] = {
        "transport": "private_inherited_socket",
        "descriptor": listener.fileno(),
        "store_id": store_id,
        "credential_persisted": False,
    }


def supervise(
    config_path: Path,
    *,
    implement: bool,
    dry_run: bool = False,
    duration_seconds: float = float("inf"),
) -> int:
    """Run the native configured-board fabric with exact daemon-birth grants."""

    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        _apply_configured_board_environment,
        configured_board_launch_plan,
        preflight_configured_board,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        main as multi_supervisor_main,
    )
    board, _config = _load_config(config_path)
    preflight = preflight_configured_board(board)
    if preflight.get("valid") is not True:
        raise OperatorError("configured-board preflight rejected the sealed SPAR board")
    plan = configured_board_launch_plan(
        board,
        implement=implement,
        detach=False,
        duration_seconds=duration_seconds,
    )
    program = board.resolved_database_program()
    if dry_run:
        listener = _new_bootstrap_listener(lane_count=board.max_lanes)
        try:
            _bind_bootstrap_launch_plan(
                plan,
                listener=listener,
                store_id=program.store_id,
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
        finally:
            listener.close()
        return 0
    paths = _runtime_paths(board)
    route_policy = _execution_route_policy(paths)
    server, paths, program, identity, ready = _start_state_owner(config_path)
    listener: socket.socket | None = None
    broker: _SparStateOwnerBootstrapBroker | None = None
    monitor: _OwnerProjectionMonitor | None = None
    prior_sigterm: Any = None
    try:
        listener = _new_bootstrap_listener(lane_count=board.max_lanes)
        broker = _SparStateOwnerBootstrapBroker(
            channel=listener,
            server=server,
            board=board,
            paths=paths,
            execution_route_policy=route_policy,
        )
        broker.start()
        monitor = _OwnerProjectionMonitor(
            server,
            paths,
            on_failure=broker._terminal_failure,
        )
        monitor.start()
        _bind_bootstrap_launch_plan(
            plan,
            listener=listener,
            store_id=program.store_id,
        )
        argv = list(plan["argv"])
        _apply_configured_board_environment(plan)
        for name in (
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
            "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET",
            "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE",
        ):
            os.environ.pop(name, None)
        prior_sigterm = signal.getsignal(signal.SIGTERM)

        def fail_before_runner_signal_install(signum: int, _frame: Any) -> None:
            raise OperatorError(f"SPAR controller received fail-fast signal {signum}")

        signal.signal(signal.SIGTERM, fail_before_runner_signal_install)
        broker.enable_fail_fast()
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": "supervise",
                    "ready": True,
                    "implement": bool(implement),
                    "identity": identity.to_dict(),
                    "live": ready,
                    "lanes": board.max_lanes,
                    "execution_route_policy_id": route_policy.policy_id,
                    "credential_transport": "private_inherited_socket",
                    "credential_in_environment_argv_or_file": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        returncode = int(multi_supervisor_main(argv))
        signal.signal(signal.SIGTERM, prior_sigterm)
        prior_sigterm = None
        if broker.failure or monitor.failure:
            raise OperatorError("SPAR owner control monitor failed during supervisor execution")
        return returncode
    finally:
        failures: list[str] = []
        if prior_sigterm is not None:
            signal.signal(signal.SIGTERM, prior_sigterm)
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as exc:
                failures.append(type(exc).__name__)
        if broker is not None:
            try:
                broker.stop()
            except Exception as exc:
                failures.append(type(exc).__name__)
        elif listener is not None:
            listener.close()
        try:
            server.stop()
        except Exception as exc:
            failures.append(type(exc).__name__)
        if failures and sys.exc_info()[0] is None:
            raise OperatorError("SPAR supervisor cleanup failed: " + ",".join(failures))


def _owner_liveness(status_payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    identity = status_payload.get("identity")
    if not isinstance(identity, Mapping):
        return "absent"
    birth_payload = identity.get("process_birth")
    if not isinstance(birth_payload, Mapping):
        return "unknown"
    try:
        observed = owner_liveness(ProcessBirthIdentity.from_dict(birth_payload))
    except Exception:
        return "unknown"
    if observed is OwnerLiveness.ALIVE:
        return "alive"
    if observed is OwnerLiveness.DEAD:
        return "dead"
    return "unknown"


def _task_status(connection: Any) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
    ).fetchall()
    counts = {str(row[0]): int(row[1]) for row in rows}
    # The current Quack table transport supports simple scans but can reject a
    # correlated NOT EXISTS plan as unimplemented.  Read the three canonical
    # relations separately and calculate this read-only projection locally;
    # task/dependency/block rows remain authoritative in DuckDB.
    task_rows = connection.execute(
        "SELECT task_cid, task_alias, ordinal, status "
        "FROM tasks ORDER BY ordinal, task_alias"
    ).fetchall()
    dependency_rows = connection.execute(
        "SELECT task_cid, dependency_task_cid FROM task_dependencies"
    ).fetchall()
    blocked_rows = connection.execute(
        "SELECT task_cid FROM task_blocks WHERE state = 'active'"
    ).fetchall()
    status_by_cid = {str(row[0]): str(row[3]) for row in task_rows}
    dependencies_by_cid: dict[str, list[str]] = {}
    for row in dependency_rows:
        dependencies_by_cid.setdefault(str(row[0]), []).append(str(row[1]))
    actively_blocked = {str(row[0]) for row in blocked_rows}
    ready_ids = [
        str(row[1])
        for row in task_rows
        if str(row[3]) in READY_STATUSES
        and str(row[0]) not in actively_blocked
        and all(
            status_by_cid.get(dependency) in COMPLETED_STATUSES
            for dependency in dependencies_by_cid.get(str(row[0]), ())
        )
    ][:100]
    active_rows = connection.execute(
        "SELECT task_alias FROM tasks WHERE status IN (?, ?, ?) "
        "ORDER BY ordinal, task_alias LIMIT 100",
        list(ACTIVE_STATUSES),
    ).fetchall()
    return {
        "status_counts": counts,
        "dependency_ready_task_ids": ready_ids,
        "active_task_ids": [str(row[0]) for row in active_rows],
        "blocked_count": int(counts.get("blocked", 0)),
        "terminal_count": sum(counts.get(item, 0) for item in TERMINAL_STATUSES),
        "task_count": sum(counts.values()),
    }


def _read_live_projection(
    paths: Mapping[str, Path], owner_status: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _json_object(paths["owner"] / "spar-live-projection.json")
    claimed = str(payload.get("projection_cid") or "")
    unsigned = dict(payload)
    unsigned.pop("projection_cid", None)
    identity = owner_status.get("identity")
    expected_birth = (
        str(identity.get("process_birth_id") or "")
        if isinstance(identity, Mapping)
        else ""
    )
    observed_at = payload.get("observed_at_unix_ns")
    task_projection = payload.get("task_projection")
    projection_fields = {
        "status_counts",
        "dependency_ready_task_ids",
        "active_task_ids",
        "blocked_count",
        "terminal_count",
        "task_count",
    }
    projection_shape_valid = bool(
        isinstance(task_projection, Mapping)
        and set(task_projection) == projection_fields
        and isinstance(task_projection.get("status_counts"), Mapping)
        and all(
            isinstance(key, str)
            and type(value) is int
            and value >= 0
            for key, value in task_projection.get("status_counts", {}).items()
        )
        and all(
            isinstance(task_projection.get(name), list)
            and all(isinstance(item, str) and item for item in task_projection[name])
            and len(task_projection[name]) == len(set(task_projection[name]))
            for name in ("dependency_ready_task_ids", "active_task_ids")
        )
        and all(
            type(task_projection.get(name)) is int
            and task_projection[name] >= 0
            for name in ("blocked_count", "terminal_count", "task_count")
        )
        and task_projection.get("task_count")
        == sum(task_projection.get("status_counts", {}).values())
        and task_projection.get("blocked_count")
        == task_projection.get("status_counts", {}).get("blocked", 0)
    )
    if (
        set(payload)
        != {
            "schema",
            "observed_at_unix_ns",
            "owner_process_birth_id",
            "server_id",
            "store_id",
            "generation",
            "schema_revision",
            "quack_authenticated_live_query",
            "task_projection",
            "projection_cid",
        }
        or payload.get("schema") != "spar/live-owner-projection@1"
        or claimed != _identity(unsigned)
        or type(observed_at) is not int
        or observed_at > time.time_ns() + 5_000_000_000
        or time.time_ns() - observed_at > 10_000_000_000
        or not expected_birth
        or payload.get("owner_process_birth_id") != expected_birth
        or not isinstance(identity, Mapping)
        or payload.get("server_id") != identity.get("server_id")
        or payload.get("store_id") != identity.get("store_id")
        or payload.get("generation") != identity.get("generation")
        or payload.get("schema_revision") != identity.get("schema_revision")
        or payload.get("quack_authenticated_live_query") is not True
        or not projection_shape_valid
    ):
        raise OperatorError("live owner task projection is stale or invalid")
    return dict(task_projection)


def status(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    board, _config = _load_config(config_path)
    paths = _runtime_paths(board)
    state_status_path = paths["owner"] / "quack-state-server.status.json"
    owner_status: dict[str, Any] = {}
    if state_status_path.is_file():
        try:
            owner_status = _json_object(state_status_path)
        except OperatorError:
            owner_status = {"lifecycle": "malformed"}
    liveness = _owner_liveness(owner_status)
    lifecycle = str(owner_status.get("lifecycle") or "absent")
    live_ready = lifecycle == "ready" and liveness == "alive"
    task_projection: dict[str, Any] = {
        "available": False,
        "reason_code": "control_plane_unavailable",
    }
    connection = None
    try:
        if live_ready:
            task_projection = {
                **_read_live_projection(paths, owner_status),
                "available": True,
                "transport": "exclusive_owner_authenticated_quack_projection",
                "authoritative": False,
                "authority_source": "DuckDB/DatabaseTaskSource@1",
            }
        elif paths["database"].is_file() and liveness in {"absent", "dead"}:
            connection = open_duckdb_connection(paths["database"])
            task_projection = {
                "available": True,
                "transport": "direct_offline",
                **_task_status(connection),
            }
    except Exception as exc:
        task_projection = {
            "available": False,
            "reason_code": "control_plane_probe_failed",
            "error_class": type(exc).__name__,
        }
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
    ducklake: dict[str, Any] = {
        "status": "absent",
        "authoritative": False,
        "scheduler_gate": False,
    }
    if paths["ducklake_receipt"].is_file():
        try:
            observed = _json_object(paths["ducklake_receipt"])
            ducklake = {
                "status": str(observed.get("status") or "unknown"),
                "authoritative": False,
                "scheduler_gate": False,
                "projection_receipt_id": str(
                    observed.get("projection_receipt_id") or ""
                ),
            }
        except OperatorError:
            ducklake["status"] = "malformed"
    return {
        "schema": OPERATOR_SCHEMA,
        "command": "status",
        "materialized": paths["database"].is_file()
        and paths["bootstrap_receipt"].is_file(),
        "state_owner": {
            "ready": live_ready,
            "lifecycle": lifecycle,
            "liveness": liveness,
            "identity": owner_status.get("identity"),
        },
        "task_authority": task_projection,
        "ducklake_projection": ducklake,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="repository-relative or absolute configured-board JSON",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "materialize",
        help="seal the committed Markdown bootstrap into DuckDB and DuckLake",
    )
    commands.add_parser(
        "state-owner",
        help="serve the materialized DuckDB authority through fenced loopback Quack",
    )
    supervise_parser = commands.add_parser(
        "supervise",
        help="run the native Quack owner and configured-board scheduler together",
    )
    supervise_parser.add_argument(
        "--implement",
        action="store_true",
        help="authorize implementation-provider dispatch",
    )
    supervise_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the exact native launch without starting owner or workers",
    )
    supervise_parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float("inf"),
        help="bounded supervisor lifetime; defaults to the board terminal",
    )
    status_parser = commands.add_parser(
        "status",
        help="report owner liveness and durable task readiness without exposing tokens",
    )
    status_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit nonzero unless Quack is live and task authority is queryable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        if arguments.command == "materialize":
            result = materialize(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if arguments.command == "state-owner":
            return state_owner(config_path)
        if arguments.command == "supervise":
            return supervise(
                config_path,
                implement=bool(arguments.implement),
                dry_run=bool(arguments.dry_run),
                duration_seconds=float(arguments.duration_seconds),
            )
        if arguments.command == "status":
            result = status(config_path)
            print(json.dumps(result, indent=2, sort_keys=True))
            if arguments.require_ready and not (
                result["state_owner"]["ready"]
                and result["task_authority"].get("available") is True
            ):
                return 1
            return 0
        raise OperatorError(f"unsupported command: {arguments.command}")
    except OperatorError as exc:
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        # Third-party transport exception text is not a trusted secret-
        # redaction surface, so unexpected failures publish only their class.
        print(
            json.dumps(
                {
                    "schema": OPERATOR_SCHEMA,
                    "command": str(arguments.command),
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": "operation failed closed",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
