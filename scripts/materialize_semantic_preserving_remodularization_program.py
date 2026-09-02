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
import errno
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
LAUNCH_SOURCE_FOREST_CURRENT_NAME: Final = "current.json"
LAUNCH_SOURCE_AMENDMENT_CURRENT_NAME: Final = "current.json"
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


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


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
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
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
            True,
        ),
        (
            "ipfs_kit",
            ("ipfs_kit_submodule_path", "kit_submodule_path"),
            ("ipfs_kit_planning_revision", "kit_planning_revision"),
            True,
        ),
        (
            "mcp_plus_plus",
            ("mcp_plus_plus_submodule_path",),
            ("mcp_plus_plus_planning_revision",),
            False,
        ),
    )
    for (
        prefix,
        path_fields,
        revision_fields,
        permit_accepted_descendants,
    ) in configured_repositories:
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
            or not tree
        ):
            raise OperatorError(f"{prefix} nested revision is unavailable")
        planning_revision = str(raw_revision)
        resolved_planning_revision = subprocess.run(
            [
                "git",
                "rev-parse",
                "--verify",
                f"{planning_revision}^{{commit}}",
            ],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        planning_is_ancestor = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                planning_revision,
                revision,
            ],
            cwd=nested_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            resolved_planning_revision.returncode != 0
            or resolved_planning_revision.stdout.strip()
            != planning_revision
            or planning_is_ancestor.returncode != 0
            or (
                not permit_accepted_descendants
                and revision != planning_revision
            )
        ):
            policy = (
                "does not descend from its seal"
                if permit_accepted_descendants
                else "differs from its exact read-only seal"
            )
            raise OperatorError(f"{prefix} nested revision {policy}")
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
                "planning_revision": planning_revision,
                "planning_revision_is_ancestor": True,
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
        "launch_source_forest_dir": evidence / "launch" / "source-forest",
        "launch_source_forest_current": (
            evidence
            / "launch"
            / "source-forest"
            / LAUNCH_SOURCE_FOREST_CURRENT_NAME
        ),
        "launch_source_amendment_dir": evidence / "launch" / "source-amendment",
        "launch_source_amendment_current": (
            evidence
            / "launch"
            / "source-amendment"
            / LAUNCH_SOURCE_AMENDMENT_CURRENT_NAME
        ),
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
        paths["launch_source_forest_dir"],
        paths["launch_source_amendment_dir"],
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


def _launch_source_forest_receipt(
    *,
    source_head: str,
    repository_tree: str,
    source_forest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one closed immutable launch-forest receipt."""
    receipt = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "semantic-preserving-remodularization-launch-source-forest@1"
        ),
        "source_head": source_head,
        "repository_tree": repository_tree,
        "source_forest_root": source_forest.get("source_forest_root"),
        "source_forest": dict(source_forest),
    }
    receipt["receipt_id"] = _identity(receipt)
    return receipt


def _record_launch_source_forest(
    paths: Mapping[str, Path],
    *,
    source_head: str,
    repository_tree: str,
    source_forest: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an immutable launch forest and an atomic current pointer."""

    receipt = _launch_source_forest_receipt(
        source_head=source_head,
        repository_tree=repository_tree,
        source_forest=source_forest,
    )
    receipt_path = paths["launch_source_forest_dir"] / (
        str(receipt["receipt_id"]).removeprefix("sha256:") + ".json"
    )
    if receipt_path.exists():
        if _json_object(receipt_path) != receipt:
            raise OperatorError("launch source-forest receipt identity conflicts")
    else:
        _atomic_json(receipt_path, receipt)

    current = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "semantic-preserving-remodularization-current-source-forest@1"
        ),
        "receipt_id": receipt["receipt_id"],
        "source_forest_root": source_forest.get("source_forest_root"),
        "source_head": source_head,
        "repository_tree": repository_tree,
    }
    current["current_pointer_id"] = _identity(current)
    _atomic_json(paths["launch_source_forest_current"], current)
    return {**receipt, "current_pointer": current}


def _database_tasks(source: Any) -> tuple[Any, ...]:
    tasks: list[Any] = []
    cursor = ""
    while True:
        page = source.list_tasks(cursor=cursor, limit=500)
        tasks.extend(page.tasks)
        cursor = page.next_cursor
        if not cursor:
            break
    if len(tasks) != int(source.snapshot().task_count):
        raise OperatorError("launch task population is incomplete")
    return tuple(tasks)


def _task_history_guard(source: Any, tasks: Sequence[Any]) -> str:
    material = []
    for task in sorted(tasks, key=lambda item: item.task_cid):
        material.append(
            {
                "task": _plain_json(task.to_dict()),
                "history": _plain_json(
                    source.task_revision_history_projection(task.task_cid)
                ),
            }
        )
    return _identity(
        {
            "schema": "spar/launch-task-history-guard@1",
            "tasks": material,
        }
    )


def _launch_source_amendment(
    *,
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_head: str,
    repository_tree: str,
    source_forest_receipt: Mapping[str, Any],
    plan_alias: str,
    parent_plan_revision: int,
    predecessor_amendment_id: str | None,
    task_contract_set_id: str,
) -> Any:
    """Build one route-independent current-source plan amendment."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.launch_source_amendment import (
        LaunchSourceAmendment,
        LaunchSourceAmendmentError,
    )

    bootstrap = _json_object(paths["bootstrap_receipt"])
    bootstrap_id = str(bootstrap.get("bootstrap_receipt_id") or "")
    bootstrap_body = dict(bootstrap)
    bootstrap_body.pop("bootstrap_receipt_id", None)
    if not bootstrap_id or _identity(bootstrap_body) != bootstrap_id:
        raise OperatorError("bootstrap receipt identity does not rehash")
    bootstrap_head = str(bootstrap.get("source_head") or "")
    bootstrap_tree = str(bootstrap.get("repository_tree_id") or "")
    bootstrap_plan = str(bootstrap.get("plan_root_cid") or "")
    if (
        not bootstrap_head
        or not bootstrap_tree
        or not bootstrap_plan
    ):
        raise OperatorError("launch bootstrap identity is incomplete")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", bootstrap_head, source_head],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise OperatorError("launch source does not descend from its bootstrap")

    source_paths = {
        "config": board.config_path,
        "taskboard": board.path(board.taskboard_path),
        "objectives": board.path(board.objectives_path),
        "plan": board.path(board.plan_path),
        "validator": board.path(board.validator_path),
    }
    current_source_ids = {
        name: _identity(_tracked_bytes(path, head=source_head))
        for name, path in sorted(source_paths.items())
    }
    bootstrap_source_ids = bootstrap.get("source_identities")
    if (
        not isinstance(bootstrap_source_ids, Mapping)
        or set(bootstrap_source_ids) != set(source_paths)
    ):
        raise OperatorError("bootstrap source identity inventory is incomplete")
    for name in ("taskboard", "objectives", "plan", "validator"):
        if current_source_ids[name] != bootstrap_source_ids.get(name):
            raise OperatorError(
                f"immutable {name} changed outside the R1 task authority"
            )

    launch_receipt_fields = {
        "schema",
        "source_head",
        "repository_tree",
        "source_forest_root",
        "source_forest",
        "receipt_id",
    }
    if not isinstance(source_forest_receipt, Mapping) or set(
        source_forest_receipt
    ) != launch_receipt_fields:
        raise OperatorError("launch source-forest receipt schema is invalid")
    launch_receipt = dict(source_forest_receipt)
    launch_receipt_id = str(source_forest_receipt.get("receipt_id") or "")
    launch_receipt_body = dict(launch_receipt)
    launch_receipt_body.pop("receipt_id", None)
    launch_source_forest = launch_receipt.get("source_forest")
    if (
        not launch_receipt_id
        or _identity(launch_receipt_body) != launch_receipt_id
        or launch_receipt.get("source_head") != source_head
        or launch_receipt.get("repository_tree") != repository_tree
        or not isinstance(launch_source_forest, Mapping)
        or launch_receipt.get("source_forest_root")
        != launch_source_forest.get("source_forest_root")
    ):
        raise OperatorError("launch source-forest receipt does not rehash")

    seal_path = _safe_path(
        ROOT,
        config.get("dependency_seal_path"),
        field="dependency_seal_path",
    )
    dependency_seal = _json_object(seal_path)
    dependency_seal_id = str(dependency_seal.get("seal_cid") or "")
    dependency_seal_body = dict(dependency_seal)
    dependency_seal_body.pop("seal_cid", None)
    if (
        not dependency_seal_id
        or _identity(dependency_seal_body) != dependency_seal_id
        or config.get("dependency_seal_cid") != dependency_seal_id
    ):
        raise OperatorError("launch dependency seal identity is invalid")

    try:
        amendment = LaunchSourceAmendment(
            board_namespace=board.board_namespace,
            plan_alias=plan_alias,
            bootstrap_receipt_id=bootstrap_id,
            bootstrap_plan_root_cid=bootstrap_plan,
            bootstrap_source_head=bootstrap_head,
            bootstrap_repository_tree_id=bootstrap_tree,
            launch_source_forest_receipt_id=launch_receipt_id,
            launch_source_forest_root=str(
                launch_receipt.get("source_forest_root") or ""
            ),
            launch_source_forest_receipt=launch_receipt,
            launch_source_head=source_head,
            launch_repository_tree_id=repository_tree,
            immutable_objectives_cid=str(bootstrap_source_ids["objectives"]),
            immutable_plan_cid=str(bootstrap_source_ids["plan"]),
            immutable_taskboard_cid=str(bootstrap_source_ids["taskboard"]),
            immutable_validator_cid=str(bootstrap_source_ids["validator"]),
            bootstrap_config_cid=str(bootstrap_source_ids["config"]),
            launch_config_cid=current_source_ids["config"],
            dependency_seal_cid=dependency_seal_id,
            task_contract_set_cid=task_contract_set_id,
            parent_plan_revision=parent_plan_revision,
            amended_plan_revision=parent_plan_revision + 1,
            predecessor_amendment_id=predecessor_amendment_id,
        )
        amendment.validate_launch_git(
            source_head=source_head,
            repository_tree_id=repository_tree,
        )
    except LaunchSourceAmendmentError as exc:
        raise OperatorError("launch source amendment is invalid") from exc
    return amendment


def _assert_launch_route_population(
    *,
    tasks: Sequence[Any],
    execution_route_policy: Any,
    plan_cid: str,
    repository_tree_id: str,
) -> Mapping[str, Any]:
    """Require live tasks to match the sealed launch route identity.

    Idle repair CAS advances revision and operational body after the route is
    sealed. Contract CID is required only at the sealed revision so SPAR-018
    requeue and SPAR-017 validation do not mint a new launch population.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        TaskExecutionRoutePolicy,
        task_execution_contract_cid,
    )

    if not isinstance(execution_route_policy, TaskExecutionRoutePolicy):
        raise OperatorError(
            "launch task population differs from its immutable execution route"
        )
    route_entries = execution_route_policy.entries_by_cid
    if (
        not route_entries
        or execution_route_policy.plan_root_cid != plan_cid
        or execution_route_policy.repository_tree_id != repository_tree_id
        or set(route_entries) != {task.task_cid for task in tasks}
    ):
        raise OperatorError(
            "launch task population differs from its immutable execution route"
        )
    for task in tasks:
        entry = route_entries[task.task_cid]
        if task.task_alias != entry.task_alias or int(task.revision) < int(
            entry.task_revision
        ):
            raise OperatorError(
                "launch task population differs from its immutable execution route"
            )
        if int(task.revision) == int(entry.task_revision) and (
            task_execution_contract_cid(task) != entry.task_contract_cid
        ):
            raise OperatorError(
                "launch task population differs from its immutable execution route"
            )
    return route_entries


def _launch_source_amendment_context(
    *,
    source: Any,
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_head: str,
    repository_tree: str,
    source_forest_receipt: Mapping[str, Any],
    execution_route_policy: Any,
) -> tuple[Any, bool, Mapping[str, Any], tuple[Any, ...]]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.launch_source_amendment import (
        LaunchSourceAmendment,
        LaunchSourceAmendmentError,
        task_contract_set_cid_from_route_entries,
    )

    bootstrap = _json_object(paths["bootstrap_receipt"])
    plan_cid = str(bootstrap.get("plan_root_cid") or "")
    raw_plan = source.plans.get(plan_cid)
    if raw_plan is None:
        raise OperatorError("immutable bootstrap plan is absent from DuckDB")
    plan = _plain_json(raw_plan)
    expected_alias = str(config.get("accepted_plan_revision_alias") or "")
    if (
        plan.get("plan_cid") != plan_cid
        or plan.get("plan_alias") != expected_alias
        or plan.get("status") != "active"
        or isinstance(plan.get("revision"), bool)
        or not isinstance(plan.get("revision"), int)
        or int(plan["revision"]) < 1
    ):
        raise OperatorError("DuckDB bootstrap plan head is not the active R1 plan")
    plan_body = plan.get("body")
    plan_revisions = tuple(
        _plain_json(item) for item in source.plans.list_revisions(plan_cid)
    )
    if (
        len(plan_revisions) != int(plan["revision"])
        or any(
            item.get("plan_cid") != plan_cid
            or item.get("revision") != index
            or not isinstance(item.get("body"), Mapping)
            for index, item in enumerate(plan_revisions, start=1)
        )
        or plan_revisions[-1].get("body") != plan_body
    ):
        raise OperatorError("DuckDB plan revision lineage is incomplete or divergent")
    if not isinstance(plan_body, Mapping) or any(
        plan_body.get(name) != bootstrap.get(value)
        for name, value in (
            ("plan_cid", "plan_root_cid"),
            ("repository_tree_id", "repository_tree_id"),
            ("source_head", "source_head"),
        )
    ):
        raise OperatorError("DuckDB plan body differs from the immutable bootstrap")

    current: LaunchSourceAmendment | None = None
    raw_current = plan_body.get("launch_source_amendment")
    current_id = plan_body.get("launch_source_amendment_id")
    if raw_current is not None or current_id is not None:
        if not isinstance(raw_current, Mapping):
            raise OperatorError("active launch source amendment is malformed")
        try:
            current = LaunchSourceAmendment.from_dict(raw_current)
        except LaunchSourceAmendmentError as exc:
            raise OperatorError("active launch source amendment is invalid") from exc
        if current_id != current.amendment_id:
            raise OperatorError("active launch source amendment identity differs")
        predecessor_body = (
            plan_revisions[-2]["body"] if len(plan_revisions) > 1 else {}
        )
        predecessor_raw = predecessor_body.get("launch_source_amendment")
        predecessor_id = predecessor_body.get("launch_source_amendment_id")
        if current.predecessor_amendment_id is None:
            if predecessor_raw is not None or predecessor_id is not None:
                raise OperatorError(
                    "first launch source amendment has a predecessor revision"
                )
        else:
            if not isinstance(predecessor_raw, Mapping):
                raise OperatorError("launch source amendment predecessor is absent")
            try:
                predecessor = LaunchSourceAmendment.from_dict(predecessor_raw)
            except LaunchSourceAmendmentError as exc:
                raise OperatorError(
                    "launch source amendment predecessor is invalid"
                ) from exc
            if (
                predecessor_id != predecessor.amendment_id
                or predecessor.amendment_id != current.predecessor_amendment_id
                or predecessor.amended_plan_revision != int(plan["revision"]) - 1
            ):
                raise OperatorError(
                    "launch source amendment predecessor lineage differs"
                )

    tasks = _database_tasks(source)
    route_entries = _assert_launch_route_population(
        tasks=tasks,
        execution_route_policy=execution_route_policy,
        plan_cid=plan_cid,
        repository_tree_id=str(bootstrap.get("repository_tree_id") or ""),
    )
    try:
        contract_set_id = task_contract_set_cid_from_route_entries(
            route_entries.values()
        )
    except LaunchSourceAmendmentError as exc:
        raise OperatorError(
            "launch task population differs from its immutable execution route"
        ) from exc
    if current is not None:
        predecessor_ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                current.launch_source_head,
                source_head,
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if predecessor_ancestry.returncode != 0:
            raise OperatorError(
                "launch source does not descend from its predecessor amendment"
            )
    if current is not None and (
        current.board_namespace != board.board_namespace
        or current.plan_alias != expected_alias
        or current.bootstrap_receipt_id
        != str(bootstrap.get("bootstrap_receipt_id") or "")
        or current.bootstrap_plan_root_cid != plan_cid
        or current.bootstrap_source_head != str(bootstrap.get("source_head") or "")
        or current.bootstrap_repository_tree_id
        != str(bootstrap.get("repository_tree_id") or "")
        or current.parent_plan_revision != int(plan["revision"]) - 1
        or current.amended_plan_revision != int(plan["revision"])
        or current.task_contract_set_cid != contract_set_id
    ):
        raise OperatorError(
            "active launch source amendment differs from its plan lineage"
        )
    candidate = _launch_source_amendment(
        board=board,
        config=config,
        paths=paths,
        source_head=source_head,
        repository_tree=repository_tree,
        source_forest_receipt=source_forest_receipt,
        plan_alias=expected_alias,
        parent_plan_revision=int(plan["revision"]),
        predecessor_amendment_id=(
            current.amendment_id if current is not None else None
        ),
        task_contract_set_id=contract_set_id,
    )
    if current is not None and current.same_launch_generation(candidate):
        return current, True, plan, tasks
    return candidate, False, plan, tasks


def _prepare_launch_source_amendment(
    *,
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_head: str,
    repository_tree: str,
    source_forest_receipt: Mapping[str, Any],
    execution_route_policy: Any,
) -> tuple[Any, bool]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    bootstrap = _json_object(paths["bootstrap_receipt"])
    with DatabaseTaskSource(
        paths["database"],
        owner_id="spar-source-amendment:prepare",
        install_schema=False,
        repository_tree_id=str(bootstrap["repository_tree_id"]),
        plan_root_cid=str(bootstrap["plan_root_cid"]),
    ) as source:
        amendment, replay, _plan, _tasks = _launch_source_amendment_context(
            source=source,
            board=board,
            config=config,
            paths=paths,
            source_head=source_head,
            repository_tree=repository_tree,
            source_forest_receipt=source_forest_receipt,
            execution_route_policy=execution_route_policy,
        )
    return amendment, replay


def _admit_launch_source_amendment(
    *,
    board: Any,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_head: str,
    repository_tree: str,
    source_forest_receipt: Mapping[str, Any],
    execution_route_policy: Any,
) -> tuple[Any, dict[str, Any]]:
    """CAS-append and read back one amendment through current plan authority."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    bootstrap = _json_object(paths["bootstrap_receipt"])
    with DatabaseTaskSource(
        paths["database"],
        owner_id="spar-source-amendment:single-writer",
        install_schema=False,
        repository_tree_id=str(bootstrap["repository_tree_id"]),
        plan_root_cid=str(bootstrap["plan_root_cid"]),
    ) as source:
        amendment, replay, plan, tasks = _launch_source_amendment_context(
            source=source,
            board=board,
            config=config,
            paths=paths,
            source_head=source_head,
            repository_tree=repository_tree,
            source_forest_receipt=source_forest_receipt,
            execution_route_policy=execution_route_policy,
        )
        plan_cid = str(bootstrap["plan_root_cid"])
        before_revisions = source.plans.list_revisions(plan_cid)
        before_snapshot = source.snapshot()
        before_task_guard = _task_history_guard(source, tasks)
        if replay:
            if (
                not before_revisions
                or int(before_revisions[-1]["revision"]) != int(plan["revision"])
                or _plain_json(before_revisions[-1]["body"])
                != _plain_json(plan["body"])
            ):
                raise OperatorError(
                    "active launch amendment lacks exact plan-revision readback"
                )
            return amendment, {
                "schema": "spar/launch-source-amendment-admission@1",
                "authoritative_store": "DuckDB/PlanRevisionRepository@1",
                "amendment_id": amendment.amendment_id,
                "plan_cid": plan_cid,
                "plan_revision": int(plan["revision"]),
                "event_id": "",
                "idempotent_replay": True,
                "task_history_guard": before_task_guard,
            }

        delta = {
            "schema": "spar/control-amendment-delta@1",
            "operation": "launch_source_amendment",
            "amendment_id": amendment.amendment_id,
            "predecessor_amendment_id": amendment.predecessor_amendment_id,
        }
        receipt = source.plans.append_revision(
            plan_cid=plan_cid,
            expected_revision=int(plan["revision"]),
            body={
                "launch_source_amendment": amendment.to_dict(),
                "launch_source_amendment_id": amendment.amendment_id,
            },
            delta=delta,
        )
        after_plan_raw = source.plans.get(plan_cid)
        after_revisions = source.plans.list_revisions(plan_cid)
        after_tasks = _database_tasks(source)
        after_snapshot = source.snapshot()
        after_task_guard = _task_history_guard(source, after_tasks)
        if after_plan_raw is None:
            raise OperatorError("launch amendment plan disappeared after CAS")
        after_plan = _plain_json(after_plan_raw)
        after_body = after_plan.get("body")
        if (
            after_plan.get("plan_alias") != plan.get("plan_alias")
            or after_plan.get("goal_cid") != plan.get("goal_cid")
            or after_plan.get("status") != plan.get("status")
            or int(after_plan.get("revision") or 0)
            != amendment.amended_plan_revision
            or len(after_revisions) != len(before_revisions) + 1
            or int(after_revisions[-1]["revision"])
            != amendment.amended_plan_revision
            or _plain_json(after_revisions[-1]["body"]) != after_body
            or not isinstance(after_body, Mapping)
            or after_body.get("launch_source_amendment") != amendment.to_dict()
            or after_body.get("launch_source_amendment_id")
            != amendment.amendment_id
            or after_body.get("plan_cid") != plan_cid
            or after_body.get("repository_tree_id")
            != bootstrap["repository_tree_id"]
            or after_body.get("source_head") != bootstrap["source_head"]
            or before_task_guard != after_task_guard
            or int(after_snapshot.event_cursor)
            != int(before_snapshot.event_cursor) + 1
            or int(receipt.details.get("revision") or 0)
            != amendment.amended_plan_revision
        ):
            raise OperatorError("launch source amendment authoritative readback failed")
        return amendment, {
            "schema": "spar/launch-source-amendment-admission@1",
            "authoritative_store": "DuckDB/PlanRevisionRepository@1",
            "amendment_id": amendment.amendment_id,
            "plan_cid": plan_cid,
            "plan_revision": amendment.amended_plan_revision,
            "event_id": receipt.event_id,
            "event_sequence": int(receipt.global_sequence),
            "idempotent_replay": False,
            "task_history_guard": after_task_guard,
        }


def _record_launch_source_amendment(
    paths: Mapping[str, Path],
    amendment: Any,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist a non-authoritative diagnostic projection after DB readback."""

    receipt = {
        "schema": "spar/launch-source-amendment-projection@1",
        "authoritative": False,
        "amendment": amendment.to_dict(),
        "admission": dict(admission),
    }
    receipt["projection_id"] = _identity(receipt)
    receipt_path = paths["launch_source_amendment_dir"] / (
        str(amendment.amendment_id).replace(":", "_") + ".json"
    )
    if receipt_path.exists():
        existing = _json_object(receipt_path)
        existing_admission = existing.get("admission")
        if (
            existing.get("amendment") != amendment.to_dict()
            or not isinstance(existing_admission, Mapping)
            or existing_admission.get("amendment_id") != amendment.amendment_id
            or existing_admission.get("plan_revision")
            != admission.get("plan_revision")
        ):
            raise OperatorError("launch source amendment projection conflicts")
        receipt = existing
    else:
        _atomic_json(receipt_path, receipt)
    current = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "semantic-preserving-remodularization-current-source-amendment@1"
        ),
        "authoritative": False,
        "amendment_id": amendment.amendment_id,
        "authoritative_plan_revision": admission["plan_revision"],
        "launch_source_head": amendment.launch_source_head,
        "launch_repository_tree_id": amendment.launch_repository_tree_id,
        "launch_source_forest_root": amendment.launch_source_forest_root,
    }
    current["current_pointer_id"] = _identity(current)
    _atomic_json(paths["launch_source_amendment_current"], current)
    return receipt


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


def _owner_command_inbox(program: Any) -> Path:
    """Return the mutation inbox lanes already write, not quack-owner/mutations.

    SPAR workers bind ``IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR`` to
    ``runtime_registry_path/mutations``.  The exclusive owner used to drain
    ``quack-owner/mutations``, so idle compare_and_set_status timed out.
    """

    registry = Path(str(program.runtime_registry_path or "")).expanduser()
    if not str(registry):
        raise OperatorError("SPAR owner command inbox requires runtime_registry_path")
    if not registry.is_absolute():
        registry = ROOT / registry
    inbox = registry.resolve() / "mutations"
    inbox.mkdir(parents=True, exist_ok=True)
    os.chmod(inbox, 0o700)
    return inbox


def _bind_owner_command_inbox(server: Any, inbox: Path) -> None:
    server._mutation_inbox_override = inbox


def _owner_connection(path: Path) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_state_owner_connection,
    )

    # wrap() leaves DuckDBConnection.path = None, which made poisoned-handle
    # recovery return False forever and never restart quack_serve.
    return open_quack_state_owner_connection(path)


def _owner_database_path(server: Any, connection: Any | None) -> Path | None:
    path = getattr(connection, "path", None)
    if path is not None:
        return Path(path)
    config = getattr(server, "config", None)
    fallback = getattr(config, "database_path", None) if config is not None else None
    if fallback is None:
        return None
    return Path(fallback)


def _normalized_owner_dml(sql: str) -> str:
    normalized = " ".join(str(sql or "").strip().upper().split())
    if not normalized.startswith(OWNER_DML_PREFIXES):
        raise OperatorError("mutation inbox accepts only the closed owner-DML vocabulary")
    if ";" in normalized.rstrip(";"):
        raise OperatorError("mutation inbox accepts exactly one SQL statement")
    return normalized


def _process_mutations(server: Any, mutation_dir: Path) -> None:
    mutation_dir.mkdir(parents=True, exist_ok=True)
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        apply_owner_command_payload,
    )

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
            # Signed owner-command envelopes are applied by
            # process_mutation_inbox. Do not treat them as SQL and unlink.
            if str(payload.get("schema") or "") == QUACK_OWNER_COMMAND_REQUEST_SCHEMA:
                continue
            owner_connection = getattr(server, "_connection", None)
            if owner_connection is None:
                raise OperatorError("state-owner connection is unavailable")
            op = str(payload.get("op") or "").strip()
            if op:
                lock = getattr(server, "_owner_transaction_lock", None)
                if lock is not None:
                    with lock:
                        result = apply_owner_command_payload(
                            owner_connection,
                            payload,
                        )
                else:
                    result = apply_owner_command_payload(
                        owner_connection,
                        payload,
                    )
                _atomic_json(done, dict(result))
                try:
                    request.unlink()
                except FileNotFoundError:
                    pass
                continue
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
    _bind_owner_command_inbox(server, _owner_command_inbox(program))
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


def _owner_listener_ready(server: Any) -> bool:
    """Return whether the Quack listener is bound.

    Only ``ECONNREFUSED`` proves the port is closed.  A handshake timeout or
    reset is the full accept backlog of attached lanes, not a down listener.
    Treating those as down bounced ``quack_serve`` and poisoned every typed
    client with ``DuckDBConnectionPolicyError``.
    """

    config = getattr(server, "config", None)
    host = ""
    port = 0
    if config is not None:
        host = str(
            getattr(config, "container_bind_host", "")
            or getattr(config, "host", "")
            or ""
        ).strip()
        port = int(
            getattr(config, "container_port", 0) or getattr(config, "port", 0) or 0
        )
    if port <= 0:
        port = int(getattr(server, "_bound_port", 0) or 0)
    if host in {"", "0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    if not host or port <= 0:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.1):
            return True
    except ConnectionRefusedError:
        return False
    except TimeoutError:
        return True
    except OSError as exc:
        return getattr(exc, "errno", None) != errno.ECONNREFUSED


def _owner_connection_unusable(connection: Any) -> bool:
    """Return whether the exclusive wrapper must be reconnected in place."""

    return (
        getattr(connection, "_poisoned", False) is True
        or getattr(connection, "_closed", False) is True
        or getattr(connection, "_connection", True) is None
    )


def _recoverable_owner_control_error(exc: BaseException) -> bool:
    """Return whether owner poison should reconnect instead of SIGTERM.

    Recovered exclusive-handle reconnect used to be followed by grant renewal
    or persist failing on the same poison marker, which fail-fast SIGTERM'd
    SPAR before SPAR-018 could be claimed.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnectionPolicyError,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TypedStateOwnerAuthorizationError,
    )

    if isinstance(exc, (DuckDBConnectionPolicyError, TypedStateOwnerAuthorizationError)):
        return True
    text = str(exc)
    name = type(exc).__name__
    return name in {
        "DuckDBConnectionPolicyError",
        "TypedStateOwnerAuthorizationError",
        "QuackClientError",
        "TransactionError",
    } or (
        "unusable after an uncertain transaction" in text
        or "typed Quack authority binding is no longer live" in text
    )


def _restart_owner_transport(server: Any, *, previous: Any, replacement: Any) -> None:
    transport_connection = getattr(server, "_transport_connection", None)
    if transport_connection not in {previous, None, replacement}:
        return
    server._transport_connection = None
    refresh = getattr(server, "_refresh_read_replica", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass
    if getattr(server, "_transport_connection", None) is None:
        server._transport_connection = replacement


def _recover_poisoned_owner_connection(
    server: Any,
    *,
    force: bool = False,
) -> bool:
    """Replace a poisoned exclusive owner handle without SIGTERM'ing SPAR.

    A single interrupted typed-client transaction marks the shared DuckDB
    wrapper unusable.  The projection monitor previously treated that as a
    terminal owner failure and killed every lane.  Reopening the exclusive
    file owner lets live claims continue.

    ``force=True`` restores a down listener without reconnecting a usable
    writer. Reconnecting a healthy owner bounces ``quack_serve`` and drops
    every attached lane.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_state_owner_connection,
    )

    connection = getattr(server, "_connection", None)
    lock = getattr(server, "_owner_transaction_lock", None)
    path = _owner_database_path(server, connection)
    if connection is None or lock is None or path is None:
        return False
    unusable = _owner_connection_unusable(connection)
    if not force and not unusable:
        return False
    with lock:
        current = getattr(server, "_connection", None)
        current_path = _owner_database_path(server, current)
        if current is None or current_path is None:
            return False
        unusable = _owner_connection_unusable(current)
        if not force and not unusable:
            return False
        replacement = current
        native_replaced = False
        if unusable:
            # Same-process reopen cannot take exclusive_file_lock again: the
            # live handle still holds the path's thread RLock. Reconnect the
            # native owner in place. Fall back to close+open only when
            # reconnect is absent or the exclusive lock was already released.
            reconnect = getattr(current, "reconnect_exclusive_owner", None)
            reconnected = False
            if callable(reconnect) and getattr(current, "path", None) is not None:
                try:
                    reconnect()
                    replacement = current
                    reconnected = True
                except Exception:
                    reconnected = False
            if not reconnected:
                close = getattr(current, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                replacement = open_quack_state_owner_connection(current_path)
            server._connection = replacement
            gateway = getattr(server, "_command_gateway", None)
            if gateway is not None:
                gateway._connection = replacement
            if getattr(server, "_transport_connection", None) in {current, None}:
                server._transport_connection = replacement
            native_replaced = True
        restored_serve = False
        if native_replaced:
            # Only rebind quack_serve after this recover actually replaced
            # the native handle. A false ECONNREFUSED probe used to restart
            # a live serve and drop SPAR-018's typed bind.
            restored_serve = _restart_owner_serve_if_down(
                server,
                getattr(server, "_connection", None),
            )
        return native_replaced or restored_serve


def _restart_owner_serve_if_down(server: Any, connection: Any) -> bool:
    """Bind quack_serve again after exclusive-handle reconnect dropped TCP."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        ServerLifecycle,
    )

    if connection is None or _owner_listener_ready(server):
        return False
    transport = getattr(server, "transport", None)
    start = getattr(transport, "start", None)
    identity = getattr(server, "_identity", None)
    config = getattr(server, "config", None)
    vault = getattr(server, "_vault", None)
    if not callable(start) or identity is None or config is None or vault is None:
        return False
    try:
        secret_handle = config.resolved_secret_handle(
            identity.server_id,
            int(identity.generation),
        )
        token = vault.resolve(secret_handle)
        port = int(
            getattr(server, "_bound_port", 0)
            or getattr(config, "port", 0)
            or 0
        )
        host = str(getattr(config, "host", "") or "127.0.0.1")
        if port <= 0:
            return False
        start(
            connection,
            host=host,
            port=port,
            token=token,
            identity=identity,
        )
        server._transport_connection = connection
        if getattr(server, "_lifecycle", None) is ServerLifecycle.FAILED:
            server._lifecycle = ServerLifecycle.READY
        return True
    except Exception:
        return False


def _owner_task_projection(server: Any) -> dict[str, Any]:
    connection = getattr(server, "_connection", None)
    transaction_lock = getattr(server, "_owner_transaction_lock", None)
    if connection is None or transaction_lock is None:
        raise OperatorError("state-owner task projection connection is unavailable")
    with transaction_lock:
        return _task_status(connection)


def _owner_identity_snapshot(server: Any) -> dict[str, Any]:
    """Return ready fields without executing quack_query on the serve connection."""

    identity = getattr(server, "_identity", None)
    if identity is None:
        raise OperatorError("state-owner identity is unavailable")
    return {
        "process_birth_id": str(identity.process_birth_id),
        "server_id": str(identity.server_id),
        "store_id": str(identity.store_id),
        "generation": int(identity.generation),
        "schema_revision": int(identity.schema_revision),
        "live": True,
    }


def _cached_task_projection(paths: Mapping[str, Path]) -> dict[str, Any]:
    path = paths["owner"] / "spar-live-projection.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    projection = payload.get("task_projection") if isinstance(payload, dict) else None
    return dict(projection) if isinstance(projection, dict) else {}


def _publish_live_projection(server: Any, paths: Mapping[str, Path]) -> dict[str, Any]:
    # Periodic projection must not call server.ready(), TCP-probe the
    # listener, or SELECT on the exclusive serve connection. quack_serve
    # owns that native handle; a second statement poisons it with
    # DuckDBConnectionPolicyError and drops SPAR-018's typed CAS.
    ready = _owner_identity_snapshot(server)
    connection = getattr(server, "_connection", None)
    transport_connection = getattr(server, "_transport_connection", None)
    serve_owns_exclusive = (
        connection is not None
        and transport_connection is not None
        and connection is transport_connection
    )
    if serve_owns_exclusive:
        task_projection = _cached_task_projection(paths)
    else:
        try:
            task_projection = _owner_task_projection(server)
        except Exception:
            task_projection = _cached_task_projection(paths)
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
    mutation_dir = server.mutation_inbox_path()
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
            process_inbox = getattr(server, "process_mutation_inbox", None)
            if callable(process_inbox):
                try:
                    process_inbox()
                except Exception:
                    pass
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
                process_inbox = getattr(server, "process_mutation_inbox", None)
                if callable(process_inbox):
                    try:
                        process_inbox()
                    except Exception:
                        pass
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
                "mutation_dir": str(server.mutation_inbox_path().relative_to(ROOT)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result, _child_returncode = _serve_state_owner(server, paths)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _resume_execution_route_policy(
    *,
    bootstrap: Mapping[str, Any],
    snapshot: Any,
    tasks: Sequence[Any],
    histories_by_task: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Any:
    """Reuse the exact first-launch route when authoritative tasks carry it.

    Operational task revisions advance after claims, retries, and completion,
    while the immutable plan revision can remain unchanged.  Re-sealing those
    later revisions would mint a different policy ID with the same source
    revision and make the daemon reject its own carried retry lineage.  The
    bootstrap projection plus owner-written task receipts are sufficient to
    reconstruct the original policy byte-for-byte; no sidecar or stale local
    checkpoint is admitted as authority.

    A fresh board has no carried bindings and simply seals the current
    population.  Once any carried binding exists, every advanced ordinary task
    must carry the same original policy and all untouched tasks must still be
    at their initial revision.  Any mismatch fails closed.
    """

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskSourceIntegrityError,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
        TaskExecutionRouteBinding,
        TaskExecutionRouteEntry,
        TaskExecutionRoutePolicy,
        task_execution_contract_cid,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        validated_post_merge_retry_predecessor_lineage,
    )

    execution_modes = {
        task.task_alias: GROK_CODEX_EXECUTION_MODE for task in tasks
    }
    current = TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=tasks,
        execution_modes=execution_modes,
    )
    carried_by_task: dict[str, Any] = {}
    route_receipt_fields = {
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
    }
    for task in tasks:
        body = task.body if isinstance(task.body, Mapping) else {}
        receipt = body.get("completion_receipt")
        if not isinstance(receipt, Mapping):
            continue
        present_route_fields = route_receipt_fields.intersection(receipt)
        if not present_route_fields:
            history = (histories_by_task or {}).get(task.task_cid)
            if history is None:
                continue
            try:
                recovered = validated_post_merge_retry_predecessor_lineage(
                    task,
                    history,
                )
            except TaskSourceIntegrityError as exc:
                raise OperatorError(
                    "advanced ordinary task lacks exact post-merge route history"
                ) from exc
            receipt = recovered
            present_route_fields = route_receipt_fields
        if present_route_fields != route_receipt_fields or any(
            receipt.get(field) is None for field in route_receipt_fields
        ):
            raise OperatorError(
                "authoritative task carries a partial execution-route receipt"
            )
        raw_binding = receipt["execution_route_binding"]
        try:
            binding = TaskExecutionRouteBinding.from_dict(raw_binding)
        except TaskSourceIntegrityError as exc:
            raise OperatorError(
                "authoritative task carries a malformed execution-route binding"
            ) from exc
        if (
            binding.task_cid != task.task_cid
            or binding.task_alias != task.task_alias
            or binding.task_revision > task.revision
            or (
                binding.task_revision == task.revision
                and binding.task_contract_cid != task_execution_contract_cid(task)
            )
            or binding.execution_mode != GROK_CODEX_EXECUTION_MODE
            or not isinstance(receipt, Mapping)
            or receipt.get("execution_route_policy_id") != binding.policy_id
            or receipt.get("execution_route_origin_revision")
            != binding.task_revision
        ):
            raise OperatorError(
                "authoritative task execution-route lineage is inconsistent"
            )
        carried_by_task[task.task_cid] = binding

    if not carried_by_task:
        advanced_ordinary = [
            task.task_alias
            for task in tasks
            if task.task_alias != "SPAR-000" and int(task.revision) != 1
        ]
        if advanced_ordinary:
            raise OperatorError(
                "advanced ordinary task lacks carried execution-route lineage"
            )
        return current

    operator_completed = bootstrap.get("operator_completed_task_ids")
    if operator_completed != ["SPAR-000"]:
        raise OperatorError(
            "bootstrap operator-completion identity is not exact"
        )
    projection_cid = str(bootstrap.get("projection_cid") or "")
    if (
        int(bootstrap.get("task_count") or 0) != len(tasks)
        or bootstrap.get("plan_root_cid") != snapshot.plan_root_cid
        or bootstrap.get("repository_tree_id") != snapshot.repository_tree_id
        or not projection_cid
    ):
        raise OperatorError(
            "bootstrap projection differs from the current route population"
        )

    policy_lineages = {
        (
            binding.policy_id,
            binding.plan_root_cid,
            binding.repository_tree_id,
            int(binding.source_revision),
        )
        for binding in carried_by_task.values()
    }
    if len(policy_lineages) != 1:
        raise OperatorError(
            "authoritative tasks carry multiple execution-route policies"
        )
    policy_id, plan_root_cid, repository_tree_id, source_revision = next(
        iter(policy_lineages)
    )
    if (
        plan_root_cid != snapshot.plan_root_cid
        or repository_tree_id != snapshot.repository_tree_id
        or source_revision < 1
        or source_revision > int(snapshot.revision)
    ):
        raise OperatorError(
            "carried execution-route policy is outside the bootstrap lineage"
        )

    entries: list[Any] = []
    for task in tasks:
        binding = carried_by_task.get(task.task_cid)
        if binding is not None:
            entry = TaskExecutionRouteEntry(
                task_cid=binding.task_cid,
                task_alias=binding.task_alias,
                task_revision=binding.task_revision,
                task_contract_cid=binding.task_contract_cid,
                execution_mode=binding.execution_mode,
            )
        else:
            is_operator_bootstrap = (
                task.task_alias == "SPAR-000"
                and task.status in COMPLETED_STATUSES
            )
            if not is_operator_bootstrap and int(task.revision) != 1:
                raise OperatorError(
                    "advanced ordinary task lacks carried execution-route lineage"
                )
            entry = TaskExecutionRouteEntry(
                task_cid=task.task_cid,
                task_alias=task.task_alias,
                task_revision=int(task.revision),
                task_contract_cid=task_execution_contract_cid(task),
                execution_mode=GROK_CODEX_EXECUTION_MODE,
            )
        entries.append(entry)

    try:
        return TaskExecutionRoutePolicy(
            plan_root_cid=plan_root_cid,
            repository_tree_id=repository_tree_id,
            source_revision=source_revision,
            source_projection_cid=projection_cid,
            entries=tuple(sorted(entries, key=lambda item: item.task_cid)),
            policy_id=policy_id,
        )
    except TaskSourceIntegrityError as exc:
        raise OperatorError(
            "original execution-route policy does not reconstruct exactly"
        ) from exc


def _execution_route_policy(paths: Mapping[str, Path]) -> Any:
    """Seal or resume one exact route before the owner takes the DuckDB lease."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        EXECUTION_ROUTE_RECEIPT_FIELDS,
        POST_MERGE_RETRY_RECOVERY_OPERATIONS,
        VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS,
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
        histories_by_task: dict[str, Sequence[Mapping[str, Any]]] = {}
        cursor = ""
        while True:
            page = source.list_tasks(cursor=cursor, limit=500)
            tasks.extend(page.tasks)
            cursor = page.next_cursor
            if not cursor:
                break
        for task in tasks:
            body = task.body if isinstance(task.body, Mapping) else {}
            receipt = body.get("completion_receipt")
            if (
                task.task_alias != "SPAR-000"
                and int(task.revision) > 1
                and task.status == "retrying"
                and isinstance(receipt, Mapping)
                and receipt.get("operation")
                in POST_MERGE_RETRY_RECOVERY_OPERATIONS
                and not set(receipt).intersection(
                    EXECUTION_ROUTE_RECEIPT_FIELDS
                    | VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS
                )
            ):
                history = source.task_revision_history_projection(task.task_cid)
                revisions = history.get("revisions")
                if isinstance(revisions, list):
                    histories_by_task[task.task_cid] = revisions
    if len(tasks) != int(snapshot.task_count):
        raise OperatorError("execution-route task population is incomplete")
    return _resume_execution_route_policy(
        bootstrap=bootstrap,
        snapshot=snapshot,
        tasks=tasks,
        histories_by_task=histories_by_task,
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

    def _terminal_failure(self, exc: BaseException) -> bool:
        """Return True when the broker must stop the owner process.

        After lanes are live, exclusive-owner poison and grant-renewal races
        must reconnect in place.  SIGTERM here killed SPAR ~20s after ready
        (generation 51) before SPAR-018 could be claimed.
        """

        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "spar-owner-broker-failure@1"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[-1000:],
                    "recoverable": _recoverable_owner_control_error(exc),
                    "fail_fast_enabled": self.fail_fast_enabled.is_set(),
                    "stopping": self.stopping.is_set(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if self.fail_fast_enabled.is_set() and not self.stopping.is_set():
            try:
                _recover_poisoned_owner_connection(self.server, force=False)
            except Exception:
                pass
            return False
        with self._lock:
            self.failure = self.failure or type(exc).__name__
        self.ready.set()
        return True

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
                if _recover_poisoned_owner_connection(self.server, force=False):
                    continue
                raise
            except Exception as exc:
                if _recoverable_owner_control_error(exc):
                    _recover_poisoned_owner_connection(self.server, force=False)
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
                if _recoverable_owner_control_error(exc) and not self.stopping.is_set():
                    try:
                        _recover_poisoned_owner_connection(self.server, force=False)
                    except Exception:
                        pass
                    self.stopping.wait(0.25)
                    continue
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
                        if self._terminal_failure(persist_exc):
                            return
                continue
            except OSError as exc:
                if not self.stopping.is_set() and accepted is None:
                    if self._terminal_failure(exc):
                        return
                    self.stopping.wait(0.25)
                continue
            except BaseException as exc:
                if self._terminal_failure(exc):
                    return
                self.stopping.wait(0.25)
                continue
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
        mutation_dir: Path,
        on_failure: Callable[[BaseException], None],
    ) -> None:
        self.server = server
        self.paths = paths
        self.mutation_dir = mutation_dir
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

    def _drain_owner_commands(self) -> None:
        process_inbox = getattr(self.server, "process_mutation_inbox", None)
        if callable(process_inbox):
            try:
                process_inbox()
            except Exception:
                if _owner_connection_unusable(
                    getattr(self.server, "_connection", None)
                ):
                    raise
        _process_mutations(self.server, self.mutation_dir)

    def _run(self) -> None:
        initial = True
        next_projection = 0.0
        while not self.stopping.is_set():
            try:
                self._drain_owner_commands()
                now = time.monotonic()
                if now >= next_projection:
                    _publish_live_projection(self.server, self.paths)
                    next_projection = now + 1.0
                connection = getattr(self.server, "_connection", None)
                if (
                    connection is not None
                    and not _owner_connection_unusable(connection)
                    and not _owner_listener_ready(self.server)
                ):
                    # quack_serve can exit without poisoning the Python
                    # wrapper. Restart it so SPAR-018 typed grants attach.
                    _restart_owner_serve_if_down(self.server, connection)
            except BaseException as exc:
                recovered = False
                recover_error_type = ""
                recover_error = ""
                connection = getattr(self.server, "_connection", None)
                # Never force-recover from a TCP probe. A handshake timeout
                # or false ECONNREFUSED during backlog used to bounce
                # quack_serve and invalidate every typed grant, so SPAR-018
                # never completed CAS. Reconnect only when the exclusive
                # wrapper is actually unusable.
                listener_ready = _owner_listener_ready(self.server)
                try:
                    recovered = _recover_poisoned_owner_connection(
                        self.server,
                        force=False,
                    )
                except Exception as recover_exc:
                    recovered = False
                    recover_error_type = type(recover_exc).__name__
                    recover_error = str(recover_exc)[-1000:]
                if initial and not recovered:
                    self.failure = type(exc).__name__
                    self.ready.set()
                    self.on_failure(exc)
                    return
                print(
                    json.dumps(
                        {
                            "schema": (
                                "ipfs_accelerate_py/agent-supervisor/"
                                "spar-owner-projection-retry@1"
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[-1000:],
                            "recovered": recovered,
                            "recover_error_type": recover_error_type,
                            "recover_error": recover_error,
                            "listener_ready": listener_ready,
                            "connection_path_present": (
                                getattr(connection, "path", None) is not None
                            ),
                            "config_database_path_present": (
                                getattr(
                                    getattr(self.server, "config", None),
                                    "database_path",
                                    None,
                                )
                                is not None
                            ),
                            "initial": initial,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if recovered:
                    self.stopping.wait(0.25)
                    continue
                self.stopping.wait(1.0)
                continue
            if initial:
                initial = False
                self.ready.set()
            self.stopping.wait(0.05)


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
    launch_source_amendment: Any,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        _generic_state_owner_bootstrap_binding,
    )

    argv = list(plan.get("argv") or ())
    amendment_json = launch_source_amendment.to_json()
    for value in (
        "--state-owner-bootstrap-fd",
        str(listener.fileno()),
        "--state-owner-bootstrap-store-id",
        store_id,
        "--launch-source-amendment-json",
        amendment_json,
    ):
        argv.append(f"--common-arg={value}")
    common_args = tuple(
        item.split("=", 1)[1]
        for item in argv
        if item.startswith("--common-arg=")
    )
    if _generic_state_owner_bootstrap_binding(common_args) != listener.fileno():
        raise OperatorError("SPAR bootstrap launch-plan descriptor changed")
    if "--require-launch-source-amendment" not in common_args:
        raise OperatorError("SPAR launch plan does not require its source amendment")
    if any("IPFS_ACCELERATE_AGENT_QUACK_TOKEN=" in item for item in argv):
        raise OperatorError("SPAR bootstrap launch plan contains a raw credential")
    plan["argv"] = argv
    plan["state_owner_bootstrap"] = {
        "transport": "private_inherited_socket",
        "descriptor": listener.fileno(),
        "store_id": store_id,
        "credential_persisted": False,
    }
    plan["launch_source_amendment"] = {
        "amendment_id": launch_source_amendment.amendment_id,
        "bootstrap_plan_root_cid": (
            launch_source_amendment.bootstrap_plan_root_cid
        ),
        "launch_source_head": launch_source_amendment.launch_source_head,
        "launch_repository_tree_id": (
            launch_source_amendment.launch_repository_tree_id
        ),
        "launch_source_forest_root": (
            launch_source_amendment.launch_source_forest_root
        ),
        "task_contract_set_cid": launch_source_amendment.task_contract_set_cid,
        "task_history_policy": launch_source_amendment.task_history_policy,
        "completion_policy": launch_source_amendment.completion_policy,
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
    board, config = _load_config(config_path)
    current_head, current_tree = _assert_clean_current_tree(config)
    current_source_forest = _source_forest(config, head=current_head)
    preflight = preflight_configured_board(board)
    if preflight.get("valid") is not True:
        raise OperatorError("configured-board preflight rejected the sealed SPAR board")
    plan = configured_board_launch_plan(
        board,
        implement=implement,
        detach=False,
        duration_seconds=duration_seconds,
    )
    plan["source_forest_root"] = current_source_forest[
        "source_forest_root"
    ]
    program = board.resolved_database_program()
    paths = _runtime_paths(board)
    route_policy = _execution_route_policy(paths)
    launch_source_forest_candidate = _launch_source_forest_receipt(
        source_head=current_head,
        repository_tree=current_tree,
        source_forest=current_source_forest,
    )
    if dry_run:
        launch_source_amendment, idempotent_replay = (
            _prepare_launch_source_amendment(
                board=board,
                config=config,
                paths=paths,
                source_head=current_head,
                repository_tree=current_tree,
                source_forest_receipt=launch_source_forest_candidate,
                execution_route_policy=route_policy,
            )
        )
        listener = _new_bootstrap_listener(lane_count=board.max_lanes)
        try:
            _bind_bootstrap_launch_plan(
                plan,
                listener=listener,
                store_id=program.store_id,
                launch_source_amendment=launch_source_amendment,
            )
            plan["launch_source_amendment"]["idempotent_replay"] = (
                idempotent_replay
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
        finally:
            listener.close()
        return 0
    _harden_runtime_directories(board, paths)
    launch_source_amendment, amendment_admission = (
        _admit_launch_source_amendment(
            board=board,
            config=config,
            paths=paths,
            source_head=current_head,
            repository_tree=current_tree,
            source_forest_receipt=launch_source_forest_candidate,
            execution_route_policy=route_policy,
        )
    )
    launch_source_forest = _record_launch_source_forest(
        paths,
        source_head=current_head,
        repository_tree=current_tree,
        source_forest=current_source_forest,
    )
    recorded_launch_source_amendment = _record_launch_source_amendment(
        paths,
        launch_source_amendment,
        amendment_admission,
    )
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
            mutation_dir=server.mutation_inbox_path(),
            on_failure=broker._terminal_failure,
        )
        monitor.start()
        _bind_bootstrap_launch_plan(
            plan,
            listener=listener,
            store_id=program.store_id,
            launch_source_amendment=launch_source_amendment,
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
                    "launch_source_forest_receipt_id": (
                        launch_source_forest["receipt_id"]
                    ),
                    "launch_source_amendment_id": (
                        launch_source_amendment.amendment_id
                    ),
                    "launch_source_amendment_plan_revision": (
                        amendment_admission["plan_revision"]
                    ),
                    "launch_source_amendment_projection_id": (
                        recorded_launch_source_amendment["projection_id"]
                    ),
                    "source_forest_root": current_source_forest[
                        "source_forest_root"
                    ],
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
