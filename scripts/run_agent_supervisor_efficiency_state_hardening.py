#!/usr/bin/env python3
"""Bootstrap and operate the bounded ASEH board on the existing supervisor.

This is a program adapter, not a supervisor. It materializes sealed Markdown
once into DatabaseTaskSource, starts the existing Quack owner, enables the
PID-bound typed-grant handoff repaired by ASEH-BOOTSTRAP-001, and invokes the
existing configured-board implementation supervisor in the foreground.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG: Final = Path(
    "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
)
PROGRAM: Final = "agent-supervisor-efficiency-and-state-hardening-v1"
OPERATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-program-operator@1"
)
POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-population@1"
)
BOOTSTRAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap@1"
)
DUCKLAKE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-ducklake-projection@1"
)
GOAL_RE: Final = re.compile(r"^## (ASEH-G\d{3}) (.+)$", re.MULTILINE)
META_RE: Final = re.compile(r"^- ([^:\n]+):[ \t]*(.*)$", re.MULTILINE)
READY_STATUSES: Final = frozenset(
    {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
)
ACTIVE_STATUSES: Final = frozenset({"claimed", "in_progress", "running"})
COMPLETED_STATUSES: Final = frozenset({"complete", "completed", "done", "skipped"})
TERMINAL_STATUSES: Final = frozenset(
    {*COMPLETED_STATUSES, "cancelled", "failed", "quarantined", "rejected"}
)
LIVE_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/aseh-live-status@2"
)
OWNER_LOCK_SUFFIX: Final = ".state-owner.lock"
OWNER_MARKER_SUFFIX: Final = ".state-owner.json"
STATUS_SAMPLE_INTERVAL_SECONDS: Final = 0.5
STATUS_RECEIPT_MAX_BYTES: Final = 1_048_576


class OperatorError(RuntimeError):
    """Fail-closed ASEH program error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _run(
    argv: Sequence[str],
    *,
    timeout: float = 600.0,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv), cwd=ROOT, env=None if env is None else dict(env),
        text=True, capture_output=True, check=False, timeout=timeout,
    )


def _git(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=cwd, text=True, capture_output=True,
        check=False, timeout=60,
    )
    if completed.returncode != 0:
        raise OperatorError(
            f"git {' '.join(args)} failed: {completed.stderr[-1000:]}"
        )
    return completed.stdout.strip()


def _safe_path(value: str, *, field: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise OperatorError(f"{field} escapes repository") from exc
    return candidate


def _load(config_path: Path) -> tuple[Any, dict[str, Any]]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
    )

    path = config_path if config_path.is_absolute() else ROOT / config_path
    board = load_configured_board(path, repo_root=ROOT)
    payload = dict(board.payload)
    if payload.get("program_identifier") != PROGRAM:
        raise OperatorError("scheduler is not the ASEH program")
    return board, payload


def _paths(board: Any) -> dict[str, Path]:
    program = board.resolved_database_program()
    runtime = board.path(board.runtime_paths["root"])
    raw = board.payload.get("runtime_paths")
    raw = raw if isinstance(raw, Mapping) else {}
    ducklake = board.payload.get("ducklake_projection_program")
    ducklake = ducklake if isinstance(ducklake, Mapping) else {}
    result = {
        "runtime": runtime,
        "database": _safe_path(program.store_id, field="database_program.store_id"),
        "owner": _safe_path(
            str(raw.get("quack_owner") or f"{board.runtime_paths['root']}/quack-owner"),
            field="runtime_paths.quack_owner",
        ),
        "evidence": _safe_path(
            str(raw.get("evidence") or f"{board.runtime_paths['root']}/evidence"),
            field="runtime_paths.evidence",
        ),
        "ducklake_catalog": _safe_path(
            str(ducklake.get("catalog_path") or f"{board.runtime_paths['root']}/ducklake/catalog.duckdb"),
            field="ducklake_projection_program.catalog_path",
        ),
        "ducklake_data": _safe_path(
            str(ducklake.get("data_path") or f"{board.runtime_paths['root']}/ducklake/data"),
            field="ducklake_projection_program.data_path",
        ),
    }
    registry = _safe_path(
        program.runtime_registry_path,
        field="database_program.runtime_registry_path",
    )
    if registry != result["owner"]:
        raise OperatorError(
            "database runtime registry must equal the canonical Quack owner "
            "directory"
        )
    result["registry"] = registry
    result["bootstrap_receipt"] = (
        result["evidence"] / "bootstrap" / "bootstrap-materialization.json"
    )
    result["ducklake_receipt"] = (
        result["evidence"] / "bootstrap" / "ducklake-history-projection.json"
    )
    result["status_receipt"] = (
        result["evidence"] / "control-plane" / "live-status.json"
    )
    result["inbox_failure_receipt"] = (
        result["evidence"] / "control-plane" / "owner-inbox-failure.json"
    )
    for name, path in result.items():
        if name == "runtime":
            continue
        try:
            path.relative_to(runtime)
        except ValueError as exc:
            raise OperatorError(f"{name} must remain below runtime root") from exc
    return result


def _tracked_bytes(path: Path, *, head: str) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ("git", "show", f"{head}:{relative}"),
        cwd=ROOT, capture_output=True, check=False, timeout=60,
    )
    if completed.returncode != 0:
        raise OperatorError(f"control input is not tracked at HEAD: {relative}")
    observed = path.read_bytes()
    if observed != completed.stdout:
        raise OperatorError(f"control input differs from HEAD: {relative}")
    return observed


def _metadata_value(value: Any) -> str:
    return str(value or "").strip()


def _goal_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    matches = list(GOAL_RE.finditer(text))
    rows: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields = {
            key.lower().replace(" ", "_").replace("-", "_"): value.strip()
            for key, value in META_RE.findall(text[match.end():end])
        }
        rows.append((match.group(1), match.group(2).strip(), fields))
    return rows


def _split(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _assert_clean_tree(board: Any) -> tuple[str, str]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise OperatorError("materialization/launch requires a clean checkout")
    branch = _git("branch", "--show-current")
    if branch != board.merge_target_branch:
        raise OperatorError("current branch differs from sealed merge target")
    return _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")


def _source_forest(
    board: Any,
    *,
    outer_head: str,
    outer_tree: str,
) -> dict[str, Any]:
    """Build the exact clean source identity for every task owner."""

    if _git("rev-parse", "--show-toplevel") != str(ROOT):
        raise OperatorError("accelerator repository root differs")
    entries: list[dict[str, Any]] = [
        {
            "owning_repository": "ipfs_accelerate_py",
            "path": ".",
            "commit": outer_head,
            "tree": outer_tree,
            "gitlink_commit": "",
        }
    ]
    for relative in board.worktree_submodule_paths:
        lexical_path = board.path(relative)
        observed = os.lstat(lexical_path)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise OperatorError(
                f"configured source owner is not a real directory: {relative}"
            )
        path = lexical_path.resolve(strict=True)
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise OperatorError(
                f"configured source owner escapes repository: {relative}"
            ) from exc
        if _git("rev-parse", "--show-toplevel", cwd=path) != str(path):
            raise OperatorError(
                f"configured source owner root differs: {relative}"
            )
        nested_status = _git(
            "status", "--porcelain=v1", "--untracked-files=all", cwd=path
        )
        if nested_status:
            raise OperatorError(
                f"configured source owner is not clean: {relative}"
            )
        nested_head = _git("rev-parse", "HEAD", cwd=path)
        nested_tree = _git("rev-parse", "HEAD^{tree}", cwd=path)
        gitlink_commit = _git("rev-parse", f"{outer_head}:{relative}")
        expected_row = f"160000 commit {nested_head}\t{relative}"
        if (
            gitlink_commit != nested_head
            or _git("ls-tree", outer_head, "--", relative) != expected_row
        ):
            raise OperatorError(
                f"configured source owner differs from gitlink: {relative}"
            )
        entries.append(
            {
                "owning_repository": Path(relative).name,
                "path": relative,
                "commit": nested_head,
                "tree": nested_tree,
                "gitlink_commit": gitlink_commit,
            }
        )
    by_owner = {
        str(entry["owning_repository"]): dict(entry) for entry in entries
    }
    if set(by_owner) != {
        "ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py"
    }:
        raise OperatorError(
            "source forest must contain exactly Accelerate, Datasets, and Kit"
        )
    body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-source-forest@1",
        "entries": [
            {**entry, "source_identity": _identity(entry)}
            for entry in entries
        ],
    }
    return {
        **body,
        "forest_cid": _identity(body),
        "by_owner": {
            owner: {**entry, "source_identity": _identity(entry)}
            for owner, entry in sorted(by_owner.items())
        },
    }


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

    head, tree = _assert_clean_tree(board)
    source_forest = _source_forest(
        board, outer_head=head, outer_tree=tree,
    )
    source_paths = {
        "config": board.config_path,
        "taskboard": board.path(board.taskboard_path),
        "objectives": board.path(board.objectives_path),
        "plan": board.path(board.plan_path),
        "program_registry": ROOT / "docs/architecture/agent_supervisor/PROGRAMS.md",
        "validator": board.path(board.validator_path),
        "operator": Path(__file__).resolve(),
        "baseline": ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/bootstrap_baseline.json",
        "bootstrap_test": ROOT / "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        "requirements": board.path(str(config.get("requirements_path") or "")),
    }
    sources = {
        name: _tracked_bytes(path, head=head)
        for name, path in source_paths.items()
    }
    if _identity(sources["requirements"]) != config.get("requirements_digest"):
        raise OperatorError("protected requirements digest differs from config")
    plan_root = content_identity(
        {
            "schema": "aseh-plan-root@1",
            "source_head": head,
            "repository_tree_id": tree,
            "source_forest_cid": source_forest["forest_cid"],
            "sources": {
                name: _identity(payload) for name, payload in sorted(sources.items())
            },
        }
    )

    goal_rows = _goal_blocks(sources["objectives"].decode("utf-8"))
    if [item[0] for item in goal_rows] != [
        "ASEH-G000", "ASEH-G010", "ASEH-G020", "ASEH-G030", "ASEH-G040",
        "ASEH-G050", "ASEH-G060", "ASEH-G070", "ASEH-G080",
    ]:
        raise OperatorError("goal heap differs from sealed ASEH IDs/order")
    goal_cids = {
        goal_id: content_identity(
            {
                "goal_id": goal_id, "title": title, "metadata": fields,
                "plan_root_cid": plan_root,
            }
        )
        for goal_id, title, fields in goal_rows
    }
    goals: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    observed_goals: set[str] = set()
    for ordinal, (goal_id, title, fields) in enumerate(goal_rows, start=1):
        parent = fields.get("parent", "")
        if parent and parent not in observed_goals:
            raise OperatorError(f"{goal_id} parent must precede it")
        goal = {
            "goal_cid": goal_cids[goal_id],
            "goal_id": goal_id,
            "goal_alias": goal_id,
            "title": title,
            "ordinal": ordinal,
            "status": fields.get("status", "active").lower(),
            "objective_id": "objective:aseh-root" if goal_id == "ASEH-G000" else "",
            "objective_alias": "ASEH-G000",
            "priority": fields.get("priority", "P0"),
            "body": dict(fields),
        }
        if parent:
            goal["parent_goal_cid"] = goal_cids[parent]
            edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in _split(fields.get("depends_on")):
            if dependency not in goal_cids:
                raise OperatorError(f"{goal_id} has an unknown dependency")
            edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
        goals.append(goal)
        observed_goals.add(goal_id)

    parsed = parse_todo_blocks(
        sources["taskboard"].decode("utf-8"),
        task_header_prefix="## ASEH-",
    )
    expected_ids = [
        "ASEH-000", "ASEH-001",
        *[f"ASEH-{group}{item}" for group, count in (
            ("01", 6), ("02", 5), ("03", 6), ("04", 6), ("05", 6),
            ("06", 3), ("07", 6),
        ) for item in range(count)],
    ]
    # The compact expression above would produce 010..015, 020..024, etc.
    task_ids = [item[0] for item in parsed]
    if task_ids != expected_ids:
        raise OperatorError("task board differs from the sealed 40-task ID/order")
    normalized = [
        (
            task_id, title, source_line,
            {key: _metadata_value(value) for key, value in fields.items()},
        )
        for task_id, title, source_line, fields in parsed
    ]
    task_cids = {
        task_id: content_identity(
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "metadata": fields,
                "plan_root_cid": plan_root,
                "source_forest_cid": source_forest["forest_cid"],
                "owner_source": source_forest["by_owner"].get(
                    fields.get("owning_repository", "")
                ),
            }
        )
        for task_id, title, source_line, fields in normalized
    }
    tasks: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    for ordinal, (task_id, title, source_line, fields) in enumerate(normalized, start=1):
        dependencies = _split(fields.get("depends_on"))
        future = [item for item in dependencies if item not in observed_tasks]
        if future:
            raise OperatorError(f"{task_id} dependencies must precede it: {future}")
        goal_id = fields.get("subgoal_id") or fields.get("goal_id") or ""
        if goal_id not in goal_cids:
            raise OperatorError(f"{task_id} refers to an unknown goal")
        owner = fields.get("owning_repository", "")
        owner_source = source_forest["by_owner"].get(owner)
        if not isinstance(owner_source, Mapping):
            raise OperatorError(f"{task_id} has an unknown source owner: {owner}")
        outputs = _split(
            fields.get("exact_declared_outputs") or fields.get("outputs")
        )
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
                "objective_id": "objective:aseh-root",
                "ordinal": ordinal,
                "status": "todo",
                "priority": fields.get("priority", "P1"),
                "dependencies": [task_cids[item] for item in dependencies],
                "depends_on": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"task_cid": task_cids[task_id], "path": path}
                        ),
                    }
                    for path in outputs
                ],
                "acceptance": [fields.get("acceptance_conditions", "")],
                "validations": list(
                    split_validation_commands(fields.get("validation", ""))
                ),
                "accepted_plan_root_cid": plan_root,
                "source_forest_cid": source_forest["forest_cid"],
                "base_revision": owner_source["commit"],
                "base_repository_tree_id": owner_source["tree"],
                "owner_source_identity": owner_source["source_identity"],
                "owning_repository": owner,
            }
        )
        tasks.append(task)
        observed_tasks.add(task_id)

    projection = config.get("initial_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    dependency_count = sum(
        len(_split(item[3].get("depends_on"))) for item in normalized
    )
    if (
        int(projection.get("task_count", -1)) != len(tasks)
        or int(projection.get("goal_count", -1)) != len(goals)
        or int(projection.get("task_dependency_count", -1)) != dependency_count
    ):
        raise OperatorError("materialized population differs from initial projection")
    return {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "source_forest": source_forest,
        "source_identities": {
            name: _identity(payload) for name, payload in sorted(sources.items())
        },
        "objectives": goals,
        "goal_edges": edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "ASEH-PLAN-R1",
                "goal_cid": goal_cids["ASEH-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _validate_bootstrap(board: Any) -> dict[str, Any]:
    commands = (
        (sys.executable, str(board.path(board.validator_path)), "--check-all", "--json"),
        (
            sys.executable, "-m", "pytest", "-q",
            "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
        ),
    )
    results: list[dict[str, Any]] = []
    for command in commands:
        completed = _run(command, timeout=900)
        observation = {
            "argv": list(command),
            "returncode": completed.returncode,
            "stdout_digest": _identity(completed.stdout.encode()),
            "stderr_digest": _identity(completed.stderr.encode()),
        }
        results.append(observation)
        if completed.returncode != 0:
            raise OperatorError(
                f"bootstrap validation failed: {' '.join(command)}"
            )
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-bootstrap-validation@1",
        "hermetic": True,
        "live": False,
        "commands": results,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _ducklake_projection(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "schema": DUCKLAKE_SCHEMA,
        "authoritative": False,
        "scheduler_gate": False,
        "acceptance_gate": False,
        "completion_gate": False,
        "status": "unavailable",
        "reason_code": "ducklake_projection_unavailable",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
    }
    try:
        import duckdb

        catalog = paths["ducklake_catalog"]
        data_path = paths["ducklake_data"]
        catalog.parent.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data_path).replace("'", "''")
            connection.execute(
                f"ATTACH 'ducklake:{catalog_sql}' AS aseh_history "
                f"(DATA_PATH '{data_sql}')"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS aseh_history.bootstrap_history (
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
            if int(connection.execute(
                "SELECT COUNT(*) FROM aseh_history.bootstrap_history WHERE event_id = ?",
                [event_id],
            ).fetchone()[0]) == 0:
                connection.execute(
                    "INSERT INTO aseh_history.bootstrap_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        event_id, time.time(), population["source_head"],
                        population["repository_tree_id"], population["plan_root_cid"],
                        str(control_receipt.get("projection_cid") or ""),
                        int(control_receipt.get("task_count") or 0),
                        int(control_receipt.get("goal_count") or 0),
                        json.dumps(
                            {
                                "control_authority": "DuckDB/DatabaseTaskSource@1",
                                "transport": "QuackStateServer@1",
                                "projection": "DuckLake/non-authoritative",
                            },
                            sort_keys=True,
                        ),
                    ],
                )
            count = int(connection.execute(
                "SELECT COUNT(*) FROM aseh_history.bootstrap_history"
            ).fetchone()[0])
            connection.execute("DETACH aseh_history")
        finally:
            connection.close()
        projection.update(
            {
                "status": "available", "reason_code": "", "event_id": event_id,
                "row_count": count,
                "catalog_path": str(catalog.relative_to(ROOT)),
                "data_path": str(data_path.relative_to(ROOT)),
            }
        )
    except Exception as exc:
        projection["error_class"] = type(exc).__name__
    projection["projection_receipt_id"] = _identity(projection)
    _atomic_json(paths["ducklake_receipt"], projection)
    return projection


@contextmanager
def _offline_database_guard(paths: Mapping[str, Path]) -> Any:
    """Hold the Quack owner's exact lock for every direct-file operation."""

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        owner_liveness,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OwnerMarker,
    )

    database = paths["database"]
    lock_path = database.with_name(f".{database.name}{OWNER_LOCK_SUFFIX}")
    marker_path = database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OperatorError(
                "offline database access refused while the Quack owner is live"
            ) from exc
        if marker_path.exists():
            try:
                marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
                marker = OwnerMarker.from_dict(marker_payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OperatorError(
                    "offline database access refused: owner marker is invalid"
                ) from exc
            liveness = owner_liveness(marker.process_birth)
            if liveness is not OwnerLiveness.DEAD:
                raise OperatorError(
                    "offline database access refused: owner liveness is not dead"
                )
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _verify_materialized_source(
    source: Any,
    *,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Prove the bounded projection and its admitted-event replay agree."""

    expected_tasks = list(population["tasks"])
    expected_aliases = [str(item["task_alias"]) for item in expected_tasks]
    expected_cids = {
        str(item["task_alias"]): str(item["task_cid"])
        for item in expected_tasks
    }
    if source.projection_matches_events() is not True:
        raise OperatorError("materialized projection differs from admitted events")
    snapshot = source.snapshot().to_dict()
    page = source.list_tasks(limit=100)
    if page.next_cursor:
        raise OperatorError("materialized task population exceeds its sealed bound")
    observed_aliases = [item.task_alias for item in page.tasks]
    if observed_aliases != expected_aliases:
        raise OperatorError("materialized task order/aliases differ from the board")
    for item in page.tasks:
        if item.task_cid != expected_cids.get(item.task_alias):
            raise OperatorError(
                f"materialized task identity differs: {item.task_alias}"
            )
        expected = expected_tasks[expected_aliases.index(item.task_alias)]
        for key in (
            "owning_repository",
            "base_revision",
            "base_repository_tree_id",
            "source_forest_cid",
            "owner_source_identity",
        ):
            if item.body.get(key) != expected.get(key):
                raise OperatorError(
                    f"materialized owner binding differs: {item.task_alias}:{key}"
                )
    ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    expected_ready = list(config["initial_projection"]["ready_task_ids"])
    if ready != expected_ready:
        raise OperatorError(
            f"initial ready frontier differs: expected {expected_ready}, "
            f"observed {ready}"
        )
    projection = config["initial_projection"]
    expected_counts = {
        "task_count": int(projection["task_count"]),
        "goal_count": int(projection["goal_count"]),
        "dependency_count": int(projection["task_dependency_count"]),
    }
    for key, expected in expected_counts.items():
        if int(snapshot.get(key, -1)) != expected:
            raise OperatorError(f"materialized {key} differs from the board")
    integrity = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-integrity@1",
        "projection_matches_events": True,
        "projection_cid": snapshot["projection_cid"],
        "event_cursor": snapshot["event_cursor"],
        **expected_counts,
    }
    integrity["integrity_receipt_id"] = _identity(integrity)
    return snapshot, ready, integrity


def _recovered_control_receipt(
    population: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-task-source@1",
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "projection_cid": snapshot["projection_cid"],
        "task_count": len(population["tasks"]),
        "goal_count": len(population["objectives"]),
        "goal_edge_count": len(population["goal_edges"]),
        "plan_count": len(population["plans"]),
        "event_watermark": snapshot["event_cursor"],
        "task_cids": [str(item["task_cid"]) for item in population["tasks"]],
        "recovered_from_exact_projection": True,
    }


def _write_bootstrap_receipt(
    *,
    paths: Mapping[str, Path],
    population: Mapping[str, Any],
    validation: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    integrity: Mapping[str, Any],
    ready: Sequence[str],
    recovered: bool,
) -> dict[str, Any]:
    ducklake = _ducklake_projection(
        paths=paths, population=population, control_receipt=control_receipt,
    )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "source_forest": population["source_forest"],
        "source_identities": population["source_identities"],
        "database_task_source_receipt": dict(control_receipt),
        "snapshot": dict(snapshot),
        "integrity": dict(integrity),
        "initial_ready_task_ids": list(ready),
        "bootstrap_validation": dict(validation),
        "recovered_after_interrupted_materialization": bool(recovered),
        "authority": {
            "operational_state": "DuckDB/DatabaseTaskSource@1",
            "live_transport": "QuackStateServer@1/TypedStateOwnerCommandGateway@1",
            "ducklake": "non_authoritative_history_projection",
        },
        "ducklake_projection": ducklake,
    }
    receipt["bootstrap_receipt_id"] = _identity(receipt)
    _atomic_json(paths["bootstrap_receipt"], receipt)
    return receipt


def materialize(config_path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    board, config = _load(config_path)
    paths = _paths(board)
    population = _population(board, config)
    validation = _validate_bootstrap(board)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    database = paths["database"]
    bootstrap = paths["bootstrap_receipt"]
    stage = database.with_name(f".{database.name}.bootstrap-stage")
    with _offline_database_guard(paths):
        if bootstrap.exists() and not database.is_file():
            raise OperatorError("bootstrap receipt exists without its database")
        if database.exists() and not database.is_file():
            raise OperatorError("database authority is not a regular file")
        if database.is_file():
            with DatabaseTaskSource(
                database, owner_id="aseh-bootstrap:verify",
                install_schema=False,
                repository_tree_id=population["repository_tree_id"],
                plan_root_cid=population["plan_root_cid"],
            ) as source:
                snapshot, ready, integrity = _verify_materialized_source(
                    source, population=population, config=config,
                )
            if bootstrap.is_file():
                prior = json.loads(bootstrap.read_text(encoding="utf-8"))
                unsigned = dict(prior)
                prior_id = unsigned.pop("bootstrap_receipt_id", "")
                if prior_id != _identity(unsigned):
                    raise OperatorError("existing bootstrap receipt CID is invalid")
                if any(
                    prior.get(key) != population.get(key)
                    for key in (
                        "source_head", "repository_tree_id", "plan_root_cid",
                    )
                ) or prior.get("source_forest") != population["source_forest"]:
                    raise OperatorError(
                        "existing authority differs from the sealed source forest"
                    )
                return {
                    "schema": OPERATOR_SCHEMA, "command": "materialize",
                    "ok": True, "idempotent_replay": True,
                    "bootstrap_receipt": prior, "snapshot": snapshot,
                    "ready_task_ids": ready,
                }
            control_receipt = _recovered_control_receipt(population, snapshot)
            receipt = _write_bootstrap_receipt(
                paths=paths, population=population, validation=validation,
                control_receipt=control_receipt, snapshot=snapshot,
                integrity=integrity, ready=ready, recovered=True,
            )
            return {
                "schema": OPERATOR_SCHEMA, "command": "materialize",
                "ok": True, "idempotent_replay": False,
                "recovered": True, "bootstrap_receipt": receipt,
                "snapshot": snapshot,
            }

        for candidate in (stage, Path(f"{stage}.wal")):
            candidate.unlink(missing_ok=True)
        try:
            with DatabaseTaskSource(
                stage, owner_id="aseh-bootstrap:single-writer",
                repository_tree_id=population["repository_tree_id"],
                plan_root_cid=population["plan_root_cid"],
            ) as source:
                control_receipt = dict(source.materialize(population))
                snapshot, ready, integrity = _verify_materialized_source(
                    source, population=population, config=config,
                )
            with stage.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(stage, database)
            directory = os.open(
                database.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            stage.unlink(missing_ok=True)
            Path(f"{stage}.wal").unlink(missing_ok=True)
            raise
        receipt = _write_bootstrap_receipt(
            paths=paths, population=population, validation=validation,
            control_receipt=control_receipt, snapshot=snapshot,
            integrity=integrity, ready=ready, recovered=False,
        )
    return {
        "schema": OPERATOR_SCHEMA, "command": "materialize",
        "ok": True, "idempotent_replay": False,
        "bootstrap_receipt": receipt, "snapshot": snapshot,
    }


def _build_server(board: Any, paths: Mapping[str, Path]) -> Any:
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
        TYPED_STATE_OWNER_SOCKET_FILENAME,
    )

    program = board.resolved_database_program()
    endpoint = str(program.quack_endpoint)
    port = int(endpoint.rsplit(":", 1)[1])
    if Path.cwd().resolve() != ROOT:
        raise OperatorError(
            "the configured owner must start from the sealed repository root"
        )
    owner_relative = paths["owner"].relative_to(ROOT)
    socket_parent = Path("/proc/self/cwd") / owner_relative
    typed_socket = socket_parent / TYPED_STATE_OWNER_SOCKET_FILENAME
    broker_socket = socket_parent / TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME
    if any(
        len(os.fsencode(str(path))) >= 108
        for path in (typed_socket, broker_socket)
    ):
        raise OperatorError("configured owner socket aliases exceed AF_UNIX bounds")
    return build_server(
        database_path=paths["database"],
        state_dir=paths["owner"],
        repository_root=ROOT,
        host="127.0.0.1",
        port=port,
        repository_id=PROGRAM,
        store_id=program.store_id,
        secret_handle=program.endpoint_secret_handle,
        typed_command_socket_path=typed_socket,
    )


def _record_control_failure(
    paths: Mapping[str, Path],
    failure: dict[str, Any],
    failure_event: threading.Event,
    *,
    reason_code: str,
    error_type: str,
) -> None:
    if failure_event.is_set():
        return
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-control-failure@1",
        "reason_code": reason_code,
        "error_type": error_type,
        "observed_at": time.time(),
    }
    payload["receipt_cid"] = _identity(payload)
    failure.update(payload)
    _atomic_json(paths["inbox_failure_receipt"], payload)
    failure_event.set()


def _owner_inbox_loop(
    server: Any,
    paths: Mapping[str, Path],
    stop: threading.Event,
    failure: dict[str, Any],
    failure_event: threading.Event,
    expected_store_generation: str,
) -> None:
    while not stop.wait(0.1):
        if server.lifecycle.value != "ready":
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="owner_left_ready_state",
                error_type="QuackStateServerNotReady",
            )
            return
        try:
            server.service_database_task_command_inbox(
                expected_store_generation=expected_store_generation,
                max_requests=32,
            )
            # The content-CID bundle protocol has a disjoint request identity.
            # Keep that compatibility service without invoking the legacy
            # generic SQL-envelope scanner.
            server.service_mutation_inbox(max_requests=32)
        except Exception as exc:
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="owner_mutation_inbox_failed",
                error_type=type(exc).__name__,
            )
            return


def _terminate_scheduler(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10.0)


def run_supervisor(config_path: Path, *, implement: bool, duration: float) -> int:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        preflight_configured_board,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
        state_authority_pass_fds,
    )

    board, _config = _load(config_path)
    paths = _paths(board)
    if not paths["bootstrap_receipt"].is_file() or not paths["database"].is_file():
        raise OperatorError("materialize the sealed board before starting the owner")
    preflight = preflight_configured_board(board)
    if preflight.get("valid") is not True:
        raise OperatorError(
            "configured-board preflight failed: "
            + json.dumps(preflight.get("errors") or [])
        )
    server = _build_server(board, paths)
    stop = threading.Event()
    failure_event = threading.Event()
    failure: dict[str, Any] = {}
    inbox_thread: threading.Thread | None = None
    monitor_thread: threading.Thread | None = None
    scheduler: subprocess.Popen[Any] | None = None
    prior_environment: dict[str, str | None] = {}
    try:
        identity = server.start()
        launched_at = time.time()
        program_environment = dict(
            board.resolved_database_program().environment(repository_root=ROOT)
        )
        expected_mutations = str((paths["owner"] / "mutations").resolve())
        if program_environment.get(
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"
        ) != expected_mutations:
            raise OperatorError("scheduler and owner mutation inboxes differ")
        program_environment["IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET"] = str(
            server.typed_command_socket_path()
        )
        broker_environment = dict(server.start_supervisor_grant_broker())
        launch_environment = {**program_environment, **broker_environment}
        raw_token_name = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
        prior_environment[raw_token_name] = os.environ.get(raw_token_name)
        os.environ.pop(raw_token_name, None)
        for name, value in launch_environment.items():
            prior_environment[name] = os.environ.get(name)
            os.environ[name] = value
        harden_state_authority_process()
        inbox_thread = threading.Thread(
            target=_owner_inbox_loop,
            args=(
                server,
                paths,
                stop,
                failure,
                failure_event,
                str(identity.generation),
            ),
            name="aseh-owner-mutation-inbox",
            daemon=True,
        )
        inbox_thread.start()
        argv = [
            sys.executable,
            str(ROOT / "scripts/ops/agent_supervisor/configured_board_scheduler.py"),
            "--repo-root", str(ROOT), "--config", str(board.config_path),
            "launch", "--foreground", "--duration-seconds", str(duration),
        ]
        if implement:
            argv.append("--implement")
        scheduler = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=dict(os.environ),
            start_new_session=True,
            pass_fds=state_authority_pass_fds(os.environ),
        )
        initial_health, last_progress_at = _await_initial_health(
            board, paths, server, scheduler, launched_at=launched_at,
            failure=failure, failure_event=failure_event,
        )
        monitor_thread = threading.Thread(
            target=_status_monitor_loop,
            kwargs={
                "board": board,
                "paths": paths,
                "server": server,
                "scheduler": scheduler,
                "launched_at": launched_at,
                "previous": initial_health["samples"][-1],
                "last_progress_at": last_progress_at,
                "stop": stop,
                "failure": failure,
                "failure_event": failure_event,
            },
            name="aseh-live-health-monitor",
            daemon=True,
        )
        monitor_thread.start()
        launch_record = {
            "schema": "ipfs_accelerate_py/agent-supervisor/aseh-owner-launch@1",
            "identity": identity.to_dict(),
            "typed_grant_broker": {
                "available": True,
                "socket_path": broker_environment[
                    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET"
                ],
                "secret_published": False,
            },
            "initial_health_receipt_cid": initial_health["receipt_cid"],
            "implement": implement,
        }
        launch_record["receipt_cid"] = _identity(launch_record)
        _atomic_json(
            paths["evidence"] / "control-plane" / "owner-launch.json",
            launch_record,
        )
        while scheduler.poll() is None:
            if failure_event.wait(0.25):
                _terminate_scheduler(scheduler)
                raise OperatorError(
                    f"foreground control plane failed: {failure.get('reason_code')}"
                )
        return int(scheduler.returncode or 0)
    finally:
        stop.set()
        if scheduler is not None:
            _terminate_scheduler(scheduler)
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        if inbox_thread is not None:
            inbox_thread.join(timeout=2.0)
        try:
            try:
                from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                    reset_quack_transport_cache,
                )

                reset_quack_transport_cache()
            finally:
                server.stop()
        finally:
            for name, prior in prior_environment.items():
                if prior is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = prior


def _broker_status_query(board: Any, paths: Mapping[str, Path]) -> dict[str, Any]:
    """Read the live portfolio through the sealed-broker Quack path."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    program = board.resolved_database_program()
    broker_socket = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET", "")
        or ""
    )
    broker_fd = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD", "")
        or ""
    )
    if (
        Path(broker_socket).resolve(strict=False)
        != (paths["owner"] / "typed-state-owner-grants.sock").resolve(strict=False)
        or not broker_fd.isdecimal()
        or str(os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or "")
    ):
        raise OperatorError("live status lacks an exclusive sealed-broker binding")
    os.fstat(int(broker_fd))
    bootstrap = json.loads(paths["bootstrap_receipt"].read_text(encoding="utf-8"))
    with DatabaseTaskSource(
        program.quack_endpoint,
        owner_id=f"aseh-status:{os.getpid()}:{time.time_ns()}",
        install_schema=False,
        repository_tree_id=str(bootstrap.get("repository_tree_id") or ""),
        plan_root_cid=str(bootstrap.get("plan_root_cid") or ""),
    ) as source:
        if source.intent.uses_quack_transport is not True:
            raise OperatorError("live status did not use the Quack transport")
        snapshot = source.snapshot().to_dict()
        page = source.list_tasks(limit=100)
        if page.next_cursor:
            raise OperatorError("live task portfolio exceeds the sealed bound")
        ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    statuses: Counter[str] = Counter()
    aliases: dict[str, str] = {}
    revisions: dict[str, int] = {}
    for task in page.tasks:
        status_name = str(task.status or "").lower()
        aliases[task.task_alias] = status_name
        revisions[task.task_alias] = int(task.revision)
        statuses[status_name] += 1
    return {
        "available": True,
        "transport": "quack",
        "credential_path": "sealed_memfd_broker",
        "snapshot": snapshot,
        "task_statuses": dict(sorted(aliases.items())),
        "task_revisions": dict(sorted(revisions.items())),
        "status_counts": dict(sorted(statuses.items())),
        "ready_task_ids": ready,
        "ready_count": len(ready),
        "active_count": sum(statuses.get(item, 0) for item in ACTIVE_STATUSES),
        "blocked_count": int(statuses.get("blocked", 0)),
        "completed_count": sum(statuses.get(item, 0) for item in COMPLETED_STATUSES),
        "terminal_count": sum(statuses.get(item, 0) for item in TERMINAL_STATUSES),
        "event_cursor": int(snapshot.get("event_cursor") or 0),
    }


def _lane_status_observations(board: Any, *, now: float) -> list[dict[str, Any]]:
    state_root = board.path(board.runtime_paths["state"])
    prefix = re.sub(
        r"[^a-z0-9._-]+", "-", board.task_prefix.strip().lower()
    ).strip("-") or "configured-board"
    rows: list[dict[str, Any]] = []
    for index in range(board.max_lanes):
        path = (
            state_root / f"lane-{index}"
            / f"{prefix}_lane_{index}_supervisor_status.json"
        )
        row: dict[str, Any] = {
            "lane": index,
            "path": str(path.relative_to(ROOT)),
            "present": False,
            "fresh": False,
        }
        try:
            observed = path.stat()
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            rows.append(row)
            continue
        age = max(0.0, now - observed.st_mtime)
        row.update(
            {
                "present": True,
                "mtime_ns": observed.st_mtime_ns,
                "age_seconds": age,
                "fresh": age <= 60.0,
                "phase": payload.get("phase") or payload.get("status") or "",
                "admissible": bool(
                    payload.get("schema")
                    == (
                        "ipfs_accelerate_py.agent_supervisor."
                        "todo_implementation_supervisor.supervisor"
                    )
                    and payload.get("repo_root") == str(board.repo_root)
                    and payload.get("task_prefix") == board.task_header_prefix
                    and payload.get("state_prefix")
                    == f"{prefix}_lane_{index}"
                    and str(payload.get("status") or "")
                    in {
                        "starting",
                        "running",
                        "restarting",
                        "agentic_maintenance_started",
                    }
                ),
                "receipt_cid": (
                    payload.get("receipt_cid") or payload.get("status_cid") or ""
                ),
            }
        )
        rows.append(row)
    return rows


def _status_sample(
    board: Any,
    paths: Mapping[str, Path],
    server: Any,
    scheduler: subprocess.Popen[Any],
) -> dict[str, Any]:
    observed_at = time.time()
    try:
        authority = _broker_status_query(board, paths)
    except Exception as exc:
        authority = {
            "available": False,
            "error_type": type(exc).__name__,
            "ready_count": 0,
            "active_count": 0,
            "blocked_count": 0,
            "terminal_count": 0,
            "event_cursor": 0,
            "task_statuses": {},
            "task_revisions": {},
        }
    scheduler_returncode = scheduler.poll()
    try:
        scheduler_process_group = os.getpgid(scheduler.pid)
    except ProcessLookupError:
        scheduler_process_group = -1
    sample = {
        "observed_at": observed_at,
        "monotonic_ns": time.monotonic_ns(),
        "owner_status": server.status(),
        "scheduler": {
            "pid": scheduler.pid,
            "process_group": scheduler_process_group,
            "alive": scheduler_returncode is None,
            "returncode": scheduler_returncode,
        },
        "authority": authority,
        "lanes": _lane_status_observations(board, now=observed_at),
    }
    sample["sample_cid"] = _identity(sample)
    return sample


def _progress_between(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    evidence: list[str] = []
    prior_authority = before.get("authority")
    current_authority = after.get("authority")
    prior_authority = prior_authority if isinstance(prior_authority, Mapping) else {}
    current_authority = (
        current_authority if isinstance(current_authority, Mapping) else {}
    )
    if int(current_authority.get("event_cursor") or 0) > int(
        prior_authority.get("event_cursor") or 0
    ):
        evidence.append("authoritative_event_advanced")
    if current_authority.get("task_statuses") != prior_authority.get("task_statuses"):
        evidence.append("task_status_changed")
    if current_authority.get("task_revisions") != prior_authority.get("task_revisions"):
        evidence.append("task_revision_changed")
    prior_lanes = {
        int(item.get("lane", -1)): int(item.get("mtime_ns") or 0)
        for item in before.get("lanes", [])
        if isinstance(item, Mapping)
    }
    if any(
        int(item.get("mtime_ns") or 0)
        > prior_lanes.get(int(item.get("lane", -1)), 0)
        for item in after.get("lanes", [])
        if isinstance(item, Mapping)
    ):
        evidence.append("lane_heartbeat_advanced")
    return evidence


def _health_receipt(
    board: Any,
    paths: Mapping[str, Path],
    *,
    samples: Sequence[Mapping[str, Any]],
    launched_at: float,
    last_progress_at: float,
    failure: dict[str, Any],
) -> dict[str, Any]:
    if len(samples) != 2:
        raise OperatorError("health receipt requires exactly two samples")
    before, current = samples
    scheduler_samples = [
        item.get("scheduler")
        if isinstance(item.get("scheduler"), Mapping)
        else {}
        for item in (before, current)
    ]
    scheduler_alive = bool(
        all(item.get("alive") is True for item in scheduler_samples)
        and len({int(item.get("pid") or 0) for item in scheduler_samples}) == 1
        and all(
            int(item.get("process_group") or -1) == int(item.get("pid") or 0)
            for item in scheduler_samples
        )
    )
    prior_authority = before.get("authority")
    prior_authority = (
        prior_authority if isinstance(prior_authority, Mapping) else {}
    )
    authority = current.get("authority")
    authority = authority if isinstance(authority, Mapping) else {}
    owner_status = current.get("owner_status")
    owner_status = owner_status if isinstance(owner_status, Mapping) else {}
    progress = _progress_between(before, current)
    now = float(current["observed_at"])
    stale_seconds = float(board.payload.get("stale_seconds") or 1800.0)
    startup_grace = float(
        board.payload.get("watchdog_startup_grace_seconds") or 300.0
    )
    lanes = [item for item in current.get("lanes", []) if isinstance(item, Mapping)]
    lane_fresh = bool(
        len(lanes) == board.max_lanes
        and all(
            item.get("fresh") is True
            and item.get("admissible") is True
            and int(item.get("mtime_ns") or 0) >= int(launched_at * 1_000_000_000)
            for item in lanes
        )
    )
    owner_ready = owner_status.get("lifecycle") == "ready"
    broker = owner_status.get("configured_supervisor_credential_broker")
    broker = broker if isinstance(broker, Mapping) else {}
    broker_ready = bool(
        broker.get("available") is True
        and not str(broker.get("last_error_type") or "")
    )
    broker_samples_authenticated = all(
        sample.get("available") is True
        and sample.get("transport") == "quack"
        and sample.get("credential_path") == "sealed_memfd_broker"
        for sample in (prior_authority, authority)
    )
    expected_tasks = int(board.payload["initial_projection"]["task_count"])
    snapshot = authority.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    task_count = int(snapshot.get("task_count", -1))
    blocked_count = int(authority.get("blocked_count") or 0)
    terminal_count = int(authority.get("terminal_count") or 0)
    ready_count = int(authority.get("ready_count") or 0)
    active_count = int(authority.get("active_count") or 0)
    terminal = terminal_count == expected_tasks
    startup_active = now - launched_at <= startup_grace
    evidence_fresh = bool(progress or lane_fresh or startup_active or terminal)
    stuck = bool(
        owner_ready
        and authority.get("available") is True
        and not terminal
        and (ready_count or active_count)
        and not evidence_fresh
        and now - last_progress_at >= stale_seconds
    )
    blocked = bool(blocked_count or failure)
    healthy = bool(
        owner_ready
        and scheduler_alive
        and broker_ready
        and broker_samples_authenticated
        and task_count == expected_tasks
        and lane_fresh
        and not blocked
        and not stuck
        and (ready_count or active_count or terminal)
    )
    bootstrap = json.loads(paths["bootstrap_receipt"].read_text(encoding="utf-8"))
    receipt = {
        "schema": LIVE_STATUS_SCHEMA,
        "program_id": PROGRAM,
        "source_head": bootstrap.get("source_head"),
        "repository_tree_id": bootstrap.get("repository_tree_id"),
        "plan_root_cid": bootstrap.get("plan_root_cid"),
        "bootstrap_receipt_id": bootstrap.get("bootstrap_receipt_id"),
        "broker_authenticated": broker_samples_authenticated,
        "samples": [dict(before), dict(current)],
        "progress_evidence": progress,
        "last_progress_at": last_progress_at,
        "startup_grace_active": startup_active,
        "lane_heartbeat_fresh": lane_fresh,
        "owner_ready": owner_ready,
        "scheduler_alive": scheduler_alive,
        "healthy": healthy,
        "blocked": blocked,
        "stuck": stuck,
        "terminal": terminal,
        "failure": dict(failure),
        "observed_at": now,
    }
    receipt["receipt_cid"] = _identity(receipt)
    return receipt


def _await_initial_health(
    board: Any,
    paths: Mapping[str, Path],
    server: Any,
    scheduler: subprocess.Popen[Any],
    *,
    launched_at: float,
    failure: Mapping[str, Any],
    failure_event: threading.Event,
) -> tuple[dict[str, Any], float]:
    first = _status_sample(board, paths, server, scheduler)
    last_progress_at = launched_at
    timeout = min(
        600.0,
        max(
            5.0,
            float(board.payload.get("watchdog_startup_grace_seconds") or 300.0),
        ),
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if scheduler.poll() is not None:
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="scheduler_exited_before_health_admission",
                error_type="ASEHForegroundSchedulerExit",
            )
            raise OperatorError("scheduler exited before health admission")
        if failure_event.wait(STATUS_SAMPLE_INTERVAL_SECONDS):
            raise OperatorError("control failure occurred before health admission")
        second = _status_sample(board, paths, server, scheduler)
        if _progress_between(first, second):
            last_progress_at = float(second["observed_at"])
        receipt = _health_receipt(
            board, paths, samples=(first, second), launched_at=launched_at,
            last_progress_at=last_progress_at, failure=failure,
        )
        _atomic_json(paths["status_receipt"], receipt)
        if receipt.get("blocked") is True or receipt.get("stuck") is True:
            _record_control_failure(
                paths, failure, failure_event,
                reason_code=(
                    "authoritative_board_blocked"
                    if receipt.get("blocked") is True
                    else "authoritative_board_stuck"
                ),
                error_type="ASEHHealthGateFailure",
            )
            raise OperatorError("foreground health admission failed closed")
        prior_authority = first.get("authority")
        current_authority = second.get("authority")
        if not (
            isinstance(prior_authority, Mapping)
            and prior_authority.get("available") is True
        ) and not (
            isinstance(current_authority, Mapping)
            and current_authority.get("available") is True
        ):
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="broker_status_unavailable_two_samples",
                error_type="ASEHHealthQueryFailure",
            )
            raise OperatorError("broker health unavailable for two samples")
        if receipt.get("healthy") is True:
            return receipt, last_progress_at
        first = second
    _record_control_failure(
        paths, failure, failure_event,
        reason_code="foreground_health_admission_timeout",
        error_type="ASEHHealthAdmissionTimeout",
    )
    raise OperatorError("two-sample foreground health admission timed out")


def _status_monitor_loop(
    board: Any,
    paths: Mapping[str, Path],
    server: Any,
    scheduler: subprocess.Popen[Any],
    *,
    launched_at: float,
    previous: Mapping[str, Any],
    last_progress_at: float,
    stop: threading.Event,
    failure: dict[str, Any],
    failure_event: threading.Event,
) -> None:
    interval = min(
        30.0,
        max(1.0, float(board.payload.get("check_interval_seconds") or 10.0)),
    )
    prior = dict(previous)
    while not stop.wait(interval):
        try:
            current = _status_sample(board, paths, server, scheduler)
            if _progress_between(prior, current):
                last_progress_at = float(current["observed_at"])
            receipt = _health_receipt(
                board, paths, samples=(prior, current), launched_at=launched_at,
                last_progress_at=last_progress_at, failure=failure,
            )
            _atomic_json(paths["status_receipt"], receipt)
            prior_authority = prior.get("authority")
            current_authority = current.get("authority")
            prior_available = (
                isinstance(prior_authority, Mapping)
                and prior_authority.get("available") is True
            )
            current_available = (
                isinstance(current_authority, Mapping)
                and current_authority.get("available") is True
            )
            if receipt.get("blocked") is True or receipt.get("stuck") is True:
                _record_control_failure(
                    paths, failure, failure_event,
                    reason_code=(
                        "authoritative_board_blocked"
                        if receipt.get("blocked") is True
                        else "authoritative_board_stuck"
                    ),
                    error_type="ASEHHealthGateFailure",
                )
                return
            if not prior_available and not current_available:
                _record_control_failure(
                    paths, failure, failure_event,
                    reason_code="broker_status_unavailable_two_samples",
                    error_type="ASEHHealthQueryFailure",
                )
                return
            prior = current
        except Exception as exc:
            _record_control_failure(
                paths, failure, failure_event,
                reason_code="health_monitor_failed",
                error_type=type(exc).__name__,
            )
            return


def _read_live_status_receipt(
    board: Any, paths: Mapping[str, Path]
) -> tuple[dict[str, Any], float]:
    path = paths["status_receipt"]
    observed = path.stat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size > STATUS_RECEIPT_MAX_BYTES
    ):
        raise OperatorError("live status receipt file identity is unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != LIVE_STATUS_SCHEMA or payload.get("program_id") != PROGRAM:
        raise OperatorError("live status receipt identity differs")
    unsigned = dict(payload)
    receipt_cid = unsigned.pop("receipt_cid", "")
    if receipt_cid != _identity(unsigned):
        raise OperatorError("live status receipt CID is invalid")
    if not isinstance(payload.get("samples"), list) or len(payload["samples"]) != 2:
        raise OperatorError("live status receipt is not a two-sample observation")
    bootstrap = json.loads(paths["bootstrap_receipt"].read_text(encoding="utf-8"))
    for field in (
        "source_head", "repository_tree_id", "plan_root_cid", "bootstrap_receipt_id",
    ):
        if payload.get(field) != bootstrap.get(field):
            raise OperatorError(f"live status receipt has stale {field}")
    age = max(0.0, time.time() - float(payload.get("observed_at") or 0.0))
    max_age = min(
        60.0,
        max(15.0, 3.0 * float(board.payload.get("check_interval_seconds") or 10.0)),
    )
    if age > max_age:
        raise OperatorError("live status receipt is stale")
    return payload, age


def status(config_path: Path, *, require_ready: bool) -> tuple[int, dict[str, Any]]:
    board, _config = _load(config_path)
    paths = _paths(board)
    owner_status: dict[str, Any] = {}
    status_path = paths["owner"] / "quack-state-server.status.json"
    if status_path.is_file():
        try:
            owner_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner_status = {}
    try:
        receipt, age = _read_live_status_receipt(board, paths)
        receipt_available = True
        receipt_error: dict[str, Any] = {}
    except Exception as exc:
        receipt = {}
        age = float("inf")
        receipt_available = False
        receipt_error = {
            "error_type": type(exc).__name__,
            "reason": "live_status_receipt_unavailable_or_invalid",
        }
    owner_ready = owner_status.get("lifecycle") == "ready"
    healthy = bool(
        receipt_available and owner_ready and receipt.get("healthy") is True
    )
    report = {
        "schema": LIVE_STATUS_SCHEMA,
        "program_id": PROGRAM,
        "owner_ready": owner_ready,
        "owner_status": owner_status,
        "broker_authenticated_receipt": bool(
            receipt_available and receipt.get("broker_authenticated") is True
        ),
        "receipt_age_seconds": age if receipt_available else None,
        "receipt": receipt,
        "receipt_error": receipt_error,
        "healthy": healthy,
        "blocked": bool(receipt.get("blocked", False)),
        "stuck": bool(receipt.get("stuck", False)),
        "terminal": bool(receipt.get("terminal", False)),
        "observed_at": time.time(),
    }
    return (0 if healthy or not require_ready else 1), report


def preflight(config_path: Path) -> tuple[int, dict[str, Any]]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        preflight_configured_board,
    )

    board, _config = _load(config_path)
    report = preflight_configured_board(board)
    return (0 if report.get("valid") is True else 1), dict(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("materialize")
    commands.add_parser("preflight")
    run = commands.add_parser("run")
    run.add_argument("--implement", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--duration-seconds", type=float, default=float("inf"))
    show = commands.add_parser("status")
    show.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "materialize":
            payload = materialize(args.config)
            code = 0
        elif args.command == "preflight":
            code, payload = preflight(args.config)
        elif args.command == "status":
            code, payload = status(args.config, require_ready=args.require_ready)
        else:
            return run_supervisor(
                args.config,
                implement=bool(args.implement),
                duration=float(args.duration_seconds),
            )
    except (OperatorError, OSError, RuntimeError, ValueError) as exc:
        payload = {
            "schema": OPERATOR_SCHEMA, "command": args.command, "ok": False,
            "error_type": type(exc).__name__, "error": str(exc),
        }
        code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
