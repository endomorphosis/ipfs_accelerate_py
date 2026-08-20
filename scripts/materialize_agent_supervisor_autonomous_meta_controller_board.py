#!/usr/bin/env python3
"""Materialize and verify the sealed APMC board in its DuckDB authority.

Initial materialization is allowed only while no Quack owner is serving the
new database.  Once Quack owns the file, verification must use ``--endpoint``
so this script never becomes a second file owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (  # noqa: E402
    MAX_QUERY_LIMIT,
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (  # noqa: E402
    is_quack_transport_target,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    DatabaseImplementationDaemon,
    parse_task_text,
)

PROGRAM_ID = "agent-supervisor-autonomous-meta-controller-v1"
ROOT_OBJECTIVE = "APMC-G000"
BRANCH = "codex/agent-supervisor-autonomous-meta-controller-v1"
TODO_PATH = REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md"
OBJECTIVES_PATH = (
    REPO_ROOT / "docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md"
)
PLAN_PATH = REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_agent_supervisor_autonomous_meta_controller_board.py"
EXPECTED_TASK_IDS = tuple(f"APMC-{index:03d}" for index in range(21))
EXPECTED_GOAL_IDS = tuple(["APMC-G000", *(f"APMC-G{index:03d}" for index in range(10, 111, 10))])
EXPECTED_MANUAL_REVIEW_TASK_IDS = ("APMC-019", "APMC-020")


class MaterializationError(RuntimeError):
    """The board cannot be safely materialized or its projection changed."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise MaterializationError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _require_clean_committed_source() -> tuple[str, str]:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise MaterializationError(
            "refusing to bind APMC state to a dirty worktree; commit the exact source first"
        )
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise MaterializationError(f"expected branch {BRANCH!r}, observed {branch!r}")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    return head, tree


def _safe_new_database(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise MaterializationError("database must be inside this repository worktree") from exc
    if not relative.parts or relative.parts[0] not in {"data", "state"}:
        raise MaterializationError("database must be under ignored data/ or state/")
    if lexical.suffix.lower() not in {".duckdb", ".ddb"}:
        raise MaterializationError("database must have a .duckdb or .ddb suffix")
    current = REPO_ROOT
    for component in relative.parts[:-1]:
        current /= component
        if current.is_symlink():
            raise MaterializationError("database parent may not traverse a symlink")
    return lexical


def _validated_inputs() -> tuple[list[Any], list[Any]]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--check-all"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise MaterializationError(
            "APMC validator refused materialization: " + (result.stdout or result.stderr).strip()
        )
    tasks = parse_task_text(
        TODO_PATH.read_text(encoding="utf-8"),
        path=TODO_PATH,
        task_header_prefix="## APMC-",
    )
    goals = parse_goal_heap(OBJECTIVES_PATH.read_text(encoding="utf-8"))
    if tuple(task.task_id for task in tasks) != EXPECTED_TASK_IDS:
        raise MaterializationError("task population changed after validation")
    if tuple(goal.goal_id for goal in goals) != EXPECTED_GOAL_IDS:
        raise MaterializationError("goal population changed after validation")
    return tasks, goals


def build_population(*, source_head: str, source_tree: str) -> dict[str, Any]:
    tasks, goals = _validated_inputs()
    source_bindings = {
        "source_head": source_head,
        "repository_tree_id": source_tree,
        "plan_sha256": _sha256(PLAN_PATH),
        "objectives_sha256": _sha256(OBJECTIVES_PATH),
        "taskboard_sha256": _sha256(TODO_PATH),
    }
    plan_root = content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/accepted-plan-root@1",
            "program_id": PROGRAM_ID,
            **source_bindings,
        }
    )
    goal_cids = {
        goal.goal_id: content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/goal@1",
                "program_id": PROGRAM_ID,
                "goal_id": goal.goal_id,
                "title": goal.title,
                "source_tree": source_tree,
            }
        )
        for goal in goals
    }
    objective_rows: list[dict[str, Any]] = []
    for ordinal, goal in enumerate(goals, start=1):
        parent = str(goal.fields.get("parent") or "").strip()
        objective_rows.append(
            {
                "goal_cid": goal_cids[goal.goal_id],
                "goal_id": goal.goal_id,
                "goal_alias": goal.goal_id,
                "title": goal.title,
                "objective_id": ROOT_OBJECTIVE if goal.goal_id == ROOT_OBJECTIVE else "",
                "objective_alias": ROOT_OBJECTIVE,
                "parent_goal_cid": goal_cids[parent] if parent else "",
                "ordinal": ordinal,
                "status": "open",
                "priority": str(goal.fields.get("priority") or "P2"),
                "program_id": PROGRAM_ID,
                "source_tree": source_tree,
                "fields": dict(goal.fields),
            }
        )
    goal_edges: list[dict[str, str]] = []
    for goal in goals:
        parent = str(goal.fields.get("parent") or "").strip()
        if parent:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal.goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in _csv(str(goal.fields.get("depends_on") or "")):
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": goal_cids[goal.goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
    task_cids_by_alias = {task.task_id: task.canonical_task_cid for task in tasks}
    task_rows: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks, start=1):
        validation_rows = [
            {
                "argv": shlex.split(command),
                "current_tree_required": True,
                "shell_interpolation_permitted": False,
            }
            for command in task.validation
        ]
        task_rows.append(
            {
                "task_cid": task.canonical_task_cid,
                "task_id": task.task_id,
                "task_alias": task.task_id,
                "task_key": task.canonical_task_key,
                "goal_cid": goal_cids[str(task.metadata["goal id"])],
                "goal_id": str(task.metadata["goal id"]),
                "plan_cid": plan_root,
                "objective_id": ROOT_OBJECTIVE,
                "ordinal": ordinal,
                "status": "todo",
                "priority": task.priority,
                "title": task.title,
                "objective": task.title,
                "track": task.track,
                "completion": task.completion,
                "review_only": str(task.metadata["review only"]).strip().casefold()
                in {"1", "true", "yes"},
                "board_namespace": PROGRAM_ID,
                "repository_tree_id": source_tree,
                "source_head": source_head,
                "source_line": task.source_line,
                "metadata": dict(task.metadata),
                "provenance": {
                    "board_namespace": PROGRAM_ID,
                    "source_path": TODO_PATH.relative_to(REPO_ROOT).as_posix(),
                    "acceptance": task.acceptance,
                },
                # Bind aliases before insertion so a dependency on a later
                # task ordinal cannot remain an unresolved display alias.
                "dependencies": [task_cids_by_alias[dependency] for dependency in task.depends_on],
                "outputs": [
                    {
                        "path": path,
                        "effect_class": "bounded_repository_path",
                        "effect_id": content_identity({"task_id": task.task_id, "path": path}),
                    }
                    for path in task.outputs
                ],
                "acceptance": [
                    {
                        "criterion": task.acceptance,
                        "evidence_policy": {
                            "current_tree_required": True,
                            "declared_validation_required": True,
                            "markdown_status_is_authority": False,
                        },
                    }
                ],
                "validations": validation_rows,
                # The attempt-local Portal projection also understands this
                # singular compatibility field.  It remains a declaration,
                # never completion evidence.
                "validation": " ; ".join(task.validation),
            }
        )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-population@1",
        "program_id": PROGRAM_ID,
        "repository_tree_id": source_tree,
        "source_head": source_head,
        "plan_root_cid": plan_root,
        "objectives": objective_rows,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "APMC-PLAN-R1",
                "goal_cid": goal_cids[ROOT_OBJECTIVE],
                "status": "active",
                "program_id": PROGRAM_ID,
                **source_bindings,
            }
        ],
        "tasks": task_rows,
        "task_cids_by_alias": task_cids_by_alias,
        "goal_cids_by_alias": goal_cids,
    }


def _verify_source(source: DatabaseTaskSource, population: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = source.snapshot()
    page = source.list_tasks(limit=100)
    ready = source.ready_tasks(limit=100)
    tasks = tuple(page.tasks)
    aliases = tuple(item.task_alias for item in tasks)
    expected_tasks = tuple(population["tasks"])
    by_alias = {item.task_alias: item for item in tasks}
    observed_goal_edges = tuple(
        (
            str(item["parent_goal_cid"]),
            str(item["child_goal_cid"]),
            str(item["edge_kind"]),
        )
        for item in source.list_goal_edges(limit=MAX_QUERY_LIMIT)
    )
    expected_goal_edges = tuple(
        sorted(
            (
                str(item["parent_goal_cid"]),
                str(item["child_goal_cid"]),
                str(item["edge_kind"]),
            )
            for item in population["goal_edges"]
        )
    )
    errors: list[str] = []
    if aliases != EXPECTED_TASK_IDS:
        errors.append("task aliases/order changed")
    if snapshot.task_count != len(EXPECTED_TASK_IDS) or snapshot.goal_count != len(
        EXPECTED_GOAL_IDS
    ):
        errors.append("task/goal population count changed")
    if snapshot.plan_root_cid != population["plan_root_cid"]:
        errors.append("plan root changed")
    if snapshot.repository_tree_id != population["repository_tree_id"]:
        errors.append("repository tree changed")
    if observed_goal_edges != expected_goal_edges:
        errors.append("goal parent/dependency edges changed")
    for expected in expected_tasks:
        observed = by_alias.get(str(expected["task_alias"]))
        if observed is None:
            continue
        expected_dependencies = tuple(
            sorted(
                population["task_cids_by_alias"].get(item, item)
                for item in expected["dependencies"]
            )
        )
        if (
            observed.task_cid != expected["task_cid"]
            or tuple(observed.dependencies) != expected_dependencies
        ):
            errors.append(f"{observed.task_alias} identity/dependencies changed")
        if tuple(str(item.get("path") or "") for item in observed.outputs) != tuple(
            str(item["path"]) for item in expected["outputs"]
        ):
            errors.append(f"{observed.task_alias} outputs changed")
        if not observed.acceptance or not observed.validations:
            errors.append(f"{observed.task_alias} evidence declarations are missing")
        if (
            observed.body.get("review_only") is not expected["review_only"]
            or str(observed.body.get("completion") or "") != expected["completion"]
        ):
            errors.append(f"{observed.task_alias} completion/review gate changed")
        expected_auto_claim_forbidden = (
            expected["review_only"] is True or str(expected["completion"]).casefold() == "manual"
        )
        if (
            DatabaseImplementationDaemon._automatic_claim_forbidden(observed)  # noqa: SLF001
            is not expected_auto_claim_forbidden
        ):
            errors.append(f"{observed.task_alias} automatic-claim gate changed")
    ready_aliases = tuple(item.task_alias for item in ready.tasks)
    if ready_aliases != ("APMC-000",):
        errors.append(f"initial ready set changed: {ready_aliases!r}")
    if errors:
        raise MaterializationError("; ".join(errors))
    review_only_aliases = tuple(
        str(item["task_alias"]) for item in expected_tasks if item["review_only"] is True
    )
    manual_completion_aliases = tuple(
        str(item["task_alias"])
        for item in expected_tasks
        if str(item["completion"]).casefold() == "manual"
    )
    if review_only_aliases != EXPECTED_MANUAL_REVIEW_TASK_IDS:
        errors.append(f"review-only task set changed: {review_only_aliases!r}")
    if manual_completion_aliases != EXPECTED_MANUAL_REVIEW_TASK_IDS:
        errors.append(f"manual-completion task set changed: {manual_completion_aliases!r}")
    if errors:
        raise MaterializationError("; ".join(errors))
    receipt: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-materialization-receipt@1",
        "program_id": PROGRAM_ID,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "projection_cid": snapshot.projection_cid,
        "source_identity": snapshot.source_identity,
        "task_count": snapshot.task_count,
        "goal_count": snapshot.goal_count,
        "goal_edge_count": len(observed_goal_edges),
        "goal_edges_cid": content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/goal-edge-set@1",
                "edges": observed_goal_edges,
            }
        ),
        "dependency_count": snapshot.dependency_count,
        "ready_task_aliases": list(ready_aliases),
        "task_cids_by_alias": dict(population["task_cids_by_alias"]),
        "review_only_task_aliases": list(review_only_aliases),
        "manual_completion_task_aliases": list(manual_completion_aliases),
        "duckdb_authoritative": True,
        "quack_required_after_materialization": True,
        "ducklake_authoritative": False,
        "ducklake_projection_status": "disabled_activation_held",
        "markdown_status_authoritative": False,
    }
    receipt["receipt_id"] = content_identity(receipt)
    return receipt


def materialize(database: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    if database.exists():
        raise MaterializationError(
            "database already exists; refuse a second direct owner "
            "(use --verify-only --endpoint after Quack starts)"
        )
    database.parent.mkdir(parents=True, exist_ok=True)
    source = DatabaseTaskSource(
        database,
        owner_id="apmc-materializer:exclusive-bootstrap",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=True,
    )
    try:
        source.materialize(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        return _verify_source(source, population)
    finally:
        source.close()


def verify(endpoint_or_database: str, population: Mapping[str, Any]) -> dict[str, Any]:
    if not is_quack_transport_target(endpoint_or_database):
        raise MaterializationError("post-bootstrap verification requires a loopback Quack endpoint")
    source = DatabaseTaskSource(
        endpoint_or_database,
        owner_id="apmc-materializer:read-only-verifier",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    try:
        return _verify_source(source, population)
    finally:
        source.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="New repository-local DuckDB target.")
    parser.add_argument(
        "--endpoint",
        default="",
        help="Loopback quack: endpoint for verification after the owner starts.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Perform no population writes.")
    parser.add_argument(
        "--receipt",
        default="",
        help="Optional repository-local runtime receipt path under data/ or state/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        head, tree = _require_clean_committed_source()
        population = build_population(source_head=head, source_tree=tree)
        database = _safe_new_database(args.database)
        if args.verify_only:
            if not args.endpoint:
                raise MaterializationError(
                    "--verify-only requires --endpoint; direct DuckDB verification "
                    "would create a second owner"
                )
            receipt = verify(str(args.endpoint), population)
        else:
            if args.endpoint:
                raise MaterializationError("--endpoint is valid only with --verify-only")
            receipt = materialize(database, population)
        if args.receipt:
            receipt_path = (
                _safe_new_database(args.receipt)
                if str(args.receipt).endswith((".duckdb", ".ddb"))
                else Path(args.receipt)
            )
            if not receipt_path.is_absolute():
                receipt_path = REPO_ROOT / receipt_path
            receipt_path = Path(os.path.abspath(receipt_path))
            try:
                relative = receipt_path.relative_to(REPO_ROOT)
            except ValueError as exc:
                raise MaterializationError("receipt must remain inside the repository") from exc
            if not relative.parts or relative.parts[0] not in {"data", "state"}:
                raise MaterializationError("receipt must be under ignored data/ or state/")
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/autonomy/board-materialization-error@1",
            "program_id": PROGRAM_ID,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
