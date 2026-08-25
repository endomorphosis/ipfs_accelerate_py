#!/usr/bin/env python3
"""Catalog the Plan-R2-held board into the active embedded DuckDB store.

Ingests EAAEF-010..179 (and any later template) as blocked rows so DuckDB holds
the full 116-task board. This is not Plan R2 materialization: held tasks stay
blocked, is_schedulable remains false, and configured-board-launch is not started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
BOARD_PATH = CAMPAIGN / "task_board.json"
RECEIPT_PATH = CAMPAIGN / "receipts" / "host_admission" / "held_board_catalog.json"
CURSOR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "generation-cursor.json"
)
MATERIALIZATION = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "run-v14"
    / "registry"
    / "bootstrap-materialization.json"
)
HOST_EVIDENCE_MIN = 180
HOST_EVIDENCE_MAX = 191


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally argument-free offline ingestion contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _ensure_repository_importable() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _database_task_source_class() -> type[DatabaseTaskSource]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    return DatabaseTaskSource


def _acquire_state_owner_lease(control: Path) -> Any:
    """Fence the offline ingestion against the live Quack state owner."""

    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OWNER_LOCK_SUFFIX,
        OWNER_MARKER_SUFFIX,
        ExclusiveOwnerLease,
    )

    database = Path(control)
    lease = ExclusiveOwnerLease(
        lock_path=database.with_name(f".{database.name}{OWNER_LOCK_SUFFIX}"),
        marker_path=database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}"),
    )
    lease.acquire(
        server_id=f"offline:eaaef-held-board-ingest:{os.getpid()}",
        process_birth=current_process_birth(),
        database_path=database,
        generation=1,
    )
    return lease


def _cid(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _task_number(task_id: str) -> int:
    return int(str(task_id).split("-")[-1])


def _is_bootstrap(task_id: str) -> bool:
    number = _task_number(task_id)
    return number < 10 or HOST_EVIDENCE_MIN <= number <= HOST_EVIDENCE_MAX


def _active_control_db() -> Path:
    generation = "eaaef-run-v14"
    if CURSOR.is_file():
        cursor = json.loads(CURSOR.read_text(encoding="utf-8"))
        generation = str(cursor.get("active_generation") or generation)
    number = generation.rsplit("-v", 1)[-1]
    return (
        ROOT
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / f"run-v{number}"
        / "control.duckdb"
    )


def _materialization_binding(control: Path) -> dict[str, str]:
    receipt_path = control.parent / "registry" / "bootstrap-materialization.json"
    if not receipt_path.is_file():
        receipt_path = MATERIALIZATION
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "plan_root_cid": str(receipt["plan_root_cid"]),
        "source_head": str(receipt["source_head"]),
        "source_tree": str(receipt["source_tree"]),
        "repository_tree_id": str(
            receipt.get("database_materialization", {})
            .get("task_source", {})
            .get("repository_tree_id")
            or receipt["source_tree"]
        ),
    }


def _ingest_under_lease(control: Path) -> dict[str, object]:
    board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    held_catalog_task_ids = [
        str(task.get("stable_task_id") or "")
        for task in board.get("tasks") or ()
        if str(task.get("stable_task_id") or "")
        and not _is_bootstrap(str(task.get("stable_task_id") or ""))
    ]
    binding = _materialization_binding(control)
    plan_root_cid = binding["plan_root_cid"]
    tree = binding["repository_tree_id"]
    source_head = binding["source_head"]
    skipped: list[str] = []
    database_task_source = _database_task_source_class()
    with database_task_source(control, install_schema=False) as source:
        existing_page = source.list_tasks(limit=1000)
        existing = {item.task_alias: item.task_cid for item in existing_page.tasks}
        goal_cids: dict[str, str] = {}
        for goal in board.get("goals") or ():
            alias = str(goal.get("goal_id") or "")
            if not alias:
                continue
            row = source.get_goal(alias)
            if row is None:
                # get_goal may require CID; resolve via alias scan of snapshot goals
                continue
            goal_cids[alias] = str(row["goal_cid"])
        if len(goal_cids) != len(board.get("goals") or ()):
            import duckdb

            con = duckdb.connect(str(control), read_only=True)
            try:
                for alias, cid in con.execute("SELECT goal_alias, goal_cid FROM goals").fetchall():
                    goal_cids[str(alias)] = str(cid)
            finally:
                con.close()
        declared = dict(existing)
        held_specs: list[dict] = []
        for task in board.get("tasks") or ():
            alias = str(task.get("stable_task_id") or "")
            if not alias or _is_bootstrap(alias) or alias in existing:
                if alias in existing:
                    skipped.append(alias)
                continue
            task_cid = _cid(
                {
                    "schema": "EAAEFTaskIdentity@1",
                    "task_spec_cid": task.get("task_spec_cid"),
                    "plan_root_cid": plan_root_cid,
                    "source_head": source_head,
                    "repository_tree_id": tree,
                    "population": "held_awaiting_plan_r2",
                }
            )
            declared[alias] = task_cid
            declared[task_cid] = task_cid
            held_specs.append(task)
        tasks: list[dict] = []
        for ordinal, task in enumerate(held_specs, start=len(existing) + 1):
            alias = str(task["stable_task_id"])
            task_cid = declared[alias]
            goal_id = str(task.get("subgoal_id") or "EAAEF-G000")
            goal_cid = goal_cids.get(goal_id) or next(iter(goal_cids.values()))
            dependencies = [str(item) for item in task.get("dependencies") or ()]
            resolved = [declared.get(item, item) for item in dependencies]
            execution_owned = list(task.get("execution_owned_files") or [])
            execution_validation = list(task.get("execution_validation") or [])
            body = dict(task)
            body.update(
                {
                    "task_id": alias,
                    "task_alias": alias,
                    "accepted_plan_root_cid": plan_root_cid,
                    "held_board_catalog": True,
                    "future_tasks_materialized": False,
                    "configured_board_launch": False,
                }
            )
            tasks.append(
                {
                    **body,
                    "task_cid": task_cid,
                    "task_id": alias,
                    "task_alias": alias,
                    "goal_cid": goal_cid,
                    "plan_cid": plan_root_cid,
                    "objective_id": "objective:eaaef-root",
                    "ordinal": ordinal,
                    "status": "blocked",
                    "priority": str(task.get("priority") or "P0"),
                    "title": str(task.get("title") or alias),
                    "dependencies": resolved,
                    "outputs": [
                        {
                            "path": str(path),
                            "effect_id": _cid({"task": alias, "path": str(path)}),
                        }
                        for path in execution_owned
                    ],
                    "acceptance": [str(task.get("acceptance") or "")],
                    "validations": execution_validation,
                }
            )
        goals = [
            {
                "goal_cid": cid,
                "goal_alias": alias,
                "goal_id": alias,
                "title": alias,
                "ordinal": index,
                "status": "open",
            }
            for index, (alias, cid) in enumerate(sorted(goal_cids.items()), start=1)
        ]
        receipt = source.materialize(
            {
                "repository_tree_id": tree,
                "plan_root_cid": plan_root_cid,
                "goals": goals,
                "plans": [
                    {
                        "plan_cid": plan_root_cid,
                        "plan_alias": str(board.get("plan_revision") or "EAAEF-PLAN-R1"),
                        "goal_cid": goal_cids["EAAEF-G000"],
                        "status": "active",
                    }
                ],
                "tasks": tasks,
            },
            repository_tree_id=tree,
            plan_root_cid=plan_root_cid,
        )
        after = source.list_tasks(limit=1000)
        statuses: dict[str, int] = {}
        for item in after.tasks:
            statuses[item.status] = statuses.get(item.status, 0) + 1
        ready = [
            item.task_alias
            for item in source.ready_tasks(limit=1000).tasks
            if not _is_bootstrap(item.task_alias)
        ]
        if ready:
            raise RuntimeError(
                "held-board catalog leaked non-bootstrap ready tasks: " + ",".join(ready)
            )
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-held-board-catalog@1",
        "process_started": False,
        "configured_board_launch": False,
        "future_tasks_materialized": False,
        "plan_r2_applied": False,
        "control_db": str(control.relative_to(ROOT)),
        "plan_root_cid": plan_root_cid,
        "board_cid": board.get("board_cid"),
        # These established receipt fields describe the durable catalog
        # population, not only mutations made by this invocation.  Keeping
        # them stable makes a recovered/replayed ingestion byte-idempotent.
        "inserted_task_ids": held_catalog_task_ids,
        "skipped_existing_task_ids": sorted(set(skipped)),
        "inserted_count": len(held_catalog_task_ids),
        "task_count_after": len(after.tasks),
        "status_counts": statuses,
        "materialize_task_count": receipt.get("task_count"),
        "live_launch_allowed": False,
    }
    payload["receipt_cid"] = _cid(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def ingest() -> dict[str, object]:
    control = _active_control_db()
    lease = _acquire_state_owner_lease(control)
    try:
        return _ingest_under_lease(control)
    finally:
        lease.release(fence_token=lease.fence_token)


def main(argv: list[str] | None = None) -> int:
    _parse_args(argv)
    print(json.dumps(ingest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
