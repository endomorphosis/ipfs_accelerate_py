#!/usr/bin/env python3
"""Run scheduler-delegated completion for seal-gated manual PDR tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ipfs_accelerate_py.agent_supervisor.control.delegated_operator_completion import (
    DelegatedOperatorCompletionPolicy,
    complete_ready_sealed_manual_tasks,
    complete_sealed_manual_task,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    load_supervisor_scheduler_config,
)


DEFAULT_SCHEDULER = (
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)
DEFAULT_TODO = (
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md"
)


def _parse_board(todo_path: Path, prefix: str = "## PDR-") -> list[dict]:
    text = todo_path.read_text(encoding="utf-8")
    tasks: list[dict] = []
    blocks = re.split(r"(?=^## PDR-)", text, flags=re.M)
    for block in blocks:
        m = re.match(r"^## (PDR-\d+)\s*(.*)", block)
        if not m:
            continue
        task_id = m.group(1)
        status_m = re.search(r"(?m)^- Status:\s*(\S+)", block)
        depends_m = re.search(r"(?m)^- Depends on:\s*(.+)$", block)
        validation_m = re.search(r"(?m)^- Validation:\s*(.+)$", block)
        depends = []
        if depends_m:
            depends = [
                part.strip()
                for part in depends_m.group(1).split(",")
                if part.strip()
            ]
        tasks.append(
            {
                "task_id": task_id,
                "status": status_m.group(1) if status_m else "pending",
                "depends_on": depends,
                "validation": validation_m.group(1).strip()
                if validation_m
                else "",
            }
        )
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delegated operator completion for seal-gated manual tasks"
    )
    parser.add_argument(
        "--scheduler-config",
        default=DEFAULT_SCHEDULER,
        help="Repository-relative scheduler config",
    )
    parser.add_argument(
        "--todo-path",
        default=DEFAULT_TODO,
        help="Repository-relative taskboard",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Optional task id filter (repeatable)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_REPO),
        help="Repository root",
    )
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    profile = load_supervisor_scheduler_config(
        root / args.scheduler_config,
        repo_root=root,
    )
    policy = DelegatedOperatorCompletionPolicy.from_mapping(
        profile.get("delegated_operator_completion")
    )
    if not policy.enabled:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "delegated_operator_completion is disabled",
                },
                indent=2,
            )
        )
        return 2

    board = _parse_board(root / args.todo_path)
    completed = [t["task_id"] for t in board if t["status"] == "completed"]
    pending = [t["task_id"] for t in board if t["status"] != "completed"]
    if args.tasks:
        pending = [t for t in pending if t in set(args.tasks)]
    depends = {t["task_id"]: t["depends_on"] for t in board}
    validations = {t["task_id"]: t["validation"] for t in board}
    seal_configs = profile.get("manual_completion_seals") or {}

    if args.tasks and len(args.tasks) == 1:
        task_id = args.tasks[0]
        seal = seal_configs.get(task_id)
        if seal is None:
            print(json.dumps({"ok": False, "error": "no seal config"}, indent=2))
            return 2
        outcome = complete_sealed_manual_task(
            repo_root=root,
            todo_path=root / args.todo_path,
            scheduler_path=root / args.scheduler_config,
            task_id=task_id,
            board_namespace=str(profile["board_namespace"]),
            seal_config=seal,
            validation_command=validations.get(task_id, ""),
            policy=policy,
        )
        print(json.dumps({"ok": True, "result": outcome}, indent=2, sort_keys=True))
        return 0

    results = complete_ready_sealed_manual_tasks(
        repo_root=root,
        todo_path=root / args.todo_path,
        scheduler_path=root / args.scheduler_config,
        board_namespace=str(profile["board_namespace"]),
        seal_configs=seal_configs,
        validation_commands=validations,
        completed_task_ids=completed,
        pending_task_ids=pending,
        depends_on=depends,
        policy=policy,
    )
    print(json.dumps({"ok": True, "results": results}, indent=2, sort_keys=True))
    return 0 if not results.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
