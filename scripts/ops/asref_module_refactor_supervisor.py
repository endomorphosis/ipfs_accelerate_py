#!/usr/bin/env python3
"""Preflight and launch the ASREF agent_supervisor module-refactor supervisors.

Binds lanes to the ASREF todo/objective heap on branch
``refactor/agent-supervisor-layout``. Prefer running from the isolated
worktree so concurrent main-checkout supervisors cannot thrash the board.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (  # noqa: E402
    ImplementationSupervisorTrackConfig,
    build_configured_multi_supervisor_cli_runner,
    utc_run_stamp,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_file,
)


TODO_RELATIVE = Path("docs/architecture/agent_supervisor_module_refactor.todo.md")
OBJECTIVE_RELATIVE = Path(
    "docs/architecture/agent_supervisor_module_refactor.objectives.md"
)
PLAN_RELATIVE = Path("docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md")
INVENTORY_RELATIVE = Path("docs/architecture/asref/move_map.json")
TASK_PREFIX = "ASREF-"
TASK_HEADER_PREFIX = f"## {TASK_PREFIX}"
PROTECTED_PATHS = (
    TODO_RELATIVE.as_posix(),
    OBJECTIVE_RELATIVE.as_posix(),
    PLAN_RELATIVE.as_posix(),
)
DEFAULT_NAMESPACE = "asref-v1"
DEFAULT_LANES = 4
DEFAULT_MERGE_BRANCH = "refactor/agent-supervisor-layout"
DEFAULT_REFILL_OPEN_TASK_THRESHOLD = 3


def _runtime_root(namespace: str) -> Path:
    configured = os.environ.get("IPFS_ACCELERATE_AGENT_SUPERVISOR_ROOT", "").strip()
    base = (
        Path(configured).expanduser()
        if configured
        else Path.home()
        / ".local"
        / "share"
        / "ipfs_accelerate_py"
        / "agent-supervisor"
    )
    return base / namespace


def inspect_board(
    *,
    repo_root: Path = REPO_ROOT,
    lanes: int = DEFAULT_LANES,
    merge_branch: str = DEFAULT_MERGE_BRANCH,
) -> dict[str, Any]:
    todo_path = repo_root / TODO_RELATIVE
    objective_path = repo_root / OBJECTIVE_RELATIVE
    errors: list[str] = []
    if not todo_path.is_file():
        errors.append(f"missing todo board: {todo_path}")
    if not objective_path.is_file():
        errors.append(f"missing objective heap: {objective_path}")

    branch = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    head_branch = (branch.stdout or "").strip()
    if head_branch and head_branch != merge_branch:
        errors.append(
            f"repo HEAD is {head_branch!r}; expected {merge_branch!r} for ASREF"
        )

    tasks = (
        parse_task_file(todo_path, TASK_HEADER_PREFIX) if todo_path.is_file() else []
    )
    goals = (
        parse_goal_heap(objective_path.read_text(encoding="utf-8"))
        if objective_path.is_file()
        else []
    )
    open_tasks = [
        task
        for task in tasks
        if str(getattr(task, "status", "")).lower()
        not in {"completed", "done", "closed", "cancelled", "canceled"}
    ]
    ready = [
        task.task_id
        for task in open_tasks
        if not getattr(task, "depends_on", ())
        or all(
            any(
                other.task_id == dep
                and str(getattr(other, "status", "")).lower()
                in {"completed", "done", "closed"}
                for other in tasks
            )
            for dep in task.depends_on
        )
    ]
    # Also treat depends_on satisfied when dep missing from board (loose seed boards)
    ready_loose = []
    completed_ids = {
        task.task_id
        for task in tasks
        if str(getattr(task, "status", "")).lower()
        in {"completed", "done", "closed"}
    }
    task_ids = {task.task_id for task in tasks}
    for task in open_tasks:
        deps = tuple(getattr(task, "depends_on", ()) or ())
        if all(dep in completed_ids or dep not in task_ids for dep in deps):
            ready_loose.append(task.task_id)

    if not open_tasks:
        errors.append("no open ASREF tasks on the board")
    if not ready_loose:
        errors.append("no ready ASREF tasks (all open tasks blocked)")

    return {
        "schema": "ipfs_accelerate_py.asref.preflight.v1",
        "ok": not errors,
        "errors": errors,
        "repo_root": str(repo_root),
        "head_branch": head_branch,
        "merge_branch": merge_branch,
        "task_count": len(tasks),
        "open_task_count": len(open_tasks),
        "ready_task_ids": ready_loose,
        "goal_count": len(goals),
        "lanes": lanes,
        "protected_paths": list(PROTECTED_PATHS),
        "inventory_present": (repo_root / INVENTORY_RELATIVE).is_file(),
    }


def _common_args(
    *,
    repo_root: Path,
    runtime_root: Path,
    merge_branch: str,
    enable_objective_refill: bool,
    refill_open_task_threshold: int,
) -> tuple[str, ...]:
    python = sys.executable
    entry = repo_root / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
    args = [
        "--todo-path",
        str(repo_root / TODO_RELATIVE),
        "--task-prefix",
        TASK_PREFIX,
        "--worktree-root",
        str(runtime_root / "worktrees"),
        "--merge-target-branch",
        merge_branch,
        "--stale-seconds",
        "1800",
        "--check-interval",
        "15",
        "--watchdog-startup-grace-seconds",
        "600",
        "--max-restarts",
        "50",
        "--max-task-attempts",
        "5",
        "--daemon-interval",
        "20",
        "--implementation-timeout",
        "3600",
        "--implementation-log-stall-seconds",
        "1200",
        "--llm-merge-resolver-command",
        f"{python} -m ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback",
        "--llm-merge-resolver-timeout-seconds",
        "1800",
        "--worktree-reconciliation-max-merges",
        "1",
        "--no-objective-task-janitor",
        "--no-objective-goal-completion-reconcile",
        "--implement",
        "--log-level",
        "INFO",
    ]
    for protected in PROTECTED_PATHS:
        args.extend(["--implementation-protected-path", protected])
    if enable_objective_refill:
        args.extend(
            [
                "--objective-refill-scan",
                "--objective-path",
                str(repo_root / OBJECTIVE_RELATIVE),
                "--objective-graph-path",
                str(runtime_root / "objective_graph.json"),
                "--objective-bundle-dir",
                str(runtime_root / "bundles"),
                "--objective-dataset-dir",
                str(runtime_root / "datasets"),
                "--objective-discovery-dir",
                str(runtime_root / "discovery"),
                "--objective-todo-vector-index-path",
                str(runtime_root / "bundles" / "todo_vector_index.json"),
                "--objective-scan-min-open-tasks",
                str(refill_open_task_threshold),
                "--objective-scan-max-findings",
                "4",
                "--objective-scan-cooldown-seconds",
                "900",
                "--objective-refill-timeout-seconds",
                "600",
                "--objective-goal-prefix",
                "ASREF-G",
                "--objective-root-goal-id",
                "ASREF-G000",
                "--objective-root-goal-title",
                "Clear agent_supervisor package layout and monorepo root hygiene",
                "--objective-tracking-document-title",
                "Agent Supervisor Module Refactor Objective Heap",
                "--objective-mission-term",
                "agent_supervisor,package,refactor,import,layout,module,readme",
            ]
        )
    # entry is used only by multi_supervisor track script path
    _ = entry
    return tuple(args)


def launch(args: argparse.Namespace) -> int:
    report = inspect_board(
        repo_root=REPO_ROOT,
        lanes=args.lanes,
        merge_branch=args.merge_branch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"] and not args.force:
        return 2

    provider = str(getattr(args, "implementation_provider", "") or "").strip()
    if provider:
        os.environ["IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"] = provider

    runtime_root = _runtime_root(args.namespace)
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "master").mkdir(parents=True, exist_ok=True)
    (runtime_root / "state").mkdir(parents=True, exist_ok=True)
    (runtime_root / "worktrees").mkdir(parents=True, exist_ok=True)
    (runtime_root / "bundles").mkdir(parents=True, exist_ok=True)
    (runtime_root / "discovery").mkdir(parents=True, exist_ok=True)
    (runtime_root / "datasets").mkdir(parents=True, exist_ok=True)

    stamp = utc_run_stamp()
    entry = REPO_ROOT / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
    if not entry.is_file():
        print(json.dumps({"error": f"missing entry script: {entry}"}), file=sys.stderr)
        return 2

    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=REPO_ROOT,
        duration_seconds=args.duration_seconds,
        heartbeat_interval_seconds=30,
        supervisor_status_stale_seconds=300,
        stop_grace_seconds=900,
        stamp=stamp,
        master_dir=runtime_root / "master",
        master_log=runtime_root / "master" / f"asref_{stamp}.log",
        master_pid_path=runtime_root / "master" / "asref.pid",
        label="asref-module-refactor",
        python_executable=sys.executable,
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="asref-module-refactor",
                script_path=entry,
                state_dir=runtime_root / "state",
                state_prefix="asref",
            ),
        ),
        common_args=_common_args(
            repo_root=REPO_ROOT,
            runtime_root=runtime_root,
            merge_branch=args.merge_branch,
            enable_objective_refill=args.enable_objective_refill,
            refill_open_task_threshold=args.refill_open_task_threshold,
        ),
        detach=not args.foreground,
    )
    lane_args = [
        "--implementation-supervisor-lanes-per-track",
        str(args.lanes),
    ]
    payload = {
        "schema": "ipfs_accelerate_py.asref.launch.v1",
        "argv": [*runner.args(), *lane_args],
        "runtime_root": str(runtime_root),
        "stamp": stamp,
        "master_log": str(runtime_root / "master" / f"asref_{stamp}.log"),
        "master_pid_path": str(runtime_root / "master" / "asref.pid"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return runner.run(lane_args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or launch ASREF module-refactor supervisors"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--lanes", type=int, default=DEFAULT_LANES)
        p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
        p.add_argument("--merge-branch", default=DEFAULT_MERGE_BRANCH)
        p.add_argument(
            "--duration-seconds",
            type=float,
            default=float("inf"),
            help="Master supervisor duration (default: inf)",
        )
        p.add_argument(
            "--enable-objective-refill",
            action="store_true",
            help="Enable objective-heap refill into the ASREF board",
        )
        p.add_argument(
            "--refill-open-task-threshold",
            type=int,
            default=DEFAULT_REFILL_OPEN_TASK_THRESHOLD,
        )
        p.add_argument(
            "--force",
            action="store_true",
            help="Launch even if preflight reports errors",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Print launch argv without starting supervisors",
        )
        p.add_argument(
            "--foreground",
            action="store_true",
            help="Do not detach the multi-supervisor master",
        )
        p.add_argument(
            "--implementation-provider",
            default=os.environ.get(
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER", ""
            ),
            help=(
                "Set IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER for child "
                "daemons (e.g. grok, goose, codex, auto)"
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = inspect_board(
            repo_root=REPO_ROOT,
            lanes=args.lanes,
            merge_branch=args.merge_branch,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    if args.command == "launch":
        return launch(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
