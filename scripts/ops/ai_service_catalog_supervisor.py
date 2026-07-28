#!/usr/bin/env python3
"""Preflight and launch the six-lane AI service catalog supervisor."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    task_implementation_protected_path_conflicts,
)


TODO_RELATIVE = Path("docs/architecture/ai_service_catalog.todo.md")
OBJECTIVE_RELATIVE = Path("docs/architecture/ai_service_catalog.objectives.md")
TASK_PREFIX = "AICAT-"
TASK_HEADER_PREFIX = f"## {TASK_PREFIX}"
BASELINE_TASK_IDS = frozenset(f"AICAT-{index:03d}" for index in range(1, 21))
PROTECTED_PATHS = (TODO_RELATIVE.as_posix(), OBJECTIVE_RELATIVE.as_posix())
DEFAULT_NAMESPACE = "ai-service-catalog-v2"
DEFAULT_LANES = 6
DEFAULT_REFILL_OPEN_TASK_THRESHOLD = 2


def _git_path_state(repo_root: Path, relative_path: Path) -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path.as_posix()],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 0
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative_path.as_posix()],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "path": relative_path.as_posix(),
        "tracked": tracked,
        "clean": status.returncode == 0 and not status.stdout.strip(),
        "status": status.stdout.strip(),
    }


def _dependency_errors(tasks_by_id: dict[str, Any]) -> tuple[list[str], list[str]]:
    unknown = sorted(
        {
            f"{task.task_id}->{dependency}"
            for task in tasks_by_id.values()
            for dependency in task.depends_on
            if dependency not in tasks_by_id
        }
    )
    incoming = {
        task_id: {
            dependency
            for dependency in task.depends_on
            if dependency in tasks_by_id
        }
        for task_id, task in tasks_by_id.items()
    }
    ready = sorted(task_id for task_id, dependencies in incoming.items() if not dependencies)
    visited: list[str] = []
    while ready:
        task_id = ready.pop(0)
        visited.append(task_id)
        for candidate_id in sorted(incoming):
            if task_id not in incoming[candidate_id]:
                continue
            incoming[candidate_id].remove(task_id)
            if not incoming[candidate_id] and candidate_id not in visited:
                ready.append(candidate_id)
        ready.sort()
    cyclic = sorted(task_id for task_id, dependencies in incoming.items() if dependencies)
    return unknown, cyclic


def inspect_board(
    *,
    repo_root: Path = REPO_ROOT,
    lanes: int = DEFAULT_LANES,
    enable_objective_refill: bool = False,
    refill_open_task_threshold: int = DEFAULT_REFILL_OPEN_TASK_THRESHOLD,
) -> dict[str, Any]:
    """Return a machine-readable launch preflight for the AICAT board."""

    todo_path = repo_root / TODO_RELATIVE
    objective_path = repo_root / OBJECTIVE_RELATIVE
    tasks = parse_task_file(todo_path, TASK_HEADER_PREFIX)
    tasks_by_id = {task.task_id: task for task in tasks}
    task_ids = [task.task_id for task in tasks]
    duplicate_task_ids = sorted(
        task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1
    )
    missing_baseline_task_ids = sorted(BASELINE_TASK_IDS.difference(tasks_by_id))
    unknown_dependencies, cyclic_task_ids = _dependency_errors(tasks_by_id)
    completed_task_ids = {
        task.task_id for task in tasks if task.status == "completed"
    }
    blocked_task_ids = {
        task.task_id
        for task in tasks
        if task.status == "blocked"
        or task.metadata.get("is schedulable", "true").lower() == "false"
        or task.metadata.get("review only", "false").lower() == "true"
    }
    ready_tasks = [
        task
        for task in tasks
        if task.task_id not in completed_task_ids
        and task.task_id not in blocked_task_ids
        and all(dependency in completed_task_ids for dependency in task.depends_on)
    ]
    nominal_ready_by_lane = {
        str(index): sorted(
            task.task_id
            for task in ready_tasks
            if (
                int(re.search(r"(\d+)$", task.task_id).group(1)) % lanes
                if re.search(r"(\d+)$", task.task_id)
                else 0
            )
            == index
        )
        for index in range(lanes)
    }
    all_ready_task_ids = sorted(task.task_id for task in ready_tasks)
    fallback_ready_by_lane = {
        lane: ([] if task_ids else list(all_ready_task_ids))
        for lane, task_ids in nominal_ready_by_lane.items()
    }
    scheduler_candidate_ready_by_lane = {
        lane: (list(task_ids) if task_ids else list(all_ready_task_ids))
        for lane, task_ids in nominal_ready_by_lane.items()
    }
    protected_conflicts = {
        task.task_id: list(conflicts)
        for task in tasks
        if task.task_id not in completed_task_ids
        and task.task_id not in blocked_task_ids
        for conflicts in [
            task_implementation_protected_path_conflicts(
                task,
                PROTECTED_PATHS,
            )
        ]
        if conflicts
    }
    path_states = [
        _git_path_state(repo_root, TODO_RELATIVE),
        _git_path_state(repo_root, OBJECTIVE_RELATIVE),
    ]
    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    open_task_count = len(tasks) - len(completed_task_ids)
    errors: list[str] = []
    if duplicate_task_ids:
        errors.append("duplicate task IDs")
    if missing_baseline_task_ids:
        errors.append("baseline task IDs are missing")
    if unknown_dependencies:
        errors.append("task dependencies reference unknown IDs")
    if cyclic_task_ids:
        errors.append("task dependency graph contains a cycle")
    if protected_conflicts:
        errors.append("schedulable tasks declare operator-protected outputs")
    if any(not state["tracked"] for state in path_states):
        errors.append("control-plane documents must be tracked")
    if any(not state["clean"] for state in path_states):
        errors.append("control-plane documents must be clean before launch")
    if not goals:
        errors.append("objective heap contains no goals")
    if (
        completed_task_ids == {"AICAT-001"}
        and set(nominal_ready_by_lane) != {
            lane
            for lane, task_ids_for_lane in nominal_ready_by_lane.items()
            if task_ids_for_lane
        }
    ):
        errors.append("initial ready wave does not cover every configured lane")
    if enable_objective_refill and open_task_count > refill_open_task_threshold:
        errors.append(
            "objective refill is gated until the baseline backlog is nearly drained"
        )
    return {
        "schema": "ipfs_accelerate_py.ai_service_catalog.preflight.v1",
        "ok": not errors,
        "errors": errors,
        "repo_root": str(repo_root),
        "task_count": len(tasks),
        "goal_count": len(goals),
        "completed_task_ids": sorted(completed_task_ids),
        "open_task_count": open_task_count,
        "ready_task_ids": all_ready_task_ids,
        "ready_by_lane": nominal_ready_by_lane,
        "fallback_ready_by_lane": fallback_ready_by_lane,
        "scheduler_candidate_ready_by_lane": scheduler_candidate_ready_by_lane,
        "lane_assignment_policy": (
            "numeric_suffix_modulo_with_global_unclaimed_ready_fallback"
        ),
        "duplicate_task_ids": duplicate_task_ids,
        "missing_baseline_task_ids": missing_baseline_task_ids,
        "unknown_dependencies": unknown_dependencies,
        "cyclic_task_ids": cyclic_task_ids,
        "protected_path_conflicts": protected_conflicts,
        "control_paths": path_states,
        "objective_refill_enabled": enable_objective_refill,
        "refill_open_task_threshold": refill_open_task_threshold,
    }


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


def _common_args(
    *,
    runtime_root: Path,
    enable_objective_refill: bool,
    refill_open_task_threshold: int,
) -> tuple[str, ...]:
    python = sys.executable
    args = [
        "--todo-path",
        str(REPO_ROOT / TODO_RELATIVE),
        "--task-prefix",
        TASK_PREFIX,
        "--worktree-root",
        str(runtime_root / "worktrees"),
        "--merge-target-branch",
        "main",
        "--stale-seconds",
        "1800",
        "--check-interval",
        "15",
        "--watchdog-startup-grace-seconds",
        "900",
        "--max-restarts",
        "100",
        "--max-task-attempts",
        "3",
        "--daemon-interval",
        "20",
        "--implementation-timeout",
        "3600",
        "--implementation-log-stall-seconds",
        "1200",
        "--llm-merge-resolver-command",
        f"{python} -m ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback",
        "--llm-merge-resolver-timeout-seconds",
        "1800",
        "--worktree-reconciliation-max-merges",
        "1",
        "--implementation-protected-path",
        TODO_RELATIVE.as_posix(),
        "--implementation-protected-path",
        OBJECTIVE_RELATIVE.as_posix(),
        "--no-objective-task-janitor",
        "--implement",
        "--log-level",
        "INFO",
    ]
    if enable_objective_refill:
        args.extend(
            [
                "--objective-refill-scan",
                "--objective-path",
                str(REPO_ROOT / OBJECTIVE_RELATIVE),
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
                "AICAT-G",
                "--objective-root-goal-id",
                "AICAT-G000",
                "--objective-root-goal-title",
                "Unified AI service discovery and routing",
                "--objective-tracking-document-title",
                "AI Service Catalog Objective Heap",
                "--objective-mission-term",
                "catalog,model,provider,router,mcp,mcplusplus,voice,embedding,multimodal",
            ]
        )
    return tuple(args)


def launch(args: argparse.Namespace) -> int:
    report = inspect_board(
        lanes=args.lanes,
        enable_objective_refill=args.enable_objective_refill,
        refill_open_task_threshold=args.refill_open_task_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        return 2

    namespace = args.namespace
    runtime_root = _runtime_root(namespace)
    stamp = utc_run_stamp()
    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=REPO_ROOT,
        duration_seconds=args.duration_seconds,
        heartbeat_interval_seconds=30,
        supervisor_status_stale_seconds=300,
        stop_grace_seconds=900,
        stamp=stamp,
        master_dir=runtime_root / "master",
        master_log=runtime_root / "master" / f"aicat_{stamp}.log",
        master_pid_path=runtime_root / "master" / "aicat.pid",
        label="ai-service-catalog-six-lane",
        python_executable=sys.executable,
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="ai-service-catalog",
                script_path=Path(__file__).resolve(),
                state_dir=runtime_root / "state",
                state_prefix="aicat",
            ),
        ),
        common_args=_common_args(
            runtime_root=runtime_root,
            enable_objective_refill=args.enable_objective_refill,
            refill_open_task_threshold=args.refill_open_task_threshold,
        ),
        detach=not args.foreground,
    )
    lane_args = [
        "--implementation-supervisor-lanes-per-track",
        str(args.lanes),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "ipfs_accelerate_py.ai_service_catalog.launch.v1",
                    "argv": [*runner.args(), *lane_args],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return runner.run(lane_args)


def _build_operator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or launch the AI service catalog supervisor"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "launch"):
        child = subparsers.add_parser(command)
        child.add_argument("--lanes", type=int, default=DEFAULT_LANES)
        child.add_argument("--namespace", default=DEFAULT_NAMESPACE)
        child.add_argument("--enable-objective-refill", action="store_true")
        child.add_argument(
            "--refill-open-task-threshold",
            type=int,
            default=DEFAULT_REFILL_OPEN_TASK_THRESHOLD,
        )
        if command == "launch":
            child.add_argument("--duration-seconds", default="inf")
            child.add_argument("--foreground", action="store_true")
            child.add_argument("--dry-run", action="store_true")
    return parser


def _validate_operator_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.lanes < 1:
        parser.error("--lanes must be positive")
    if args.refill_open_task_threshold < 0:
        parser.error("--refill-open-task-threshold cannot be negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.namespace):
        parser.error("--namespace must be a safe path segment")


def main(argv: Sequence[str] | None = None) -> int:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if cli_args and cli_args[0] not in {"preflight", "launch"}:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
            main as implementation_supervisor_main,
        )

        return implementation_supervisor_main(cli_args)

    parser = _build_operator_parser()
    args = parser.parse_args(cli_args)
    _validate_operator_args(parser, args)
    if args.command == "preflight":
        report = inspect_board(
            lanes=args.lanes,
            enable_objective_refill=args.enable_objective_refill,
            refill_open_task_threshold=args.refill_open_task_threshold,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
