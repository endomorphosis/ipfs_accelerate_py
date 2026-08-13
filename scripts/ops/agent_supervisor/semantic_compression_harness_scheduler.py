#!/usr/bin/env python3
"""SCH-specific supervisor adapter for the completed semantic-compression board.

The generic configured-board control plane currently cannot import on this
tree. This adapter binds the sealed SCH scheduler JSON to the existing
implementation supervisor without that import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

SCHEDULER_REL = Path("config/agent_supervisor_semantic_compression_harness_scheduler.json")
SEAL_REL = Path("config/semantic_state_dependencies.seal.json")
ENTRY_REL = Path("scripts/ops/agent_supervisor/implementation_supervisor_entry.py")
SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.semantic_compression_harness.scheduler_config@1"
)
BOARD_NAMESPACE = "semantic-compression-harness-v1"
TASK_PREFIX = "## SCH-"
SEALED_ROLES = {
    "accelerate_harness": "271e331af802f37d759c000666282631a99f7aab",
    "incremental_semantic_index": "1330038f626ef92993f03d46f21e1a57719e9c25",
    "semantic_state_contracts": "1330038f626ef92993f03d46f21e1a57719e9c25",
    "kit_state_roots": "df2f9cc092456329de9724c45a50c54b410875d1",
    "mcp_plus_plus": "dc3164653a48d059ae9812078359daeafb451c07",
}


class SCHSchedulerError(RuntimeError):
    """Raised when the SCH supervisor boundary cannot be admitted."""


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SCHSchedulerError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SCHSchedulerError(f"{path} must be a JSON object")
    return payload


def load_scheduler(repo_root: Path) -> dict[str, object]:
    payload = _load_json(repo_root / SCHEDULER_REL)
    if payload.get("schema") != SCHEDULER_SCHEMA:
        raise SCHSchedulerError(f"unsupported scheduler schema: {payload.get('schema')!r}")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        raise SCHSchedulerError("scheduler board_namespace mismatch")
    if payload.get("task_prefix") != "SCH-":
        raise SCHSchedulerError("scheduler task_prefix must be SCH-")
    return payload


def preflight(repo_root: Path) -> dict[str, object]:
    scheduler = load_scheduler(repo_root)
    seal = _load_json(repo_root / SEAL_REL)
    authorities = {
        str(item.get("role")): str(item.get("commit"))
        for item in seal.get("authorities", [])
        if isinstance(item, dict)
    }
    pin_errors: list[str] = []
    for role, commit in SEALED_ROLES.items():
        if authorities.get(role) != commit:
            pin_errors.append(f"{role} seal commit {authorities.get(role)!r} != {commit}")
    binding = scheduler.get("source_binding")
    if not isinstance(binding, dict):
        pin_errors.append("source_binding missing")
    else:
        if binding.get("ipfs_kit_planning_revision") != SEALED_ROLES["kit_state_roots"]:
            pin_errors.append("scheduler kit pin does not match SCH-000 seal")
        if binding.get("ipfs_datasets_planning_revision") != SEALED_ROLES["semantic_state_contracts"]:
            pin_errors.append("scheduler datasets pin does not match SCH-000 seal")
        if binding.get("accelerator_required_ancestor") != SEALED_ROLES["accelerate_harness"]:
            pin_errors.append("scheduler accelerate pin does not match SCH-000 seal")
    tasks = parse_task_file(
        repo_root / str(scheduler["taskboard_path"]),
        task_header_prefix=TASK_PREFIX,
    )
    if [task.task_id for task in tasks] != [f"SCH-{index:03d}" for index in range(19)]:
        pin_errors.append("taskboard is not SCH-000 through SCH-018")
    if {task.status for task in tasks} != {"completed"}:
        pin_errors.append("SCH board is not fully completed")
    if pin_errors:
        raise SCHSchedulerError("; ".join(pin_errors))
    return {
        "valid": True,
        "board_namespace": BOARD_NAMESPACE,
        "tasks": len(tasks),
        "terminal_task_id": "SCH-018",
        "seal_roles": SEALED_ROLES,
    }


def launch_plan(repo_root: Path, *, implement: bool, once: bool = True) -> dict[str, object]:
    scheduler = load_scheduler(repo_root)
    runtime = scheduler["runtime_paths"]
    if not isinstance(runtime, dict):
        raise SCHSchedulerError("runtime_paths must be an object")
    argv = [
        sys.executable,
        str(repo_root / ENTRY_REL),
        "--todo-path",
        str(PurePosixPath(str(scheduler["taskboard_path"]))),
        "--task-prefix",
        TASK_PREFIX,
        "--state-dir",
        str(PurePosixPath(str(runtime["state"]))),
        "--state-prefix",
        "sch",
        "--stale-seconds",
        str(scheduler.get("stale_seconds", 1200)),
        "--check-interval",
        str(scheduler.get("check_interval_seconds", 20)),
        "--daemon-interval",
        str(scheduler.get("daemon_interval_seconds", 45)),
        "--max-restarts",
        str(scheduler.get("max_restarts", 8)),
        "--max-task-attempts",
        str(scheduler.get("max_task_attempts", 4)),
        "--implementation-timeout",
        str(scheduler.get("implementation_timeout_seconds", 5400)),
    ]
    if once:
        argv.append("--once")
    if implement:
        argv.append("--implement")
    provider = scheduler.get("provider")
    environment = {}
    if isinstance(provider, dict):
        environment = {
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": provider.get(
                "primary_provider_id", "grok_cli"
            ),
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": provider.get(
                "fallback_provider_id", "codex"
            ),
            "IPFS_ACCELERATE_AGENT_GROK_MODEL": provider.get("primary_model_id", "grok-4.5"),
            "IPFS_ACCELERATE_AGENT_CODEX_MODEL": provider.get("fallback_model_id", "gpt-5.6-terra"),
        }
    return {
        "cwd": str(repo_root),
        "argv": argv,
        "environment": environment,
        "lanes": scheduler.get("max_lanes"),
        "implement": implement,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Admit the sealed SCH board and pin bindings")
    launch = sub.add_parser("launch", help="Render or start the implementation supervisor")
    launch.add_argument("--implement", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--once", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(repo_root), indent=2, sort_keys=True))
            return 0
        plan = launch_plan(repo_root, implement=bool(args.implement), once=bool(args.once))
        if args.dry_run or not args.implement:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        preflight(repo_root)
        import os
        import subprocess

        env = dict(os.environ)
        env.update({key: str(value) for key, value in dict(plan["environment"]).items()})
        return subprocess.call(list(plan["argv"]), cwd=str(plan["cwd"]), env=env)
    except SCHSchedulerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
