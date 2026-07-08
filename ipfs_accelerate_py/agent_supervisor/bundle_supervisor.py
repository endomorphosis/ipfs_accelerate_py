"""Plan and launch per-bundle todo daemon lanes."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .objective_graph import DEFAULT_TASK_PREFIX, build_bundle_task_payloads, repo_relative_path, safe_bundle_key, utc_now


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleLaneSpec:
    """One isolated daemon/supervisor lane for an objective bundle shard."""

    bundle_key: str
    parallel_lane: str
    todo_path: Path
    state_dir: Path
    worktree_root: Path
    state_prefix: str
    task_ids: list[str]
    conflict_policy: str
    command: list[str]
    log_path: Path
    source_todo: str = ""

    def to_dict(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("todo_path", "state_dir", "worktree_root", "log_path"):
            path = Path(payload[key])
            payload[key] = repo_relative_path(repo_root, path) if repo_root is not None else str(path)
        return payload


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def lane_state_prefix(bundle_key: str) -> str:
    return f"agent_{safe_bundle_key(bundle_key).replace('-', '_')}"


def implementation_supervisor_command(
    *,
    todo_path: Path,
    state_dir: Path,
    worktree_root: Path,
    state_prefix: str,
    task_prefix: str,
    implement: bool,
    daemon_interval: float,
    stale_seconds: float,
    check_interval: float,
    max_restarts: int,
    implementation_timeout: float,
    implementation_command: str = "",
    log_level: str = "INFO",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
        "--todo-path",
        str(todo_path),
        "--state-dir",
        str(state_dir),
        "--state-prefix",
        state_prefix,
        "--task-prefix",
        task_prefix,
        "--worktree-root",
        str(worktree_root),
        "--daemon-interval",
        str(daemon_interval),
        "--stale-seconds",
        str(stale_seconds),
        "--check-interval",
        str(check_interval),
        "--max-restarts",
        str(max_restarts),
        "--implementation-timeout",
        str(implementation_timeout),
        "--log-level",
        log_level,
    ]
    command.append("--implement" if implement else "--no-implement")
    if implementation_command:
        command.extend(["--implementation-command", implementation_command])
    return command


def plan_bundle_lanes(
    *,
    bundle_index_path: Path,
    repo_root: Path,
    state_root: Path,
    worktree_root: Path,
    log_dir: Path,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    implement: bool = False,
    daemon_interval: float = 300.0,
    stale_seconds: float = 1800.0,
    check_interval: float = 60.0,
    max_restarts: int = 10,
    implementation_timeout: float = 1800.0,
    implementation_command: str = "",
    log_level: str = "INFO",
    max_lanes: int | None = None,
) -> list[BundleLaneSpec]:
    """Return one isolated supervisor command for each objective bundle."""

    lanes: list[BundleLaneSpec] = []
    for payload in build_bundle_task_payloads(bundle_index_path):
        if max_lanes is not None and len(lanes) >= max_lanes:
            break
        bundle_key = str(payload.get("bundle_key") or "objective/general")
        safe_key = safe_bundle_key(bundle_key)
        todo_path = resolve_repo_path(repo_root, str(payload.get("todo_path") or ""))
        state_dir = state_root / safe_key / "state"
        lane_worktree_root = worktree_root / safe_key
        log_path = log_dir / f"{safe_key}.log"
        state_prefix = lane_state_prefix(bundle_key)
        task_ids = [
            str(item.get("task_id"))
            for item in payload.get("tasks", [])
            if isinstance(item, dict) and item.get("task_id")
        ]
        command = implementation_supervisor_command(
            todo_path=todo_path,
            state_dir=state_dir,
            worktree_root=lane_worktree_root,
            state_prefix=state_prefix,
            task_prefix=task_prefix,
            implement=implement,
            daemon_interval=daemon_interval,
            stale_seconds=stale_seconds,
            check_interval=check_interval,
            max_restarts=max_restarts,
            implementation_timeout=implementation_timeout,
            implementation_command=implementation_command,
            log_level=log_level,
        )
        lanes.append(
            BundleLaneSpec(
                bundle_key=bundle_key,
                parallel_lane=str(payload.get("parallel_lane") or bundle_key),
                todo_path=todo_path,
                state_dir=state_dir,
                worktree_root=lane_worktree_root,
                state_prefix=state_prefix,
                task_ids=task_ids,
                conflict_policy=str(payload.get("conflict_policy") or ""),
                command=command,
                log_path=log_path,
                source_todo=str(payload.get("source_todo") or ""),
            )
        )
    return lanes


def launch_bundle_lanes(
    lanes: Sequence[BundleLaneSpec],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Launch lane supervisors in detached process sessions."""

    results: list[dict[str, Any]] = []
    for lane in lanes:
        lane.state_dir.mkdir(parents=True, exist_ok=True)
        lane.worktree_root.mkdir(parents=True, exist_ok=True)
        lane.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lane.log_path.open("ab")
        try:
            process = subprocess.Popen(
                lane.command,
                cwd=repo_root,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            handle.close()
        pid_path = lane.state_dir / f"{lane.state_prefix}_bundle_supervisor.pid"
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        results.append(
            {
                "bundle_key": lane.bundle_key,
                "pid": process.pid,
                "pid_path": repo_relative_path(repo_root, pid_path),
                "log_path": repo_relative_path(repo_root, lane.log_path),
                "command": lane.command,
            }
        )
    return results


def check_lane_health(
    lanes: Sequence[BundleLaneSpec],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Check health of all launched lanes and restart any dead ones.

    Returns a list of health reports, one per lane. Dead lanes are
    automatically relaunched so the bundle supervisor can run indefinitely.
    """
    import os
    import signal

    reports: list[dict[str, Any]] = []
    for lane in lanes:
        pid_path = lane.state_dir / f"{lane.state_prefix}_bundle_supervisor.pid"
        report: dict[str, Any] = {"bundle_key": lane.bundle_key, "alive": False, "restarted": False}

        if not pid_path.exists():
            report["reason"] = "no_pid_file"
        else:
            try:
                pid = int(pid_path.read_text().strip())
                # Check if process is alive
                os.kill(pid, 0)
                report["alive"] = True
                report["pid"] = pid
            except (ValueError, ProcessLookupError, PermissionError):
                report["reason"] = "process_dead"
            except OSError:
                report["reason"] = "check_failed"

        if not report["alive"]:
            # Restart the dead lane
            try:
                lane.log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = lane.log_path.open("ab")
                try:
                    process = subprocess.Popen(
                        lane.command,
                        cwd=repo_root,
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    handle.close()
                pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
                report["restarted"] = True
                report["new_pid"] = process.pid
                logger.info("Restarted dead lane %s with PID %d", lane.bundle_key, process.pid)
            except OSError as exc:
                report["restart_error"] = str(exc)
                logger.error("Failed to restart lane %s: %s", lane.bundle_key, exc)

        reports.append(report)
    return reports


def write_bundle_lane_manifest(
    *,
    manifest_path: Path,
    repo_root: Path,
    bundle_index_path: Path,
    lanes: Sequence[BundleLaneSpec],
    started: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.bundle_supervisor",
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "bundle_index_path": repo_relative_path(repo_root, bundle_index_path),
        "planned_count": len(lanes),
        "started_count": len(started),
        "lanes": [lane.to_dict(repo_root=repo_root) for lane in lanes],
        "started": list(started),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def default_state_root(repo_root: Path) -> Path:
    return repo_root / "data" / "agent_supervisor" / "bundle_lanes"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or launch isolated daemon lanes for objective bundle shards")
    parser.add_argument("--bundle-index-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--worktree-root", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--start", action="store_true", help="Launch the planned lane supervisors")
    parser.add_argument("--max-lanes", type=int, default=None)
    implement_group = parser.add_mutually_exclusive_group()
    implement_group.add_argument("--implement", dest="implement", action="store_true")
    implement_group.add_argument("--no-implement", dest="implement", action="store_false")
    parser.set_defaults(implement=False)
    parser.add_argument("--daemon-interval", type=float, default=300.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--check-interval", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--implementation-timeout", type=float, default=1800.0)
    parser.add_argument("--implementation-command", default="")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run_bundle_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    state_root = (args.state_root or default_state_root(repo_root)).resolve()
    worktree_root = (args.worktree_root or state_root / "worktrees").resolve()
    log_dir = (args.log_dir or state_root / "logs").resolve()
    manifest_path = (args.manifest_path or state_root / "bundle_lanes.json").resolve()
    bundle_index_path = args.bundle_index_path.resolve()
    lanes = plan_bundle_lanes(
        bundle_index_path=bundle_index_path,
        repo_root=repo_root,
        state_root=state_root,
        worktree_root=worktree_root,
        log_dir=log_dir,
        task_prefix=args.task_prefix,
        implement=args.implement,
        daemon_interval=args.daemon_interval,
        stale_seconds=args.stale_seconds,
        check_interval=args.check_interval,
        max_restarts=args.max_restarts,
        implementation_timeout=args.implementation_timeout,
        implementation_command=args.implementation_command,
        log_level=args.log_level,
        max_lanes=args.max_lanes,
    )
    started = launch_bundle_lanes(lanes, repo_root=repo_root) if args.start else []
    payload = write_bundle_lane_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root,
        bundle_index_path=bundle_index_path,
        lanes=lanes,
        started=started,
    )
    logger.info("Planned %s bundle lanes; started %s", len(lanes), len(started))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run_bundle_supervisor(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
