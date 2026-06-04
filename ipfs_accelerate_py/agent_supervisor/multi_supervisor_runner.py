"""Reusable timed runner for multiple implementation supervisor scripts."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .todo_daemon.core import pid_alive, read_pid_file, terminate_pid_tree


OutputFn = Callable[[str], None]


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return int(default)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


@dataclass(frozen=True)
class SupervisorTrack:
    """One supervisor process managed by the multi-supervisor runner."""

    name: str
    script_path: Path
    log_path: Path
    supervisor_pid_path: Path
    daemon_pid_path: Path

    def resolve(self, repo_root: Path) -> "SupervisorTrack":
        return SupervisorTrack(
            name=self.name,
            script_path=_resolve_path(repo_root, self.script_path),
            log_path=_resolve_path(repo_root, self.log_path),
            supervisor_pid_path=_resolve_path(repo_root, self.supervisor_pid_path),
            daemon_pid_path=_resolve_path(repo_root, self.daemon_pid_path),
        )


class SupervisorRunInterrupted(Exception):
    """Raised internally when a signal requests orderly shutdown."""


def utc_run_stamp() -> str:
    """Return a UTC run stamp suitable for log/pid filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_timestamp() -> str:
    """Return a compact local timestamp for operator logs."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_track_spec(spec: str, *, stamp: str = "") -> SupervisorTrack:
    """Parse ``NAME|SCRIPT|LOG|SUPERVISOR_PID|DAEMON_PID`` track specs."""

    rendered = spec.format(stamp=stamp) if stamp else spec
    parts = rendered.split("|")
    if len(parts) != 5 or not parts[0].strip():
        raise ValueError(
            "track specs must have NAME|SCRIPT|LOG|SUPERVISOR_PID|DAEMON_PID"
        )
    name, script, log, supervisor_pid, daemon_pid = (part.strip() for part in parts)
    return SupervisorTrack(
        name=name,
        script_path=Path(script),
        log_path=Path(log),
        supervisor_pid_path=Path(supervisor_pid),
        daemon_pid_path=Path(daemon_pid),
    )


def implementation_supervisor_track_spec(
    *,
    name: str,
    script_path: Path | str,
    state_dir: Path | str,
    state_prefix: str,
) -> str:
    """Return a standard implementation-supervisor track spec."""

    state_path = Path(state_dir).as_posix()
    return "|".join(
        (
            str(name),
            Path(script_path).as_posix(),
            f"{state_path}/{state_prefix}_8h_run_{{stamp}}.log",
            f"{state_path}/{state_prefix}_supervisor.pid",
            f"{state_path}/{state_prefix}_managed_daemon.pid",
        )
    )


def parse_implementation_track_spec(spec: str, *, stamp: str = "") -> SupervisorTrack:
    """Parse ``NAME|SCRIPT|STATE_DIR|STATE_PREFIX`` implementation-track specs."""

    parts = [part.strip() for part in spec.split("|")]
    if len(parts) != 4 or not parts[0]:
        raise ValueError("implementation track specs must have NAME|SCRIPT|STATE_DIR|STATE_PREFIX")
    name, script, state_dir, state_prefix = parts
    return parse_track_spec(
        implementation_supervisor_track_spec(
            name=name,
            script_path=script,
            state_dir=state_dir,
            state_prefix=state_prefix,
        ),
        stamp=stamp,
    )


def supervisor_track_payload(track: SupervisorTrack) -> dict[str, str]:
    """Return a serializable track description for tests and diagnostics."""

    return {
        "name": track.name,
        "script_path": str(track.script_path),
        "log_path": str(track.log_path),
        "supervisor_pid_path": str(track.supervisor_pid_path),
        "daemon_pid_path": str(track.daemon_pid_path),
    }


def implementation_supervisor_common_args(
    *,
    implementation_command: str = "",
    llm_merge_resolver_command: str = "",
    stale_seconds: int = 1800,
    check_interval: int = 60,
    daemon_interval: int = 120,
    implementation_timeout: int = 1800,
    implementation_log_stall_seconds: int = 900,
    max_restarts: int = 0,
    objective_scan_min_open_tasks: int = 20,
    objective_scan_max_findings: int = 12,
    objective_scan_cooldown_seconds: int = 900,
    objective_surplus_findings_per_goal: int = 6,
    objective_surplus_min_terms_per_todo: int = 4,
    codebase_scan_cooldown_seconds: int = 900,
    llm_merge_resolver_timeout_seconds: int = 1800,
) -> list[str]:
    """Return standard common args for long-running implementation supervisors."""

    effective_llm_merge_resolver_command = llm_merge_resolver_command or implementation_command
    args = [
        "--implement",
        "--stale-seconds",
        str(stale_seconds),
        "--check-interval",
        str(check_interval),
        "--daemon-interval",
        str(daemon_interval),
        "--implementation-timeout",
        str(implementation_timeout),
        "--implementation-log-stall-seconds",
        str(implementation_log_stall_seconds),
        "--max-restarts",
        str(max_restarts),
        "--objective-scan-min-open-tasks",
        str(objective_scan_min_open_tasks),
        "--objective-scan-max-findings",
        str(objective_scan_max_findings),
        "--objective-scan-cooldown-seconds",
        str(objective_scan_cooldown_seconds),
        "--objective-surplus-findings-per-goal",
        str(objective_surplus_findings_per_goal),
        "--objective-surplus-min-terms-per-todo",
        str(objective_surplus_min_terms_per_todo),
        "--codebase-scan-cooldown-seconds",
        str(codebase_scan_cooldown_seconds),
        "--llm-merge-resolver-timeout-seconds",
        str(llm_merge_resolver_timeout_seconds),
    ]
    if implementation_command:
        args.extend(["--implementation-command", implementation_command])
    if effective_llm_merge_resolver_command:
        args.extend(["--llm-merge-resolver-command", effective_llm_merge_resolver_command])
    return args


def _emit(output: OutputFn, message: str) -> None:
    output(f"{iso_timestamp()} {message}")


def _default_output(message: str) -> None:
    print(message, flush=True)


def start_track(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    common_args: Sequence[str],
    python_executable: str = "python3",
    output: OutputFn = _default_output,
) -> subprocess.Popen[bytes]:
    """Start one supervisor track and write its supervisor PID marker."""

    resolved = track.resolve(repo_root)
    resolved.log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.supervisor_pid_path.parent.mkdir(parents=True, exist_ok=True)
    command = [python_executable, str(resolved.script_path), *common_args]
    out_handle = resolved.log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        out_handle.close()
    resolved.supervisor_pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    _emit(
        output,
        f"started {resolved.name} supervisor pid={process.pid} script={resolved.script_path} log={resolved.log_path}",
    )
    return process


def _terminate_pid(pid: int | None, *, grace_seconds: float) -> bool:
    if not pid:
        return False
    return terminate_pid_tree(int(pid), grace_seconds=grace_seconds)


def stop_tracks(
    tracks: Sequence[SupervisorTrack],
    processes: dict[str, subprocess.Popen[bytes]],
    *,
    repo_root: Path,
    grace_seconds: float = 10.0,
    output: OutputFn = _default_output,
) -> dict[str, object]:
    """Stop supervisor wrapper processes and their managed daemons."""

    stopped: list[int] = []
    _emit(output, "stopping supervisor wrapper and managed daemons")
    for track in tracks:
        resolved = track.resolve(repo_root)
        process = processes.get(track.name)
        candidate_pids = [
            process.pid if process is not None else None,
            read_pid_file(resolved.supervisor_pid_path),
            read_pid_file(resolved.daemon_pid_path),
        ]
        for pid in candidate_pids:
            if pid and _terminate_pid(pid, grace_seconds=grace_seconds):
                stopped.append(int(pid))
        if process is not None:
            try:
                process.wait(timeout=max(0.1, grace_seconds))
            except subprocess.TimeoutExpired:
                pass
    return {"stopped_pids": stopped, "stopped_count": len(stopped)}


def run_supervisor_tracks(
    tracks: Sequence[SupervisorTrack],
    *,
    repo_root: Path,
    common_args: Sequence[str],
    duration_seconds: float,
    heartbeat_interval_seconds: float = 60.0,
    stop_grace_seconds: float = 10.0,
    python_executable: str = "python3",
    master_pid_path: Path | None = None,
    label: str = "multi-supervisor",
    output: OutputFn = _default_output,
) -> dict[str, object]:
    """Run and supervise multiple tracks for the requested duration."""

    resolved_repo_root = repo_root.resolve()
    if master_pid_path is not None:
        resolved_master_pid = _resolve_path(resolved_repo_root, master_pid_path)
        resolved_master_pid.parent.mkdir(parents=True, exist_ok=True)
        resolved_master_pid.write_text(f"{os.getpid()}\n", encoding="utf-8")
    processes: dict[str, subprocess.Popen[bytes]] = {}

    def _handle_signal(signum: int, _frame: object) -> None:
        raise SupervisorRunInterrupted(f"received signal {signum}")

    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    interrupted = ""
    try:
        _emit(output, f"starting {label} duration_seconds={duration_seconds:g}")
        for track in tracks:
            processes[track.name] = start_track(
                track,
                repo_root=resolved_repo_root,
                common_args=common_args,
                python_executable=python_executable,
                output=output,
            )

        deadline = time.monotonic() + max(0.0, float(duration_seconds))
        while time.monotonic() < deadline:
            sleep_for = min(
                max(0.05, heartbeat_interval_seconds),
                max(0.0, deadline - time.monotonic()),
            )
            time.sleep(sleep_for)
            for track in tracks:
                process = processes.get(track.name)
                resolved = track.resolve(resolved_repo_root)
                daemon_pid = read_pid_file(resolved.daemon_pid_path)
                if process is not None and process.poll() is None and pid_alive(process.pid):
                    _emit(
                        output,
                        f"heartbeat {track.name} supervisor_pid={process.pid} daemon_pid={daemon_pid or 'unknown'}",
                    )
                    continue
                old_pid = None if process is None else process.pid
                _emit(output, f"restarting exited {track.name} supervisor old_pid={old_pid or 'none'}")
                processes[track.name] = start_track(
                    track,
                    repo_root=resolved_repo_root,
                    common_args=common_args,
                    python_executable=python_executable,
                    output=output,
                )
        _emit(output, "completed requested run window")
    except SupervisorRunInterrupted as exc:
        interrupted = str(exc)
        _emit(output, f"interrupted: {interrupted}")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        stop_payload = stop_tracks(
            tracks,
            processes,
            repo_root=resolved_repo_root,
            grace_seconds=stop_grace_seconds,
            output=output,
        )
    return {
        "completed": not interrupted,
        "interrupted": interrupted,
        "track_count": len(tracks),
        "stopped_count": stop_payload["stopped_count"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple implementation supervisors for a fixed window")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--duration-seconds", type=float, default=28800.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=60.0)
    parser.add_argument("--stop-grace-seconds", type=float, default=10.0)
    parser.add_argument("--stamp", default=utc_run_stamp())
    parser.add_argument("--master-dir", type=Path, default=Path("data/agent_supervisor"))
    parser.add_argument("--master-log", type=Path, default=None)
    parser.add_argument("--master-pid-path", type=Path, default=None)
    parser.add_argument("--label", default="multi-supervisor")
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument(
        "--implementation-track",
        action="append",
        default=[],
        help="Compact NAME|SCRIPT|STATE_DIR|STATE_PREFIX implementation-supervisor track.",
    )
    parser.add_argument("--common-arg", action="append", default=[])
    parser.add_argument(
        "--implementation-supervisor-defaults",
        action="store_true",
        help="Prepend standard long-running implementation-supervisor args before --common-arg values.",
    )
    parser.add_argument("--implementation-supervisor-command", default="")
    parser.add_argument("--implementation-supervisor-stale-seconds", type=int, default=1800)
    parser.add_argument("--implementation-supervisor-check-interval", type=int, default=60)
    parser.add_argument("--implementation-supervisor-daemon-interval", type=int, default=120)
    parser.add_argument("--implementation-supervisor-timeout", type=int, default=1800)
    parser.add_argument("--implementation-supervisor-log-stall-seconds", type=int, default=900)
    parser.add_argument("--implementation-supervisor-max-restarts", type=int, default=0)
    parser.add_argument(
        "--implementation-supervisor-objective-scan-min-open-tasks",
        type=int,
        default=_env_int("OBJECTIVE_SCAN_MIN_OPEN_TASKS", 20),
    )
    parser.add_argument(
        "--implementation-supervisor-objective-scan-max-findings",
        type=int,
        default=_env_int("OBJECTIVE_SCAN_MAX_FINDINGS", 12),
    )
    parser.add_argument("--implementation-supervisor-objective-scan-cooldown-seconds", type=int, default=900)
    parser.add_argument(
        "--implementation-supervisor-objective-surplus-findings-per-goal",
        type=int,
        default=_env_int("OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", 6),
    )
    parser.add_argument(
        "--implementation-supervisor-objective-surplus-min-terms-per-todo",
        type=int,
        default=_env_int("OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", 4),
    )
    parser.add_argument("--implementation-supervisor-codebase-scan-cooldown-seconds", type=int, default=900)
    parser.add_argument("--implementation-supervisor-llm-merge-resolver-command", default="")
    parser.add_argument("--implementation-supervisor-llm-merge-resolver-timeout-seconds", type=int, default=1800)
    parser.add_argument("--detach", action="store_true")
    return parser


def _master_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = args.repo_root.resolve()
    master_dir = _resolve_path(repo_root, args.master_dir)
    master_log = _resolve_path(repo_root, args.master_log) if args.master_log else master_dir / f"8h_run_{args.stamp}.log"
    master_pid = (
        _resolve_path(repo_root, args.master_pid_path)
        if args.master_pid_path
        else master_dir / f"8h_run_{args.stamp}.pid"
    )
    return master_log, master_pid


def _without_detach(argv: Sequence[str]) -> list[str]:
    removed = False
    cleaned: list[str] = []
    for item in argv:
        if item == "--detach" and not removed:
            removed = True
            continue
        cleaned.append(item)
    return cleaned


def launch_detached(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, object]:
    """Launch this runner detached, redirecting output to the master log."""

    master_log, master_pid = _master_paths(args)
    master_log.parent.mkdir(parents=True, exist_ok=True)
    master_pid.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.multi_supervisor_runner",
        *_without_detach(argv),
    ]
    out_handle = master_log.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            stdin=subprocess.DEVNULL,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        out_handle.close()
    master_pid.write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "stamp": args.stamp,
        "master_pid": process.pid,
        "master_log": str(master_log),
        "master_pid_file": str(master_pid),
    }


def common_args_from_parsed_args(args: argparse.Namespace) -> list[str]:
    """Return the effective common supervisor args for parsed runner options."""

    common_args: list[str] = []
    if args.implementation_supervisor_defaults:
        command = args.implementation_supervisor_command
        common_args.extend(
            implementation_supervisor_common_args(
                implementation_command=command,
                llm_merge_resolver_command=args.implementation_supervisor_llm_merge_resolver_command or command,
                stale_seconds=args.implementation_supervisor_stale_seconds,
                check_interval=args.implementation_supervisor_check_interval,
                daemon_interval=args.implementation_supervisor_daemon_interval,
                implementation_timeout=args.implementation_supervisor_timeout,
                implementation_log_stall_seconds=args.implementation_supervisor_log_stall_seconds,
                max_restarts=args.implementation_supervisor_max_restarts,
                objective_scan_min_open_tasks=args.implementation_supervisor_objective_scan_min_open_tasks,
                objective_scan_max_findings=args.implementation_supervisor_objective_scan_max_findings,
                objective_scan_cooldown_seconds=args.implementation_supervisor_objective_scan_cooldown_seconds,
                objective_surplus_findings_per_goal=args.implementation_supervisor_objective_surplus_findings_per_goal,
                objective_surplus_min_terms_per_todo=args.implementation_supervisor_objective_surplus_min_terms_per_todo,
                codebase_scan_cooldown_seconds=args.implementation_supervisor_codebase_scan_cooldown_seconds,
                llm_merge_resolver_timeout_seconds=args.implementation_supervisor_llm_merge_resolver_timeout_seconds,
            )
        )
    common_args.extend(args.common_arg)
    return common_args


def tracks_from_parsed_args(args: argparse.Namespace) -> list[SupervisorTrack]:
    """Return supervisor tracks from raw and compact parsed track specs."""

    tracks = [parse_track_spec(track, stamp=args.stamp) for track in args.track]
    tracks.extend(
        parse_implementation_track_spec(track, stamp=args.stamp)
        for track in args.implementation_track
    )
    return tracks


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(args_list)
    if not args.track and not args.implementation_track:
        parser.error("at least one --track or --implementation-track is required")
    if args.detach:
        payload = launch_detached(args, args_list)
        for key in ("stamp", "master_pid", "master_log", "master_pid_file"):
            print(f"{key}={payload[key]}")
        return 0

    master_log, master_pid = _master_paths(args)
    tracks = tracks_from_parsed_args(args)
    master_log.parent.mkdir(parents=True, exist_ok=True)
    with master_log.open("ab") as log_handle:
        def output(message: str) -> None:
            print(message, flush=True)
            log_handle.write((message + "\n").encode("utf-8"))
            log_handle.flush()

        run_supervisor_tracks(
            tracks,
            repo_root=args.repo_root,
            common_args=common_args_from_parsed_args(args),
            duration_seconds=args.duration_seconds,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            stop_grace_seconds=args.stop_grace_seconds,
            python_executable=args.python_executable,
            master_pid_path=master_pid,
            label=args.label,
            output=output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
