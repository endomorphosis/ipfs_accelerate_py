"""Reusable child-process runtime helpers for todo-daemon supervisors."""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ..event_log import unique_backup_path
from ..wrapper_utils import with_exclusive_flag_default
from .core import now_iso, parse_timestamp, pid_alive, process_args, read_json, read_pid_file, remove_runtime_marker, terminate_pid_tree, write_json


@dataclass(frozen=True)
class RestartPolicy:
    """Restart delays for a supervised daemon child."""

    restart_backoff_seconds: float = 30.0
    fast_restart_backoff_seconds: float = 2.0
    fast_restart_statuses: frozenset[str] = frozenset(
        {
            "dirty_recovery_skipped_clean",
            "repeated_rejection_recovery_skipped_clean",
            "no_change",
        }
    )

    def delay_for_status(self, status: str) -> float:
        if status in self.fast_restart_statuses:
            return max(0.0, float(self.fast_restart_backoff_seconds))
        return max(0.0, float(self.restart_backoff_seconds))


@dataclass(frozen=True)
class SupervisedChildSpec:
    """Configuration for one supervisor-owned child process."""

    repo_root: Path
    command: tuple[str, ...]
    log_path: Path
    child_pid_path: Path
    latest_log_path: Optional[Path] = None
    env: Mapping[str, str] = field(default_factory=dict)
    stdin_devnull: bool = True
    start_new_session: bool = True

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path


@dataclass(frozen=True)
class SupervisedChild:
    """A launched supervisor child process and its resolved artifacts."""

    pid: int
    command: tuple[str, ...]
    log_path: Path
    child_pid_path: Path
    latest_log_path: Optional[Path] = None
    started_at: str = ""


@dataclass(frozen=True)
class ProcessTerminationResult:
    """Result from terminating a supervisor-owned child process group."""

    pid: int
    initial_exit_code: Optional[int]
    final_exit_code: Optional[int]
    terminate_sent: bool = False
    kill_sent: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class ChildSummaryHealthSpec:
    """Field mapping for summarizing supervisor child status files."""

    active_ids_field: str = "active_packet_claimed_todo_ids"
    active_phase_field: str = "active_packet_phase"
    active_phases: frozenset[str] = frozenset(
        {"claimed_program_synthesis_todos", "executing_codex_packet"}
    )
    latest_reason_field: str = "latest_stop_reason"
    numeric_total_fields: tuple[str, ...] = ()
    scope_field: str = "scope"
    timestamp_fields: tuple[str, ...] = (
        "heartbeat_at",
        "updated_at",
        "active_packet_last_heartbeat_at",
        "finished_at",
        "created_at",
        "started_at",
    )
    waiting_reasons: frozenset[str] = frozenset({"waiting_for_todos"})
    worker_id_field: str = "worker_id"


@dataclass
class StopSignalState:
    """Mutable stop request state installed by reusable signal handlers."""

    previous_signal_handlers: dict[int, Any] = field(default_factory=dict)
    received_at: Optional[str] = None
    signal_count: int = 0
    stop_requested: bool = False
    stop_signal: Optional[int] = None

    def mark_requested(self, signum: int) -> None:
        self.stop_requested = True
        self.stop_signal = int(signum)
        self.signal_count += 1
        self.received_at = now_iso()

    def restore(self) -> None:
        """Restore signal handlers captured during installation."""

        for signum, handler in self.previous_signal_handlers.items():
            signal.signal(signum, handler)


DEFAULT_SUPERVISOR_RUNNING_STATES = frozenset({"running", "starting", "recycling", "restarting"})


def launch_process_child(
    command: Sequence[str],
    *,
    cwd: Path | str,
    env: Optional[Mapping[str, object]] = None,
    stdin: Any = subprocess.DEVNULL,
    stdout: Any = None,
    stderr: Any = None,
    start_new_session: bool = True,
    text: bool = False,
) -> subprocess.Popen[Any]:
    """Launch a supervisor-owned child process with normalized runtime defaults."""

    child_env = dict(os.environ)
    if env:
        child_env.update({str(key): str(value) for key, value in env.items()})
    kwargs = {
        "cwd": cwd,
        "env": child_env,
        "stdin": stdin,
        "stdout": stdout,
        "stderr": stderr,
        "start_new_session": start_new_session,
    }
    if text:
        kwargs["text"] = True
    return subprocess.Popen([str(part) for part in command], **kwargs)


class SupervisorRuntimeEnsureCallback(Protocol):
    """Callable signature for launching a project-bound supervisor wrapper."""

    def __call__(self, argv: Sequence[str], *, state_dir: Path, state_prefix: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SupervisorRuntimeOperations:
    """Project-bound runtime operations for a reusable supervisor wrapper."""

    repair_runtime: Callable[[Path, str], dict[str, Any]]
    is_running: Callable[[Path, str], bool]
    ensure_running: SupervisorRuntimeEnsureCallback


def pop_bool_flag(argv: list[str], flag: str) -> bool:
    """Remove a boolean flag from argv in place and return whether it was present."""

    found = False
    kept: list[str] = []
    for item in argv:
        if item == flag:
            found = True
            continue
        kept.append(item)
    argv[:] = kept
    return found


def supervisor_runtime_paths(
    state_dir: Path,
    state_prefix: str,
    *,
    implementation_lock_name: str = "implementation.lock",
) -> dict[str, Path]:
    """Return the conventional runtime marker paths for an implementation supervisor."""

    return {
        "supervisor_status": state_dir / f"{state_prefix}_supervisor_status.json",
        "managed_daemon_pid": state_dir / f"{state_prefix}_managed_daemon.pid",
        "wrapper_pid": state_dir / f"{state_prefix}_supervisor_wrapper.pid",
        "wrapper_out": state_dir / f"{state_prefix}_supervisor_wrapper.out",
        "implementation_lock": state_dir / implementation_lock_name,
    }


def runtime_lock_owner_is_alive(path: Path) -> bool:
    """Return whether an implementation lock still belongs to a live owner process."""

    metadata = read_json(path)
    try:
        pid = int(metadata.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if not pid_alive(pid):
        return False
    owner_script = str(metadata.get("owner_script") or "")
    if owner_script and owner_script not in process_args(pid):
        return False
    return True


def repair_supervisor_runtime(
    state_dir: Path,
    state_prefix: str,
    *,
    running_states: frozenset[str] = DEFAULT_SUPERVISOR_RUNNING_STATES,
    implementation_lock_name: str = "implementation.lock",
) -> dict[str, Any]:
    """Clear stale supervisor pid files, daemon pid files, locks, and running status."""

    paths = supervisor_runtime_paths(
        state_dir,
        state_prefix,
        implementation_lock_name=implementation_lock_name,
    )
    repairs: dict[str, Any] = {"removed": [], "updated_status": False}
    for key in ("managed_daemon_pid", "wrapper_pid"):
        path = paths[key]
        pid = read_pid_file(path)
        if not path.exists():
            continue
        if pid and pid_alive(pid):
            continue
        if remove_runtime_marker(path):
            repairs["removed"].append(str(path))

    lock_path = paths["implementation_lock"]
    if lock_path.exists() and not runtime_lock_owner_is_alive(lock_path):
        if remove_runtime_marker(lock_path):
            repairs["removed"].append(str(lock_path))

    status_path = paths["supervisor_status"]
    status = read_json(status_path)
    try:
        supervisor_pid = int(status.get("supervisor_pid") or 0)
    except (TypeError, ValueError):
        supervisor_pid = 0
    try:
        daemon_pid = int(status.get("daemon_pid") or 0)
    except (TypeError, ValueError):
        daemon_pid = 0
    status_value = str(status.get("status") or "")
    supervisor_alive = pid_alive(supervisor_pid)
    daemon_alive = pid_alive(daemon_pid)
    if status and status_value in running_states and not supervisor_alive:
        status.update(
            {
                "status": "stale",
                "repaired_at": now_iso(),
                "repair_reason": "supervisor_pid_not_running",
                "supervisor_pid_alive": False,
                "daemon_pid_alive": daemon_alive,
            }
        )
        write_json(status_path, status)
        repairs["updated_status"] = True
    return repairs


def supervisor_pid_matches(
    pid: int,
    *,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
) -> bool:
    """Return whether a live pid looks like the expected supervisor process."""

    if not pid_alive(pid):
        return False
    if process_predicate is not None and process_predicate(pid):
        return True
    if not process_match_any:
        return True
    command_line = process_args(pid)
    return any(marker and marker in command_line for marker in process_match_any)


def supervisor_is_running(
    state_dir: Path,
    state_prefix: str,
    *,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    implementation_lock_name: str = "implementation.lock",
) -> bool:
    """Return whether the conventional wrapper/status markers point to a live supervisor."""

    paths = supervisor_runtime_paths(
        state_dir,
        state_prefix,
        implementation_lock_name=implementation_lock_name,
    )
    supervisor_status = read_json(paths["supervisor_status"])
    candidates = [
        read_pid_file(paths["wrapper_pid"]),
        supervisor_status.get("supervisor_pid"),
    ]
    for candidate in candidates:
        try:
            pid = int(candidate or 0)
        except (TypeError, ValueError):
            continue
        if supervisor_pid_matches(pid, process_match_any=process_match_any, process_predicate=process_predicate):
            return True
    return False


def background_supervisor_args(
    argv: Sequence[str],
    *,
    once_flag: str = "--once",
    implement_flag: str = "--implement",
    no_implement_flag: str = "--no-implement",
) -> list[str]:
    """Return argv suitable for background execution of a supervisor."""

    args = [item for item in argv if item != once_flag]
    return implementation_supervisor_args(
        args,
        implement_flag=implement_flag,
        no_implement_flag=no_implement_flag,
    )


def implementation_supervisor_args(
    argv: Sequence[str],
    *,
    implement_flag: str = "--implement",
    no_implement_flag: str = "--no-implement",
) -> list[str]:
    """Return supervisor argv with implementation mode enabled unless explicitly disabled."""

    return with_exclusive_flag_default(argv, implement_flag, (no_implement_flag,))


@dataclass(frozen=True)
class ConfiguredSupervisorEntrypoint:
    """Project-bound entrypoint that applies supervisor argv defaults before dispatch."""

    supervisor_main: Callable[[list[str]], Any]
    default_args: Callable[[Sequence[str]], list[str]] = implementation_supervisor_args

    def with_defaults(self, argv: Sequence[str]) -> list[str]:
        """Return argv with the configured supervisor defaults applied."""

        return self.default_args(list(argv))

    def run(self, argv: Sequence[str] | None = None) -> Any:
        """Run the bound supervisor main after applying defaults."""

        return self.supervisor_main(self.with_defaults(sys.argv[1:] if argv is None else argv))


def build_configured_implementation_supervisor_entrypoint(
    supervisor_main: Callable[[list[str]], Any],
    *,
    default_args: Callable[[Sequence[str]], list[str]] = implementation_supervisor_args,
) -> ConfiguredSupervisorEntrypoint:
    """Build reusable implementation-supervisor entrypoint wiring."""

    return ConfiguredSupervisorEntrypoint(
        supervisor_main=supervisor_main,
        default_args=default_args,
    )


def build_module_implementation_supervisor_entrypoint(
    module_name: str,
    *,
    main_name: str = "main",
    default_args: Callable[[Sequence[str]], list[str]] = implementation_supervisor_args,
) -> ConfiguredSupervisorEntrypoint:
    """Build an implementation-supervisor entrypoint from an importable module main."""

    def supervisor_main(argv: list[str]) -> Any:
        module = importlib.import_module(module_name)
        main = getattr(module, main_name)
        if not callable(main):
            raise TypeError(f"{module_name}.{main_name} is not callable")
        return main(argv)

    return build_configured_implementation_supervisor_entrypoint(
        supervisor_main,
        default_args=default_args,
    )


def ensure_supervisor_running(
    argv: Sequence[str],
    *,
    state_dir: Path,
    state_prefix: str,
    repo_root: Path,
    script_path: Path,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Repair stale markers and launch a background supervisor when none is live."""

    repairs = repair_supervisor_runtime(
        state_dir,
        state_prefix,
        implementation_lock_name=implementation_lock_name,
    )
    if supervisor_is_running(
        state_dir,
        state_prefix,
        process_match_any=process_match_any,
        process_predicate=process_predicate,
        implementation_lock_name=implementation_lock_name,
    ):
        return {"started": False, "reason": "already_running", "repairs": repairs}

    paths = supervisor_runtime_paths(
        state_dir,
        state_prefix,
        implementation_lock_name=implementation_lock_name,
    )
    launch_args = background_supervisor_args(argv)
    command = [sys.executable, str(script_path), *launch_args]
    if prepare_environment is not None:
        prepare_environment()
    paths["wrapper_out"].parent.mkdir(parents=True, exist_ok=True)
    out_handle = paths["wrapper_out"].open("ab")
    try:
        process = launch_process_child(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        out_handle.close()
    paths["wrapper_pid"].write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(max(0.0, float(startup_delay_seconds)))
    return {
        "started": pid_alive(process.pid),
        "pid": process.pid,
        "command": command,
        "wrapper_out": str(paths["wrapper_out"]),
        "repairs": repairs,
    }


def build_supervisor_runtime_operations(
    *,
    repo_root: Path,
    script_path: Path,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
) -> SupervisorRuntimeOperations:
    """Bind generic supervisor runtime helpers to a project wrapper."""

    def repair_runtime(state_dir: Path, state_prefix: str) -> dict[str, Any]:
        return repair_supervisor_runtime(
            state_dir,
            state_prefix,
            implementation_lock_name=implementation_lock_name,
        )

    def is_running(state_dir: Path, state_prefix: str) -> bool:
        return supervisor_is_running(
            state_dir,
            state_prefix,
            process_match_any=process_match_any,
            process_predicate=process_predicate,
            implementation_lock_name=implementation_lock_name,
        )

    def ensure_running(argv: Sequence[str], *, state_dir: Path, state_prefix: str) -> dict[str, Any]:
        return ensure_supervisor_running(
            argv,
            state_dir=state_dir,
            state_prefix=state_prefix,
            repo_root=repo_root,
            script_path=script_path,
            process_match_any=process_match_any,
            process_predicate=process_predicate,
            prepare_environment=prepare_environment,
            implementation_lock_name=implementation_lock_name,
            startup_delay_seconds=startup_delay_seconds,
        )

    return SupervisorRuntimeOperations(
        repair_runtime=repair_runtime,
        is_running=is_running,
        ensure_running=ensure_running,
    )


def supervisor_run_id(now: Optional[datetime] = None) -> str:
    """Return the stable UTC run id format used by unattended supervisors."""

    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def supervised_log_path(
    daemon_dir: Path,
    *,
    prefix: str,
    run_id: str,
    suffix: str = ".log",
) -> Path:
    """Return a supervisor child log path under ``daemon_dir``."""

    return daemon_dir / f"{prefix}_{run_id}{suffix}"


def build_python_module_command(
    module: str,
    args: Sequence[str] = (),
    *,
    python_executable: str = "python3",
    unbuffered: bool = True,
) -> tuple[str, ...]:
    """Build a ``python -m`` command tuple for a reusable daemon module."""

    command = [python_executable]
    if unbuffered:
        command.append("-u")
    command.extend(["-m", module])
    command.extend(str(arg) for arg in args)
    return tuple(command)


def child_exit_should_restart(
    *,
    exit_code: Optional[int],
    restart_count: int,
    restart_limit: int,
    stop_requested: bool = False,
    restart_on_clean_exit: bool = False,
) -> bool:
    """Return whether a supervised child should be replaced after exit."""

    if exit_code is None or stop_requested:
        return False
    try:
        count = int(restart_count)
    except (TypeError, ValueError):
        count = 0
    try:
        limit = int(restart_limit)
    except (TypeError, ValueError):
        limit = 0
    if count >= limit:
        return False
    if int(exit_code) == 0 and not restart_on_clean_exit:
        return False
    return True


def install_stop_signal_handlers(
    signals: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
    *,
    on_signal: Optional[Callable[[int, Any], None]] = None,
) -> StopSignalState:
    """Install reusable stop-request signal handlers and return mutable state."""

    state = StopSignalState()

    def request_stop(signum: int, frame: Any) -> None:
        state.mark_requested(signum)
        if on_signal is not None:
            on_signal(signum, frame)

    for signum in signals:
        signum_int = int(signum)
        state.previous_signal_handlers[signum_int] = signal.getsignal(signum_int)
        signal.signal(signum_int, request_stop)
    return state


def supervised_child_succeeded(
    *,
    child_id: str,
    exit_code: Optional[int],
    runner_terminated_child_ids: Sequence[str] = (),
    stop_requested: bool = False,
    allow_runner_terminated: bool = False,
    runner_terminated_success_codes: frozenset[int] = frozenset(
        {-signal.SIGTERM, -signal.SIGKILL}
    ),
) -> bool:
    """Return whether one supervised child should count as successful."""

    terminated_ids = {str(item) for item in runner_terminated_child_ids}
    if exit_code == 0:
        return allow_runner_terminated or str(child_id) not in terminated_ids
    if not allow_runner_terminated or stop_requested:
        return False
    return bool(
        str(child_id) in terminated_ids
        and exit_code in runner_terminated_success_codes
    )


def supervised_child_group_succeeded(
    exit_codes: Mapping[str, Optional[int]],
    *,
    runner_terminated_child_ids: Sequence[str] = (),
    stop_requested: bool = False,
    allow_runner_terminated: bool = False,
    require_children: bool = True,
) -> bool:
    """Return whether all supervised children reached acceptable exits."""

    if require_children and not exit_codes:
        return False
    return all(
        supervised_child_succeeded(
            child_id=child_id,
            exit_code=exit_code,
            runner_terminated_child_ids=runner_terminated_child_ids,
            stop_requested=stop_requested,
            allow_runner_terminated=allow_runner_terminated,
        )
        for child_id, exit_code in exit_codes.items()
    )


def child_summary_age_seconds(
    path: Path,
    data: Mapping[str, Any],
    *,
    timestamp_fields: Sequence[str] = (
        "heartbeat_at",
        "updated_at",
        "active_packet_last_heartbeat_at",
        "finished_at",
        "created_at",
        "started_at",
    ),
    now: Optional[float] = None,
    ) -> Optional[float]:
    """Return the age of a child summary from known timestamps or mtime."""

    now_epoch = time.time() if now is None else float(now)
    ages: list[float] = []
    for key in timestamp_fields:
        age_seconds = timestamp_age_seconds(data.get(key), now=now_epoch)
        if age_seconds is not None:
            ages.append(age_seconds)
    if ages:
        return min(ages)
    try:
        return max(0.0, now_epoch - path.stat().st_mtime)
    except OSError:
        return None


def timestamp_age_seconds(value: Any, *, now: Optional[float] = None) -> Optional[float]:
    """Return the age in seconds for an ISO timestamp-like value."""

    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now_epoch = time.time() if now is None else float(now)
    return max(0.0, now_epoch - parsed.timestamp())


def summarize_child_summary_files(
    paths: Sequence[Path],
    *,
    spec: ChildSummaryHealthSpec,
    stale_seconds: float = 0.0,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Summarize reusable health signals from supervisor child JSON files."""

    summary_count = 0
    active_count = 0
    waiting_count = 0
    scope_counts: Counter[str] = Counter()
    latest_reasons: Counter[str] = Counter()
    summary_age_seconds: dict[str, float] = {}
    stale_child_ids: set[str] = set()
    numeric_totals: dict[str, int] = {field: 0 for field in spec.numeric_total_fields}
    threshold = max(0.0, float(stale_seconds))
    for path in paths:
        if not path.exists():
            continue
        data = read_json(path)
        if not data:
            continue
        summary_count += 1
        worker_id = str(data.get(spec.worker_id_field) or path.stem)
        scope = str(data.get(spec.scope_field) or "unscoped")
        scope_counts[scope] += 1
        age_seconds = child_summary_age_seconds(
            path,
            data,
            timestamp_fields=spec.timestamp_fields,
            now=now,
        )
        if age_seconds is not None:
            summary_age_seconds[worker_id] = round(float(age_seconds), 3)
            if threshold > 0.0 and age_seconds >= threshold:
                stale_child_ids.add(worker_id)
        latest_reason = str(data.get(spec.latest_reason_field) or "")
        if latest_reason:
            latest_reasons[latest_reason] += 1
        active_phase = str(data.get(spec.active_phase_field) or "")
        active_ids = data.get(spec.active_ids_field) or []
        has_active_work = bool(
            active_phase in spec.active_phases
            and isinstance(active_ids, list)
            and active_ids
        )
        if has_active_work:
            active_count += 1
        if latest_reason in spec.waiting_reasons and not has_active_work:
            waiting_count += 1
        for field in spec.numeric_total_fields:
            try:
                numeric_totals[field] += int(data.get(field, 0) or 0)
            except (TypeError, ValueError):
                pass

    return {
        "active_count": active_count,
        "latest_stop_reasons": dict(sorted(latest_reasons.items())),
        "numeric_totals": dict(sorted(numeric_totals.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "stale_child_ids": sorted(stale_child_ids),
        "stale_count": len(stale_child_ids),
        "summary_age_seconds": dict(sorted(summary_age_seconds.items())),
        "summary_count": summary_count,
        "waiting_count": waiting_count,
    }


def terminate_process_group(process: subprocess.Popen[Any], signum: int) -> bool:
    """Signal a child process group, falling back to the child process itself."""

    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signum)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        try:
            process.send_signal(signum)
            return True
        except OSError:
            return False


def _captured_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_process_group_capture(
    command: Sequence[str],
    *,
    cwd: Path | str,
    env: Optional[Mapping[str, object]] = None,
    input_text: Optional[str] = None,
    timeout_seconds: float,
    kill_wait_seconds: float = 5.0,
    start_new_session: bool = True,
    text: bool = True,
) -> dict[str, Any]:
    """Run a child process group, capturing output and killing leaks on timeout."""

    started = time.time()
    process: Optional[subprocess.Popen[Any]] = None
    try:
        input_value: Any = input_text
        if input_text is not None and not text:
            input_value = input_text.encode("utf-8")
        process = launch_process_child(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=start_new_session,
            text=text,
        )
        stdout, stderr = process.communicate(
            input=input_value,
            timeout=max(1.0, float(timeout_seconds)),
        )
        return {
            "duration_seconds": round(time.time() - started, 3),
            "exit_code": process.returncode,
            "status": "completed",
            "stderr": _captured_process_text(stderr),
            "stdout": _captured_process_text(stdout),
        }
    except OSError as exc:
        return {
            "duration_seconds": round(time.time() - started, 3),
            "exit_code": None,
            "status": "failed",
            "stderr": str(exc),
            "stdout": "",
        }
    except subprocess.TimeoutExpired as exc:
        stdout: Any = exc.stdout
        stderr: Any = exc.stderr
        if process is not None:
            terminate_process_group(process, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(
                    timeout=max(0.0, float(kill_wait_seconds))
                )
            except subprocess.TimeoutExpired as term_exc:
                stdout = term_exc.stdout if term_exc.stdout is not None else stdout
                stderr = term_exc.stderr if term_exc.stderr is not None else stderr
                terminate_process_group(process, signal.SIGKILL)
                try:
                    stdout, stderr = process.communicate(
                        timeout=max(0.0, float(kill_wait_seconds))
                    )
                except subprocess.TimeoutExpired as kill_exc:
                    stdout = kill_exc.stdout if kill_exc.stdout is not None else stdout
                    stderr = kill_exc.stderr if kill_exc.stderr is not None else stderr
        return {
            "duration_seconds": round(time.time() - started, 3),
            "exit_code": process.returncode if process is not None else None,
            "status": "timeout",
            "stderr": _captured_process_text(stderr),
            "stdout": _captured_process_text(stdout),
            "timeout_seconds": float(timeout_seconds),
        }


def terminate_process_with_grace(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 10.0,
    kill_wait_seconds: float = 5.0,
    terminate_signal: int = signal.SIGTERM,
    kill_signal: int = signal.SIGKILL,
) -> ProcessTerminationResult:
    """Terminate a child process group, escalating to kill after a grace period."""

    pid = int(process.pid)
    initial_exit_code = process.poll()
    if initial_exit_code is not None:
        return ProcessTerminationResult(
            pid=pid,
            initial_exit_code=int(initial_exit_code),
            final_exit_code=int(initial_exit_code),
        )

    terminate_sent = terminate_process_group(process, terminate_signal)
    kill_sent = False
    timed_out = False
    try:
        process.wait(timeout=max(0.0, float(grace_seconds)))
    except subprocess.TimeoutExpired:
        kill_sent = terminate_process_group(process, kill_signal)
        try:
            process.wait(timeout=max(0.0, float(kill_wait_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True

    final_exit_code = process.poll()
    return ProcessTerminationResult(
        pid=pid,
        initial_exit_code=initial_exit_code,
        final_exit_code=int(final_exit_code) if final_exit_code is not None else None,
        terminate_sent=terminate_sent,
        kill_sent=kill_sent,
        timed_out=timed_out,
    )


def terminate_processes_with_grace(
    processes: (
        Mapping[str, Optional[subprocess.Popen[Any]]]
        | Sequence[tuple[str, Optional[subprocess.Popen[Any]]]]
    ),
    *,
    grace_seconds: float = 10.0,
    kill_wait_seconds: float = 5.0,
    terminate_signal: int = signal.SIGTERM,
    kill_signal: int = signal.SIGKILL,
) -> dict[str, ProcessTerminationResult]:
    """Terminate many child process groups after signaling all active children first."""

    items = processes.items() if isinstance(processes, Mapping) else processes
    active: list[tuple[str, subprocess.Popen[Any], Optional[int], bool]] = []
    results: dict[str, ProcessTerminationResult] = {}
    for child_id, process in items:
        if process is None:
            continue
        child_key = str(child_id)
        initial_exit_code = process.poll()
        if initial_exit_code is not None:
            results[child_key] = ProcessTerminationResult(
                pid=int(process.pid),
                initial_exit_code=int(initial_exit_code),
                final_exit_code=int(initial_exit_code),
            )
            continue
        terminate_sent = terminate_process_group(process, terminate_signal)
        active.append((child_key, process, initial_exit_code, terminate_sent))

    deadline = time.time() + max(0.0, float(grace_seconds))
    kill_candidates: list[tuple[str, subprocess.Popen[Any], Optional[int], bool]] = []
    for child_key, process, initial_exit_code, terminate_sent in active:
        try:
            process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
        final_exit_code = process.poll()
        if final_exit_code is None:
            kill_candidates.append(
                (child_key, process, initial_exit_code, terminate_sent)
            )
            continue
        results[child_key] = ProcessTerminationResult(
            pid=int(process.pid),
            initial_exit_code=initial_exit_code,
            final_exit_code=int(final_exit_code) if final_exit_code is not None else None,
            terminate_sent=terminate_sent,
            kill_sent=False,
            timed_out=False,
        )

    kill_results: list[
        tuple[str, subprocess.Popen[Any], Optional[int], bool, bool]
    ] = []
    for child_key, process, initial_exit_code, terminate_sent in kill_candidates:
        kill_results.append(
            (
                child_key,
                process,
                initial_exit_code,
                terminate_sent,
                terminate_process_group(process, kill_signal),
            )
        )

    kill_deadline = time.time() + max(0.0, float(kill_wait_seconds))
    for child_key, process, initial_exit_code, terminate_sent, kill_sent in kill_results:
        timed_out = False
        try:
            process.wait(timeout=max(0.0, kill_deadline - time.time()))
        except subprocess.TimeoutExpired:
            timed_out = True
        final_exit_code = process.poll()
        results[child_key] = ProcessTerminationResult(
            pid=int(process.pid),
            initial_exit_code=initial_exit_code,
            final_exit_code=int(final_exit_code) if final_exit_code is not None else None,
            terminate_sent=terminate_sent,
            kill_sent=kill_sent,
            timed_out=timed_out,
        )
    return results


def _prepare_marker_path(path: Path, *, remove_existing_file: bool) -> Optional[Path]:
    if path.is_symlink():
        path.unlink()
        return None
    if path.is_dir():
        backup_path = unique_backup_path(path, "directory-backup")
        path.rename(backup_path)
        return backup_path
    if remove_existing_file:
        path.unlink(missing_ok=True)
    return None


def launch_supervised_child(spec: SupervisedChildSpec) -> SupervisedChild:
    """Launch a supervisor-owned child process and write its marker files."""

    log_path = spec.resolve(spec.log_path)
    child_pid_path = spec.resolve(spec.child_pid_path)
    latest_log_path = spec.resolve(spec.latest_log_path) if spec.latest_log_path is not None else None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_pid_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_marker_path(log_path, remove_existing_file=False)
    _prepare_marker_path(child_pid_path, remove_existing_file=False)
    if latest_log_path is not None:
        latest_log_path.parent.mkdir(parents=True, exist_ok=True)
        _prepare_marker_path(latest_log_path, remove_existing_file=True)
        latest_log_path.symlink_to(log_path.name)

    env = {key: str(value) for key, value in spec.env.items()}
    out_handle = log_path.open("ab")
    try:
        process = launch_process_child(
            spec.command,
            cwd=str(spec.repo_root),
            env=env,
            stdin=subprocess.DEVNULL if spec.stdin_devnull else None,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            start_new_session=spec.start_new_session,
        )
    finally:
        out_handle.close()
    child_pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return SupervisedChild(
        pid=int(process.pid),
        command=tuple(spec.command),
        log_path=log_path,
        child_pid_path=child_pid_path,
        latest_log_path=latest_log_path,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def clear_child_pid_file(child: SupervisedChild | SupervisedChildSpec, *, pid: Optional[int] = None) -> bool:
    """Remove a child pid file if it still refers to the expected child."""

    child_pid_path = child.child_pid_path
    if isinstance(child, SupervisedChildSpec):
        child_pid_path = child.resolve(child.child_pid_path)
    expected = str(pid if pid is not None else getattr(child, "pid", "")).strip()
    try:
        current = child_pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    except (OSError, UnicodeDecodeError):
        if child_pid_path.is_dir() or child_pid_path.is_symlink():
            _prepare_marker_path(child_pid_path, remove_existing_file=False)
            return True
        return False
    if expected and current != expected:
        return False
    child_pid_path.unlink(missing_ok=True)
    return True


def terminate_supervised_child(
    child: SupervisedChild,
    *,
    grace_seconds: float = 10.0,
    clear_pid_file: bool = True,
) -> bool:
    """Terminate a supervisor child process tree and optionally clear its pid file."""

    stopped = terminate_pid_tree(child.pid, grace_seconds=grace_seconds)
    if clear_pid_file:
        clear_child_pid_file(child)
    return stopped


def wait_for_child_exit(child: SupervisedChild, *, poll_interval_seconds: float = 0.2) -> int:
    """Wait for a child process id to disappear and return a process-style code."""

    while True:
        try:
            waited_pid, status = os.waitpid(child.pid, os.WNOHANG)
        except ChildProcessError:
            return 0
        if waited_pid == child.pid:
            if os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            if os.WIFSIGNALED(status):
                return 128 + os.WTERMSIG(status)
            return status
        time.sleep(max(0.01, float(poll_interval_seconds)))


def current_python_executable_command(module: str, args: Sequence[str] = ()) -> tuple[str, ...]:
    """Build a ``sys.executable -m`` command for in-package supervisors."""

    return build_python_module_command(
        module,
        args,
        python_executable=sys.executable,
        unbuffered=True,
    )
