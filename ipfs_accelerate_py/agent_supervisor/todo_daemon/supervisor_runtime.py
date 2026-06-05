"""Reusable child-process runtime helpers for todo-daemon supervisors."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ..event_log import unique_backup_path
from ..wrapper_utils import with_exclusive_flag_default
from .core import now_iso, pid_alive, process_args, read_json, read_pid_file, remove_runtime_marker, terminate_pid_tree, write_json


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


DEFAULT_SUPERVISOR_RUNNING_STATES = frozenset({"running", "starting", "recycling", "restarting"})


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
    env = dict(os.environ)
    paths["wrapper_out"].parent.mkdir(parents=True, exist_ok=True)
    out_handle = paths["wrapper_out"].open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
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

    env = dict(os.environ)
    env.update({key: str(value) for key, value in spec.env.items()})
    out_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
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
