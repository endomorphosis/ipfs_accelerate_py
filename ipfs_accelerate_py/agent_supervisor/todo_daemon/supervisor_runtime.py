"""Reusable child-process runtime helpers for todo-daemon supervisors."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ..core.wrapper_utils import with_exclusive_flag_default
from ..merge.checkout_lock import serialized_lock_update
from ..runtime.event_log import unique_backup_path
from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
    read_process_birth,
)
from .core import now_iso, parse_timestamp, pid_alive, process_args, read_json, read_pid_file, remove_runtime_marker, terminate_pid_tree, write_json


SUPERVISED_CHILD_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.supervised-child-identity@1"
)
SUPERVISED_CHILD_IDENTITY_PATH_ENV = (
    "IPFS_ACCELERATE_SUPERVISED_CHILD_IDENTITY_PATH"
)
SUPERVISED_CHILD_OWNER_SCOPE_ENV = (
    "IPFS_ACCELERATE_SUPERVISED_CHILD_OWNER_SCOPE"
)


@dataclass
class RestartPolicy:
    """Restart delays for a supervised daemon child.

    Supports exponential backoff: each consecutive restart multiplies the delay
    by ``backoff_factor`` up to ``max_backoff_seconds``. The counter resets
    after a successful run longer than ``healthy_run_seconds``.
    """

    restart_backoff_seconds: float = 30.0
    fast_restart_backoff_seconds: float = 2.0
    backoff_factor: float = 1.5
    max_backoff_seconds: float = 600.0  # cap at 10 minutes
    healthy_run_seconds: float = 120.0  # reset backoff after this runtime
    fast_restart_statuses: frozenset[str] = frozenset(
        {
            "dirty_recovery_skipped_clean",
            "repeated_rejection_recovery_skipped_clean",
            "no_change",
        }
    )
    _consecutive_failures: int = 0

    def delay_for_status(self, status: str, *, run_duration: float = 0.0) -> float:
        if run_duration >= self.healthy_run_seconds:
            object.__setattr__(self, "_consecutive_failures", 0)
        else:
            object.__setattr__(
                self, "_consecutive_failures", self._consecutive_failures + 1
            )

        if status in self.fast_restart_statuses:
            return max(0.0, float(self.fast_restart_backoff_seconds))

        base = max(0.0, float(self.restart_backoff_seconds))
        multiplier = self.backoff_factor ** min(max(self._consecutive_failures - 1, 0), 10)
        return min(base * multiplier, self.max_backoff_seconds)

    def reset(self) -> None:
        """Reset backoff state (e.g. after a healthy run)."""
        object.__setattr__(self, "_consecutive_failures", 0)


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
    identity_path: Optional[Path] = None
    identity_record_id: str = ""
    identity_process_birth: Optional[ProcessBirthIdentity] = None
    owned_process_group_id: Optional[int] = None


@dataclass(frozen=True)
class SupervisedChildIdentity:
    """Durable PID-reuse-resistant identity for one supervisor-owned child."""

    process_birth: ProcessBirthIdentity
    command: tuple[str, ...]
    owner_scope: Mapping[str, str]
    created_at: str
    record_id: str = ""

    def to_dict(self, *, include_record_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SUPERVISED_CHILD_IDENTITY_SCHEMA,
            "process_birth": self.process_birth.to_dict(),
            "command": list(self.command),
            "command_id": _supervised_child_content_identity(
                {"command": list(self.command)}
            ),
            "owner_scope": {
                str(key): str(value)
                for key, value in sorted(self.owner_scope.items())
            },
            "created_at": str(self.created_at),
        }
        if include_record_id:
            payload["record_id"] = self.record_id or (
                _supervised_child_content_identity(payload)
            )
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "SupervisedChildIdentity | None":
        data = dict(payload or {})
        expected_fields = {
            "schema",
            "process_birth",
            "command",
            "command_id",
            "owner_scope",
            "created_at",
            "record_id",
        }
        if set(data) != expected_fields:
            return None
        if data.get("schema") != SUPERVISED_CHILD_IDENTITY_SCHEMA:
            return None
        raw_birth = data.get("process_birth")
        raw_command = data.get("command")
        raw_scope = data.get("owner_scope")
        if (
            not isinstance(raw_birth, Mapping)
            or not isinstance(raw_command, list)
            or not raw_command
            or not all(isinstance(item, str) and item for item in raw_command)
            or not isinstance(raw_scope, Mapping)
            or not all(
                isinstance(key, str)
                and key
                and isinstance(value, str)
                and value
                for key, value in raw_scope.items()
            )
            or not isinstance(data.get("created_at"), str)
            or not data.get("created_at")
            or not isinstance(data.get("record_id"), str)
        ):
            return None
        try:
            birth = ProcessBirthIdentity.from_dict(raw_birth)
        except (TypeError, ValueError):
            return None
        if (
            birth.pid <= 1
            or birth.start_time_ticks <= 0
            or not birth.boot_id
        ):
            return None
        command = tuple(raw_command)
        if data.get("command_id") != _supervised_child_content_identity(
            {"command": list(command)}
        ):
            return None
        normalized = dict(data)
        record_id = str(normalized.pop("record_id") or "")
        if record_id != _supervised_child_content_identity(normalized):
            return None
        return cls(
            process_birth=birth,
            command=command,
            owner_scope={
                str(key): str(value)
                for key, value in raw_scope.items()
            },
            created_at=str(data["created_at"]),
            record_id=record_id,
        )


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
    pass_fds: Sequence[int] = (),
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
    normalized_pass_fds = tuple(int(item) for item in pass_fds)
    if normalized_pass_fds:
        # Preserve compatibility with injected/fake ``Popen`` callables and
        # non-POSIX launchers when no descriptor inheritance was requested.
        kwargs["pass_fds"] = normalized_pass_fds
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
    command_line = process_args(pid)
    if owner_script and command_line and owner_script not in command_line:
        owner_module_stem = Path(owner_script).stem
        if not owner_module_stem or owner_module_stem not in command_line:
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
    try:
        with serialized_lock_update(lock_path):
            if lock_path.exists() and not runtime_lock_owner_is_alive(lock_path):
                if remove_runtime_marker(lock_path):
                    repairs["removed"].append(str(lock_path))
    except (OSError, RuntimeError) as exc:
        # Runtime repair is a recovery convenience, never authority to mutate
        # a lease whose ownership could not be inspected atomically.
        repairs["implementation_lock_repair_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

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


class ProcessGroupCancelled(RuntimeError):
    """Raised after a cancellation predicate fences an owned process group."""

    def __init__(self, reason: str = "cancellation_requested") -> None:
        self.reason = str(reason or "cancellation_requested")
        super().__init__(self.reason)


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


def run_process_group_stream(
    command: Sequence[str],
    *,
    cwd: Path | str,
    stdout: Any,
    stderr: Any = subprocess.STDOUT,
    input_text: Optional[str] = None,
    env: Optional[Mapping[str, object]] = None,
    timeout_seconds: float,
    progress_timeout_seconds: float | None = None,
    max_timeout_seconds: float | None = None,
    progress_paths: Sequence[Path | str] = (),
    on_started: Callable[[subprocess.Popen[Any]], None] | None = None,
    on_progress: Callable[[Mapping[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool | str] | None = None,
    cancellation_reason: str = "cancellation_requested",
    progress_poll_seconds: float = 1.0,
    termination_grace_seconds: float = 5.0,
    text: bool = True,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[Any]:
    """Run a streamed child in an owned process group and fence it on timeout.

    ``timeout_seconds`` retains the historical absolute-deadline behaviour.
    Supplying ``progress_timeout_seconds`` enables an idle-progress deadline:
    output-file or ``progress_paths`` mutations renew that deadline until the
    independent ``max_timeout_seconds`` hard cap.  The hard cap prevents a
    noisy or malicious child from extending its lease forever. Supplying only
    ``on_progress`` retains the absolute deadline while polling the same
    progress marker for telemetry.
    """

    input_value: Any = input_text
    if input_text is not None and not text:
        input_value = input_text.encode("utf-8")
    idle_timeout: float | None = None
    hard_timeout: float | None = None
    poll_seconds: float | None = None
    monitor_progress = (
        progress_timeout_seconds is not None
        or on_progress is not None
        or cancel_requested is not None
    )
    if progress_timeout_seconds is not None:
        idle_timeout = float(progress_timeout_seconds)
        hard_timeout = float(
            timeout_seconds
            if max_timeout_seconds is None
            else max_timeout_seconds
        )
        if (
            not math.isfinite(idle_timeout)
            or not math.isfinite(hard_timeout)
            or idle_timeout <= 0
            or hard_timeout <= 0
        ):
            raise ValueError(
                "progress and maximum timeouts must be finite and positive"
            )
        if hard_timeout < idle_timeout:
            raise ValueError(
                "max_timeout_seconds cannot be shorter than "
                "progress_timeout_seconds"
            )
        requested_poll_seconds = float(progress_poll_seconds)
        if (
            not math.isfinite(requested_poll_seconds)
            or requested_poll_seconds <= 0
        ):
            raise ValueError(
                "progress_poll_seconds must be finite and positive"
            )
        poll_seconds = max(
            0.01,
            min(requested_poll_seconds, idle_timeout),
        )
    elif monitor_progress:
        hard_timeout = max(0.0, float(timeout_seconds))
        requested_poll_seconds = float(progress_poll_seconds)
        if (
            not math.isfinite(requested_poll_seconds)
            or requested_poll_seconds <= 0
        ):
            raise ValueError(
                "progress_poll_seconds must be finite and positive"
            )
        poll_seconds = max(0.01, requested_poll_seconds)
    process = launch_process_child(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        text=text,
        pass_fds=pass_fds,
    )
    callback_failure_cleaned = False
    try:
        if on_started is not None:
            try:
                on_started(process)
            except BaseException as callback_error:
                # A caller that requires durable birth evidence must never
                # leave an unaccounted child or daemonized group running when
                # persistence fails.  Fence before poll/wait can reap the
                # leader and make its dedicated group undiscoverable.
                try:
                    birth = read_process_birth(process.pid)
                except OSError:
                    birth = None
                fenced = terminate_pid_tree(
                    process.pid,
                    grace_seconds=max(
                        0.1,
                        float(termination_grace_seconds),
                    ),
                    freeze_first=True,
                    require_gone=True,
                    owned_process_group_id=process.pid,
                    expected_root_start_time_ticks=(
                        birth.start_time_ticks if birth is not None else None
                    ),
                )
                try:
                    process.wait(
                        timeout=max(0.1, float(termination_grace_seconds))
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "provider runner callback failure could not be reaped"
                    ) from exc
                # The leader is now reaped and its numeric PID may be reused.
                # Prevent the generic exception cleanup below from acting on
                # that unbound number a second time.
                callback_failure_cleaned = True
                if not fenced:
                    raise RuntimeError(
                        "provider runner callback failure could not be fenced"
                    ) from callback_error
                raise
        if not monitor_progress:
            process.communicate(
                input=input_value,
                timeout=max(0.0, float(timeout_seconds)),
            )
        else:
            assert hard_timeout is not None
            assert poll_seconds is not None
            started = time.monotonic()
            hard_deadline = started + hard_timeout
            idle_deadline = (
                started + idle_timeout
                if idle_timeout is not None
                else None
            )
            progress_marker = _stream_progress_marker(
                stdout,
                stderr=stderr,
                progress_paths=progress_paths,
            )
            progress_events = 0

            input_thread: threading.Thread | None = None
            if process.stdin is not None:
                stdin_stream = process.stdin
                # ``communicate`` must not race the bounded writer during
                # timeout cleanup.
                process.stdin = None

                def write_input() -> None:
                    try:
                        if input_value is not None:
                            stdin_stream.write(input_value)
                            stdin_stream.flush()
                    except (BrokenPipeError, OSError, ValueError):
                        pass
                    finally:
                        try:
                            stdin_stream.close()
                        except OSError:
                            pass

                input_thread = threading.Thread(
                    target=write_input,
                    name="supervisor-stream-input",
                    daemon=True,
                )
                input_thread.start()

            while process.poll() is None:
                now = time.monotonic()
                if cancel_requested is not None:
                    cancellation = cancel_requested()
                    if cancellation:
                        reason = (
                            cancellation
                            if isinstance(cancellation, str)
                            else cancellation_reason
                        )
                        raise ProcessGroupCancelled(str(reason))
                timeout_reason = ""
                if now >= hard_deadline:
                    timeout_reason = (
                        "hard_timeout"
                        if idle_timeout is not None
                        else "absolute_timeout"
                    )
                elif idle_deadline is not None and now >= idle_deadline:
                    timeout_reason = "progress_idle_timeout"
                if timeout_reason:
                    timeout_value = hard_timeout
                    if timeout_reason == "progress_idle_timeout":
                        assert idle_timeout is not None
                        timeout_value = idle_timeout
                    exc = subprocess.TimeoutExpired(
                        cmd=list(command),
                        timeout=timeout_value,
                    )
                    setattr(exc, "timeout_reason", timeout_reason)
                    setattr(exc, "elapsed_seconds", max(0.0, now - started))
                    setattr(exc, "progress_events", progress_events)
                    setattr(exc, "progress_timeout_seconds", idle_timeout)
                    setattr(exc, "max_timeout_seconds", hard_timeout)
                    raise exc

                wait_deadlines = [
                    poll_seconds,
                    max(0.001, hard_deadline - now),
                ]
                if idle_deadline is not None:
                    wait_deadlines.append(
                        max(0.001, idle_deadline - now)
                    )
                wait_seconds = min(wait_deadlines)
                try:
                    process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    pass
                next_marker = _stream_progress_marker(
                    stdout,
                    stderr=stderr,
                    progress_paths=progress_paths,
                )
                if next_marker != progress_marker:
                    progress_marker = next_marker
                    progress_events += 1
                    observed_at = time.monotonic()
                    if idle_timeout is not None:
                        idle_deadline = observed_at + idle_timeout
                    if on_progress is not None:
                        try:
                            on_progress(
                                {
                                    "elapsed_seconds": max(
                                        0.0, observed_at - started
                                    ),
                                    "progress_events": progress_events,
                                    "progress_timeout_seconds": idle_timeout,
                                    "max_timeout_seconds": hard_timeout,
                                }
                            )
                        except Exception:
                            # Progress telemetry must never change child
                            # execution semantics.
                            pass
            if input_thread is not None:
                input_thread.join(timeout=0.1)
    except ProcessGroupCancelled:
        # A canonical-board cancellation is an authority hand-off, so the
        # provider tree must be unable to execute before the caller can release
        # its task/resource claims. Freeze first to close TERM-handler fork
        # races, include the stable process group owned by the session leader,
        # and do not return until the strict fence proves every member gone.
        terminate_pid_tree(
            process.pid,
            grace_seconds=max(0.0, float(termination_grace_seconds)),
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=process.pid,
        )
        # The strict fence treats zombies as non-executable; reap the direct
        # child before propagating cancellation so no process resource leaks.
        process.wait()
        raise
    except subprocess.TimeoutExpired as exc:
        terminate_pid_tree(
            process.pid,
            grace_seconds=max(0.0, float(termination_grace_seconds)),
        )
        try:
            process.communicate(timeout=max(0.1, float(termination_grace_seconds)))
        except subprocess.TimeoutExpired:
            terminate_process_group(process, signal.SIGKILL)
            process.communicate()
        timeout_exc = subprocess.TimeoutExpired(
            cmd=list(command),
            timeout=float(getattr(exc, "timeout", timeout_seconds)),
            output=exc.output,
            stderr=exc.stderr,
        )
        for attribute in (
            "timeout_reason",
            "elapsed_seconds",
            "progress_events",
            "progress_timeout_seconds",
            "max_timeout_seconds",
        ):
            if hasattr(exc, attribute):
                setattr(timeout_exc, attribute, getattr(exc, attribute))
        raise timeout_exc from exc
    except BaseException:
        if callback_failure_cleaned:
            raise
        # The implementation daemon can be recycled while a provider runner
        # owns a separate process session. Fence that complete tree before the
        # daemon releases its canonical task claim.
        terminate_pid_tree(
            process.pid,
            grace_seconds=max(0.0, float(termination_grace_seconds)),
        )
        try:
            process.wait(timeout=max(0.1, float(termination_grace_seconds)))
        except subprocess.TimeoutExpired:
            terminate_process_group(process, signal.SIGKILL)
            process.wait()
        raise
    # A successful CLI process may have daemonized descendants that closed
    # their inherited output descriptors.  They remain in the owned session
    # and could mutate the checkout after the implementation fence's final
    # check, so quiesce the complete group before returning to validation.
    def group_alive() -> bool:
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return False

    if group_alive():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + max(0.0, float(termination_grace_seconds))
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if group_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=int(process.returncode or 0),
    )


def _stream_progress_marker(
    stdout: Any,
    *,
    stderr: Any = None,
    progress_paths: Sequence[Path | str],
    max_entries: int = 512,
) -> tuple[tuple[str, int, int], ...]:
    """Return a bounded metadata marker for streamed output and checkpoints."""

    marker: list[tuple[str, int, int]] = []
    try:
        stat = os.fstat(stdout.fileno())
        marker.append(("<stdout>", int(stat.st_size), int(stat.st_mtime_ns)))
    except (AttributeError, OSError, ValueError):
        pass
    if not any(
        stderr is sentinel
        for sentinel in (
            None,
            subprocess.STDOUT,
            subprocess.PIPE,
            subprocess.DEVNULL,
        )
    ):
        try:
            stat = os.fstat(stderr.fileno())
            marker.append(("<stderr>", int(stat.st_size), int(stat.st_mtime_ns)))
        except (AttributeError, OSError, ValueError):
            pass

    remaining = max(0, int(max_entries))
    for raw_path in progress_paths:
        if remaining <= 0:
            break
        root = Path(raw_path)
        try:
            root_stat = root.lstat()
        except OSError:
            marker.append((str(root), -1, -1))
            remaining -= 1
            continue
        marker.append(
            (str(root), int(root_stat.st_size), int(root_stat.st_mtime_ns))
        )
        remaining -= 1
        if not root.is_dir() or root.is_symlink():
            continue
        try:
            descendants = sorted(root.rglob("*"), key=lambda item: str(item))
        except OSError:
            continue
        for descendant in descendants:
            if remaining <= 0:
                break
            try:
                stat = descendant.lstat()
            except OSError:
                continue
            marker.append(
                (
                    str(descendant),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
            )
            remaining -= 1
    return tuple(marker)


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


def _supervised_child_content_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def supervised_child_identity_path(child_pid_path: Path) -> Path:
    """Return the sidecar path paired with a legacy raw PID marker."""

    suffix = child_pid_path.suffix
    stem = child_pid_path.name[: -len(suffix)] if suffix else child_pid_path.name
    return child_pid_path.with_name(f"{stem}.identity.json")


def _configured_child_identity_path(spec: SupervisedChildSpec) -> Path | None:
    raw_path = str(
        spec.env.get(SUPERVISED_CHILD_IDENTITY_PATH_ENV, "") or ""
    ).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else spec.repo_root / path


def _configured_child_owner_scope(
    spec: SupervisedChildSpec,
) -> dict[str, str] | None:
    raw_scope = str(
        spec.env.get(SUPERVISED_CHILD_OWNER_SCOPE_ENV, "") or ""
    ).strip()
    if not raw_scope:
        return None
    try:
        payload = json.loads(raw_scope)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    if not all(
        isinstance(key, str)
        and key
        and isinstance(value, str)
        and value
        for key, value in payload.items()
    ):
        return None
    return {str(key): str(value) for key, value in payload.items()}


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_supervised_child_identity(
    path: Path,
) -> SupervisedChildIdentity | None:
    """Load one closed, content-addressed child identity record."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return SupervisedChildIdentity.from_dict(payload)


def write_supervised_child_identity(
    path: Path,
    *,
    pid: int,
    command: Sequence[str],
    owner_scope: Mapping[str, str],
    require_direct_child: bool = False,
) -> SupervisedChildIdentity:
    """Capture and atomically persist the exact birth identity of ``pid``."""

    try:
        process_birth = read_process_birth(int(pid))
    except OSError as exc:
        raise RuntimeError("supervised child process identity unavailable") from exc
    if (
        process_birth is None
        or process_birth.start_time_ticks <= 0
        or not process_birth.boot_id
    ):
        raise RuntimeError("supervised child process identity unavailable")
    if require_direct_child and process_birth.parent_pid != os.getpid():
        raise RuntimeError("supervised child is not owned by this launcher")
    identity = SupervisedChildIdentity(
        process_birth=process_birth,
        command=tuple(str(part) for part in command),
        owner_scope={str(key): str(value) for key, value in owner_scope.items()},
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    payload = identity.to_dict()
    persisted = SupervisedChildIdentity.from_dict(payload)
    if persisted is None:
        raise RuntimeError("could not construct supervised child identity")
    _write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return persisted


def supervised_child_identity_liveness(
    identity: SupervisedChildIdentity,
) -> OwnerLiveness:
    """Evaluate an identity without treating a reused numeric PID as alive."""

    return owner_liveness(identity.process_birth)


def read_process_command_argv(pid: int) -> tuple[str, ...] | None:
    """Read exact argv from procfs, returning None when it cannot be proven."""

    try:
        raw = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    if not raw:
        return None
    try:
        return tuple(
            part.decode("utf-8")
            for part in raw.split(b"\0")
            if part
        )
    except UnicodeError:
        return None


def terminate_direct_child_process(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 1.0,
) -> bool:
    """Reap a just-launched direct child without relying on procfs identity."""

    try:
        if process.poll() is not None:
            process.wait(timeout=0)
            return True
        process.terminate()
        try:
            process.wait(timeout=max(0.0, float(grace_seconds)))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=max(1.0, float(grace_seconds)))
        return process.poll() is not None
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        return False


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
    identity_path = _configured_child_identity_path(spec)
    owner_scope = _configured_child_owner_scope(spec)
    if (identity_path is None) != (owner_scope is None):
        raise RuntimeError(
            "supervised child identity path and owner scope must be configured together"
        )
    if identity_path is not None and not spec.start_new_session:
        raise RuntimeError(
            "identity-protected child requires a dedicated process session"
        )
    latest_log_path = spec.resolve(spec.latest_log_path) if spec.latest_log_path is not None else None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_pid_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_marker_path(log_path, remove_existing_file=False)
    _prepare_marker_path(child_pid_path, remove_existing_file=False)
    if identity_path is not None:
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        if identity_path.exists() or identity_path.is_symlink():
            raise RuntimeError(
                "supervised child identity marker was not reconciled"
            )
    if latest_log_path is not None:
        latest_log_path.parent.mkdir(parents=True, exist_ok=True)
        _prepare_marker_path(latest_log_path, remove_existing_file=True)
        latest_log_path.symlink_to(log_path.name)

    env = {key: str(value) for key, value in spec.env.items()}
    env.pop(SUPERVISED_CHILD_IDENTITY_PATH_ENV, None)
    env.pop(SUPERVISED_CHILD_OWNER_SCOPE_ENV, None)
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
    persisted_identity: SupervisedChildIdentity | None = None
    if identity_path is not None and owner_scope is not None:
        try:
            persisted_identity = write_supervised_child_identity(
                identity_path,
                pid=int(process.pid),
                command=spec.command,
                owner_scope=owner_scope,
                require_direct_child=True,
            )
        except Exception:
            direct_child_stopped = terminate_direct_child_process(
                process,
                grace_seconds=1.0,
            )
            launched_birth = None
            if not direct_child_stopped:
                try:
                    launched_birth = read_process_birth(int(process.pid))
                except OSError:
                    launched_birth = None
            if (
                not direct_child_stopped
                and launched_birth is not None
                and launched_birth.parent_pid == os.getpid()
            ):
                terminate_pid_tree(
                    int(process.pid),
                    grace_seconds=1.0,
                    freeze_first=True,
                    require_gone=True,
                    owned_process_group_id=(
                        int(process.pid) if spec.start_new_session else None
                    ),
                    expected_root_start_time_ticks=(
                        launched_birth.start_time_ticks
                    ),
                )
            raise
    # The raw PID remains the compatibility/commit marker and is written only
    # after the PID-reuse-resistant identity record is durable.
    _write_bytes_atomic(child_pid_path, f"{process.pid}\n".encode("ascii"))
    return SupervisedChild(
        pid=int(process.pid),
        command=tuple(spec.command),
        log_path=log_path,
        child_pid_path=child_pid_path,
        latest_log_path=latest_log_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        identity_path=identity_path,
        identity_record_id=(
            persisted_identity.record_id
            if persisted_identity is not None
            else ""
        ),
        identity_process_birth=(
            persisted_identity.process_birth
            if persisted_identity is not None
            else None
        ),
        owned_process_group_id=(
            int(process.pid) if persisted_identity is not None else None
        ),
    )


def adopt_or_launch_supervised_child(
    spec: SupervisedChildSpec,
    *,
    launch_lock_path: Path,
) -> SupervisedChild:
    """Atomically adopt or launch one child for a supervisor scope."""

    with serialized_lock_update(launch_lock_path):
        adopted = adopt_supervised_child(spec)
        if adopted is not None:
            return adopted
        return launch_supervised_child(spec)


def supervised_child_command_matches(command_line: str, command: Sequence[str]) -> bool:
    """Return whether a live process command line matches a supervisor child command."""

    if not command_line:
        return False
    required_fragments = [str(part) for part in command[1:] if str(part)]
    if not required_fragments:
        required_fragments = [str(part) for part in command if str(part)]
    return all(fragment in command_line for fragment in required_fragments)


def adopt_supervised_child(spec: SupervisedChildSpec) -> SupervisedChild | None:
    """Return a live matching child from the PID marker instead of launching a duplicate."""

    child_pid_path = spec.resolve(spec.child_pid_path)
    identity_path = _configured_child_identity_path(spec)
    owner_scope = _configured_child_owner_scope(spec)
    if (identity_path is None) != (owner_scope is None):
        raise RuntimeError(
            "supervised child identity path and owner scope must be configured together"
        )
    if identity_path is not None and not spec.start_new_session:
        raise RuntimeError(
            "identity-protected child requires a dedicated process session"
        )
    pid = read_pid_file(child_pid_path)
    identity: SupervisedChildIdentity | None = None
    if identity_path is not None and owner_scope is not None and (
        not pid or not pid_alive(pid)
    ):
        identity_exists = identity_path.exists() or identity_path.is_symlink()
        if identity_exists:
            identity = load_supervised_child_identity(identity_path)
            if identity is None:
                raise RuntimeError(
                    "orphaned supervised child identity is invalid"
                )
            liveness = supervised_child_identity_liveness(identity)
            if liveness is OwnerLiveness.UNKNOWN:
                raise RuntimeError(
                    "orphaned supervised child identity liveness is unknown"
                )
            if liveness is OwnerLiveness.DEAD:
                for path in (child_pid_path, identity_path):
                    if not path.exists() and not path.is_symlink():
                        continue
                    backup = unique_backup_path(path, "stale-child-identity")
                    path.rename(backup)
                return None
            if (
                identity.command != tuple(spec.command)
                or dict(identity.owner_scope) != owner_scope
                or read_process_command_argv(identity.process_birth.pid)
                != identity.command
            ):
                raise RuntimeError(
                    "orphaned supervised child ownership identity mismatch"
                )
            _write_bytes_atomic(
                child_pid_path,
                f"{identity.process_birth.pid}\n".encode("ascii"),
            )
            pid = identity.process_birth.pid
    if not pid or not pid_alive(pid):
        return None
    command_line = process_args(pid)
    if not supervised_child_command_matches(command_line, spec.command):
        return None
    if identity_path is not None and owner_scope is not None:
        identity = load_supervised_child_identity(identity_path)
        process_argv = read_process_command_argv(int(pid))
        if identity is None:
            # A live legacy child is migratable only when it already matches
            # the exact desired command. A config-mismatched legacy PID is
            # rejected by the implementation supervisor before this point.
            if process_argv != tuple(spec.command):
                raise RuntimeError(
                    "legacy supervised child exact command is unproven"
                )
            identity = write_supervised_child_identity(
                identity_path,
                pid=int(pid),
                command=spec.command,
                owner_scope=owner_scope,
            )
        if (
            identity.process_birth.pid != int(pid)
            or identity.command != tuple(spec.command)
            or dict(identity.owner_scope) != owner_scope
            or supervised_child_identity_liveness(identity)
            is not OwnerLiveness.ALIVE
        ):
            raise RuntimeError("supervised child ownership identity mismatch")
        if process_argv != identity.command:
            raise RuntimeError("supervised child command identity mismatch")
    latest_log_path = spec.resolve(spec.latest_log_path) if spec.latest_log_path is not None else None
    log_path = spec.resolve(spec.log_path)
    if latest_log_path is not None:
        try:
            # A supervisor restart creates a fresh run id, but an adopted
            # daemon continues writing the log selected by its original
            # supervisor. The stable latest-log marker preserves that
            # provenance across the supervisor-only restart.
            adopted_log_path = latest_log_path.resolve(strict=True)
            if adopted_log_path.is_file():
                log_path = adopted_log_path
        except (OSError, RuntimeError):
            pass
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return SupervisedChild(
        pid=int(pid),
        command=tuple(spec.command),
        log_path=log_path,
        child_pid_path=child_pid_path,
        latest_log_path=latest_log_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        identity_path=identity_path,
        identity_record_id=(identity.record_id if identity is not None else ""),
        identity_process_birth=(
            identity.process_birth if identity is not None else None
        ),
        owned_process_group_id=(int(pid) if identity is not None else None),
    )


def _supervised_child_identity_matches_handle(
    child: SupervisedChild,
    identity: SupervisedChildIdentity | None,
) -> bool:
    """Bind a durable child handle to the exact identity generation it adopted."""

    if identity is None:
        return False
    if (
        not child.identity_record_id
        or child.identity_process_birth is None
        or child.owned_process_group_id != int(child.pid)
    ):
        return False
    if child.identity_record_id and identity.record_id != child.identity_record_id:
        return False
    if (
        child.identity_process_birth is not None
        and identity.process_birth != child.identity_process_birth
    ):
        return False
    return bool(
        identity.process_birth.pid == int(child.pid)
        and identity.command == tuple(child.command)
    )


def clear_child_pid_file(child: SupervisedChild | SupervisedChildSpec, *, pid: Optional[int] = None) -> bool:
    """Remove a child pid file if it still refers to the expected child."""

    child_pid_path = child.child_pid_path
    identity_path = getattr(child, "identity_path", None)
    if isinstance(child, SupervisedChildSpec):
        child_pid_path = child.resolve(child.child_pid_path)
        identity_path = _configured_child_identity_path(child)
    expected = str(pid if pid is not None else getattr(child, "pid", "")).strip()
    identity_path = identity_path or supervised_child_identity_path(
        child_pid_path
    )
    identity = load_supervised_child_identity(identity_path)
    if isinstance(child, SupervisedChild):
        identity_required = child.identity_path is not None
        if (
            (identity_required and identity is None)
            or (
                identity is not None
                and not _supervised_child_identity_matches_handle(
                    child,
                    identity,
                )
            )
        ):
            return False
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
    if identity is not None and (
        not expected or str(identity.process_birth.pid) == expected
    ):
        identity_path.unlink(missing_ok=True)
    return True


def terminate_supervised_child(
    child: SupervisedChild,
    *,
    grace_seconds: float = 10.0,
    clear_pid_file: bool = True,
) -> bool:
    """Terminate a supervisor child process tree and optionally clear its pid file."""

    default_identity_path = supervised_child_identity_path(
        child.child_pid_path
    )
    identity_path = child.identity_path or default_identity_path
    identity_required = child.identity_path is not None
    identity_enabled = bool(
        identity_required
        or identity_path.exists()
        or identity_path.is_symlink()
    )
    identity = (
        load_supervised_child_identity(identity_path)
        if identity_enabled
        else None
    )
    if identity_enabled:
        if (
            not _supervised_child_identity_matches_handle(child, identity)
            or supervised_child_identity_liveness(identity)
            is not OwnerLiveness.ALIVE
            or read_process_command_argv(int(child.pid))
            != identity.command
        ):
            # A numeric PID can be recycled between supervisor observations.
            # Preserve both markers when exact ownership cannot be proven.
            return False
    stopped = terminate_pid_tree(
        child.pid,
        grace_seconds=grace_seconds,
        freeze_first=identity_enabled,
        require_gone=identity_enabled,
        owned_process_group_id=(
            child.owned_process_group_id if identity_enabled else None
        ),
        expected_root_start_time_ticks=(
            identity.process_birth.start_time_ticks
            if identity is not None
            else None
        ),
    )
    may_clear_identity_markers = bool(
        not identity_enabled
        or stopped
        or (
            identity is not None
            and supervised_child_identity_liveness(identity)
            is OwnerLiveness.DEAD
        )
    )
    if clear_pid_file and may_clear_identity_markers:
        clear_child_pid_file(child)
    return stopped


def wait_for_child_exit(child: SupervisedChild, *, poll_interval_seconds: float = 0.2) -> int:
    """Wait for a child process id to disappear and return a process-style code."""

    while True:
        try:
            waited_pid, status = os.waitpid(child.pid, os.WNOHANG)
        except ChildProcessError:
            identity = (
                load_supervised_child_identity(child.identity_path)
                if child.identity_path is not None
                else None
            )
            if child.identity_path is not None:
                if not _supervised_child_identity_matches_handle(
                    child,
                    identity,
                ):
                    raise RuntimeError(
                        "supervised child exit identity does not match its handle"
                    )
                liveness = supervised_child_identity_liveness(identity)
                if liveness is OwnerLiveness.DEAD:
                    return 0
                if liveness is OwnerLiveness.UNKNOWN:
                    raise RuntimeError(
                        "supervised child exit liveness is unknown"
                    )
            elif not pid_alive(child.pid):
                return 0
            time.sleep(max(0.01, float(poll_interval_seconds)))
            continue
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
