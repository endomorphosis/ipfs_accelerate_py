"""Reusable timed runner for multiple implementation supervisor scripts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .todo_daemon.core import pid_alive, read_pid_file, remove_runtime_marker, terminate_pid_tree
from .wrapper_utils import AgentSupervisorNamespacePaths, apply_env_defaults, env_str


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
    supervisor_status_path: Path | None = None

    def resolve(self, repo_root: Path) -> "SupervisorTrack":
        return SupervisorTrack(
            name=self.name,
            script_path=_resolve_path(repo_root, self.script_path),
            log_path=_resolve_path(repo_root, self.log_path),
            supervisor_pid_path=_resolve_path(repo_root, self.supervisor_pid_path),
            daemon_pid_path=_resolve_path(repo_root, self.daemon_pid_path),
            supervisor_status_path=(
                _resolve_path(repo_root, self.supervisor_status_path)
                if self.supervisor_status_path is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ImplementationSupervisorTrackConfig:
    """Structured inputs for one implementation-supervisor track."""

    name: str
    script_path: Path | str
    state_dir: Path | str
    state_prefix: str

    def compact_spec(self) -> str:
        """Return the compact CLI ``--implementation-track`` spec."""

        return implementation_supervisor_compact_track_spec(
            name=self.name,
            script_path=self.script_path,
            state_dir=self.state_dir,
            state_prefix=self.state_prefix,
        )

    def track_spec(self) -> str:
        """Return the expanded supervisor track spec with log and PID paths."""

        return implementation_supervisor_track_spec(
            name=self.name,
            script_path=self.script_path,
            state_dir=self.state_dir,
            state_prefix=self.state_prefix,
        )


@dataclass(frozen=True)
class ImplementationSupervisorNamespaceTrackSpec:
    """Minimal namespace-based inputs for one implementation-supervisor track."""

    name: str
    script_path: Path | str
    namespace: str
    state_prefix: str | None = None


def implementation_supervisor_namespace_track_config(
    *,
    name: str,
    script_path: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    state_prefix: str | None = None,
) -> ImplementationSupervisorTrackConfig:
    """Return a track config using the standard namespace state directory."""

    return ImplementationSupervisorTrackConfig(
        name=name,
        script_path=script_path,
        state_dir=namespace_paths.state_dir,
        state_prefix=state_prefix or namespace_paths.namespace,
    )


def _implementation_supervisor_namespace_track_spec(
    spec: (
        ImplementationSupervisorNamespaceTrackSpec
        | tuple[str, Path | str, str]
        | tuple[str, Path | str, str, str]
    ),
) -> ImplementationSupervisorNamespaceTrackSpec:
    if isinstance(spec, ImplementationSupervisorNamespaceTrackSpec):
        return spec
    if len(spec) == 3:
        name, script_path, namespace = spec
        state_prefix = None
    elif len(spec) == 4:
        name, script_path, namespace, state_prefix = spec
    else:
        raise ValueError(
            "namespace track specs must have NAME|SCRIPT|NAMESPACE or "
            "NAME|SCRIPT|NAMESPACE|STATE_PREFIX"
        )
    return ImplementationSupervisorNamespaceTrackSpec(
        name=name,
        script_path=script_path,
        namespace=namespace,
        state_prefix=state_prefix,
    )


def implementation_supervisor_namespace_track_configs(
    *,
    repo_root: Path | str,
    track_specs: Sequence[
        ImplementationSupervisorNamespaceTrackSpec
        | tuple[str, Path | str, str]
        | tuple[str, Path | str, str, str]
    ],
    data_root: Path | str = "data",
) -> tuple[ImplementationSupervisorTrackConfig, ...]:
    """Return implementation-supervisor track configs from namespace-based specs."""

    from .wrapper_utils import agent_supervisor_namespace_paths

    return tuple(
        implementation_supervisor_namespace_track_config(
            name=resolved_spec.name,
            script_path=resolved_spec.script_path,
            namespace_paths=agent_supervisor_namespace_paths(
                repo_root,
                resolved_spec.namespace,
                data_root=data_root,
            ),
            state_prefix=resolved_spec.state_prefix,
        )
        for resolved_spec in (
            _implementation_supervisor_namespace_track_spec(spec) for spec in track_specs
        )
    )


@dataclass(frozen=True)
class ConfiguredMultiSupervisorCliRunner:
    """Project-bound CLI argv for launching the reusable multi-supervisor runner."""

    argv: tuple[str, ...]

    def args(self) -> list[str]:
        """Return the configured runner argv as a mutable list."""

        return list(self.argv)

    def run(self, extra_argv: Sequence[str] | None = None) -> int:
        """Run the multi-supervisor CLI with configured args plus any overrides."""

        return main([*self.argv, *(extra_argv or ())])

    def run_cli(self, argv: Sequence[str] | None = None) -> int:
        """Run from a wrapper CLI, defaulting overrides from ``sys.argv``."""

        return self.run(sys.argv[1:] if argv is None else argv)


@dataclass(frozen=True)
class ConfiguredMultiSupervisorLauncher:
    """Prepared launcher for a configured multi-supervisor runner."""

    runner: ConfiguredMultiSupervisorCliRunner
    env_defaults: tuple[tuple[str, str], ...] = ()
    prepare_environment: Callable[[], None] | None = None

    def args(self) -> list[str]:
        """Return the configured runner argv as a mutable list."""

        return self.runner.args()

    def prepare(self) -> None:
        """Apply environment defaults and run the optional preparation callback."""

        if self.env_defaults:
            apply_env_defaults(dict(self.env_defaults))
        if self.prepare_environment is not None:
            self.prepare_environment()

    def run(self, extra_argv: Sequence[str] | None = None) -> int:
        """Prepare the environment and run the configured multi-supervisor CLI."""

        self.prepare()
        return self.runner.run(extra_argv)

    def run_cli(self, argv: Sequence[str] | None = None) -> int:
        """Prepare and run from a wrapper CLI, defaulting overrides from ``sys.argv``."""

        self.prepare()
        return self.runner.run_cli(argv)


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
    """Parse ``NAME|SCRIPT|LOG|SUPERVISOR_PID|DAEMON_PID[|SUPERVISOR_STATUS]`` specs."""

    rendered = spec.format(stamp=stamp) if stamp else spec
    parts = rendered.split("|")
    if len(parts) not in {5, 6} or not parts[0].strip():
        raise ValueError(
            "track specs must have NAME|SCRIPT|LOG|SUPERVISOR_PID|DAEMON_PID"
            "[|SUPERVISOR_STATUS]"
        )
    name, script, log, supervisor_pid, daemon_pid = (part.strip() for part in parts[:5])
    supervisor_status = parts[5].strip() if len(parts) == 6 else ""
    return SupervisorTrack(
        name=name,
        script_path=Path(script),
        log_path=Path(log),
        supervisor_pid_path=Path(supervisor_pid),
        daemon_pid_path=Path(daemon_pid),
        supervisor_status_path=Path(supervisor_status) if supervisor_status else None,
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


def implementation_supervisor_compact_track_spec(
    *,
    name: str,
    script_path: Path | str,
    state_dir: Path | str,
    state_prefix: str,
) -> str:
    """Return a compact ``NAME|SCRIPT|STATE_DIR|STATE_PREFIX`` implementation-track spec."""

    return "|".join(
        (
            str(name),
            Path(script_path).as_posix(),
            Path(state_dir).as_posix(),
            str(state_prefix),
        )
    )


def implementation_supervisor_compact_track_specs(
    track_configs: Sequence[ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]],
) -> tuple[str, ...]:
    """Return compact implementation-track specs from structured track configs."""

    specs: list[str] = []
    for config in track_configs:
        if isinstance(config, ImplementationSupervisorTrackConfig):
            specs.append(config.compact_spec())
            continue
        name, script_path, state_dir, state_prefix = config
        specs.append(
            implementation_supervisor_compact_track_spec(
                name=name,
                script_path=script_path,
                state_dir=state_dir,
                state_prefix=state_prefix,
            )
        )
    return tuple(specs)


def parse_implementation_track_spec(spec: str, *, stamp: str = "") -> SupervisorTrack:
    """Parse ``NAME|SCRIPT|STATE_DIR|STATE_PREFIX`` implementation-track specs."""

    parts = [part.strip() for part in spec.split("|")]
    if len(parts) != 4 or not parts[0]:
        raise ValueError("implementation track specs must have NAME|SCRIPT|STATE_DIR|STATE_PREFIX")
    name, script, state_dir, state_prefix = parts
    track = parse_track_spec(
        implementation_supervisor_track_spec(
            name=name,
            script_path=script,
            state_dir=state_dir,
            state_prefix=state_prefix,
        ),
        stamp=stamp,
    )
    return SupervisorTrack(
        name=track.name,
        script_path=track.script_path,
        log_path=track.log_path,
        supervisor_pid_path=track.supervisor_pid_path,
        daemon_pid_path=track.daemon_pid_path,
        supervisor_status_path=Path(state_dir) / f"{state_prefix}_supervisor_status.json",
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


def _env_default_items(
    defaults: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(defaults, Mapping):
        iterable = defaults.items()
    else:
        iterable = defaults
    return tuple((str(name), str(value)) for name, value in iterable)


def _env_default_value(value: bool | int | str) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def implementation_multi_supervisor_env_defaults(
    *,
    python_unbuffered: bool | int | str | None = True,
    codex_merge_resolver_timeout_seconds: int | str | None = 60,
    prefer_copilot_merge_resolver: bool | int | str | None = None,
) -> dict[str, str]:
    """Return reusable environment defaults for long-running implementation supervisors."""

    defaults: dict[str, str] = {}
    if python_unbuffered is not None:
        defaults["PYTHONUNBUFFERED"] = _env_default_value(python_unbuffered)
    if codex_merge_resolver_timeout_seconds is not None:
        defaults["CODEX_MERGE_RESOLVER_TIMEOUT_SECONDS"] = _env_default_value(
            codex_merge_resolver_timeout_seconds
        )
    if prefer_copilot_merge_resolver is not None:
        defaults["PREFER_COPILOT_MERGE_RESOLVER"] = _env_default_value(
            prefer_copilot_merge_resolver
        )
    return defaults


def build_configured_multi_supervisor_cli_runner(
    *,
    repo_root: Path | str,
    duration_seconds: float | int | str = 28800.0,
    duration_seconds_env_var: str = "",
    heartbeat_interval_seconds: float | int | str | None = None,
    supervisor_status_stale_seconds: float | int | str | None = None,
    stop_grace_seconds: float | int | str | None = None,
    stamp: str = "",
    stamp_env_var: str = "",
    master_dir: Path | str = Path("data/agent_supervisor"),
    master_log: Path | str | None = None,
    master_pid_path: Path | str | None = None,
    label: str = "multi-supervisor",
    python_executable: str = "python3",
    implementation_supervisor_defaults: bool = False,
    implementation_supervisor_command: str = "",
    implementation_supervisor_llm_merge_resolver_command: str = "",
    implementation_tracks: Sequence[str] = (),
    implementation_track_configs: Sequence[
        ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]
    ] = (),
    tracks: Sequence[str] = (),
    common_args: Sequence[str] = (),
    detach: bool = False,
) -> ConfiguredMultiSupervisorCliRunner:
    """Build reusable multi-supervisor CLI argv from project-specific tracks."""

    effective_duration_seconds = (
        env_str(duration_seconds_env_var, str(duration_seconds))
        if duration_seconds_env_var
        else duration_seconds
    )
    effective_stamp_default = stamp or utc_run_stamp()
    effective_stamp = (
        env_str(stamp_env_var, effective_stamp_default)
        if stamp_env_var
        else effective_stamp_default
    )
    argv = [
        "--repo-root",
        str(repo_root),
        "--duration-seconds",
        str(effective_duration_seconds),
        "--stamp",
        effective_stamp,
        "--master-dir",
        str(master_dir),
        "--label",
        label,
        "--python-executable",
        python_executable,
    ]
    if heartbeat_interval_seconds is not None:
        argv.extend(["--heartbeat-interval-seconds", str(heartbeat_interval_seconds)])
    if supervisor_status_stale_seconds is not None:
        argv.extend(["--supervisor-status-stale-seconds", str(supervisor_status_stale_seconds)])
    if stop_grace_seconds is not None:
        argv.extend(["--stop-grace-seconds", str(stop_grace_seconds)])
    if master_log is not None:
        argv.extend(["--master-log", str(master_log)])
    if master_pid_path is not None:
        argv.extend(["--master-pid-path", str(master_pid_path)])
    if implementation_supervisor_defaults:
        argv.append("--implementation-supervisor-defaults")
    if implementation_supervisor_command:
        argv.extend(["--implementation-supervisor-command", implementation_supervisor_command])
    if implementation_supervisor_llm_merge_resolver_command:
        argv.extend(
            [
                "--implementation-supervisor-llm-merge-resolver-command",
                implementation_supervisor_llm_merge_resolver_command,
            ]
        )
    for track in tracks:
        argv.extend(["--track", str(track)])
    for track in implementation_tracks:
        argv.extend(["--implementation-track", str(track)])
    for track in implementation_supervisor_compact_track_specs(implementation_track_configs):
        argv.extend(["--implementation-track", str(track)])
    for arg in common_args:
        argv.extend(["--common-arg", str(arg)])
    if detach:
        argv.append("--detach")
    return ConfiguredMultiSupervisorCliRunner(tuple(argv))


def build_configured_multi_supervisor_launcher(
    *,
    repo_root: Path | str,
    duration_seconds: float | int | str = 28800.0,
    duration_seconds_env_var: str = "",
    heartbeat_interval_seconds: float | int | str | None = None,
    supervisor_status_stale_seconds: float | int | str | None = None,
    stop_grace_seconds: float | int | str | None = None,
    stamp: str = "",
    stamp_env_var: str = "",
    master_dir: Path | str = Path("data/agent_supervisor"),
    master_log: Path | str | None = None,
    master_pid_path: Path | str | None = None,
    label: str = "multi-supervisor",
    python_executable: str = "python3",
    implementation_supervisor_defaults: bool = False,
    implementation_supervisor_command: str = "",
    implementation_supervisor_llm_merge_resolver_command: str = "",
    implementation_tracks: Sequence[str] = (),
    implementation_track_configs: Sequence[
        ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]
    ] = (),
    tracks: Sequence[str] = (),
    common_args: Sequence[str] = (),
    detach: bool = False,
    env_defaults: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredMultiSupervisorLauncher:
    """Build a prepared multi-supervisor launcher from project-specific inputs."""

    return ConfiguredMultiSupervisorLauncher(
        runner=build_configured_multi_supervisor_cli_runner(
            repo_root=repo_root,
            duration_seconds=duration_seconds,
            duration_seconds_env_var=duration_seconds_env_var,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            supervisor_status_stale_seconds=supervisor_status_stale_seconds,
            stop_grace_seconds=stop_grace_seconds,
            stamp=stamp,
            stamp_env_var=stamp_env_var,
            master_dir=master_dir,
            master_log=master_log,
            master_pid_path=master_pid_path,
            label=label,
            python_executable=python_executable,
            implementation_supervisor_defaults=implementation_supervisor_defaults,
            implementation_supervisor_command=implementation_supervisor_command,
            implementation_supervisor_llm_merge_resolver_command=(
                implementation_supervisor_llm_merge_resolver_command
            ),
            implementation_tracks=implementation_tracks,
            implementation_track_configs=implementation_track_configs,
            tracks=tracks,
            common_args=common_args,
            detach=detach,
        ),
        env_defaults=_env_default_items(env_defaults),
        prepare_environment=prepare_environment,
    )


def build_repo_implementation_multi_supervisor_launcher(
    *,
    repo_root: Path | str,
    implementation_track_configs: Sequence[
        ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]
    ],
    resolver_script_path: Path | str = "",
    implementation_supervisor_command: str = "",
    implementation_supervisor_llm_merge_resolver_command: str = "",
    duration_seconds: float | int | str = 28800.0,
    duration_seconds_env_var: str = "DURATION_SECONDS",
    heartbeat_interval_seconds: float | int | str | None = None,
    supervisor_status_stale_seconds: float | int | str | None = None,
    stop_grace_seconds: float | int | str | None = None,
    stamp: str = "",
    stamp_env_var: str = "STAMP",
    master_dir: Path | str = Path("data/agent_supervisor"),
    master_log: Path | str | None = None,
    master_pid_path: Path | str | None = None,
    label: str = "implementation supervisor run",
    python_executable: str = "python3",
    common_args: Sequence[str] = (),
    detach: bool = False,
    env_defaults: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    prepare_environment: Callable[[], None] | None = None,
    runtime_package_names: Sequence[Path | str] | None = ("ipfs_accelerate", "ipfs_datasets"),
    runtime_external_dir: Path | str = "external",
    runtime_env_var: str = "PYTHONPATH",
) -> ConfiguredMultiSupervisorLauncher:
    """Build a repo-local implementation multi-supervisor launcher."""

    from .llm_merge_resolver_fallback import llm_merge_resolver_fallback_command
    from .wrapper_utils import build_repo_runtime_environment_callbacks, repo_script_command

    llm_merge_resolver_command = implementation_supervisor_llm_merge_resolver_command
    if not llm_merge_resolver_command and resolver_script_path:
        llm_merge_resolver_command = repo_script_command(repo_root, resolver_script_path)
    if not llm_merge_resolver_command:
        llm_merge_resolver_command = llm_merge_resolver_fallback_command(
            python_executable=python_executable
        )
    effective_prepare_environment = prepare_environment
    if effective_prepare_environment is None and runtime_package_names is not None:
        runtime_environment = build_repo_runtime_environment_callbacks(
            repo_root,
            package_names=runtime_package_names,
            external_dir=runtime_external_dir,
            env_var=runtime_env_var,
        )
        effective_prepare_environment = runtime_environment.ensure_pythonpath
    return build_configured_multi_supervisor_launcher(
        repo_root=repo_root,
        duration_seconds=duration_seconds,
        duration_seconds_env_var=duration_seconds_env_var,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        supervisor_status_stale_seconds=supervisor_status_stale_seconds,
        stop_grace_seconds=stop_grace_seconds,
        stamp=stamp,
        stamp_env_var=stamp_env_var,
        master_dir=master_dir,
        master_log=master_log,
        master_pid_path=master_pid_path,
        label=label,
        python_executable=python_executable,
        implementation_supervisor_defaults=True,
        implementation_supervisor_command=implementation_supervisor_command,
        implementation_supervisor_llm_merge_resolver_command=llm_merge_resolver_command,
        implementation_track_configs=implementation_track_configs,
        common_args=common_args,
        detach=detach,
        env_defaults=env_defaults,
        prepare_environment=effective_prepare_environment,
    )


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
    objective_refill_timeout_seconds: int = 600,
    objective_surplus_findings_per_goal: int = 6,
    objective_surplus_min_terms_per_todo: int = 4,
    codebase_scan_cooldown_seconds: int = 900,
    llm_merge_resolver_timeout_seconds: int = 1800,
) -> list[str]:
    """Return standard common args for long-running implementation supervisors."""

    effective_llm_merge_resolver_command = llm_merge_resolver_command or implementation_command
    args = [
        "--implement",
        "--objective-refill-scan",
        "--codebase-refill-scan",
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
        "--objective-refill-timeout-seconds",
        str(objective_refill_timeout_seconds),
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


def _remove_stale_pid_marker_if_unchanged(pid_path: Path, stale_pid: int) -> bool:
    """Remove a dead PID marker only if it still names the dead process."""

    current_pid = read_pid_file(pid_path)
    if current_pid != stale_pid or pid_alive(current_pid):
        return False
    return remove_runtime_marker(pid_path)


def daemon_pid_health_fields(
    pid_path: Path,
    *,
    cleanup_stale_marker: bool = False,
) -> dict[str, object]:
    """Return heartbeat fields for a managed daemon PID marker."""

    daemon_pid = read_pid_file(pid_path)
    if not daemon_pid:
        return {"daemon_pid": None, "daemon_status": "missing"}
    if pid_alive(daemon_pid):
        return {"daemon_pid": daemon_pid, "daemon_status": "live"}
    removed = False
    if cleanup_stale_marker:
        removed = _remove_stale_pid_marker_if_unchanged(pid_path, daemon_pid)
    return {
        "daemon_pid": None,
        "daemon_status": "stale",
        "stale_daemon_pid": daemon_pid,
        "removed_stale_daemon_pid_file": removed,
    }


def format_daemon_heartbeat_fields(fields: Mapping[str, object]) -> str:
    """Return compact daemon health fields for master heartbeat logs."""

    daemon_pid = fields.get("daemon_pid")
    parts = [f"daemon_pid={daemon_pid if daemon_pid else 'unknown'}"]
    stale_pid = fields.get("stale_daemon_pid")
    if stale_pid:
        parts.append(f"stale_daemon_pid={stale_pid}")
    status = fields.get("daemon_status")
    if status and status != "live":
        parts.append(f"daemon_status={status}")
    if fields.get("removed_stale_daemon_pid_file"):
        parts.append("removed_stale_daemon_pid_file=true")
    return " ".join(parts)


def _read_json_dict(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_status_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _inferred_supervisor_status_path(track: SupervisorTrack) -> Path | None:
    if track.supervisor_status_path is not None:
        return track.supervisor_status_path
    name = track.supervisor_pid_path.name
    suffix = "_supervisor.pid"
    if name.endswith(suffix):
        prefix = name[: -len(suffix)]
        return track.supervisor_pid_path.with_name(f"{prefix}_supervisor_status.json")
    return None


def _relative_or_absolute_path(repo_root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else repo_root / path


def supervisor_status_health_fields(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    stale_seconds: float,
) -> dict[str, object]:
    """Return heartbeat fields for the wrapper supervisor status file."""

    status_path = _inferred_supervisor_status_path(track)
    if status_path is None:
        return {"supervisor_status": "untracked"}
    payload = _read_json_dict(status_path)
    if not payload:
        return {
            "supervisor_status": "missing",
            "supervisor_status_path": str(status_path),
        }
    updated_at = _parse_status_timestamp(payload.get("updated_at") or payload.get("heartbeat_at"))
    if updated_at is None:
        return {
            "supervisor_status": "unknown",
            "supervisor_status_path": str(status_path),
        }
    age_seconds = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
    if stale_seconds <= 0 or age_seconds <= stale_seconds:
        return {
            "supervisor_status": "live",
            "supervisor_status_path": str(status_path),
            "supervisor_status_age_seconds": round(age_seconds, 1),
        }

    child_state_path = _relative_or_absolute_path(
        repo_root,
        payload.get("current_status_path") or payload.get("progress_path") or payload.get("state_path"),
    )
    child_state = _read_json_dict(child_state_path)
    active_task_id = str(child_state.get("active_task_id") or "").strip()
    implementation_in_progress = bool(child_state.get("implementation_in_progress"))
    active_child = bool(active_task_id or implementation_in_progress)
    return {
        "supervisor_status": "stale_active" if active_child else "stale",
        "supervisor_status_path": str(status_path),
        "supervisor_status_age_seconds": round(age_seconds, 1),
        "supervisor_active_task_id": active_task_id,
        "supervisor_child_in_progress": implementation_in_progress,
        "restart_supervisor": not active_child,
    }


def format_supervisor_status_fields(fields: Mapping[str, object]) -> str:
    """Return compact supervisor health fields for master heartbeat logs."""

    status = fields.get("supervisor_status")
    if not status or status == "untracked":
        return ""
    parts = [f"supervisor_status={status}"]
    age = fields.get("supervisor_status_age_seconds")
    if age is not None:
        parts.append(f"supervisor_status_age_seconds={age}")
    active_task_id = fields.get("supervisor_active_task_id")
    if active_task_id:
        parts.append(f"supervisor_active_task_id={active_task_id}")
    if fields.get("restart_supervisor"):
        parts.append("restart_supervisor=true")
    return " ".join(parts)


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
    supervisor_status_stale_seconds: float = 600.0,
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
                daemon_fields = daemon_pid_health_fields(
                    resolved.daemon_pid_path,
                    cleanup_stale_marker=True,
                )
                supervisor_fields = supervisor_status_health_fields(
                    resolved,
                    repo_root=resolved_repo_root,
                    stale_seconds=float(supervisor_status_stale_seconds),
                )
                if process is not None and process.poll() is None and pid_alive(process.pid):
                    supervisor_summary = format_supervisor_status_fields(supervisor_fields)
                    heartbeat_parts = [
                        f"heartbeat {track.name} supervisor_pid={process.pid}",
                        format_daemon_heartbeat_fields(daemon_fields),
                    ]
                    if supervisor_summary:
                        heartbeat_parts.append(supervisor_summary)
                    _emit(
                        output,
                        " ".join(heartbeat_parts),
                    )
                    if supervisor_fields.get("restart_supervisor"):
                        daemon_pid = daemon_fields.get("daemon_pid")
                        _emit(
                            output,
                            (
                                f"restarting stale {track.name} supervisor old_pid={process.pid} "
                                f"daemon_pid={daemon_pid or 'unknown'} "
                                f"supervisor_status_age_seconds="
                                f"{supervisor_fields.get('supervisor_status_age_seconds')}"
                            ),
                        )
                        _terminate_pid(process.pid, grace_seconds=stop_grace_seconds)
                        if isinstance(daemon_pid, int):
                            _terminate_pid(daemon_pid, grace_seconds=stop_grace_seconds)
                        try:
                            process.wait(timeout=max(0.1, stop_grace_seconds))
                        except subprocess.TimeoutExpired:
                            pass
                        processes[track.name] = start_track(
                            track,
                            repo_root=resolved_repo_root,
                            common_args=common_args,
                            python_executable=python_executable,
                            output=output,
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
    parser.add_argument("--supervisor-status-stale-seconds", type=float, default=600.0)
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
        "--implementation-supervisor-objective-refill-timeout-seconds",
        type=int,
        default=_env_int("OBJECTIVE_REFILL_TIMEOUT_SECONDS", 600),
    )
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
                objective_refill_timeout_seconds=args.implementation_supervisor_objective_refill_timeout_seconds,
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
            supervisor_status_stale_seconds=args.supervisor_status_stale_seconds,
            stop_grace_seconds=args.stop_grace_seconds,
            python_executable=args.python_executable,
            master_pid_path=master_pid,
            label=args.label,
            output=output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
