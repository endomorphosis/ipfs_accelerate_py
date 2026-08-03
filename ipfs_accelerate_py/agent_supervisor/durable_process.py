"""Launch long-lived agent-supervisor processes under a durable host owner.

``nohup`` and a new POSIX session protect a process from terminal hangups, but
they do not move it out of a caller's service or job-control cgroup.  A caller
shutdown can therefore terminate an otherwise healthy detached supervisor.
This module launches the process as a transient user service so the user
service manager, rather than the invoking shell or automation job, owns its
lifetime.

Only explicitly named environment variables are copied from the invoking
process.  The command is always passed as an argument vector and is never
evaluated by a shell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UNIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}\Z")
_Runner = Callable[..., subprocess.CompletedProcess]


class DurableProcessError(RuntimeError):
    """A durable process could not be launched or inspected safely."""


@dataclass(frozen=True)
class DurableProcessLaunch:
    """Verified identity of one host-managed process."""

    backend: str
    unit_name: str
    pid: int
    active_state: str
    sub_state: str
    working_directory: str
    log_path: str

    def to_dict(self) -> dict:
        """Return a compact JSON-compatible launch receipt."""

        return asdict(self)


def _normalize_unit_name(value: str) -> str:
    unit_name = str(value).strip()
    if unit_name.endswith(".service"):
        unit_name = unit_name[: -len(".service")]
    if not _UNIT_NAME.fullmatch(unit_name):
        raise DurableProcessError(
            "unit name must contain only ASCII letters, digits, '.', '_', "
            "'@', and '-' and must start with a letter or digit"
        )
    return f"{unit_name}.service"


def _validated_command(command: Sequence[str]) -> tuple:
    command_argv = tuple(str(item) for item in command)
    if not command_argv:
        raise DurableProcessError("a non-empty command is required")
    if any(not item or "\x00" in item for item in command_argv):
        raise DurableProcessError(
            "command arguments must be non-empty and contain no NUL bytes"
        )
    return command_argv


def _validated_environment(
    environment: Optional[Mapping[str, str]],
) -> tuple:
    items = []
    for raw_name, raw_value in sorted((environment or {}).items()):
        name = str(raw_name)
        value = str(raw_value)
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise DurableProcessError(
                f"invalid environment variable name: {name!r}"
            )
        if "\x00" in value:
            raise DurableProcessError(
                f"environment variable {name!r} contains a NUL byte"
            )
        items.append((name, value))
    return tuple(items)


def _resolve_tool(name: str, explicit_path: Optional[str]) -> str:
    if explicit_path:
        return str(explicit_path)
    resolved = shutil.which(name)
    if not resolved:
        raise DurableProcessError(
            f"{name} is required for the systemd-user launch backend"
        )
    return resolved


def _run(
    runner: _Runner,
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess:
    try:
        return runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        # ``TimeoutExpired.__str__`` embeds the entire argv, which may include
        # explicitly forwarded environment values.  Keep failures secret-safe.
        raise DurableProcessError(
            "service-manager command timed out"
        ) from exc
    except OSError as exc:
        raise DurableProcessError(
            f"service-manager command failed: {type(exc).__name__}"
        ) from exc
    except subprocess.SubprocessError as exc:
        raise DurableProcessError(
            f"service-manager command failed: {type(exc).__name__}"
        ) from exc


def _service_properties(output: str) -> dict:
    properties = {}
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            properties[name.strip()] = value.strip()
    return properties


def launch_systemd_user_service(
    command: Sequence[str],
    *,
    unit_name: str,
    working_directory: Path,
    log_path: Path,
    environment: Optional[Mapping[str, str]] = None,
    startup_timeout_seconds: float = 15.0,
    systemd_run_path: Optional[str] = None,
    systemctl_path: Optional[str] = None,
    runner: _Runner = subprocess.run,
) -> DurableProcessLaunch:
    """Launch and verify one transient systemd user service.

    The returned receipt is created only after ``systemd-run`` reports that
    the command reached ``exec`` and ``systemctl`` proves a live main PID.
    Failure after service creation triggers a best-effort stop of that exact
    unit so an untracked process is not left behind.
    """

    if startup_timeout_seconds <= 0:
        raise DurableProcessError("startup timeout must be positive")
    service_unit = _normalize_unit_name(unit_name)
    command_argv = _validated_command(command)
    environment_items = _validated_environment(environment)
    cwd = Path(working_directory).expanduser().resolve()
    if not cwd.is_dir():
        raise DurableProcessError(
            f"working directory does not exist or is not a directory: {cwd}"
        )
    output_path = Path(log_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    systemd_run = _resolve_tool("systemd-run", systemd_run_path)
    systemctl = _resolve_tool("systemctl", systemctl_path)
    launch_command = [
        systemd_run,
        "--user",
        "--quiet",
        f"--unit={service_unit}",
        "--collect",
        "--service-type=exec",
        f"--working-directory={cwd}",
        "--property=StandardInput=null",
        f"--property=StandardOutput=append:{output_path}",
        f"--property=StandardError=append:{output_path}",
        "--property=Restart=no",
        "--property=KillMode=mixed",
        "--property=TimeoutStopSec=90s",
        "--property=SuccessExitStatus=143 SIGTERM",
    ]
    for name, value in environment_items:
        launch_command.append(f"--setenv={name}={value}")
    launch_command.extend(("--", *command_argv))

    launched = _run(
        runner,
        launch_command,
        timeout_seconds=startup_timeout_seconds,
    )
    if launched.returncode != 0:
        detail = (launched.stderr or launched.stdout or "").strip()
        raise DurableProcessError(
            "systemd user service launch failed"
            + (f": {detail}" if detail else "")
        )

    inspect_command = [
        systemctl,
        "--user",
        "show",
        "--no-pager",
        "--property=ActiveState",
        "--property=SubState",
        "--property=MainPID",
        "--property=Result",
        service_unit,
    ]
    inspected = _run(
        runner,
        inspect_command,
        timeout_seconds=startup_timeout_seconds,
    )
    properties = (
        _service_properties(inspected.stdout)
        if inspected.returncode == 0
        else {}
    )
    active_state = properties.get("ActiveState", "")
    sub_state = properties.get("SubState", "")
    try:
        pid = int(properties.get("MainPID", "0"))
    except ValueError:
        pid = 0
    if active_state != "active" or pid <= 0:
        _run(
            runner,
            [systemctl, "--user", "stop", service_unit],
            timeout_seconds=startup_timeout_seconds,
        )
        detail = (inspected.stderr or inspected.stdout or "").strip()
        raise DurableProcessError(
            "systemd user service did not expose a live main process"
            + (f": {detail}" if detail else "")
        )

    return DurableProcessLaunch(
        backend="systemd-user",
        unit_name=service_unit,
        pid=pid,
        active_state=active_state,
        sub_state=sub_state,
        working_directory=str(cwd),
        log_path=str(output_path),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the durable-launch command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch a long-running command as a transient systemd user service"
        )
    )
    parser.add_argument("--unit", required=True)
    parser.add_argument(
        "--working-directory",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Copy the named variable from this process into the service. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _passed_environment(names: Sequence[str]) -> dict:
    environment = {}
    for name in names:
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise DurableProcessError(
                f"invalid environment variable name: {name!r}"
            )
        if name not in os.environ:
            raise DurableProcessError(
                f"requested environment variable is not set: {name}"
            )
        environment[name] = os.environ[name]
    return environment


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch a service and print its verified JSON receipt."""

    args = build_arg_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        launch = launch_systemd_user_service(
            command,
            unit_name=args.unit,
            working_directory=args.working_directory,
            log_path=args.log_path,
            environment=_passed_environment(args.pass_env),
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
    except DurableProcessError as exc:
        print(f"durable process launch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(launch.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
