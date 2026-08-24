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
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UNIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}\Z")
_STRICT_PID = re.compile(r"[1-9][0-9]*\n\Z")
_QUACK_TOKEN_ENVIRONMENT_NAME = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
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


def _validated_absolute_path(value: Path, *, label: str) -> Path:
    """Return one normalized absolute path without following its final leaf."""

    path = Path(value).expanduser()
    if "\x00" in os.fspath(path):
        raise DurableProcessError(f"{label} contains a NUL byte")
    if not path.is_absolute():
        raise DurableProcessError(f"{label} must be an absolute path")
    # ``abspath`` removes redundant path components without resolving a
    # possibly preplaced symlink at the authority-bearing final leaf.
    return Path(os.path.abspath(os.fspath(path)))


def _strict_pid_file_value(path: Path) -> int:
    """Read a small, owner-only, single-link regular PID file safely."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif path.is_symlink():
        raise DurableProcessError(
            "forking PID file must not be a symbolic link"
        )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DurableProcessError(
            "forking PID file could not be read safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DurableProcessError(
                "forking PID file is not a single-link regular file"
            )
        if metadata.st_uid != os.geteuid():
            raise DurableProcessError(
                "forking PID file is not owned by the launching user"
            )
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise DurableProcessError(
                "forking PID file is not owner-only mode 0600"
            )
        content = os.read(descriptor, 64)
        if os.read(descriptor, 1):
            raise DurableProcessError("forking PID file is too large")
        try:
            observed = os.lstat(path)
        except OSError as exc:
            raise DurableProcessError(
                "forking PID file changed while it was read"
            ) from exc
        if (
            (metadata.st_dev, metadata.st_ino)
            != (observed.st_dev, observed.st_ino)
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise DurableProcessError(
                "forking PID file changed while it was read"
            )
    finally:
        os.close(descriptor)
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DurableProcessError(
            "forking PID file is not strict ASCII"
        ) from exc
    if not _STRICT_PID.fullmatch(text):
        raise DurableProcessError(
            "forking PID file does not contain one strict positive PID"
        )
    return int(text)


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


def _best_effort_stop(
    runner: _Runner,
    systemctl: str,
    service_unit: str,
    *,
    timeout_seconds: float,
) -> None:
    """Try to stop one exact unit without hiding the launch failure."""

    try:
        _run(
            runner,
            [systemctl, "--user", "stop", service_unit],
            timeout_seconds=timeout_seconds,
        )
    except DurableProcessError:
        pass


def launch_systemd_user_service(
    command: Sequence[str],
    *,
    unit_name: str,
    working_directory: Path,
    log_path: Path,
    environment: Optional[Mapping[str, str]] = None,
    forking_pid_file: Optional[Path] = None,
    startup_timeout_seconds: float = 15.0,
    systemd_run_path: Optional[str] = None,
    systemctl_path: Optional[str] = None,
    runner: _Runner = subprocess.run,
) -> DurableProcessLaunch:
    """Launch and verify one transient systemd user service.

    By default, the returned receipt is created only after ``systemd-run``
    reports that the command reached ``exec`` and ``systemctl`` proves a live
    main PID.  Supplying ``forking_pid_file`` opts into ``Type=forking`` for a
    command that launches its durable child and then exits.  That path also
    configures ``PIDFile=`` with ``GuessMainPID=no`` and requires the service
    manager's main PID to equal a strict read of the absolute PID file.

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
    pid_file_path = (
        _validated_absolute_path(
            forking_pid_file,
            label="forking PID file",
        )
        if forking_pid_file is not None
        else None
    )
    if pid_file_path is not None and any(
        name == _QUACK_TOKEN_ENVIRONMENT_NAME
        for name, _value in environment_items
    ):
        raise DurableProcessError(
            "forking service must obtain its Quack token from the one-use vault"
        )
    output_path = Path(log_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    systemd_run = _resolve_tool("systemd-run", systemd_run_path)
    systemctl = _resolve_tool("systemctl", systemctl_path)
    if pid_file_path is not None:
        preflight = _run(
            runner,
            [
                systemctl,
                "--user",
                "show",
                "--no-pager",
                "--property=LoadState",
                "--property=ActiveState",
                service_unit,
            ],
            timeout_seconds=startup_timeout_seconds,
        )
        preflight_properties = (
            _service_properties(preflight.stdout)
            if preflight.returncode == 0
            else {}
        )
        if not (
            preflight_properties.get("LoadState") == "not-found"
            and preflight_properties.get("ActiveState") == "inactive"
        ):
            raise DurableProcessError(
                "forking service unit must be absent before launch"
            )
    launch_command = [
        systemd_run,
        "--user",
        "--quiet",
        f"--unit={service_unit}",
        "--collect",
        (
            "--service-type=forking"
            if pid_file_path is not None
            else "--service-type=exec"
        ),
        f"--working-directory={cwd}",
        "--property=StandardInput=null",
        f"--property=StandardOutput=append:{output_path}",
        f"--property=StandardError=append:{output_path}",
        "--property=Restart=no",
        "--property=KillMode=mixed",
        "--property=TimeoutStopSec=90s",
        "--property=SuccessExitStatus=143 SIGTERM",
    ]
    if pid_file_path is not None:
        launch_command.extend(
            (
                f"--property=PIDFile={pid_file_path}",
                "--property=GuessMainPID=no",
            )
        )
    for name, value in environment_items:
        launch_command.append(f"--setenv={name}={value}")
    launch_command.extend(("--", *command_argv))

    try:
        launched = _run(
            runner,
            launch_command,
            timeout_seconds=startup_timeout_seconds,
        )
    except DurableProcessError:
        if pid_file_path is not None:
            # A timed-out Type=forking start can have created the unit even
            # though ``systemd-run`` did not return its outcome.
            _best_effort_stop(
                runner,
                systemctl,
                service_unit,
                timeout_seconds=startup_timeout_seconds,
            )
        raise
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
        "--property=Type",
        "--property=PIDFile",
        "--property=GuessMainPID",
        service_unit,
    ]
    try:
        inspected = _run(
            runner,
            inspect_command,
            timeout_seconds=startup_timeout_seconds,
        )
    except DurableProcessError:
        _best_effort_stop(
            runner,
            systemctl,
            service_unit,
            timeout_seconds=startup_timeout_seconds,
        )
        raise
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
    inspection_valid = active_state == "active" and pid > 0
    if pid_file_path is not None:
        inspection_valid = inspection_valid and pid > 1 and (
            properties.get("Type") == "forking"
            and properties.get("PIDFile") == str(pid_file_path)
            and properties.get("GuessMainPID") == "no"
        )
        if inspection_valid:
            try:
                inspection_valid = _strict_pid_file_value(pid_file_path) == pid
            except DurableProcessError:
                inspection_valid = False
        if inspection_valid:
            try:
                reinspected = _run(
                    runner,
                    inspect_command,
                    timeout_seconds=startup_timeout_seconds,
                )
            except DurableProcessError:
                inspection_valid = False
            else:
                reobserved = (
                    _service_properties(reinspected.stdout)
                    if reinspected.returncode == 0
                    else {}
                )
                inspection_valid = (
                    reobserved.get("ActiveState") == "active"
                    and reobserved.get("MainPID") == str(pid)
                    and reobserved.get("Type") == "forking"
                    and reobserved.get("PIDFile") == str(pid_file_path)
                    and reobserved.get("GuessMainPID") == "no"
                )
                if inspection_valid:
                    active_state = reobserved.get("ActiveState", active_state)
                    sub_state = reobserved.get("SubState", sub_state)
    if not inspection_valid:
        _best_effort_stop(
            runner,
            systemctl,
            service_unit,
            timeout_seconds=startup_timeout_seconds,
        )
        detail = (inspected.stderr or inspected.stdout or "").strip()
        message = "systemd user service did not expose a live main process"
        if pid_file_path is not None:
            message = (
                "systemd forking service did not expose the exact PID-file "
                "main process"
            )
        raise DurableProcessError(message + (f": {detail}" if detail else ""))

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
        "--forking-pid-file",
        type=Path,
        default=None,
        help=(
            "Opt into a Type=forking service and require systemd's MainPID "
            "to match this absolute PID file."
        ),
    )
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
            forking_pid_file=args.forking_pid_file,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
    except DurableProcessError as exc:
        print(f"durable process launch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(launch.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
