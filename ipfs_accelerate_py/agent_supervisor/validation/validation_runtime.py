"""Fail-closed subprocess policy for supervisor validation commands.

Validation commands are reviewed shell text, but the process environment that
launches the supervisor is not part of that text.  Keep profile hooks, secrets,
and transient executable search paths outside the validation boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shlex
import shutil
import signal
import site
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

VALIDATION_PATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PATH"
VALIDATION_PYTHON_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON"
VALIDATION_PYTHONPATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH"
VALIDATION_NPM_CACHE_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_NPM_CACHE"
VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV = (
    "IPFS_ACCELERATE_AGENT_VALIDATION_PLAYWRIGHT_BROWSERS_PATH"
)
_CHILD_PYTHON_ENV = "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"
_NEUTRAL_HOME = "/nonexistent/ipfs-accelerate-validation"
_NPM_DISABLED_USER_CONFIG = "/dev/null/npmrc"
HERMETIC_VALIDATION_RUNTIME_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hermetic-validation-runtime@1"
)
_RUNTIME_ID_ENV = "IPFS_ACCELERATE_VALIDATION_RUNTIME_ID"
_CANCELLATION_ID_ENV = "IPFS_ACCELERATE_VALIDATION_CANCELLATION_ID"

# These values affect deterministic/offline validation without carrying the
# provider, wallet, registry, signing, or cloud credentials commonly present
# in a long-running supervisor environment.
VALIDATION_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "CARGO_NET_OFFLINE",
        "CI",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "IPFS_ACCEL_SKIP_CORE",
        "IPFS_KIT_DISABLE",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "NODE_ENV",
        "NPM_CONFIG_OFFLINE",
        "PIP_NO_INDEX",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "TRANSFORMERS_OFFLINE",
        "TZ",
        "WORLD_AID_EXTERNAL_CALLS_ENABLED",
        "WORLD_AID_WLD_TRANSFERS_ENABLED",
        "WORLD_ID_ENABLED",
    }
)

_SYSTEM_VALIDATION_PATHS = (
    Path("/usr/local/sbin"),
    Path("/usr/local/bin"),
    Path("/usr/sbin"),
    Path("/usr/bin"),
    Path("/sbin"),
    Path("/bin"),
)


class ValidationRuntimeError(ValueError):
    """Raised before execution when the validation runtime policy is invalid."""


class ValidationNetworkMode(str, Enum):
    """Network boundary applied to a validation process."""

    NONE = "none"


class ValidationFilesystemMode(str, Enum):
    """Filesystem boundary applied to a validation process."""

    READ_ONLY_ROOT_WORKSPACE = "read_only_root_workspace"


@dataclass(frozen=True)
class ValidationResourceBounds:
    """Portable hard limits for one validation process tree.

    Wall-clock timeout is carried by :class:`HermeticValidationRuntime`; the
    fields here are kernel-enforced limits applied before the validation image
    starts.  Values are deliberately finite by default.
    """

    cpu_seconds: int = 900
    memory_bytes: int = 2 * 1024 * 1024 * 1024
    output_file_bytes: int = 256 * 1024 * 1024
    open_files: int = 512
    processes: int = 256

    def __post_init__(self) -> None:
        for name in (
            "cpu_seconds",
            "memory_bytes",
            "output_file_bytes",
            "open_files",
            "processes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValidationRuntimeError(
                    f"validation resource bound {name} must be positive"
                )
            object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, int]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "memory_bytes": self.memory_bytes,
            "output_file_bytes": self.output_file_bytes,
            "open_files": self.open_files,
            "processes": self.processes,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "ValidationResourceBounds":
        return cls(
            cpu_seconds=int(value.get("cpu_seconds", 900)),
            memory_bytes=int(
                value.get("memory_bytes", 2 * 1024 * 1024 * 1024)
            ),
            output_file_bytes=int(
                value.get("output_file_bytes", 256 * 1024 * 1024)
            ),
            open_files=int(value.get("open_files", 512)),
            processes=int(value.get("processes", 256)),
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        details = resolved.stat()
    except OSError as exc:
        raise ValidationRuntimeError(
            f"validation toolchain entry is unavailable: {path}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValidationRuntimeError(
            f"validation toolchain entry is not executable: {path}"
        )
    _reject_writable_path(resolved, source="validation toolchain")
    hasher = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise ValidationRuntimeError(
            f"validation toolchain entry cannot be read: {resolved}"
        ) from exc
    return {
        "path": str(resolved),
        "sha256": hasher.hexdigest(),
        "size": details.st_size,
        "mode": stat.S_IMODE(details.st_mode),
    }


@dataclass(frozen=True)
class HermeticValidationRuntime:
    """Content-addressed execution boundary for one validation attempt."""

    command: str
    command_argv: tuple[str, ...]
    workspace_path: str
    repository_tree_id: str
    environment: tuple[tuple[str, str], ...]
    toolchain: tuple[tuple[str, str], ...]
    timeout_seconds: float
    cancellation_id: str
    resource_bounds: ValidationResourceBounds = field(
        default_factory=ValidationResourceBounds
    )
    network_mode: ValidationNetworkMode = ValidationNetworkMode.NONE
    filesystem_mode: ValidationFilesystemMode = (
        ValidationFilesystemMode.READ_ONLY_ROOT_WORKSPACE
    )
    isolation_executable: str = ""
    runtime_id: str = ""

    def __post_init__(self) -> None:
        command = str(self.command or "").strip()
        workspace = Path(str(self.workspace_path or ""))
        tree_id = str(self.repository_tree_id or "").strip()
        cancellation_id = str(self.cancellation_id or "").strip()
        if not command or not tree_id or not cancellation_id:
            raise ValidationRuntimeError(
                "hermetic runtime requires command, tree, and cancellation identity"
            )
        if not workspace.is_absolute():
            raise ValidationRuntimeError(
                "hermetic validation workspace must be absolute"
            )
        try:
            resolved_workspace = workspace.resolve(strict=True)
        except OSError as exc:
            raise ValidationRuntimeError(
                f"hermetic validation workspace is unavailable: {workspace}"
            ) from exc
        if not resolved_workspace.is_dir():
            raise ValidationRuntimeError(
                "hermetic validation workspace must be a directory"
            )
        argv = tuple(str(value) for value in self.command_argv)
        if not argv or any(not value for value in argv):
            raise ValidationRuntimeError(
                "hermetic validation command argv must be complete"
            )
        timeout = float(self.timeout_seconds)
        if timeout <= 0:
            raise ValidationRuntimeError(
                "hermetic validation timeout must be positive"
            )
        environment = tuple(
            sorted((str(key), str(value)) for key, value in self.environment)
        )
        if len({key for key, _value in environment}) != len(environment):
            raise ValidationRuntimeError(
                "hermetic validation environment has duplicate keys"
            )
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "command_argv", argv)
        object.__setattr__(self, "workspace_path", str(resolved_workspace))
        object.__setattr__(self, "repository_tree_id", tree_id)
        object.__setattr__(self, "environment", environment)
        object.__setattr__(
            self,
            "toolchain",
            tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in self.toolchain
                )
            ),
        )
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "cancellation_id", cancellation_id)
        object.__setattr__(
            self, "network_mode", ValidationNetworkMode(self.network_mode)
        )
        object.__setattr__(
            self,
            "filesystem_mode",
            ValidationFilesystemMode(self.filesystem_mode),
        )
        claimed = str(self.runtime_id or "").strip()
        object.__setattr__(self, "runtime_id", "")
        actual = _sha256(_canonical_json(self._identity_payload()).encode())
        if claimed and claimed != actual:
            raise ValidationRuntimeError(
                "hermetic validation runtime identity mismatch"
            )
        object.__setattr__(self, "runtime_id", actual)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": HERMETIC_VALIDATION_RUNTIME_SCHEMA,
            "command": self.command,
            "command_argv": list(self.command_argv),
            "workspace_path": self.workspace_path,
            "repository_tree_id": self.repository_tree_id,
            "environment": dict(self.environment),
            "toolchain": dict(self.toolchain),
            "timeout_seconds": self.timeout_seconds,
            "cancellation_id": self.cancellation_id,
            "resource_bounds": self.resource_bounds.to_dict(),
            "network_mode": self.network_mode.value,
            "filesystem_mode": self.filesystem_mode.value,
            "isolation_executable": self.isolation_executable,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "runtime_id": self.runtime_id}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "HermeticValidationRuntime":
        schema = str(value.get("schema") or HERMETIC_VALIDATION_RUNTIME_SCHEMA)
        if schema != HERMETIC_VALIDATION_RUNTIME_SCHEMA:
            raise ValidationRuntimeError(
                f"unsupported hermetic validation runtime schema: {schema}"
            )
        bounds = value.get("resource_bounds") or {}
        if not isinstance(bounds, Mapping):
            raise ValidationRuntimeError(
                "hermetic validation resource bounds are malformed"
            )
        return cls(
            command=str(value.get("command") or ""),
            command_argv=tuple(value.get("command_argv") or ()),
            workspace_path=str(value.get("workspace_path") or ""),
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            environment=tuple(
                (str(key), str(item))
                for key, item in dict(value.get("environment") or {}).items()
            ),
            toolchain=tuple(
                (str(key), str(item))
                for key, item in dict(value.get("toolchain") or {}).items()
            ),
            timeout_seconds=float(value.get("timeout_seconds") or 0),
            cancellation_id=str(value.get("cancellation_id") or ""),
            resource_bounds=ValidationResourceBounds.from_dict(bounds),
            network_mode=ValidationNetworkMode(
                str(value.get("network_mode") or "none")
            ),
            filesystem_mode=ValidationFilesystemMode(
                str(
                    value.get("filesystem_mode")
                    or "read_only_root_workspace"
                )
            ),
            isolation_executable=str(
                value.get("isolation_executable") or ""
            ),
            runtime_id=str(value.get("runtime_id") or ""),
        )


class ValidationCancellationToken:
    """Thread-safe cancellation token fenced by a stable identity."""

    def __init__(self, cancellation_id: str) -> None:
        identity = str(cancellation_id or "").strip()
        if not identity:
            raise ValidationRuntimeError("cancellation identity is required")
        self.cancellation_id = identity
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    def cancel(self, *, cancellation_id: str, reason: str = "cancelled") -> bool:
        """Cancel only when the caller presents the exact fencing identity."""

        if str(cancellation_id or "").strip() != self.cancellation_id:
            return False
        with self._lock:
            if self._event.is_set():
                return True
            self._reason = str(reason or "cancelled").strip() or "cancelled"
            self._event.set()
            return True

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def cancelled(self) -> bool:
        return self.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


def _reject_writable_path(path: Path, *, source: str) -> None:
    inspected = path
    while True:
        try:
            mode = inspected.stat().st_mode
        except OSError as exc:
            raise ValidationRuntimeError(
                f"{source} path cannot be inspected: {inspected}"
            ) from exc
        if mode & (stat.S_IWGRP | stat.S_IWOTH) or os.access(inspected, os.W_OK):
            relationship = "path" if inspected == path else "ancestor"
            raise ValidationRuntimeError(
                f"{source} {relationship} must not be writable by the "
                f"validation user or group: {inspected}"
            )
        parent = inspected.parent
        if parent == inspected:
            break
        inspected = parent


def _validated_path_entries(
    value: str,
    *,
    source: str,
) -> tuple[str, ...]:
    entries: list[str] = []
    for raw_entry in value.split(os.pathsep):
        if not raw_entry:
            raise ValidationRuntimeError(f"{source} contains an empty PATH entry")
        path = Path(raw_entry)
        if not path.is_absolute():
            raise ValidationRuntimeError(
                f"{source} entries must be absolute: {raw_entry!r}"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValidationRuntimeError(
                f"{source} entry is unavailable: {raw_entry!r}"
            ) from exc
        if not resolved.is_dir():
            raise ValidationRuntimeError(
                f"{source} entry is not a directory: {raw_entry!r}"
            )
        _reject_writable_path(resolved, source=source)
        rendered = str(resolved)
        if rendered not in entries:
            entries.append(rendered)
    if not entries:
        raise ValidationRuntimeError(f"{source} must contain at least one directory")
    return tuple(entries)


def validation_executable_path(
    environment: Mapping[str, object] | None = None,
) -> str:
    """Return the exact executable search path for validation children.

    The inherited ``PATH`` is never reused.  Operators that need an approved
    non-system toolchain must supply its complete absolute path list through
    :data:`VALIDATION_PATH_ENV`; otherwise only standard system directories are
    exposed.
    """

    source = os.environ if environment is None else environment
    override = str(source.get(VALIDATION_PATH_ENV) or "").strip()
    if override:
        return os.pathsep.join(
            _validated_path_entries(override, source=VALIDATION_PATH_ENV)
        )

    candidates = list(_SYSTEM_VALIDATION_PATHS)

    entries: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        try:
            _reject_writable_path(resolved, source="default validation PATH")
        except ValidationRuntimeError:
            continue
        rendered = str(resolved)
        if rendered not in entries:
            entries.append(rendered)
    if not entries:
        raise ValidationRuntimeError(
            "no trusted executable directories are available for validation"
        )
    return os.pathsep.join(entries)


def validation_python_executable(
    environment: Mapping[str, object] | None = None,
) -> str:
    """Return the canonical approved Python executable for validation commands.

    Never execute the original launcher spelling after validating only its
    target: a launcher in a writable directory could be replaced between the
    check and ``exec``.  The child instead executes the already-resolved,
    non-writable system binary.  Approved package roots are supplied
    separately by :func:`_runtime_python_path_entries`.
    """

    source = os.environ if environment is None else environment
    configured = str(source.get(VALIDATION_PYTHON_ENV) or "").strip()
    candidate = Path(configured) if configured else Path(sys.executable)
    if not candidate.is_absolute():
        raise ValidationRuntimeError(
            f"{VALIDATION_PYTHON_ENV} must be an absolute executable path"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationRuntimeError(
            f"validation Python is unavailable: {candidate}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValidationRuntimeError(
            f"validation Python is not executable: {candidate}"
        )
    _reject_writable_path(resolved, source="validation Python")
    return str(resolved)


def _known_runtime_package_roots() -> set[Path]:
    """Return package roots derived from Python installation metadata.

    This deliberately does not trust arbitrary entries already present in
    ``sys.path``.  In particular, a process-start ``PYTHONPATH`` ending in a
    directory named ``site-packages`` must not be laundered into the scrubbed
    validation environment.
    """

    candidates: set[Path] = set()
    try:
        candidates.update(Path(item) for item in site.getsitepackages())
    except (AttributeError, OSError):
        pass
    for key in ("purelib", "platlib"):
        configured = sysconfig.get_path(key)
        if configured:
            candidates.add(Path(configured))
    if os.name == "posix":
        try:
            import pwd

            account_uid = os.getuid()
            # ``unshare -Ur`` maps the invoking host account to namespace
            # uid 0. Resolve that mapping through the kernel-owned uid map so
            # the approved canonical user site remains stable without
            # trusting inherited HOME or PYTHONUSERBASE values.
            try:
                uid_map = Path("/proc/self/uid_map").read_text(
                    encoding="utf-8"
                )
            except OSError:
                uid_map = ""
            for line in uid_map.splitlines():
                fields = line.split()
                if len(fields) != 3:
                    continue
                try:
                    inside_start, outside_start, length = (
                        int(field) for field in fields
                    )
                except ValueError:
                    continue
                if inside_start <= account_uid < inside_start + length:
                    account_uid = outside_start + (account_uid - inside_start)
                    break
            account_home = Path(pwd.getpwuid(account_uid).pw_dir)
            user_base = account_home / ".local"
            for key in ("purelib", "platlib"):
                configured = sysconfig.get_path(
                    key,
                    scheme="posix_user",
                    vars={"userbase": str(user_base)},
                )
                if configured:
                    candidates.add(Path(configured))
        except (ImportError, KeyError, OSError):
            pass

    resolved: set[Path] = set()
    for candidate in candidates:
        try:
            root = candidate.resolve(strict=True)
        except OSError:
            continue
        if root.is_dir():
            resolved.add(root)
    return resolved


def _runtime_python_path_entries(
    source: Mapping[str, object],
) -> tuple[str, ...]:
    """Return explicitly approved or already-active Python package roots.

    Automatic user-site discovery remains disabled in the child.  For the
    default supervisor interpreter, retain only active ``site-packages`` and
    ``dist-packages`` directories from this process.  This supports a
    supervisor installed with ``pip --user`` without re-reading ``HOME`` or an
    inherited ``PYTHONPATH``.  A separately configured validation interpreter
    is assumed to carry its own isolated package environment unless an
    explicit validation PYTHONPATH is also supplied.
    """

    configured = str(source.get(VALIDATION_PYTHONPATH_ENV) or "").strip()
    if configured:
        entries: list[str] = []
        for raw_entry in configured.split(os.pathsep):
            if not raw_entry:
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} contains an empty entry"
                )
            path = Path(raw_entry)
            if not path.is_absolute():
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} entries must be absolute: "
                    f"{raw_entry!r}"
                )
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} entry is unavailable: "
                    f"{raw_entry!r}"
                ) from exc
            if not resolved.is_dir():
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} entry is not a directory: "
                    f"{raw_entry!r}"
                )
            _reject_writable_path(
                resolved,
                source=VALIDATION_PYTHONPATH_ENV,
            )
            rendered = str(resolved)
            if rendered not in entries:
                entries.append(rendered)
        return tuple(entries)

    if str(source.get(VALIDATION_PYTHON_ENV) or "").strip():
        return ()

    approved_roots = _known_runtime_package_roots()
    entries = []
    for raw_entry in sys.path:
        if not raw_entry:
            continue
        path = Path(raw_entry)
        if not path.is_absolute():
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir() or resolved not in approved_roots:
            continue
        rendered = str(resolved)
        if rendered not in entries:
            entries.append(rendered)
    return tuple(entries)


def _approved_directory(
    source: Mapping[str, object],
    variable: str,
) -> str | None:
    raw = str(source.get(variable) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValidationRuntimeError(f"{variable} must be an absolute directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationRuntimeError(f"{variable} is unavailable: {path}") from exc
    if not resolved.is_dir():
        raise ValidationRuntimeError(f"{variable} is not a directory: {path}")
    return str(resolved)


def build_validation_environment(
    environment: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build the complete allowlisted environment for a validation child."""

    source = os.environ if environment is None else environment
    result = {
        key: str(source[key])
        for key in sorted(VALIDATION_ENVIRONMENT_ALLOWLIST)
        if key in source and source[key] is not None
    }
    result.update(
        {
            _CHILD_PYTHON_ENV: validation_python_executable(source),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": _NEUTRAL_HOME,
            "NO_COLOR": "1",
            "NPM_CONFIG_GLOBALCONFIG": "/dev/null",
            # npm rejects loading one path in two config scopes.  A child of
            # /dev/null is both distinct from the global config and guaranteed
            # to remain unavailable, so neither scope can import host settings.
            "NPM_CONFIG_USERCONFIG": _NPM_DISABLED_USER_CONFIG,
            "PAGER": "cat",
            "PATH": validation_executable_path(source),
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "TERM": "dumb",
            "XDG_CACHE_HOME": _NEUTRAL_HOME,
            "XDG_CONFIG_HOME": _NEUTRAL_HOME,
            "XDG_DATA_HOME": _NEUTRAL_HOME,
            "XDG_STATE_HOME": _NEUTRAL_HOME,
        }
    )
    npm_cache = _approved_directory(source, VALIDATION_NPM_CACHE_ENV)
    if npm_cache is not None:
        result["NPM_CONFIG_CACHE"] = npm_cache
    playwright_browsers = _approved_directory(
        source,
        VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV,
    )
    if playwright_browsers is not None:
        result["PLAYWRIGHT_BROWSERS_PATH"] = playwright_browsers
    python_path = _runtime_python_path_entries(source)
    if python_path:
        result["PYTHONPATH"] = os.pathsep.join(python_path)
    result.setdefault("LANG", "C")
    result.setdefault("LC_ALL", "C")
    result.setdefault("PYTHONHASHSEED", "0")
    result.setdefault("TZ", "UTC")
    return result


def validation_shell_command(command: str) -> list[str]:
    """Return a non-login, non-interactive Bash invocation for reviewed text."""

    text = str(command)
    if not text.strip():
        raise ValidationRuntimeError("validation command must not be empty")
    if "`" in text or "$(" in text:
        raise ValidationRuntimeError(
            "dynamic command substitution is not permitted for validation"
        )
    try:
        lexer = shlex.shlex(
            text,
            posix=True,
            punctuation_chars=";&|()<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        leading = list(lexer)
    except ValueError as exc:
        raise ValidationRuntimeError(
            "validation command has invalid shell quoting"
        ) from exc
    if any(Path(token).name in {"bash", "sh"} for token in leading):
        raise ValidationRuntimeError(
            "nested validation shells are not permitted; provide the inner "
            "command directly"
        )
    if any(Path(token).name == "eval" for token in leading):
        raise ValidationRuntimeError(
            "dynamic shell evaluation is not permitted for validation"
        )
    guarded = (
        f"readonly {_CHILD_PYTHON_ENV}; "
        f'python() {{ "${{{_CHILD_PYTHON_ENV}}}" "$@"; }}; '
        f'python3() {{ "${{{_CHILD_PYTHON_ENV}}}" "$@"; }}; '
        f'pytest() {{ "${{{_CHILD_PYTHON_ENV}}}" -m pytest "$@"; }}; '
        "readonly -f python python3 pytest; "
        f"{text}"
    )
    return ["/bin/bash", "--noprofile", "--norc", "-c", guarded]


def validation_argv_command(command: Sequence[str]) -> list[str]:
    """Normalize an argv validation without permitting a login shell.

    Most argv validations execute directly.  Historical callers may have
    stored ``bash -lc`` or ``sh -c`` arrays; retain their command text and
    positional arguments while routing them through the same guarded Bash
    contract as string validations.  Bare Python and pytest launchers are
    pinned to the supervisor's approved interpreter because its parent
    directory is intentionally absent from the restricted executable PATH.
    """

    parts = [str(part) for part in command]
    if not parts or not parts[0]:
        raise ValidationRuntimeError("validation argv must not be empty")
    if parts[0] in {"python", "python3"}:
        return [validation_python_executable(), *parts[1:]]
    if parts[0] == "pytest":
        return [validation_python_executable(), "-m", "pytest", *parts[1:]]
    executable = Path(parts[0]).name
    if executable not in {"bash", "sh"}:
        if any(Path(part).name in {"bash", "sh"} for part in parts[1:]):
            raise ValidationRuntimeError(
                "wrapped validation shells are not permitted"
            )
        return parts

    command_index: int | None = None
    for index, argument in enumerate(parts[1:], start=1):
        if argument == "--login":
            continue
        if argument == "-c" or (
            argument.startswith("-")
            and not argument.startswith("--")
            and "c" in argument[1:]
        ):
            command_index = index + 1
            break

    if command_index is None:
        raise ValidationRuntimeError(
            "validation shell argv must provide command text with -c"
        )
    if command_index >= len(parts):
        raise ValidationRuntimeError("validation shell command text is missing")
    normalized = validation_shell_command(parts[command_index])
    normalized.extend(parts[command_index + 1 :])
    return normalized


def build_hermetic_validation_runtime(
    *,
    command: str,
    workspace_path: Path | str,
    repository_tree_id: str,
    environment: Mapping[str, object] | None = None,
    timeout_seconds: float,
    cancellation_id: str,
    resource_bounds: ValidationResourceBounds | Mapping[str, object] | None = None,
    isolation_executable: Path | str | None = None,
) -> HermeticValidationRuntime:
    """Pin every execution input used by a strict validation attempt.

    Strict execution uses Bubblewrap because an environment variable cannot
    enforce a network or filesystem boundary.  Missing namespace support is
    reported by the runner as an infrastructure failure; it never silently
    falls back to an unisolated process.
    """

    child_environment = build_validation_environment(environment)
    shell_argv = tuple(validation_shell_command(command))
    if isolation_executable is None:
        discovered = shutil.which("bwrap", path=child_environment["PATH"])
        if not discovered:
            raise ValidationRuntimeError(
                "strict hermetic validation requires bubblewrap"
            )
        isolation = Path(discovered)
    else:
        isolation = Path(isolation_executable)
    isolation_identity = _file_identity(isolation)
    bash_identity = _file_identity(Path(shell_argv[0]))
    python_identity = _file_identity(
        Path(child_environment[_CHILD_PYTHON_ENV])
    )
    path_identity = _sha256(
        _canonical_json(
            [
                {
                    "path": entry,
                    "mode": stat.S_IMODE(Path(entry).stat().st_mode),
                    "device": Path(entry).stat().st_dev,
                    "inode": Path(entry).stat().st_ino,
                }
                for entry in child_environment["PATH"].split(os.pathsep)
            ]
        ).encode()
    )
    bounds = (
        resource_bounds
        if isinstance(resource_bounds, ValidationResourceBounds)
        else ValidationResourceBounds.from_dict(resource_bounds or {})
    )
    return HermeticValidationRuntime(
        command=command,
        command_argv=shell_argv,
        workspace_path=str(Path(workspace_path).resolve()),
        repository_tree_id=repository_tree_id,
        environment=tuple(child_environment.items()),
        toolchain=(
            ("bash_path", str(bash_identity["path"])),
            ("bash_sha256", str(bash_identity["sha256"])),
            ("python_path", str(python_identity["path"])),
            ("python_sha256", str(python_identity["sha256"])),
            ("path_identity", path_identity),
            ("isolation_sha256", str(isolation_identity["sha256"])),
        ),
        timeout_seconds=timeout_seconds,
        cancellation_id=cancellation_id,
        resource_bounds=bounds,
        isolation_executable=str(isolation_identity["path"]),
    )


def hermetic_validation_command(
    runtime: HermeticValidationRuntime,
) -> list[str]:
    """Build the exact Bubblewrap argv for a pinned runtime."""

    workspace = runtime.workspace_path
    return [
        runtime.isolation_executable,
        "--die-with-parent",
        "--unshare-net",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        workspace,
        workspace,
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        workspace,
        "--",
        *runtime.command_argv,
    ]


def _bounded_limit(requested: int, resource_name: int) -> tuple[int, int]:
    _soft, hard = resource.getrlimit(resource_name)
    if hard == resource.RLIM_INFINITY:
        value = requested
    else:
        value = min(requested, int(hard))
    return value, value


def _resource_preexec(bounds: ValidationResourceBounds) -> None:
    """Apply child-only limits after fork and before exec."""

    resource.setrlimit(
        resource.RLIMIT_CPU,
        _bounded_limit(bounds.cpu_seconds, resource.RLIMIT_CPU),
    )
    resource.setrlimit(
        resource.RLIMIT_AS,
        _bounded_limit(bounds.memory_bytes, resource.RLIMIT_AS),
    )
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        _bounded_limit(bounds.output_file_bytes, resource.RLIMIT_FSIZE),
    )
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        _bounded_limit(bounds.open_files, resource.RLIMIT_NOFILE),
    )
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(
            resource.RLIMIT_NPROC,
            _bounded_limit(bounds.processes, resource.RLIMIT_NPROC),
        )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=0.25)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def run_hermetic_validation_process(
    runtime: HermeticValidationRuntime,
    *,
    cancellation_token: ValidationCancellationToken | None = None,
) -> dict[str, object]:
    """Execute a pinned runtime with hard timeout and cancellation fencing."""

    if (
        cancellation_token is not None
        and cancellation_token.cancellation_id != runtime.cancellation_id
    ):
        raise ValidationRuntimeError(
            "validation cancellation token identity mismatch"
        )
    started_at = time.time()
    environment = dict(runtime.environment)
    environment[_RUNTIME_ID_ENV] = runtime.runtime_id
    environment[_CANCELLATION_ID_ENV] = runtime.cancellation_id
    with tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8", errors="replace"
    ) as output_file:
        try:
            process = subprocess.Popen(
                hermetic_validation_command(runtime),
                cwd=runtime.workspace_path,
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=(os.name == "posix"),
                preexec_fn=(
                    (lambda: _resource_preexec(runtime.resource_bounds))
                    if os.name == "posix"
                    else None
                ),
            )
        except OSError as exc:
            return {
                "returncode": 75,
                "output": "",
                "error": f"hermetic_runtime_start_failed:{type(exc).__name__}",
                "infrastructure_failure": True,
                "runtime_id": runtime.runtime_id,
                "cancellation_id": runtime.cancellation_id,
            }

        timed_out = False
        cancelled = False
        deadline = time.monotonic() + runtime.timeout_seconds
        while process.poll() is None:
            if cancellation_token is not None and cancellation_token.is_set():
                cancelled = True
                _terminate_process_tree(process)
                break
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                timed_out = True
                _terminate_process_tree(process)
                break
            try:
                process.wait(timeout=min(0.05, remaining_seconds))
            except subprocess.TimeoutExpired:
                continue

        output_file.flush()
        output_size = output_file.tell()
        output_file.seek(0)
        output = output_file.read(runtime.resource_bounds.output_file_bytes)
        output_truncated = output_size > len(output.encode("utf-8"))

    if cancelled:
        return {
            "returncode": 130,
            "output": output,
            "output_bytes": output_size,
            "output_truncated": output_truncated,
            "cancelled": True,
            "error": (
                cancellation_token.reason
                if cancellation_token is not None
                else "cancelled"
            ),
            "runtime_id": runtime.runtime_id,
            "cancellation_id": runtime.cancellation_id,
        }
    if timed_out:
        return {
            "returncode": 124,
            "output": output,
            "output_bytes": output_size,
            "output_truncated": output_truncated,
            "timed_out": True,
            "runtime_id": runtime.runtime_id,
            "cancellation_id": runtime.cancellation_id,
        }

    returncode = int(process.returncode or 0)
    infrastructure_failure = bool(
        returncode != 0
        and (
            "bwrap:" in output
            or "Operation not permitted" in output
            or "Creating new namespace failed" in output
        )
    )
    return {
        "returncode": 75 if infrastructure_failure else returncode,
        "output": output,
        "output_bytes": output_size,
        "output_truncated": output_truncated,
        "infrastructure_failure": infrastructure_failure,
        "error": (
            "hermetic_isolation_unavailable"
            if infrastructure_failure
            else ""
        ),
        "runtime_id": runtime.runtime_id,
        "cancellation_id": runtime.cancellation_id,
        "execution_elapsed_seconds": max(0.0, time.time() - started_at),
    }
