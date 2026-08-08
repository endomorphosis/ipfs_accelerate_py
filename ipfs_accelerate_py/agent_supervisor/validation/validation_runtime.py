"""Fail-closed subprocess policy for supervisor validation commands.

Validation commands are reviewed shell text, but the process environment that
launches the supervisor is not part of that text.  Keep profile hooks, secrets,
and transient executable search paths outside the validation boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
PROOF_REUSE_STATE_ROOT_ENV = "IPFS_PROOF_REUSE_STATE_ROOT"
VALIDATION_FILESYSTEM_BOUNDARY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "validation-filesystem-boundary@1"
)
VALIDATION_PYTHON_LAUNCHER_SHA256_ENV = (
    "IPFS_ACCELERATE_VALIDATION_PYTHON_LAUNCHER_SHA256"
)
VALIDATION_PYTHON_LAUNCHER_MODE_ENV = (
    "IPFS_ACCELERATE_VALIDATION_PYTHON_LAUNCHER_MODE"
)
VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV = (
    "IPFS_ACCELERATE_VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256"
)
VALIDATION_PYTHON_INTERPRETER_SHA256_ENV = (
    "IPFS_ACCELERATE_VALIDATION_PYTHON_INTERPRETER_SHA256"
)
VALIDATION_PYTHON_INTERPRETER_STAT_ENV = (
    "IPFS_ACCELERATE_VALIDATION_PYTHON_INTERPRETER_STAT"
)
_CHILD_PYTHON_ENV = "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"
_NEUTRAL_HOME = "/nonexistent/ipfs-accelerate-validation"
_NPM_DISABLED_USER_CONFIG = "/dev/null/npmrc"
HERMETIC_VALIDATION_RUNTIME_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hermetic-validation-runtime@1"
)
_RUNTIME_ID_ENV = "IPFS_ACCELERATE_VALIDATION_RUNTIME_ID"
_CANCELLATION_ID_ENV = "IPFS_ACCELERATE_VALIDATION_CANCELLATION_ID"
_VALIDATION_PYTHON_LAUNCHER_POLICY_BASE = (
    "ipfs_accelerate_py/agent-supervisor/"
    "nested-validation-python-launcher@1;"
    "seals=write,grow,shrink,seal;"
    "shell-startup=privileged-no-bash-env;"
    "user-site=interpreter-s-flag;"
    "pythonpath=task-local-then-approved"
)
_SEALED_VALIDATION_PYTHON_RUNNER_ATTRIBUTE = (
    "__ipfs_accelerate_sealed_validation_python__"
)
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_MINIMUM_ABI = 3
_LANDLOCK_SYSCALL_CREATE_RULESET = 444
_LANDLOCK_SYSCALL_ADD_RULE = 445
_LANDLOCK_SYSCALL_RESTRICT_SELF = 446
_LANDLOCK_WRITE_ACCESS = (
    (1 << 1)  # WRITE_FILE
    | (1 << 4)  # REMOVE_DIR
    | (1 << 5)  # REMOVE_FILE
    | (1 << 6)  # MAKE_CHAR
    | (1 << 7)  # MAKE_DIR
    | (1 << 8)  # MAKE_REG
    | (1 << 9)  # MAKE_SOCK
    | (1 << 10)  # MAKE_FIFO
    | (1 << 11)  # MAKE_BLOCK
    | (1 << 12)  # MAKE_SYM
    | (1 << 13)  # REFER (ABI 2)
    | (1 << 14)  # TRUNCATE (ABI 3)
)
VALIDATION_LANDLOCK_FAILURE_MARKER = (
    "ipfs-accelerate-validation-landlock-error:"
)

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


@dataclass(frozen=True)
class ValidationPythonLauncherReceipt:
    """Identity and isolation evidence for a nested-process Python launcher."""

    executable: str
    content_sha256: str
    interpreter_sha256: str
    interpreter_stat: str
    mode: str
    policy_sha256: str
    sealed: bool


@dataclass(frozen=True)
class ValidationFilesystemBoundaryReceipt:
    """Body-free evidence for the state-root write boundary."""

    landlock_abi: int
    policy_sha256: str

    def to_dict(self, *, applied: bool = True) -> dict[str, object]:
        return {
            "schema": VALIDATION_FILESYSTEM_BOUNDARY_SCHEMA,
            "mode": "landlock-read-only-host-v1",
            "landlock_abi": self.landlock_abi,
            "policy_sha256": self.policy_sha256,
            "applied": bool(applied),
            "proof_reuse_control_state_read_only": bool(applied),
            "proof_reuse_state_write_exception": (
                "exact-workspace-private-home-and-std-devices"
            ),
            "protected_hardlink_aliases_checked": True,
            "workspace_writable": True,
            "private_home_writable": True,
            "standard_device_nodes_writable": True,
            "proof_authoritative": False,
            "completion_authority": False,
        }


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


def validation_landlock_abi() -> int:
    """Return the host Landlock ABI, or fail before untrusted execution."""

    if not sys.platform.startswith("linux"):
        raise ValidationRuntimeError(
            "proof-reuse state validation requires Linux Landlock"
        )
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        ctypes.set_errno(0)
        abi = int(
            libc.syscall(
                _LANDLOCK_SYSCALL_CREATE_RULESET,
                None,
                0,
                _LANDLOCK_CREATE_RULESET_VERSION,
            )
        )
        error_number = ctypes.get_errno()
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise ValidationRuntimeError(
            "proof-reuse state validation Landlock probe failed"
        ) from exc
    if abi < _LANDLOCK_MINIMUM_ABI:
        detail = f"abi={abi}" if abi >= 0 else f"errno={error_number}"
        raise ValidationRuntimeError(
            "proof-reuse state validation requires Landlock ABI 3 or newer "
            f"({detail})"
        )
    return abi


def _validation_standard_device_write_paths() -> tuple[str, ...]:
    """Return host device nodes that must stay writable under the fence.

    Pytest, cargo, and many hermetic runners open ``/dev/null`` (and similar
    sinks) for logging.  Denying those nodes breaks ordinary validation even
    though they cannot forge proof-state evidence.
    """

    allowed: list[str] = []
    for candidate in ("/dev/null", "/dev/full", "/dev/zero"):
        path = Path(candidate)
        try:
            if path.is_symlink():
                continue
            details = path.lstat()
            if not stat.S_ISCHR(details.st_mode):
                continue
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        allowed.append(str(resolved))
    return tuple(allowed)


def _validation_landlock_launcher_source() -> str:
    """Render the isolated interpreter source that applies the write fence."""

    return """\
import ctypes
import json
import os
import stat as stat_module
import sys

CREATE_RULESET = 444
ADD_RULE = 445
RESTRICT_SELF = 446
CREATE_RULESET_VERSION = 1
RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
MINIMUM_ABI = 3
WRITE_ACCESS = ((1 << 1) | (1 << 4) | (1 << 5) | (1 << 6) |
                (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) |
                (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14))
# Character devices (e.g. /dev/null) only accept file-write bits under
# PATH_BENEATH; MAKE_*/REMOVE_* bits return EINVAL.
DEVICE_WRITE_ACCESS = (1 << 1) | (1 << 14)  # WRITE_FILE | TRUNCATE
FAILURE_MARKER = "ipfs-accelerate-validation-landlock-error:"

class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]

class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]

def checked(value):
    if int(value) < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return int(value)

try:
    payload = json.loads(sys.argv[1])
    if set(payload) != {"writable_paths"}:
        raise ValueError("invalid policy keys")
    writable_paths = payload["writable_paths"]
    if (
        not isinstance(writable_paths, list)
        or len(writable_paths) < 2
        or len(writable_paths) > 8
    ):
        raise ValueError("invalid writable path population")
    command = sys.argv[2:]
    if not command:
        raise ValueError("validation command is missing")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = checked(libc.syscall(CREATE_RULESET, None, 0, CREATE_RULESET_VERSION))
    if abi < MINIMUM_ABI:
        raise RuntimeError("Landlock ABI is too old")
    ruleset_attr = RulesetAttr(WRITE_ACCESS)
    ruleset_fd = checked(
        libc.syscall(
            CREATE_RULESET,
            ctypes.byref(ruleset_attr),
            ctypes.sizeof(ruleset_attr),
            0,
        )
    )
    try:
        for raw_path in writable_paths:
            if not isinstance(raw_path, str) or not os.path.isabs(raw_path):
                raise ValueError("writable path is not absolute")
            if os.path.realpath(raw_path) != raw_path:
                raise ValueError("writable path is not canonical")
            identity = os.lstat(raw_path)
            open_flags = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW
            allowed_access = WRITE_ACCESS
            if stat_module.S_ISDIR(identity.st_mode):
                open_flags |= os.O_DIRECTORY
            elif stat_module.S_ISCHR(identity.st_mode):
                allowed_access = DEVICE_WRITE_ACCESS
            elif not stat_module.S_ISREG(identity.st_mode):
                raise ValueError("writable path has unsupported file type")
            path_fd = os.open(raw_path, open_flags)
            try:
                path_attr = PathBeneathAttr(allowed_access, path_fd)
                checked(
                    libc.syscall(
                        ADD_RULE,
                        ruleset_fd,
                        RULE_PATH_BENEATH,
                        ctypes.byref(path_attr),
                        0,
                    )
                )
            finally:
                os.close(path_fd)
        checked(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
        checked(libc.syscall(RESTRICT_SELF, ruleset_fd, 0))
    finally:
        os.close(ruleset_fd)
    os.execvpe(command[0], command, os.environ)
except BaseException as exc:
    message = (FAILURE_MARKER + type(exc).__name__ + "\\n").encode(
        "ascii", errors="replace"
    )
    os.write(2, message[:256])
    os._exit(75)
"""


def _reviewed_proof_state_roots(state_root: Path) -> tuple[Path, ...]:
    """Return the current root and versioned siblings sharing its profile."""

    match = re.fullmatch(r"(?P<prefix>.+)-v\d+", state_root.name)
    if match is None:
        return (state_root,)
    prefix = str(match.group("prefix"))
    roots = [state_root]
    try:
        siblings = tuple(state_root.parent.iterdir())
    except OSError as exc:
        raise ValidationRuntimeError(
            "proof-reuse state-root siblings are unavailable"
        ) from exc
    for sibling in siblings:
        if sibling == state_root or sibling.is_symlink():
            continue
        if re.fullmatch(rf"{re.escape(prefix)}-v\d+", sibling.name) is None:
            continue
        try:
            resolved = sibling.resolve(strict=True)
        except OSError as exc:
            raise ValidationRuntimeError(
                "reviewed proof-reuse state root is unavailable"
            ) from exc
        if not resolved.is_dir():
            raise ValidationRuntimeError(
                "reviewed proof-reuse state root is not a directory"
            )
        roots.append(resolved)
    return tuple(sorted(set(roots), key=str))


def _workspace_multilink_inodes(workspace: Path) -> set[tuple[int, int]]:
    """Collect regular-file identities that have aliases outside one name."""

    identities: set[tuple[int, int]] = set()
    try:
        for directory, directory_names, file_names in os.walk(
            workspace,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            directory_names[:] = [
                name
                for name in directory_names
                if not (directory_path / name).is_symlink()
            ]
            for name in file_names:
                candidate = directory_path / name
                identity = candidate.lstat()
                if stat.S_ISREG(identity.st_mode) and identity.st_nlink > 1:
                    identities.add((int(identity.st_dev), int(identity.st_ino)))
    except OSError as exc:
        raise ValidationRuntimeError(
            "validation workspace hardlink inventory is unavailable"
        ) from exc
    return identities


def _reject_protected_state_hardlink_aliases(
    *,
    state_root: Path,
    workspace: Path,
) -> None:
    """Reject a writable worktree name aliasing proof-control evidence."""

    candidate_inodes = _workspace_multilink_inodes(workspace)
    if not candidate_inodes:
        return
    for reviewed_root in _reviewed_proof_state_roots(state_root):
        try:
            for directory, directory_names, file_names in os.walk(
                reviewed_root,
                topdown=True,
                followlinks=False,
            ):
                directory_path = Path(directory)
                if directory_path == reviewed_root:
                    # Other task worktrees are mutable implementation scratch,
                    # not receipt/control authority.  They may legitimately
                    # share compiler artifacts with this task.
                    directory_names[:] = [
                        name for name in directory_names if name != "worktrees"
                    ]
                retained_directories: list[str] = []
                for name in directory_names:
                    child = directory_path / name
                    if child.is_symlink():
                        continue
                    try:
                        workspace.relative_to(child)
                    except ValueError:
                        retained_directories.append(name)
                    else:
                        # Never compare the authorized writable subtree to
                        # itself when the configured root contains worktrees.
                        continue
                directory_names[:] = retained_directories
                for name in file_names:
                    candidate = directory_path / name
                    identity = candidate.lstat()
                    if not stat.S_ISREG(identity.st_mode):
                        continue
                    if (
                        int(identity.st_dev),
                        int(identity.st_ino),
                    ) in candidate_inodes:
                        raise ValidationRuntimeError(
                            "validation workspace aliases protected "
                            "proof-reuse state through a hardlink"
                        )
        except ValidationRuntimeError:
            raise
        except OSError as exc:
            raise ValidationRuntimeError(
                "protected proof-reuse state hardlink inventory is unavailable"
            ) from exc


def validation_readonly_state_command(
    command: Sequence[str],
    *,
    workspace_path: Path | str,
    private_home_path: Path | str,
    environment: Mapping[str, str],
) -> tuple[list[str], ValidationFilesystemBoundaryReceipt | None]:
    """Fence an exposed proof-state root against validation-process writes.

    The validation worktree and its fresh private home remain writable.  Every
    other filesystem object, including the current and reviewed historical
    proof-state roots, is readable but cannot be created, removed, renamed, or
    truncated by the command or any descendant.  Landlock is inherited across
    ``exec`` and cannot be relaxed by the restricted process.
    """

    argv = [str(value) for value in command]
    if not argv:
        raise ValidationRuntimeError("validation command must not be empty")
    state_root_text = str(
        environment.get(PROOF_REUSE_STATE_ROOT_ENV) or ""
    ).strip()
    if not state_root_text:
        return argv, None
    state_root = Path(state_root_text).resolve(strict=True)
    workspace = Path(workspace_path).resolve(strict=True)
    private_home = Path(private_home_path).resolve(strict=True)
    if not state_root.is_dir():
        raise ValidationRuntimeError(
            "proof-reuse state root is not a directory"
        )
    for writable in (workspace, private_home):
        try:
            state_root.relative_to(writable)
        except ValueError:
            pass
        else:
            raise ValidationRuntimeError(
                "proof-reuse state root overlaps a writable validation path"
            )
    _reject_protected_state_hardlink_aliases(
        state_root=state_root,
        workspace=workspace,
    )
    abi = validation_landlock_abi()
    source = _validation_landlock_launcher_source()
    executable = validation_python_executable(
        {VALIDATION_PYTHON_ENV: environment.get(_CHILD_PYTHON_ENV, "")}
    )
    writable_paths = [str(workspace), str(private_home)]
    writable_paths.extend(_validation_standard_device_write_paths())
    policy = json.dumps(
        {
            "writable_paths": writable_paths,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    receipt = ValidationFilesystemBoundaryReceipt(
        landlock_abi=abi,
        policy_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    return [executable, "-I", "-c", source, policy, *argv], receipt


def _validation_python_launcher_mode(*, sealed: bool = False) -> str:
    delivery = "sealed-memfd" if sealed else "canonical-direct"
    return f"{sys.platform}:{delivery}"


def _validation_python_launcher_policy_sha256(mode: str) -> str:
    return hashlib.sha256(
        (
            f"{_VALIDATION_PYTHON_LAUNCHER_POLICY_BASE};"
            f"delivery={mode}"
        ).encode("utf-8")
    ).hexdigest()


def _validation_python_interpreter_identity(
    executable: Path | str,
) -> tuple[str, str]:
    resolved = Path(executable).resolve(strict=True)
    identity = _file_identity(resolved)
    details = resolved.stat()
    stat_identity = _canonical_json(
        {
            "path": str(resolved),
            "device": int(details.st_dev),
            "inode": int(details.st_ino),
            "size": int(details.st_size),
            "mode": stat.S_IMODE(details.st_mode),
            "mtime_ns": int(details.st_mtime_ns),
        }
    )
    return str(identity["sha256"]), stat_identity


def sealed_validation_python_runner(runner: Any) -> Any:
    """Mark a trusted runner as requiring sealed nested-Python delivery."""

    setattr(runner, _SEALED_VALIDATION_PYTHON_RUNNER_ATTRIBUTE, True)
    return runner


def runner_requires_sealed_validation_python(runner: Any) -> bool:
    """Return whether a runner declares the sealed nested-Python contract."""

    target = getattr(runner, "__func__", runner)
    return (
        getattr(
            target,
            _SEALED_VALIDATION_PYTHON_RUNNER_ATTRIBUTE,
            False,
        )
        is True
    )


def validation_environment_for_runner(
    environment: Mapping[str, str],
    runner: Any,
) -> dict[str, str]:
    """Bind runner-specific launcher policy before scheduler cache lookup."""

    result = {str(key): str(value) for key, value in environment.items()}
    if (
        not sys.platform.startswith("linux")
        or not runner_requires_sealed_validation_python(runner)
    ):
        return result
    executable = str(result.get(_CHILD_PYTHON_ENV) or "").strip()
    if not executable:
        raise ValidationRuntimeError(
            "validation environment is missing its canonical Python"
        )
    mode = _validation_python_launcher_mode(sealed=True)
    result.update(
        {
            VALIDATION_PYTHON_LAUNCHER_MODE_ENV: mode,
            VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV: (
                _validation_python_launcher_policy_sha256(mode)
            ),
            VALIDATION_PYTHON_LAUNCHER_SHA256_ENV: hashlib.sha256(
                _validation_python_launcher_source(
                    executable=executable,
                    approved_pythonpath=result.get("PYTHONPATH", ""),
                )
            ).hexdigest(),
        }
    )
    return result


def build_validation_environment(
    environment: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build the complete allowlisted environment for a validation child."""

    source = os.environ if environment is None else environment
    python_executable = validation_python_executable(source)
    result = {
        key: str(source[key])
        for key in sorted(VALIDATION_ENVIRONMENT_ALLOWLIST)
        if key in source and source[key] is not None
    }
    result.update(
        {
            _CHILD_PYTHON_ENV: python_executable,
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
            # Nested tools commonly honor PYTHON when starting a Python
            # helper.  Pin its harmless baseline to the same canonical
            # interpreter; command runners can replace it with a sealed
            # launcher when approved package roots must survive a nested
            # PYTHONPATH replacement.
            "PYTHON": python_executable,
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
    proof_reuse_state_root = _approved_directory(
        source,
        PROOF_REUSE_STATE_ROOT_ENV,
    )
    if proof_reuse_state_root is not None:
        result[PROOF_REUSE_STATE_ROOT_ENV] = proof_reuse_state_root
    python_path = _runtime_python_path_entries(source)
    if python_path:
        result["PYTHONPATH"] = os.pathsep.join(python_path)
    interpreter_sha256, interpreter_stat = (
        _validation_python_interpreter_identity(python_executable)
    )
    launcher_mode = _validation_python_launcher_mode()
    result.update(
        {
            VALIDATION_PYTHON_LAUNCHER_MODE_ENV: launcher_mode,
            VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV: (
                _validation_python_launcher_policy_sha256(launcher_mode)
            ),
            VALIDATION_PYTHON_LAUNCHER_SHA256_ENV: interpreter_sha256,
            VALIDATION_PYTHON_INTERPRETER_SHA256_ENV: interpreter_sha256,
            VALIDATION_PYTHON_INTERPRETER_STAT_ENV: interpreter_stat,
        }
    )
    result.setdefault("LANG", "C")
    result.setdefault("LC_ALL", "C")
    result.setdefault("PYTHONHASHSEED", "0")
    result.setdefault("TZ", "UTC")
    return result


def _validation_python_launcher_source(
    *,
    executable: str,
    approved_pythonpath: str,
) -> bytes:
    """Render a launcher containing no child-controlled configuration."""

    return (
        "#!/bin/bash -p\n"
        f"readonly executable={shlex.quote(executable)}\n"
        f"readonly approved={shlex.quote(approved_pythonpath)}\n"
        "unset BASH_ENV ENV PYTHONHOME PYTHONSTARTUP\n"
        "export PYTHONNOUSERSITE=1\n"
        'requested="${PYTHONPATH-}"\n'
        'if [[ -n "$approved" && "$requested" != "$approved" ]]; then\n'
        '    if [[ -n "$requested" ]]; then\n'
        '        export PYTHONPATH="$requested:$approved"\n'
        "    else\n"
        '        export PYTHONPATH="$approved"\n'
        "    fi\n"
        "fi\n"
        'exec "$executable" -s "$@"\n'
    ).encode("utf-8")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise ValidationRuntimeError(
                "sealed validation Python launcher write was incomplete"
            )
        offset += written


@contextmanager
def validation_python_launcher_environment(
    environment: Mapping[str, str],
) -> Iterator[tuple[dict[str, str], ValidationPythonLauncherReceipt]]:
    """Yield an environment that pins nested Python to approved packages.

    Linux uses a sealed anonymous executable addressed through the supervisor's
    procfs descriptor.  A child can replace ``PYTHONPATH`` for workspace-local
    imports, but the launcher appends the roots already admitted by
    :func:`build_validation_environment`.  ``PYTHONNOUSERSITE`` remains set, so
    Python never rediscovers packages through a child-controlled HOME.

    The descriptor stays open only for the yielded subprocess lifetime and is
    closed on every exit path.  Linux fails closed if immutable delivery cannot
    be established.  Other platforms retain the canonical interpreter without
    attempting a weaker mutable launcher.
    """

    child_environment = {
        str(key): str(value) for key, value in environment.items()
    }
    executable_text = str(
        child_environment.get(_CHILD_PYTHON_ENV) or ""
    ).strip()
    if not executable_text:
        raise ValidationRuntimeError(
            "validation environment is missing its canonical Python"
        )
    executable = Path(executable_text)
    if not executable.is_absolute():
        raise ValidationRuntimeError(
            "validation environment Python must be absolute"
        )
    try:
        resolved_executable = executable.resolve(strict=True)
    except OSError as exc:
        raise ValidationRuntimeError(
            f"validation Python is unavailable: {executable}"
        ) from exc
    if not resolved_executable.is_file() or not os.access(
        resolved_executable, os.X_OK
    ):
        raise ValidationRuntimeError(
            f"validation Python is not executable: {executable}"
        )
    _reject_writable_path(
        resolved_executable,
        source="validation Python",
    )
    rendered_executable = str(resolved_executable)
    approved_pythonpath = str(child_environment.get("PYTHONPATH") or "")
    expected_mode = _validation_python_launcher_mode(
        sealed=sys.platform.startswith("linux")
    )
    recorded_mode = str(
        child_environment.get(VALIDATION_PYTHON_LAUNCHER_MODE_ENV) or ""
    )
    if recorded_mode != expected_mode:
        raise ValidationRuntimeError(
            "validation Python launcher mode does not match runtime policy"
        )
    recorded_policy_sha256 = str(
        child_environment.get(
            VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
        )
        or ""
    )
    expected_policy_sha256 = _validation_python_launcher_policy_sha256(
        expected_mode
    )
    if recorded_policy_sha256 != expected_policy_sha256:
        raise ValidationRuntimeError(
            "validation Python launcher policy identity mismatch"
        )
    recorded_content_sha256 = str(
        child_environment.get(VALIDATION_PYTHON_LAUNCHER_SHA256_ENV) or ""
    )
    recorded_interpreter_sha256 = str(
        child_environment.get(VALIDATION_PYTHON_INTERPRETER_SHA256_ENV)
        or ""
    )
    recorded_interpreter_stat = str(
        child_environment.get(VALIDATION_PYTHON_INTERPRETER_STAT_ENV)
        or ""
    )
    interpreter_sha256, interpreter_stat = (
        _validation_python_interpreter_identity(resolved_executable)
    )
    if (
        recorded_interpreter_sha256 != interpreter_sha256
        or recorded_interpreter_stat != interpreter_stat
    ):
        raise ValidationRuntimeError(
            "validation Python interpreter identity mismatch"
        )

    if not sys.platform.startswith("linux"):
        if recorded_content_sha256 != interpreter_sha256:
            raise ValidationRuntimeError(
                "validation Python launcher content identity mismatch"
            )
        child_environment["PYTHON"] = rendered_executable
        yield child_environment, ValidationPythonLauncherReceipt(
            executable=rendered_executable,
            content_sha256=interpreter_sha256,
            interpreter_sha256=interpreter_sha256,
            interpreter_stat=interpreter_stat,
            mode=expected_mode,
            policy_sha256=recorded_policy_sha256,
            sealed=False,
        )
        return

    try:
        import fcntl

        required_seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        creation_flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    except (AttributeError, ImportError) as exc:
        raise ValidationRuntimeError(
            "sealed validation Python launcher is unavailable on Linux"
        ) from exc

    payload = _validation_python_launcher_source(
        executable=rendered_executable,
        approved_pythonpath=approved_pythonpath,
    )
    content_sha256 = hashlib.sha256(payload).hexdigest()
    if recorded_content_sha256 != content_sha256:
        raise ValidationRuntimeError(
            "validation Python launcher content identity mismatch"
        )
    fd = -1
    try:
        fd = os.memfd_create(
            "ipfs-accelerate-validation-python",
            creation_flags,
        )
        _write_all(fd, payload)
        os.fchmod(fd, 0o500)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, required_seals)
        actual_seals = int(fcntl.fcntl(fd, fcntl.F_GET_SEALS))
        if actual_seals & required_seals != required_seals:
            raise ValidationRuntimeError(
                "validation Python launcher did not acquire all required seals"
            )
        persisted = os.pread(fd, len(payload) + 1, 0)
        if (
            len(persisted) != len(payload)
            or hashlib.sha256(persisted).hexdigest() != content_sha256
        ):
            raise ValidationRuntimeError(
                "sealed validation Python launcher content mismatch"
            )
        launcher_path = f"/proc/{os.getpid()}/fd/{fd}"
        if not os.access(launcher_path, os.R_OK | os.X_OK):
            raise ValidationRuntimeError(
                "sealed validation Python launcher is not executable"
            )
        child_environment["PYTHON"] = launcher_path
        yield child_environment, ValidationPythonLauncherReceipt(
            executable=launcher_path,
            content_sha256=content_sha256,
            interpreter_sha256=interpreter_sha256,
            interpreter_stat=interpreter_stat,
            mode=expected_mode,
            policy_sha256=recorded_policy_sha256,
            sealed=True,
        )
    except ValidationRuntimeError:
        raise
    except OSError as exc:
        raise ValidationRuntimeError(
            "sealed validation Python launcher construction failed"
        ) from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


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
    # A reviewed command may prepend workspace-local import roots with an
    # assignment such as ``PYTHONPATH=src:. python -m pytest``.  Bash applies
    # that assignment to the shell function itself, which would otherwise
    # replace the approved site-packages roots supplied by
    # ``build_validation_environment``.  Capture those roots before executing
    # the reviewed text and append them inside every guarded Python launcher.
    # This retains the command's intentional workspace imports while keeping
    # the pinned pytest/runtime packages available.
    guarded = (
        f"readonly {_CHILD_PYTHON_ENV}; "
        '_IPFS_ACCELERATE_APPROVED_PYTHONPATH="${PYTHONPATH-}"; '
        "readonly _IPFS_ACCELERATE_APPROVED_PYTHONPATH; "
        "_ipfs_accelerate_validation_python() { "
        'local requested="${PYTHONPATH-}"; '
        'local approved="${_IPFS_ACCELERATE_APPROVED_PYTHONPATH}"; '
        'if [[ -n "$approved" && "$requested" != "$approved" ]]; then '
        'if [[ -n "$requested" ]]; then '
        'PYTHONPATH="$requested:$approved" '
        f'"${{{_CHILD_PYTHON_ENV}}}" -s "$@"; '
        "else "
        'PYTHONPATH="$approved" '
        f'"${{{_CHILD_PYTHON_ENV}}}" -s "$@"; '
        "fi; "
        "else "
        f'"${{{_CHILD_PYTHON_ENV}}}" -s "$@"; '
        "fi; "
        "}; "
        'python() { _ipfs_accelerate_validation_python "$@"; }; '
        'python3() { _ipfs_accelerate_validation_python "$@"; }; '
        'pytest() { _ipfs_accelerate_validation_python -m pytest "$@"; }; '
        "readonly -f _ipfs_accelerate_validation_python python python3 pytest; "
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
        return [validation_python_executable(), "-s", *parts[1:]]
    if parts[0] == "pytest":
        return [
            validation_python_executable(),
            "-s",
            "-m",
            "pytest",
            *parts[1:],
        ]
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
    falls back to an unisolated process.  The supplied environment is always
    rebuilt through the validation allowlist; callers cannot bypass secret and
    startup-hook scrubbing by claiming that a mapping was already sanitized.
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
