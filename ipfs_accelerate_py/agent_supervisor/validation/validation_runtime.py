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
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

VALIDATION_PATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PATH"
VALIDATION_PYTHON_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON"
VALIDATION_PYTHONPATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH"
VALIDATION_NPM_CACHE_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_NPM_CACHE"
VALIDATION_CARGO_HOME_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_CARGO_HOME"
VALIDATION_RUSTUP_HOME_ENV = (
    "IPFS_ACCELERATE_AGENT_VALIDATION_RUSTUP_HOME"
)
VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV = (
    "IPFS_ACCELERATE_AGENT_VALIDATION_PLAYWRIGHT_BROWSERS_PATH"
)
FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV = (
    "IPFS_ACCELERATE_AGENT_FORMAL_TOOLCHAIN_CONTRACT_SHA256"
)
FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV = (
    "IPFS_ACCELERATE_AGENT_REQUIRED_COMMANDS"
)
FORMAL_TOOLCHAIN_PATH_ENV = (
    "IPFS_ACCELERATE_VALIDATION_FORMAL_TOOLCHAIN_PATH"
)
FORMAL_TOOLCHAIN_ROOT_ENV_NAMES = (
    "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT",
    "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
)
FORMAL_TOOLCHAIN_DEPLOYMENT_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "formal-toolchain-deployment-manifest@1"
)
_FORMAL_TOOL_COMMAND_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}"
)
_MAX_FORMAL_TOOL_COMMANDS = 64
PROOF_REUSE_STATE_ROOT_ENV = "IPFS_PROOF_REUSE_STATE_ROOT"
PROVIDER_PROTECTED_STATE_ROOT_ENV = (
    "IPFS_ACCELERATE_AGENT_PROTECTED_STATE_ROOT"
)
VALIDATION_FILESYSTEM_BOUNDARY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "validation-filesystem-boundary@1"
)
PROVIDER_FILESYSTEM_BOUNDARY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "provider-filesystem-boundary@1"
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
VALIDATION_RUFF_EXECUTABLE_MODE_ENV = (
    "IPFS_ACCELERATE_VALIDATION_RUFF_EXECUTABLE_MODE"
)
VALIDATION_RUFF_EXECUTABLE_SHA256_ENV = (
    "IPFS_ACCELERATE_VALIDATION_RUFF_EXECUTABLE_SHA256"
)
VALIDATION_RUFF_EXECUTABLE_STAT_ENV = (
    "IPFS_ACCELERATE_VALIDATION_RUFF_EXECUTABLE_STAT"
)
_CHILD_PYTHON_ENV = "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"
VALIDATION_RUFF_EXECUTABLE_ENV = (
    "IPFS_ACCELERATE_VALIDATION_RUFF_EXECUTABLE"
)
VALIDATION_RUFF_UNAVAILABLE_MARKER = (
    "ipfs-accelerate-validation-ruff-error:unavailable"
)
_NEUTRAL_HOME = "/nonexistent/ipfs-accelerate-validation"
_NPM_DISABLED_USER_CONFIG = "/dev/null/npmrc"
HERMETIC_VALIDATION_RUNTIME_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hermetic-validation-runtime@1"
)
_RUNTIME_ID_ENV = "IPFS_ACCELERATE_VALIDATION_RUNTIME_ID"
_CANCELLATION_ID_ENV = "IPFS_ACCELERATE_VALIDATION_CANCELLATION_ID"
_VALIDATION_PYTHON_LAUNCHER_POLICY_BASE = (
    "ipfs_accelerate_py/agent-supervisor/"
    "nested-validation-python-launcher@2;"
    "seals=write,grow,shrink,seal;"
    "shell-startup=privileged-no-bash-env;"
    "user-site=interpreter-s-flag;"
    "pythonpath=task-local-then-approved;"
    "ruff=active-distribution-content-stat-sealed-memfd-exact-module"
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
LANDLOCK_APPLIED_ACK_FD_ENV = (
    "IPFS_ACCELERATE_AGENT_LANDLOCK_APPLIED_ACK_FD"
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


@dataclass(frozen=True)
class ProviderFilesystemBoundaryReceipt:
    """Body-free evidence for one autonomous provider write boundary."""

    landlock_abi: int
    policy_sha256: str
    checkpoint_writable: bool
    provider_profile_count: int

    def to_dict(
        self,
        *,
        task_id: str,
        attempt: int,
        stage: str,
        applied: bool = True,
    ) -> dict[str, object]:
        return {
            "schema": PROVIDER_FILESYSTEM_BOUNDARY_SCHEMA,
            "mode": "landlock-provider-write-fence-v1",
            "task_id": str(task_id),
            "attempt": int(attempt),
            "stage": str(stage),
            "landlock_abi": self.landlock_abi,
            "policy_sha256": self.policy_sha256,
            "applied": bool(applied),
            "provider_descendants_fenced": bool(applied),
            "proof_reuse_authority_content_and_names_read_only": bool(applied),
            "task_checkpoint_writable": self.checkpoint_writable,
            "provider_private_home_writable": True,
            "provider_profile_count": self.provider_profile_count,
            "shared_git_metadata_writable": False,
            "protected_hardlink_aliases_checked": True,
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


_SEALED_NODE_RELATIVE = (
    "data/agent_supervisor/verified_gui_optimizer/"
    "toolchain/node_modules/.bin/node"
)
_SEALED_NODE_VERSION = "v22.19.0"
_NODE_COMMAND_RE = re.compile(r"(?<![A-Za-z0-9_/-])node(?![A-Za-z0-9_-])")


def resolve_sealed_node_executable(start: Path | str | None) -> Path | None:
    """Locate the digest/version-checked VGO Node 22 runtime.

    Official validation PATH cannot include the operator-writable toolchain
    directory.  Browser tasks still require that exact Node.  Walk from the
    workspace toward the supervisor state root and accept only ``v22.19.0``.
    """

    if start is None:
        return None
    here = Path(start)
    try:
        here = here.resolve()
    except OSError:
        return None
    for _ in range(14):
        for candidate in (
            here / _SEALED_NODE_RELATIVE,
            here / "toolchain/node_modules/.bin/node",
        ):
            if _usable_sealed_node(candidate):
                try:
                    return candidate.resolve()
                except OSError:
                    return None
        parent = here.parent
        if parent == here:
            break
        here = parent
    return None


def _usable_sealed_node(path: Path) -> bool:
    try:
        if not path.is_file() or not os.access(path, os.X_OK):
            return False
    except OSError:
        return False
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    version = (completed.stdout or completed.stderr or "").strip().splitlines()
    return bool(version) and version[0].strip() == _SEALED_NODE_VERSION


def apply_sealed_node_toolchain(
    environment: Mapping[str, str],
    *,
    workspace_path: Path | str,
    command: str,
) -> dict[str, str]:
    """Prepend the sealed Node 22 bin when the command invokes ``node``."""

    env = dict(environment)
    if not _NODE_COMMAND_RE.search(str(command or "")):
        return env
    sealed = resolve_sealed_node_executable(workspace_path)
    if sealed is None:
        return env
    bin_dir = str(sealed.parent)
    current = str(env.get("PATH") or "")
    entries = [item for item in current.split(os.pathsep) if item]
    if bin_dir in entries:
        entries = [bin_dir, *[item for item in entries if item != bin_dir]]
    else:
        entries = [bin_dir, *entries]
    env["PATH"] = os.pathsep.join(entries)
    return env


def _formal_toolchain_required_commands(
    source: Mapping[str, object],
) -> tuple[str, ...]:
    raw = str(
        source.get(FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV) or ""
    ).strip()
    if not raw:
        return ()
    commands: list[str] = []
    for item in raw.split(","):
        command = item.strip()
        if not command or not _FORMAL_TOOL_COMMAND_RE.fullmatch(command):
            raise ValidationRuntimeError(
                f"{FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV} must contain "
                "comma-separated bare executable names"
            )
        if command not in commands:
            commands.append(command)
        if len(commands) > _MAX_FORMAL_TOOL_COMMANDS:
            raise ValidationRuntimeError(
                f"{FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV} contains too "
                "many commands"
            )
    return tuple(commands)


def _formal_toolchain_root(
    source: Mapping[str, object],
    variable: str,
) -> str | None:
    raw = str(source.get(variable) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValidationRuntimeError(
            f"{variable} must be an absolute deployed toolchain root"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationRuntimeError(
            f"{variable} deployed toolchain root is unavailable"
        ) from exc
    if not resolved.is_dir():
        raise ValidationRuntimeError(
            f"{variable} deployed toolchain root is not a directory"
        )
    try:
        _reject_writable_path(
            resolved,
            source=f"{variable} deployed toolchain root",
        )
    except ValidationRuntimeError as exc:
        raise ValidationRuntimeError(
            f"{variable} is not an immutable formal-toolchain deployment; "
            "stage reviewed assets under a root-owned/read-only root before "
            "supervisor dispatch"
        ) from exc
    return str(resolved)


def formal_toolchain_deployment_manifest(
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build and verify the cross-boundary formal-toolchain manifest.

    The manifest is derived only from the already fail-closed validation PATH,
    explicitly allowlisted managed roots, and explicitly required bare command
    names.  Every required executable is content hashed after writable-path
    rejection.  A caller-supplied expected identity is a fence: mismatch fails
    before provider dispatch or validation child creation.
    """

    source = os.environ if environment is None else environment
    bound_path = str(source.get(FORMAL_TOOLCHAIN_PATH_ENV) or "").strip()
    path = (
        os.pathsep.join(
            _validated_path_entries(
                bound_path,
                source=FORMAL_TOOLCHAIN_PATH_ENV,
            )
        )
        if bound_path
        else validation_executable_path(source)
    )
    roots = {
        variable: resolved
        for variable in FORMAL_TOOLCHAIN_ROOT_ENV_NAMES
        if (resolved := _formal_toolchain_root(source, variable)) is not None
    }
    commands = _formal_toolchain_required_commands(source)
    executable_identities: dict[str, dict[str, object]] = {}
    for command in commands:
        found = shutil.which(command, path=path)
        if not found:
            raise ValidationRuntimeError(
                f"required formal toolchain command is unavailable: {command}"
            )
        executable_identities[command] = _file_identity(Path(found))
    manifest: dict[str, object] = {
        "schema": FORMAL_TOOLCHAIN_DEPLOYMENT_MANIFEST_SCHEMA,
        "path_entries": path.split(os.pathsep),
        "managed_roots": roots,
        "required_executables": executable_identities,
        "writable_sources_rejected": True,
    }
    identity = _sha256(_canonical_json(manifest).encode("utf-8"))
    expected = str(
        source.get(FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV) or ""
    ).strip()
    if expected:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValidationRuntimeError(
                f"{FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV} is malformed"
            )
        if expected != identity:
            raise ValidationRuntimeError(
                "formal toolchain deployment contract identity mismatch"
            )
    return {**manifest, "manifest_sha256": identity}


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

def inventory_walk_error(error):
    raise error

try:
    payload = json.loads(sys.argv[1])
    if not isinstance(payload, dict) or set(payload) - {
        "writable_paths",
        "expected_path_identities",
        "hardlink_guard",
    }:
        raise ValueError("invalid policy keys")
    writable_paths = payload["writable_paths"]
    if (
        not isinstance(writable_paths, list)
        or len(writable_paths) < 2
        or len(writable_paths) > 16
    ):
        raise ValueError("invalid writable path population")
    expected_identities = payload.get("expected_path_identities", [])
    if not isinstance(expected_identities, list) or len(expected_identities) > 16:
        raise ValueError("invalid expected path identities")
    for expected in expected_identities:
        if not isinstance(expected, list) or len(expected) != 3:
            raise ValueError("invalid expected path identity")
        raw_path, raw_dev, raw_ino = expected
        if not isinstance(raw_path, str) or not os.path.isabs(raw_path):
            raise ValueError("invalid identity path")
        identity = os.lstat(raw_path)
        if (int(identity.st_dev), int(identity.st_ino)) != (raw_dev, raw_ino):
            raise RuntimeError("writable path identity changed")
    hardlink_guard = payload.get("hardlink_guard")
    if hardlink_guard is not None:
        if not isinstance(hardlink_guard, dict) or set(hardlink_guard) != {
            "excluded_protected_roots",
            "protected_roots",
            "writable_roots",
        }:
            raise ValueError("invalid hardlink guard")
        writable_roots = hardlink_guard["writable_roots"]
        protected_roots = hardlink_guard["protected_roots"]
        excluded_roots = set(hardlink_guard["excluded_protected_roots"])
        if (
            not isinstance(writable_roots, list)
            or not isinstance(protected_roots, list)
            or len(writable_roots) > 16
            or len(protected_roots) > 16
            or any(not isinstance(item, str) or not os.path.isabs(item) for item in writable_roots)
            or any(not isinstance(item, str) or not os.path.isabs(item) for item in protected_roots)
            or any(not isinstance(item, str) or not os.path.isabs(item) for item in excluded_roots)
        ):
            raise ValueError("invalid hardlink guard roots")
        candidate_inodes = set()
        visited = 0
        for writable_root in writable_roots:
            for directory, directory_names, file_names in os.walk(
                writable_root,
                topdown=True,
                onerror=inventory_walk_error,
                followlinks=False,
            ):
                directory_names[:] = [
                    name
                    for name in directory_names
                    if not os.path.islink(os.path.join(directory, name))
                ]
                for name in file_names:
                    visited += 1
                    if visited > 1000000:
                        raise RuntimeError("hardlink inventory bound exceeded")
                    identity = os.lstat(os.path.join(directory, name))
                    if stat_module.S_ISREG(identity.st_mode) and identity.st_nlink > 1:
                        candidate_inodes.add((int(identity.st_dev), int(identity.st_ino)))
        if candidate_inodes:
            for protected_root in protected_roots:
                for directory, directory_names, file_names in os.walk(
                    protected_root,
                    topdown=True,
                    onerror=inventory_walk_error,
                    followlinks=False,
                ):
                    if directory == protected_root:
                        directory_names[:] = [
                            name for name in directory_names if name != "worktrees"
                        ]
                    retained = []
                    for name in directory_names:
                        child = os.path.join(directory, name)
                        if os.path.islink(child) or child in excluded_roots:
                            continue
                        retained.append(name)
                    directory_names[:] = retained
                    for name in file_names:
                        visited += 1
                        if visited > 2000000:
                            raise RuntimeError("protected inventory bound exceeded")
                        identity = os.lstat(os.path.join(directory, name))
                        if (
                            stat_module.S_ISREG(identity.st_mode)
                            and (int(identity.st_dev), int(identity.st_ino)) in candidate_inodes
                        ):
                            raise RuntimeError("writable hardlink aliases protected state")
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
        for executable_variable in (
            "PYTHON",
            "IPFS_ACCELERATE_VALIDATION_RUFF_EXECUTABLE",
        ):
            raw_executable = os.environ.get(executable_variable, "")
            if raw_executable.startswith("/proc/"):
                executable_fd = os.open(raw_executable, os.O_RDONLY)
                os.set_inheritable(executable_fd, True)
                os.environ[executable_variable] = (
                    "/proc/" + str(os.getpid()) + "/fd/" + str(executable_fd)
                )
        checked(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
        checked(libc.syscall(RESTRICT_SELF, ruleset_fd, 0))
    finally:
        os.close(ruleset_fd)
    raw_ack_fd = os.environ.pop(
        "IPFS_ACCELERATE_AGENT_LANDLOCK_APPLIED_ACK_FD", ""
    ).strip()
    if raw_ack_fd:
        ack_fd = int(raw_ack_fd)
        if ack_fd < 3:
            raise ValueError("invalid Landlock acknowledgement descriptor")
        os.write(ack_fd, b"landlock-applied-v1\\n")
        os.close(ack_fd)
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

    def reject_walk_error(error: OSError) -> None:
        raise error

    try:
        for directory, directory_names, file_names in os.walk(
            workspace,
            topdown=True,
            onerror=reject_walk_error,
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
    additional_writable_paths: Sequence[Path] = (),
    protected_scan_exclusions: Sequence[Path] = (),
) -> None:
    """Reject any writable name aliasing protected proof-control evidence."""

    writable_paths = tuple(
        dict.fromkeys((workspace, *additional_writable_paths))
    )
    candidate_inodes: set[tuple[int, int]] = set()
    for writable_path in writable_paths:
        candidate_inodes.update(_workspace_multilink_inodes(writable_path))
    if not candidate_inodes:
        return
    excluded_paths = tuple(
        dict.fromkeys((workspace, *protected_scan_exclusions))
    )

    def reject_walk_error(error: OSError) -> None:
        raise error

    for reviewed_root in _reviewed_proof_state_roots(state_root):
        try:
            for directory, directory_names, file_names in os.walk(
                reviewed_root,
                topdown=True,
                onerror=reject_walk_error,
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
                    if child in excluded_paths:
                        # Never compare an explicitly non-authoritative
                        # writable subtree to itself.  Ancestors are retained
                        # until this exact path is reached so neighboring
                        # control state remains in the protected inventory.
                        continue
                    retained_directories.append(name)
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
                            "writable provider or validation state aliases protected "
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


def provider_readonly_state_command(
    command: Sequence[str],
    *,
    state_root_path: Path | str,
    workspace_path: Path | str,
    private_home_path: Path | str,
    checkpoint_path: Path | str | None,
    provider_profile_paths: Sequence[Path | str],
    environment: Mapping[str, str],
) -> tuple[list[str], ProviderFilesystemBoundaryReceipt]:
    """Fence one autonomous provider child from proof-authority writes.

    The exact worktree, fresh private home, canonical provider profile roots,
    and (when present) the task's non-authoritative checkpoint directory stay
    writable.  Shared Git metadata and every current/historical proof-control
    path remain read-only.  The returned launcher is inherited by every tool
    process spawned by the provider.
    """

    argv = [str(value) for value in command]
    if not argv:
        raise ValidationRuntimeError("provider command must not be empty")
    state_root = Path(state_root_path).resolve(strict=True)
    workspace = Path(workspace_path).resolve(strict=True)
    private_home = Path(private_home_path).resolve(strict=True)
    if not state_root.is_dir():
        raise ValidationRuntimeError(
            "proof-reuse state root is not a directory"
        )
    if not workspace.is_dir() or not private_home.is_dir():
        raise ValidationRuntimeError(
            "provider workspace and private home must be directories"
        )

    checkpoint: Path | None = None
    if checkpoint_path is not None and str(checkpoint_path).strip():
        checkpoint = Path(checkpoint_path).resolve(strict=True)
        if not checkpoint.is_dir():
            raise ValidationRuntimeError(
                "provider checkpoint path is not a directory"
            )
        try:
            checkpoint_relative = checkpoint.relative_to(state_root)
        except ValueError as exc:
            raise ValidationRuntimeError(
                "provider checkpoint path is outside the current state root"
            ) from exc
        if (
            len(checkpoint_relative.parts) != 4
            or checkpoint_relative.parts[0] != "state"
            or checkpoint_relative.parts[2] != "implementation_checkpoints"
            or not checkpoint_relative.parts[1].startswith("ptr_lane_")
            or not checkpoint_relative.parts[3]
        ):
            raise ValidationRuntimeError(
                "provider checkpoint path is not task checkpoint scratch"
            )

    profiles: list[Path] = []
    for raw_path in provider_profile_paths:
        text = str(raw_path).strip()
        if not text:
            continue
        profile = Path(text).resolve(strict=True)
        if not profile.is_dir():
            raise ValidationRuntimeError(
                "provider profile path is not a directory"
            )
        profiles.append(profile)
    profiles = list(dict.fromkeys(profiles))

    protected_roots = _reviewed_proof_state_roots(state_root)
    for protected_root in protected_roots:
        try:
            workspace_relative = workspace.relative_to(protected_root)
        except ValueError:
            continue
        if (
            protected_root != state_root
            or len(workspace_relative.parts) != 3
            or workspace_relative.parts[0] != "worktrees"
            or re.fullmatch(r"ptr_lane_\d+", workspace_relative.parts[1])
            is None
            or not workspace_relative.parts[2]
        ):
            raise ValidationRuntimeError(
                "provider workspace is not an exact current-lane worktree"
            )
    for writable in (private_home, *profiles):
        for protected_root in protected_roots:
            try:
                writable.relative_to(protected_root)
            except ValueError:
                pass
            else:
                raise ValidationRuntimeError(
                    "provider private/profile path overlaps proof state"
                )
            try:
                protected_root.relative_to(writable)
            except ValueError:
                pass
            else:
                raise ValidationRuntimeError(
                    "provider writable path contains proof state"
                )
    for writable in (workspace, checkpoint):
        if writable is None:
            continue
        try:
            state_root.relative_to(writable)
        except ValueError:
            pass
        else:
            raise ValidationRuntimeError(
                "provider writable path contains proof state"
            )

    additional_hardlink_surfaces = [private_home, *profiles]
    protected_scan_exclusions: list[Path] = []
    if checkpoint is not None:
        additional_hardlink_surfaces.append(checkpoint)
        protected_scan_exclusions.append(checkpoint)
    _reject_protected_state_hardlink_aliases(
        state_root=state_root,
        workspace=workspace,
        additional_writable_paths=tuple(additional_hardlink_surfaces),
        protected_scan_exclusions=tuple(protected_scan_exclusions),
    )

    source = _validation_landlock_launcher_source()
    writable_paths = [str(workspace), str(private_home)]
    if checkpoint is not None:
        writable_paths.append(str(checkpoint))
    writable_paths.extend(str(path) for path in profiles)
    null_devices = tuple(
        path
        for path in _validation_standard_device_write_paths()
        if path == "/dev/null"
    )
    writable_paths.extend(null_devices)
    writable_paths = list(dict.fromkeys(writable_paths))
    if len(writable_paths) > 8:
        raise ValidationRuntimeError(
            "provider filesystem boundary has too many writable paths"
        )
    writable_directories = [workspace, private_home]
    if checkpoint is not None:
        writable_directories.append(checkpoint)
    writable_directories.extend(profiles)
    excluded_protected_roots = []
    for candidate in (workspace, checkpoint):
        if candidate is None:
            continue
        if any(
            candidate == protected_root
            or protected_root in candidate.parents
            for protected_root in protected_roots
        ):
            excluded_protected_roots.append(str(candidate))
    expected_path_identities: list[list[object]] = []
    for raw_path in writable_paths:
        identity = Path(raw_path).lstat()
        expected_path_identities.append(
            [raw_path, int(identity.st_dev), int(identity.st_ino)]
        )
    policy = json.dumps(
        {
            "expected_path_identities": expected_path_identities,
            "hardlink_guard": {
                "excluded_protected_roots": excluded_protected_roots,
                "protected_roots": [str(path) for path in protected_roots],
                "writable_roots": [str(path) for path in writable_directories],
            },
            "writable_paths": writable_paths,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    executable = validation_python_executable(
        {VALIDATION_PYTHON_ENV: environment.get(_CHILD_PYTHON_ENV, "")}
    )
    receipt = ProviderFilesystemBoundaryReceipt(
        landlock_abi=validation_landlock_abi(),
        policy_sha256=hashlib.sha256(
            source.encode("utf-8") + b"\0" + policy.encode("utf-8")
        ).hexdigest(),
        checkpoint_writable=checkpoint is not None,
        provider_profile_count=len(profiles),
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


def _active_ruff_executable_snapshot(
    *, approved_pythonpath: str
) -> tuple[bytes, str, str] | None:
    """Read the approved active Ruff distribution binary with stable identity."""

    try:
        approved_roots = {
            Path(entry).resolve(strict=True)
            for entry in approved_pythonpath.split(os.pathsep)
            if entry
        }
        distribution = importlib_metadata.distribution("ruff")
        if Path(distribution.locate_file("")).resolve(
            strict=True
        ) not in approved_roots:
            return None
        executable_name = "ruff.exe" if os.name == "nt" else "ruff"
        candidates = {
            Path(distribution.locate_file(entry)).resolve(strict=True)
            for entry in distribution.files or ()
            if Path(str(entry)).name == executable_name
        }
    except importlib_metadata.PackageNotFoundError:
        return None
    except OSError as exc:
        raise ValidationRuntimeError(
            "active Ruff distribution is unavailable"
        ) from exc
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValidationRuntimeError(
            "active Ruff distribution has ambiguous executable entries"
        )
    executable = candidates.pop()
    try:
        before = executable.stat()
        if not stat.S_ISREG(before.st_mode) or not (
            stat.S_IMODE(before.st_mode) & 0o111
        ):
            raise ValidationRuntimeError(
                "active Ruff executable is not a regular executable"
            )
        content = executable.read_bytes()
        after = executable.stat()

        def stat_payload(value: os.stat_result) -> str:
            return _canonical_json(
                {
                    "path": str(executable),
                    "device": int(value.st_dev),
                    "inode": int(value.st_ino),
                    "size": int(value.st_size),
                    "mode": stat.S_IMODE(value.st_mode),
                    "mtime_ns": int(value.st_mtime_ns),
                    "ctime_ns": int(value.st_ctime_ns),
                }
            )
        before_identity = stat_payload(before)
        after_identity = stat_payload(after)
        if before_identity != after_identity:
            raise ValidationRuntimeError(
                "active Ruff executable changed while being read"
            )
        if len(content) != int(after.st_size):
            raise ValidationRuntimeError(
                "active Ruff executable read was incomplete"
            )
        return content, hashlib.sha256(content).hexdigest(), after_identity
    except ValidationRuntimeError:
        raise
    except OSError as exc:
        raise ValidationRuntimeError(
            "active Ruff executable cannot be read"
        ) from exc


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
    for variable in (
        VALIDATION_RUFF_EXECUTABLE_MODE_ENV,
        VALIDATION_RUFF_EXECUTABLE_SHA256_ENV,
        VALIDATION_RUFF_EXECUTABLE_STAT_ENV,
        VALIDATION_RUFF_EXECUTABLE_ENV,
    ):
        result.pop(variable, None)
    ruff_snapshot = _active_ruff_executable_snapshot(
        approved_pythonpath=str(result.get("PYTHONPATH") or "")
    )
    if ruff_snapshot is None:
        result[VALIDATION_RUFF_EXECUTABLE_MODE_ENV] = "unavailable"
    else:
        _ruff_content, ruff_sha256, ruff_stat = ruff_snapshot
        result.update(
            {
                VALIDATION_RUFF_EXECUTABLE_MODE_ENV: "sealed-memfd",
                VALIDATION_RUFF_EXECUTABLE_SHA256_ENV: ruff_sha256,
                VALIDATION_RUFF_EXECUTABLE_STAT_ENV: ruff_stat,
            }
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
                    ruff_sha256=(
                        ruff_snapshot[1] if ruff_snapshot is not None else ""
                    ),
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
    cargo_home = _approved_directory(source, VALIDATION_CARGO_HOME_ENV)
    if cargo_home is not None:
        # Private validation HOME cannot see the supervisor cargo registry.
        # Bind the pre-populated host cache read-only and stay offline so
        # validation never mutates or redownloads crates.io content.
        result["CARGO_HOME"] = cargo_home
        result.setdefault("CARGO_NET_OFFLINE", "true")
    rustup_home = _approved_directory(source, VALIDATION_RUSTUP_HOME_ENV)
    if rustup_home is not None:
        result["RUSTUP_HOME"] = rustup_home
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
    formal_toolchain = formal_toolchain_deployment_manifest(source)
    result[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = str(
        formal_toolchain["manifest_sha256"]
    )
    result[FORMAL_TOOLCHAIN_PATH_ENV] = os.pathsep.join(
        str(item) for item in formal_toolchain["path_entries"]
    )
    required_commands = tuple(
        dict(formal_toolchain["required_executables"])
    )
    if required_commands:
        result[FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV] = ",".join(
            required_commands
        )
    for variable, root in dict(formal_toolchain["managed_roots"]).items():
        result[str(variable)] = str(root)
    return result


def _validation_python_launcher_source(
    *,
    executable: str,
    approved_pythonpath: str,
    ruff_sha256: str,
) -> bytes:
    """Render a launcher that discovers Ruff beside its sealed parent FD."""

    ruff_broker = (
        "import fcntl,hashlib,os,re,stat,sys\n"
        "launcher,expected=sys.argv[1:3]\n"
        "match=re.fullmatch(r'/proc/(?:([1-9][0-9]*)|self)/fd/([0-9]+)',launcher)\n"
        "if match is None: sys.exit(75)\n"
        "directory=('/proc/'+match.group(1)+'/fd') if match.group(1) else '/proc/self/fd'\n"
        "launcher_fd=int(match.group(2))\n"
        "entries=[int(entry) for entry in os.listdir(directory) if entry.isdecimal()]\n"
        "if len(entries)>512: sys.exit(75)\n"
        "required=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL\n"
        "fd=-1\n"
        "for candidate in sorted(entries):\n"
        " if candidate==launcher_fd: continue\n"
        " opened=-1\n"
        " try:\n"
        "  opened=os.open(directory+'/'+str(candidate),os.O_RDONLY|os.O_CLOEXEC)\n"
        "  before=os.fstat(opened)\n"
        "  if not stat.S_ISREG(before.st_mode) or not before.st_mode&0o111: continue\n"
        "  if fcntl.fcntl(opened,fcntl.F_GET_SEALS)&required != required: continue\n"
        "  digest=hashlib.sha256()\n"
        "  while chunk:=os.read(opened,1024*1024): digest.update(chunk)\n"
        "  after=os.fstat(opened)\n"
        "  if (before.st_dev,before.st_ino,before.st_size,before.st_mode)!=(after.st_dev,after.st_ino,after.st_size,after.st_mode): continue\n"
        "  if digest.hexdigest()!=expected: continue\n"
        "  os.lseek(opened,0,os.SEEK_SET)\n"
        "  fd=opened\n"
        "  opened=-1\n"
        "  break\n"
        " except OSError:\n"
        "  pass\n"
        " finally:\n"
        "  if opened>=0: os.close(opened)\n"
        "if fd<0: sys.exit(75)\n"
        "env=dict(os.environ)\n"
        f"[env.pop(key,None) for key in {(VALIDATION_RUFF_EXECUTABLE_ENV, VALIDATION_RUFF_EXECUTABLE_MODE_ENV, VALIDATION_RUFF_EXECUTABLE_SHA256_ENV, VALIDATION_RUFF_EXECUTABLE_STAT_ENV)!r}]\n"
        "os.execve(fd,['ruff',*sys.argv[3:]],env)\n"
    )

    return (
        "#!/bin/bash -p\n"
        f"readonly executable={shlex.quote(executable)}\n"
        f"readonly approved={shlex.quote(approved_pythonpath)}\n"
        'readonly launcher_path="$0"\n'
        f"readonly ruff_sha256={shlex.quote(ruff_sha256)}\n"
        f"readonly ruff_broker={shlex.quote(ruff_broker)}\n"
        "unset BASH_ENV ENV PYTHONHOME PYTHONSTARTUP "
        f"{VALIDATION_RUFF_EXECUTABLE_ENV} "
        f"{VALIDATION_RUFF_EXECUTABLE_MODE_ENV} "
        f"{VALIDATION_RUFF_EXECUTABLE_SHA256_ENV} "
        f"{VALIDATION_RUFF_EXECUTABLE_STAT_ENV}\n"
        "export PYTHONNOUSERSITE=1\n"
        "ruff_invocation=0\n"
        'if [[ "$#" -ge 2 && "$1" =~ ^-[^-]*m$ '
        '&& "$2" == "ruff" ]]; then\n'
        "    shift 2\n"
        "    ruff_invocation=1\n"
        'elif [[ "$#" -ge 1 && "$1" == "-mruff" ]]; then\n'
        "    shift\n"
        "    ruff_invocation=1\n"
        "else\n"
        "    previous_module_flag=0\n"
        '    for argument in "$@"; do\n'
        '        if [[ "$previous_module_flag" == 1 && '
        '"$argument" =~ ^ruff(\\.|$) ]]; then\n'
        "            printf '%s\\n' "
        "ipfs-accelerate-validation-ruff-error:unsupported-module-spelling >&2\n"
        "            exit 75\n"
        "        fi\n"
        '        if [[ "$argument" =~ ^-[^-]*mruff(\\.|$) ]]; then\n'
        "            printf '%s\\n' "
        "ipfs-accelerate-validation-ruff-error:unsupported-module-spelling >&2\n"
        "            exit 75\n"
        "        fi\n"
        '        if [[ "$argument" =~ ^-[^-]*m$ ]]; then\n'
        "            previous_module_flag=1\n"
        "        else\n"
        "            previous_module_flag=0\n"
        "        fi\n"
        "    done\n"
        "fi\n"
        'if [[ "$ruff_invocation" == 1 ]]; then\n'
        '    if [[ -z "$ruff_sha256" ]]; then\n'
        f"        printf '%s\\n' {shlex.quote(VALIDATION_RUFF_UNAVAILABLE_MARKER)} >&2\n"
        "        exit 75\n"
        "    fi\n"
        '    exec "$executable" -I -c "$ruff_broker" '
        '"$launcher_path" "$ruff_sha256" "$@"\n'
        "fi\n"
        'requested="${PYTHONPATH-}"\n'
        'if [[ -n "$approved" && "$requested" != "$approved" ]]; then\n'
        '    if [[ -n "$requested" ]]; then\n'
        '        export PYTHONPATH="$requested:$approved"\n'
        "    else\n"
        '        export PYTHONPATH="$approved"\n'
        "    fi\n"
        "fi\n"
        'exec "$executable" -s "$@"\n'
    ).encode()


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise ValidationRuntimeError(
                "sealed validation Python launcher write was incomplete"
            )
        offset += written


def _sealed_executable_memfd(
    *,
    name: str,
    payload: bytes,
    creation_flags: int,
    required_seals: int,
    fcntl_module: Any,
) -> tuple[int, str]:
    """Copy executable bytes to a verified sealed memfd."""

    fd = -1
    try:
        fd = os.memfd_create(name, creation_flags)
        _write_all(fd, payload)
        os.fchmod(fd, 0o500)
        fcntl_module.fcntl(fd, fcntl_module.F_ADD_SEALS, required_seals)
        actual_seals = int(
            fcntl_module.fcntl(fd, fcntl_module.F_GET_SEALS)
        )
        if actual_seals & required_seals != required_seals:
            raise ValidationRuntimeError(
                f"sealed {name} did not acquire all required seals"
            )
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        persisted_hasher = hashlib.sha256()
        offset = 0
        while offset < len(payload):
            chunk = os.pread(
                fd,
                min(1024 * 1024, len(payload) - offset),
                offset,
            )
            if not chunk:
                break
            persisted_hasher.update(chunk)
            offset += len(chunk)
        if (
            offset != len(payload)
            or os.pread(fd, 1, offset)
            or persisted_hasher.hexdigest() != expected_sha256
        ):
            raise ValidationRuntimeError(
                f"sealed {name} content mismatch"
            )
        executable_path = f"/proc/{os.getpid()}/fd/{fd}"
        if not os.access(executable_path, os.R_OK | os.X_OK):
            raise ValidationRuntimeError(
                f"sealed {name} is not executable"
            )
        return fd, executable_path
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


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

    recorded_ruff_mode = str(
        child_environment.get(VALIDATION_RUFF_EXECUTABLE_MODE_ENV) or ""
    )
    recorded_ruff_sha256 = str(
        child_environment.get(VALIDATION_RUFF_EXECUTABLE_SHA256_ENV) or ""
    )
    recorded_ruff_stat = str(
        child_environment.get(VALIDATION_RUFF_EXECUTABLE_STAT_ENV) or ""
    )
    if recorded_ruff_mode not in {"sealed-memfd", "unavailable"}:
        raise ValidationRuntimeError(
            "validation Ruff executable mode does not match runtime policy"
        )
    ruff_snapshot = _active_ruff_executable_snapshot(
        approved_pythonpath=approved_pythonpath
    )
    if recorded_ruff_mode == "unavailable":
        if (
            recorded_ruff_sha256
            or recorded_ruff_stat
            or ruff_snapshot is not None
        ):
            raise ValidationRuntimeError(
                "validation Ruff unavailable identity mismatch"
            )
    elif (
        ruff_snapshot is None
        or recorded_ruff_sha256 != ruff_snapshot[1]
        or recorded_ruff_stat != ruff_snapshot[2]
    ):
        raise ValidationRuntimeError(
            "validation Ruff executable identity mismatch"
        )

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

    identity_payload = _validation_python_launcher_source(
        executable=rendered_executable,
        approved_pythonpath=approved_pythonpath,
        ruff_sha256=(ruff_snapshot[1] if ruff_snapshot is not None else ""),
    )
    content_sha256 = hashlib.sha256(identity_payload).hexdigest()
    if recorded_content_sha256 != content_sha256:
        raise ValidationRuntimeError(
            "validation Python launcher content identity mismatch"
        )
    open_fds: set[int] = set()
    try:
        if ruff_snapshot is not None:
            ruff_fd, ruff_path = (
                _sealed_executable_memfd(
                    name="validation Ruff executable",
                    payload=ruff_snapshot[0],
                    creation_flags=creation_flags,
                    required_seals=required_seals,
                    fcntl_module=fcntl,
                )
            )
            open_fds.add(ruff_fd)
            child_environment[VALIDATION_RUFF_EXECUTABLE_ENV] = ruff_path
        python_fd, launcher_path = (
            _sealed_executable_memfd(
                name="validation Python launcher",
                payload=identity_payload,
                creation_flags=creation_flags,
                required_seals=required_seals,
                fcntl_module=fcntl,
            )
        )
        open_fds.add(python_fd)
        for variable in (
            VALIDATION_RUFF_EXECUTABLE_MODE_ENV,
            VALIDATION_RUFF_EXECUTABLE_SHA256_ENV,
            VALIDATION_RUFF_EXECUTABLE_STAT_ENV,
        ):
            child_environment.pop(variable, None)
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
        for fd in open_fds:
            try:
                os.close(fd)
            except OSError:
                pass


_SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
_SHELL_CONTROL_TOKEN = re.compile(r"[;&|()]+")
_PYTHON_MODULE_FLAG = re.compile(r"^-[^-]*m$")
_PYTHON_COMPACT_RUFF_MODULE = re.compile(r"^-[^-]*mruff(?:\.|$)")
_RUFF_MODULE_NAME = re.compile(r"^ruff(?:\.|$)")
_DYNAMIC_SHELL_COMMAND_WORD = re.compile(r"[$*?\[\]{}~]")
_COMMAND_WRAPPERS = frozenset(
    {"chrt", "command", "env", "ionice", "nice", "nohup", "setsid", "stdbuf", "timeout"}
)


def _ruff_module_indices(
    arguments: Sequence[str],
) -> tuple[tuple[int, int | None], ...]:
    """Locate Python Ruff module spellings without executing argv wrappers."""

    found: list[tuple[int, int | None]] = []
    for index, argument in enumerate(arguments):
        if _PYTHON_COMPACT_RUFF_MODULE.match(argument):
            found.append((index, None))
        elif (
            _PYTHON_MODULE_FLAG.match(argument)
            and index + 1 < len(arguments)
            and _RUFF_MODULE_NAME.match(arguments[index + 1])
        ):
            found.append((index, index + 1))
    return tuple(found)


def _shell_command_start(arguments: Sequence[str], index: int) -> int:
    """Return the first token after the nearest shell control operator."""

    for candidate in range(index - 1, -1, -1):
        if _SHELL_CONTROL_TOKEN.fullmatch(arguments[candidate]):
            return candidate + 1
    return 0


def _shell_command_segments(
    arguments: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Split reviewed shell words at command-control operators."""

    segments: list[tuple[str, ...]] = []
    start = 0
    for index, argument in enumerate(arguments):
        if not _SHELL_CONTROL_TOKEN.fullmatch(argument):
            continue
        if index > start:
            segments.append(tuple(arguments[start:index]))
        start = index + 1
    if start < len(arguments):
        segments.append(tuple(arguments[start:]))
    return tuple(segments)


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
        analysis_text = text.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
        lexer = shlex.shlex(
            analysis_text,
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
    for token in leading:
        if not any(character.isspace() for character in token):
            continue
        try:
            embedded = shlex.split(token)
        except ValueError as exc:
            raise ValidationRuntimeError(
                "validation command has invalid embedded shell arguments"
            ) from exc
        if (
            _ruff_module_indices(embedded)
            or any(Path(part).name == "ruff" for part in embedded)
            or any(Path(part).name in {"python", "python3"} for part in embedded)
        ):
            raise ValidationRuntimeError(
                "embedded Python or Ruff validation arguments are not permitted"
            )
    ruff_module_indices = _ruff_module_indices(leading)
    for module_index, _value_index in ruff_module_indices:
        command_start = _shell_command_start(leading, module_index)
        while (
            command_start < module_index
            and _SHELL_ASSIGNMENT.fullmatch(leading[command_start])
        ):
            command_start += 1
        if (
            command_start >= module_index
            or leading[command_start] not in {"python", "python3"}
        ):
            raise ValidationRuntimeError(
                "Ruff validation must use the sealed python or python3 launcher"
            )
    module_value_indices = {
        value_index
        for _flag_index, value_index in ruff_module_indices
        if value_index is not None
    }
    for index, token in enumerate(leading):
        if index in module_value_indices or Path(token).name != "ruff":
            continue
        raise ValidationRuntimeError(
            "direct Ruff validation requires the sealed Python launcher"
        )
    for segment in _shell_command_segments(leading):
        command_index = 0
        while (
            command_index < len(segment)
            and _SHELL_ASSIGNMENT.fullmatch(segment[command_index])
        ):
            command_index += 1
        if command_index >= len(segment):
            continue
        if _DYNAMIC_SHELL_COMMAND_WORD.search(segment[command_index]):
            raise ValidationRuntimeError(
                "dynamic validation command names are not permitted"
            )
        for later in segment[command_index + 1 :]:
            if Path(later).name in {"python", "python3"}:
                raise ValidationRuntimeError(
                    "wrapped Python validation commands are not permitted"
                )
        command_name = segment[command_index]
        if Path(command_name).name in _COMMAND_WRAPPERS and any(
            _DYNAMIC_SHELL_COMMAND_WORD.search(argument)
            for argument in segment[command_index + 1 :]
        ):
            raise ValidationRuntimeError(
                "dynamic wrapped validation command names are not permitted"
            )
        if Path(command_name).name in {"python", "python3"} and command_name not in {
            "python",
            "python3",
        }:
            raise ValidationRuntimeError(
                "validation Python must use the sealed python or python3 launcher"
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
        '_IPFS_ACCELERATE_SEALED_PYTHON="${PYTHON}"; '
        "readonly _IPFS_ACCELERATE_SEALED_PYTHON; "
        '_IPFS_ACCELERATE_APPROVED_PYTHONPATH="${PYTHONPATH-}"; '
        "readonly _IPFS_ACCELERATE_APPROVED_PYTHONPATH; "
        "_ipfs_accelerate_validation_python() { "
        'local requested="${PYTHONPATH-}"; '
        'local approved="${_IPFS_ACCELERATE_APPROVED_PYTHONPATH}"; '
        'local -a prefix=(); '
        f'if [[ "${{_IPFS_ACCELERATE_SEALED_PYTHON}}" '
        f'== "${{{_CHILD_PYTHON_ENV}}}" ]]; then prefix=(-s); fi; '
        'if [[ -n "$approved" && "$requested" != "$approved" ]]; then '
        'if [[ -n "$requested" ]]; then '
        'PYTHONPATH="$requested:$approved" '
        '"${_IPFS_ACCELERATE_SEALED_PYTHON}" "${prefix[@]}" "$@"; '
        "else "
        'PYTHONPATH="$approved" '
        '"${_IPFS_ACCELERATE_SEALED_PYTHON}" "${prefix[@]}" "$@"; '
        "fi; "
        "else "
        '"${_IPFS_ACCELERATE_SEALED_PYTHON}" "${prefix[@]}" "$@"; '
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
    executable_name = Path(parts[0]).name
    split_env_argv: list[str] = []
    for index, part in enumerate(parts):
        split_text = ""
        if part in {"-S", "--split-string"} and index + 1 < len(parts):
            split_text = parts[index + 1]
        elif part.startswith("--split-string="):
            split_text = part.partition("=")[2]
        elif part.startswith("-S") and len(part) > 2:
            split_text = part[2:]
        if split_text:
            try:
                split_env_argv.extend(shlex.split(split_text))
            except ValueError as exc:
                raise ValidationRuntimeError(
                    "invalid env split-string validation argv"
                ) from exc
    embedded_argv: list[str] = []
    for part in parts:
        if not any(character.isspace() for character in part):
            continue
        try:
            embedded_argv.extend(shlex.split(part))
        except ValueError as exc:
            raise ValidationRuntimeError(
                "invalid embedded validation argv"
            ) from exc
    ruff_argv = [parts, split_env_argv, embedded_argv]
    if (
        any(Path(part).name == "ruff" for argv in ruff_argv for part in argv)
        or any(_ruff_module_indices(argv) for argv in ruff_argv)
    ):
        raise ValidationRuntimeError(
            "direct argv Ruff validation requires the sealed string-command launcher"
        )
    if executable_name in {"python", "python3"} and parts[0] not in {
        "python",
        "python3",
    }:
        raise ValidationRuntimeError(
            "direct argv Python must use the canonical launcher name"
        )
    if any(
        Path(part).name in {"python", "python3"}
        for part in parts[1:]
    ):
        raise ValidationRuntimeError(
            "wrapped direct argv Python validation is not permitted"
        )
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
    if executable_name not in {"bash", "sh"}:
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

    child_environment = apply_sealed_node_toolchain(
        build_validation_environment(environment),
        workspace_path=workspace_path,
        command=command,
    )
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
