"""Admitted explicit-argv verification process runner.

This module is the single shared subprocess boundary for incremental
verification check adapters.  It is intentionally a private IVP adapter over
existing supervisor infrastructure:

* :class:`~ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler.ResourceScheduler`
  for resource admission and revocable leases
* process-group / process-tree termination from the todo-daemon fencing helpers
* content-addressed artifact digests from the multiformats identity profile
* closed hermetic sandbox and network policy observations matching the
  verification contracts

Non-goals (hard refuse):

* shell-string interpolation (``shell`` is always ``False``)
* ambient secret inheritance
* auto-install, download, network-policy widening
* mock hardware / mock inference / simulated success paths
* unbounded stdout/stderr capture
* publication of late success after cancellation or lease revocation

Importing this module starts no processes and performs no network I/O.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

from ..core.multiformats_identity import cid_for_bytes
from ..runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourceAdmissionLease,
    ResourceScheduler,
)
from ..todo_daemon.core import pid_alive, terminate_pid_tree
from .contracts import TerminalStatus

# ---------------------------------------------------------------------------
# Schemas and closed policy constants
# ---------------------------------------------------------------------------

PROCESS_RUNNER_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/verification-process-runner@1"
)
PROCESS_RUNNER_EVIDENCE: str = "ivp/process-runner@1"
PROCESS_TREE_CANCELLATION_EVIDENCE: str = "ivp/process-tree-cancellation@1"

SANDBOX_SCHEMA: str = "hermetic-sandbox@1"
SANDBOX_POLICY_SCHEMA: str = "hermetic-sandbox-policy@1"
FILESYSTEM_POLICY_SCHEMA: str = "verification-filesystem-policy@1"
SANDBOX_IDENTITY_SCHEMA: str = "verification-sandbox-identity@1"
NETWORK_POLICY_DENY_ALL: str = "deny_all"

DEFAULT_MAX_STDOUT_BYTES: int = 256 * 1024
DEFAULT_MAX_STDERR_BYTES: int = 256 * 1024
DEFAULT_MAX_ENV_KEYS: int = 256
DEFAULT_MAX_ENV_VALUE_CHARS: int = 65_536
DEFAULT_MAX_ARGV_ITEMS: int = 512
DEFAULT_MAX_ARGV_ITEM_CHARS: int = 16_384
DEFAULT_MAX_TIMEOUT_SECONDS: float = 86_400.0
DEFAULT_MIN_TIMEOUT_SECONDS: float = 0.001
DEFAULT_TERM_GRACE_SECONDS: float = 0.25
DEFAULT_KILL_WAIT_SECONDS: float = 1.0
DEFAULT_POLL_INTERVAL_SECONDS: float = 0.02
DEFAULT_PREVIEW_CHARS: int = 4_096
DEFAULT_RESOURCE_CLASS: str = "cpu-validation"
DEFAULT_STAGE: str = "validation"

# Environment keys that widen network, install, auth, or home-cache policy.
_FORBIDDEN_ENV_MARKERS: frozenset[str] = frozenset(
    {
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "ftp_proxy",
        "pip_index_url",
        "pip_extra_index_url",
        "npm_config_registry",
        "uv_index_url",
        "poetry_http_basic",
        "aws_access_key_id",
        "aws_secret_access_key",
        "openai_api_key",
        "anthropic_api_key",
        "hf_token",
        "huggingface_token",
        "github_token",
        "gh_token",
    }
)

_AUTO_INSTALL_ARGV_MARKERS: frozenset[str] = frozenset(
    {
        "install",
        "pip",
        "pip3",
        "npm",
        "yarn",
        "pnpm",
        "uv",
        "poetry",
        "conda",
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "brew",
        "cargo",
        "go",
        "gem",
        "bundle",
    }
)

_CLOSED_SANDBOX_POLICY: Mapping[str, str] = MappingProxyType(
    {
        "schema": SANDBOX_POLICY_SCHEMA,
        "network": "deny",
        "auto_install": "deny",
        "home_cache": "deny",
        "auth_material": "deny",
    }
)

_CLOSED_FILESYSTEM_POLICY: Mapping[str, str] = MappingProxyType(
    {
        "schema": FILESYSTEM_POLICY_SCHEMA,
        "source": "read_only",
        "artifacts": "private_writable",
    }
)

Clock = Callable[[], float]
PopenFactory = Callable[..., Any]
Sleep = Callable[[float], None]
_PROCESS_HANDLE_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    subprocess.SubprocessError,
)


@contextmanager
def _temporary_binary_stream() -> Iterator[BinaryIO]:
    """Yield one automatically closed binary stream for bounded capture."""

    with tempfile.TemporaryFile(mode="w+b") as stream:
        yield stream


def _process_returncode(process: Any) -> tuple[bool, int | None]:
    """Return whether a generic process handle could be polled safely."""

    try:
        returncode = process.poll()
    except _PROCESS_HANDLE_ERRORS:
        return False, None
    return True, None if returncode is None else int(returncode)


def _best_effort_wait(process: Any, *, timeout: float) -> None:
    """Reap a generic process handle without overriding fence evidence."""

    try:
        process.wait(timeout=timeout)
    except _PROCESS_HANDLE_ERRORS:
        return


def _close_stream_files(stream_files: ExitStack) -> None:
    """Close capture streams without overriding the run's terminal result."""

    try:
        stream_files.close()
    except OSError:
        return


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VerificationProcessRunnerError(ValueError):
    """Base error for fail-closed process-runner contract violations."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_command",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class VerificationProcessPolicyError(VerificationProcessRunnerError):
    """Caller requested a disallowed policy (shell, install, network widen)."""


class VerificationProcessBoundsError(VerificationProcessRunnerError):
    """A command field exceeds a closed runner bound."""


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class VerificationCancellation:
    """Thread-safe, identity-fenced cooperative cancellation token.

    Cancellation only sticks when the caller presents the exact fencing
    identity that was issued with the token.  That identity is observed on the
    run result so late publication can be fenced independently of process exit.
    """

    def __init__(self, cancellation_id: str | None = None) -> None:
        identity = str(cancellation_id or "").strip() or f"cancel:{uuid.uuid4().hex}"
        self.cancellation_id = identity
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    def cancel(self, *, cancellation_id: str | None = None, reason: str = "cancelled") -> bool:
        """Cancel only when *cancellation_id* matches this token's identity.

        When *cancellation_id* is omitted the token cancels itself (owner path).
        """

        presented = (
            self.cancellation_id
            if cancellation_id is None
            else str(cancellation_id or "").strip()
        )
        if presented != self.cancellation_id:
            return False
        with self._lock:
            if self._event.is_set():
                return True
            self._reason = str(reason or "cancelled").strip() or "cancelled"
            self._event.set()
            return True

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def cancelled(self) -> bool:
        return self.is_cancelled()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def __bool__(self) -> bool:
        return self.is_cancelled()


# ---------------------------------------------------------------------------
# Command / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationSandboxIdentity:
    """Effective hermetic sandbox observation required for admission."""

    sandbox_schema: str = SANDBOX_SCHEMA
    sandbox_policy: Mapping[str, str] = field(
        default_factory=lambda: dict(_CLOSED_SANDBOX_POLICY)
    )
    filesystem_policy: Mapping[str, str] = field(
        default_factory=lambda: dict(_CLOSED_FILESYSTEM_POLICY)
    )
    sandbox_id: str = ""
    source_root: str = ""
    artifact_root: str = ""
    platform: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        schema = str(self.sandbox_schema or "").strip()
        if not schema:
            raise VerificationProcessRunnerError(
                "sandbox schema is required",
                reason_code="sandbox_unavailable",
            )
        policy = {
            str(key): str(value)
            for key, value in dict(self.sandbox_policy or {}).items()
        }
        filesystem = {
            str(key): str(value)
            for key, value in dict(self.filesystem_policy or {}).items()
        }
        source = str(self.source_root or "").strip()
        artifacts = str(self.artifact_root or "").strip()
        if not source or not artifacts:
            raise VerificationProcessRunnerError(
                "sandbox source_root and artifact_root are required",
                reason_code="sandbox_unavailable",
            )
        try:
            source_path = Path(source).expanduser().resolve(strict=True)
            artifact_path = Path(artifacts).expanduser().resolve(strict=True)
        except OSError as exc:
            raise VerificationProcessRunnerError(
                "sandbox roots are unavailable",
                reason_code="sandbox_unavailable",
                details={"error": type(exc).__name__},
            ) from exc
        if not source_path.is_dir() or not artifact_path.is_dir():
            raise VerificationProcessRunnerError(
                "sandbox roots must be existing directories",
                reason_code="sandbox_unavailable",
            )
        object.__setattr__(self, "sandbox_schema", schema)
        object.__setattr__(self, "sandbox_policy", MappingProxyType(policy))
        object.__setattr__(self, "filesystem_policy", MappingProxyType(filesystem))
        object.__setattr__(self, "source_root", str(source_path))
        object.__setattr__(self, "artifact_root", str(artifact_path))
        object.__setattr__(
            self,
            "platform",
            MappingProxyType(
                {str(key): str(value) for key, value in dict(self.platform or {}).items()}
            ),
        )
        object.__setattr__(
            self,
            "extra",
            MappingProxyType(dict(self.extra or {})),
        )
        claimed = str(self.sandbox_id or "").strip()
        object.__setattr__(self, "sandbox_id", "")
        actual = _sha256_hex(_canonical_json_bytes(self.to_observation()))
        if claimed and claimed != actual:
            raise VerificationProcessRunnerError(
                "sandbox identity mismatch",
                reason_code="sandbox_unavailable",
                details={"claimed": claimed, "actual": actual},
            )
        object.__setattr__(self, "sandbox_id", actual)

    def is_closed_deny_default(self) -> bool:
        policy = dict(self.sandbox_policy)
        filesystem = dict(self.filesystem_policy)
        return (
            set(policy) == set(_CLOSED_SANDBOX_POLICY)
            and all(policy.get(name) == "deny" for name in ("network", "auto_install", "home_cache", "auth_material"))
            and policy.get("schema") == SANDBOX_POLICY_SCHEMA
            and set(filesystem) == set(_CLOSED_FILESYSTEM_POLICY)
            and filesystem.get("source") == "read_only"
            and filesystem.get("artifacts") == "private_writable"
            and filesystem.get("schema") == FILESYSTEM_POLICY_SCHEMA
        )

    def to_observation(self) -> dict[str, Any]:
        return {
            "schema": SANDBOX_IDENTITY_SCHEMA,
            "sandbox_schema": self.sandbox_schema,
            "sandbox_policy": dict(self.sandbox_policy),
            "filesystem_policy": dict(self.filesystem_policy),
            "source_root": self.source_root,
            "artifact_root": self.artifact_root,
            "platform": dict(self.platform),
            "extra": dict(self.extra),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_observation()
        payload["sandbox_id"] = self.sandbox_id
        return payload


@dataclass(frozen=True)
class VerificationCommand:
    """Explicit-argv process request with hermetic observations required.

    *argv* is the only execution form.  There is no shell string field and the
    runner never enables ``shell=True``.  *cwd*, *environment*, *timeout*,
    *sandbox*, and *network_policy* are all explicit and observed on the result.
    """

    argv: Sequence[str]
    cwd: str
    environment: Mapping[str, str]
    timeout_seconds: float
    sandbox: VerificationSandboxIdentity
    network_policy: str = NETWORK_POLICY_DENY_ALL
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES
    lane_id: str = ""
    resource_class: str = DEFAULT_RESOURCE_CLASS
    stage: str = DEFAULT_STAGE
    metadata: Mapping[str, str] = field(default_factory=dict)
    stdin: bytes | str | None = None

    def __post_init__(self) -> None:
        argv = _normalize_argv(self.argv)
        cwd = _require_absolute_existing_dir(self.cwd, field_name="cwd")
        env = _normalize_environment(self.environment)
        timeout = _normalize_timeout(self.timeout_seconds)
        network = str(self.network_policy or "").strip()
        if network != NETWORK_POLICY_DENY_ALL:
            raise VerificationProcessPolicyError(
                "network policy must be deny_all",
                reason_code="network_policy_denied",
                details={"network_policy": network},
            )
        if not isinstance(self.sandbox, VerificationSandboxIdentity):
            raise VerificationProcessRunnerError(
                "sandbox must be a VerificationSandboxIdentity",
                reason_code="sandbox_unavailable",
            )
        if not self.sandbox.is_closed_deny_default():
            raise VerificationProcessPolicyError(
                "sandbox policy must be the closed deny-by-default hermetic policy",
                reason_code="sandbox_policy_denied",
                details={"sandbox_policy": dict(self.sandbox.sandbox_policy)},
            )
        for name in ("max_stdout_bytes", "max_stderr_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise VerificationProcessBoundsError(
                    f"{name} must be a positive integer",
                    reason_code="bounds_exceeded",
                )
        cwd_path = Path(cwd)
        if not (
            _is_relative_to(cwd_path, Path(self.sandbox.source_root))
            or _is_relative_to(cwd_path, Path(self.sandbox.artifact_root))
        ):
            raise VerificationProcessPolicyError(
                "cwd must lie under the sandbox source_root or artifact_root",
                reason_code="cwd_escape",
                details={"cwd": cwd},
            )
        _reject_auto_install_argv(argv)
        _reject_forbidden_environment(env)
        stdin_bytes = _normalize_stdin(self.stdin)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "environment", MappingProxyType(env))
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "network_policy", network)
        object.__setattr__(
            self,
            "lane_id",
            str(self.lane_id or "").strip()
            or f"verification:{uuid.uuid4().hex[:16]}",
        )
        object.__setattr__(
            self,
            "resource_class",
            str(self.resource_class or DEFAULT_RESOURCE_CLASS).strip()
            or DEFAULT_RESOURCE_CLASS,
        )
        object.__setattr__(
            self,
            "stage",
            str(self.stage or DEFAULT_STAGE).strip() or DEFAULT_STAGE,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(key): str(value) for key, value in dict(self.metadata or {}).items()}
            ),
        )
        object.__setattr__(self, "stdin", stdin_bytes)

    @property
    def executable(self) -> str:
        return self.argv[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment": dict(self.environment),
            "timeout_seconds": self.timeout_seconds,
            "sandbox": self.sandbox.to_dict(),
            "network_policy": self.network_policy,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "lane_id": self.lane_id,
            "resource_class": self.resource_class,
            "stage": self.stage,
            "metadata": dict(self.metadata),
            "stdin_bytes": 0 if self.stdin is None else len(self.stdin),
        }


@dataclass(frozen=True)
class VerificationStreamArtifact:
    """Bounded stdout or stderr capture with deterministic digests."""

    digest: str
    cid: str
    truncated: bool
    byte_count: int
    captured_byte_count: int
    preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "cid": self.cid,
            "truncated": self.truncated,
            "byte_count": self.byte_count,
            "captured_byte_count": self.captured_byte_count,
            "preview": self.preview,
        }


class VerificationRunDisposition(str, Enum):
    """Runner-local disposition before adapters project receipts."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationRunResult:
    """Observed outcome of one admitted verification process run."""

    terminal_status: TerminalStatus
    disposition: VerificationRunDisposition
    exit_code: int | None
    duration_ms: int
    command_argv: tuple[str, ...]
    executable: str
    cwd: str
    environment: Mapping[str, str]
    sandbox: Mapping[str, Any]
    network_policy: str
    timeout_seconds: float
    stdout: VerificationStreamArtifact
    stderr: VerificationStreamArtifact
    process_started: bool
    publication_allowed: bool
    timed_out: bool = False
    cancelled: bool = False
    unavailable: bool = False
    pid: int | None = None
    process_group_id: int | None = None
    lease_id: str = ""
    cancellation_id: str = ""
    reason_codes: tuple[str, ...] = ()
    reason: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema: str = PROCESS_RUNNER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_status",
            TerminalStatus(self.terminal_status),
        )
        object.__setattr__(
            self,
            "disposition",
            VerificationRunDisposition(self.disposition),
        )
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(self.environment or {})),
        )
        object.__setattr__(
            self,
            "sandbox",
            MappingProxyType(dict(self.sandbox or {})),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(key): str(value) for key, value in dict(self.metadata or {}).items()}
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes),
        )
        object.__setattr__(self, "command_argv", tuple(str(item) for item in self.command_argv))

    @property
    def ok(self) -> bool:
        return (
            self.publication_allowed
            and self.terminal_status is TerminalStatus.PASSED
            and self.exit_code == 0
            and not self.timed_out
            and not self.cancelled
            and not self.unavailable
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence": [PROCESS_RUNNER_EVIDENCE, PROCESS_TREE_CANCELLATION_EVIDENCE],
            "terminal_status": self.terminal_status.value,
            "disposition": self.disposition.value,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "command_argv": list(self.command_argv),
            "executable": self.executable,
            "cwd": self.cwd,
            "environment": dict(self.environment),
            "sandbox": dict(self.sandbox),
            "network_policy": self.network_policy,
            "timeout_seconds": self.timeout_seconds,
            "stdout": self.stdout.to_dict(),
            "stderr": self.stderr.to_dict(),
            "process_started": self.process_started,
            "publication_allowed": self.publication_allowed,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "unavailable": self.unavailable,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "lease_id": self.lease_id,
            "cancellation_id": self.cancellation_id,
            "reason_codes": list(self.reason_codes),
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "ok": self.ok,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_for_bytes(data: bytes) -> str:
    return f"sha256:{_sha256_hex(data)}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (ValueError, OSError):
        return False


def _require_absolute_existing_dir(value: str, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VerificationProcessRunnerError(
            f"{field_name} is required",
            reason_code="invalid_command",
            details={"field": field_name},
        )
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise VerificationProcessPolicyError(
            f"{field_name} must be an absolute path",
            reason_code="cwd_not_absolute",
            details={"field": field_name, "value": text},
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise VerificationProcessRunnerError(
            f"{field_name} is unavailable",
            reason_code="cwd_unavailable",
            details={"field": field_name, "error": type(exc).__name__},
        ) from exc
    if not resolved.is_dir():
        raise VerificationProcessRunnerError(
            f"{field_name} must be a directory",
            reason_code="cwd_not_directory",
            details={"field": field_name},
        )
    return str(resolved)


def _normalize_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise VerificationProcessPolicyError(
            "argv must be a non-string sequence of strings (shell interpolation is impossible)",
            reason_code="shell_interpolation_impossible",
        )
    if not argv:
        raise VerificationProcessRunnerError(
            "argv must not be empty",
            reason_code="empty_argv",
        )
    if len(argv) > DEFAULT_MAX_ARGV_ITEMS:
        raise VerificationProcessBoundsError(
            f"argv exceeds {DEFAULT_MAX_ARGV_ITEMS} items",
            reason_code="bounds_exceeded",
        )
    normalized: list[str] = []
    for index, item in enumerate(argv):
        if not isinstance(item, str):
            raise VerificationProcessRunnerError(
                f"argv[{index}] must be a string",
                reason_code="invalid_argv",
                details={"index": index, "type": type(item).__name__},
            )
        if "\x00" in item:
            raise VerificationProcessPolicyError(
                "argv items must not contain NUL bytes",
                reason_code="invalid_argv",
                details={"index": index},
            )
        if len(item) > DEFAULT_MAX_ARGV_ITEM_CHARS:
            raise VerificationProcessBoundsError(
                f"argv[{index}] exceeds {DEFAULT_MAX_ARGV_ITEM_CHARS} characters",
                reason_code="bounds_exceeded",
            )
        if not item and index == 0:
            raise VerificationProcessRunnerError(
                "executable must be a non-empty string",
                reason_code="empty_executable",
            )
        normalized.append(item)
    return tuple(normalized)


def _normalize_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(environment, Mapping):
        raise VerificationProcessRunnerError(
            "environment must be a mapping",
            reason_code="invalid_environment",
        )
    if len(environment) > DEFAULT_MAX_ENV_KEYS:
        raise VerificationProcessBoundsError(
            f"environment exceeds {DEFAULT_MAX_ENV_KEYS} keys",
            reason_code="bounds_exceeded",
        )
    result: dict[str, str] = {}
    for raw_key, raw_value in environment.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise VerificationProcessRunnerError(
                "environment keys must be non-empty strings",
                reason_code="invalid_environment",
            )
        if "\x00" in raw_key:
            raise VerificationProcessPolicyError(
                "environment keys must not contain NUL bytes",
                reason_code="invalid_environment",
            )
        if not isinstance(raw_value, str):
            raise VerificationProcessRunnerError(
                f"environment value for {raw_key!r} must be a string",
                reason_code="invalid_environment",
            )
        if "\x00" in raw_value:
            raise VerificationProcessPolicyError(
                "environment values must not contain NUL bytes",
                reason_code="invalid_environment",
                details={"key": raw_key},
            )
        if len(raw_value) > DEFAULT_MAX_ENV_VALUE_CHARS:
            raise VerificationProcessBoundsError(
                f"environment value for {raw_key!r} exceeds bound",
                reason_code="bounds_exceeded",
            )
        result[raw_key] = raw_value
    return result


def _normalize_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise VerificationProcessRunnerError(
            "timeout_seconds must be a number",
            reason_code="invalid_timeout",
        )
    value = float(timeout_seconds)
    if value < DEFAULT_MIN_TIMEOUT_SECONDS:
        raise VerificationProcessRunnerError(
            f"timeout_seconds must be >= {DEFAULT_MIN_TIMEOUT_SECONDS}",
            reason_code="invalid_timeout",
        )
    if value > DEFAULT_MAX_TIMEOUT_SECONDS:
        raise VerificationProcessBoundsError(
            f"timeout_seconds exceeds {DEFAULT_MAX_TIMEOUT_SECONDS}",
            reason_code="bounds_exceeded",
        )
    return value


def _normalize_stdin(stdin: bytes | str | None) -> bytes | None:
    if stdin is None:
        return None
    if isinstance(stdin, str):
        return stdin.encode("utf-8")
    if isinstance(stdin, (bytes, bytearray, memoryview)):
        return bytes(stdin)
    raise VerificationProcessRunnerError(
        "stdin must be bytes, str, or None",
        reason_code="invalid_stdin",
    )


def _reject_auto_install_argv(argv: Sequence[str]) -> None:
    """Fail closed on install/package-manager command shapes.

    The runner never auto-installs tools.  Callers must point at reviewed
    executables already present on the hermetic filesystem.
    """

    if not argv:
        return
    tokens = [Path(item).name.lower() for item in argv[:6]]
    joined = " ".join(tokens)
    if tokens[0] in _AUTO_INSTALL_ARGV_MARKERS and "install" in tokens:
        raise VerificationProcessPolicyError(
            "auto-install commands are refused by the verification process runner",
            reason_code="auto_install_denied",
            details={"argv0": argv[0]},
        )
    if any(token in {"pip", "pip3"} for token in tokens) and "install" in tokens:
        raise VerificationProcessPolicyError(
            "pip install is refused by the verification process runner",
            reason_code="auto_install_denied",
            details={"argv_preview": joined},
        )
    if tokens[0] in {"npm", "yarn", "pnpm"} and any(
        token in {"install", "add", "ci"} for token in tokens[1:4]
    ):
        raise VerificationProcessPolicyError(
            "package-manager install is refused by the verification process runner",
            reason_code="auto_install_denied",
            details={"argv_preview": joined},
        )


def _reject_forbidden_environment(environment: Mapping[str, str]) -> None:
    for key in environment:
        lowered = key.strip().lower().replace("-", "_")
        if lowered in _FORBIDDEN_ENV_MARKERS or any(
            marker in lowered
            for marker in (
                "api_key",
                "access_token",
                "secret",
                "password",
                "private_key",
                "auth_token",
            )
        ):
            raise VerificationProcessPolicyError(
                "environment contains forbidden secret or network-widening keys",
                reason_code="forbidden_environment",
                details={"key": key},
            )


def _resolve_executable(executable: str) -> tuple[str | None, str | None]:
    """Return ``(resolved_path, unavailability_reason)``."""

    text = str(executable or "").strip()
    if not text:
        return None, "empty_executable"
    path = Path(text).expanduser()
    if not path.is_absolute():
        # Explicit absolute executable is required for hermetic observation.
        return None, "executable_not_absolute"
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None, "executable_missing"
    if not resolved.is_file():
        return None, "executable_not_file"
    if not os.access(resolved, os.X_OK) and os.name == "posix":
        # Still allow non-executable scripts only when the platform would; treat
        # lack of execute bit as unavailable for fail-closed hermetic runs.
        return None, "executable_not_executable"
    return str(resolved), None


def _spawn_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "shell": False,
        "close_fds": True,
    }
    if os.name == "nt":
        create_new = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = create_new
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _empty_stream_artifact() -> VerificationStreamArtifact:
    data = b""
    digest = _digest_for_bytes(data)
    return VerificationStreamArtifact(
        digest=digest,
        cid=cid_for_bytes(data),
        truncated=False,
        byte_count=0,
        captured_byte_count=0,
        preview="",
    )


def _capture_stream(
    handle: BinaryIO,
    *,
    maximum: int,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> VerificationStreamArtifact:
    """Read a stream with deterministic truncation and content digests.

    Digests and CIDs are computed over the *captured* (post-truncation) bytes so
    receipt identity is stable for the artifact actually retained.
    """

    handle.seek(0, os.SEEK_END)
    total = handle.tell()
    handle.seek(0)
    data = handle.read(maximum)
    truncated = total > len(data)
    digest = _digest_for_bytes(data)
    preview = data[:preview_chars].decode("utf-8", errors="replace")
    return VerificationStreamArtifact(
        digest=digest,
        cid=cid_for_bytes(data),
        truncated=truncated,
        byte_count=total,
        captured_byte_count=len(data),
        preview=preview,
    )


def _duration_ms(started: float, clock: Clock) -> int:
    elapsed = max(0.0, clock() - started)
    return int(elapsed * 1000)


def _terminal_for_disposition(
    disposition: VerificationRunDisposition,
    *,
    exit_code: int | None,
) -> TerminalStatus:
    if disposition is VerificationRunDisposition.TIMEOUT:
        return TerminalStatus.TIMEOUT
    if disposition is VerificationRunDisposition.CANCELLED:
        return TerminalStatus.CANCELLED
    if disposition is VerificationRunDisposition.UNAVAILABLE:
        return TerminalStatus.UNAVAILABLE
    if disposition is VerificationRunDisposition.FAILED:
        return TerminalStatus.FAILED
    if exit_code == 0:
        return TerminalStatus.PASSED
    return TerminalStatus.FAILED


def _default_host_snapshot() -> HostResourceSnapshot:
    return HostResourceSnapshot(
        worker_limit=32,
        available_worker_capacity=32,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=(
            "cpu-validation",
            "cpu-proof-type-check",
            "cpu-proof-solver",
            "cpu-small",
        ),
    )


# ---------------------------------------------------------------------------
# Process tree fencing
# ---------------------------------------------------------------------------


def fence_process_tree(
    process: Any,
    *,
    grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS,
    kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS,
    require_gone: bool = True,
) -> bool:
    """Terminate a process and all descendants, including escaped sessions.

    Uses the supervisor's identity-aware process-tree fence so children that
    called ``start_new_session`` (and their grandchildren) cannot outlive
    cancellation or timeout.
    """

    if process is None:
        return True
    poll_succeeded, returncode = _process_returncode(process)
    if poll_succeeded and returncode is not None:
        # Still walk descendants that may have been reparented before wait.
        pid = getattr(process, "pid", None)
        if pid is None:
            return True
        try:
            exited_tree_gone = terminate_pid_tree(
                int(pid),
                grace_seconds=grace_seconds,
                freeze_first=True,
                require_gone=require_gone,
                owned_process_group_id=int(pid) if os.name == "posix" else None,
            )
        except _PROCESS_HANDLE_ERRORS:
            exited_tree_gone = None
        if exited_tree_gone is not None:
            return exited_tree_gone

    pid = getattr(process, "pid", None)
    if pid is None:
        try:
            process.kill()
        except _PROCESS_HANDLE_ERRORS:
            return False
        return True

    root_pid = int(pid)
    gone = terminate_pid_tree(
        root_pid,
        grace_seconds=grace_seconds,
        freeze_first=True,
        require_gone=require_gone,
        owned_process_group_id=root_pid if os.name == "posix" else None,
    )
    # Best-effort wait on the Popen handle so zombie state is reaped here.
    deadline = time.monotonic() + max(0.0, kill_wait_seconds)
    while time.monotonic() < deadline:
        poll_succeeded, returncode = _process_returncode(process)
        if not poll_succeeded or returncode is not None:
            break
        time.sleep(0.02)
    _best_effort_wait(process, timeout=0.05)
    if require_gone and pid_alive(root_pid):
        try:
            if os.name == "posix":
                os.killpg(root_pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        return not pid_alive(root_pid)
    return bool(gone) or process.poll() is not None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class VerificationProcessRunner:
    """Admitted explicit-argv runner with lease, bounds, and tree fencing.

    Parameters
    ----------
    resource_scheduler:
        Optional shared :class:`ResourceScheduler`.  When omitted a private
        scheduler is created for this runner instance.
    host_snapshot:
        Optional host capacity snapshot used for every lease acquisition.
        Tests inject a generous snapshot; production callers may leave this
        ``None`` so the scheduler samples live host telemetry.
    popen_factory / clock / sleep:
        Injectable seams for deterministic tests.  Production uses
        :class:`subprocess.Popen` and wall clocks.
    """

    def __init__(
        self,
        *,
        resource_scheduler: ResourceScheduler | None = None,
        host_snapshot: HostResourceSnapshot | Mapping[str, Any] | None = None,
        popen_factory: PopenFactory | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
        term_grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS,
        kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        require_resource_lease: bool = True,
    ) -> None:
        self._scheduler = resource_scheduler or ResourceScheduler()
        self._host_snapshot = host_snapshot
        self._popen: PopenFactory = popen_factory or subprocess.Popen
        self._clock: Clock = clock or time.monotonic
        self._sleep: Sleep = sleep or time.sleep
        self._term_grace_seconds = float(term_grace_seconds)
        self._kill_wait_seconds = float(kill_wait_seconds)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._require_resource_lease = bool(require_resource_lease)

    # -- public API --------------------------------------------------------

    def run(
        self,
        command: VerificationCommand,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> VerificationRunResult:
        """Acquire a lease, execute *command*, and return a fenced result."""

        if not isinstance(command, VerificationCommand):
            raise VerificationProcessRunnerError(
                "command must be a VerificationCommand",
                reason_code="invalid_command",
            )
        started = self._clock()
        cancel = cancellation
        cancellation_id = cancel.cancellation_id if cancel is not None else ""

        def early(
            *,
            disposition: VerificationRunDisposition,
            reason_code: str,
            reason: str,
            exit_code: int | None = None,
            process_started: bool = False,
            lease_id: str = "",
            pid: int | None = None,
            process_group_id: int | None = None,
            publication_allowed: bool | None = None,
        ) -> VerificationRunResult:
            timed_out = disposition is VerificationRunDisposition.TIMEOUT
            cancelled = disposition is VerificationRunDisposition.CANCELLED
            unavailable = disposition is VerificationRunDisposition.UNAVAILABLE
            allow = (
                False
                if publication_allowed is None
                else publication_allowed
            )
            if publication_allowed is None:
                allow = not (cancelled or timed_out or unavailable)
                if disposition is VerificationRunDisposition.COMPLETED:
                    allow = True
                if disposition is VerificationRunDisposition.FAILED:
                    allow = True
            return VerificationRunResult(
                terminal_status=_terminal_for_disposition(
                    disposition, exit_code=exit_code
                ),
                disposition=disposition,
                exit_code=exit_code,
                duration_ms=_duration_ms(started, self._clock),
                command_argv=tuple(command.argv),
                executable=command.executable,
                cwd=command.cwd,
                environment=dict(command.environment),
                sandbox=command.sandbox.to_dict(),
                network_policy=command.network_policy,
                timeout_seconds=command.timeout_seconds,
                stdout=_empty_stream_artifact(),
                stderr=_empty_stream_artifact(),
                process_started=process_started,
                publication_allowed=allow,
                timed_out=timed_out,
                cancelled=cancelled,
                unavailable=unavailable,
                pid=pid,
                process_group_id=process_group_id,
                lease_id=lease_id,
                cancellation_id=cancellation_id,
                reason_codes=(reason_code,),
                reason=reason,
                metadata=dict(command.metadata),
            )

        # Pre-spawn cancellation fence.
        if cancel is not None and cancel.is_cancelled():
            return early(
                disposition=VerificationRunDisposition.CANCELLED,
                reason_code="cancelled_before_spawn",
                reason=cancel.reason or "cancelled before spawn",
                publication_allowed=False,
            )

        resolved_executable, missing_reason = _resolve_executable(command.executable)
        if resolved_executable is None:
            return early(
                disposition=VerificationRunDisposition.UNAVAILABLE,
                reason_code=missing_reason or "executable_missing",
                reason="verification executable is unavailable",
                publication_allowed=False,
            )
        # Rebuild argv with the observed absolute executable path.
        observed_argv = (resolved_executable, *command.argv[1:])

        if not command.sandbox.is_closed_deny_default():
            return early(
                disposition=VerificationRunDisposition.UNAVAILABLE,
                reason_code="sandbox_unavailable",
                reason="required hermetic sandbox is unavailable",
                publication_allowed=False,
            )

        lease: ResourceAdmissionLease | None = None
        lease_id = ""
        if self._require_resource_lease:
            lease = self._acquire_lease(command)
            if lease is None:
                return early(
                    disposition=VerificationRunDisposition.UNAVAILABLE,
                    reason_code="resource_lease_denied",
                    reason="resource admission denied the verification process lease",
                    publication_allowed=False,
                )
            lease_id = lease.lease_id

        process: Any = None
        stdout_file: Any = None
        stderr_file: Any = None
        process_started = False
        pid: int | None = None
        process_group_id: int | None = None
        timed_out = False
        cancelled = False
        exit_code: int | None = None
        stdout_artifact = _empty_stream_artifact()
        stderr_artifact = _empty_stream_artifact()
        reason_codes: list[str] = []
        reason = ""
        stream_files = ExitStack()

        try:
            # Re-check cancellation after lease acquisition (fence).
            if cancel is not None and cancel.is_cancelled():
                cancelled = True
                reason_codes.append("cancelled_before_spawn")
                reason = cancel.reason or "cancelled before spawn"
                return early(
                    disposition=VerificationRunDisposition.CANCELLED,
                    reason_code=reason_codes[0],
                    reason=reason,
                    lease_id=lease_id,
                    publication_allowed=False,
                )

            stdout_file = stream_files.enter_context(
                _temporary_binary_stream()
            )
            stderr_file = stream_files.enter_context(
                _temporary_binary_stream()
            )
            spawn_kwargs = _spawn_kwargs()
            # Shell is always False — never accept override via factory kwargs alone.
            spawn_kwargs["shell"] = False
            try:
                process = self._popen(
                    list(observed_argv),
                    cwd=command.cwd,
                    env=dict(command.environment),
                    stdin=subprocess.PIPE if command.stdin is not None else subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    **spawn_kwargs,
                )
            except FileNotFoundError:
                return early(
                    disposition=VerificationRunDisposition.UNAVAILABLE,
                    reason_code="executable_missing",
                    reason="verification executable was not found at spawn",
                    lease_id=lease_id,
                    publication_allowed=False,
                )
            except OSError as exc:
                return early(
                    disposition=VerificationRunDisposition.UNAVAILABLE,
                    reason_code="spawn_failed",
                    reason=f"process spawn failed: {type(exc).__name__}",
                    lease_id=lease_id,
                    publication_allowed=False,
                )

            process_started = True
            pid = int(getattr(process, "pid", 0) or 0) or None
            if pid is not None and os.name == "posix":
                process_group_id = pid

            if command.stdin is not None and process.stdin is not None:
                try:
                    process.stdin.write(command.stdin)
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass

            deadline = self._clock() + command.timeout_seconds
            while True:
                returncode = process.poll()
                if returncode is not None:
                    exit_code = int(returncode)
                    break
                if cancel is not None and cancel.is_cancelled():
                    cancelled = True
                    reason_codes.append("cancelled")
                    reason = cancel.reason or "cancelled"
                    fence_process_tree(
                        process,
                        grace_seconds=self._term_grace_seconds,
                        kill_wait_seconds=self._kill_wait_seconds,
                        require_gone=True,
                    )
                    _poll_succeeded, exit_code = _process_returncode(process)
                    break
                now = self._clock()
                if now >= deadline:
                    timed_out = True
                    reason_codes.append("timeout")
                    reason = "verification process exceeded its wall-time limit"
                    fence_process_tree(
                        process,
                        grace_seconds=self._term_grace_seconds,
                        kill_wait_seconds=self._kill_wait_seconds,
                        require_gone=True,
                    )
                    _poll_succeeded, exit_code = _process_returncode(process)
                    break
                remaining = deadline - now
                self._sleep(min(self._poll_interval_seconds, max(0.0, remaining)))

            # Capture bounded artifacts before publication fencing.
            if stdout_file is not None:
                stdout_artifact = _capture_stream(
                    stdout_file, maximum=command.max_stdout_bytes
                )
            if stderr_file is not None:
                stderr_artifact = _capture_stream(
                    stderr_file, maximum=command.max_stderr_bytes
                )

            # Late-success fence: cancellation wins even if exit_code == 0.
            if cancel is not None and cancel.is_cancelled():
                cancelled = True
                if "cancelled" not in reason_codes and "cancelled_before_spawn" not in reason_codes:
                    reason_codes.append("cancelled")
                if not reason:
                    reason = cancel.reason or "cancelled"
            if lease is not None and not self._lease_still_held(lease):
                # Lease revoked while running — fence publication.
                cancelled = True
                reason_codes.append("lease_revoked")
                reason = reason or "resource lease revoked before publication"
                if process is not None and process.poll() is None:
                    fence_process_tree(
                        process,
                        grace_seconds=self._term_grace_seconds,
                        kill_wait_seconds=self._kill_wait_seconds,
                        require_gone=True,
                    )

            if cancelled:
                disposition = VerificationRunDisposition.CANCELLED
                publication_allowed = False
            elif timed_out:
                disposition = VerificationRunDisposition.TIMEOUT
                publication_allowed = False
            elif exit_code == 0:
                disposition = VerificationRunDisposition.COMPLETED
                publication_allowed = True
            else:
                disposition = VerificationRunDisposition.FAILED
                publication_allowed = True
                if not reason_codes:
                    reason_codes.append("nonzero_exit")
                if not reason:
                    reason = f"process exited with code {exit_code}"

            if stdout_artifact.truncated:
                reason_codes.append("stdout_truncated")
            if stderr_artifact.truncated:
                reason_codes.append("stderr_truncated")

            return VerificationRunResult(
                terminal_status=_terminal_for_disposition(
                    disposition, exit_code=exit_code
                ),
                disposition=disposition,
                exit_code=exit_code,
                duration_ms=_duration_ms(started, self._clock),
                command_argv=observed_argv,
                executable=resolved_executable,
                cwd=command.cwd,
                environment=dict(command.environment),
                sandbox=command.sandbox.to_dict(),
                network_policy=command.network_policy,
                timeout_seconds=command.timeout_seconds,
                stdout=stdout_artifact,
                stderr=stderr_artifact,
                process_started=process_started,
                publication_allowed=publication_allowed,
                timed_out=timed_out,
                cancelled=cancelled,
                unavailable=False,
                pid=pid,
                process_group_id=process_group_id,
                lease_id=lease_id,
                cancellation_id=cancellation_id,
                reason_codes=tuple(dict.fromkeys(reason_codes)),
                reason=reason,
                metadata=dict(command.metadata),
            )
        finally:
            if process is not None and process.poll() is None:
                fence_process_tree(
                    process,
                    grace_seconds=self._term_grace_seconds,
                    kill_wait_seconds=self._kill_wait_seconds,
                    require_gone=False,
                )
            _close_stream_files(stream_files)
            if lease is not None:
                self._scheduler.release(
                    lease, reason="verification_process_complete"
                )

    # -- lease helpers -----------------------------------------------------

    def _acquire_lease(
        self, command: VerificationCommand
    ) -> ResourceAdmissionLease | None:
        requirement = LaneResourceRequirements(
            lane_id=command.lane_id,
            stage=command.stage,
            resource_class=command.resource_class,
            process_slots=1,
            requires_provider=False,
            memory_bytes=0,
            disk_bytes=0,
        )
        host = self._host_snapshot
        if host is None:
            host = _default_host_snapshot()
        elif not isinstance(host, HostResourceSnapshot):
            host = HostResourceSnapshot.from_mapping(host)
        _decision, lease = self._scheduler.acquire(requirement, host=host)
        return lease

    def _lease_still_held(self, lease: ResourceAdmissionLease) -> bool:
        active_leases = self._scheduler.active_leases
        if callable(active_leases):
            active_leases = active_leases()
        active = {item.lease_id for item in active_leases}
        return lease.lease_id in active


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_closed_sandbox(
    *,
    source_root: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    platform: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> VerificationSandboxIdentity:
    """Build the closed deny-by-default hermetic sandbox identity."""

    return VerificationSandboxIdentity(
        sandbox_schema=SANDBOX_SCHEMA,
        sandbox_policy=dict(_CLOSED_SANDBOX_POLICY),
        filesystem_policy=dict(_CLOSED_FILESYSTEM_POLICY),
        source_root=str(source_root),
        artifact_root=str(artifact_root),
        platform=dict(platform or {"schema": "verification-platform@1", "os": sys.platform}),
        extra=dict(extra or {}),
    )


def build_hermetic_environment(
    values: Mapping[str, str] | None = None,
    *,
    path: str | None = None,
) -> dict[str, str]:
    """Return a minimal deterministic environment for verification children.

    Secrets, proxy variables, and install-index URLs are never injected.
    """

    env: dict[str, str] = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LANGUAGE": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    if path is not None:
        env["PATH"] = str(path)
    if values:
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise VerificationProcessRunnerError(
                    "hermetic environment values must be strings",
                    reason_code="invalid_environment",
                )
            env[key] = value
    _reject_forbidden_environment(env)
    return env


__all__ = [
    "DEFAULT_MAX_STDERR_BYTES",
    "DEFAULT_MAX_STDOUT_BYTES",
    "NETWORK_POLICY_DENY_ALL",
    "PROCESS_RUNNER_EVIDENCE",
    "PROCESS_RUNNER_SCHEMA",
    "PROCESS_TREE_CANCELLATION_EVIDENCE",
    "VerificationCancellation",
    "VerificationCommand",
    "VerificationProcessBoundsError",
    "VerificationProcessPolicyError",
    "VerificationProcessRunner",
    "VerificationProcessRunnerError",
    "VerificationRunDisposition",
    "VerificationRunResult",
    "VerificationSandboxIdentity",
    "VerificationStreamArtifact",
    "build_closed_sandbox",
    "build_hermetic_environment",
    "fence_process_tree",
]
