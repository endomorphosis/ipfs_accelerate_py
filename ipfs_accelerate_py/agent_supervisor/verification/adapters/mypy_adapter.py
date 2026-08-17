"""Mypy verification adapter for incremental verification (IVP-006).

``MypyVerificationAdapter`` executes an explicit file/module/config selector
through the admitted :class:`VerificationProcessRunner`.  It never interpolates
a shell string, never installs packages, and never upgrades a timeout,
cancellation, or unavailability into a pass.

Authority rules
---------------
* Explicit argv (direct mypy executable or ``python -m mypy``) is the only
  execution form.
* Selector, config, observed environment, and tool identity must bind the
  supplied :class:`VerificationReceiptKey`.
* ``pass`` / ``fail`` / ``timeout`` / ``unavailable`` / ``cancelled`` map
  losslessly from the runner and exit/diagnostics projection.
* Usage errors and malformed diagnostic reports are ``invalid``.
* Missing mypy remains ``unavailable``; there is no success-on-unavailable
  fallback and no auto-install.
* Bounded diagnostics are retained as content-addressed artifacts.
* Tool executable, version, and configuration mutations change receipt keys
  (via the contracts compiler identity surface).
* Simulated mode is retained as ``simulated`` and never satisfies production.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    DirectExecutionObservation,
    TerminalStatus,
    TypeCheckReceipt,
    VerificationContractError,
    VerificationIdentityError,
    VerificationReceiptKey,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    PROCESS_RUNNER_EVIDENCE,
    PROCESS_TREE_CANCELLATION_EVIDENCE,
    VerificationCancellation,
    VerificationCommand,
    VerificationProcessRunner,
    VerificationProcessRunnerError,
    VerificationRunDisposition,
    VerificationRunResult,
    VerificationSandboxIdentity,
    build_hermetic_environment,
)

# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

MYPY_VERIFICATION_ADAPTER_INTERFACE: Final[str] = "MypyVerificationAdapter@1"
MYPY_VERIFICATION_ADAPTER_SCHEMA: Final[str] = "mypy-verification-adapter@1"
MYPY_ADAPTER_EVIDENCE: Final[str] = "ivp/mypy-adapter@1"
MYPY_DIAGNOSTICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mypy-diagnostics@1"
)
MYPY_DIAGNOSTICS_ACCOUNTING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mypy-diagnostics-accounting@1"
)
DEFAULT_DIAGNOSTICS_RELPATH: Final[str] = "mypy-diagnostics.json"

_MAX_SELECTORS: Final[int] = 10_000
_MAX_SELECTOR_CHARS: Final[int] = 2_048
_MAX_CONFIG_ARGS: Final[int] = 256
_MAX_REASON_CODES: Final[int] = 64
_MAX_DIAGNOSTICS: Final[int] = 10_000
_MAX_DIAGNOSTIC_MESSAGE_CHARS: Final[int] = 4_096

# mypy text diagnostic: path:line:col: severity: message [code]
_MYPY_DIAGNOSTIC_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<path>[^\x00\r\n:]+)"
    r":(?P<line>\d+)"
    r"(?::(?P<column>\d+))?"
    r":\s*(?P<severity>error|note|warning)"
    r":\s*(?P<message>.+?)"
    r"(?:\s+\[(?P<code>[^\]]+)\])?"
    r"\s*$"
)
_MYPY_SUMMARY_RE: Final[re.Pattern[str]] = re.compile(
    r"Found\s+(?P<errors>\d+)\s+error(?:s)?\s+in\s+(?P<files>\d+)\s+file",
    re.IGNORECASE,
)
_MYPY_SUCCESS_SUMMARY_RE: Final[re.Pattern[str]] = re.compile(
    r"Success:\s+no\s+issues\s+found\s+in\s+(?P<files>\d+)\s+source\s+file",
    re.IGNORECASE,
)
_USAGE_MARKERS: Final[tuple[str, ...]] = (
    "usage: mypy",
    "mypy: error: unrecognized arguments",
    "mypy: error: argument",
    "error: unused section(s):",
    "there are no .mypy.ini, setup.cfg, or pyproject.toml",
)

_REASON_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_.:/+-]{0,127}$"
)
_CLOSED_SEVERITIES: Final[frozenset[str]] = frozenset({"error", "note", "warning"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MypyVerificationAdapterError(ValueError):
    """Fail-closed adapter contract violation (pre-execution)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_request",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Request / result types
# ---------------------------------------------------------------------------


class MypyInvocation(str, Enum):
    """Closed invocation forms for the mypy adapter."""

    DIRECT = "direct"
    PYTHON_MODULE = "python_module"


class MypyRunMode(str, Enum):
    """Closed execution modes for the mypy adapter."""

    SELECTED_TARGETS = "selected_targets"
    PACKAGE_OR_MODULE = "package_or_module"


@dataclass(frozen=True)
class MypyDiagnostic:
    """One bounded mypy diagnostic line (artifact content, not authority)."""

    path: str
    line: int
    severity: str
    message: str
    column: int | None = None
    error_code: str = ""

    def __post_init__(self) -> None:
        path = str(self.path or "").strip()
        if not path or "\x00" in path or "\n" in path or "\r" in path:
            raise MypyVerificationAdapterError(
                "diagnostic path must be a non-empty single-line path",
                reason_code="invalid_diagnostic",
            )
        if len(path) > _MAX_SELECTOR_CHARS:
            raise MypyVerificationAdapterError(
                "diagnostic path exceeds bound",
                reason_code="bounds_exceeded",
            )
        object.__setattr__(self, "path", path)
        line = self.line
        if isinstance(line, bool) or not isinstance(line, int) or line < 0:
            raise MypyVerificationAdapterError(
                "diagnostic line must be a non-negative integer",
                reason_code="invalid_diagnostic",
            )
        column = self.column
        if column is not None and (
            isinstance(column, bool) or not isinstance(column, int) or column < 0
        ):
            raise MypyVerificationAdapterError(
                "diagnostic column must be a non-negative integer or None",
                reason_code="invalid_diagnostic",
            )
        severity = str(self.severity or "").strip().lower()
        if severity not in _CLOSED_SEVERITIES:
            raise MypyVerificationAdapterError(
                f"diagnostic severity must be one of {sorted(_CLOSED_SEVERITIES)}",
                reason_code="invalid_diagnostic",
            )
        object.__setattr__(self, "severity", severity)
        message = str(self.message or "").strip()
        if not message:
            raise MypyVerificationAdapterError(
                "diagnostic message must not be empty",
                reason_code="invalid_diagnostic",
            )
        if len(message) > _MAX_DIAGNOSTIC_MESSAGE_CHARS:
            message = message[:_MAX_DIAGNOSTIC_MESSAGE_CHARS]
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "error_code", str(self.error_code or "").strip())

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "message": self.message,
            "error_code": self.error_code,
            "is_error": self.is_error,
        }


@dataclass(frozen=True)
class MypyVerificationRequest:
    """One admitted mypy type-check verification request.

    *receipt_key* must already bind the selector argv that
    :meth:`MypyVerificationAdapter.build_argv` will emit for this request
    (tool name ``mypy``, matching configuration and environment CIDs).
    """

    receipt_key: VerificationReceiptKey
    mode: MypyRunMode
    sandbox: VerificationSandboxIdentity
    cwd: str
    timeout_seconds: float
    mypy_executable: str = ""
    python_executable: str = ""
    invocation: MypyInvocation = MypyInvocation.DIRECT
    paths: Sequence[str] = ()
    modules: Sequence[str] = ()
    packages: Sequence[str] = ()
    config_args: Sequence[str] = ()
    extra_mypy_args: Sequence[str] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    simulated: bool = False
    network_policy: str = NETWORK_POLICY_DENY_ALL
    max_stdout_bytes: int = 256 * 1024
    max_stderr_bytes: int = 256 * 1024
    diagnostics_relpath: str = DEFAULT_DIAGNOSTICS_RELPATH
    lane_id: str = ""
    resource_class: str = "cpu-validation"
    stage: str = "validation"
    metadata: Mapping[str, str] = field(default_factory=dict)
    # When set, used instead of reading the artifact/stdout report (tests inject).
    injected_diagnostics: Mapping[str, Any] | bytes | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_key, VerificationReceiptKey):
            raise MypyVerificationAdapterError(
                "receipt_key must be a VerificationReceiptKey",
                reason_code="invalid_receipt_key",
            )
        if self.receipt_key.receipt_kind is not VerificationReceiptKind.TYPE_CHECK:
            raise MypyVerificationAdapterError(
                "receipt_key must be receipt_kind=type_check",
                reason_code="invalid_receipt_kind",
            )
        if self.receipt_key.tool_name != "mypy":
            raise MypyVerificationAdapterError(
                "receipt_key.tool_name must be mypy",
                reason_code="invalid_tool",
            )
        if self.receipt_key.adapter_schema != MYPY_VERIFICATION_ADAPTER_SCHEMA:
            raise MypyVerificationAdapterError(
                "receipt_key.adapter_schema must be mypy-verification-adapter@1",
                reason_code="invalid_adapter_schema",
            )
        mode = (
            self.mode
            if isinstance(self.mode, MypyRunMode)
            else MypyRunMode(str(self.mode))
        )
        object.__setattr__(self, "mode", mode)
        invocation = (
            self.invocation
            if isinstance(self.invocation, MypyInvocation)
            else MypyInvocation(str(self.invocation))
        )
        object.__setattr__(self, "invocation", invocation)

        mypy_exe = str(self.mypy_executable or "").strip()
        python_exe = str(self.python_executable or "").strip()
        if invocation is MypyInvocation.DIRECT:
            if not mypy_exe:
                raise MypyVerificationAdapterError(
                    "mypy_executable is required for direct invocation",
                    reason_code="invalid_mypy_executable",
                )
            if not _is_absolute_path(mypy_exe):
                raise MypyVerificationAdapterError(
                    "mypy_executable must be absolute",
                    reason_code="invalid_mypy_executable",
                )
        else:
            if not python_exe:
                raise MypyVerificationAdapterError(
                    "python_executable is required for python_module invocation",
                    reason_code="invalid_python",
                )
            if not _is_absolute_path(python_exe):
                raise MypyVerificationAdapterError(
                    "python_executable must be absolute",
                    reason_code="invalid_python",
                )
        object.__setattr__(self, "mypy_executable", mypy_exe)
        object.__setattr__(self, "python_executable", python_exe)

        if not isinstance(self.sandbox, VerificationSandboxIdentity):
            raise MypyVerificationAdapterError(
                "sandbox must be a VerificationSandboxIdentity",
                reason_code="sandbox_unavailable",
            )
        cwd = str(self.cwd or "").strip()
        if not cwd:
            raise MypyVerificationAdapterError(
                "cwd is required",
                reason_code="invalid_cwd",
            )
        object.__setattr__(self, "cwd", cwd)
        timeout = float(self.timeout_seconds)
        if not (timeout > 0.0):
            raise MypyVerificationAdapterError(
                "timeout_seconds must be positive",
                reason_code="invalid_timeout",
            )
        object.__setattr__(self, "timeout_seconds", timeout)

        paths = _normalize_selectors(self.paths, field_name="paths")
        modules = _normalize_selectors(self.modules, field_name="modules")
        packages = _normalize_selectors(self.packages, field_name="packages")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "modules", modules)
        object.__setattr__(self, "packages", packages)
        config_args = _normalize_arg_sequence(
            self.config_args, field_name="config_args"
        )
        object.__setattr__(self, "config_args", config_args)
        extra = _normalize_arg_sequence(
            self.extra_mypy_args, field_name="extra_mypy_args"
        )
        object.__setattr__(self, "extra_mypy_args", extra)
        env = {
            str(key): str(value)
            for key, value in dict(self.environment or {}).items()
        }
        object.__setattr__(self, "environment", MappingProxyType(env))
        object.__setattr__(self, "simulated", bool(self.simulated))
        network = str(self.network_policy or "").strip() or NETWORK_POLICY_DENY_ALL
        if network != NETWORK_POLICY_DENY_ALL:
            raise MypyVerificationAdapterError(
                "network policy must be deny_all",
                reason_code="network_policy_denied",
            )
        object.__setattr__(self, "network_policy", network)
        report_path = str(
            self.diagnostics_relpath or DEFAULT_DIAGNOSTICS_RELPATH
        ).strip()
        if (
            not report_path
            or report_path.startswith("/")
            or ".." in PurePosixPath(report_path).parts
            or "\x00" in report_path
        ):
            raise MypyVerificationAdapterError(
                "diagnostics_relpath must be a sandbox-relative path",
                reason_code="invalid_diagnostics_path",
            )
        object.__setattr__(self, "diagnostics_relpath", report_path)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(k): str(v) for k, v in dict(self.metadata or {}).items()}
            ),
        )
        if not paths and not modules and not packages:
            raise MypyVerificationAdapterError(
                "at least one path, module, or package selector is required",
                reason_code="empty_selector",
            )
        if mode is MypyRunMode.PACKAGE_OR_MODULE and not modules and not packages:
            raise MypyVerificationAdapterError(
                "package_or_module mode requires modules or packages",
                reason_code="empty_selector",
            )


@dataclass(frozen=True)
class MypyVerificationResult:
    """Observed mypy verification outcome with retained argv and artifacts."""

    terminal_status: TerminalStatus
    receipt: TypeCheckReceipt | None
    command_argv: tuple[str, ...]
    mode: MypyRunMode
    invocation: MypyInvocation
    diagnostics: tuple[MypyDiagnostic, ...]
    error_count: int
    checked_files: int
    artifact_cids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    production_admissible: bool
    simulated: bool
    run_result: VerificationRunResult | None
    diagnostics_cid: str
    evidence: tuple[str, ...] = (
        MYPY_ADAPTER_EVIDENCE,
        PROCESS_RUNNER_EVIDENCE,
        PROCESS_TREE_CANCELLATION_EVIDENCE,
    )
    duration_ms: int = 0
    exit_code: int | None = None
    publication_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_status",
            TerminalStatus(self.terminal_status),
        )
        object.__setattr__(
            self,
            "mode",
            self.mode
            if isinstance(self.mode, MypyRunMode)
            else MypyRunMode(str(self.mode)),
        )
        object.__setattr__(
            self,
            "invocation",
            self.invocation
            if isinstance(self.invocation, MypyInvocation)
            else MypyInvocation(str(self.invocation)),
        )
        object.__setattr__(
            self,
            "command_argv",
            tuple(str(item) for item in self.command_argv),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "artifact_cids",
            tuple(str(item) for item in self.artifact_cids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )
        object.__setattr__(self, "production_admissible", bool(self.production_admissible))
        object.__setattr__(self, "simulated", bool(self.simulated))
        object.__setattr__(self, "publication_allowed", bool(self.publication_allowed))
        object.__setattr__(
            self,
            "evidence",
            tuple(str(item) for item in self.evidence),
        )

    @property
    def ok(self) -> bool:
        return (
            self.production_admissible
            and self.terminal_status is TerminalStatus.PASSED
            and self.receipt is not None
            and self.receipt.terminal_success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MYPY_VERIFICATION_ADAPTER_SCHEMA,
            "interface": MYPY_VERIFICATION_ADAPTER_INTERFACE,
            "evidence": list(self.evidence),
            "terminal_status": self.terminal_status.value,
            "receipt": self.receipt.to_record() if self.receipt is not None else None,
            "command_argv": list(self.command_argv),
            "mode": self.mode.value,
            "invocation": self.invocation.value,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "error_count": self.error_count,
            "checked_files": self.checked_files,
            "artifact_cids": list(self.artifact_cids),
            "reason_codes": list(self.reason_codes),
            "production_admissible": self.production_admissible,
            "simulated": self.simulated,
            "diagnostics_cid": self.diagnostics_cid,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "publication_allowed": self.publication_allowed,
            "ok": self.ok,
            "run_result": self.run_result.to_dict() if self.run_result else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_absolute_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if PurePosixPath(text).is_absolute():
        return True
    return Path(text).expanduser().is_absolute()


def _normalize_selectors(
    values: Sequence[str] | None, *, field_name: str
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MypyVerificationAdapterError(
            f"{field_name} must be a sequence of strings",
            reason_code="invalid_selectors",
        )
    if len(values) > _MAX_SELECTORS:
        raise MypyVerificationAdapterError(
            f"{field_name} exceeds {_MAX_SELECTORS} items",
            reason_code="bounds_exceeded",
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise MypyVerificationAdapterError(
                f"{field_name}[{index}] must be a string",
                reason_code="invalid_selectors",
            )
        item = raw.strip()
        if not item:
            raise MypyVerificationAdapterError(
                f"{field_name}[{index}] must not be empty",
                reason_code="invalid_selectors",
            )
        if "\x00" in item or "\n" in item or "\r" in item:
            raise MypyVerificationAdapterError(
                f"{field_name}[{index}] contains control characters",
                reason_code="invalid_selectors",
            )
        if len(item) > _MAX_SELECTOR_CHARS:
            raise MypyVerificationAdapterError(
                f"{field_name}[{index}] exceeds {_MAX_SELECTOR_CHARS} characters",
                reason_code="bounds_exceeded",
            )
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _normalize_arg_sequence(
    values: Sequence[str] | None, *, field_name: str
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MypyVerificationAdapterError(
            f"{field_name} must be a non-string sequence of strings",
            reason_code="invalid_args",
        )
    if len(values) > _MAX_CONFIG_ARGS:
        raise MypyVerificationAdapterError(
            f"{field_name} exceeds {_MAX_CONFIG_ARGS} items",
            reason_code="bounds_exceeded",
        )
    result: list[str] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise MypyVerificationAdapterError(
                f"{field_name}[{index}] must be a string",
                reason_code="invalid_args",
            )
        if "\x00" in raw:
            raise MypyVerificationAdapterError(
                f"{field_name}[{index}] must not contain NUL",
                reason_code="invalid_args",
            )
        result.append(raw)
    return tuple(result)


def build_mypy_argv(
    *,
    invocation: MypyInvocation = MypyInvocation.DIRECT,
    mypy_executable: str = "",
    python_executable: str = "",
    mode: MypyRunMode = MypyRunMode.SELECTED_TARGETS,
    paths: Sequence[str] = (),
    modules: Sequence[str] = (),
    packages: Sequence[str] = (),
    config_args: Sequence[str] = (),
    extra_mypy_args: Sequence[str] = (),
    cache_dir: str | None = None,
) -> tuple[str, ...]:
    """Build the explicit reproducible mypy argv sequence."""

    invocation = (
        invocation
        if isinstance(invocation, MypyInvocation)
        else MypyInvocation(str(invocation))
    )
    mode = mode if isinstance(mode, MypyRunMode) else MypyRunMode(str(mode))
    path_targets = (
        _normalize_selectors(paths, field_name="paths") if paths else ()
    )
    module_targets = (
        _normalize_selectors(modules, field_name="modules") if modules else ()
    )
    package_targets = (
        _normalize_selectors(packages, field_name="packages") if packages else ()
    )
    config = _normalize_arg_sequence(config_args, field_name="config_args")
    extra = _normalize_arg_sequence(extra_mypy_args, field_name="extra_mypy_args")

    if not path_targets and not module_targets and not package_targets:
        raise MypyVerificationAdapterError(
            "at least one path, module, or package selector is required",
            reason_code="empty_selector",
        )
    if mode is MypyRunMode.PACKAGE_OR_MODULE and not module_targets and not package_targets:
        raise MypyVerificationAdapterError(
            "package_or_module mode requires modules or packages",
            reason_code="empty_selector",
        )

    if invocation is MypyInvocation.DIRECT:
        executable = str(mypy_executable or "").strip()
        if not executable:
            raise MypyVerificationAdapterError(
                "mypy_executable is required for direct invocation",
                reason_code="invalid_mypy_executable",
            )
        argv: list[str] = [executable]
    else:
        python = str(python_executable or "").strip()
        if not python:
            raise MypyVerificationAdapterError(
                "python_executable is required for python_module invocation",
                reason_code="invalid_python",
            )
        argv = [python, "-m", "mypy"]

    argv.extend(config)

    # Reproducible hermetic defaults.  Callers may override via config/extra.
    present = set(config) | set(extra)
    default_flags: list[str] = []
    if "--no-incremental" not in present and "--incremental" not in present:
        default_flags.append("--no-incremental")
    if "--hide-error-context" not in present:
        default_flags.append("--hide-error-context")
    if cache_dir and "--cache-dir" not in present:
        default_flags.extend(["--cache-dir", str(cache_dir)])
    argv.extend(default_flags)
    argv.extend(extra)

    for package in package_targets:
        argv.extend(["-p", package])
    for module in module_targets:
        argv.extend(["-m", module])
    argv.extend(path_targets)
    return tuple(argv)


def encode_diagnostics_report(
    *,
    items: Sequence[Mapping[str, Any]] | Sequence[MypyDiagnostic] = (),
    exit_code: int | None = 0,
    checked_files: int | None = None,
    error_count: int | None = None,
    usage_error: bool = False,
    malformed: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical diagnostics-report mapping for injection or persistence."""

    serialized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, MypyDiagnostic):
            serialized.append(item.to_dict())
        else:
            serialized.append(dict(item))
    errors = (
        int(error_count)
        if error_count is not None
        else sum(1 for item in serialized if str(item.get("severity", "")).lower() == "error")
    )
    checked = (
        int(checked_files)
        if checked_files is not None
        else len({str(item.get("path") or "") for item in serialized if item.get("path")})
    )
    payload: dict[str, Any] = {
        "schema": MYPY_DIAGNOSTICS_SCHEMA,
        "exit_code": exit_code,
        "checked_files": checked,
        "error_count": errors,
        "items": serialized,
        "usage_error": bool(usage_error),
        "malformed": bool(malformed),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def parse_diagnostics_report(
    source: Mapping[str, Any] | bytes | str | None,
    *,
    fallback_exit_code: int | None = None,
) -> tuple[
    tuple[MypyDiagnostic, ...],
    int | None,
    int,
    int,
    bool,
    bool,
]:
    """Parse a diagnostics report or free-form mypy text.

    Returns
    ``(items, exit_code, error_count, checked_files, usage_error, malformed)``.
    """

    if source is None:
        return (), fallback_exit_code, 0, 0, False, True

    text: str | None = None
    payload: Any = None
    if isinstance(source, (bytes, bytearray)):
        try:
            text = bytes(source).decode("utf-8")
        except UnicodeDecodeError:
            return (), fallback_exit_code, 0, 0, False, True
        source = text
    if isinstance(source, str):
        text = source
        stripped = source.strip()
        if not stripped:
            # Empty output is valid free-form (exit code decides).
            return (), fallback_exit_code, 0, 0, False, False
        # Prefer structured JSON when present.
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            if start >= 0:
                try:
                    payload = json.loads(stripped[start:])
                except json.JSONDecodeError:
                    payload = None
            else:
                payload = None
        if payload is None:
            return _parse_mypy_text(text, fallback_exit_code=fallback_exit_code)
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        return (), fallback_exit_code, 0, 0, False, True

    if not isinstance(payload, Mapping):
        return (), fallback_exit_code, 0, 0, False, True

    schema = str(payload.get("schema") or "").strip()
    # Structured reports must use the diagnostics schema when a schema is set.
    if schema and schema != MYPY_DIAGNOSTICS_SCHEMA:
        return (), fallback_exit_code, 0, 0, False, True
    if payload.get("malformed") is True:
        return (), fallback_exit_code, 0, 0, False, True
    # Schema-less mappings that look like free-form bags without items are
    # treated as structured only when they declare known fields.
    structured_keys = {
        "items",
        "exit_code",
        "error_count",
        "checked_files",
        "usage_error",
        "malformed",
        "schema",
    }
    if not schema and not structured_keys.intersection(payload.keys()):
        return (), fallback_exit_code, 0, 0, False, True

    usage_error = bool(payload.get("usage_error"))
    exit_code = payload.get("exit_code", fallback_exit_code)
    if exit_code is not None:
        try:
            exit_code = int(exit_code)
        except (TypeError, ValueError):
            return (), fallback_exit_code, 0, 0, False, True
    try:
        checked_files = int(payload.get("checked_files", 0))
    except (TypeError, ValueError):
        return (), fallback_exit_code, 0, 0, False, True
    if checked_files < 0:
        return (), fallback_exit_code, 0, 0, False, True
    try:
        declared_errors = int(payload.get("error_count", -1))
    except (TypeError, ValueError):
        return (), fallback_exit_code, 0, 0, False, True

    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = ()
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return (), fallback_exit_code, 0, 0, False, True
    if len(raw_items) > _MAX_DIAGNOSTICS:
        return (), fallback_exit_code, 0, 0, False, True

    items: list[MypyDiagnostic] = []
    try:
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, Mapping):
                return (), fallback_exit_code, 0, 0, False, True
            column_raw = raw.get("column")
            column: int | None
            if column_raw is None or column_raw == "":
                column = None
            else:
                column = int(column_raw)
            items.append(
                MypyDiagnostic(
                    path=str(raw.get("path") or ""),
                    line=int(raw.get("line", 0)),
                    column=column,
                    severity=str(raw.get("severity") or "error"),
                    message=str(raw.get("message") or ""),
                    error_code=str(raw.get("error_code") or ""),
                )
            )
    except (TypeError, ValueError, MypyVerificationAdapterError):
        return (), fallback_exit_code, 0, 0, False, True

    error_count = (
        declared_errors
        if declared_errors >= 0
        else sum(1 for item in items if item.is_error)
    )
    return (
        tuple(items),
        exit_code,
        error_count,
        checked_files,
        usage_error,
        False,
    )


def _parse_mypy_text(
    text: str,
    *,
    fallback_exit_code: int | None,
) -> tuple[
    tuple[MypyDiagnostic, ...],
    int | None,
    int,
    int,
    bool,
    bool,
]:
    lower = text.lower()
    usage_error = any(marker in lower for marker in _USAGE_MARKERS)
    items: list[MypyDiagnostic] = []
    for line in text.splitlines():
        if len(items) >= _MAX_DIAGNOSTICS:
            break
        match = _MYPY_DIAGNOSTIC_RE.match(line.strip())
        if not match:
            continue
        try:
            column_text = match.group("column")
            items.append(
                MypyDiagnostic(
                    path=match.group("path"),
                    line=int(match.group("line")),
                    column=int(column_text) if column_text is not None else None,
                    severity=match.group("severity"),
                    message=match.group("message").strip(),
                    error_code=(match.group("code") or "").strip(),
                )
            )
        except (TypeError, ValueError, MypyVerificationAdapterError):
            continue

    error_count = sum(1 for item in items if item.is_error)
    checked_files = 0
    summary = _MYPY_SUMMARY_RE.search(text)
    if summary:
        try:
            error_count = max(error_count, int(summary.group("errors")))
            checked_files = int(summary.group("files"))
        except (TypeError, ValueError):
            pass
    else:
        success = _MYPY_SUCCESS_SUMMARY_RE.search(text)
        if success:
            try:
                checked_files = int(success.group("files"))
            except (TypeError, ValueError):
                checked_files = 0
        elif items:
            checked_files = len({item.path for item in items})

    return (
        tuple(items),
        fallback_exit_code,
        error_count,
        checked_files,
        usage_error,
        False,
    )


def project_terminal_status(
    *,
    run_result: VerificationRunResult | None,
    diagnostics: Sequence[MypyDiagnostic],
    exit_code: int | None,
    error_count: int,
    usage_error: bool,
    malformed: bool,
    simulated: bool,
) -> tuple[TerminalStatus, tuple[str, ...]]:
    """Project closed terminal status from runner + diagnostics/exit code."""

    reasons: list[str] = []

    if simulated:
        reasons.append("simulated_mode")
        return TerminalStatus.SIMULATED, tuple(reasons)

    if run_result is not None:
        if run_result.timed_out or run_result.disposition is VerificationRunDisposition.TIMEOUT:
            reasons.append("timeout")
            reasons.extend(run_result.reason_codes)
            return TerminalStatus.TIMEOUT, _unique_reasons(reasons)
        if run_result.cancelled or run_result.disposition is VerificationRunDisposition.CANCELLED:
            reasons.append("cancelled")
            reasons.extend(run_result.reason_codes)
            return TerminalStatus.CANCELLED, _unique_reasons(reasons)
        if (
            run_result.unavailable
            or run_result.disposition is VerificationRunDisposition.UNAVAILABLE
        ):
            reasons.append("unavailable")
            reasons.extend(run_result.reason_codes)
            return TerminalStatus.UNAVAILABLE, _unique_reasons(reasons)

    if usage_error:
        reasons.append("usage_error")
        return TerminalStatus.INVALID, tuple(reasons)
    if malformed:
        reasons.append("malformed_output")
        return TerminalStatus.INVALID, tuple(reasons)

    observed_exit = exit_code
    if observed_exit is None and run_result is not None:
        observed_exit = run_result.exit_code

    # mypy exit codes: 0 success, 1 type errors, 2 usage/blocked.
    if observed_exit == 2:
        reasons.append("usage_error")
        reasons.append("mypy_exit_2")
        return TerminalStatus.INVALID, _unique_reasons(reasons)

    if observed_exit is None:
        reasons.append("missing_exit_code")
        return TerminalStatus.INVALID, _unique_reasons(reasons)

    error_diagnostics = sum(1 for item in diagnostics if item.is_error)
    total_errors = max(int(error_count), error_diagnostics)

    if observed_exit == 0:
        if total_errors > 0:
            # Inconsistent tool report: do not manufacture a pass.
            reasons.append("exit_zero_with_errors")
            return TerminalStatus.INVALID, _unique_reasons(reasons)
        reasons.append("type_check_passed")
        return TerminalStatus.PASSED, _unique_reasons(reasons)

    if observed_exit == 1 or total_errors > 0:
        reasons.append("type_check_failed")
        if total_errors > 0:
            reasons.append(f"error_count:{min(total_errors, 9999)}")
        for item in diagnostics:
            if item.is_error:
                token = _path_reason_token(item.path)
                reasons.append(f"error:{token}:{item.line}")
                if len(reasons) >= _MAX_REASON_CODES - 2:
                    break
        return TerminalStatus.FAILED, _unique_reasons(reasons)

    reasons.append("nonzero_exit")
    reasons.append(f"exit_code:{observed_exit}")
    return TerminalStatus.FAILED, _unique_reasons(reasons)


def _path_reason_token(path: str) -> str:
    text = str(path or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_.:/+-]+", "_", text)
    cleaned = cleaned.strip("._:-/+") or "unknown"
    if not cleaned[0].isalpha():
        cleaned = "p_" + cleaned
    return cleaned[:96]


def _sanitize_reason_token(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if _REASON_TOKEN_RE.fullmatch(text) and len(text) <= 128:
        return text
    cleaned = re.sub(r"[^a-z0-9_.:/+-]+", "_", text)
    cleaned = cleaned.strip("._:-/+") or "reason"
    if not cleaned[0].isalpha():
        cleaned = "r_" + cleaned
    return cleaned[:128]


def _unique_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in reasons:
        text = _sanitize_reason_token(str(raw or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= _MAX_REASON_CODES:
            break
    return tuple(ordered)


def _exit_code_for_status(
    status: TerminalStatus, run_result: VerificationRunResult | None, observed: int | None
) -> int | None:
    if status in {
        TerminalStatus.TIMEOUT,
        TerminalStatus.CANCELLED,
        TerminalStatus.UNAVAILABLE,
        TerminalStatus.SIMULATED,
    }:
        if run_result is not None:
            return run_result.exit_code
        return None
    if status is TerminalStatus.PASSED:
        return 0
    if observed is not None:
        return observed if observed != 0 else 1
    if run_result is not None and run_result.exit_code is not None:
        return run_result.exit_code if run_result.exit_code != 0 else 1
    return 1


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MypyVerificationAdapter:
    """Execute explicit mypy file/module/config selectors via the shared runner."""

    interface: Final[str] = MYPY_VERIFICATION_ADAPTER_INTERFACE
    schema: Final[str] = MYPY_VERIFICATION_ADAPTER_SCHEMA
    evidence: Final[str] = MYPY_ADAPTER_EVIDENCE

    def __init__(
        self,
        process_runner: VerificationProcessRunner | None = None,
        *,
        require_production: bool = True,
    ) -> None:
        self._runner = process_runner or VerificationProcessRunner()
        self._require_production = bool(require_production)

    @property
    def process_runner(self) -> VerificationProcessRunner:
        return self._runner

    def build_argv(self, request: MypyVerificationRequest) -> tuple[str, ...]:
        """Return the reproducible explicit mypy argv list."""

        if not isinstance(request, MypyVerificationRequest):
            raise MypyVerificationAdapterError(
                "request must be a MypyVerificationRequest",
                reason_code="invalid_request",
            )
        cache_dir = str(Path(request.sandbox.artifact_root) / ".mypy_cache")
        return build_mypy_argv(
            invocation=request.invocation,
            mypy_executable=request.mypy_executable,
            python_executable=request.python_executable,
            mode=request.mode,
            paths=request.paths,
            modules=request.modules,
            packages=request.packages,
            config_args=request.config_args,
            extra_mypy_args=request.extra_mypy_args,
            cache_dir=cache_dir,
        )

    def execute(
        self,
        request: MypyVerificationRequest,
        *,
        cancellation: VerificationCancellation | None = None,
    ) -> MypyVerificationResult:
        """Run mypy (or project from injected diagnostics) and emit a TypeCheckReceipt."""

        if not isinstance(request, MypyVerificationRequest):
            raise MypyVerificationAdapterError(
                "request must be a MypyVerificationRequest",
                reason_code="invalid_request",
            )
        argv = self.build_argv(request)
        self._validate_selector_binding(request, argv)

        if request.simulated:
            return self._finalize(
                request=request,
                argv=argv,
                run_result=None,
                diagnostics=(),
                exit_code=None,
                error_count=0,
                checked_files=0,
                usage_error=False,
                malformed=False,
                forced_status=TerminalStatus.SIMULATED,
                extra_reasons=("simulated_mode",),
            )

        run_result: VerificationRunResult | None = None
        if request.injected_diagnostics is None:
            command = self._build_command(request, argv)
            try:
                run_result = self._runner.run(command, cancellation=cancellation)
            except VerificationProcessRunnerError as exc:
                return self._finalize(
                    request=request,
                    argv=argv,
                    run_result=None,
                    diagnostics=(),
                    exit_code=None,
                    error_count=0,
                    checked_files=0,
                    usage_error=False,
                    malformed=False,
                    forced_status=TerminalStatus.UNAVAILABLE,
                    extra_reasons=(
                        getattr(exc, "reason_code", None) or "runner_error",
                        "unavailable",
                    ),
                )
            # Preserve runner-terminal outcomes before report parsing.
            if (
                run_result.timed_out
                or run_result.cancelled
                or run_result.unavailable
                or run_result.disposition
                in {
                    VerificationRunDisposition.TIMEOUT,
                    VerificationRunDisposition.CANCELLED,
                    VerificationRunDisposition.UNAVAILABLE,
                }
            ):
                return self._finalize(
                    request=request,
                    argv=tuple(run_result.command_argv) or argv,
                    run_result=run_result,
                    diagnostics=(),
                    exit_code=run_result.exit_code,
                    error_count=0,
                    checked_files=0,
                    usage_error=False,
                    malformed=False,
                    forced_status=None,
                    extra_reasons=(),
                )
            observed_argv = tuple(run_result.command_argv) or argv
            report_source = self._load_diagnostics_report(request, run_result)
            fallback_exit = run_result.exit_code
        else:
            observed_argv = argv
            report_source = request.injected_diagnostics
            fallback_exit = None
            if isinstance(report_source, Mapping) and "exit_code" in report_source:
                try:
                    fallback_exit = int(report_source["exit_code"])  # type: ignore[arg-type]
                except (TypeError, ValueError, KeyError):
                    fallback_exit = None

        (
            diagnostics,
            exit_code,
            error_count,
            checked_files,
            usage_error,
            malformed,
        ) = parse_diagnostics_report(
            report_source, fallback_exit_code=fallback_exit
        )
        return self._finalize(
            request=request,
            argv=observed_argv,
            run_result=run_result,
            diagnostics=diagnostics,
            exit_code=exit_code,
            error_count=error_count,
            checked_files=checked_files,
            usage_error=usage_error,
            malformed=malformed,
            forced_status=None,
            extra_reasons=(),
        )

    # -- internals ---------------------------------------------------------

    def _validate_selector_binding(
        self,
        request: MypyVerificationRequest,
        argv: Sequence[str],
    ) -> None:
        key = request.receipt_key
        if request.invocation is MypyInvocation.DIRECT:
            if not argv or argv[0] != request.mypy_executable:
                raise MypyVerificationAdapterError(
                    "argv[0] must equal mypy_executable for direct invocation",
                    reason_code="invalid_argv_form",
                    details={"argv_preview": list(argv[:6])},
                )
            # Ensure mypy appears as the tool; reject shell-like forms.
            if any(token in {"&&", "||", ";", "|"} for token in argv):
                raise MypyVerificationAdapterError(
                    "argv must not contain shell operators",
                    reason_code="invalid_argv_form",
                )
        else:
            if (
                len(argv) < 3
                or argv[0] != request.python_executable
                or argv[1] != "-m"
                or argv[2] != "mypy"
            ):
                raise MypyVerificationAdapterError(
                    "argv must be explicit python -m mypy",
                    reason_code="invalid_argv_form",
                    details={"argv_preview": list(argv[:6])},
                )

        from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
            _SELECTOR_IDENTITY_INPUT_SCHEMA,
            _structured_cid,
        )

        try:
            observed_selector = _structured_cid(
                _SELECTOR_IDENTITY_INPUT_SCHEMA,
                {"argv": tuple(argv)},
                field_name="selector_argv",
            )
        except (VerificationContractError, VerificationIdentityError) as exc:
            raise MypyVerificationAdapterError(
                "selector argv cannot be identity-bound",
                reason_code="selector_binding_error",
                details={"error": str(exc)},
            ) from exc
        if observed_selector != key.selector_cid:
            raise MypyVerificationAdapterError(
                "built argv does not match receipt_key.selector_cid",
                reason_code="selector_binding_mismatch",
                details={
                    "expected_selector_cid": key.selector_cid,
                    "observed_selector_cid": observed_selector,
                },
            )
        env = key.environment_observation
        if env.get("tool_name") != "mypy":
            raise MypyVerificationAdapterError(
                "environment tool_name must be mypy",
                reason_code="environment_binding_mismatch",
            )
        if env.get("adapter_schema") != MYPY_VERIFICATION_ADAPTER_SCHEMA:
            raise MypyVerificationAdapterError(
                "environment adapter_schema mismatch",
                reason_code="environment_binding_mismatch",
            )
        if env.get("network_policy") != request.network_policy:
            raise MypyVerificationAdapterError(
                "environment network_policy mismatch",
                reason_code="environment_binding_mismatch",
            )

    def _build_command(
        self,
        request: MypyVerificationRequest,
        argv: Sequence[str],
    ) -> VerificationCommand:
        env = dict(request.environment)
        if not env:
            env = build_hermetic_environment()
        # Keep mypy cache under the private artifact root when possible.
        cache_dir = str(Path(request.sandbox.artifact_root) / ".mypy_cache")
        env.setdefault("MYPY_CACHE_DIR", cache_dir)
        return VerificationCommand(
            argv=list(argv),
            cwd=request.cwd,
            environment=env,
            timeout_seconds=request.timeout_seconds,
            sandbox=request.sandbox,
            network_policy=request.network_policy,
            max_stdout_bytes=request.max_stdout_bytes,
            max_stderr_bytes=request.max_stderr_bytes,
            lane_id=request.lane_id,
            resource_class=request.resource_class,
            stage=request.stage,
            metadata={
                **dict(request.metadata),
                "adapter": MYPY_VERIFICATION_ADAPTER_SCHEMA,
                "mode": request.mode.value,
                "invocation": request.invocation.value,
            },
        )

    def _load_diagnostics_report(
        self,
        request: MypyVerificationRequest,
        run_result: VerificationRunResult,
    ) -> Any:
        artifact_root = Path(request.sandbox.artifact_root)
        report_path = artifact_root / request.diagnostics_relpath
        if report_path.is_file():
            try:
                return report_path.read_bytes()
            except OSError:
                pass
        # Combine bounded stdout/stderr previews as free-form mypy text.
        parts: list[str] = []
        if run_result.stdout and run_result.stdout.preview:
            parts.append(run_result.stdout.preview)
        if run_result.stderr and run_result.stderr.preview:
            parts.append(run_result.stderr.preview)
        if parts:
            return "\n".join(parts)
        return ""

    def _finalize(
        self,
        *,
        request: MypyVerificationRequest,
        argv: Sequence[str],
        run_result: VerificationRunResult | None,
        diagnostics: Sequence[MypyDiagnostic],
        exit_code: int | None,
        error_count: int,
        checked_files: int,
        usage_error: bool,
        malformed: bool,
        forced_status: TerminalStatus | None,
        extra_reasons: Sequence[str],
    ) -> MypyVerificationResult:
        if forced_status is not None:
            status = forced_status
            reasons = _unique_reasons(extra_reasons)
        else:
            status, reasons = project_terminal_status(
                run_result=run_result,
                diagnostics=diagnostics,
                exit_code=exit_code,
                error_count=error_count,
                usage_error=usage_error,
                malformed=malformed,
                simulated=request.simulated,
            )
            if extra_reasons:
                reasons = _unique_reasons((*extra_reasons, *reasons))

        duration_ms = int(run_result.duration_ms) if run_result is not None else 0
        observation_exit = _exit_code_for_status(status, run_result, exit_code)

        stdout_cid = ""
        stderr_cid = ""
        artifact_cids: list[str] = []
        if run_result is not None:
            if run_result.stdout and run_result.stdout.cid:
                stdout_cid = run_result.stdout.cid
                artifact_cids.append(stdout_cid)
            if run_result.stderr and run_result.stderr.cid:
                stderr_cid = run_result.stderr.cid
                artifact_cids.append(stderr_cid)

        diagnostics_payload = encode_diagnostics_report(
            items=diagnostics,
            exit_code=observation_exit if observation_exit is not None else exit_code,
            checked_files=checked_files,
            error_count=error_count,
            usage_error=usage_error,
            malformed=malformed,
            extra={
                "terminal_status": status.value,
                "mode": request.mode.value,
                "invocation": request.invocation.value,
                "reason_codes": list(reasons),
            },
        )
        diagnostics_bytes = json.dumps(
            diagnostics_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        diagnostics_cid = cid_for_bytes(diagnostics_bytes)
        artifact_cids.append(diagnostics_cid)

        accounting_payload = {
            "schema": MYPY_DIAGNOSTICS_ACCOUNTING_SCHEMA,
            "items": [item.to_dict() for item in diagnostics],
            "error_count": error_count,
            "checked_files": checked_files,
        }
        accounting_cid = cid_for_bytes(
            json.dumps(
                accounting_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        artifact_cids.append(accounting_cid)

        if status in {
            TerminalStatus.TIMEOUT,
            TerminalStatus.CANCELLED,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.SIMULATED,
        }:
            if run_result is None or not run_result.process_started:
                observation_exit = None
                if not stdout_cid:
                    empty = cid_for_bytes(b"")
                    stdout_cid = stdout_cid or empty
                    stderr_cid = stderr_cid or empty
                    if empty not in artifact_cids:
                        artifact_cids.extend([stdout_cid, stderr_cid])
            elif not (stdout_cid and stderr_cid):
                empty = cid_for_bytes(b"")
                stdout_cid = stdout_cid or empty
                stderr_cid = stderr_cid or empty
        else:
            if observation_exit is None:
                observation_exit = 0 if status is TerminalStatus.PASSED else 1
            if status is TerminalStatus.PASSED:
                observation_exit = 0
            empty = cid_for_bytes(b"")
            stdout_cid = stdout_cid or empty
            stderr_cid = stderr_cid or empty
            for cid in (stdout_cid, stderr_cid):
                if cid not in artifact_cids:
                    artifact_cids.append(cid)

        deduped: list[str] = []
        seen_cids: set[str] = set()
        for cid in artifact_cids:
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                deduped.append(cid)

        key = request.receipt_key
        execution = DirectExecutionObservation(
            receipt_key_cid=key.key_id,
            repository_tree_cid=key.repository_tree_cid,
            environment_cid=key.environment_cid,
            repository_tree_observation=key.repository_tree_observation,
            environment_observation=key.environment_observation,
            terminal_status=status,
            command_argv=tuple(argv),
            duration_ms=duration_ms,
            exit_code=observation_exit,
            stdout_artifact_cid=stdout_cid,
            stderr_artifact_cid=stderr_cid,
            artifact_cids=tuple(deduped),
            reason_codes=reasons,
        )

        try:
            receipt = TypeCheckReceipt(
                key=key,
                execution=execution,
                artifact_cids=tuple(deduped),
                reason_codes=reasons,
            )
        except (VerificationContractError, VerificationIdentityError) as exc:
            raise MypyVerificationAdapterError(
                f"failed to project TypeCheckReceipt: {exc}",
                reason_code="receipt_projection_failed",
                details={"error": str(exc)},
            ) from exc

        production_admissible = (
            not request.simulated
            and status is TerminalStatus.PASSED
            and receipt is not None
            and receipt.terminal_success
            and receipt.status is TerminalStatus.PASSED
        )
        if self._require_production and request.simulated:
            production_admissible = False

        publication_allowed = True
        if run_result is not None:
            publication_allowed = bool(run_result.publication_allowed)
        if status in {
            TerminalStatus.CANCELLED,
            TerminalStatus.TIMEOUT,
            TerminalStatus.UNAVAILABLE,
            TerminalStatus.SIMULATED,
        }:
            if status is TerminalStatus.SIMULATED:
                publication_allowed = False
            elif run_result is not None:
                publication_allowed = bool(run_result.publication_allowed)
            else:
                publication_allowed = False

        return MypyVerificationResult(
            terminal_status=status,
            receipt=receipt,
            command_argv=tuple(argv),
            mode=request.mode,
            invocation=request.invocation,
            diagnostics=tuple(diagnostics),
            error_count=error_count,
            checked_files=checked_files,
            artifact_cids=tuple(deduped),
            reason_codes=reasons,
            production_admissible=production_admissible,
            simulated=request.simulated,
            run_result=run_result,
            diagnostics_cid=diagnostics_cid,
            duration_ms=duration_ms,
            exit_code=observation_exit,
            publication_allowed=publication_allowed,
        )


def create_mypy_verification_adapter(
    process_runner: VerificationProcessRunner | None = None,
    *,
    require_production: bool = True,
) -> MypyVerificationAdapter:
    """Factory for the production mypy verification adapter."""

    return MypyVerificationAdapter(
        process_runner=process_runner,
        require_production=require_production,
    )


__all__ = [
    "DEFAULT_DIAGNOSTICS_RELPATH",
    "MYPY_ADAPTER_EVIDENCE",
    "MYPY_DIAGNOSTICS_ACCOUNTING_SCHEMA",
    "MYPY_DIAGNOSTICS_SCHEMA",
    "MYPY_VERIFICATION_ADAPTER_INTERFACE",
    "MYPY_VERIFICATION_ADAPTER_SCHEMA",
    "MypyDiagnostic",
    "MypyInvocation",
    "MypyRunMode",
    "MypyVerificationAdapter",
    "MypyVerificationAdapterError",
    "MypyVerificationRequest",
    "MypyVerificationResult",
    "build_mypy_argv",
    "create_mypy_verification_adapter",
    "encode_diagnostics_report",
    "parse_diagnostics_report",
    "project_terminal_status",
]
