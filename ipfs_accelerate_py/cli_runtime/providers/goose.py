"""Canonical Goose CLI adapter: command construction, parsing, and errors.

This module is the sole owner of Goose argv construction, version capability
gating, JSON / stream-json parsing, and typed failure classification. Other
surfaces (llm_router, endpoints, MCP, ACP) must delegate here.

Safety invariants:

- Chat mode is always non-side-effecting: ``--no-session``, ``--no-profile``
  (when supported), JSON output, low max-turn and max-tool-repetition bounds,
  and **no** builtin or external extensions.
- Prompt text is supplied only via stdin through ``--instructions -`` / ``-i -``.
- Dynamic values remain single argv entries; shell execution is forbidden.
- Agent mode requires an explicit :class:`GooseAgentPolicy` and never runs
  under ordinary chat defaults.
- Router ``model_name`` maps to Goose's model; ``goose_provider`` maps to the
  underlying provider (``--provider`` / ``GOOSE_PROVIDER``). They are never
  collapsed into one field.
- Required safety flags cannot be silently omitted: unsupported versions fail
  closed when a required flag is missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from ..contracts import (
    MAX_EVENT_COUNT,
    MAX_METADATA_VALUE_CHARS,
    MAX_TEXT_CHARS,
    CLICapabilities,
    CLIEvent,
    CLIRequest,
    CLIResult,
    EventKind,
    ExecutionMode,
    ProviderSpec,
)
from ..errors import (
    CLIErrorRecord,
    CLIRuntimeError,
    CLIRuntimeErrorCode,
    ContractValidationError,
    InvalidStateError,
    MalformedOutputError,
    NonzeroExitError,
    PolicyDeniedError,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimeoutError,
)
from ..installers.goose import (
    GooseInstallResult,
    GooseReadiness,
    assess_goose_readiness,
    discover_goose,
    ensure_goose,
    goose_auth_available,
)
from ..process_runner import (
    CancellationToken,
    ProcessBounds,
    ProcessRunner,
    ProcessRunResult,
    ProcessSpec,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDER_NAME: str = "goose_cli"
PROVIDER_ALIASES: tuple[str, ...] = ("goose",)

# Chat defaults: deliberately low so ordinary generate_text cannot run away.
DEFAULT_CHAT_MAX_TURNS: int = 2
DEFAULT_CHAT_MAX_TOOL_REPETITIONS: int = 1
DEFAULT_AGENT_MAX_TURNS: int = 40
DEFAULT_AGENT_MAX_TOOL_REPETITIONS: int = 8
DEFAULT_CHAT_TIMEOUT_SECONDS: float = 180.0
DEFAULT_AGENT_TIMEOUT_SECONDS: float = 600.0

# Pinned release used by the installer; capability matrix is keyed off this.
PINNED_GOOSE_VERSION: str = "1.44.0"

# Flags that chat mode must never omit when the binary supports them.
REQUIRED_CHAT_SAFETY_FLAGS: tuple[str, ...] = (
    "--no-session",
    "--no-profile",
    "--output-format",
    "--max-turns",
    "--max-tool-repetitions",
    "--instructions",
)

# Output formats Goose accepts for structured automation.
OUTPUT_FORMAT_JSON: str = "json"
OUTPUT_FORMAT_STREAM_JSON: str = "stream-json"
OUTPUT_FORMAT_TEXT: str = "text"
ALLOWED_OUTPUT_FORMATS: frozenset[str] = frozenset(
    {OUTPUT_FORMAT_JSON, OUTPUT_FORMAT_STREAM_JSON, OUTPUT_FORMAT_TEXT}
)

# Approval modes Goose understands (``GOOSE_MODE`` / slash ``/mode``).
APPROVAL_MODES: frozenset[str] = frozenset(
    {"chat", "auto", "approve", "smart_approve"}
)

_VERSION_RE = re.compile(
    r"(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?",
)
_SENSITIVE_ENV_MARKERS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "keyring",
)

RunFn = Callable[..., Any]
WhichFn = Callable[[str], Optional[str]]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class GooseErrorKind(str, Enum):
    """Fine-grained Goose failure kinds (mapped onto CLIRuntimeErrorCode)."""

    NOT_INSTALLED = "not_installed"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNCONFIGURED_PROVIDER = "unconfigured_provider"
    AUTHENTICATION = "authentication"
    QUOTA_RATE_LIMIT = "quota_rate_limit"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIAL = "policy_denial"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    MALFORMED_OUTPUT = "malformed_output"
    NONZERO_EXIT = "nonzero_exit"
    SPAWN_FAILED = "spawn_failed"
    INTERNAL = "internal"


_KIND_TO_CODE: Mapping[GooseErrorKind, CLIRuntimeErrorCode] = {
    GooseErrorKind.NOT_INSTALLED: CLIRuntimeErrorCode.SPAWN_FAILED,
    GooseErrorKind.UNSUPPORTED_VERSION: CLIRuntimeErrorCode.UNSUPPORTED_CAPABILITY,
    GooseErrorKind.UNCONFIGURED_PROVIDER: CLIRuntimeErrorCode.INVALID_STATE,
    GooseErrorKind.AUTHENTICATION: CLIRuntimeErrorCode.AUTHENTICATION_FAILED,
    GooseErrorKind.QUOTA_RATE_LIMIT: CLIRuntimeErrorCode.CAPACITY_EXCEEDED,
    GooseErrorKind.APPROVAL_REQUIRED: CLIRuntimeErrorCode.POLICY_DENIED,
    GooseErrorKind.POLICY_DENIAL: CLIRuntimeErrorCode.POLICY_DENIED,
    GooseErrorKind.TIMEOUT: CLIRuntimeErrorCode.TIMEOUT,
    GooseErrorKind.CANCELLATION: CLIRuntimeErrorCode.CANCELLED,
    GooseErrorKind.MALFORMED_OUTPUT: CLIRuntimeErrorCode.MALFORMED_OUTPUT,
    GooseErrorKind.NONZERO_EXIT: CLIRuntimeErrorCode.NONZERO_EXIT,
    GooseErrorKind.SPAWN_FAILED: CLIRuntimeErrorCode.SPAWN_FAILED,
    GooseErrorKind.INTERNAL: CLIRuntimeErrorCode.INTERNAL,
}


def goose_error_code(kind: GooseErrorKind) -> CLIRuntimeErrorCode:
    """Map a Goose-specific kind onto the shared runtime error code."""
    return _KIND_TO_CODE.get(kind, CLIRuntimeErrorCode.INTERNAL)


def _clip(value: Any, maximum: int = MAX_METADATA_VALUE_CHARS) -> str:
    text = str("" if value is None else value)
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def make_goose_error(
    kind: GooseErrorKind,
    message: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    retryable: bool = False,
) -> CLIErrorRecord:
    """Build a bounded :class:`CLIErrorRecord` for a Goose failure."""
    payload: dict[str, str] = {"goose_error_kind": kind.value}
    if details:
        for key, raw in details.items():
            if len(payload) >= 32:
                break
            k = _clip(key, 128)
            if not k:
                continue
            lowered = k.lower()
            if any(marker in lowered for marker in _SENSITIVE_ENV_MARKERS):
                payload[k] = "[redacted]"
            else:
                payload[k] = _clip(raw)
    return CLIErrorRecord(
        code=goose_error_code(kind),
        message=_clip(message, 4096) or kind.value,
        retryable=retryable,
        details=payload,
    )


class GooseProviderError(CLIRuntimeError):
    """Raised by the Goose adapter with a typed :class:`GooseErrorKind`."""

    def __init__(
        self,
        message: str,
        *,
        kind: GooseErrorKind = GooseErrorKind.INTERNAL,
        details: Optional[Mapping[str, Any]] = None,
        retryable: bool = False,
        side_effects_started: bool = False,
    ) -> None:
        record = make_goose_error(
            kind, message, details=details, retryable=retryable
        )
        super().__init__(
            record.message,
            code=record.code,
            retryable=record.retryable,
            details=record.details,
        )
        self.kind = kind
        self.side_effects_started = bool(side_effects_started)


# ---------------------------------------------------------------------------
# Version capabilities / gating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GooseVersionCapabilities:
    """Which ``goose run`` safety flags a binary is known to support.

    When a required chat safety flag is unsupported the adapter fails closed
    rather than silently building a less-safe command.
    """

    version: str
    supports_no_session: bool = True
    supports_no_profile: bool = True
    supports_output_format: bool = True
    supports_max_turns: bool = True
    supports_max_tool_repetitions: bool = True
    supports_instructions_stdin: bool = True
    supports_provider_flag: bool = True
    supports_model_flag: bool = True
    supports_with_builtin: bool = True
    supports_with_extension: bool = True
    supports_quiet: bool = True
    supports_stream_json: bool = True

    def required_chat_flags(self) -> tuple[str, ...]:
        """Return the safety flags that must be present for chat mode."""
        return REQUIRED_CHAT_SAFETY_FLAGS

    def missing_required_chat_flags(self) -> tuple[str, ...]:
        """Return required chat flags that this version cannot supply."""
        missing: list[str] = []
        if not self.supports_no_session:
            missing.append("--no-session")
        if not self.supports_no_profile:
            missing.append("--no-profile")
        if not self.supports_output_format:
            missing.append("--output-format")
        if not self.supports_max_turns:
            missing.append("--max-turns")
        if not self.supports_max_tool_repetitions:
            missing.append("--max-tool-repetitions")
        if not self.supports_instructions_stdin:
            missing.append("--instructions")
        return tuple(missing)

    def ensure_chat_safe(self) -> None:
        """Fail closed when required chat safety flags cannot be applied."""
        missing = self.missing_required_chat_flags()
        if missing:
            raise GooseProviderError(
                "Goose version does not support required chat safety flags: "
                + ", ".join(missing),
                kind=GooseErrorKind.UNSUPPORTED_VERSION,
                details={
                    "version": self.version,
                    "missing_flags": ",".join(missing),
                },
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "supports_no_session": self.supports_no_session,
            "supports_no_profile": self.supports_no_profile,
            "supports_output_format": self.supports_output_format,
            "supports_max_turns": self.supports_max_turns,
            "supports_max_tool_repetitions": self.supports_max_tool_repetitions,
            "supports_instructions_stdin": self.supports_instructions_stdin,
            "supports_provider_flag": self.supports_provider_flag,
            "supports_model_flag": self.supports_model_flag,
            "supports_with_builtin": self.supports_with_builtin,
            "supports_with_extension": self.supports_with_extension,
            "supports_quiet": self.supports_quiet,
            "supports_stream_json": self.supports_stream_json,
        }


def parse_version_tuple(version: str) -> tuple[int, int, int]:
    """Parse a Goose version string into ``(major, minor, patch)``."""
    text = str(version or "").strip().lstrip("v")
    match = _VERSION_RE.search(text)
    if not match:
        return (0, 0, 0)
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch") or 0)
    return (major, minor, patch)


def capabilities_for_version(version: str) -> GooseVersionCapabilities:
    """Return known flag support for a Goose release.

    Capability matrix (conservative fail-closed floor):

    - ``>= 1.0.0``: ``--no-session``, ``--max-turns``, ``-i -``
    - ``>= 1.8.0``: ``--no-profile``, ``--output-format``, ``--provider``,
      ``--model``, ``--with-builtin``, ``--with-extension``, ``--quiet``
    - ``>= 1.12.0``: ``--max-tool-repetitions``, ``stream-json``

    Versions below 1.8.0 cannot satisfy chat safety requirements and will be
    rejected by :meth:`GooseVersionCapabilities.ensure_chat_safe`.
    """
    major, minor, patch = parse_version_tuple(version)
    ver = f"{major}.{minor}.{patch}" if version else "0.0.0"
    at_least_1 = (major, minor, patch) >= (1, 0, 0)
    at_least_1_8 = (major, minor, patch) >= (1, 8, 0)
    at_least_1_12 = (major, minor, patch) >= (1, 12, 0)
    return GooseVersionCapabilities(
        version=ver if version else "unknown",
        supports_no_session=at_least_1,
        supports_no_profile=at_least_1_8,
        supports_output_format=at_least_1_8,
        supports_max_turns=at_least_1,
        supports_max_tool_repetitions=at_least_1_12,
        supports_instructions_stdin=at_least_1,
        supports_provider_flag=at_least_1_8,
        supports_model_flag=at_least_1_8,
        supports_with_builtin=at_least_1_8,
        supports_with_extension=at_least_1_8,
        supports_quiet=at_least_1_8,
        supports_stream_json=at_least_1_12,
    )


def capabilities_from_help(help_text: str, *, version: str = "") -> GooseVersionCapabilities:
    """Derive capabilities by inspecting ``goose run --help`` output.

    Used by version-gate tests and optional live probes. Missing flags yield
    ``False`` so chat safety checks fail closed.
    """
    text = help_text or ""
    return GooseVersionCapabilities(
        version=version or "probed",
        supports_no_session="--no-session" in text,
        supports_no_profile="--no-profile" in text,
        supports_output_format="--output-format" in text,
        supports_max_turns="--max-turns" in text,
        supports_max_tool_repetitions="--max-tool-repetitions" in text,
        supports_instructions_stdin=(
            "--instructions" in text or "-i," in text or "-i " in text
        ),
        supports_provider_flag="--provider" in text,
        supports_model_flag="--model" in text,
        supports_with_builtin="--with-builtin" in text,
        supports_with_extension="--with-extension" in text,
        supports_quiet="--quiet" in text or "-q," in text,
        supports_stream_json="stream-json" in text,
    )


# ---------------------------------------------------------------------------
# Agent policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GooseAgentPolicy:
    """Explicit authorization record required for agent-mode execution.

    All path fields must be absolute after validation. Relative paths and
    missing allow_side_effects are rejected fail-closed.
    """

    allow_side_effects: bool
    cwd: str
    path_root: str
    approval_mode: str = "approve"
    session_id: Optional[str] = None
    resume_session: bool = False
    builtins: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    max_turns: int = DEFAULT_AGENT_MAX_TURNS
    max_tool_repetitions: int = DEFAULT_AGENT_MAX_TOOL_REPETITIONS
    timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS
    max_output_bytes: Optional[int] = None
    allowed_cwd_roots: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.allow_side_effects, bool):
            raise ContractValidationError("allow_side_effects must be a boolean")
        if not self.allow_side_effects:
            raise PolicyDeniedError(
                "agent mode requires allow_side_effects=True",
                details={"allow_side_effects": False},
            )

        cwd = _require_absolute_path(self.cwd, "cwd")
        path_root = _require_absolute_path(self.path_root, "path_root")
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "path_root", path_root)

        mode = str(self.approval_mode or "").strip().lower()
        if mode not in APPROVAL_MODES:
            raise ContractValidationError(
                f"approval_mode must be one of {sorted(APPROVAL_MODES)}",
                details={"approval_mode": self.approval_mode},
            )
        if mode == "chat":
            raise PolicyDeniedError(
                "agent policy cannot use approval_mode=chat",
                details={"approval_mode": mode},
            )
        object.__setattr__(self, "approval_mode", mode)

        if self.session_id is not None:
            sid = str(self.session_id).strip()
            if not sid:
                raise ContractValidationError("session_id must be non-empty when set")
            object.__setattr__(self, "session_id", sid)

        object.__setattr__(self, "builtins", _normalize_name_tuple(self.builtins, "builtin"))
        object.__setattr__(
            self, "extensions", _normalize_name_tuple(self.extensions, "extension")
        )

        max_turns = _require_positive_int(self.max_turns, "max_turns")
        max_reps = _require_positive_int(
            self.max_tool_repetitions, "max_tool_repetitions"
        )
        timeout = float(self.timeout_seconds)
        if timeout <= 0:
            raise ContractValidationError("timeout_seconds must be positive")
        object.__setattr__(self, "max_turns", max_turns)
        object.__setattr__(self, "max_tool_repetitions", max_reps)
        object.__setattr__(self, "timeout_seconds", timeout)

        if self.max_output_bytes is not None:
            if (
                not isinstance(self.max_output_bytes, int)
                or isinstance(self.max_output_bytes, bool)
                or self.max_output_bytes < 1
            ):
                raise ContractValidationError(
                    "max_output_bytes must be a positive integer when set"
                )

        roots = tuple(
            _require_absolute_path(item, "allowed_cwd_roots")
            for item in (self.allowed_cwd_roots or ())
        )
        object.__setattr__(self, "allowed_cwd_roots", roots)

        # Ensure cwd is under path_root and any configured roots.
        self._assert_path_under_root(cwd, path_root, field_name="cwd")
        for root in roots:
            # cwd must fall under at least one allowed root when roots given;
            # path_root itself is always required.
            pass
        if roots and not any(_is_relative_to(Path(cwd), Path(r)) for r in roots):
            raise PolicyDeniedError(
                "cwd is outside allowed_cwd_roots",
                details={"cwd": cwd, "allowed_cwd_roots": ",".join(roots)},
            )

        meta = {
            str(k): _clip(v)
            for k, v in dict(self.metadata or {}).items()
            if str(k).strip()
        }
        object.__setattr__(self, "metadata", meta)

    @staticmethod
    def _assert_path_under_root(
        path: str, root: str, *, field_name: str
    ) -> None:
        if not _is_relative_to(Path(path), Path(root)):
            raise PolicyDeniedError(
                f"{field_name} must be under path_root / GOOSE_PATH_ROOT",
                details={field_name: path, "path_root": root},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_side_effects": self.allow_side_effects,
            "cwd": self.cwd,
            "path_root": self.path_root,
            "approval_mode": self.approval_mode,
            "session_id": self.session_id,
            "resume_session": self.resume_session,
            "builtins": list(self.builtins),
            "extensions": list(self.extensions),
            "max_turns": self.max_turns,
            "max_tool_repetitions": self.max_tool_repetitions,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "allowed_cwd_roots": list(self.allowed_cwd_roots),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GooseAgentPolicy:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("agent policy must be a mapping")
        builtins = payload.get("builtins") or ()
        extensions = payload.get("extensions") or ()
        roots = payload.get("allowed_cwd_roots") or ()
        if isinstance(builtins, list):
            builtins = tuple(builtins)
        if isinstance(extensions, list):
            extensions = tuple(extensions)
        if isinstance(roots, list):
            roots = tuple(roots)
        return cls(
            allow_side_effects=bool(payload.get("allow_side_effects", False)),
            cwd=str(payload.get("cwd", "")),
            path_root=str(
                payload.get("path_root")
                or payload.get("GOOSE_PATH_ROOT")
                or payload.get("goose_path_root")
                or ""
            ),
            approval_mode=str(payload.get("approval_mode", "approve")),
            session_id=payload.get("session_id"),
            resume_session=bool(payload.get("resume_session", False)),
            builtins=builtins,
            extensions=extensions,
            max_turns=int(payload.get("max_turns", DEFAULT_AGENT_MAX_TURNS)),
            max_tool_repetitions=int(
                payload.get(
                    "max_tool_repetitions", DEFAULT_AGENT_MAX_TOOL_REPETITIONS
                )
            ),
            timeout_seconds=float(
                payload.get("timeout_seconds", DEFAULT_AGENT_TIMEOUT_SECONDS)
            ),
            max_output_bytes=payload.get("max_output_bytes"),
            allowed_cwd_roots=roots,
            metadata=payload.get("metadata") or {},
        )


def _require_absolute_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise PolicyDeniedError(
            f"{field_name} must be an absolute path",
            details={field_name: value},
        )
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        try:
            return os.path.commonpath([str(path), str(root)]) == str(root)
        except ValueError:
            return False


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"{field_name} must be a positive integer"
            ) from exc
    if value < 1:
        raise ContractValidationError(f"{field_name} must be >= 1")
    return value


def _normalize_name_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ContractValidationError(
            f"{field_name}s must be a sequence of names, not a string"
        )
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) > 64:
            raise ContractValidationError(f"too many {field_name}s")
    return tuple(out)


# ---------------------------------------------------------------------------
# Command / environment construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GooseCommandPlan:
    """Validated argv + environment + cwd for one Goose invocation."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Optional[str]
    mode: ExecutionMode
    model_name: Optional[str]
    goose_provider: Optional[str]
    output_format: str
    max_turns: int
    max_tool_repetitions: int
    side_effecting: bool
    metadata: Mapping[str, str] = field(default_factory=dict)

    def required_flags_present(self) -> bool:
        """Return True when every required chat safety flag is in argv."""
        if self.mode is not ExecutionMode.CHAT:
            return True
        joined = list(self.argv)
        # --instructions may appear as -i
        has_instructions = (
            "--instructions" in joined
            or "-i" in joined
        )
        checks = {
            "--no-session": "--no-session" in joined,
            "--no-profile": "--no-profile" in joined,
            "--output-format": "--output-format" in joined,
            "--max-turns": "--max-turns" in joined,
            "--max-tool-repetitions": "--max-tool-repetitions" in joined,
            "--instructions": has_instructions,
        }
        return all(checks.values())


def build_goose_command(
    *,
    executable: str,
    mode: ExecutionMode | str = ExecutionMode.CHAT,
    model_name: Optional[str] = None,
    goose_provider: Optional[str] = None,
    max_turns: Optional[int] = None,
    max_tool_repetitions: Optional[int] = None,
    output_format: str = OUTPUT_FORMAT_JSON,
    streaming: bool = False,
    quiet: bool = True,
    agent_policy: Optional[GooseAgentPolicy] = None,
    capabilities: Optional[GooseVersionCapabilities] = None,
    session_id: Optional[str] = None,
    builtins: Sequence[str] = (),
    extensions: Sequence[str] = (),
) -> GooseCommandPlan:
    """Build a fail-closed Goose ``run`` command plan.

    Chat mode never enables builtins/extensions and always requests JSON
    (or stream-json when streaming). Agent mode requires *agent_policy*.
    """
    binary = str(executable or "").strip()
    if not binary:
        raise GooseProviderError(
            "Goose executable path is empty",
            kind=GooseErrorKind.NOT_INSTALLED,
        )

    exec_mode = (
        mode
        if isinstance(mode, ExecutionMode)
        else ExecutionMode(str(mode).strip().lower())
    )
    caps = capabilities or capabilities_for_version(PINNED_GOOSE_VERSION)

    if exec_mode is ExecutionMode.CHAT:
        caps.ensure_chat_safe()
        if agent_policy is not None:
            raise InvalidStateError(
                "chat mode cannot accept an agent policy",
                details={"mode": "chat"},
            )
        if builtins or extensions:
            raise InvalidStateError(
                "chat mode cannot enable builtins or extensions",
                details={
                    "builtins": list(builtins),
                    "extensions": list(extensions),
                },
            )
        turns = int(max_turns if max_turns is not None else DEFAULT_CHAT_MAX_TURNS)
        reps = int(
            max_tool_repetitions
            if max_tool_repetitions is not None
            else DEFAULT_CHAT_MAX_TOOL_REPETITIONS
        )
        turns = max(1, min(turns, DEFAULT_CHAT_MAX_TURNS))
        reps = max(1, min(reps, DEFAULT_CHAT_MAX_TOOL_REPETITIONS))
        side_effecting = False
        cwd: Optional[str] = None
        approval_mode = "chat"
        path_root: Optional[str] = None
        effective_session: Optional[str] = None
        effective_builtins: tuple[str, ...] = ()
        effective_extensions: tuple[str, ...] = ()
    else:
        if agent_policy is None:
            raise PolicyDeniedError(
                "agent mode requires an explicit GooseAgentPolicy",
                details={"mode": "agent"},
            )
        if not agent_policy.allow_side_effects:
            raise PolicyDeniedError(
                "agent mode requires allow_side_effects=True",
            )
        turns = int(
            max_turns if max_turns is not None else agent_policy.max_turns
        )
        reps = int(
            max_tool_repetitions
            if max_tool_repetitions is not None
            else agent_policy.max_tool_repetitions
        )
        turns = max(1, turns)
        reps = max(1, reps)
        side_effecting = True
        cwd = agent_policy.cwd
        approval_mode = agent_policy.approval_mode
        path_root = agent_policy.path_root
        effective_session = session_id or agent_policy.session_id
        effective_builtins = tuple(builtins) or agent_policy.builtins
        effective_extensions = tuple(extensions) or agent_policy.extensions
        # Agent still needs the structural flags that exist.
        if not caps.supports_max_turns or not caps.supports_instructions_stdin:
            raise GooseProviderError(
                "Goose version lacks required agent execution flags",
                kind=GooseErrorKind.UNSUPPORTED_VERSION,
                details={"version": caps.version},
            )

    fmt = str(output_format or OUTPUT_FORMAT_JSON).strip().lower()
    if streaming and fmt == OUTPUT_FORMAT_JSON:
        fmt = OUTPUT_FORMAT_STREAM_JSON
    if fmt not in ALLOWED_OUTPUT_FORMATS:
        raise ContractValidationError(
            f"unsupported output format: {output_format!r}"
        )
    if fmt == OUTPUT_FORMAT_STREAM_JSON and not caps.supports_stream_json:
        raise GooseProviderError(
            "Goose version does not support stream-json output",
            kind=GooseErrorKind.UNSUPPORTED_VERSION,
            details={"version": caps.version},
        )
    if exec_mode is ExecutionMode.CHAT and fmt == OUTPUT_FORMAT_TEXT:
        raise InvalidStateError(
            "chat mode must request structured JSON output",
            details={"output_format": fmt},
        )

    model = (model_name or "").strip() or None
    provider = (goose_provider or "").strip() or None

    argv: list[str] = [binary, "run"]

    # Session / profile safety ------------------------------------------------
    if caps.supports_no_session:
        # Chat always; agent only when policy does not resume a session.
        if exec_mode is ExecutionMode.CHAT or not (
            agent_policy and agent_policy.resume_session and effective_session
        ):
            argv.append("--no-session")
    elif exec_mode is ExecutionMode.CHAT:
        # ensure_chat_safe already guards this; belt-and-suspenders.
        caps.ensure_chat_safe()

    if exec_mode is ExecutionMode.CHAT and caps.supports_no_profile:
        argv.append("--no-profile")

    if quiet and caps.supports_quiet:
        argv.append("--quiet")

    if caps.supports_max_turns:
        argv.extend(["--max-turns", str(turns)])
    if caps.supports_max_tool_repetitions:
        argv.extend(["--max-tool-repetitions", str(reps)])

    if caps.supports_output_format:
        argv.extend(["--output-format", fmt])

    if provider and caps.supports_provider_flag:
        argv.extend(["--provider", provider])
    if model and caps.supports_model_flag:
        argv.extend(["--model", model])

    # Agent-only extensions / builtins (never for chat).
    if exec_mode is ExecutionMode.AGENT:
        if effective_builtins and caps.supports_with_builtin:
            # Goose accepts comma-separated builtin names.
            argv.extend(["--with-builtin", ",".join(effective_builtins)])
        if effective_extensions and caps.supports_with_extension:
            for ext in effective_extensions:
                argv.extend(["--with-extension", ext])
        if (
            agent_policy
            and agent_policy.resume_session
            and effective_session
        ):
            argv.extend(["--resume", "--session-id", effective_session])
        elif effective_session and not (
            agent_policy and agent_policy.resume_session
        ):
            # Name a new session without resuming.
            argv.extend(["--name", effective_session])

    # Prompt always via stdin — never as a free-text argv blob.
    if caps.supports_instructions_stdin:
        argv.extend(["--instructions", "-"])
    else:
        raise GooseProviderError(
            "Goose version cannot accept instructions on stdin",
            kind=GooseErrorKind.UNSUPPORTED_VERSION,
            details={"version": caps.version},
        )

    env: dict[str, str] = {}
    if exec_mode is ExecutionMode.CHAT:
        env["GOOSE_MODE"] = "chat"
    else:
        env["GOOSE_MODE"] = approval_mode
    if path_root:
        env["GOOSE_PATH_ROOT"] = path_root
    if model:
        env["GOOSE_MODEL"] = model
    if provider:
        env["GOOSE_PROVIDER"] = provider

    meta = {
        "mode": exec_mode.value,
        "output_format": fmt,
        "max_turns": str(turns),
        "max_tool_repetitions": str(reps),
        "goose_version": caps.version,
    }
    if model:
        meta["model_name"] = model
    if provider:
        meta["goose_provider"] = provider

    plan = GooseCommandPlan(
        argv=tuple(argv),
        env=env,
        cwd=cwd,
        mode=exec_mode,
        model_name=model,
        goose_provider=provider,
        output_format=fmt,
        max_turns=turns,
        max_tool_repetitions=reps,
        side_effecting=side_effecting,
        metadata=meta,
    )

    if exec_mode is ExecutionMode.CHAT and not plan.required_flags_present():
        # Hard fail: safety flags must never be silently dropped.
        raise GooseProviderError(
            "chat command plan is missing required safety flags",
            kind=GooseErrorKind.UNSUPPORTED_VERSION,
            details={"argv": " ".join(argv[:12])},
        )
    return plan


def build_goose_process_env(
    plan: GooseCommandPlan,
    *,
    base_env: Optional[Mapping[str, str]] = None,
    extra: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, Optional[str]]:
    """Return a process-runner env overlay for *plan*.

    Values of ``None`` mean *remove this key* (process runner convention).
    Secrets already present in the base environment are left alone; this
    function never injects credential material of its own.
    """
    overlay: dict[str, Optional[str]] = dict(plan.env)
    if extra:
        for key, value in extra.items():
            overlay[str(key)] = None if value is None else str(value)
    # Ensure GOOSE_MODE from the plan wins over ambient GOOSE_MODE=auto.
    overlay["GOOSE_MODE"] = plan.env.get("GOOSE_MODE", "chat")
    _ = base_env  # base is applied by the process runner; kept for API clarity
    return overlay


# ---------------------------------------------------------------------------
# Structured parsers (no terminal-text cleanup)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GooseParsedOutput:
    """Bounded extraction from Goose JSON / stream-json stdout."""

    text: str
    events: tuple[CLIEvent, ...]
    metadata: Mapping[str, str]
    side_effects_started: bool
    tool_call_count: int
    message_count: int
    status: str
    raw_format: str

    def to_metadata(self) -> dict[str, str]:
        out = dict(self.metadata)
        out["side_effects_started"] = "true" if self.side_effects_started else "false"
        out["tool_call_count"] = str(self.tool_call_count)
        out["message_count"] = str(self.message_count)
        out["status"] = self.status
        out["raw_format"] = self.raw_format
        return out


def _extract_text_from_content(content: Any) -> str:
    """Pull concatenated text blocks from a Goose message content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                btype = str(block.get("type") or "").lower()
                if btype in {"text", "output_text", ""}:
                    text = block.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif "text" in block and block.get("text") is not None:
                    parts.append(str(block["text"]))
        return "".join(parts)
    if isinstance(content, Mapping):
        if "text" in content:
            return str(content.get("text") or "")
    return ""


def _message_has_tool_use(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, Mapping):
            continue
        btype = str(block.get("type") or "").lower()
        if btype in {
            "tool_use",
            "tool_call",
            "toolrequest",
            "tool_request",
            "function_call",
        }:
            return True
        if "toolCall" in block or "tool_call" in block or "toolUse" in block:
            return True
    return bool(message.get("toolCalls") or message.get("tool_calls"))


def parse_goose_json(
    stdout: str,
    *,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> GooseParsedOutput:
    """Parse Goose ``--output-format json`` stdout without terminal cleanup.

    The pinned Goose release emits::

        {
          "messages": [ {"role": "...", "content": [...] }, ... ],
          "metadata": { "status": "completed", "total_tokens": ... }
        }

    Final text is the last assistant message's text content. Tool activity
    sets ``side_effects_started``.
    """
    raw = stdout if isinstance(stdout, str) else str(stdout or "")
    # Do not strip ANSI or reflow text — only ignore pure surrounding whitespace
    # so JSON loaders can find the document boundary.
    payload_text = raw.strip()
    if not payload_text:
        raise MalformedOutputError(
            "Goose JSON output is empty",
            details={"format": OUTPUT_FORMAT_JSON},
        )
    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise MalformedOutputError(
            f"Goose JSON output is not valid JSON: {exc}",
            details={"format": OUTPUT_FORMAT_JSON, "error": str(exc)[:200]},
        ) from exc

    if not isinstance(data, Mapping):
        raise MalformedOutputError(
            "Goose JSON root must be an object",
            details={"format": OUTPUT_FORMAT_JSON},
        )

    messages = data.get("messages") or []
    if not isinstance(messages, list):
        raise MalformedOutputError(
            "Goose JSON messages must be a list",
            details={"format": OUTPUT_FORMAT_JSON},
        )

    assistant_texts: list[str] = []
    tool_call_count = 0
    events: list[CLIEvent] = []
    seq = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").lower()
        text = _extract_text_from_content(message.get("content"))
        has_tools = _message_has_tool_use(message)
        if has_tools:
            tool_call_count += 1
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.TOOL_CALL,
                        sequence=seq,
                        message="tool call observed",
                        side_effecting=True,
                    )
                )
                seq += 1
        if role == "assistant" and text:
            assistant_texts.append(text)
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.TEXT_DELTA,
                        sequence=seq,
                        message=_clip(text, 512),
                    )
                )
                seq += 1

    meta_raw = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    status = str(meta_raw.get("status") or "completed")
    text_out = assistant_texts[-1] if assistant_texts else ""
    if len(text_out) > max_text_chars:
        text_out = text_out[: max_text_chars - 3] + "..."

    bounded_meta: dict[str, str] = {}
    for key in (
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "status",
    ):
        if key in meta_raw and meta_raw[key] is not None:
            bounded_meta[key] = _clip(meta_raw[key])

    side_effects = tool_call_count > 0
    return GooseParsedOutput(
        text=text_out,
        events=tuple(events),
        metadata=bounded_meta,
        side_effects_started=side_effects,
        tool_call_count=tool_call_count,
        message_count=len(messages),
        status=status,
        raw_format=OUTPUT_FORMAT_JSON,
    )


def parse_goose_stream_json(
    stdout: str,
    *,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> GooseParsedOutput:
    """Parse Goose ``--output-format stream-json`` NDJSON without cleanup.

    Observed event shapes (pinned release)::

        {"type":"message","message":{...}}
        {"type":"complete","total_tokens":0,...}

    Final text is the last assistant message text; tool events flip
    ``side_effects_started``.
    """
    raw = stdout if isinstance(stdout, str) else str(stdout or "")
    if not raw.strip():
        raise MalformedOutputError(
            "Goose stream-json output is empty",
            details={"format": OUTPUT_FORMAT_STREAM_JSON},
        )

    assistant_texts: list[str] = []
    tool_call_count = 0
    events: list[CLIEvent] = []
    seq = 0
    status = "completed"
    bounded_meta: dict[str, str] = {}
    message_count = 0
    parsed_any = False

    for line_no, line in enumerate(raw.splitlines()):
        # Preserve content; only skip blank separators between NDJSON records.
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MalformedOutputError(
                f"Goose stream-json line {line_no} is not valid JSON: {exc}",
                details={
                    "format": OUTPUT_FORMAT_STREAM_JSON,
                    "line": str(line_no),
                },
            ) from exc
        if not isinstance(event, Mapping):
            raise MalformedOutputError(
                f"Goose stream-json line {line_no} must be an object",
                details={"format": OUTPUT_FORMAT_STREAM_JSON},
            )
        parsed_any = True
        etype = str(event.get("type") or "").lower()

        if etype == "message" or "message" in event:
            message = event.get("message") if etype == "message" else event
            if not isinstance(message, Mapping):
                continue
            message_count += 1
            role = str(message.get("role") or "").lower()
            text = _extract_text_from_content(message.get("content"))
            has_tools = _message_has_tool_use(message)
            if has_tools:
                tool_call_count += 1
                if len(events) < MAX_EVENT_COUNT:
                    events.append(
                        CLIEvent(
                            kind=EventKind.TOOL_CALL,
                            sequence=seq,
                            message="tool call observed",
                            side_effecting=True,
                        )
                    )
                    seq += 1
            if role == "assistant" and text:
                assistant_texts.append(text)
                if len(events) < MAX_EVENT_COUNT:
                    events.append(
                        CLIEvent(
                            kind=EventKind.TEXT_DELTA,
                            sequence=seq,
                            message=_clip(text, 512),
                        )
                    )
                    seq += 1
            # Some streams put deltas under type=message with partial text.
            if not role and text:
                assistant_texts.append(text)

        elif etype in {"complete", "completed", "result", "done"}:
            status = "completed"
            for key in (
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_write_input_tokens",
                "status",
            ):
                if key in event and event[key] is not None:
                    bounded_meta[key] = _clip(event[key])
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.COMPLETED,
                        sequence=seq,
                        message="stream complete",
                        payload=dict(bounded_meta),
                    )
                )
                seq += 1

        elif etype in {"tool", "tool_call", "tool_use", "tool_result"}:
            tool_call_count += 1
            kind = (
                EventKind.TOOL_RESULT
                if "result" in etype
                else EventKind.TOOL_CALL
            )
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=kind,
                        sequence=seq,
                        message=_clip(etype, 64),
                        side_effecting=True,
                    )
                )
                seq += 1

        elif etype in {"error", "failed"}:
            status = "failed"
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.FAILED,
                        sequence=seq,
                        message=_clip(event.get("message") or etype, 512),
                    )
                )
                seq += 1

        elif etype in {"text", "text_delta", "delta"}:
            delta = event.get("text") or event.get("delta") or event.get("content")
            if delta is not None:
                assistant_texts.append(str(delta))
                if len(events) < MAX_EVENT_COUNT:
                    events.append(
                        CLIEvent(
                            kind=EventKind.TEXT_DELTA,
                            sequence=seq,
                            message=_clip(delta, 512),
                        )
                    )
                    seq += 1

    if not parsed_any:
        raise MalformedOutputError(
            "Goose stream-json produced no parseable events",
            details={"format": OUTPUT_FORMAT_STREAM_JSON},
        )

    text_out = assistant_texts[-1] if assistant_texts else "".join(assistant_texts)
    # Prefer the last complete assistant message; if only deltas were collected
    # join them.
    if len(assistant_texts) > 1 and all(
        len(t) < 200 for t in assistant_texts
    ):
        # Heuristic: many small deltas → join; otherwise last full message wins.
        joined = "".join(assistant_texts)
        if len(joined) <= max_text_chars and not any(
            "\n" in t and len(t) > 80 for t in assistant_texts
        ):
            # Keep last full message when it looks complete.
            text_out = assistant_texts[-1]
    if len(text_out) > max_text_chars:
        text_out = text_out[: max_text_chars - 3] + "..."

    return GooseParsedOutput(
        text=text_out,
        events=tuple(events),
        metadata=bounded_meta,
        side_effects_started=tool_call_count > 0,
        tool_call_count=tool_call_count,
        message_count=message_count,
        status=status,
        raw_format=OUTPUT_FORMAT_STREAM_JSON,
    )


def parse_goose_output(
    stdout: str,
    *,
    output_format: str = OUTPUT_FORMAT_JSON,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> GooseParsedOutput:
    """Dispatch to the JSON or stream-json parser. Text format is rejected."""
    fmt = str(output_format or OUTPUT_FORMAT_JSON).strip().lower()
    if fmt == OUTPUT_FORMAT_STREAM_JSON:
        return parse_goose_stream_json(stdout, max_text_chars=max_text_chars)
    if fmt == OUTPUT_FORMAT_JSON:
        return parse_goose_json(stdout, max_text_chars=max_text_chars)
    raise MalformedOutputError(
        "refusing to parse unstructured Goose text output",
        details={"output_format": fmt},
    )


# ---------------------------------------------------------------------------
# Error classification from process results / stderr
# ---------------------------------------------------------------------------


_AUTH_MARKERS = (
    "authentication",
    "unauthorized",
    "401",
    "api key",
    "invalid api key",
    "missing api key",
    "not authenticated",
    "auth failed",
    "bearer",
)
_QUOTA_MARKERS = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "quota",
    "429",
    "capacity",
    "too many requests",
    "resource exhausted",
)
_UNCONFIGURED_MARKERS = (
    "no provider",
    "provider not configured",
    "not configured",
    "please configure",
    "run goose configure",
    "missing provider",
    "unknown provider",
    "provider is not set",
)
_APPROVAL_MARKERS = (
    "approval required",
    "awaiting approval",
    "waiting for approval",
    "user approval",
    "needs approval",
)
_POLICY_MARKERS = (
    "policy denied",
    "permission denied",
    "not allowed",
    "forbidden",
    "blocked by policy",
)


def classify_goose_failure(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: Optional[int] = None,
    timed_out: bool = False,
    cancelled: bool = False,
    process_started: bool = True,
    spawn_error: bool = False,
) -> tuple[GooseErrorKind, str, bool]:
    """Classify a Goose process failure into a stable kind.

    Returns ``(kind, message, retryable)``.
    """
    if cancelled:
        return GooseErrorKind.CANCELLATION, "Goose run was cancelled", False
    if timed_out:
        return GooseErrorKind.TIMEOUT, "Goose run timed out", False
    if spawn_error or not process_started:
        return (
            GooseErrorKind.NOT_INSTALLED
            if exit_code is None
            else GooseErrorKind.SPAWN_FAILED,
            "Goose process failed to start",
            False,
        )

    blob = f"{stderr}\n{stdout}".strip()
    lowered = blob.lower()
    message = _clip(blob or f"goose exited with code {exit_code}", 500)

    if any(m in lowered for m in _AUTH_MARKERS):
        return GooseErrorKind.AUTHENTICATION, message, False
    if any(m in lowered for m in _QUOTA_MARKERS):
        return GooseErrorKind.QUOTA_RATE_LIMIT, message, True
    if any(m in lowered for m in _APPROVAL_MARKERS):
        return GooseErrorKind.APPROVAL_REQUIRED, message, False
    if any(m in lowered for m in _UNCONFIGURED_MARKERS):
        return GooseErrorKind.UNCONFIGURED_PROVIDER, message, False
    if any(m in lowered for m in _POLICY_MARKERS):
        return GooseErrorKind.POLICY_DENIAL, message, False
    if "unsupported" in lowered and "version" in lowered:
        return GooseErrorKind.UNSUPPORTED_VERSION, message, False

    if exit_code not in (0, None):
        return GooseErrorKind.NONZERO_EXIT, message or f"exit {exit_code}", False
    return GooseErrorKind.INTERNAL, message or "unknown Goose failure", False


# ---------------------------------------------------------------------------
# Provider adapter
# ---------------------------------------------------------------------------


def goose_provider_spec() -> ProviderSpec:
    """Side-effect-free metadata for registry registration."""
    return ProviderSpec(
        name=PROVIDER_NAME,
        aliases=PROVIDER_ALIASES,
        description=(
            "Block/AAIF Goose CLI — chat-only by default; agent mode requires "
            "an explicit GooseAgentPolicy."
        ),
        capabilities=CLICapabilities.chat_defaults(),
        locality="local",
        metadata={
            "pinned_version": PINNED_GOOSE_VERSION,
            "default_chat_max_turns": str(DEFAULT_CHAT_MAX_TURNS),
            "default_chat_max_tool_repetitions": str(
                DEFAULT_CHAT_MAX_TOOL_REPETITIONS
            ),
        },
    )


@dataclass
class GooseCLIProvider:
    """Canonical Goose adapter implementing rich and string provider surfaces.

    Construction never installs software or starts processes. Discovery is
    detect-only unless ``allow_install=True`` is passed to an explicit resolve
    path such as :meth:`ensure_ready`.
    """

    executable: Optional[str] = None
    version: str = ""
    capabilities: Optional[GooseVersionCapabilities] = None
    runner: Optional[ProcessRunner] = None
    base_env: Optional[Mapping[str, str]] = None
    default_goose_provider: Optional[str] = None
    default_model: Optional[str] = None
    allow_install: bool = False
    managed_root: Optional[str] = None
    discover_kwargs: Mapping[str, Any] = field(default_factory=dict)
    _resolved: bool = field(default=False, init=False, repr=False)

    # -- discovery ---------------------------------------------------------

    def discover(
        self,
        *,
        explicit_path: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> GooseInstallResult:
        """Detect-only Goose discovery (never installs)."""
        merged = dict(self.discover_kwargs)
        merged.update(kwargs)
        if self.managed_root and "managed_root" not in merged:
            merged["managed_root"] = self.managed_root
        result = discover_goose(
            explicit_path=explicit_path or self.executable,
            environ=environ if environ is not None else self.base_env,
            **merged,
        )
        if result.available:
            self.executable = result.executable
            self.version = result.version or self.version
            if self.capabilities is None and self.version:
                self.capabilities = capabilities_for_version(self.version)
            self._resolved = True
        return result

    def ensure_ready(
        self,
        *,
        auto_install: Optional[bool] = None,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> GooseInstallResult:
        """Explicit readiness path; may install only when policy allows."""
        do_install = self.allow_install if auto_install is None else bool(auto_install)
        env = environ if environ is not None else self.base_env
        merged = dict(self.discover_kwargs)
        merged.update(kwargs)
        if self.managed_root and "managed_root" not in merged:
            merged["managed_root"] = self.managed_root
        if do_install:
            result = ensure_goose(
                explicit_path=self.executable,
                auto_install=True,
                environ=env,
                **merged,
            )
        else:
            result = discover_goose(
                explicit_path=self.executable,
                environ=env,
                **merged,
            )
        if result.available:
            self.executable = result.executable
            self.version = result.version or self.version
            if self.capabilities is None and self.version:
                self.capabilities = capabilities_for_version(self.version)
            self._resolved = True
        return result

    def readiness(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        auto_install: bool = False,
    ) -> GooseReadiness:
        """Combine binary discovery with authentication markers."""
        install = self.discover(environ=environ) if not auto_install else None
        return assess_goose_readiness(
            install_result=install,
            environ=environ if environ is not None else self.base_env,
            auto_install=auto_install and self.allow_install,
            **dict(self.discover_kwargs),
        )

    def _require_executable(self) -> str:
        if self.executable and str(self.executable).strip():
            return str(self.executable).strip()
        result = self.discover()
        if not result.available or not result.executable:
            raise GooseProviderError(
                "Goose CLI is not installed",
                kind=GooseErrorKind.NOT_INSTALLED,
                details={"reason": result.reason or "not_installed"},
            )
        return str(result.executable)

    def _capabilities(self) -> GooseVersionCapabilities:
        if self.capabilities is not None:
            return self.capabilities
        if self.version:
            return capabilities_for_version(self.version)
        return capabilities_for_version(PINNED_GOOSE_VERSION)

    # -- request execution -------------------------------------------------

    def generate_result(
        self,
        request: CLIRequest,
        *,
        agent_policy: Optional[GooseAgentPolicy] = None,
        goose_provider: Optional[str] = None,
        cancel_token: Optional[CancellationToken] = None,
        output_format: Optional[str] = None,
    ) -> CLIResult:
        """Execute a typed :class:`CLIRequest` and return a :class:`CLIResult`."""
        if not isinstance(request, CLIRequest):
            raise ContractValidationError("request must be a CLIRequest")

        mode = request.mode
        policy = agent_policy
        if mode is ExecutionMode.AGENT and policy is None:
            # Allow policy via request.metadata JSON keys.
            policy = self._policy_from_request(request)
        if mode is ExecutionMode.AGENT and policy is None:
            raise PolicyDeniedError(
                "agent mode requires an explicit GooseAgentPolicy",
                details={"mode": "agent"},
            )
        if mode is ExecutionMode.CHAT and (
            request.side_effecting or request.tools or request.session_id
        ):
            raise InvalidStateError(
                "chat Goose requests cannot be side-effecting or use tools/sessions"
            )

        provider = (
            goose_provider
            or request.provider_override
            or request.metadata.get("goose_provider")
            or self.default_goose_provider
        )
        model = request.effective_model() or self.default_model
        streaming = bool(request.streaming)
        fmt = (
            output_format
            or request.metadata.get("output_format")
            or (
                OUTPUT_FORMAT_STREAM_JSON
                if streaming
                else OUTPUT_FORMAT_JSON
            )
        )

        try:
            executable = self._require_executable()
            caps = self._capabilities()
            max_turns = None
            max_reps = None
            if "max_turns" in request.metadata:
                max_turns = int(request.metadata["max_turns"])
            if "max_tool_repetitions" in request.metadata:
                max_reps = int(request.metadata["max_tool_repetitions"])

            plan = build_goose_command(
                executable=executable,
                mode=mode,
                model_name=model,
                goose_provider=provider,
                max_turns=max_turns,
                max_tool_repetitions=max_reps,
                output_format=fmt,
                streaming=streaming,
                agent_policy=policy,
                capabilities=caps,
                session_id=request.session_id,
                builtins=request.tools if mode is ExecutionMode.AGENT else (),
            )
        except GooseProviderError:
            raise
        except (PolicyDeniedError, InvalidStateError, ContractValidationError):
            raise
        except CLIRuntimeError:
            raise
        except Exception as exc:
            raise GooseProviderError(
                f"Goose command construction failed: {exc}",
                kind=GooseErrorKind.INTERNAL,
            ) from exc

        timeout = request.timeout_seconds
        if timeout is None:
            timeout = (
                policy.timeout_seconds
                if policy is not None
                else DEFAULT_CHAT_TIMEOUT_SECONDS
            )

        env_overlay = build_goose_process_env(plan, base_env=self.base_env)
        bounds = None
        if policy is not None and policy.max_output_bytes is not None:
            bounds = ProcessBounds(
                max_stdout_bytes=int(policy.max_output_bytes),
                max_stderr_bytes=int(policy.max_output_bytes),
            )

        runner = self.runner or ProcessRunner(
            bounds=bounds,
            base_env=dict(self.base_env) if self.base_env is not None else None,
        )

        if request.cancellation_requested or (
            cancel_token is not None and cancel_token.is_cancelled()
        ):
            return self._error_result(
                request,
                kind=GooseErrorKind.CANCELLATION,
                message="Goose run cancelled before start",
                plan=plan,
                side_effects_started=False,
            )

        allowed_roots: Sequence[str] = ()
        if policy is not None:
            roots = list(policy.allowed_cwd_roots) or [policy.path_root]
            allowed_roots = roots

        spec = ProcessSpec(
            argv=plan.argv,
            cwd=plan.cwd or request.workspace,
            env=env_overlay,
            env_overlay=True,
            stdin=request.prompt,
            timeout_seconds=float(timeout),
            allowed_cwd_roots=allowed_roots,
            side_effecting=plan.side_effecting,
            mode=plan.mode,
            provider_name=PROVIDER_NAME,
            model_name=plan.model_name,
            metadata=dict(plan.metadata),
            cancel_token=cancel_token,
        )

        try:
            proc_result = runner.run(spec)
        except ProcessTimeoutError as exc:
            return self._error_result(
                request,
                kind=GooseErrorKind.TIMEOUT,
                message=str(exc),
                plan=plan,
                side_effects_started=plan.side_effecting,
            )
        except ProcessCancelledError as exc:
            return self._error_result(
                request,
                kind=GooseErrorKind.CANCELLATION,
                message=str(exc),
                plan=plan,
                side_effects_started=plan.side_effecting,
            )
        except ProcessSpawnError as exc:
            return self._error_result(
                request,
                kind=GooseErrorKind.NOT_INSTALLED,
                message=str(exc),
                plan=plan,
                side_effects_started=False,
            )
        except CLIRuntimeError as exc:
            kind = GooseErrorKind.INTERNAL
            if exc.code is CLIRuntimeErrorCode.POLICY_DENIED:
                kind = GooseErrorKind.POLICY_DENIAL
            return self._error_result(
                request,
                kind=kind,
                message=str(exc),
                plan=plan,
                side_effects_started=plan.side_effecting,
            )

        return self._result_from_process(request, plan, proc_result)

    def generate(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Legacy string surface. Chat-only unless agent kwargs are explicit."""
        agent = bool(
            kwargs.pop("agent", False)
            or kwargs.pop("side_effecting", False)
            or kwargs.pop("with_tools", False)
        )
        goose_provider = kwargs.pop("goose_provider", None) or kwargs.pop(
            "provider_override", None
        )
        timeout = kwargs.pop("timeout", None)
        workspace = kwargs.pop("workspace", None) or kwargs.pop("cwd", None)
        session_id = kwargs.pop("session_id", None)
        streaming = bool(kwargs.pop("stream", False) or kwargs.pop("streaming", False))
        max_turns = kwargs.pop("max_turns", None)
        max_tool_repetitions = kwargs.pop("max_tool_repetitions", None)
        output_format = kwargs.pop("output_format", None)
        cancel_token = kwargs.pop("cancel_token", None)
        policy_raw = kwargs.pop("agent_policy", None) or kwargs.pop("policy", None)
        path_root = kwargs.pop("path_root", None) or kwargs.pop(
            "GOOSE_PATH_ROOT", None
        )
        approval_mode = kwargs.pop("approval_mode", None)
        builtins = kwargs.pop("builtins", None) or kwargs.pop("with_builtin", None)
        extensions = kwargs.pop("extensions", None) or kwargs.pop(
            "with_extension", None
        )
        allow_side_effects = kwargs.pop("allow_side_effects", agent)

        # Swallow remaining unknown kwargs so callers can pass router noise.
        _ = kwargs

        policy: Optional[GooseAgentPolicy] = None
        if agent:
            if isinstance(policy_raw, GooseAgentPolicy):
                policy = policy_raw
            elif isinstance(policy_raw, Mapping):
                policy = GooseAgentPolicy.from_mapping(policy_raw)
            else:
                if not workspace or not path_root:
                    raise GooseProviderError(
                        "agent mode requires workspace/cwd and path_root "
                        "(GOOSE_PATH_ROOT) via GooseAgentPolicy",
                        kind=GooseErrorKind.POLICY_DENIAL,
                    )
                policy = GooseAgentPolicy(
                    allow_side_effects=bool(allow_side_effects),
                    cwd=str(workspace),
                    path_root=str(path_root),
                    approval_mode=str(approval_mode or "approve"),
                    session_id=session_id,
                    builtins=_coerce_seq(builtins),
                    extensions=_coerce_seq(extensions),
                    max_turns=int(max_turns or DEFAULT_AGENT_MAX_TURNS),
                    max_tool_repetitions=int(
                        max_tool_repetitions or DEFAULT_AGENT_MAX_TOOL_REPETITIONS
                    ),
                    timeout_seconds=float(
                        timeout or DEFAULT_AGENT_TIMEOUT_SECONDS
                    ),
                )

        metadata: dict[str, str] = {}
        if goose_provider:
            metadata["goose_provider"] = str(goose_provider)
        if max_turns is not None and not agent:
            metadata["max_turns"] = str(max_turns)
        if max_tool_repetitions is not None and not agent:
            metadata["max_tool_repetitions"] = str(max_tool_repetitions)
        if output_format:
            metadata["output_format"] = str(output_format)

        request = CLIRequest(
            prompt=str(prompt),
            mode=ExecutionMode.AGENT if agent else ExecutionMode.CHAT,
            model_name=model_name,
            provider_name=PROVIDER_NAME,
            provider_override=str(goose_provider) if goose_provider else None,
            side_effecting=bool(agent),
            cacheable=not agent,
            retryable=not agent,
            streaming=streaming,
            session_id=session_id if agent else None,
            tools=_coerce_seq(builtins) if agent else (),
            timeout_seconds=float(timeout) if timeout is not None else None,
            workspace=str(workspace) if workspace else None,
            metadata=metadata,
            capabilities=(
                CLICapabilities.agent_defaults()
                if agent
                else CLICapabilities.chat_defaults()
            ),
        )
        result = self.generate_result(
            request,
            agent_policy=policy,
            goose_provider=str(goose_provider) if goose_provider else None,
            cancel_token=cancel_token,
            output_format=str(output_format) if output_format else None,
        )
        if not result.ok:
            kind_name = (result.metadata or {}).get("goose_error_kind") or (
                result.error.details.get("goose_error_kind")
                if result.error
                else None
            )
            kind = GooseErrorKind.INTERNAL
            if kind_name:
                try:
                    kind = GooseErrorKind(kind_name)
                except ValueError:
                    kind = GooseErrorKind.INTERNAL
            raise GooseProviderError(
                result.error.message if result.error else "Goose run failed",
                kind=kind,
                details=dict(result.error.details) if result.error else {},
                side_effects_started=bool(result.had_side_effect_event),
            )
        return result.text

    def stream_events(
        self,
        request: CLIRequest,
        *,
        agent_policy: Optional[GooseAgentPolicy] = None,
        goose_provider: Optional[str] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> Iterator[CLIEvent]:
        """Run with stream-json and yield parsed events, then a completed event."""
        # Force streaming format through metadata.
        meta = dict(request.metadata)
        meta["output_format"] = OUTPUT_FORMAT_STREAM_JSON
        streaming_request = CLIRequest(
            prompt=request.prompt,
            mode=request.mode,
            model_name=request.model_name,
            provider_name=request.provider_name,
            provider_override=request.provider_override,
            model_override=request.model_override,
            side_effecting=request.side_effecting,
            cacheable=request.cacheable,
            retryable=request.retryable,
            streaming=True,
            session_id=request.session_id,
            tools=request.tools,
            cancellation_requested=request.cancellation_requested,
            timeout_seconds=request.timeout_seconds,
            workspace=request.workspace,
            metadata=meta,
            capabilities=request.capabilities,
        )
        result = self.generate_result(
            streaming_request,
            agent_policy=agent_policy,
            goose_provider=goose_provider,
            cancel_token=cancel_token,
            output_format=OUTPUT_FORMAT_STREAM_JSON,
        )
        yield from result.events
        if result.ok:
            yield CLIEvent(
                kind=EventKind.COMPLETED,
                sequence=len(result.events),
                message="goose completed",
                payload={
                    "side_effects_started": (
                        "true" if result.had_side_effect_event else "false"
                    )
                },
            )
        else:
            yield CLIEvent(
                kind=EventKind.FAILED,
                sequence=len(result.events),
                message=(
                    result.error.message if result.error else "goose failed"
                ),
            )

    # -- internals ---------------------------------------------------------

    def _policy_from_request(
        self, request: CLIRequest
    ) -> Optional[GooseAgentPolicy]:
        meta = request.metadata or {}
        if "agent_policy_json" in meta:
            try:
                payload = json.loads(meta["agent_policy_json"])
            except json.JSONDecodeError as exc:
                raise ContractValidationError(
                    "agent_policy_json is not valid JSON"
                ) from exc
            if isinstance(payload, Mapping):
                return GooseAgentPolicy.from_mapping(payload)
        if meta.get("allow_side_effects", "").lower() in {"1", "true", "yes"}:
            cwd = request.workspace or meta.get("cwd")
            path_root = meta.get("path_root") or meta.get("GOOSE_PATH_ROOT")
            if cwd and path_root:
                return GooseAgentPolicy(
                    allow_side_effects=True,
                    cwd=str(cwd),
                    path_root=str(path_root),
                    approval_mode=meta.get("approval_mode", "approve"),
                    session_id=request.session_id,
                    builtins=request.tools,
                    max_turns=int(
                        meta.get("max_turns", DEFAULT_AGENT_MAX_TURNS)
                    ),
                    max_tool_repetitions=int(
                        meta.get(
                            "max_tool_repetitions",
                            DEFAULT_AGENT_MAX_TOOL_REPETITIONS,
                        )
                    ),
                    timeout_seconds=float(
                        request.timeout_seconds
                        or DEFAULT_AGENT_TIMEOUT_SECONDS
                    ),
                )
        return None

    def _result_from_process(
        self,
        request: CLIRequest,
        plan: GooseCommandPlan,
        proc: ProcessRunResult,
    ) -> CLIResult:
        side_effects_started = bool(
            plan.side_effecting or proc.had_side_effect_event
        )
        meta: dict[str, str] = {
            **dict(plan.metadata),
            "process_started": "true" if proc.process_started else "false",
            "had_output": "true" if proc.had_output else "false",
            "exit_code": "" if proc.exit_code is None else str(proc.exit_code),
        }

        if proc.cancelled:
            kind, message, retryable = classify_goose_failure(cancelled=True)
            return self._error_result(
                request,
                kind=kind,
                message=message,
                plan=plan,
                side_effects_started=side_effects_started,
                exit_code=proc.exit_code,
                elapsed=proc.elapsed_seconds,
                retryable=retryable,
                extra_meta=meta,
            )
        if proc.timed_out:
            kind, message, retryable = classify_goose_failure(timed_out=True)
            return self._error_result(
                request,
                kind=kind,
                message=message,
                plan=plan,
                side_effects_started=side_effects_started,
                exit_code=proc.exit_code,
                elapsed=proc.elapsed_seconds,
                retryable=retryable,
                extra_meta=meta,
            )
        if not proc.process_started:
            kind, message, retryable = classify_goose_failure(
                process_started=False, spawn_error=True
            )
            return self._error_result(
                request,
                kind=kind,
                message=message,
                plan=plan,
                side_effects_started=False,
                exit_code=proc.exit_code,
                elapsed=proc.elapsed_seconds,
                retryable=retryable,
                extra_meta=meta,
            )

        # Prefer structured parse; classify auth etc. even on zero exit when
        # the assistant message embeds an authentication error (Goose often
        # exits 0 after surfacing provider failures as assistant text).
        parsed: Optional[GooseParsedOutput] = None
        parse_error: Optional[Exception] = None
        if proc.stdout and plan.output_format in {
            OUTPUT_FORMAT_JSON,
            OUTPUT_FORMAT_STREAM_JSON,
        }:
            try:
                parsed = parse_goose_output(
                    proc.stdout, output_format=plan.output_format
                )
            except MalformedOutputError as exc:
                parse_error = exc

        if parsed is not None:
            side_effects_started = side_effects_started or parsed.side_effects_started
            meta.update(parsed.to_metadata())
            # Classify embedded provider failures from assistant text / stderr.
            kind, message, retryable = classify_goose_failure(
                stdout=parsed.text,
                stderr=proc.stderr,
                exit_code=proc.exit_code,
                process_started=True,
            )
            auth_or_config = kind in {
                GooseErrorKind.AUTHENTICATION,
                GooseErrorKind.QUOTA_RATE_LIMIT,
                GooseErrorKind.UNCONFIGURED_PROVIDER,
                GooseErrorKind.APPROVAL_REQUIRED,
            }
            if auth_or_config or (proc.exit_code not in (0, None) and not proc.ok):
                if not auth_or_config:
                    kind, message, retryable = classify_goose_failure(
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        exit_code=proc.exit_code,
                        process_started=True,
                    )
                return self._error_result(
                    request,
                    kind=kind,
                    message=message,
                    plan=plan,
                    side_effects_started=side_effects_started,
                    exit_code=proc.exit_code,
                    elapsed=proc.elapsed_seconds,
                    retryable=retryable,
                    extra_meta=meta,
                    events=parsed.events,
                    text=parsed.text,
                )

            ok = proc.exit_code in (0, None) and parsed.status != "failed"
            if not ok:
                kind, message, retryable = classify_goose_failure(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.exit_code,
                    process_started=True,
                )
                return self._error_result(
                    request,
                    kind=kind,
                    message=message,
                    plan=plan,
                    side_effects_started=side_effects_started,
                    exit_code=proc.exit_code,
                    elapsed=proc.elapsed_seconds,
                    retryable=retryable,
                    extra_meta=meta,
                    events=parsed.events,
                    text=parsed.text,
                )

            return CLIResult(
                text=parsed.text,
                ok=True,
                mode=plan.mode,
                provider_name=PROVIDER_NAME,
                model_name=plan.model_name,
                side_effecting=side_effects_started or plan.side_effecting,
                cacheable=not (side_effects_started or plan.side_effecting),
                retryable=not (side_effects_started or plan.side_effecting),
                streaming=plan.output_format == OUTPUT_FORMAT_STREAM_JSON,
                truncated=proc.truncated_stdout or proc.truncated_stderr,
                cancelled=False,
                exit_code=proc.exit_code,
                elapsed_seconds=proc.elapsed_seconds,
                events=parsed.events,
                error=None,
                metadata=meta,
                had_side_effect_event=side_effects_started,
            )

        # No structured parse available.
        if parse_error is not None or (
            plan.output_format
            in {OUTPUT_FORMAT_JSON, OUTPUT_FORMAT_STREAM_JSON}
            and proc.ok
        ):
            return self._error_result(
                request,
                kind=GooseErrorKind.MALFORMED_OUTPUT,
                message=(
                    str(parse_error)
                    if parse_error
                    else "Goose produced no parseable structured output"
                ),
                plan=plan,
                side_effects_started=side_effects_started,
                exit_code=proc.exit_code,
                elapsed=proc.elapsed_seconds,
                extra_meta=meta,
            )

        kind, message, retryable = classify_goose_failure(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.exit_code,
            process_started=proc.process_started,
        )
        return self._error_result(
            request,
            kind=kind,
            message=message,
            plan=plan,
            side_effects_started=side_effects_started,
            exit_code=proc.exit_code,
            elapsed=proc.elapsed_seconds,
            retryable=retryable,
            extra_meta=meta,
        )

    def _error_result(
        self,
        request: CLIRequest,
        *,
        kind: GooseErrorKind,
        message: str,
        plan: Optional[GooseCommandPlan],
        side_effects_started: bool,
        exit_code: Optional[int] = None,
        elapsed: Optional[float] = None,
        retryable: bool = False,
        extra_meta: Optional[Mapping[str, str]] = None,
        events: Sequence[CLIEvent] = (),
        text: str = "",
    ) -> CLIResult:
        error = make_goose_error(
            kind,
            message,
            details={
                "mode": plan.mode.value if plan else request.mode.value,
                "side_effects_started": str(side_effects_started).lower(),
            },
            retryable=retryable and not side_effects_started,
        )
        meta: dict[str, str] = {
            "goose_error_kind": kind.value,
            "side_effects_started": "true" if side_effects_started else "false",
        }
        if plan is not None:
            meta.update(dict(plan.metadata))
        if extra_meta:
            meta.update(dict(extra_meta))
        meta["side_effects_started"] = "true" if side_effects_started else "false"
        return CLIResult(
            text=text,
            ok=False,
            mode=plan.mode if plan else request.mode,
            provider_name=PROVIDER_NAME,
            model_name=plan.model_name if plan else request.effective_model(),
            side_effecting=side_effects_started
            or (plan.side_effecting if plan else request.side_effecting),
            cacheable=False,
            retryable=False if side_effects_started else retryable,
            streaming=bool(request.streaming),
            truncated=False,
            cancelled=kind is GooseErrorKind.CANCELLATION,
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            events=tuple(events),
            error=error,
            metadata=meta,
            had_side_effect_event=side_effects_started,
        )


def _coerce_seq(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(parts)
    return _normalize_name_tuple(value, "item")


def create_goose_provider(
    *,
    executable: Optional[str] = None,
    allow_install: bool = False,
    runner: Optional[ProcessRunner] = None,
    base_env: Optional[Mapping[str, str]] = None,
    default_goose_provider: Optional[str] = None,
    default_model: Optional[str] = None,
    **discover_kwargs: Any,
) -> GooseCLIProvider:
    """Factory used by the lazy registry / llm_router integration."""
    return GooseCLIProvider(
        executable=executable,
        allow_install=allow_install,
        runner=runner,
        base_env=base_env,
        default_goose_provider=default_goose_provider,
        default_model=default_model,
        discover_kwargs=discover_kwargs,
    )


__all__ = [
    "PROVIDER_NAME",
    "PROVIDER_ALIASES",
    "DEFAULT_CHAT_MAX_TURNS",
    "DEFAULT_CHAT_MAX_TOOL_REPETITIONS",
    "DEFAULT_AGENT_MAX_TURNS",
    "DEFAULT_AGENT_MAX_TOOL_REPETITIONS",
    "DEFAULT_CHAT_TIMEOUT_SECONDS",
    "DEFAULT_AGENT_TIMEOUT_SECONDS",
    "PINNED_GOOSE_VERSION",
    "REQUIRED_CHAT_SAFETY_FLAGS",
    "OUTPUT_FORMAT_JSON",
    "OUTPUT_FORMAT_STREAM_JSON",
    "OUTPUT_FORMAT_TEXT",
    "ALLOWED_OUTPUT_FORMATS",
    "APPROVAL_MODES",
    "GooseErrorKind",
    "GooseProviderError",
    "GooseVersionCapabilities",
    "GooseAgentPolicy",
    "GooseCommandPlan",
    "GooseParsedOutput",
    "GooseCLIProvider",
    "goose_error_code",
    "make_goose_error",
    "parse_version_tuple",
    "capabilities_for_version",
    "capabilities_from_help",
    "build_goose_command",
    "build_goose_process_env",
    "parse_goose_json",
    "parse_goose_stream_json",
    "parse_goose_output",
    "classify_goose_failure",
    "goose_provider_spec",
    "create_goose_provider",
]
