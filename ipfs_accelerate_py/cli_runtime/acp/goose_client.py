"""Persistent Goose ACP (Agent Client Protocol) streaming client.

Manages a local ``goose acp`` stdio subprocess with:

- explicit executable path and isolated ``GOOSE_PATH_ROOT`` state root
- NDJSON JSON-RPC framing (newline-delimited, no embedded newlines)
- protocol ``initialize`` + capability validation before accepting work
- request/response correlation and session-scoped event routing
- hard bounds on pending requests, sessions, serialized bytes, output,
  idle time, and restarts
- cancellation that clears pending state and terminates the child tree
- unexpected exit → typed *uncertain side-effect* failure for in-flight work
- restart policy that recovers the *transport only* and never auto-replays
  agent prompts or session work
- endpoint-local session state (no shared process-wide session map)

This module never enables ``goose serve`` or any network listener.
Importing this module starts no processes.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from ..contracts import (
    MAX_EVENT_COUNT,
    MAX_EVENT_PAYLOAD_CHARS,
    MAX_PROMPT_CHARS,
    MAX_SERIALIZED_BYTES,
    MAX_SESSION_ID_CHARS,
    MAX_TEXT_CHARS,
    CLIEvent,
    EventKind,
    _clip_text,
)
from ..errors import (
    BoundsExceededError,
    CLIRuntimeError,
    CLIRuntimeErrorCode,
    ContractValidationError,
    InvalidStateError,
    MalformedOutputError,
    PolicyDeniedError,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimeoutError,
)
from ..process_runner import (
    DEFAULT_KILL_WAIT_SECONDS,
    DEFAULT_TERM_GRACE_SECONDS,
    terminate_process_tree,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

ACP_PROTOCOL_VERSION: int = 1
ACP_JSONRPC_VERSION: str = "2.0"
CLIENT_NAME: str = "ipfs_accelerate_py"
CLIENT_TITLE: str = "IPFS Accelerate Goose ACP Client"
CLIENT_VERSION: str = "1.0.0"

METHOD_INITIALIZE: str = "initialize"
METHOD_SESSION_NEW: str = "session/new"
METHOD_SESSION_LOAD: str = "session/load"
METHOD_SESSION_PROMPT: str = "session/prompt"
METHOD_SESSION_CANCEL: str = "session/cancel"
METHOD_SESSION_CLOSE: str = "session/close"
METHOD_SESSION_UPDATE: str = "session/update"
METHOD_SESSION_REQUEST_PERMISSION: str = "session/request_permission"

# Typed failure marker for unexpected process exit mid-work.
FAILURE_KIND_UNCERTAIN_SIDE_EFFECT: str = "uncertain_side_effect"
STATUS_UNCERTAIN_SIDE_EFFECT: str = "uncertain_side_effect"

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

DEFAULT_MAX_PENDING_REQUESTS: int = 32
DEFAULT_MAX_SESSIONS: int = 16
DEFAULT_MAX_SERIALIZED_BYTES: int = MAX_SERIALIZED_BYTES
DEFAULT_MAX_OUTPUT_BYTES: int = MAX_TEXT_CHARS
DEFAULT_MAX_IDLE_SECONDS: float = 300.0
DEFAULT_MAX_RESTARTS: int = 3
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 120.0
DEFAULT_INIT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_READ_CHUNK_BYTES: int = 65536
DEFAULT_EVENT_QUEUE_SIZE: int = 256
DEFAULT_STDERR_DIAGNOSTIC_CHARS: int = 1024

PopenFactory = Callable[..., Any]
Clock = Callable[[], float]
EventCallback = Callable[[Mapping[str, Any]], None]


# ---------------------------------------------------------------------------
# Errors (local; do not extend shared CLIRuntimeErrorCode enum)
# ---------------------------------------------------------------------------


class ACPError(CLIRuntimeError):
    """Base ACP client error."""

    def __init__(
        self,
        message: str,
        *,
        code: CLIRuntimeErrorCode | str = CLIRuntimeErrorCode.INTERNAL,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        failure_kind: str | None = None,
        uncertain_side_effects: bool = False,
    ) -> None:
        payload: dict[str, Any] = dict(details or {})
        if failure_kind is not None:
            payload.setdefault("failure_kind", failure_kind)
        if uncertain_side_effects:
            payload.setdefault("uncertain_side_effects", "true")
            payload.setdefault("failure_kind", FAILURE_KIND_UNCERTAIN_SIDE_EFFECT)
        super().__init__(
            message,
            code=code,
            retryable=retryable,
            details=payload,
        )
        self.failure_kind = failure_kind or payload.get("failure_kind")
        self.uncertain_side_effects = bool(uncertain_side_effects)


class ACPNotReadyError(ACPError):
    def __init__(self, message: str = "ACP client is not ready", **kwargs: Any) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.INVALID_STATE,
            retryable=False,
            **kwargs,
        )


class ACPUncertainSideEffectError(ACPError):
    """Raised when the ACP child exits while agent work may have started."""

    def __init__(
        self,
        message: str = "ACP process exited with uncertain side effects",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.INTERNAL,
            retryable=False,
            details=details,
            failure_kind=FAILURE_KIND_UNCERTAIN_SIDE_EFFECT,
            uncertain_side_effects=True,
        )


class ACPCapacityError(ACPError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.CAPACITY_EXCEEDED,
            retryable=False,
            details=details,
        )


class ACPProtocolError(ACPError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.MALFORMED_OUTPUT,
            retryable=False,
            details=details,
        )


class ACPRestartExhaustedError(ACPError):
    def __init__(
        self,
        message: str = "ACP restart budget exhausted",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.SPAWN_FAILED,
            retryable=False,
            details=details,
            failure_kind="restart_exhausted",
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ACPClientState(str, Enum):
    """Lifecycle state of the managed ACP subprocess."""

    CREATED = "created"
    STARTING = "starting"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    RESTARTING = "restarting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ACPBounds:
    """Hard limits enforced by the ACP client."""

    max_pending_requests: int = DEFAULT_MAX_PENDING_REQUESTS
    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_serialized_bytes: int = DEFAULT_MAX_SERIALIZED_BYTES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_idle_seconds: float = DEFAULT_MAX_IDLE_SECONDS
    max_restarts: int = DEFAULT_MAX_RESTARTS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    init_timeout_seconds: float = DEFAULT_INIT_TIMEOUT_SECONDS
    read_chunk_bytes: int = DEFAULT_READ_CHUNK_BYTES
    event_queue_size: int = DEFAULT_EVENT_QUEUE_SIZE
    term_grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS
    kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS

    def __post_init__(self) -> None:
        for name in (
            "max_pending_requests",
            "max_sessions",
            "max_serialized_bytes",
            "max_output_bytes",
            "max_restarts",
            "read_chunk_bytes",
            "event_queue_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                if name == "max_restarts" and value == 0:
                    continue
                if name == "max_restarts" and isinstance(value, int) and value >= 0:
                    continue
                raise ContractValidationError(f"{name} must be a positive integer")
        if self.max_restarts < 0:
            raise ContractValidationError("max_restarts must be >= 0")
        for name in (
            "max_idle_seconds",
            "request_timeout_seconds",
            "init_timeout_seconds",
            "term_grace_seconds",
            "kill_wait_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ContractValidationError(f"{name} must be a positive number")


@dataclass(frozen=True)
class ACPRestartPolicy:
    """Explicit transport restart policy.

    Restarts recover the *subprocess and protocol handshake only*. Pending
    agent work is never automatically replayed; callers must re-issue prompts
    after receiving an uncertain-side-effect failure.
    """

    enabled: bool = True
    max_restarts: int = DEFAULT_MAX_RESTARTS
    restart_on_unexpected_exit: bool = True
    # Hard invariant: never auto-replay.
    auto_replay_agent_work: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ContractValidationError("enabled must be a boolean")
        if not isinstance(self.restart_on_unexpected_exit, bool):
            raise ContractValidationError(
                "restart_on_unexpected_exit must be a boolean"
            )
        if not isinstance(self.auto_replay_agent_work, bool):
            raise ContractValidationError(
                "auto_replay_agent_work must be a boolean"
            )
        if self.auto_replay_agent_work:
            raise PolicyDeniedError(
                "ACP restart policy forbids automatic agent work replay",
                details={"auto_replay_agent_work": True},
            )
        if not isinstance(self.max_restarts, int) or isinstance(
            self.max_restarts, bool
        ):
            raise ContractValidationError("max_restarts must be an integer")
        if self.max_restarts < 0:
            raise ContractValidationError("max_restarts must be >= 0")


@dataclass
class ACPSessionRecord:
    """Endpoint-local session metadata (not shared across clients)."""

    session_id: str
    cwd: str
    created_at: float
    last_activity: float
    closed: bool = False
    prompt_count: int = 0
    side_effects_started: bool = False
    pending_prompt_ids: set[Any] = field(default_factory=set)
    metadata: dict[str, str] = field(default_factory=dict)

    def touch(self, now: float) -> None:
        self.last_activity = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "closed": self.closed,
            "prompt_count": self.prompt_count,
            "side_effects_started": self.side_effects_started,
            "pending_prompts": len(self.pending_prompt_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class _PendingRequest:
    request_id: Any
    method: str
    session_id: Optional[str]
    created_at: float
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Mapping[str, Any]] = None
    error: Optional[BaseException] = None
    side_effecting: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    output_bytes: int = 0
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------


def encode_acp_message(message: Mapping[str, Any], *, max_bytes: int) -> bytes:
    """Serialize a JSON-RPC message as one NDJSON line (no embedded newlines)."""
    if not isinstance(message, Mapping):
        raise ContractValidationError("ACP message must be a mapping")
    text = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if "\n" in text or "\r" in text:
        # Defensive: re-encode without whitespace that could break NDJSON.
        text = json.dumps(
            message, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).replace("\n", " ").replace("\r", " ")
    data = (text + "\n").encode("utf-8")
    if len(data) > max_bytes:
        raise BoundsExceededError(
            f"ACP message exceeds {max_bytes} serialized bytes",
            details={"length": len(data), "maximum": max_bytes},
        )
    return data


def split_ndjson_buffer(
    buffer: bytes,
    *,
    max_line_bytes: int,
) -> tuple[list[bytes], bytes]:
    """Split a byte buffer into complete NDJSON lines and a residual partial frame.

    Raises :class:`BoundsExceededError` when a single line grows beyond
    ``max_line_bytes`` without a newline (runaway partial frame).
    """
    lines: list[bytes] = []
    start = 0
    while True:
        idx = buffer.find(b"\n", start)
        if idx < 0:
            residual = buffer[start:]
            if len(residual) > max_line_bytes:
                raise BoundsExceededError(
                    f"ACP partial frame exceeds {max_line_bytes} bytes",
                    details={
                        "length": len(residual),
                        "maximum": max_line_bytes,
                    },
                )
            return lines, residual
        line = buffer[start:idx]
        if len(line) > max_line_bytes:
            raise BoundsExceededError(
                f"ACP frame exceeds {max_line_bytes} bytes",
                details={"length": len(line), "maximum": max_line_bytes},
            )
        # Tolerate CRLF by stripping trailing CR.
        if line.endswith(b"\r"):
            line = line[:-1]
        if line:
            lines.append(line)
        start = idx + 1


def parse_acp_line(line: bytes) -> dict[str, Any]:
    """Parse one NDJSON line into a JSON-RPC object."""
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedOutputError(
            "ACP frame is not valid UTF-8",
            details={"error": type(exc).__name__},
        ) from exc
    text = text.strip()
    if not text:
        raise MalformedOutputError("ACP frame is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedOutputError(
            "ACP frame is not valid JSON",
            details={"error": type(exc).__name__},
        ) from exc
    if not isinstance(payload, dict):
        raise MalformedOutputError(
            "ACP frame must be a JSON object",
            details={"type": type(payload).__name__},
        )
    return payload


def build_text_prompt_blocks(prompt: str) -> list[dict[str, Any]]:
    """Build baseline ACP ``ContentBlock::Text`` prompt payload."""
    if not isinstance(prompt, str):
        raise ContractValidationError("prompt must be a string")
    if "\x00" in prompt:
        raise ContractValidationError("prompt must not contain null bytes")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise BoundsExceededError(
            f"prompt exceeds {MAX_PROMPT_CHARS} characters",
            details={"length": len(prompt), "maximum": MAX_PROMPT_CHARS},
        )
    return [{"type": "text", "text": prompt}]


def _require_absolute_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PolicyDeniedError(
            f"{field_name} must be an absolute path",
            details={"field": field_name},
        )
    return str(path.resolve(strict=False))


def _spawn_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"shell": False}
    if os.name == "nt":
        create_new = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = create_new
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _clip_diag(value: Any, maximum: int = DEFAULT_STDERR_DIAGNOSTIC_CHARS) -> str:
    return _clip_text(value, maximum)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GooseACPClient:
    """Managed client for a local ``goose acp`` NDJSON stdio session.

    Session records are local to this instance (endpoint-local when the
    endpoint owns the client). Restarts never replay agent work.
    """

    def __init__(
        self,
        executable: str,
        state_root: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, Optional[str]]] = None,
        bounds: Optional[ACPBounds] = None,
        restart_policy: Optional[ACPRestartPolicy] = None,
        client_info: Optional[Mapping[str, str]] = None,
        client_capabilities: Optional[Mapping[str, Any]] = None,
        permission_handler: Optional[
            Callable[[Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
        popen_factory: Optional[PopenFactory] = None,
        clock: Optional[Clock] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        if not isinstance(executable, str) or not executable.strip():
            raise ContractValidationError(
                "executable must be an explicit non-empty path or name"
            )
        # Refuse network serve modes entirely.
        exe_lower = executable.strip().lower()
        if "serve" in Path(executable).name.lower() and "acp" not in exe_lower:
            raise PolicyDeniedError(
                "goose serve / network ACP listeners are not supported",
                details={"executable": _clip_diag(executable, 256)},
            )

        self.executable = executable.strip()
        self.state_root = _require_absolute_path(state_root, "state_root")
        self.cwd = (
            _require_absolute_path(cwd, "cwd")
            if cwd is not None
            else self.state_root
        )
        self._env_overlay = dict(env or {})
        self.bounds = bounds or ACPBounds()
        self.restart_policy = restart_policy or ACPRestartPolicy(
            max_restarts=self.bounds.max_restarts
        )
        if self.restart_policy.auto_replay_agent_work:
            raise PolicyDeniedError(
                "ACP restart policy forbids automatic agent work replay"
            )
        self.client_info = {
            "name": CLIENT_NAME,
            "title": CLIENT_TITLE,
            "version": CLIENT_VERSION,
        }
        if client_info:
            self.client_info.update(
                {str(k): str(v) for k, v in client_info.items()}
            )
        # Minimal client capabilities: we do not advertise fs/terminal by
        # default so the agent cannot request host side-effects through us
        # unless the caller explicitly opts in.
        self.client_capabilities: dict[str, Any] = {
            "fs": {"readTextFile": False, "writeTextFile": False},
            "terminal": False,
        }
        if client_capabilities:
            self.client_capabilities.update(dict(client_capabilities))
        self._permission_handler = permission_handler
        self._popen = popen_factory or subprocess.Popen
        self._clock = clock or time.monotonic
        self.endpoint_id = endpoint_id

        self._lock = threading.RLock()
        self._state = ACPClientState.CREATED
        self._process: Any = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._idle_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pending: dict[Any, _PendingRequest] = {}
        self._sessions: dict[str, ACPSessionRecord] = {}
        self._next_id = 1
        self._agent_capabilities: dict[str, Any] = {}
        self._agent_info: dict[str, Any] = {}
        self._protocol_version: Optional[int] = None
        self._restart_count = 0
        self._last_activity = self._clock()
        self._stderr_tail = ""
        self._output_bytes_total = 0
        self._event_listeners: list[EventCallback] = []
        self._global_event_queue: queue.Queue = queue.Queue(
            maxsize=self.bounds.event_queue_size
        )
        self._write_lock = threading.Lock()
        self._initialized = False
        self._last_exit_code: Optional[int] = None
        self._last_failure: Optional[dict[str, Any]] = None

    # -- properties --------------------------------------------------------

    @property
    def state(self) -> ACPClientState:
        with self._lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return (
                self._state is ACPClientState.READY
                and self._initialized
                and self._process is not None
                and self._process.poll() is None
            )

    @property
    def restart_count(self) -> int:
        with self._lock:
            return self._restart_count

    @property
    def agent_capabilities(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._agent_capabilities)

    @property
    def agent_info(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._agent_info)

    @property
    def protocol_version(self) -> Optional[int]:
        with self._lock:
            return self._protocol_version

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._sessions.values() if not s.closed]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            rec = self._sessions.get(session_id)
            return None if rec is None else rec.to_dict()

    def add_event_listener(self, callback: EventCallback) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._event_listeners.append(callback)

    def remove_event_listener(self, callback: EventCallback) -> None:
        with self._lock:
            try:
                self._event_listeners.remove(callback)
            except ValueError:
                pass

    def describe(self) -> dict[str, Any]:
        with self._lock:
            pid = getattr(self._process, "pid", None) if self._process else None
            return {
                "endpoint_id": self.endpoint_id,
                "state": self._state.value,
                "ready": self.is_ready,
                "executable": self.executable,
                "state_root": self.state_root,
                "cwd": self.cwd,
                "pid": pid,
                "protocol_version": self._protocol_version,
                "agent_capabilities": dict(self._agent_capabilities),
                "agent_info": dict(self._agent_info),
                "pending_requests": len(self._pending),
                "sessions": len(
                    [s for s in self._sessions.values() if not s.closed]
                ),
                "restart_count": self._restart_count,
                "max_restarts": self.restart_policy.max_restarts,
                "restart_enabled": self.restart_policy.enabled,
                "auto_replay_agent_work": False,
                "last_exit_code": self._last_exit_code,
                "last_failure": dict(self._last_failure or {}),
                "output_bytes_total": self._output_bytes_total,
                "bounds": {
                    "max_pending_requests": self.bounds.max_pending_requests,
                    "max_sessions": self.bounds.max_sessions,
                    "max_serialized_bytes": self.bounds.max_serialized_bytes,
                    "max_output_bytes": self.bounds.max_output_bytes,
                    "max_idle_seconds": self.bounds.max_idle_seconds,
                    "max_restarts": self.bounds.max_restarts,
                },
            }

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Spawn ``goose acp``, initialize protocol, validate capabilities."""
        with self._lock:
            if self._state is ACPClientState.READY and self._initialized:
                return {
                    "status": "success",
                    "success": True,
                    "state": self._state.value,
                    "already_started": True,
                    "protocol_version": self._protocol_version,
                    "agent_capabilities": dict(self._agent_capabilities),
                }
            if self._state in (
                ACPClientState.STARTING,
                ACPClientState.INITIALIZING,
                ACPClientState.RESTARTING,
            ):
                raise ACPNotReadyError(
                    f"ACP client is already {self._state.value}",
                    details={"state": self._state.value},
                )
            if self._state is ACPClientState.FAILED:
                # Only recoverable via explicit restart or stop+start after
                # budget reset by stop().
                if (
                    self.restart_policy.enabled
                    and self._restart_count > self.restart_policy.max_restarts
                ):
                    raise ACPRestartExhaustedError(
                        details={
                            "restart_count": self._restart_count,
                            "max_restarts": self.restart_policy.max_restarts,
                        }
                    )
            self._state = ACPClientState.STARTING
            self._stop_event.clear()

        try:
            self._spawn_process()
            with self._lock:
                self._state = ACPClientState.INITIALIZING
            init_result = self._initialize()
            with self._lock:
                self._state = ACPClientState.READY
                self._initialized = True
                self._last_activity = self._clock()
            return {
                "status": "success",
                "success": True,
                "state": ACPClientState.READY.value,
                "protocol_version": self._protocol_version,
                "agent_capabilities": dict(self._agent_capabilities),
                "agent_info": dict(self._agent_info),
                "initialize_result": init_result,
                "pid": getattr(self._process, "pid", None),
                "state_root": self.state_root,
                "restart_count": self._restart_count,
            }
        except Exception as exc:
            with self._lock:
                self._state = ACPClientState.FAILED
                self._initialized = False
                self._last_failure = {
                    "error_type": type(exc).__name__,
                    "message": _clip_diag(str(exc)),
                }
            self._kill_process(reason="start_failed")
            if isinstance(exc, CLIRuntimeError):
                raise
            raise ProcessSpawnError(
                f"failed to start ACP client: {type(exc).__name__}",
                details={"error_type": type(exc).__name__},
            ) from exc

    def stop(self, *, timeout: float = 5.0) -> dict[str, Any]:
        """Cancel pending work, clear sessions, and terminate the child tree."""
        with self._lock:
            if self._state is ACPClientState.STOPPED:
                return {
                    "status": "success",
                    "success": True,
                    "state": self._state.value,
                    "already_stopped": True,
                }
            self._state = ACPClientState.STOPPING
            self._stop_event.set()

        self._fail_all_pending(
            ProcessCancelledError(
                "ACP client stopped",
                details={"reason": "client_stop"},
            ),
            uncertain=False,
        )
        self._kill_process(reason="client_stop", timeout=timeout)
        with self._lock:
            for session in self._sessions.values():
                session.closed = True
                session.pending_prompt_ids.clear()
            self._sessions.clear()
            self._initialized = False
            self._state = ACPClientState.STOPPED
            # Stopping resets restart budget for a future explicit start.
            # Unexpected-exit restarts consume the budget; clean stop does not
            # permanently fail the client.
        return {
            "status": "success",
            "success": True,
            "state": ACPClientState.STOPPED.value,
            "cancelled_pending": True,
        }

    def restart_transport(self, *, explicit: bool = True) -> dict[str, Any]:
        """Restart the ACP subprocess without replaying agent work.

        In-flight requests fail with uncertain-side-effect status. Sessions are
        marked closed locally; the remote process starts clean. Callers must
        create/load sessions again and re-issue prompts deliberately.
        """
        with self._lock:
            if not explicit and not self.restart_policy.enabled:
                raise PolicyDeniedError(
                    "ACP automatic restart is disabled",
                    details={"explicit": False},
                )
            if not explicit and not self.restart_policy.restart_on_unexpected_exit:
                raise PolicyDeniedError(
                    "ACP restart_on_unexpected_exit is disabled",
                    details={"explicit": False},
                )
            if self._restart_count >= self.restart_policy.max_restarts:
                self._state = ACPClientState.FAILED
                raise ACPRestartExhaustedError(
                    details={
                        "restart_count": self._restart_count,
                        "max_restarts": self.restart_policy.max_restarts,
                        "explicit": explicit,
                    }
                )
            self._state = ACPClientState.RESTARTING
            self._restart_count += 1
            restart_n = self._restart_count

        # Fail pending — never replay.
        self._fail_all_pending(
            ACPUncertainSideEffectError(
                "ACP transport restart; in-flight work not replayed",
                details={
                    "restart_count": restart_n,
                    "auto_replay_agent_work": False,
                    "explicit": explicit,
                },
            ),
            uncertain=True,
        )
        # Local sessions are invalidated by process death/restart.
        with self._lock:
            for session in self._sessions.values():
                session.closed = True
                session.pending_prompt_ids.clear()
            self._sessions.clear()
            self._initialized = False

        self._kill_process(reason="restart")
        self._stop_event.clear()

        try:
            self._spawn_process()
            with self._lock:
                self._state = ACPClientState.INITIALIZING
            init_result = self._initialize()
            with self._lock:
                self._state = ACPClientState.READY
                self._initialized = True
                self._last_activity = self._clock()
            return {
                "status": "success",
                "success": True,
                "state": ACPClientState.READY.value,
                "restart_count": restart_n,
                "auto_replay_agent_work": False,
                "sessions_cleared": True,
                "protocol_version": self._protocol_version,
                "agent_capabilities": dict(self._agent_capabilities),
                "initialize_result": init_result,
                "message": (
                    "transport restarted; agent work was not replayed; "
                    "create or load sessions explicitly"
                ),
            }
        except Exception as exc:
            with self._lock:
                self._state = ACPClientState.FAILED
                self._initialized = False
                self._last_failure = {
                    "error_type": type(exc).__name__,
                    "message": _clip_diag(str(exc)),
                    "restart_count": restart_n,
                }
            self._kill_process(reason="restart_failed")
            if isinstance(exc, CLIRuntimeError):
                raise
            raise ProcessSpawnError(
                f"ACP restart failed: {type(exc).__name__}",
                details={
                    "error_type": type(exc).__name__,
                    "restart_count": restart_n,
                },
            ) from exc

    # -- sessions ----------------------------------------------------------

    def session_new(
        self,
        *,
        cwd: Optional[str] = None,
        mcp_servers: Sequence[Mapping[str, Any]] = (),
        metadata: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Create a new ACP session after the client is ready."""
        self._ensure_ready()
        with self._lock:
            open_count = sum(1 for s in self._sessions.values() if not s.closed)
            if open_count >= self.bounds.max_sessions:
                raise ACPCapacityError(
                    f"session limit {self.bounds.max_sessions} reached",
                    details={
                        "sessions": open_count,
                        "maximum": self.bounds.max_sessions,
                    },
                )
        session_cwd = _require_absolute_path(cwd or self.cwd, "cwd")
        params = {
            "cwd": session_cwd,
            "mcpServers": list(mcp_servers or ()),
        }
        result = self._request(
            METHOD_SESSION_NEW,
            params,
            timeout=timeout,
            side_effecting=True,
        )
        session_id = result.get("sessionId") or result.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ACPProtocolError(
                "session/new response missing sessionId",
                details={"keys": ",".join(sorted(result))},
            )
        session_id = session_id.strip()
        if len(session_id) > MAX_SESSION_ID_CHARS:
            raise BoundsExceededError(
                f"sessionId exceeds {MAX_SESSION_ID_CHARS} characters",
                details={
                    "length": len(session_id),
                    "maximum": MAX_SESSION_ID_CHARS,
                },
            )
        now = self._clock()
        record = ACPSessionRecord(
            session_id=session_id,
            cwd=session_cwd,
            created_at=now,
            last_activity=now,
            metadata={str(k): str(v) for k, v in (metadata or {}).items()},
        )
        with self._lock:
            # Guard against cross-session id collision / leakage.
            existing = self._sessions.get(session_id)
            if existing is not None and not existing.closed:
                raise InvalidStateError(
                    "sessionId already tracked by this client",
                    details={"session_id": session_id[:64]},
                )
            self._sessions[session_id] = record
        return {
            "status": "success",
            "success": True,
            "session_id": session_id,
            "cwd": session_cwd,
            "result": self._sanitize_result(result),
            "endpoint_id": self.endpoint_id,
        }

    def session_load(
        self,
        session_id: str,
        *,
        cwd: Optional[str] = None,
        mcp_servers: Sequence[Mapping[str, Any]] = (),
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Load/resume an existing session when the agent supports it."""
        self._ensure_ready()
        sid = self._validate_session_id(session_id)
        caps = self.agent_capabilities
        if not caps.get("loadSession"):
            raise PolicyDeniedError(
                "agent does not advertise loadSession capability",
                details={"loadSession": False},
            )
        session_cwd = _require_absolute_path(cwd or self.cwd, "cwd")
        params = {
            "sessionId": sid,
            "cwd": session_cwd,
            "mcpServers": list(mcp_servers or ()),
        }
        result = self._request(
            METHOD_SESSION_LOAD,
            params,
            timeout=timeout,
            session_id=sid,
            side_effecting=True,
        )
        now = self._clock()
        with self._lock:
            open_count = sum(
                1
                for s in self._sessions.values()
                if not s.closed and s.session_id != sid
            )
            if open_count >= self.bounds.max_sessions:
                raise ACPCapacityError(
                    f"session limit {self.bounds.max_sessions} reached",
                    details={
                        "sessions": open_count,
                        "maximum": self.bounds.max_sessions,
                    },
                )
            rec = self._sessions.get(sid)
            if rec is None:
                rec = ACPSessionRecord(
                    session_id=sid,
                    cwd=session_cwd,
                    created_at=now,
                    last_activity=now,
                )
                self._sessions[sid] = rec
            else:
                rec.closed = False
                rec.cwd = session_cwd
                rec.touch(now)
        return {
            "status": "success",
            "success": True,
            "session_id": sid,
            "cwd": session_cwd,
            "loaded": True,
            "result": self._sanitize_result(result),
            "endpoint_id": self.endpoint_id,
        }

    def session_close(
        self,
        session_id: str,
        *,
        timeout: Optional[float] = None,
        remote: bool = True,
    ) -> dict[str, Any]:
        """Close a session locally and optionally notify the agent."""
        sid = self._validate_session_id(session_id)
        remote_result: Optional[dict[str, Any]] = None
        remote_error: Optional[str] = None
        if remote and self.is_ready:
            try:
                # Prefer session/close when available; ignore method-not-found.
                remote_result = self._request(
                    METHOD_SESSION_CLOSE,
                    {"sessionId": sid},
                    timeout=timeout,
                    session_id=sid,
                    side_effecting=True,
                )
            except ACPError as exc:
                # Method may be unsupported; local close still proceeds.
                remote_error = exc.record.message
            except CLIRuntimeError as exc:
                remote_error = exc.record.message

        cancelled_ids: list[Any] = []
        with self._lock:
            rec = self._sessions.get(sid)
            if rec is not None:
                cancelled_ids = list(rec.pending_prompt_ids)
                for req_id in list(rec.pending_prompt_ids):
                    pending = self._pending.get(req_id)
                    if pending is not None:
                        pending.cancelled = True
                        pending.error = ProcessCancelledError(
                            "session closed",
                            details={"session_id": sid[:64]},
                        )
                        pending.event.set()
                rec.pending_prompt_ids.clear()
                rec.closed = True
                rec.touch(self._clock())
            # Drop closed session entry to free capacity.
            self._sessions.pop(sid, None)

        # Best-effort cancel notification for in-flight prompts.
        if cancelled_ids and self.is_ready:
            try:
                self._notify(METHOD_SESSION_CANCEL, {"sessionId": sid})
            except Exception:  # noqa: BLE001
                pass

        return {
            "status": "success",
            "success": True,
            "session_id": sid,
            "closed": True,
            "cancelled_prompts": len(cancelled_ids),
            "remote_result": (
                None if remote_result is None else self._sanitize_result(remote_result)
            ),
            "remote_error": remote_error,
            "endpoint_id": self.endpoint_id,
        }

    def session_prompt(
        self,
        session_id: str,
        prompt: str,
        *,
        timeout: Optional[float] = None,
        on_event: Optional[EventCallback] = None,
        content_blocks: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Send a prompt turn and collect streamed session/update events.

        Never automatically retried. Unexpected process exit surfaces as
        :class:`ACPUncertainSideEffectError`.
        """
        self._ensure_ready()
        sid = self._validate_session_id(session_id)
        with self._lock:
            rec = self._sessions.get(sid)
            if rec is None or rec.closed:
                raise InvalidStateError(
                    "unknown or closed session on this endpoint client",
                    details={"session_id": sid[:64]},
                )
        if content_blocks is None:
            blocks = build_text_prompt_blocks(prompt)
        else:
            if not isinstance(content_blocks, (list, tuple)):
                raise ContractValidationError("content_blocks must be a sequence")
            blocks = [dict(b) for b in content_blocks]

        params = {"sessionId": sid, "prompt": blocks}
        # Collect events for this session during the request.
        collected: list[dict[str, Any]] = []
        text_parts: list[str] = []
        tool_calls = 0
        side_effects = False

        def _collector(event: Mapping[str, Any]) -> None:
            nonlocal tool_calls, side_effects
            if event.get("session_id") != sid:
                return  # hard isolation
            collected.append(dict(event))
            if len(collected) > MAX_EVENT_COUNT:
                collected.pop(0)
            update = event.get("update") or {}
            kind = str(update.get("sessionUpdate") or update.get("type") or "")
            if "tool" in kind.lower():
                tool_calls += 1
                side_effects = True
            content = update.get("content") or update.get("text")
            if isinstance(content, str) and content:
                text_parts.append(content)
            elif isinstance(content, Mapping):
                t = content.get("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # noqa: BLE001
                    logger.debug("on_event callback failed", exc_info=True)

        listener_wrapped = _collector
        self.add_event_listener(listener_wrapped)
        try:
            with self._lock:
                rec = self._sessions.get(sid)
                if rec is None or rec.closed:
                    raise InvalidStateError(
                        "session closed before prompt",
                        details={"session_id": sid[:64]},
                    )
            result = self._request(
                METHOD_SESSION_PROMPT,
                params,
                timeout=timeout,
                session_id=sid,
                side_effecting=True,
            )
        except ACPUncertainSideEffectError:
            with self._lock:
                rec = self._sessions.get(sid)
                if rec is not None:
                    rec.side_effects_started = True
            raise
        finally:
            self.remove_event_listener(listener_wrapped)

        stop_reason = result.get("stopReason") or result.get("stop_reason")
        text = "".join(text_parts)
        if len(text) > self.bounds.max_output_bytes:
            text = text[: self.bounds.max_output_bytes]
        with self._lock:
            rec = self._sessions.get(sid)
            if rec is not None:
                rec.prompt_count += 1
                rec.side_effects_started = rec.side_effects_started or side_effects
                rec.touch(self._clock())

        return {
            "status": "success",
            "success": True,
            "session_id": sid,
            "stop_reason": stop_reason,
            "text": text,
            "events": collected[:MAX_EVENT_COUNT],
            "event_count": len(collected),
            "tool_call_count": tool_calls,
            "side_effects_started": side_effects,
            "result": self._sanitize_result(result),
            "endpoint_id": self.endpoint_id,
            "cacheable": False,
            "retryable": False,
        }

    def session_cancel(self, session_id: str) -> dict[str, Any]:
        """Cancel in-flight work for a session and clear local pending state."""
        sid = self._validate_session_id(session_id)
        cancelled = 0
        with self._lock:
            rec = self._sessions.get(sid)
            if rec is not None:
                for req_id in list(rec.pending_prompt_ids):
                    pending = self._pending.get(req_id)
                    if pending is not None:
                        pending.cancelled = True
                        pending.error = ProcessCancelledError(
                            "session cancelled",
                            details={"session_id": sid[:64]},
                        )
                        pending.event.set()
                        cancelled += 1
                rec.pending_prompt_ids.clear()
                rec.touch(self._clock())

        notified = False
        if self.is_ready:
            try:
                self._notify(METHOD_SESSION_CANCEL, {"sessionId": sid})
                notified = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("session/cancel notify failed: %s", type(exc).__name__)

        return {
            "status": "success",
            "success": True,
            "session_id": sid,
            "cancelled_pending": cancelled,
            "notified": notified,
            "endpoint_id": self.endpoint_id,
        }

    def stream_prompt(
        self,
        session_id: str,
        prompt: str,
        *,
        timeout: Optional[float] = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield session events then a final completed/failed envelope."""
        yield {
            "event": "started",
            "session_id": session_id,
            "endpoint_id": self.endpoint_id,
        }
        q: queue.Queue = queue.Queue()

        def _on(event: Mapping[str, Any]) -> None:
            if event.get("session_id") == session_id:
                q.put(dict(event))

        def _runner() -> None:
            try:
                result = self.session_prompt(
                    session_id, prompt, timeout=timeout, on_event=_on
                )
                q.put({"__final__": result})
            except BaseException as exc:  # noqa: BLE001
                q.put({"__error__": exc})

        thread = threading.Thread(
            target=_runner, name="acp-stream-prompt", daemon=True
        )
        thread.start()
        while True:
            try:
                item = q.get(timeout=0.05)
            except queue.Empty:
                if not thread.is_alive() and q.empty():
                    break
                continue
            if "__final__" in item:
                final = item["__final__"]
                yield {
                    "event": "completed",
                    "session_id": session_id,
                    "endpoint_id": self.endpoint_id,
                    "text": final.get("text"),
                    "stop_reason": final.get("stop_reason"),
                    "side_effects_started": final.get("side_effects_started"),
                    "tool_call_count": final.get("tool_call_count"),
                }
                break
            if "__error__" in item:
                exc = item["__error__"]
                uncertain = isinstance(exc, ACPUncertainSideEffectError) or (
                    isinstance(exc, ACPError) and exc.uncertain_side_effects
                )
                payload: dict[str, Any] = {
                    "event": "failed",
                    "session_id": session_id,
                    "endpoint_id": self.endpoint_id,
                    "error": _clip_diag(
                        str(exc.record.message)
                        if isinstance(exc, CLIRuntimeError)
                        else type(exc).__name__
                    ),
                    "error_code": (
                        exc.code.value
                        if isinstance(exc, CLIRuntimeError)
                        else CLIRuntimeErrorCode.INTERNAL.value
                    ),
                    "uncertain_side_effects": uncertain,
                    "failure_kind": (
                        FAILURE_KIND_UNCERTAIN_SIDE_EFFECT
                        if uncertain
                        else getattr(exc, "failure_kind", None)
                    ),
                }
                yield payload
                break
            yield {
                "event": "session_update",
                "session_id": session_id,
                "endpoint_id": self.endpoint_id,
                "update": item.get("update"),
                "raw": item,
            }

    # -- internals: spawn / init -------------------------------------------

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise ACPNotReadyError(
                "ACP client is not ready; call start() and wait for initialize",
                details={"state": self.state.value},
            )
        # Idle bound: if no activity for too long, mark degraded (caller may
        # restart explicitly). We do not silently kill mid-request here.
        with self._lock:
            idle = self._clock() - self._last_activity
            if idle > self.bounds.max_idle_seconds and not self._pending:
                self._state = ACPClientState.DEGRADED
                raise ACPNotReadyError(
                    "ACP client exceeded idle time bound",
                    details={
                        "idle_seconds": idle,
                        "max_idle_seconds": self.bounds.max_idle_seconds,
                    },
                )

    def _build_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            k: v for k, v in os.environ.items() if isinstance(v, str)
        }
        # Isolated state root — never share the caller's default goose home.
        env["GOOSE_PATH_ROOT"] = self.state_root
        # Ensure the state root exists.
        Path(self.state_root).mkdir(parents=True, exist_ok=True)
        for key, value in self._env_overlay.items():
            if value is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(value)
        # Never pass through serve-related secrets that might encourage network.
        env.pop("GOOSE_SERVER__SECRET_KEY", None)
        return env

    def _spawn_process(self) -> None:
        argv = [self.executable, "acp"]
        # Defense in depth: refuse accidental serve args.
        for part in argv:
            if part in {"serve", "--dangerously-unauthenticated"}:
                raise PolicyDeniedError(
                    "refusing to launch goose serve / unauthenticated network mode",
                    details={"argv_preview": "acp"},
                )
        env = self._build_env()
        kwargs = _spawn_kwargs()
        kwargs.update(
            {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "cwd": self.cwd,
                "env": env,
                "bufsize": 0,
            }
        )
        try:
            process = self._popen(argv, **kwargs)
        except FileNotFoundError as exc:
            raise ProcessSpawnError(
                "ACP executable not found",
                details={"executable": _clip_diag(self.executable, 256)},
            ) from exc
        except OSError as exc:
            raise ProcessSpawnError(
                f"failed to spawn ACP process: {type(exc).__name__}",
                details={"error_type": type(exc).__name__},
            ) from exc

        with self._lock:
            self._process = process
            self._last_exit_code = None

        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="goose-acp-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name="goose-acp-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

    def _initialize(self) -> dict[str, Any]:
        params = {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "clientCapabilities": self.client_capabilities,
            "clientInfo": self.client_info,
        }
        result = self._request(
            METHOD_INITIALIZE,
            params,
            timeout=self.bounds.init_timeout_seconds,
            side_effecting=False,
            allow_before_ready=True,
        )
        version = result.get("protocolVersion", result.get("protocol_version"))
        if version is None:
            raise ACPProtocolError(
                "initialize response missing protocolVersion",
                details={"keys": ",".join(sorted(result))},
            )
        try:
            version_int = int(version)
        except (TypeError, ValueError) as exc:
            raise ACPProtocolError(
                "initialize protocolVersion is not an integer",
                details={"protocolVersion": str(version)},
            ) from exc
        if version_int != ACP_PROTOCOL_VERSION:
            raise ACPProtocolError(
                "unsupported ACP protocol version from agent",
                details={
                    "requested": ACP_PROTOCOL_VERSION,
                    "negotiated": version_int,
                },
            )
        agent_caps = result.get("agentCapabilities") or result.get(
            "agent_capabilities"
        ) or {}
        if not isinstance(agent_caps, Mapping):
            raise ACPProtocolError(
                "agentCapabilities must be an object",
                details={"type": type(agent_caps).__name__},
            )
        # Baseline: agents must support session/new, session/prompt,
        # session/cancel, session/update. We cannot probe methods, but we
        # require a capabilities object (possibly empty) and successful init.
        agent_info = result.get("agentInfo") or result.get("agent_info") or {}
        if agent_info is not None and not isinstance(agent_info, Mapping):
            raise ACPProtocolError(
                "agentInfo must be an object when present",
                details={"type": type(agent_info).__name__},
            )

        with self._lock:
            self._protocol_version = version_int
            self._agent_capabilities = dict(agent_caps)
            self._agent_info = dict(agent_info or {})
            self._initialized = True
        return dict(result)

    # -- RPC ---------------------------------------------------------------

    def _next_request_id(self) -> int:
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            return rid

    def _request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
        session_id: Optional[str] = None,
        side_effecting: bool = False,
        allow_before_ready: bool = False,
    ) -> dict[str, Any]:
        if not allow_before_ready:
            self._ensure_ready()
        else:
            with self._lock:
                if self._process is None or self._process.poll() is not None:
                    raise ACPNotReadyError(
                        "ACP process is not running",
                        details={"state": self._state.value},
                    )

        with self._lock:
            if len(self._pending) >= self.bounds.max_pending_requests:
                raise ACPCapacityError(
                    f"pending request limit {self.bounds.max_pending_requests} reached",
                    details={
                        "pending": len(self._pending),
                        "maximum": self.bounds.max_pending_requests,
                    },
                )
            request_id = self._next_request_id()
            pending = _PendingRequest(
                request_id=request_id,
                method=method,
                session_id=session_id,
                created_at=self._clock(),
                side_effecting=side_effecting,
            )
            self._pending[request_id] = pending
            if session_id is not None:
                rec = self._sessions.get(session_id)
                if rec is not None:
                    rec.pending_prompt_ids.add(request_id)

        message = {
            "jsonrpc": ACP_JSONRPC_VERSION,
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        try:
            self._write_message(message)
        except Exception:
            with self._lock:
                self._pending.pop(request_id, None)
                if session_id is not None:
                    rec = self._sessions.get(session_id)
                    if rec is not None:
                        rec.pending_prompt_ids.discard(request_id)
            raise

        wait_timeout = (
            float(timeout)
            if timeout is not None
            else self.bounds.request_timeout_seconds
        )
        if wait_timeout <= 0:
            wait_timeout = self.bounds.request_timeout_seconds

        finished = pending.event.wait(timeout=wait_timeout)
        with self._lock:
            self._pending.pop(request_id, None)
            if session_id is not None:
                rec = self._sessions.get(session_id)
                if rec is not None:
                    rec.pending_prompt_ids.discard(request_id)

        if not finished:
            # Timed out — best-effort cancel if session-scoped.
            if session_id is not None:
                try:
                    self._notify(
                        METHOD_SESSION_CANCEL, {"sessionId": session_id}
                    )
                except Exception:  # noqa: BLE001
                    pass
            raise ProcessTimeoutError(
                f"ACP request timed out: {method}",
                details={
                    "method": method,
                    "timeout_seconds": wait_timeout,
                    "session_id": (session_id or "")[:64],
                    "side_effecting": side_effecting,
                },
            )

        if pending.cancelled and pending.error is not None:
            raise pending.error
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise ACPProtocolError(
                "ACP request completed without response",
                details={"method": method, "id": request_id},
            )
        response = dict(pending.response)
        if "error" in response:
            err = response["error"]
            if isinstance(err, Mapping):
                msg = str(err.get("message") or "ACP error")
                code = err.get("code")
                raise ACPError(
                    _clip_diag(msg),
                    code=CLIRuntimeErrorCode.INTERNAL,
                    details={
                        "method": method,
                        "rpc_code": str(code) if code is not None else "",
                        "session_id": (session_id or "")[:64],
                    },
                )
            raise ACPError(
                "ACP error response",
                details={"method": method},
            )
        result = response.get("result")
        if result is None:
            return {}
        if not isinstance(result, Mapping):
            return {"value": result}
        return dict(result)

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        message = {
            "jsonrpc": ACP_JSONRPC_VERSION,
            "method": method,
            "params": dict(params),
        }
        self._write_message(message)

    def _write_message(self, message: Mapping[str, Any]) -> None:
        data = encode_acp_message(
            message, max_bytes=self.bounds.max_serialized_bytes
        )
        with self._lock:
            process = self._process
        if process is None or process.stdin is None:
            raise ACPNotReadyError("ACP stdin is not available")
        with self._write_lock:
            try:
                process.stdin.write(data)
                process.stdin.flush()
            except BrokenPipeError as exc:
                self._handle_unexpected_exit(exit_code=process.poll())
                raise ACPUncertainSideEffectError(
                    "ACP stdin pipe broken",
                    details={"method": str(message.get("method", ""))},
                ) from exc
            except OSError as exc:
                raise ACPError(
                    f"failed to write ACP message: {type(exc).__name__}",
                    details={"error_type": type(exc).__name__},
                ) from exc
        with self._lock:
            self._last_activity = self._clock()

    # -- reader loops ------------------------------------------------------

    def _reader_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        buffer = b""
        stdout = process.stdout
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = stdout.read(self.bounds.read_chunk_bytes)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                with self._lock:
                    self._output_bytes_total += len(chunk)
                    if self._output_bytes_total > self.bounds.max_output_bytes * 4:
                        # Soft bound on total lifetime stdout; mark degraded.
                        self._state = ACPClientState.DEGRADED
                    self._last_activity = self._clock()
                buffer += chunk
                try:
                    lines, buffer = split_ndjson_buffer(
                        buffer,
                        max_line_bytes=self.bounds.max_serialized_bytes,
                    )
                except BoundsExceededError as exc:
                    logger.warning("ACP frame bound exceeded: %s", exc)
                    buffer = b""
                    continue
                for line in lines:
                    try:
                        message = parse_acp_line(line)
                    except MalformedOutputError as exc:
                        logger.debug(
                            "dropping malformed ACP frame: %s",
                            exc.record.message,
                        )
                        continue
                    try:
                        self._dispatch_message(message)
                    except Exception:  # noqa: BLE001
                        logger.exception("ACP dispatch failed")
        finally:
            # Process stdout closed → unexpected exit path if not stopping.
            if not self._stop_event.is_set():
                exit_code = None
                try:
                    exit_code = process.poll()
                    if exit_code is None:
                        try:
                            exit_code = process.wait(timeout=0.2)
                        except Exception:  # noqa: BLE001
                            exit_code = process.poll()
                except Exception:  # noqa: BLE001
                    exit_code = None
                self._handle_unexpected_exit(exit_code=exit_code)

    def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = process.stderr.read(4096)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    text = ""
                if text:
                    with self._lock:
                        self._stderr_tail = (self._stderr_tail + text)[
                            -DEFAULT_STDERR_DIAGNOSTIC_CHARS:
                        ]
        except Exception:  # noqa: BLE001
            pass

    def _dispatch_message(self, message: Mapping[str, Any]) -> None:
        # Responses have id and result/error; requests have method+id;
        # notifications have method without id.
        msg_id = message.get("id", _MISSING)
        method = message.get("method")

        if msg_id is not _MISSING and method is None:
            # Response to our request.
            self._handle_response(msg_id, message)
            return

        if method is not None and msg_id is _MISSING:
            # Notification from agent.
            self._handle_notification(str(method), message.get("params") or {})
            return

        if method is not None and msg_id is not _MISSING:
            # Request from agent to client (e.g. session/request_permission).
            self._handle_agent_request(
                str(method), msg_id, message.get("params") or {}
            )
            return

        logger.debug("ignoring unrecognized ACP message shape")

    def _handle_response(
        self, msg_id: Any, message: Mapping[str, Any]
    ) -> None:
        with self._lock:
            pending = self._pending.get(msg_id)
            if pending is None:
                # Unknown ID — no cross-session leakage; drop.
                logger.debug("ACP response for unknown id=%r dropped", msg_id)
                return
            pending.response = dict(message)
            pending.event.set()
            self._last_activity = self._clock()

    def _handle_notification(
        self, method: str, params: Any
    ) -> None:
        if not isinstance(params, Mapping):
            params = {}
        if method == METHOD_SESSION_UPDATE:
            session_id = params.get("sessionId") or params.get("session_id")
            if not isinstance(session_id, str):
                logger.debug("session/update missing sessionId; dropped")
                return
            with self._lock:
                rec = self._sessions.get(session_id)
                if rec is None or rec.closed:
                    # Unknown session — drop to prevent cross-session leakage.
                    logger.debug(
                        "session/update for unknown session %r dropped",
                        session_id[:64],
                    )
                    return
                rec.touch(self._clock())
                update = params.get("update") or {}
                if isinstance(update, Mapping):
                    kind = str(
                        update.get("sessionUpdate") or update.get("type") or ""
                    )
                    if "tool" in kind.lower():
                        rec.side_effects_started = True
            event = {
                "method": method,
                "session_id": session_id,
                "update": params.get("update"),
                "params": self._sanitize_result(params),
            }
            self._emit_event(event)
            return
        # Other notifications: emit if session-scoped and known; else drop.
        session_id = None
        if isinstance(params, Mapping):
            session_id = params.get("sessionId") or params.get("session_id")
        if isinstance(session_id, str):
            with self._lock:
                if session_id not in self._sessions:
                    return
            self._emit_event(
                {
                    "method": method,
                    "session_id": session_id,
                    "params": self._sanitize_result(params),
                }
            )

    def _handle_agent_request(
        self, method: str, msg_id: Any, params: Any
    ) -> None:
        if not isinstance(params, Mapping):
            params = {}
        # session/request_permission — respond with cancel by default for safety
        # unless a handler is provided.
        if method == METHOD_SESSION_REQUEST_PERMISSION:
            session_id = params.get("sessionId") or params.get("session_id")
            if isinstance(session_id, str):
                with self._lock:
                    if session_id not in self._sessions:
                        # Unknown session — reject without side effects.
                        self._write_error_response(
                            msg_id,
                            code=-32000,
                            message="unknown session",
                        )
                        return
            outcome: Mapping[str, Any]
            if self._permission_handler is not None:
                try:
                    outcome = self._permission_handler(params)
                except Exception:  # noqa: BLE001
                    outcome = {"outcome": {"outcome": "cancelled"}}
            else:
                # Fail closed: do not auto-approve tool side effects.
                outcome = {"outcome": {"outcome": "cancelled"}}
            if not isinstance(outcome, Mapping):
                outcome = {"outcome": {"outcome": "cancelled"}}
            self._write_result_response(msg_id, outcome)
            return

        # Unsupported agent→client method.
        self._write_error_response(
            msg_id,
            code=-32601,
            message=f"method not supported by client: {method}",
        )

    def _write_result_response(self, msg_id: Any, result: Mapping[str, Any]) -> None:
        message = {
            "jsonrpc": ACP_JSONRPC_VERSION,
            "id": msg_id,
            "result": dict(result),
        }
        try:
            self._write_message(message)
        except Exception:  # noqa: BLE001
            logger.debug("failed to write ACP result response", exc_info=True)

    def _write_error_response(
        self, msg_id: Any, *, code: int, message: str
    ) -> None:
        payload = {
            "jsonrpc": ACP_JSONRPC_VERSION,
            "id": msg_id,
            "error": {
                "code": code,
                "message": _clip_diag(message, MAX_EVENT_PAYLOAD_CHARS),
            },
        }
        try:
            self._write_message(payload)
        except Exception:  # noqa: BLE001
            logger.debug("failed to write ACP error response", exc_info=True)

    def _emit_event(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            listeners = list(self._event_listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception:  # noqa: BLE001
                logger.debug("event listener failed", exc_info=True)
        try:
            self._global_event_queue.put_nowait(dict(event))
        except queue.Full:
            # Backpressure: drop oldest-style by discarding this event.
            try:
                self._global_event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._global_event_queue.put_nowait(dict(event))
            except queue.Full:
                pass

    # -- exit / cancel helpers ---------------------------------------------

    def _handle_unexpected_exit(self, *, exit_code: Optional[int]) -> None:
        with self._lock:
            if self._state in (
                ACPClientState.STOPPING,
                ACPClientState.STOPPED,
                ACPClientState.RESTARTING,
            ):
                return
            self._last_exit_code = exit_code
            self._initialized = False
            self._last_failure = {
                "failure_kind": FAILURE_KIND_UNCERTAIN_SIDE_EFFECT,
                "exit_code": exit_code,
                "stderr_tail": _clip_diag(self._stderr_tail),
            }
            # Mark all local sessions closed — process state is gone.
            for session in self._sessions.values():
                session.closed = True
                session.side_effects_started = True
                session.pending_prompt_ids.clear()

        self._fail_all_pending(
            ACPUncertainSideEffectError(
                "ACP process exited unexpectedly",
                details={
                    "exit_code": exit_code,
                    "stderr_tail": _clip_diag(self._stderr_tail),
                },
            ),
            uncertain=True,
        )

        # Restart policy: transport only, never replay.
        should_restart = (
            self.restart_policy.enabled
            and self.restart_policy.restart_on_unexpected_exit
            and not self._stop_event.is_set()
        )
        if should_restart:
            try:
                self.restart_transport(explicit=False)
                return
            except ACPRestartExhaustedError:
                with self._lock:
                    self._state = ACPClientState.FAILED
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ACP auto-restart failed: %s", type(exc).__name__
                )
                with self._lock:
                    self._state = ACPClientState.FAILED
                return
        with self._lock:
            self._state = ACPClientState.FAILED

    def _fail_all_pending(
        self, error: BaseException, *, uncertain: bool
    ) -> None:
        with self._lock:
            pending_items = list(self._pending.items())
            self._pending.clear()
            for rec in self._sessions.values():
                rec.pending_prompt_ids.clear()
                if uncertain:
                    rec.side_effects_started = True
        for _rid, pending in pending_items:
            if pending.event.is_set() and pending.response is not None:
                continue
            pending.error = error
            if uncertain and not isinstance(error, ACPUncertainSideEffectError):
                pending.error = ACPUncertainSideEffectError(
                    str(error),
                    details={"method": pending.method},
                )
            pending.event.set()

    def _kill_process(
        self, *, reason: str, timeout: float = 5.0
    ) -> None:
        self._stop_event.set()
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        terminate_process_tree(
            process,
            grace_seconds=min(self.bounds.term_grace_seconds, timeout),
            kill_wait_seconds=min(self.bounds.kill_wait_seconds, timeout),
            clock=self._clock,
        )
        # Join reader threads briefly.
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
        self._reader_thread = None
        self._stderr_thread = None
        logger.debug("ACP process terminated reason=%s", reason)

    def _validate_session_id(self, session_id: Any) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ContractValidationError("session_id must be a non-empty string")
        text = session_id.strip()
        if len(text) > MAX_SESSION_ID_CHARS:
            raise BoundsExceededError(
                f"session_id exceeds {MAX_SESSION_ID_CHARS} characters",
                details={
                    "length": len(text),
                    "maximum": MAX_SESSION_ID_CHARS,
                },
            )
        return text

    def _sanitize_result(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a bounded, prompt-safe diagnostic view of a result mapping."""
        out: dict[str, Any] = {}
        for key, value in list(payload.items())[:64]:
            key_s = str(key)
            lowered = key_s.lower()
            if any(
                marker in lowered
                for marker in (
                    "prompt",
                    "password",
                    "secret",
                    "token",
                    "api_key",
                    "authorization",
                    "credential",
                )
            ):
                out[key_s] = "[redacted]"
                continue
            if isinstance(value, str):
                out[key_s] = _clip_text(value, MAX_EVENT_PAYLOAD_CHARS)
            elif isinstance(value, (int, float, bool)) or value is None:
                out[key_s] = value
            elif isinstance(value, Mapping):
                out[key_s] = {
                    str(k): _clip_text(v, 256)
                    for k, v in list(value.items())[:16]
                }
            elif isinstance(value, (list, tuple)):
                out[key_s] = f"[list:{len(value)}]"
            else:
                out[key_s] = type(value).__name__
        return out

    def __enter__(self) -> GooseACPClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            if self._process is not None:
                self.stop()
        except Exception:
            pass


class _Missing:
    pass


_MISSING = _Missing()


def create_goose_acp_client(
    executable: str,
    state_root: str,
    **kwargs: Any,
) -> GooseACPClient:
    """Factory for :class:`GooseACPClient` (does not start the process)."""
    return GooseACPClient(executable, state_root, **kwargs)


__all__ = [
    "ACP_PROTOCOL_VERSION",
    "ACPBounds",
    "ACPCapacityError",
    "ACPClientState",
    "ACPError",
    "ACPNotReadyError",
    "ACPProtocolError",
    "ACPRestartExhaustedError",
    "ACPRestartPolicy",
    "ACPSessionRecord",
    "ACPUncertainSideEffectError",
    "CLIENT_NAME",
    "CLIENT_VERSION",
    "DEFAULT_MAX_IDLE_SECONDS",
    "DEFAULT_MAX_PENDING_REQUESTS",
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_MAX_SESSIONS",
    "FAILURE_KIND_UNCERTAIN_SIDE_EFFECT",
    "STATUS_UNCERTAIN_SIDE_EFFECT",
    "GooseACPClient",
    "build_text_prompt_blocks",
    "create_goose_acp_client",
    "encode_acp_message",
    "parse_acp_line",
    "split_ndjson_buffer",
]
