"""Canonical Muse Code CLI adapter: ``muse exec`` argv, JSONL parsing, errors.

This module owns Muse Code command construction, JSONL event parsing, and
typed failure classification. Other surfaces (llm_router, endpoints, MCP,
compatibility wrappers) must delegate here.

Safety invariants:

- Headless runs always use ``muse exec`` (never the interactive TUI).
- Prompt text is supplied via ``--prompt-file`` (a temp file), never
  interpolated into a shell string and never placed as a free argv blob
  when a file can be used.
- Default automation posture is ``--disable-approval``: skip prompts but
  keep the OS sandbox on. ``--yolo`` (disable approval *and* sandbox, and
  trust the workspace) requires an explicit flag.
- ``muse exec`` is always side-effecting: the agent may edit files and run
  commands. Ordinary router generation therefore disables response cache
  and cross-provider retry.
- Dynamic values remain single argv entries; shell execution is forbidden.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

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
    MalformedOutputError,
    PolicyDeniedError,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimeoutError,
)
from ..installers.muse import (
    MuseInstallResult,
    MuseReadiness,
    assess_muse_readiness,
    discover_muse,
    ensure_muse,
    muse_auth_available,
)
from ..process_runner import (
    CancellationToken,
    ProcessRunner,
    ProcessRunResult,
    ProcessSpec,
)

PROVIDER_NAME: str = "muse_code"
PROVIDER_ALIASES: tuple[str, ...] = (
    "muse",
    "muse-code",
    "musecode",
    "muse_cli",
    "muse-cli",
)

DEFAULT_MODEL: str = "muse-spark-1.2"
DEFAULT_CHAT_MAX_MODEL_STEPS: int = 8
DEFAULT_AGENT_MAX_MODEL_STEPS: int = 40
DEFAULT_CHAT_TIMEOUT_SECONDS: float = 180.0
DEFAULT_AGENT_TIMEOUT_SECONDS: float = 600.0

REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)
APPROVAL_MODES: frozenset[str] = frozenset(
    {"default", "disable_approval", "yolo", "unrestricted"}
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


class MuseErrorKind(str, Enum):
    """Fine-grained Muse Code failure kinds (mapped onto CLIRuntimeErrorCode)."""

    NOT_INSTALLED = "not_installed"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    QUOTA_RATE_LIMIT = "quota_rate_limit"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIAL = "policy_denial"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    MALFORMED_OUTPUT = "malformed_output"
    NONZERO_EXIT = "nonzero_exit"
    SPAWN_FAILED = "spawn_failed"
    USAGE_ERROR = "usage_error"
    INTERNAL = "internal"


_KIND_TO_CODE: Mapping[MuseErrorKind, CLIRuntimeErrorCode] = {
    MuseErrorKind.NOT_INSTALLED: CLIRuntimeErrorCode.SPAWN_FAILED,
    MuseErrorKind.AUTHENTICATION: CLIRuntimeErrorCode.AUTHENTICATION_FAILED,
    MuseErrorKind.BILLING: CLIRuntimeErrorCode.CAPACITY_EXCEEDED,
    MuseErrorKind.QUOTA_RATE_LIMIT: CLIRuntimeErrorCode.CAPACITY_EXCEEDED,
    MuseErrorKind.APPROVAL_REQUIRED: CLIRuntimeErrorCode.POLICY_DENIED,
    MuseErrorKind.POLICY_DENIAL: CLIRuntimeErrorCode.POLICY_DENIED,
    MuseErrorKind.TIMEOUT: CLIRuntimeErrorCode.TIMEOUT,
    MuseErrorKind.CANCELLATION: CLIRuntimeErrorCode.CANCELLED,
    MuseErrorKind.MALFORMED_OUTPUT: CLIRuntimeErrorCode.MALFORMED_OUTPUT,
    MuseErrorKind.NONZERO_EXIT: CLIRuntimeErrorCode.NONZERO_EXIT,
    MuseErrorKind.SPAWN_FAILED: CLIRuntimeErrorCode.SPAWN_FAILED,
    MuseErrorKind.USAGE_ERROR: CLIRuntimeErrorCode.INVALID_CONTRACT,
    MuseErrorKind.INTERNAL: CLIRuntimeErrorCode.INTERNAL,
}


def muse_error_code(kind: MuseErrorKind) -> CLIRuntimeErrorCode:
    return _KIND_TO_CODE.get(kind, CLIRuntimeErrorCode.INTERNAL)


def muse_kind_retryable(kind: MuseErrorKind) -> bool:
    """429-class Muse failures are retryable; billing/auth are not."""
    return kind is MuseErrorKind.QUOTA_RATE_LIMIT


def _clip(value: Any, maximum: int = MAX_METADATA_VALUE_CHARS) -> str:
    text = str("" if value is None else value)
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def make_muse_error(
    kind: MuseErrorKind,
    message: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    retryable: bool = False,
) -> CLIErrorRecord:
    payload: dict[str, str] = {"muse_error_kind": kind.value}
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
        code=muse_error_code(kind),
        message=_clip(message, 4096) or kind.value,
        retryable=retryable,
        details=payload,
    )


class MuseProviderError(CLIRuntimeError):
    """Raised by the Muse Code adapter with a typed :class:`MuseErrorKind`."""

    def __init__(
        self,
        message: str,
        *,
        kind: MuseErrorKind = MuseErrorKind.INTERNAL,
        details: Optional[Mapping[str, Any]] = None,
        retryable: bool = False,
        side_effects_started: bool = False,
    ) -> None:
        record = make_muse_error(kind, message, details=details, retryable=retryable)
        super().__init__(
            record.message,
            code=record.code,
            retryable=record.retryable,
            details=record.details,
        )
        self.kind = kind
        self.side_effects_started = bool(side_effects_started)


@dataclass(frozen=True)
class MuseCommandPlan:
    """Validated argv for one ``muse exec`` invocation."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Optional[str]
    mode: ExecutionMode
    model_name: Optional[str]
    json_events: bool
    max_model_steps: int
    side_effecting: bool
    yolo: bool
    prompt_file: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MuseParsedOutput:
    """Bounded parse of Muse Code stdout (JSONL events or plain text)."""

    text: str
    events: tuple[CLIEvent, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    side_effects_started: bool = True
    tool_call_count: int = 0
    message_count: int = 0
    status: str = "completed"
    raw_format: str = "text"


def build_muse_exec_argv(
    *,
    executable: str,
    prompt_file: str,
    model_name: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_model_steps: Optional[int] = None,
    json_events: bool = True,
    disable_approval: bool = True,
    yolo: bool = False,
    trust_workspace: bool = False,
    disable_sandbox: bool = False,
    workspace: Optional[str] = None,
    session_id: Optional[str] = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build ``muse exec`` argv. Prompt lives in ``prompt_file``, not argv."""
    binary = str(executable or "").strip()
    if not binary:
        raise ContractValidationError("muse executable is required")
    file_path = str(prompt_file or "").strip()
    if not file_path:
        raise ContractValidationError("muse --prompt-file path is required")

    argv: list[str] = [binary, "exec"]
    if json_events:
        argv.append("--json")
    if model_name and str(model_name).strip():
        argv.extend(["--model", str(model_name).strip()])
    effort = str(reasoning_effort or "").strip().lower()
    if effort:
        if effort not in REASONING_EFFORTS:
            raise ContractValidationError(
                f"unsupported Muse reasoning effort: {reasoning_effort!r}"
            )
        argv.extend(["--reasoning-effort", effort])
    steps = int(max_model_steps) if max_model_steps is not None else None
    if steps is not None:
        if steps < 1:
            raise ContractValidationError("max_model_steps must be >= 1")
        argv.extend(["--max-model-steps", str(steps)])
    if yolo:
        argv.append("--yolo")
    else:
        if disable_approval:
            argv.append("--disable-approval")
        if disable_sandbox:
            argv.append("--disable-sandbox")
        if trust_workspace:
            argv.append("--trust-workspace")
    if workspace and str(workspace).strip():
        argv.extend(["--workspace", str(workspace).strip()])
    if session_id and str(session_id).strip():
        argv.extend(["--session-id", str(session_id).strip()])
    if extra_args:
        argv.extend(str(item) for item in extra_args if str(item))
    argv.extend(["--prompt-file", file_path])
    return argv


def muse_login_present(*, environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when ``muse login`` left a local auth file. Never reads the secret."""
    home = str((environ or os.environ).get("HOME") or Path.home())
    auth = Path(home) / ".config" / "muse" / "auth.json"
    try:
        return auth.is_file() and auth.stat().st_size > 2
    except OSError:
        return False


def isolated_muse_tmpdir(workspace: Optional[str] = None) -> str:
    """Temp dir that is never inside the Muse workspace (sandbox constraint)."""
    path = tempfile.mkdtemp(prefix="muse-rt-")
    if not workspace:
        return path
    try:
        ws = Path(workspace).expanduser().resolve()
        tmp = Path(path).resolve()
        tmp.relative_to(ws)
    except (OSError, ValueError):
        return path
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    parent = Path(workspace).expanduser().resolve().parent
    return tempfile.mkdtemp(prefix="muse-rt-", dir=str(parent))


def build_muse_process_env(
    *,
    base_env: Optional[Mapping[str, str]] = None,
    extra: Optional[Mapping[str, Optional[str]]] = None,
    prefer_login: bool = True,
) -> dict[str, Optional[str]]:
    """Return a process-runner env overlay.

    ``META_API_KEY`` already in the environment is left alone. Otherwise a
    stored secrets-manager key is injected only when ``muse login`` is absent
    (login is otherwise shadowed and 402s). Never logs credential values.
    """
    overlay: dict[str, Optional[str]] = {}
    env = dict(base_env or os.environ)
    if not str(env.get("META_API_KEY") or "").strip():
        skip_stored = bool(prefer_login and muse_login_present(environ=env))
        if not skip_stored:
            for name in (
                "MODEL_API_KEY",
                "META_AI_API_KEY",
                "ipfs_accelerate_py_META_AI_API_KEY",
                "IPFS_ACCELERATE_PY_META_AI_API_KEY",
            ):
                value = str(env.get(name) or "").strip()
                if value:
                    overlay["META_API_KEY"] = value
                    break
            if "META_API_KEY" not in overlay:
                try:
                    from ...common.meta_model_api import resolve_meta_model_api_key

                    resolved = resolve_meta_model_api_key()
                except Exception:
                    resolved = None
                if resolved:
                    overlay["META_API_KEY"] = resolved
    local_bin = str(Path.home() / ".local" / "bin")
    path = env.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        overlay["PATH"] = local_bin + os.pathsep + path
    if extra:
        for key, value in extra.items():
            overlay[str(key)] = None if value is None else str(value)
    return overlay


_LAST_MUSE_SESSION = threading.local()
_LAST_MUSE_OBSERVATION = threading.local()
_PROMPT_META_KEYS = frozenset({"prompt", "input", "user_prompt", "user_input"})


def get_last_muse_session_id() -> str:
    """Return the native Muse session id from the last ``muse exec`` on this thread."""
    return str(getattr(_LAST_MUSE_SESSION, "session_id", "") or "")


def get_last_muse_observation() -> dict[str, Any]:
    """Bounded last-run metadata. Never includes prompts or credentials."""
    payload = getattr(_LAST_MUSE_OBSERVATION, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _set_last_muse_session_id(value: Optional[str]) -> None:
    _LAST_MUSE_SESSION.session_id = str(value or "").strip()[:256]


def _set_last_muse_observation(payload: Mapping[str, Any]) -> None:
    clean: dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        k = str(key or "").strip()
        if not k or k.lower() in _PROMPT_META_KEYS:
            continue
        if any(marker in k.lower() for marker in _SENSITIVE_ENV_MARKERS):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clean[k] = value
        else:
            text = _clip(value, 256)
            if text:
                clean[k] = text
        if len(clean) >= 48:
            break
    _LAST_MUSE_OBSERVATION.payload = clean
    try:
        from ..cli_metadata import set_last_cli_observation

        set_last_cli_observation(PROVIDER_NAME, clean)
    except Exception:
        pass


def _session_id_from_mapping(payload: Mapping[str, Any], *, depth: int = 0) -> str:
    if depth > 3:
        return ""
    for key in ("session_id", "sessionId", "sessionID"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:256]
    nested = payload.get("session")
    if isinstance(nested, Mapping):
        found = _session_id_from_mapping(nested, depth=depth + 1)
        if found:
            return found
        ident = nested.get("id")
        if isinstance(ident, str) and ident.strip():
            return ident.strip()[:256]
    stream = payload.get("stream")
    if isinstance(stream, Mapping) and str(stream.get("kind") or "").lower() == "session":
        ident = stream.get("id")
        if isinstance(ident, str) and ident.strip():
            return ident.strip()[:256]
    inner = payload.get("payload")
    if isinstance(inner, Mapping):
        return _session_id_from_mapping(inner, depth=depth + 1)
    return ""


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "message", "content", "delta"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return _extract_text_from_content(content.get("parts"))
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _put_meta(meta: dict[str, str], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    k = str(key or "").strip()
    if not k or k.lower() in _PROMPT_META_KEYS:
        return
    if any(marker in k.lower() for marker in _SENSITIVE_ENV_MARKERS):
        return
    meta[k] = _clip(value, 256)


def _ingest_muse_facets(payload: Mapping[str, Any], meta: dict[str, str]) -> None:
    event = payload.get("event")
    if not isinstance(event, Mapping):
        return
    details = event.get("details")
    if not isinstance(details, Mapping):
        return
    phase = details.get("phase")
    if phase:
        _put_meta(meta, "phase", phase)
    facets = details.get("facets")
    if not isinstance(facets, Sequence) or isinstance(facets, (str, bytes)):
        return
    for facet in facets:
        if not isinstance(facet, Mapping):
            continue
        kind = str(facet.get("kind") or "")
        if kind == "external_attempt":
            _put_meta(meta, "attempt", facet.get("attempt"))
            _put_meta(meta, "max_attempts", facet.get("max_attempts"))
            _put_meta(meta, "operation", facet.get("operation"))
            if facet.get("error_kind"):
                _put_meta(meta, "stream_error_kind", facet.get("error_kind"))
            if facet.get("system"):
                _put_meta(meta, "system", facet.get("system"))
        detail = facet.get("detail")
        if not isinstance(detail, Mapping):
            continue
        if detail.get("provider"):
            _put_meta(meta, "provider_id", detail.get("provider"))
        if detail.get("model"):
            _put_meta(meta, "model_id", detail.get("model"))
        _put_meta(meta, "request_id", detail.get("request_id"))
        _put_meta(meta, "response_id", detail.get("response_id"))
        for tok in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            if tok in detail:
                _put_meta(meta, tok, detail.get(tok))
        stream = detail.get("stream")
        if not isinstance(stream, Mapping):
            continue
        _put_meta(meta, "time_to_first_event_ms", stream.get("time_to_first_event_ms"))
        _put_meta(meta, "stream_bytes_read", stream.get("bytes_read"))
        _put_meta(meta, "wire_events_seen", stream.get("wire_events_seen"))
        _put_meta(meta, "last_wire_event_type", stream.get("last_wire_event_type"))


def parse_muse_jsonl(
    stdout: str,
    *,
    max_text_chars: int = MAX_TEXT_CHARS,
) -> MuseParsedOutput:
    """Parse ``muse exec --json`` JSONL events. Falls back to plain stdout.

    Collects bounded run metadata (session/run/model/request ids, TTFT, stream
    stats). Never stores prompts, credentials, or raw envelopes.
    """
    raw = stdout or ""
    output_chunks: list[str] = []
    assistant_messages: list[str] = []
    terminal_text = ""
    tool_call_count = 0
    side_effect_intents = 0
    events: list[CLIEvent] = []
    seq = 0
    status = "completed"
    bounded_meta: dict[str, str] = {}
    message_count = 0
    event_count = 0
    parsed_any = False

    for line_no, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        parsed_any = True
        event_count += 1
        session_id = _session_id_from_mapping(event)
        if session_id:
            bounded_meta["session_id"] = session_id
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        payload_type = str(
            event.get("payload_type") or payload.get("kind") or ""
        ).lower()
        etype = str(
            event.get("type")
            or event.get("event")
            or event.get("kind")
            or payload_type
            or ""
        ).lower()
        role = str(event.get("role") or payload.get("role") or "").lower()
        run_stream = payload.get("run_stream") if isinstance(payload.get("run_stream"), Mapping) else {}
        if run_stream.get("id"):
            _put_meta(bounded_meta, "run_id", run_stream.get("id"))
        _put_meta(bounded_meta, "command_id", payload.get("command_id"))
        if payload_type == "run.model.configured":
            _put_meta(bounded_meta, "model_id", payload.get("model_id"))
            _put_meta(bounded_meta, "display_label", payload.get("display_label"))
            _put_meta(bounded_meta, "profile_id", payload.get("profile_id"))
            _put_meta(bounded_meta, "provider_id", payload.get("provider_id"))
        _ingest_muse_facets(payload, bounded_meta)
        if payload_type.endswith("failed") or "failed" in etype:
            status = "failed"
            nested = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
            reason = payload.get("reason") or nested.get("reason") or payload.get("error")
            if reason:
                _put_meta(bounded_meta, "error_reason", reason)
        if payload_type == "run.terminal.completed":
            _put_meta(bounded_meta, "terminal", payload.get("terminal") or "completed")
            status = "completed"
        if payload_type == "task.lifecycle.side_effect_intent" or "side_effect" in payload_type:
            side_effect_intents += 1
        if "input.user" in payload_type or payload_type.endswith(".user"):
            continue
        text = _extract_text_from_content(
            event.get("text")
            or event.get("message")
            or event.get("content")
            or event.get("delta")
            or event.get("output")
            or payload.get("text")
            or payload.get("message")
            or payload.get("content")
            or payload.get("delta")
            or payload.get("output")
            or payload.get("body")
        )
        if etype in {"tool", "tool_call", "tool_use", "tool_result", "command", "shell"}:
            tool_call_count += 1
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.TOOL_CALL if "result" not in etype else EventKind.TOOL_RESULT,
                        sequence=seq,
                        message=_clip(etype, 64),
                        side_effecting=True,
                    )
                )
                seq += 1
        if etype in {"error", "failed"} or payload_type.endswith("failed"):
            status = "failed"
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.FAILED,
                        sequence=seq,
                        message=_clip(
                            bounded_meta.get("error_reason") or event.get("message") or etype,
                            512,
                        ),
                    )
                )
                seq += 1
        if payload_type == "run.output.delta" and text:
            output_chunks.append(text)
            message_count += 1
            if len(events) < MAX_EVENT_COUNT:
                events.append(
                    CLIEvent(
                        kind=EventKind.TEXT_DELTA,
                        sequence=seq,
                        message=_clip(text, 512),
                    )
                )
                seq += 1
            continue
        if payload_type == "run.terminal.completed" and text:
            terminal_text = text
            message_count += 1
            continue
        if role in {"assistant", "agent", "muse"} or etype in {
            "text",
            "text_delta",
            "delta",
            "message",
            "assistant",
            "output",
        } or any(
            token in payload_type
            for token in (
                "assistant",
                "output.text",
                "message.delta",
                "turn.output",
                "agent.message",
            )
        ):
            if text:
                assistant_messages.append(text)
                message_count += 1
                if len(events) < MAX_EVENT_COUNT:
                    events.append(
                        CLIEvent(
                            kind=EventKind.TEXT_DELTA,
                            sequence=seq,
                            message=_clip(text, 512),
                        )
                    )
                    seq += 1

    if not parsed_any:
        text_out = raw.strip()
        return MuseParsedOutput(
            text=text_out[:max_text_chars],
            events=(),
            metadata={},
            side_effects_started=True,
            tool_call_count=0,
            message_count=1 if text_out else 0,
            status="completed",
            raw_format="text",
        )

    text_out = (
        terminal_text
        or "".join(output_chunks)
        or (assistant_messages[-1] if assistant_messages else "")
    )
    if len(text_out) > max_text_chars:
        text_out = text_out[: max_text_chars - 3] + "..."
    _put_meta(bounded_meta, "event_count", event_count)
    _put_meta(bounded_meta, "side_effect_intent_count", side_effect_intents)
    _put_meta(bounded_meta, "tool_call_count", tool_call_count)
    return MuseParsedOutput(
        text=text_out,
        events=tuple(events),
        metadata=bounded_meta,
        side_effects_started=True,
        tool_call_count=tool_call_count or side_effect_intents,
        message_count=message_count,
        status=status,
        raw_format="jsonl",
    )


def classify_muse_failure(
    *,
    exit_code: Optional[int],
    stdout: str,
    stderr: str,
) -> MuseErrorKind:
    """Map Muse Code exit codes and diagnostic text onto a typed kind."""
    combined = f"{stdout or ''}\n{stderr or ''}"
    lowered = combined.lower()
    if exit_code in {130, 143}:
        return MuseErrorKind.CANCELLATION
    if exit_code == 2:
        return MuseErrorKind.USAGE_ERROR
    try:
        from ...common.meta_model_api import (
            MetaModelApiErrorKind,
            classify_meta_diagnostic_text,
        )

        failure = classify_meta_diagnostic_text(combined)
        if failure.kind is MetaModelApiErrorKind.AUTHENTICATION:
            return MuseErrorKind.AUTHENTICATION
        if failure.kind is MetaModelApiErrorKind.BILLING:
            return MuseErrorKind.BILLING
        if failure.kind is MetaModelApiErrorKind.RATE_LIMIT:
            return MuseErrorKind.QUOTA_RATE_LIMIT
    except Exception:
        pass
    if any(token in lowered for token in ("api key", "unauthenticated", "not logged in", "login")):
        return MuseErrorKind.AUTHENTICATION
    if any(
        token in lowered
        for token in (
            "billing",
            "payment required",
            "insufficient_quota",
            "insufficient quota",
        )
    ):
        return MuseErrorKind.BILLING
    if any(token in lowered for token in ("rate limit", "quota", "usage limit")):
        return MuseErrorKind.QUOTA_RATE_LIMIT
    if any(token in lowered for token in ("approval", "permission denied", "sandbox")):
        return MuseErrorKind.APPROVAL_REQUIRED
    if exit_code not in {0, None}:
        return MuseErrorKind.NONZERO_EXIT
    return MuseErrorKind.INTERNAL


def muse_provider_spec() -> ProviderSpec:
    return ProviderSpec(
        name=PROVIDER_NAME,
        aliases=PROVIDER_ALIASES,
        description=(
            "Meta Muse Code CLI via `muse exec`. Always side-effecting; "
            "default automation keeps the OS sandbox and skips approval prompts."
        ),
        capabilities=CLICapabilities.chat_defaults(),
        locality="remote",
        metadata={
            "command_contract": "muse exec",
            "default_model": DEFAULT_MODEL,
            "install": "curl -fsSL https://dev.meta.ai/install.sh | bash",
        },
    )


@dataclass
class MuseCLIProvider:
    """Canonical Muse Code adapter implementing the string provider surface.

    Construction never installs software or starts processes. Discovery is
    detect-only unless ``allow_install=True`` is passed to an explicit
    resolve path such as :meth:`ensure_ready`.
    """

    executable: Optional[str] = None
    version: str = ""
    runner: Optional[ProcessRunner] = None
    base_env: Optional[Mapping[str, str]] = None
    default_model: Optional[str] = None
    allow_install: bool = False
    discover_kwargs: Mapping[str, Any] = field(default_factory=dict)
    _resolved: bool = field(default=False, init=False, repr=False)

    def discover(
        self,
        *,
        explicit_path: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> MuseInstallResult:
        merged = dict(self.discover_kwargs)
        merged.update(kwargs)
        result = discover_muse(
            explicit_path=explicit_path or self.executable,
            environ=environ if environ is not None else self.base_env,
            **merged,
        )
        if result.available:
            self.executable = result.executable
            self.version = result.version or self.version
            self._resolved = True
        return result

    def ensure_ready(
        self,
        *,
        auto_install: Optional[bool] = None,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> MuseInstallResult:
        do_install = self.allow_install if auto_install is None else bool(auto_install)
        env = environ if environ is not None else self.base_env
        merged = dict(self.discover_kwargs)
        merged.update(kwargs)
        if do_install:
            result = ensure_muse(
                explicit_path=self.executable,
                auto_install=True,
                environ=env,
                **merged,
            )
        else:
            result = discover_muse(
                explicit_path=self.executable,
                environ=env,
                **merged,
            )
        if result.available:
            self.executable = result.executable
            self.version = result.version or self.version
            self._resolved = True
        return result

    def readiness(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        auto_install: bool = False,
    ) -> MuseReadiness:
        install = self.discover(environ=environ) if not auto_install else None
        return assess_muse_readiness(
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
            raise MuseProviderError(
                "Muse Code CLI is not installed",
                kind=MuseErrorKind.NOT_INSTALLED,
                details={"reason": result.reason or "not_installed"},
            )
        return str(result.executable)

    def generate(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Run ``muse exec`` and return the last assistant text."""
        agent = bool(
            kwargs.get("agent")
            or kwargs.get("side_effecting")
            or kwargs.get("allow_side_effects")
            or kwargs.get("yolo")
        )
        # muse exec can edit files / run commands, so the request is always
        # agent-mode at the contract layer. ``agent=False`` only tightens
        # max-model-steps and forbids --yolo.
        steps = kwargs.get("max_model_steps") or kwargs.get("max_turns")
        if steps is None and not agent:
            steps = DEFAULT_CHAT_MAX_MODEL_STEPS
        result = self.generate_result(
            CLIRequest(
                prompt=str(prompt),
                mode=ExecutionMode.AGENT,
                model_name=model_name,
                provider_name=PROVIDER_NAME,
                side_effecting=True,
                cacheable=False,
                retryable=False,
                session_id=kwargs.get("session_id"),
                timeout_seconds=kwargs.get("timeout"),
                workspace=kwargs.get("workspace") or kwargs.get("cwd"),
                metadata={
                    k: str(v)
                    for k, v in {
                        "reasoning_effort": kwargs.get("reasoning_effort"),
                        "max_model_steps": steps,
                        "yolo": kwargs.get("yolo"),
                        "trust_workspace": kwargs.get("trust_workspace"),
                        "disable_sandbox": kwargs.get("disable_sandbox"),
                        "json": kwargs.get("json", True),
                        "bounded_profile": "chat" if not agent else "agent",
                    }.items()
                    if v is not None
                },
            ),
            yolo=bool(kwargs.get("yolo") or kwargs.get("unrestricted")),
            disable_approval=kwargs.get("disable_approval", True),
            trust_workspace=bool(kwargs.get("trust_workspace")),
            disable_sandbox=bool(kwargs.get("disable_sandbox")),
            reasoning_effort=kwargs.get("reasoning_effort"),
            max_model_steps=steps,
            json_events=bool(kwargs.get("json", True)),
            cancel_token=kwargs.get("cancel_token"),
            bounded_profile="chat" if not agent else "agent",
        )
        if result.error is not None:
            kind = MuseErrorKind.INTERNAL
            raw_kind = result.metadata.get("muse_error_kind") if result.metadata else None
            if raw_kind:
                try:
                    kind = MuseErrorKind(str(raw_kind))
                except ValueError:
                    kind = MuseErrorKind.INTERNAL
            raise MuseProviderError(
                result.error.message,
                kind=kind,
                details=result.error.details,
                retryable=muse_kind_retryable(kind),
                side_effects_started=bool(
                    result.side_effecting or result.had_side_effect_event
                ),
            )
        return str(result.text or "")

    def generate_result(
        self,
        request: CLIRequest,
        *,
        yolo: bool = False,
        disable_approval: bool = True,
        trust_workspace: bool = False,
        disable_sandbox: bool = False,
        reasoning_effort: Optional[str] = None,
        max_model_steps: Optional[int] = None,
        json_events: bool = True,
        cancel_token: Optional[CancellationToken] = None,
        bounded_profile: str = "agent",
    ) -> CLIResult:
        if not isinstance(request, CLIRequest):
            raise ContractValidationError("request must be a CLIRequest")

        mode = request.mode
        profile = str(
            bounded_profile or request.metadata.get("bounded_profile") or "agent"
        ).strip().lower()
        if yolo and profile == "chat":
            raise PolicyDeniedError(
                "--yolo is not allowed for bounded Muse Code generate_text requests",
                details={"profile": "chat"},
            )

        model = (
            request.effective_model()
            if hasattr(request, "effective_model")
            else request.model_name
        ) or self.default_model or DEFAULT_MODEL
        steps = max_model_steps
        if steps is None and "max_model_steps" in request.metadata:
            try:
                steps = int(request.metadata["max_model_steps"])
            except (TypeError, ValueError):
                steps = None
        if steps is None:
            steps = (
                DEFAULT_CHAT_MAX_MODEL_STEPS
                if profile == "chat"
                else DEFAULT_AGENT_MAX_MODEL_STEPS
            )
        effort = reasoning_effort or request.metadata.get("reasoning_effort")
        json_flag = json_events
        if "json" in request.metadata:
            json_flag = str(request.metadata.get("json")).strip().lower() not in {
                "0",
                "false",
                "no",
            }
        if "yolo" in request.metadata:
            yolo = str(request.metadata.get("yolo")).strip().lower() in {
                "1",
                "true",
                "yes",
            }

        executable = self._require_executable()
        timeout = request.timeout_seconds
        if timeout is None:
            timeout = (
                DEFAULT_CHAT_TIMEOUT_SECONDS
                if profile == "chat"
                else DEFAULT_AGENT_TIMEOUT_SECONDS
            )

        prompt_file: Optional[str] = None
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="muse-prompt-",
            delete=False,
            encoding="utf-8",
        )
        try:
            handle.write(request.prompt)
            handle.flush()
            prompt_file = handle.name
        finally:
            handle.close()

        runtime_tmp = ""
        try:
            argv = build_muse_exec_argv(
                executable=executable,
                prompt_file=prompt_file,
                model_name=model,
                reasoning_effort=effort,
                max_model_steps=steps,
                json_events=json_flag,
                disable_approval=disable_approval,
                yolo=yolo,
                trust_workspace=trust_workspace,
                disable_sandbox=disable_sandbox,
                workspace=request.workspace,
                session_id=request.session_id,
            )
            plan = MuseCommandPlan(
                argv=tuple(argv),
                env={},
                cwd=request.workspace,
                mode=mode,
                model_name=model,
                json_events=json_flag,
                max_model_steps=int(steps),
                side_effecting=True,
                yolo=yolo,
                prompt_file=prompt_file,
                metadata={
                    "mode": mode.value,
                    "max_model_steps": str(steps),
                    "yolo": "true" if yolo else "false",
                },
            )
            runtime_tmp = isolated_muse_tmpdir(request.workspace)
            env_overlay = build_muse_process_env(
                base_env=self.base_env,
                extra={"TMPDIR": runtime_tmp},
            )
            runner = self.runner or ProcessRunner(
                base_env=dict(self.base_env) if self.base_env is not None else None,
            )
            spec = ProcessSpec(
                argv=plan.argv,
                cwd=plan.cwd,
                env=env_overlay,
                env_overlay=True,
                stdin=None,
                timeout_seconds=float(timeout),
                side_effecting=True,
                mode=mode,
                provider_name=PROVIDER_NAME,
                model_name=model,
                metadata=dict(plan.metadata),
                cancel_token=cancel_token,
            )
            try:
                proc_result = runner.run(spec)
            except ProcessTimeoutError as exc:
                return self._error_result(
                    request,
                    kind=MuseErrorKind.TIMEOUT,
                    message=str(exc),
                    model=model,
                )
            except ProcessCancelledError as exc:
                return self._error_result(
                    request,
                    kind=MuseErrorKind.CANCELLATION,
                    message=str(exc),
                    model=model,
                )
            except ProcessSpawnError as exc:
                return self._error_result(
                    request,
                    kind=MuseErrorKind.NOT_INSTALLED,
                    message=str(exc),
                    model=model,
                    side_effects_started=False,
                )
            except CLIRuntimeError as exc:
                return self._error_result(
                    request,
                    kind=MuseErrorKind.INTERNAL,
                    message=str(exc),
                    model=model,
                )
            return self._result_from_process(request, plan, proc_result, model=model)
        finally:
            if prompt_file:
                try:
                    os.unlink(prompt_file)
                except OSError:
                    pass
            if runtime_tmp:
                try:
                    import shutil

                    shutil.rmtree(runtime_tmp, ignore_errors=True)
                except Exception:
                    pass

    def _error_result(
        self,
        request: CLIRequest,
        *,
        kind: MuseErrorKind,
        message: str,
        model: Optional[str],
        side_effects_started: bool = True,
    ) -> CLIResult:
        record = make_muse_error(
            kind, message, retryable=muse_kind_retryable(kind)
        )
        return CLIResult(
            text="",
            ok=False,
            mode=ExecutionMode.AGENT,
            side_effecting=True,
            cacheable=False,
            retryable=False,
            had_side_effect_event=side_effects_started,
            error=record,
            provider_name=PROVIDER_NAME,
            model_name=model,
            metadata={"muse_error_kind": kind.value},
        )

    def _result_from_process(
        self,
        request: CLIRequest,
        plan: MuseCommandPlan,
        proc_result: ProcessRunResult,
        *,
        model: Optional[str],
    ) -> CLIResult:
        stdout = getattr(proc_result, "stdout", "") or ""
        stderr = getattr(proc_result, "stderr", "") or ""
        exit_code = getattr(proc_result, "exit_code", None)
        parsed = parse_muse_jsonl(stdout)
        native_session = str((parsed.metadata or {}).get("session_id") or request.session_id or "")
        if native_session:
            _set_last_muse_session_id(native_session)
        metadata = {
            str(k): str(v)
            for k, v in dict(parsed.metadata or {}).items()
            if str(k).lower() not in _PROMPT_META_KEYS
        }
        metadata["raw_format"] = parsed.raw_format
        metadata["exit_code"] = str(exit_code if exit_code is not None else "")
        metadata["yolo"] = "true" if plan.yolo else "false"
        metadata["max_model_steps"] = str(plan.max_model_steps)
        if native_session:
            metadata["session_id"] = native_session[:256]
        resolved_model = str(metadata.get("model_id") or model or "")
        _set_last_muse_observation(metadata)
        # Muse documents: 0 = turn completed, 1 = failed/cancelled, 2 = usage.
        if exit_code in {0, None} or parsed.text:
            if exit_code in {0, None}:
                metadata["muse_error_kind"] = ""
                metadata.setdefault("tool_call_count", str(parsed.tool_call_count))
                return CLIResult(
                    text=parsed.text,
                    ok=True,
                    mode=ExecutionMode.AGENT,
                    side_effecting=True,
                    cacheable=False,
                    retryable=False,
                    had_side_effect_event=True,
                    events=parsed.events,
                    provider_name=PROVIDER_NAME,
                    model_name=resolved_model or model,
                    metadata=metadata,
                )
        kind = classify_muse_failure(exit_code=exit_code, stdout=stdout, stderr=stderr)
        message = (stderr or parsed.text or "muse exec failed").strip()
        record = make_muse_error(
            kind,
            message or kind.value,
            details={"returncode": str(exit_code or "")},
            retryable=muse_kind_retryable(kind),
        )
        metadata["muse_error_kind"] = kind.value
        _set_last_muse_observation(metadata)
        return CLIResult(
            text=parsed.text,
            ok=False,
            mode=ExecutionMode.AGENT,
            side_effecting=True,
            cacheable=False,
            retryable=False,
            had_side_effect_event=True,
            events=parsed.events,
            error=record,
            provider_name=PROVIDER_NAME,
            model_name=resolved_model or model,
            metadata=metadata,
        )


def create_muse_provider(
    *,
    executable: Optional[str] = None,
    allow_install: bool = False,
    runner: Optional[ProcessRunner] = None,
    base_env: Optional[Mapping[str, str]] = None,
    default_model: Optional[str] = None,
    **discover_kwargs: Any,
) -> MuseCLIProvider:
    """Factory used by the lazy registry / llm_router integration."""
    return MuseCLIProvider(
        executable=executable,
        allow_install=allow_install,
        runner=runner,
        base_env=base_env,
        default_model=default_model or DEFAULT_MODEL,
        discover_kwargs=discover_kwargs,
    )


__all__ = [
    "PROVIDER_NAME",
    "PROVIDER_ALIASES",
    "DEFAULT_MODEL",
    "DEFAULT_CHAT_MAX_MODEL_STEPS",
    "DEFAULT_AGENT_MAX_MODEL_STEPS",
    "DEFAULT_CHAT_TIMEOUT_SECONDS",
    "DEFAULT_AGENT_TIMEOUT_SECONDS",
    "REASONING_EFFORTS",
    "APPROVAL_MODES",
    "MuseErrorKind",
    "MuseProviderError",
    "MuseCommandPlan",
    "MuseParsedOutput",
    "MuseCLIProvider",
    "build_muse_exec_argv",
    "build_muse_process_env",
    "parse_muse_jsonl",
    "get_last_muse_session_id",
    "get_last_muse_observation",
    "muse_login_present",
    "isolated_muse_tmpdir",
    "classify_muse_failure",
    "muse_error_code",
    "muse_kind_retryable",
    "muse_provider_spec",
    "make_muse_error",
    "create_muse_provider",
    "muse_auth_available",
]
