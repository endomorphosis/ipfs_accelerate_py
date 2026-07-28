"""Provider-neutral typed contracts for the shared CLI runtime.

These records are immutable, deterministically serializable, and length-bounded.
They preserve the existing string-returning ``LLMProvider`` surface while also
supporting richer result and event consumers.

Safety invariants enforced at construction time:

- Agent mode is always side-effecting and never cacheable or blindly retryable.
- Side-effecting requests reject ``cacheable=True`` and ``retryable=True``.
- Chat mode cannot enable tools or sessions without becoming agent/side-effecting.
- Serialization is sorted, JSON-safe, and size-bounded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from .errors import (
    BoundsExceededError,
    CLIErrorRecord,
    CLIRuntimeErrorCode,
    ContractValidationError,
    InvalidStateError,
)

CONTRACT_VERSION: int = 1
CONTRACT_SCHEMA: str = "ipfs_accelerate_py/cli_runtime/contracts@1"

MAX_NAME_CHARS: int = 128
MAX_ALIAS_COUNT: int = 32
MAX_PROMPT_CHARS: int = 524288
MAX_TEXT_CHARS: int = 1048576
MAX_MODEL_CHARS: int = 256
MAX_PROVIDER_CHARS: int = 256
MAX_SESSION_ID_CHARS: int = 256
MAX_TOOL_NAME_CHARS: int = 256
MAX_TOOL_COUNT: int = 64
MAX_METADATA_KEYS: int = 64
MAX_METADATA_KEY_CHARS: int = 128
MAX_METADATA_VALUE_CHARS: int = 4096
MAX_METADATA_BYTES: int = 32768
MAX_EVENT_COUNT: int = 256
MAX_EVENT_PAYLOAD_CHARS: int = 16384
MAX_DESCRIPTION_CHARS: int = 1024
MAX_SERIALIZED_BYTES: int = 2097152
MAX_TIMEOUT_SECONDS: float = 86400.0
MIN_TIMEOUT_SECONDS: float = 0.001
MAX_ARGV_ITEMS: int = 256
MAX_ARGV_ITEM_CHARS: int = 8192

_SENSITIVE_METADATA_MARKERS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


class ExecutionMode(str, Enum):
    """Execution profile for a CLI request."""

    CHAT = "chat"
    AGENT = "agent"


class EventKind(str, Enum):
    """Bounded event kinds emitted during CLI execution."""

    STARTED = "started"
    TEXT_DELTA = "text_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SIDE_EFFECT = "side_effect"
    DIAGNOSTIC = "diagnostic"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HEARTBEAT = "heartbeat"


class CapabilitySupport(str, Enum):
    """Whether a provider supports a capability."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"
    REQUIRES_AUTHORIZATION = "requires_authorization"


def _clip_text(value: Any, maximum: int) -> str:
    text = str("" if value is None else value)
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be a boolean")
    return value


def _require_non_empty_name(
    value: Any,
    field_name: str,
    *,
    maximum: int = MAX_NAME_CHARS,
) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ContractValidationError(f"{field_name} must not be empty")
    if len(text) > maximum:
        raise BoundsExceededError(
            f"{field_name} exceeds {maximum} characters",
            details={"length": len(text), "maximum": maximum},
        )
    return text


def _normalize_identifier(
    value: Any,
    field_name: str,
    *,
    maximum: int = MAX_NAME_CHARS,
) -> str:
    text = _require_non_empty_name(value, field_name, maximum=maximum)
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _bounded_metadata(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ContractValidationError("metadata must be a mapping")
    out: dict[str, str] = {}
    for index, (key, value) in enumerate(raw.items()):
        if index >= MAX_METADATA_KEYS:
            raise BoundsExceededError(
                f"metadata exceeds {MAX_METADATA_KEYS} keys",
                details={"maximum": MAX_METADATA_KEYS},
            )
        key_text = str(key).strip()
        if not key_text:
            raise ContractValidationError("metadata keys must be non-empty strings")
        if len(key_text) > MAX_METADATA_KEY_CHARS:
            raise BoundsExceededError(
                f"metadata key exceeds {MAX_METADATA_KEY_CHARS} characters",
                details={"key": key_text[:64], "maximum": MAX_METADATA_KEY_CHARS},
            )
        lowered = key_text.lower()
        if any(marker in lowered for marker in _SENSITIVE_METADATA_MARKERS):
            out[key_text] = "[redacted]"
        else:
            out[key_text] = _clip_text(value, MAX_METADATA_VALUE_CHARS)
    encoded = json.dumps(
        out, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise BoundsExceededError(
            f"metadata exceeds {MAX_METADATA_BYTES} bytes when serialized",
            details={"length": len(encoded), "maximum": MAX_METADATA_BYTES},
        )
    return out


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (
            value != value or value in (float("inf"), float("-inf"))
        ):
            raise ContractValidationError("non-finite float is not JSON-safe")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    raise ContractValidationError(
        f"value of type {type(value).__name__} is not JSON-safe"
    )


def canonical_json(value: Any) -> str:
    """Return deterministic sorted JSON text for *value*."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes, enforcing the serialized size bound."""
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > MAX_SERIALIZED_BYTES:
        raise BoundsExceededError(
            f"serialized payload exceeds {MAX_SERIALIZED_BYTES} bytes",
            details={"length": len(encoded), "maximum": MAX_SERIALIZED_BYTES},
        )
    return encoded


def ensure_serialized_bound(payload: Mapping[str, Any] | Any) -> None:
    """Fail closed when a mapping cannot be serialized within hard bounds."""
    canonical_json_bytes(payload)


@dataclass(frozen=True)
class CLICapabilities:
    """Capability advertisement for a provider or a concrete request policy.

    Request construction uses these flags to reject unsafe combinations before
    any process is started.
    """

    side_effecting: bool = False
    cacheable: bool = True
    retryable: bool = True
    streaming: bool = False
    sessions: bool = False
    cancellation: bool = True
    tools: bool = False
    provider_override: bool = True
    model_override: bool = True
    chat_mode: bool = True
    agent_mode: bool = False

    def __post_init__(self) -> None:
        for name in (
            "side_effecting",
            "cacheable",
            "retryable",
            "streaming",
            "sessions",
            "cancellation",
            "tools",
            "provider_override",
            "model_override",
            "chat_mode",
            "agent_mode",
        ):
            object.__setattr__(
                self, name, _require_bool(getattr(self, name), name)
            )
        self.validate()

    def validate(self) -> None:
        """Reject capability combinations that violate runtime safety invariants."""
        if self.side_effecting and self.cacheable:
            raise InvalidStateError(
                "side-effecting capabilities cannot be cacheable",
                details={"side_effecting": True, "cacheable": True},
            )
        if self.side_effecting and self.retryable:
            raise InvalidStateError(
                "side-effecting capabilities cannot be blindly retryable",
                details={"side_effecting": True, "retryable": True},
            )
        if self.tools and not self.side_effecting:
            raise InvalidStateError(
                "tools require side_effecting=True",
                details={"tools": True, "side_effecting": False},
            )
        if self.sessions and not self.side_effecting:
            raise InvalidStateError(
                "sessions require side_effecting=True",
                details={"sessions": True, "side_effecting": False},
            )
        if self.agent_mode and not self.side_effecting:
            raise InvalidStateError(
                "agent_mode requires side_effecting=True",
                details={"agent_mode": True, "side_effecting": False},
            )
        if self.agent_mode and self.cacheable:
            raise InvalidStateError(
                "agent_mode cannot be cacheable",
                details={"agent_mode": True, "cacheable": True},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_effecting": self.side_effecting,
            "cacheable": self.cacheable,
            "retryable": self.retryable,
            "streaming": self.streaming,
            "sessions": self.sessions,
            "cancellation": self.cancellation,
            "tools": self.tools,
            "provider_override": self.provider_override,
            "model_override": self.model_override,
            "chat_mode": self.chat_mode,
            "agent_mode": self.agent_mode,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | None
    ) -> CLICapabilities:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ContractValidationError("capabilities payload must be a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ContractValidationError(
                "unknown capability fields: " + ", ".join(sorted(unknown))
            )
        return cls(**{key: bool(payload[key]) for key in payload})

    @classmethod
    def chat_defaults(cls) -> CLICapabilities:
        """Defaults for ordinary side-effect-free chat generation."""
        return cls(
            side_effecting=False,
            cacheable=True,
            retryable=True,
            streaming=False,
            sessions=False,
            cancellation=True,
            tools=False,
            provider_override=True,
            model_override=True,
            chat_mode=True,
            agent_mode=False,
        )

    @classmethod
    def agent_defaults(cls) -> CLICapabilities:
        """Defaults for explicitly authorized side-effecting agent execution."""
        return cls(
            side_effecting=True,
            cacheable=False,
            retryable=False,
            streaming=True,
            sessions=True,
            cancellation=True,
            tools=True,
            provider_override=True,
            model_override=True,
            chat_mode=False,
            agent_mode=True,
        )


def _normalize_execution_mode(value: Any) -> ExecutionMode:
    if isinstance(value, ExecutionMode):
        return value
    try:
        raw = str(value).strip().lower()
        return ExecutionMode(raw)
    except ValueError as exc:
        raise ContractValidationError(
            f"unknown execution mode: {value!r}"
        ) from exc


def _normalize_tool_names(tools: Any) -> tuple[str, ...]:
    if tools is None:
        return ()
    if isinstance(tools, (str, bytes)):
        raise ContractValidationError(
            "tools must be a sequence of names, not a string"
        )
    names: list[str] = []
    seen: set[str] = set()
    for item in tools:
        name = _require_non_empty_name(item, "tool", maximum=MAX_TOOL_NAME_CHARS)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) > MAX_TOOL_COUNT:
            raise BoundsExceededError(
                f"tools exceeds {MAX_TOOL_COUNT} entries",
                details={"maximum": MAX_TOOL_COUNT},
            )
    return tuple(names)


@dataclass(frozen=True)
class CLIRequest:
    """Immutable request record for chat or agent CLI execution.

    ``prompt`` is carried for process stdin only; it is never included in
    :meth:`to_dict` diagnostics so logs and telemetry remain free of prompt
    content. Use :meth:`to_dict` for operator-visible metadata and
    :meth:`to_execution_dict` when a consumer must round-trip the full request
    (tests and trusted in-process handoff).
    """

    prompt: str
    mode: ExecutionMode = ExecutionMode.CHAT
    model_name: Optional[str] = None
    provider_name: Optional[str] = None
    provider_override: Optional[str] = None
    model_override: Optional[str] = None
    side_effecting: bool = False
    cacheable: bool = True
    retryable: bool = True
    streaming: bool = False
    session_id: Optional[str] = None
    tools: tuple[str, ...] = ()
    cancellation_requested: bool = False
    timeout_seconds: Optional[float] = None
    workspace: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    capabilities: CLICapabilities = field(
        default_factory=CLICapabilities.chat_defaults
    )

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise ContractValidationError("prompt must be a string")
        if len(self.prompt) > MAX_PROMPT_CHARS:
            raise BoundsExceededError(
                f"prompt exceeds {MAX_PROMPT_CHARS} characters",
                details={"length": len(self.prompt), "maximum": MAX_PROMPT_CHARS},
            )

        mode = _normalize_execution_mode(self.mode)
        object.__setattr__(self, "mode", mode)

        side_effecting = _require_bool(self.side_effecting, "side_effecting")
        cacheable = _require_bool(self.cacheable, "cacheable")
        retryable = _require_bool(self.retryable, "retryable")
        streaming = _require_bool(self.streaming, "streaming")
        cancellation_requested = _require_bool(
            self.cancellation_requested, "cancellation_requested"
        )

        # Reject explicitly incompatible flags before mode coercion.
        if side_effecting and cacheable:
            raise InvalidStateError(
                "side-effecting requests cannot be cacheable",
                details={"side_effecting": True, "cacheable": True},
            )
        if side_effecting and retryable:
            raise InvalidStateError(
                "side-effecting requests cannot be blindly retryable",
                details={"side_effecting": True, "retryable": True},
            )

        if mode is ExecutionMode.AGENT:
            side_effecting = True
            cacheable = False
            retryable = False

        tools = _normalize_tool_names(self.tools)
        if tools:
            side_effecting = True
            cacheable = False
            retryable = False

        session_id = self.session_id
        if session_id is not None:
            session_id = _require_non_empty_name(
                session_id, "session_id", maximum=MAX_SESSION_ID_CHARS
            )
            side_effecting = True
            cacheable = False
            retryable = False

        if side_effecting and cacheable:
            raise InvalidStateError(
                "side-effecting requests cannot be cacheable",
                details={"side_effecting": True, "cacheable": True},
            )
        if side_effecting and retryable:
            raise InvalidStateError(
                "side-effecting requests cannot be blindly retryable",
                details={"side_effecting": True, "retryable": True},
            )

        if mode is ExecutionMode.CHAT:
            if tools:
                raise InvalidStateError(
                    "chat mode cannot enable tools; use agent mode",
                    details={"mode": mode.value, "tools": list(tools)},
                )
            if session_id is not None:
                raise InvalidStateError(
                    "chat mode cannot resume sessions; use agent mode",
                    details={"mode": mode.value, "session_id": session_id},
                )
            if side_effecting:
                raise InvalidStateError(
                    "chat mode cannot be side-effecting",
                    details={"mode": mode.value, "side_effecting": True},
                )

        model_name = self.model_name
        if model_name is not None:
            model_name = _require_non_empty_name(
                model_name, "model_name", maximum=MAX_MODEL_CHARS
            )
        provider_name = self.provider_name
        if provider_name is not None:
            provider_name = _normalize_identifier(
                provider_name, "provider_name", maximum=MAX_PROVIDER_CHARS
            )
        provider_override = self.provider_override
        if provider_override is not None:
            provider_override = _normalize_identifier(
                provider_override, "provider_override", maximum=MAX_PROVIDER_CHARS
            )
        model_override = self.model_override
        if model_override is not None:
            model_override = _require_non_empty_name(
                model_override, "model_override", maximum=MAX_MODEL_CHARS
            )

        timeout_seconds = self.timeout_seconds
        if timeout_seconds is not None:
            if not isinstance(timeout_seconds, (int, float)) or isinstance(
                timeout_seconds, bool
            ):
                raise ContractValidationError("timeout_seconds must be a number")
            timeout_seconds = float(timeout_seconds)
            if timeout_seconds < MIN_TIMEOUT_SECONDS:
                raise ContractValidationError(
                    f"timeout_seconds must be >= {MIN_TIMEOUT_SECONDS}"
                )
            if timeout_seconds > MAX_TIMEOUT_SECONDS:
                raise BoundsExceededError(
                    f"timeout_seconds exceeds {MAX_TIMEOUT_SECONDS}",
                    details={
                        "timeout_seconds": timeout_seconds,
                        "maximum": MAX_TIMEOUT_SECONDS,
                    },
                )

        workspace = self.workspace
        if workspace is not None:
            if not isinstance(workspace, str) or not workspace.strip():
                raise ContractValidationError(
                    "workspace must be a non-empty string"
                )
            workspace = workspace.strip()

        capabilities = self.capabilities
        if not isinstance(capabilities, CLICapabilities):
            if isinstance(capabilities, Mapping):
                capabilities = CLICapabilities.from_dict(capabilities)
            else:
                raise ContractValidationError(
                    "capabilities must be a CLICapabilities"
                )
        if mode is ExecutionMode.CHAT and capabilities.side_effecting:
            raise InvalidStateError(
                "chat request capabilities cannot be side-effecting",
                details={"mode": mode.value},
            )

        object.__setattr__(self, "side_effecting", side_effecting)
        object.__setattr__(self, "cacheable", cacheable)
        object.__setattr__(self, "retryable", retryable)
        object.__setattr__(self, "streaming", streaming)
        object.__setattr__(self, "cancellation_requested", cancellation_requested)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "provider_name", provider_name)
        object.__setattr__(self, "provider_override", provider_override)
        object.__setattr__(self, "model_override", model_override)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "metadata", _bounded_metadata(self.metadata))
        object.__setattr__(self, "capabilities", capabilities)
        ensure_serialized_bound(self.to_dict())

    def effective_model(self) -> Optional[str]:
        """Return the model override when set, otherwise the request model."""
        return self.model_override or self.model_name

    def effective_provider(self) -> Optional[str]:
        """Return the provider override when set, otherwise the request provider."""
        return self.provider_override or self.provider_name

    def to_dict(self) -> dict[str, Any]:
        """Operator-safe diagnostic view (prompt omitted)."""
        return {
            "contract_version": CONTRACT_VERSION,
            "contract_schema": CONTRACT_SCHEMA,
            "mode": self.mode.value,
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "provider_override": self.provider_override,
            "model_override": self.model_override,
            "side_effecting": self.side_effecting,
            "cacheable": self.cacheable,
            "retryable": self.retryable,
            "streaming": self.streaming,
            "session_id": self.session_id,
            "tools": list(self.tools),
            "cancellation_requested": self.cancellation_requested,
            "timeout_seconds": self.timeout_seconds,
            "workspace": self.workspace,
            "metadata": dict(self.metadata),
            "prompt_chars": len(self.prompt),
        }

    def to_execution_dict(self) -> dict[str, Any]:
        """Full request including prompt for trusted in-process handoff."""
        payload = self.to_dict()
        payload["prompt"] = self.prompt
        ensure_serialized_bound(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CLIRequest:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("request payload must be a mapping")
        tools = payload.get("tools") or ()
        if isinstance(tools, list):
            tools = tuple(tools)
        caps_raw = payload.get("capabilities")
        if caps_raw is None:
            mode = _normalize_execution_mode(
                payload.get("mode", ExecutionMode.CHAT)
            )
            capabilities = (
                CLICapabilities.agent_defaults()
                if mode is ExecutionMode.AGENT
                else CLICapabilities.chat_defaults()
            )
        else:
            capabilities = CLICapabilities.from_dict(caps_raw)
            mode = _normalize_execution_mode(
                payload.get("mode", ExecutionMode.CHAT)
            )
        return cls(
            prompt=str(payload.get("prompt", "")),
            mode=mode,
            model_name=payload.get("model_name"),
            provider_name=payload.get("provider_name"),
            provider_override=payload.get("provider_override"),
            model_override=payload.get("model_override"),
            side_effecting=bool(
                payload.get("side_effecting", capabilities.side_effecting)
            ),
            cacheable=bool(payload.get("cacheable", capabilities.cacheable)),
            retryable=bool(payload.get("retryable", capabilities.retryable)),
            streaming=bool(payload.get("streaming", False)),
            session_id=payload.get("session_id"),
            tools=tools,
            cancellation_requested=bool(
                payload.get("cancellation_requested", False)
            ),
            timeout_seconds=payload.get("timeout_seconds"),
            workspace=payload.get("workspace"),
            metadata=payload.get("metadata") or {},
            capabilities=capabilities,
        )


@dataclass(frozen=True)
class CLIEvent:
    """Bounded progress or diagnostic event for streaming consumers."""

    kind: EventKind
    sequence: int = 0
    message: str = ""
    payload: Mapping[str, str] = field(default_factory=dict)
    side_effecting: bool = False

    def __post_init__(self) -> None:
        kind = self.kind
        if isinstance(kind, str):
            try:
                kind = EventKind(kind)
            except ValueError as exc:
                raise ContractValidationError(
                    f"unknown event kind: {self.kind!r}"
                ) from exc
        elif not isinstance(kind, EventKind):
            raise ContractValidationError("event kind must be EventKind or str")
        object.__setattr__(self, "kind", kind)

        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise ContractValidationError("event sequence must be an integer")
        if self.sequence < 0:
            raise ContractValidationError("event sequence must be non-negative")

        message = self.message if self.message is not None else ""
        if not isinstance(message, str):
            raise ContractValidationError("event message must be a string")
        if len(message) > MAX_EVENT_PAYLOAD_CHARS:
            raise BoundsExceededError(
                f"event message exceeds {MAX_EVENT_PAYLOAD_CHARS} characters",
                details={
                    "length": len(message),
                    "maximum": MAX_EVENT_PAYLOAD_CHARS,
                },
            )
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "payload", _bounded_metadata(self.payload))
        side_effecting = _require_bool(self.side_effecting, "side_effecting")
        if kind in (EventKind.TOOL_CALL, EventKind.TOOL_RESULT, EventKind.SIDE_EFFECT):
            side_effecting = True
        object.__setattr__(self, "side_effecting", side_effecting)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "sequence": self.sequence,
            "message": self.message,
            "payload": dict(self.payload),
            "side_effecting": self.side_effecting,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CLIEvent:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("event payload must be a mapping")
        return cls(
            kind=payload.get("kind", EventKind.DIAGNOSTIC),
            sequence=int(payload.get("sequence", 0)),
            message=str(payload.get("message", "")),
            payload=payload.get("payload") or {},
            side_effecting=bool(payload.get("side_effecting", False)),
        )


@dataclass(frozen=True)
class CLIResult:
    """Rich result record that also exposes a string surface via ``text``.

    Consumers that only need the legacy ``LLMProvider.generate`` contract can
    use :attr:`text`. Richer consumers inspect events, error records, and flags.
    """

    text: str
    ok: bool = True
    mode: ExecutionMode = ExecutionMode.CHAT
    provider_name: Optional[str] = None
    model_name: Optional[str] = None
    side_effecting: bool = False
    cacheable: bool = True
    retryable: bool = True
    streaming: bool = False
    truncated: bool = False
    cancelled: bool = False
    exit_code: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    events: tuple[CLIEvent, ...] = ()
    error: Optional[CLIErrorRecord] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    had_side_effect_event: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ContractValidationError("result text must be a string")
        if len(self.text) > MAX_TEXT_CHARS:
            raise BoundsExceededError(
                f"result text exceeds {MAX_TEXT_CHARS} characters",
                details={"length": len(self.text), "maximum": MAX_TEXT_CHARS},
            )

        mode = _normalize_execution_mode(self.mode)
        object.__setattr__(self, "mode", mode)

        ok = _require_bool(self.ok, "ok")
        side_effecting = _require_bool(self.side_effecting, "side_effecting")
        cacheable = _require_bool(self.cacheable, "cacheable")
        retryable = _require_bool(self.retryable, "retryable")
        streaming = _require_bool(self.streaming, "streaming")
        truncated = _require_bool(self.truncated, "truncated")
        cancelled = _require_bool(self.cancelled, "cancelled")
        had_side_effect = _require_bool(
            self.had_side_effect_event, "had_side_effect_event"
        )

        provider_name = self.provider_name
        if provider_name is not None:
            provider_name = _normalize_identifier(
                provider_name, "provider_name", maximum=MAX_PROVIDER_CHARS
            )
        model_name = self.model_name
        if model_name is not None:
            model_name = _require_non_empty_name(
                model_name, "model_name", maximum=MAX_MODEL_CHARS
            )

        exit_code = self.exit_code
        if exit_code is not None:
            if not isinstance(exit_code, int) or isinstance(exit_code, bool):
                raise ContractValidationError(
                    "exit_code must be an integer or None"
                )

        elapsed_seconds = self.elapsed_seconds
        if elapsed_seconds is not None:
            if not isinstance(elapsed_seconds, (int, float)) or isinstance(
                elapsed_seconds, bool
            ):
                raise ContractValidationError(
                    "elapsed_seconds must be a number or None"
                )
            elapsed_seconds = float(elapsed_seconds)
            if elapsed_seconds < 0:
                raise ContractValidationError(
                    "elapsed_seconds must be non-negative"
                )

        events = self.events
        if events is None:
            events = ()
        if not isinstance(events, (list, tuple)):
            raise ContractValidationError("events must be a sequence of CLIEvent")
        normalized_events: list[CLIEvent] = []
        for item in events:
            if isinstance(item, CLIEvent):
                normalized_events.append(item)
            elif isinstance(item, Mapping):
                normalized_events.append(CLIEvent.from_dict(item))
            else:
                raise ContractValidationError(
                    "events must contain CLIEvent records"
                )
        if len(normalized_events) > MAX_EVENT_COUNT:
            raise BoundsExceededError(
                f"events exceeds {MAX_EVENT_COUNT} entries",
                details={"maximum": MAX_EVENT_COUNT},
            )
        if any(event.side_effecting for event in normalized_events):
            had_side_effect = True
            side_effecting = True

        if mode is ExecutionMode.AGENT or side_effecting or had_side_effect:
            side_effecting = True
            cacheable = False
            retryable = False

        if side_effecting and cacheable:
            raise InvalidStateError(
                "side-effecting results cannot be cacheable",
                details={"side_effecting": True, "cacheable": True},
            )

        error = self.error
        if error is not None:
            if isinstance(error, Mapping):
                error = CLIErrorRecord.from_dict(error)
            elif not isinstance(error, CLIErrorRecord):
                raise ContractValidationError(
                    "error must be a CLIErrorRecord or mapping"
                )
        elif not ok:
            error = CLIErrorRecord(
                code=CLIRuntimeErrorCode.INTERNAL,
                message="CLI result marked not ok without structured error",
            )

        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "side_effecting", side_effecting)
        object.__setattr__(self, "cacheable", cacheable)
        object.__setattr__(self, "retryable", retryable)
        object.__setattr__(self, "streaming", streaming)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "cancelled", cancelled)
        object.__setattr__(self, "had_side_effect_event", had_side_effect)
        object.__setattr__(self, "provider_name", provider_name)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "exit_code", exit_code)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        object.__setattr__(self, "events", tuple(normalized_events))
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "metadata", _bounded_metadata(self.metadata))
        ensure_serialized_bound(self.to_dict())

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "contract_schema": CONTRACT_SCHEMA,
            "text": self.text,
            "ok": self.ok,
            "mode": self.mode.value,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "side_effecting": self.side_effecting,
            "cacheable": self.cacheable,
            "retryable": self.retryable,
            "streaming": self.streaming,
            "truncated": self.truncated,
            "cancelled": self.cancelled,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "events": [event.to_dict() for event in self.events],
            "error": None if self.error is None else self.error.to_dict(),
            "metadata": dict(self.metadata),
            "had_side_effect_event": self.had_side_effect_event,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CLIResult:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("result payload must be a mapping")
        events = payload.get("events") or ()
        if isinstance(events, list):
            events = tuple(events)
        error = payload.get("error")
        if error is not None and isinstance(error, Mapping):
            error = CLIErrorRecord.from_dict(error)
        return cls(
            text=str(payload.get("text", "")),
            ok=bool(payload.get("ok", True)),
            mode=payload.get("mode", ExecutionMode.CHAT),
            provider_name=payload.get("provider_name"),
            model_name=payload.get("model_name"),
            side_effecting=bool(payload.get("side_effecting", False)),
            cacheable=bool(payload.get("cacheable", True)),
            retryable=bool(payload.get("retryable", True)),
            streaming=bool(payload.get("streaming", False)),
            truncated=bool(payload.get("truncated", False)),
            cancelled=bool(payload.get("cancelled", False)),
            exit_code=payload.get("exit_code"),
            elapsed_seconds=payload.get("elapsed_seconds"),
            events=events,
            error=error,
            metadata=payload.get("metadata") or {},
            had_side_effect_event=bool(
                payload.get("had_side_effect_event", False)
            ),
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        mode: ExecutionMode = ExecutionMode.CHAT,
    ) -> CLIResult:
        return cls(
            text=text,
            ok=True,
            mode=mode,
            provider_name=provider_name,
            model_name=model_name,
        )


@dataclass(frozen=True)
class ProviderSpec:
    """Side-effect-free provider metadata stored by the lazy registry.

    Factories are *not* part of this record so listing and serialization never
    invoke provider code, install tools, or start processes.
    """

    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    capabilities: CLICapabilities = field(
        default_factory=CLICapabilities.chat_defaults
    )
    streaming: CapabilitySupport = CapabilitySupport.UNKNOWN
    tools: CapabilitySupport = CapabilitySupport.NOT_SUPPORTED
    sessions: CapabilitySupport = CapabilitySupport.NOT_SUPPORTED
    cancellation: CapabilitySupport = CapabilitySupport.SUPPORTED
    provider_override: CapabilitySupport = CapabilitySupport.SUPPORTED
    model_override: CapabilitySupport = CapabilitySupport.SUPPORTED
    locality: str = "unknown"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _normalize_identifier(
            self.name, "name", maximum=MAX_PROVIDER_CHARS
        )
        object.__setattr__(self, "name", name)

        aliases_raw = self.aliases
        if isinstance(aliases_raw, (str, bytes)):
            raise ContractValidationError(
                "aliases must be a sequence, not a string"
            )
        aliases: list[str] = []
        seen: set[str] = {name}
        for item in aliases_raw or ():
            alias = _normalize_identifier(
                item, "alias", maximum=MAX_PROVIDER_CHARS
            )
            if alias in seen:
                continue
            seen.add(alias)
            aliases.append(alias)
            if len(aliases) > MAX_ALIAS_COUNT:
                raise BoundsExceededError(
                    f"aliases exceeds {MAX_ALIAS_COUNT} entries",
                    details={"maximum": MAX_ALIAS_COUNT},
                )
        object.__setattr__(self, "aliases", tuple(sorted(aliases)))

        description = self.description or ""
        if not isinstance(description, str):
            raise ContractValidationError("description must be a string")
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise BoundsExceededError(
                f"description exceeds {MAX_DESCRIPTION_CHARS} characters",
                details={
                    "length": len(description),
                    "maximum": MAX_DESCRIPTION_CHARS,
                },
            )
        object.__setattr__(self, "description", description)

        capabilities = self.capabilities
        if isinstance(capabilities, Mapping):
            capabilities = CLICapabilities.from_dict(capabilities)
        if not isinstance(capabilities, CLICapabilities):
            raise ContractValidationError("capabilities must be a CLICapabilities")
        object.__setattr__(self, "capabilities", capabilities)

        def _support(
            value: Any, default: CapabilitySupport
        ) -> CapabilitySupport:
            if value is None:
                return default
            if isinstance(value, CapabilitySupport):
                return value
            try:
                return CapabilitySupport(str(value).strip().lower())
            except ValueError as exc:
                raise ContractValidationError(
                    f"unknown capability support: {value!r}"
                ) from exc

        object.__setattr__(
            self, "streaming", _support(self.streaming, CapabilitySupport.UNKNOWN)
        )
        object.__setattr__(
            self, "tools", _support(self.tools, CapabilitySupport.NOT_SUPPORTED)
        )
        object.__setattr__(
            self,
            "sessions",
            _support(self.sessions, CapabilitySupport.NOT_SUPPORTED),
        )
        object.__setattr__(
            self,
            "cancellation",
            _support(self.cancellation, CapabilitySupport.SUPPORTED),
        )
        object.__setattr__(
            self,
            "provider_override",
            _support(self.provider_override, CapabilitySupport.SUPPORTED),
        )
        object.__setattr__(
            self,
            "model_override",
            _support(self.model_override, CapabilitySupport.SUPPORTED),
        )

        locality = self.locality or "unknown"
        if not isinstance(locality, str) or not locality.strip():
            raise ContractValidationError("locality must be a non-empty string")
        object.__setattr__(self, "locality", locality.strip().lower())
        object.__setattr__(self, "metadata", _bounded_metadata(self.metadata))
        ensure_serialized_bound(self.to_dict())

    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "contract_schema": CONTRACT_SCHEMA,
            "name": self.name,
            "aliases": list(self.aliases),
            "description": self.description,
            "capabilities": self.capabilities.to_dict(),
            "streaming": self.streaming.value,
            "tools": self.tools.value,
            "sessions": self.sessions.value,
            "cancellation": self.cancellation.value,
            "provider_override": self.provider_override.value,
            "model_override": self.model_override.value,
            "locality": self.locality,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProviderSpec:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("provider spec payload must be a mapping")
        aliases = payload.get("aliases") or ()
        if isinstance(aliases, list):
            aliases = tuple(aliases)
        caps = payload.get("capabilities")
        capabilities = (
            CLICapabilities.from_dict(caps)
            if caps is not None
            else CLICapabilities.chat_defaults()
        )
        return cls(
            name=str(payload.get("name", "")),
            aliases=aliases,
            description=str(payload.get("description", "")),
            capabilities=capabilities,
            streaming=payload.get("streaming", CapabilitySupport.UNKNOWN),
            tools=payload.get("tools", CapabilitySupport.NOT_SUPPORTED),
            sessions=payload.get("sessions", CapabilitySupport.NOT_SUPPORTED),
            cancellation=payload.get(
                "cancellation", CapabilitySupport.SUPPORTED
            ),
            provider_override=payload.get(
                "provider_override", CapabilitySupport.SUPPORTED
            ),
            model_override=payload.get(
                "model_override", CapabilitySupport.SUPPORTED
            ),
            locality=str(payload.get("locality", "unknown")),
            metadata=payload.get("metadata") or {},
        )


@runtime_checkable
class LLMProvider(Protocol):
    """Legacy string-returning provider surface preserved for compatibility."""

    def generate(
        self, prompt: str, *, model_name: Optional[str] = None, **kwargs: object
    ) -> str: ...


@runtime_checkable
class RichCLIProvider(Protocol):
    """Provider that returns a structured :class:`CLIResult`."""

    def generate_result(self, request: CLIRequest) -> CLIResult: ...


@runtime_checkable
class StreamingCLIProvider(Protocol):
    """Provider that streams :class:`CLIEvent` values."""

    def stream_events(self, request: CLIRequest) -> Iterator[CLIEvent]: ...


@runtime_checkable
class EventConsumer(Protocol):
    """Callback consumer for streamed CLI events."""

    def on_event(self, event: CLIEvent) -> None: ...


ProviderFactory = Callable[[], LLMProvider]


def result_text(result: CLIResult | str) -> str:
    """Extract the legacy string surface from a result or plain string."""
    if isinstance(result, CLIResult):
        return result.text
    return str(result)


def adapt_string_provider(provider: LLMProvider) -> RichCLIProvider:
    """Adapt a string-returning :class:`LLMProvider` into a rich provider."""

    class _Adapter:
        def __init__(self, inner: LLMProvider) -> None:
            self._inner = inner

        def generate_result(self, request: CLIRequest) -> CLIResult:
            kwargs: dict[str, Any] = {}
            if request.side_effecting:
                kwargs["side_effecting"] = True
                kwargs["agent"] = request.mode is ExecutionMode.AGENT
            if request.workspace is not None:
                kwargs["workspace"] = request.workspace
            if request.timeout_seconds is not None:
                kwargs["timeout"] = request.timeout_seconds
            if request.session_id is not None:
                kwargs["session_id"] = request.session_id
            if request.tools:
                kwargs["tools"] = list(request.tools)
            if request.streaming:
                kwargs["stream"] = True
            text = self._inner.generate(
                request.prompt,
                model_name=request.effective_model(),
                **kwargs,
            )
            return CLIResult.from_text(
                text if isinstance(text, str) else str(text),
                provider_name=request.effective_provider(),
                model_name=request.effective_model(),
                mode=request.mode,
            )

    return _Adapter(provider)


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_SCHEMA",
    "MAX_NAME_CHARS",
    "MAX_ALIAS_COUNT",
    "MAX_PROMPT_CHARS",
    "MAX_TEXT_CHARS",
    "MAX_MODEL_CHARS",
    "MAX_PROVIDER_CHARS",
    "MAX_SESSION_ID_CHARS",
    "MAX_TOOL_NAME_CHARS",
    "MAX_TOOL_COUNT",
    "MAX_METADATA_KEYS",
    "MAX_METADATA_KEY_CHARS",
    "MAX_METADATA_VALUE_CHARS",
    "MAX_METADATA_BYTES",
    "MAX_EVENT_COUNT",
    "MAX_EVENT_PAYLOAD_CHARS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_SERIALIZED_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "MIN_TIMEOUT_SECONDS",
    "MAX_ARGV_ITEMS",
    "MAX_ARGV_ITEM_CHARS",
    "ExecutionMode",
    "EventKind",
    "CapabilitySupport",
    "CLICapabilities",
    "CLIRequest",
    "CLIEvent",
    "CLIResult",
    "ProviderSpec",
    "LLMProvider",
    "RichCLIProvider",
    "StreamingCLIProvider",
    "EventConsumer",
    "ProviderFactory",
    "canonical_json",
    "canonical_json_bytes",
    "ensure_serialized_bound",
    "result_text",
    "adapt_string_provider",
]
