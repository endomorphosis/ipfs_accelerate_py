"""Provider-neutral CLI endpoint factory and registry.

This module is the canonical registration path for CLI endpoint adapters.
Concrete adapters are created by tool-type factories; the abstract base class
is never instantiated.

Safety invariants:

- Importing, listing tool types, and listing registered endpoints never starts
  processes and never probes every provider for availability.
- Registration is lazy: only the requested tool type is constructed.
- Registry collisions fail closed unless ``replace=True``.
- Nonzero subprocess exit status is treated as failure.
- Request/response payload sizes are bounded.
- Error envelopes never echo prompts or credentials.
- Endpoint statistics counters are concurrency safe.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .contracts import (
    MAX_PROMPT_CHARS,
    MAX_TEXT_CHARS,
    _clip_text,
    _normalize_identifier,
)
from .errors import (
    BoundsExceededError,
    CLIErrorRecord,
    CLIRuntimeError,
    CLIRuntimeErrorCode,
    ContractValidationError,
    NonzeroExitError,
    ProviderNotFoundError,
    RegistryCollisionError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds (aligned with shared CLI runtime contracts)
# ---------------------------------------------------------------------------

MAX_ENDPOINT_ID_CHARS: int = 100
MAX_RESULT_CHARS: int = MAX_TEXT_CHARS
MAX_STDERR_DIAGNOSTIC_CHARS: int = 1024
DEFAULT_EXECUTE_TIMEOUT_SECONDS: int = 30

_ENDPOINT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
_SENSITIVE_KEYS: tuple[str, ...] = (
    "prompt",
    "stdin",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)

# ---------------------------------------------------------------------------
# Health / lifecycle
# ---------------------------------------------------------------------------


class EndpointHealth(str, Enum):
    """Distinct health states for a registered CLI endpoint."""

    UNKNOWN = "unknown"
    MISSING = "missing"  # tool path not found / not installed
    INSTALLED = "installed"  # binary present, config not validated
    CONFIGURED = "configured"  # config present, not yet ready
    READY = "ready"  # installed + configured + available
    DEGRADED = "degraded"  # registered but failing or partially available
    UNSUPPORTED_VERSION = "unsupported_version"  # binary present, unsafe/old


class EndpointLifecycleOp(str, Enum):
    """Supported lifecycle dispatch operations."""

    LIST = "list"
    DESCRIBE = "describe"
    LIVENESS = "liveness"
    READINESS = "readiness"
    EXECUTE = "execute"
    STREAM = "stream"
    CANCEL = "cancel"
    # Persistent Goose ACP session lifecycle (endpoint-local).
    ACP_START = "acp_start"
    ACP_STOP = "acp_stop"
    ACP_RESTART = "acp_restart"
    SESSION_NEW = "session_new"
    SESSION_LOAD = "session_load"
    SESSION_CLOSE = "session_close"
    SESSION_PROMPT = "session_prompt"
    SESSION_CANCEL = "session_cancel"
    ACP_DESCRIBE = "acp_describe"


# ---------------------------------------------------------------------------
# Protocols and tool specifications
# ---------------------------------------------------------------------------


@runtime_checkable
class EndpointAdapterProtocol(Protocol):
    """Minimal surface required of a concrete CLI endpoint adapter."""

    endpoint_id: str
    cli_path: Optional[str]
    config: Mapping[str, Any]
    stats: MutableMapping[str, Any]

    def is_available(self) -> bool: ...

    def execute(
        self,
        prompt: str,
        task_type: str = "text_generation",
        timeout: int = 30,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...

    def get_stats(self) -> Mapping[str, Any]: ...

    def validate_config(self) -> Mapping[str, Any]: ...


AdapterFactory = Callable[..., EndpointAdapterProtocol]


@dataclass(frozen=True)
class EndpointToolSpec:
    """Metadata for a supported CLI tool type (no process activity)."""

    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    adapter_class_name: str = ""
    supported_tasks: tuple[str, ...] = ("text_generation",)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = _normalize_identifier(self.name, "name")
        object.__setattr__(self, "name", canonical)
        cleaned: list[str] = []
        seen = {canonical}
        for alias in self.aliases:
            key = _normalize_identifier(alias, "alias")
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        object.__setattr__(self, "aliases", tuple(sorted(cleaned)))
        if not isinstance(self.description, str):
            raise ContractValidationError("description must be a string")
        object.__setattr__(
            self, "description", _clip_text(self.description, 1024)
        )

    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "description": self.description,
            "adapter_class_name": self.adapter_class_name,
            "supported_tasks": list(self.supported_tasks),
            "metadata": dict(self.metadata),
        }


@dataclass
class EndpointStats:
    """Concurrency-safe request counters for one endpoint."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self, elapsed: float) -> None:
        with self._lock:
            self.requests += 1
            self.successes += 1
            self.total_time += max(0.0, float(elapsed))

    def record_failure(self, elapsed: float = 0.0) -> None:
        with self._lock:
            self.requests += 1
            self.failures += 1
            self.total_time += max(0.0, float(elapsed))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = self.requests
            total_time = self.total_time
            avg = (total_time / requests) if requests else 0.0
            return {
                "requests": requests,
                "successes": self.successes,
                "failures": self.failures,
                "total_time": total_time,
                "avg_time": avg,
            }


@dataclass
class EndpointRecord:
    """Registered endpoint: concrete adapter plus lifecycle metadata.

    ACP client and sessions are *endpoint-local*: each endpoint owns its own
    managed ``goose acp`` process and session map. Restart behavior is
    explicit and never auto-replays agent work.
    """

    endpoint_id: str
    tool: str
    adapter: EndpointAdapterProtocol
    health: EndpointHealth = EndpointHealth.UNKNOWN
    stats: EndpointStats = field(default_factory=EndpointStats)
    active_request_id: Optional[str] = None
    cancel_requested: bool = False
    registered_at: float = field(default_factory=time.time)
    # Optional persistent ACP client (Goose). Typed as Any to avoid import
    # cycles; constructed lazily via CLIEndpointRegistry.acp_start.
    acp_client: Any = field(default=None, repr=False)

    def describe(self, *, probe: bool = False) -> dict[str, Any]:
        """Describe this endpoint. Probing is opt-in and never done for list."""
        available: Optional[bool] = None
        if probe:
            try:
                available = bool(self.adapter.is_available())
            except Exception:  # noqa: BLE001 - describe must not raise
                available = False
                self.health = EndpointHealth.DEGRADED
            else:
                self.health = (
                    EndpointHealth.READY
                    if available
                    else EndpointHealth.MISSING
                )
        cli_path = getattr(self.adapter, "cli_path", None)
        acp_info: Optional[dict[str, Any]] = None
        client = self.acp_client
        if client is not None:
            try:
                acp_info = {
                    "state": getattr(
                        getattr(client, "state", None), "value", None
                    )
                    or str(getattr(client, "state", None)),
                    "ready": bool(getattr(client, "is_ready", False)),
                    "sessions": len(client.list_sessions())
                    if hasattr(client, "list_sessions")
                    else 0,
                    "restart_count": getattr(client, "restart_count", 0),
                }
            except Exception:  # noqa: BLE001
                acp_info = {"state": "unknown"}
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
            "tool": self.tool,
            "cli_path": cli_path,
            "available": available,
            "health": self.health.value,
            "stats": self.stats.snapshot(),
            "registered_at": self.registered_at,
            "acp": acp_info,
        }


# ---------------------------------------------------------------------------
# Error helpers (never echo prompts)
# ---------------------------------------------------------------------------


class UnsupportedEndpointToolError(CLIRuntimeError):
    """Raised when a registration tool name is not in the factory map."""

    def __init__(
        self,
        tool: str,
        *,
        supported: Sequence[str] | None = None,
    ) -> None:
        details: dict[str, Any] = {"tool": str(tool)}
        if supported is not None:
            details["supported"] = ",".join(supported)
        super().__init__(
            f"Unsupported CLI endpoint tool: {tool}",
            code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
            retryable=False,
            details=details,
        )


class EndpointNotFoundError(CLIRuntimeError):
    """Raised when an endpoint_id is not registered."""

    def __init__(self, endpoint_id: str) -> None:
        super().__init__(
            f"CLI endpoint not found: {endpoint_id}",
            code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
            retryable=False,
            details={"endpoint_id": str(endpoint_id)},
        )


class EndpointUnavailableError(CLIRuntimeError):
    """Raised when a registered endpoint's CLI tool is not available."""

    def __init__(self, endpoint_id: str, *, tool: str | None = None) -> None:
        details: dict[str, Any] = {"endpoint_id": str(endpoint_id)}
        if tool:
            details["tool"] = tool
        super().__init__(
            f"CLI endpoint unavailable: {endpoint_id}",
            code=CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED,
            retryable=False,
            details=details,
        )


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _SENSITIVE_KEYS)


def sanitize_error_payload(
    payload: Mapping[str, Any] | None,
    *,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Return a copy of *payload* with prompts/secrets stripped."""
    if not payload:
        return {}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_sensitive_key(str(key)):
            out[str(key)] = "[redacted]"
            continue
        if isinstance(value, str) and prompt and prompt in value and prompt:
            # Never reflect the full prompt in diagnostics.
            out[str(key)] = value.replace(prompt, "[redacted]")
        else:
            out[str(key)] = value
    return out


def error_envelope(
    message: str,
    *,
    code: CLIRuntimeErrorCode | str = CLIRuntimeErrorCode.INTERNAL,
    endpoint_id: str | None = None,
    tool: str | None = None,
    returncode: int | None = None,
    details: Mapping[str, Any] | None = None,
    retryable: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a typed, prompt-safe error envelope."""
    safe_details = sanitize_error_payload(details or {})
    if endpoint_id is not None:
        safe_details.setdefault("endpoint_id", str(endpoint_id))
    if tool is not None:
        safe_details.setdefault("tool", str(tool))
    if returncode is not None:
        safe_details.setdefault("returncode", str(returncode))
    # Drop any accidental prompt keys from caller extras.
    safe_extra = sanitize_error_payload(extra)
    record = CLIErrorRecord(
        code=code,
        message=message,
        retryable=retryable,
        details=safe_details,
    )
    envelope: dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": record.message,
        "error_code": record.code.value,
        "retryable": record.retryable,
        "details": dict(record.details),
    }
    if endpoint_id is not None:
        envelope["endpoint_id"] = str(endpoint_id)
    if tool is not None:
        envelope["tool"] = str(tool)
    if returncode is not None:
        envelope["returncode"] = int(returncode)
    envelope.update(safe_extra)
    # Hard guarantee: never ship a prompt field.
    envelope.pop("prompt", None)
    return envelope


def bound_prompt(prompt: Any, *, maximum: int = MAX_PROMPT_CHARS) -> str:
    """Validate and return a prompt within the hard character bound."""
    if not isinstance(prompt, str):
        raise ContractValidationError(
            "prompt must be a string",
            details={"type": type(prompt).__name__},
        )
    if "\x00" in prompt:
        raise ContractValidationError("prompt must not contain null bytes")
    if len(prompt) > maximum:
        raise BoundsExceededError(
            f"prompt exceeds {maximum} characters",
            details={"length": len(prompt), "maximum": maximum},
        )
    return prompt


def bound_result_text(text: Any, *, maximum: int = MAX_RESULT_CHARS) -> str:
    """Clip result text to the response bound (never raise after success)."""
    if text is None:
        return ""
    return _clip_text(text, maximum)


def validate_endpoint_id(endpoint_id: Any) -> str:
    if not isinstance(endpoint_id, str):
        raise ContractValidationError(
            "endpoint_id must be a string",
            details={"type": type(endpoint_id).__name__},
        )
    text = endpoint_id.strip()
    if not text:
        raise ContractValidationError("endpoint_id must not be empty")
    if len(text) > MAX_ENDPOINT_ID_CHARS:
        raise BoundsExceededError(
            f"endpoint_id exceeds {MAX_ENDPOINT_ID_CHARS} characters",
            details={"length": len(text), "maximum": MAX_ENDPOINT_ID_CHARS},
        )
    if not _ENDPOINT_ID_PATTERN.match(text):
        raise ContractValidationError(
            "endpoint_id must match [a-zA-Z0-9_-]+",
            details={"endpoint_id": text[:64]},
        )
    return text


# ---------------------------------------------------------------------------
# Concrete tool factory map (lazy adapter class import)
# ---------------------------------------------------------------------------


def _load_adapter_class(class_name: str) -> type:
    """Import a concrete adapter class from the compatibility module."""
    # Local import avoids circular dependency at package import time.
    from ipfs_accelerate_py.mcp.tools import cli_endpoint_adapters as adapters

    cls = getattr(adapters, class_name, None)
    if cls is None:
        raise ProviderNotFoundError(
            class_name,
            details={"reason": "adapter_class_missing"},
        )
    # Refuse abstract base and incomplete subclasses.
    import inspect

    if inspect.isabstract(cls) or class_name == "CLIEndpointAdapter":
        raise ContractValidationError(
            f"refusing to instantiate abstract adapter {class_name}",
            details={"adapter_class": class_name},
        )
    return cls


def _default_tool_specs() -> tuple[EndpointToolSpec, ...]:
    return (
        EndpointToolSpec(
            name="claude",
            aliases=("claude_cli", "claude_code", "anthropic"),
            description="Anthropic Claude Code CLI",
            adapter_class_name="ClaudeCodeAdapter",
            supported_tasks=(
                "text_generation",
                "code_generation",
                "analysis",
            ),
        ),
        EndpointToolSpec(
            name="openai",
            aliases=("openai_cli", "codex", "chatgpt"),
            description="OpenAI Codex / ChatGPT CLI",
            adapter_class_name="OpenAICodexAdapter",
            supported_tasks=(
                "text_generation",
                "code_generation",
                "embedding",
            ),
        ),
        EndpointToolSpec(
            name="gemini",
            aliases=("gemini_cli", "gcloud", "google_gemini"),
            description="Google Gemini CLI (via gcloud)",
            adapter_class_name="GeminiCLIAdapter",
            supported_tasks=("text_generation", "code_generation"),
        ),
        EndpointToolSpec(
            name="vscode",
            aliases=("vscode_cli", "copilot", "code", "github_copilot"),
            description="VS Code CLI with GitHub Copilot",
            adapter_class_name="VSCodeCLIAdapter",
            supported_tasks=(
                "code_generation",
                "code_completion",
                "code_explanation",
                "text_generation",
            ),
        ),
        EndpointToolSpec(
            name="goose",
            aliases=(
                "goose_cli",
                "block_goose",
                "aaif_goose",
                "goose-cli",
            ),
            description=(
                "Block/AAIF Goose CLI — chat-safe by default; agent mode "
                "requires explicit side-effect policy"
            ),
            adapter_class_name="GooseCLIAdapter",
            supported_tasks=(
                "text_generation",
                "code_generation",
                "analysis",
            ),
            metadata={
                "provider": "goose_cli",
                "default_execution_mode": "chat",
                "agent_requires_policy": "true",
            },
        ),
    )


class CLIEndpointFactory:
    """Maps tool names to concrete adapter constructors without probing CLIs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._specs: dict[str, EndpointToolSpec] = {}
        self._index: dict[str, str] = {}  # name/alias -> canonical
        self._factories: dict[str, AdapterFactory] = {}
        for spec in _default_tool_specs():
            self.register_tool_type(spec, factory=None, replace=True)

    def register_tool_type(
        self,
        spec: EndpointToolSpec | Mapping[str, Any],
        *,
        factory: AdapterFactory | None = None,
        replace: bool = False,
    ) -> EndpointToolSpec:
        """Register metadata (+ optional factory) for a tool type. No probing."""
        if isinstance(spec, Mapping):
            tool_spec = EndpointToolSpec(
                name=str(spec.get("name", "")),
                aliases=tuple(spec.get("aliases") or ()),
                description=str(spec.get("description") or ""),
                adapter_class_name=str(spec.get("adapter_class_name") or ""),
                supported_tasks=tuple(
                    spec.get("supported_tasks") or ("text_generation",)
                ),
                metadata=dict(spec.get("metadata") or {}),
            )
        elif isinstance(spec, EndpointToolSpec):
            tool_spec = spec
        else:
            raise ContractValidationError(
                "spec must be EndpointToolSpec or mapping"
            )

        if factory is not None and not callable(factory):
            raise TypeError("adapter factory must be callable or None")

        canonical = tool_spec.name
        claimed = list(tool_spec.all_names())

        with self._lock:
            for name in claimed:
                existing = self._index.get(name)
                if existing is not None and existing != canonical:
                    raise RegistryCollisionError(
                        f"tool name/alias {name!r} collides with {existing!r}",
                        details={
                            "name": name,
                            "canonical": existing,
                            "requested": canonical,
                        },
                    )
            if not replace and canonical in self._specs:
                raise RegistryCollisionError(
                    f"tool type {canonical!r} is already registered",
                    details={"canonical": canonical},
                )
            prior = self._specs.get(canonical)
            if prior is not None:
                for old in prior.all_names():
                    if self._index.get(old) == canonical:
                        del self._index[old]
            self._specs[canonical] = tool_spec
            for name in claimed:
                self._index[name] = canonical
            if factory is not None:
                self._factories[canonical] = factory
            elif canonical not in self._factories:
                # Default lazy factory from adapter_class_name.
                class_name = tool_spec.adapter_class_name

                def _make(
                    endpoint_id: str,
                    cli_path: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    *,
                    _class_name: str = class_name,
                ) -> EndpointAdapterProtocol:
                    cls = _load_adapter_class(_class_name)
                    return cls(endpoint_id, cli_path, config or {})

                self._factories[canonical] = _make
            return tool_spec

    def resolve_tool(self, tool: str) -> str:
        key = _normalize_identifier(tool, "tool")
        with self._lock:
            canonical = self._index.get(key)
            if canonical is None:
                supported = tuple(sorted(self._specs))
                raise UnsupportedEndpointToolError(tool, supported=supported)
            return canonical

    def get_tool_spec(self, tool: str) -> EndpointToolSpec:
        canonical = self.resolve_tool(tool)
        with self._lock:
            return self._specs[canonical]

    def list_tool_specs(self) -> tuple[EndpointToolSpec, ...]:
        """List known tool types without invoking factories or probing CLIs."""
        with self._lock:
            return tuple(self._specs[name] for name in sorted(self._specs))

    def list_tool_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._specs))

    def create(
        self,
        tool: str,
        endpoint_id: str,
        *,
        cli_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EndpointAdapterProtocol:
        """Instantiate a *concrete* adapter for *tool*. Never uses the ABC."""
        endpoint_id = validate_endpoint_id(endpoint_id)
        canonical = self.resolve_tool(tool)
        with self._lock:
            factory = self._factories.get(canonical)
            spec = self._specs[canonical]
        if factory is None:
            raise UnsupportedEndpointToolError(
                tool, supported=self.list_tool_names()
            )
        adapter = factory(endpoint_id, cli_path, config or {})
        # Guard: refuse abstract instances if a bad factory was injected.
        import inspect

        adapter_cls = type(adapter)
        if inspect.isabstract(adapter_cls) or adapter_cls.__name__ == "CLIEndpointAdapter":
            raise ContractValidationError(
                "factory produced abstract CLIEndpointAdapter",
                details={
                    "tool": canonical,
                    "adapter_class": adapter_cls.__name__,
                },
            )
        # Stamp tool metadata for downstream describe/list.
        try:
            object.__setattr__(adapter, "tool_name", canonical)  # type: ignore[attr-defined]
        except Exception:
            try:
                adapter.tool_name = canonical  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        adapter.config = dict(getattr(adapter, "config", None) or {})  # type: ignore[attr-defined]
        if "tool" not in adapter.config:
            adapter.config["tool"] = canonical  # type: ignore[index]
        _ = spec  # retained for future capability checks
        return adapter


# ---------------------------------------------------------------------------
# Endpoint instance registry
# ---------------------------------------------------------------------------


class CLIEndpointRegistry:
    """Thread-safe registry of concrete CLI endpoint adapters."""

    def __init__(self, factory: CLIEndpointFactory | None = None) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, EndpointRecord] = {}
        self.factory = factory or CLIEndpointFactory()
        # Legacy-compatible mutable view of endpoint_id -> adapter.
        self.adapters: Dict[str, Any] = _RegistryAdapterView(self)

    def clear(self) -> None:
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
        for record in records:
            client = getattr(record, "acp_client", None)
            if client is not None:
                try:
                    client.stop()
                except Exception:  # noqa: BLE001
                    pass
                record.acp_client = None

    def register_adapter(
        self,
        adapter: EndpointAdapterProtocol,
        *,
        tool: str | None = None,
        replace: bool = False,
        probe: bool = False,
    ) -> dict[str, Any]:
        """Register an already-constructed concrete adapter instance."""
        if adapter is None:
            return error_envelope(
                "adapter is required",
                code=CLIRuntimeErrorCode.INVALID_CONTRACT,
            )
        import inspect

        adapter_cls = type(adapter)
        if inspect.isabstract(adapter_cls) or adapter_cls.__name__ == "CLIEndpointAdapter":
            return error_envelope(
                "cannot register abstract CLIEndpointAdapter; use the concrete factory",
                code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                details={"adapter_class": adapter_cls.__name__},
            )
        try:
            endpoint_id = validate_endpoint_id(adapter.endpoint_id)
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                details=exc.details,
            )

        tool_name = tool
        if not tool_name:
            tool_name = getattr(adapter, "tool_name", None) or (
                (adapter.config or {}).get("tool")  # type: ignore[union-attr]
                if isinstance(getattr(adapter, "config", None), Mapping)
                else None
            )
        if not tool_name:
            # Infer from class name when possible.
            class_name = adapter_cls.__name__
            for spec in self.factory.list_tool_specs():
                if spec.adapter_class_name == class_name:
                    tool_name = spec.name
                    break
        if not tool_name:
            tool_name = "custom"

        health = EndpointHealth.UNKNOWN
        if probe:
            try:
                available = bool(adapter.is_available())
            except Exception:  # noqa: BLE001
                available = False
            health = (
                EndpointHealth.READY if available else EndpointHealth.MISSING
            )
            available_flag = available
        else:
            # Lazy: do not probe on registration.
            available_flag = None
            health = EndpointHealth.INSTALLED

        record = EndpointRecord(
            endpoint_id=endpoint_id,
            tool=str(tool_name),
            adapter=adapter,
            health=health,
        )

        with self._lock:
            if endpoint_id in self._records and not replace:
                raise RegistryCollisionError(
                    f"endpoint {endpoint_id!r} is already registered",
                    details={"endpoint_id": endpoint_id},
                )
            self._records[endpoint_id] = record

        logger.info(
            "Registered CLI endpoint %s (tool=%s, probed=%s)",
            endpoint_id,
            tool_name,
            probe,
        )
        result: dict[str, Any] = {
            "status": "success",
            "success": True,
            "endpoint_id": endpoint_id,
            "tool": str(tool_name),
            "registered": True,
            "health": health.value,
            "message": f"CLI endpoint {endpoint_id} registered successfully",
        }
        if available_flag is not None:
            result["available"] = available_flag
        else:
            result["available"] = None  # not probed
        return result

    def register_tool(
        self,
        tool: str,
        *,
        endpoint_id: str | None = None,
        cli_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        replace: bool = False,
        probe: bool = False,
    ) -> dict[str, Any]:
        """Create a concrete adapter via the factory and register it."""
        try:
            canonical = self.factory.resolve_tool(tool)
        except UnsupportedEndpointToolError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                tool=str(tool),
                details=exc.details,
                registered=False,
            )

        if not endpoint_id:
            with self._lock:
                endpoint_id = f"{canonical}_{len(self._records)}"
        try:
            endpoint_id = validate_endpoint_id(endpoint_id)
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                tool=canonical,
                details=exc.details,
                registered=False,
            )

        cfg = dict(config or {})
        cfg.setdefault("tool", canonical)
        try:
            adapter = self.factory.create(
                canonical,
                endpoint_id,
                cli_path=cli_path,
                config=cfg,
            )
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                tool=canonical,
                endpoint_id=endpoint_id,
                details=exc.details,
                registered=False,
            )
        except Exception as exc:  # noqa: BLE001
            return error_envelope(
                f"failed to construct adapter: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED,
                tool=canonical,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
                registered=False,
            )

        try:
            return self.register_adapter(
                adapter, tool=canonical, replace=replace, probe=probe
            )
        except RegistryCollisionError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                tool=canonical,
                endpoint_id=endpoint_id,
                details=exc.details,
                registered=False,
            )

    def get(self, endpoint_id: str) -> Optional[EndpointAdapterProtocol]:
        with self._lock:
            record = self._records.get(endpoint_id)
            return record.adapter if record else None

    def get_record(self, endpoint_id: str) -> Optional[EndpointRecord]:
        with self._lock:
            return self._records.get(endpoint_id)

    def unregister(self, endpoint_id: str) -> bool:
        with self._lock:
            record = self._records.pop(endpoint_id, None)
        if record is None:
            return False
        # Clean up endpoint-local ACP process and sessions.
        client = getattr(record, "acp_client", None)
        if client is not None:
            try:
                client.stop()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "ACP stop on unregister failed for %s", endpoint_id
                )
            record.acp_client = None
        return True

    def list_endpoints(self, *, probe: bool = False) -> List[Dict[str, Any]]:
        """List registered endpoints. Side-effect free when probe=False."""
        with self._lock:
            records = list(self._records.values())
        return [record.describe(probe=probe) for record in records]

    def describe(
        self, endpoint_id: str, *, probe: bool = False
    ) -> dict[str, Any]:
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint {endpoint_id!r} not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        return {
            "status": "success",
            "success": True,
            "endpoint": record.describe(probe=probe),
        }

    def liveness(self, endpoint_id: str) -> dict[str, Any]:
        """Liveness: registered in-process (no CLI probe)."""
        record = self.get_record(endpoint_id)
        if record is None:
            return {
                "status": "error",
                "success": False,
                "live": False,
                "endpoint_id": endpoint_id,
                "error": f"CLI endpoint {endpoint_id!r} not found",
                "error_code": CLIRuntimeErrorCode.PROVIDER_NOT_FOUND.value,
            }
        return {
            "status": "success",
            "success": True,
            "live": True,
            "endpoint_id": endpoint_id,
            "tool": record.tool,
            "health": record.health.value,
        }

    def readiness(self, endpoint_id: str) -> dict[str, Any]:
        """Readiness: optional availability probe for a single endpoint.

        When the adapter exposes ``assess_health`` (e.g. Goose), prefer that
        so installed / configured / ready / degraded / unsupported_version
        remain distinct. List and liveness never call this path.
        """
        record = self.get_record(endpoint_id)
        if record is None:
            return {
                "status": "error",
                "success": False,
                "ready": False,
                "endpoint_id": endpoint_id,
                "error": f"CLI endpoint {endpoint_id!r} not found",
                "error_code": CLIRuntimeErrorCode.PROVIDER_NOT_FOUND.value,
            }
        assess = getattr(record.adapter, "assess_health", None)
        if callable(assess):
            try:
                health_info = dict(assess())
            except Exception as exc:  # noqa: BLE001
                record.health = EndpointHealth.DEGRADED
                return {
                    "status": "error",
                    "success": False,
                    "ready": False,
                    "endpoint_id": endpoint_id,
                    "tool": record.tool,
                    "health": record.health.value,
                    "error": f"readiness probe failed: {type(exc).__name__}",
                    "error_code": CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED.value,
                }
            health_value = str(health_info.get("health") or EndpointHealth.UNKNOWN.value)
            try:
                record.health = EndpointHealth(health_value)
            except ValueError:
                record.health = EndpointHealth.DEGRADED
                health_value = EndpointHealth.DEGRADED.value
            ready = bool(health_info.get("ready", False))
            available = bool(health_info.get("available", ready))
            result: dict[str, Any] = {
                "status": "success" if ready else "error",
                "success": ready,
                "ready": ready,
                "endpoint_id": endpoint_id,
                "tool": record.tool,
                "health": health_value,
                "available": available,
            }
            for key in (
                "installed",
                "configured",
                "goose_version",
                "version",
                "reason",
                "unsupported_version",
            ):
                if key in health_info:
                    result[key] = health_info[key]
            if health_info.get("error"):
                result["error"] = health_info["error"]
                result.setdefault(
                    "error_code",
                    health_info.get(
                        "error_code",
                        CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED.value,
                    ),
                )
            return result

        try:
            available = bool(record.adapter.is_available())
        except Exception as exc:  # noqa: BLE001
            record.health = EndpointHealth.DEGRADED
            return {
                "status": "error",
                "success": False,
                "ready": False,
                "endpoint_id": endpoint_id,
                "health": record.health.value,
                "error": f"readiness probe failed: {type(exc).__name__}",
                "error_code": CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED.value,
            }
        record.health = (
            EndpointHealth.READY if available else EndpointHealth.MISSING
        )
        return {
            "status": "success" if available else "error",
            "success": available,
            "ready": available,
            "endpoint_id": endpoint_id,
            "tool": record.tool,
            "health": record.health.value,
            "available": available,
        }

    def execute(
        self,
        endpoint_id: str,
        prompt: str,
        *,
        task_type: str = "text_generation",
        timeout: int = DEFAULT_EXECUTE_TIMEOUT_SECONDS,
        require_available: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute inference with bounds, nonzero-exit failure, and safe errors."""
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )

        try:
            bound = bound_prompt(prompt)
        except BoundsExceededError as exc:
            record.stats.record_failure(0.0)
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
            )
        except ContractValidationError as exc:
            record.stats.record_failure(0.0)
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
            )

        if require_available:
            try:
                available = bool(record.adapter.is_available())
            except Exception:  # noqa: BLE001
                available = False
            if not available:
                record.stats.record_failure(0.0)
                record.health = EndpointHealth.MISSING
                return error_envelope(
                    f"CLI tool for endpoint '{endpoint_id}' is not available",
                    code=CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED,
                    endpoint_id=endpoint_id,
                    tool=record.tool,
                )

        request_id = f"{endpoint_id}:{time.time_ns()}"
        with self._lock:
            record.active_request_id = request_id
            record.cancel_requested = False

        start = time.time()
        try:
            raw = record.adapter.execute(
                bound, task_type=task_type, timeout=timeout, **kwargs
            )
        except NonzeroExitError as exc:
            elapsed = time.time() - start
            record.stats.record_failure(elapsed)
            return error_envelope(
                exc.record.message,
                code=CLIRuntimeErrorCode.NONZERO_EXIT,
                endpoint_id=endpoint_id,
                tool=record.tool,
                details=exc.details,
                elapsed_time=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - surface typed envelope
            elapsed = time.time() - start
            record.stats.record_failure(elapsed)
            # Never include prompt or exception args that may hold it.
            return error_envelope(
                f"CLI execution error: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.INTERNAL,
                endpoint_id=endpoint_id,
                tool=record.tool,
                details={"error_type": type(exc).__name__},
                elapsed_time=elapsed,
            )
        finally:
            with self._lock:
                if record.active_request_id == request_id:
                    record.active_request_id = None
                    record.cancel_requested = False

        elapsed = time.time() - start
        result = dict(raw) if isinstance(raw, Mapping) else {"result": raw}
        result = sanitize_error_payload(result, prompt=bound)
        result.pop("prompt", None)

        returncode = result.get("returncode", 0)
        try:
            rc_int = int(returncode) if returncode is not None else 0
        except (TypeError, ValueError):
            rc_int = 0

        status = str(result.get("status") or "")
        failed = (
            rc_int != 0
            or status in {"error", "timeout", "validation_error", "nonzero_exit"}
            or bool(result.get("error"))
            or result.get("success") is False
        )

        if failed:
            record.stats.record_failure(elapsed)
            if rc_int != 0 and not result.get("error"):
                result["error"] = f"CLI exited with status {rc_int}"
            if rc_int != 0:
                result["error_code"] = result.get(
                    "error_code", CLIRuntimeErrorCode.NONZERO_EXIT.value
                )
            result["status"] = "error"
            result["success"] = False
            result["endpoint_id"] = endpoint_id
            result["elapsed_time"] = result.get("elapsed_time", elapsed)
            result["returncode"] = rc_int
            result.pop("prompt", None)
            # Bound any residual stderr/result text.
            if "result" in result and isinstance(result["result"], str):
                result["result"] = bound_result_text(result["result"])
            if "stderr" in result and isinstance(result["stderr"], str):
                result["stderr"] = _clip_text(
                    result["stderr"], MAX_STDERR_DIAGNOSTIC_CHARS
                )
            return result

        # Success path: bound the result text.
        if "result" in result and isinstance(result["result"], str):
            result["result"] = bound_result_text(result["result"])
        if "response" in result and isinstance(result["response"], str):
            result["response"] = bound_result_text(result["response"])
        if "raw_response" in result and isinstance(result["raw_response"], str):
            result["raw_response"] = bound_result_text(result["raw_response"])

        record.stats.record_success(elapsed)
        record.health = EndpointHealth.READY
        result.setdefault("status", "success")
        result["success"] = True
        result.setdefault("endpoint_id", endpoint_id)
        result.setdefault("elapsed_time", elapsed)
        result["returncode"] = rc_int
        result.pop("prompt", None)
        return result

    def stream(
        self,
        endpoint_id: str,
        prompt: str,
        **kwargs: Any,
    ):
        """Yield lifecycle events (ACP session stream when session_id set).

        When ``session_id`` is provided and the endpoint has a ready ACP
        client, events stream from the persistent ACP transport. Otherwise
        falls back to one-shot execute wrapping (no prompt echo).
        """
        session_id = kwargs.pop("session_id", None)
        record = self.get_record(endpoint_id)
        if (
            session_id
            and record is not None
            and record.acp_client is not None
            and getattr(record.acp_client, "is_ready", False)
        ):
            try:
                for event in record.acp_client.stream_prompt(
                    str(session_id),
                    prompt,
                    timeout=kwargs.get("timeout"),
                ):
                    # Never echo the prompt in streamed events.
                    safe = dict(event)
                    safe.pop("prompt", None)
                    yield safe
            except Exception as exc:  # noqa: BLE001
                from .acp.goose_client import (
                    ACPUncertainSideEffectError,
                    FAILURE_KIND_UNCERTAIN_SIDE_EFFECT,
                )

                uncertain = isinstance(exc, ACPUncertainSideEffectError) or (
                    getattr(exc, "uncertain_side_effects", False)
                )
                yield {
                    "event": "failed",
                    "endpoint_id": endpoint_id,
                    "session_id": session_id,
                    "error": (
                        exc.record.message
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
                        else None
                    ),
                }
            return

        yield {
            "event": "started",
            "endpoint_id": endpoint_id,
        }
        result = self.execute(endpoint_id, prompt, **kwargs)
        if result.get("success") is False or result.get("status") == "error":
            yield {
                "event": "failed",
                "endpoint_id": endpoint_id,
                "error": result.get("error"),
                "error_code": result.get("error_code"),
                "returncode": result.get("returncode"),
            }
        else:
            text = result.get("result") or result.get("response") or ""
            if isinstance(text, str):
                text = bound_result_text(text)
            yield {
                "event": "completed",
                "endpoint_id": endpoint_id,
                "result": text,
                "elapsed_time": result.get("elapsed_time"),
            }

    def cancel(self, endpoint_id: str, *, session_id: str | None = None) -> dict[str, Any]:
        """Request cancellation of in-flight execute and/or ACP session work."""
        with self._lock:
            record = self._records.get(endpoint_id)
            if record is None:
                return error_envelope(
                    f"CLI endpoint '{endpoint_id}' not found",
                    code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                    endpoint_id=endpoint_id,
                )
            client = record.acp_client
            active_id = record.active_request_id
            if active_id is not None:
                record.cancel_requested = True

        acp_result: Optional[dict[str, Any]] = None
        if client is not None and session_id:
            try:
                acp_result = client.session_cancel(session_id)
            except Exception as exc:  # noqa: BLE001
                acp_result = error_envelope(
                    f"ACP session cancel failed: {type(exc).__name__}",
                    code=CLIRuntimeErrorCode.INTERNAL,
                    endpoint_id=endpoint_id,
                    details={"error_type": type(exc).__name__},
                )
        elif client is not None and active_id is None and not session_id:
            # Cancel all open ACP sessions' pending prompts.
            try:
                sessions = client.list_sessions()
                cancelled = 0
                for sess in sessions:
                    sid = sess.get("session_id")
                    if sid:
                        out = client.session_cancel(str(sid))
                        cancelled += int(out.get("cancelled_pending") or 0)
                acp_result = {
                    "status": "success",
                    "success": True,
                    "cancelled_pending": cancelled,
                    "sessions": len(sessions),
                }
            except Exception as exc:  # noqa: BLE001
                acp_result = {
                    "status": "error",
                    "success": False,
                    "error": type(exc).__name__,
                }

        if active_id is None and acp_result is None:
            return {
                "status": "success",
                "success": True,
                "endpoint_id": endpoint_id,
                "cancelled": False,
                "message": "no active request",
            }
        return {
            "status": "success",
            "success": True,
            "endpoint_id": endpoint_id,
            "cancelled": active_id is not None
            or bool((acp_result or {}).get("cancelled_pending")),
            "request_id": active_id,
            "acp": acp_result,
        }

    # ------------------------------------------------------------------
    # Persistent Goose ACP lifecycle (endpoint-local)
    # ------------------------------------------------------------------

    def _get_goose_executable(
        self, record: EndpointRecord, *, cli_path: Optional[str] = None
    ) -> str:
        """Resolve an explicit goose executable for ACP (never PATH guessing)."""
        if cli_path:
            return str(cli_path)
        adapter = record.adapter
        path = getattr(adapter, "cli_path", None)
        if path:
            return str(path)
        cfg = getattr(adapter, "config", None) or {}
        if isinstance(cfg, Mapping):
            for key in ("cli_path", "executable", "goose_path"):
                if cfg.get(key):
                    return str(cfg[key])
        raise ContractValidationError(
            "ACP requires an explicit goose executable path on the endpoint",
            details={"endpoint_id": record.endpoint_id},
        )

    def _get_acp_state_root(
        self,
        record: EndpointRecord,
        *,
        state_root: Optional[str] = None,
    ) -> str:
        if state_root:
            return str(state_root)
        cfg = getattr(record.adapter, "config", None) or {}
        if isinstance(cfg, Mapping):
            for key in ("acp_state_root", "state_root", "GOOSE_PATH_ROOT", "path_root"):
                if cfg.get(key):
                    return str(cfg[key])
        raise ContractValidationError(
            "ACP requires an isolated absolute state_root (GOOSE_PATH_ROOT)",
            details={"endpoint_id": record.endpoint_id},
        )

    def acp_start(
        self,
        endpoint_id: str,
        *,
        executable: Optional[str] = None,
        state_root: Optional[str] = None,
        cwd: Optional[str] = None,
        max_restarts: int = 3,
        restart_on_unexpected_exit: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start endpoint-local ``goose acp`` after validating the executable.

        Does **not** enable ``goose serve`` or any network listener.
        """
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        # ACP is Goose-specific.
        tool = (record.tool or "").lower().replace("-", "_")
        if tool not in {"goose", "goose_cli", "block_goose", "aaif_goose"}:
            return error_envelope(
                "ACP sessions are only supported for goose endpoints",
                code=CLIRuntimeErrorCode.UNSUPPORTED_CAPABILITY,
                endpoint_id=endpoint_id,
                tool=record.tool,
            )
        try:
            from .acp.goose_client import (
                ACPBounds,
                ACPRestartPolicy,
                GooseACPClient,
            )

            exe = self._get_goose_executable(record, cli_path=executable)
            root = self._get_acp_state_root(record, state_root=state_root)
            # Refuse serve mode configuration if smuggled in kwargs.
            if kwargs.get("serve") or kwargs.get("dangerously_unauthenticated"):
                return error_envelope(
                    "goose serve / unauthenticated network mode is not supported",
                    code=CLIRuntimeErrorCode.POLICY_DENIED,
                    endpoint_id=endpoint_id,
                )
            existing = record.acp_client
            if existing is not None and getattr(existing, "is_ready", False):
                return {
                    "status": "success",
                    "success": True,
                    "endpoint_id": endpoint_id,
                    "already_started": True,
                    "acp": existing.describe(),
                }
            if existing is not None:
                try:
                    existing.stop()
                except Exception:  # noqa: BLE001
                    pass
            bounds = ACPBounds(
                max_restarts=int(kwargs.get("max_restarts", max_restarts)),
            )
            policy = ACPRestartPolicy(
                enabled=bool(kwargs.get("restart_enabled", True)),
                max_restarts=int(kwargs.get("max_restarts", max_restarts)),
                restart_on_unexpected_exit=bool(restart_on_unexpected_exit),
                auto_replay_agent_work=False,
            )
            client = GooseACPClient(
                exe,
                root,
                cwd=cwd or root,
                bounds=bounds,
                restart_policy=policy,
                endpoint_id=endpoint_id,
                env=kwargs.get("env"),
            )
            start_result = client.start()
            with self._lock:
                record.acp_client = client
                record.health = EndpointHealth.READY
            return {
                "status": "success",
                "success": True,
                "endpoint_id": endpoint_id,
                "acp": client.describe(),
                "start": start_result,
                "auto_replay_agent_work": False,
            }
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                tool=record.tool,
                details=exc.details,
            )
        except Exception as exc:  # noqa: BLE001
            return error_envelope(
                f"ACP start failed: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.SPAWN_FAILED,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
            )

    def acp_stop(self, endpoint_id: str) -> dict[str, Any]:
        """Stop the endpoint-local ACP process and clear sessions."""
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None:
            return {
                "status": "success",
                "success": True,
                "endpoint_id": endpoint_id,
                "stopped": False,
                "message": "no ACP client",
            }
        try:
            result = client.stop()
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "error",
                "success": False,
                "error": type(exc).__name__,
            }
        with self._lock:
            record.acp_client = None
        return {
            "status": "success",
            "success": True,
            "endpoint_id": endpoint_id,
            "stopped": True,
            "result": result,
        }

    def acp_restart(self, endpoint_id: str, **kwargs: Any) -> dict[str, Any]:
        """Explicit transport restart; never auto-replays agent work."""
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None:
            return self.acp_start(endpoint_id, **kwargs)
        try:
            result = client.restart_transport(explicit=True)
            return {
                "status": "success",
                "success": True,
                "endpoint_id": endpoint_id,
                "auto_replay_agent_work": False,
                "result": result,
                "acp": client.describe(),
            }
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details={
                    **dict(exc.details),
                    "auto_replay_agent_work": "false",
                },
            )
        except Exception as exc:  # noqa: BLE001
            return error_envelope(
                f"ACP restart failed: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.SPAWN_FAILED,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
            )

    def acp_describe(self, endpoint_id: str) -> dict[str, Any]:
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None:
            return {
                "status": "success",
                "success": True,
                "endpoint_id": endpoint_id,
                "acp": None,
                "message": "no ACP client",
            }
        return {
            "status": "success",
            "success": True,
            "endpoint_id": endpoint_id,
            "acp": client.describe(),
            "sessions": client.list_sessions(),
        }

    def session_new(
        self,
        endpoint_id: str,
        *,
        cwd: Optional[str] = None,
        mcp_servers: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None or not getattr(client, "is_ready", False):
            return error_envelope(
                "ACP client is not ready; call acp_start first",
                code=CLIRuntimeErrorCode.INVALID_STATE,
                endpoint_id=endpoint_id,
            )
        try:
            return client.session_new(
                cwd=cwd,
                mcp_servers=mcp_servers or (),
                metadata=kwargs.get("metadata"),
                timeout=kwargs.get("timeout"),
            )
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
            )
        except Exception as exc:  # noqa: BLE001
            return error_envelope(
                f"session_new failed: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.INTERNAL,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
            )

    def session_load(
        self,
        endpoint_id: str,
        session_id: str,
        *,
        cwd: Optional[str] = None,
        mcp_servers: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None or not getattr(client, "is_ready", False):
            return error_envelope(
                "ACP client is not ready; call acp_start first",
                code=CLIRuntimeErrorCode.INVALID_STATE,
                endpoint_id=endpoint_id,
            )
        try:
            return client.session_load(
                session_id,
                cwd=cwd,
                mcp_servers=mcp_servers or (),
                timeout=kwargs.get("timeout"),
            )
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
            )
        except Exception as exc:  # noqa: BLE001
            return error_envelope(
                f"session_load failed: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.INTERNAL,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
            )

    def session_close(
        self, endpoint_id: str, session_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None:
            return error_envelope(
                "no ACP client on endpoint",
                code=CLIRuntimeErrorCode.INVALID_STATE,
                endpoint_id=endpoint_id,
            )
        try:
            return client.session_close(
                session_id,
                timeout=kwargs.get("timeout"),
                remote=bool(kwargs.get("remote", True)),
            )
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
            )
        except Exception as exc:  # noqa: BLE001
            return error_envelope(
                f"session_close failed: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.INTERNAL,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
            )

    def session_prompt(
        self,
        endpoint_id: str,
        session_id: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a prompt on an endpoint-local ACP session (never auto-replayed)."""
        record = self.get_record(endpoint_id)
        if record is None:
            return error_envelope(
                f"CLI endpoint '{endpoint_id}' not found",
                code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
                endpoint_id=endpoint_id,
            )
        client = record.acp_client
        if client is None or not getattr(client, "is_ready", False):
            return error_envelope(
                "ACP client is not ready; call acp_start first",
                code=CLIRuntimeErrorCode.INVALID_STATE,
                endpoint_id=endpoint_id,
            )
        try:
            bound = bound_prompt(prompt)
        except CLIRuntimeError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
            )
        request_id = f"{endpoint_id}:acp:{time.time_ns()}"
        with self._lock:
            record.active_request_id = request_id
            record.cancel_requested = False
        start = time.time()
        try:
            result = client.session_prompt(
                session_id,
                bound,
                timeout=kwargs.get("timeout"),
                on_event=kwargs.get("on_event"),
            )
            elapsed = time.time() - start
            record.stats.record_success(elapsed)
            result = dict(result)
            result["elapsed_time"] = elapsed
            result["endpoint_id"] = endpoint_id
            result.pop("prompt", None)
            if isinstance(result.get("text"), str):
                result["text"] = bound_result_text(result["text"])
            return result
        except CLIRuntimeError as exc:
            elapsed = time.time() - start
            record.stats.record_failure(elapsed)
            from .acp.goose_client import (
                FAILURE_KIND_UNCERTAIN_SIDE_EFFECT,
                STATUS_UNCERTAIN_SIDE_EFFECT,
            )

            uncertain = bool(
                getattr(exc, "uncertain_side_effects", False)
                or (exc.details or {}).get("uncertain_side_effects")
                or (exc.details or {}).get("failure_kind")
                == FAILURE_KIND_UNCERTAIN_SIDE_EFFECT
            )
            envelope = error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=endpoint_id,
                details=exc.details,
                elapsed_time=elapsed,
                session_id=session_id,
                side_effects_started=True if uncertain else False,
                uncertain_side_effects=uncertain,
                failure_kind=(
                    FAILURE_KIND_UNCERTAIN_SIDE_EFFECT if uncertain else None
                ),
                cacheable=False,
                retryable=False,
            )
            if uncertain:
                envelope["status"] = STATUS_UNCERTAIN_SIDE_EFFECT
            return envelope
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            record.stats.record_failure(elapsed)
            return error_envelope(
                f"session_prompt failed: {type(exc).__name__}",
                code=CLIRuntimeErrorCode.INTERNAL,
                endpoint_id=endpoint_id,
                details={"error_type": type(exc).__name__},
                elapsed_time=elapsed,
            )
        finally:
            with self._lock:
                if record.active_request_id == request_id:
                    record.active_request_id = None
                    record.cancel_requested = False

    def dispatch(
        self,
        operation: EndpointLifecycleOp | str,
        *,
        endpoint_id: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Lifecycle dispatcher for list/describe/liveness/readiness/execute/..."""
        if isinstance(operation, str):
            try:
                operation = EndpointLifecycleOp(operation)
            except ValueError:
                return error_envelope(
                    f"unsupported lifecycle operation: {operation}",
                    code=CLIRuntimeErrorCode.UNSUPPORTED_CAPABILITY,
                    details={"operation": str(operation)},
                )
        if operation is EndpointLifecycleOp.LIST:
            return {
                "status": "success",
                "success": True,
                "endpoints": self.list_endpoints(probe=bool(kwargs.get("probe"))),
                "count": len(self._records),
            }
        if endpoint_id is None and operation is not EndpointLifecycleOp.LIST:
            return error_envelope(
                "endpoint_id is required",
                code=CLIRuntimeErrorCode.INVALID_CONTRACT,
            )
        assert endpoint_id is not None
        if operation is EndpointLifecycleOp.DESCRIBE:
            return self.describe(endpoint_id, probe=bool(kwargs.get("probe")))
        if operation is EndpointLifecycleOp.LIVENESS:
            return self.liveness(endpoint_id)
        if operation is EndpointLifecycleOp.READINESS:
            return self.readiness(endpoint_id)
        if operation is EndpointLifecycleOp.EXECUTE:
            if prompt is None:
                return error_envelope(
                    "prompt is required for execute",
                    code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                    endpoint_id=endpoint_id,
                )
            return self.execute(endpoint_id, prompt, **kwargs)
        if operation is EndpointLifecycleOp.STREAM:
            if prompt is None:
                return error_envelope(
                    "prompt is required for stream",
                    code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                    endpoint_id=endpoint_id,
                )
            return list(self.stream(endpoint_id, prompt, **kwargs))
        if operation is EndpointLifecycleOp.CANCEL:
            return self.cancel(
                endpoint_id, session_id=kwargs.get("session_id")
            )
        if operation is EndpointLifecycleOp.ACP_START:
            return self.acp_start(endpoint_id, **kwargs)
        if operation is EndpointLifecycleOp.ACP_STOP:
            return self.acp_stop(endpoint_id)
        if operation is EndpointLifecycleOp.ACP_RESTART:
            return self.acp_restart(endpoint_id, **kwargs)
        if operation is EndpointLifecycleOp.ACP_DESCRIBE:
            return self.acp_describe(endpoint_id)
        if operation is EndpointLifecycleOp.SESSION_NEW:
            return self.session_new(endpoint_id, **kwargs)
        if operation is EndpointLifecycleOp.SESSION_LOAD:
            session_id = kwargs.pop("session_id", None)
            if not session_id:
                return error_envelope(
                    "session_id is required for session_load",
                    code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                    endpoint_id=endpoint_id,
                )
            return self.session_load(endpoint_id, str(session_id), **kwargs)
        if operation is EndpointLifecycleOp.SESSION_CLOSE:
            session_id = kwargs.pop("session_id", None)
            if not session_id:
                return error_envelope(
                    "session_id is required for session_close",
                    code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                    endpoint_id=endpoint_id,
                )
            return self.session_close(endpoint_id, str(session_id), **kwargs)
        if operation is EndpointLifecycleOp.SESSION_PROMPT:
            session_id = kwargs.pop("session_id", None)
            if not session_id:
                return error_envelope(
                    "session_id is required for session_prompt",
                    code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                    endpoint_id=endpoint_id,
                )
            if prompt is None:
                return error_envelope(
                    "prompt is required for session_prompt",
                    code=CLIRuntimeErrorCode.INVALID_CONTRACT,
                    endpoint_id=endpoint_id,
                )
            return self.session_prompt(
                endpoint_id, str(session_id), prompt, **kwargs
            )
        if operation is EndpointLifecycleOp.SESSION_CANCEL:
            session_id = kwargs.get("session_id")
            return self.cancel(endpoint_id, session_id=session_id)
        return error_envelope(
            f"unsupported lifecycle operation: {operation}",
            code=CLIRuntimeErrorCode.UNSUPPORTED_CAPABILITY,
        )


class _RegistryAdapterView(MutableMapping):
    """Dict-like view over registry adapters for legacy CLI_ADAPTER_REGISTRY."""

    def __init__(self, registry: CLIEndpointRegistry) -> None:
        self._registry = registry

    def __getitem__(self, key: str) -> Any:
        adapter = self._registry.get(key)
        if adapter is None:
            raise KeyError(key)
        return adapter

    def __setitem__(self, key: str, value: Any) -> None:
        # Preserve endpoint_id consistency with the key when possible.
        if getattr(value, "endpoint_id", None) != key:
            try:
                value.endpoint_id = key
            except Exception:  # noqa: BLE001
                pass
        self._registry.register_adapter(value, replace=True)

    def __delitem__(self, key: str) -> None:
        if not self._registry.unregister(key):
            raise KeyError(key)

    def __iter__(self):
        with self._registry._lock:
            return iter(list(self._registry._records.keys()))

    def __len__(self) -> int:
        with self._registry._lock:
            return len(self._registry._records)

    def keys(self):  # type: ignore[override]
        with self._registry._lock:
            return list(self._registry._records.keys())

    def values(self):  # type: ignore[override]
        with self._registry._lock:
            return [r.adapter for r in self._registry._records.values()]

    def items(self):  # type: ignore[override]
        with self._registry._lock:
            return [
                (eid, r.adapter) for eid, r in self._registry._records.items()
            ]

    def clear(self) -> None:  # type: ignore[override]
        self._registry.clear()

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        adapter = self._registry.get(key)
        return default if adapter is None else adapter


# ---------------------------------------------------------------------------
# Process-wide defaults
# ---------------------------------------------------------------------------

_DEFAULT_FACTORY: Optional[CLIEndpointFactory] = None
_DEFAULT_REGISTRY: Optional[CLIEndpointRegistry] = None
_DEFAULT_LOCK = threading.RLock()


def get_default_endpoint_factory() -> CLIEndpointFactory:
    global _DEFAULT_FACTORY
    with _DEFAULT_LOCK:
        if _DEFAULT_FACTORY is None:
            _DEFAULT_FACTORY = CLIEndpointFactory()
        return _DEFAULT_FACTORY


def get_default_endpoint_registry() -> CLIEndpointRegistry:
    global _DEFAULT_REGISTRY, _DEFAULT_FACTORY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None:
            if _DEFAULT_FACTORY is None:
                _DEFAULT_FACTORY = CLIEndpointFactory()
            _DEFAULT_REGISTRY = CLIEndpointRegistry(factory=_DEFAULT_FACTORY)
        return _DEFAULT_REGISTRY


def reset_default_endpoint_registry() -> None:
    """Clear process-wide factory/registry (tests only)."""
    global _DEFAULT_FACTORY, _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is not None:
            # Stop any endpoint-local ACP clients before clearing.
            try:
                for record in list(
                    getattr(_DEFAULT_REGISTRY, "_records", {}).values()
                ):
                    client = getattr(record, "acp_client", None)
                    if client is not None:
                        try:
                            client.stop()
                        except Exception:  # noqa: BLE001
                            pass
                        record.acp_client = None
            except Exception:  # noqa: BLE001
                pass
            _DEFAULT_REGISTRY.clear()
        _DEFAULT_FACTORY = CLIEndpointFactory()
        _DEFAULT_REGISTRY = CLIEndpointRegistry(factory=_DEFAULT_FACTORY)


class _LazyDefaultAdapterView(MutableMapping):
    """Defers to the process-wide registry adapters mapping."""

    def _view(self) -> MutableMapping:
        return get_default_endpoint_registry().adapters

    def __getitem__(self, key: str) -> Any:
        return self._view()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._view()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._view()[key]

    def __iter__(self):
        return iter(self._view())

    def __len__(self) -> int:
        return len(self._view())

    def clear(self) -> None:  # type: ignore[override]
        self._view().clear()

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self._view().get(key, default)

    def keys(self):  # type: ignore[override]
        return self._view().keys()

    def values(self):  # type: ignore[override]
        return self._view().values()

    def items(self):  # type: ignore[override]
        return self._view().items()


# Legacy-compatible registry dict view (endpoint_id -> adapter).
CLI_ADAPTER_REGISTRY: MutableMapping[str, Any] = _LazyDefaultAdapterView()


def create_cli_endpoint(
    tool: str,
    endpoint_id: str,
    *,
    cli_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> EndpointAdapterProtocol:
    """Canonical concrete factory entrypoint (never instantiates the ABC)."""
    return get_default_endpoint_factory().create(
        tool, endpoint_id, cli_path=cli_path, config=config
    )


def register_cli_endpoint(
    adapter: EndpointAdapterProtocol | None = None,
    *,
    tool: str | None = None,
    endpoint_id: str | None = None,
    cli_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    replace: bool = False,
    probe: bool = False,
) -> dict[str, Any]:
    """Register a concrete adapter or create-and-register by tool name.

    Pass either an *adapter* instance **or** a *tool* name (with optional
    endpoint_id/config). Instantiating the abstract ``CLIEndpointAdapter`` is
    rejected with a typed error envelope.
    """
    registry = get_default_endpoint_registry()
    if adapter is not None:
        try:
            return registry.register_adapter(
                adapter, tool=tool, replace=replace, probe=probe
            )
        except RegistryCollisionError as exc:
            return error_envelope(
                exc.record.message,
                code=exc.code,
                endpoint_id=getattr(adapter, "endpoint_id", None),
                details=exc.details,
                registered=False,
            )
    if tool is None:
        return error_envelope(
            "register_cli_endpoint requires adapter or tool",
            code=CLIRuntimeErrorCode.INVALID_CONTRACT,
            registered=False,
        )
    return registry.register_tool(
        tool,
        endpoint_id=endpoint_id,
        cli_path=cli_path,
        config=config,
        replace=replace,
        probe=probe,
    )


def get_cli_endpoint(endpoint_id: str) -> Optional[EndpointAdapterProtocol]:
    return get_default_endpoint_registry().get(endpoint_id)


def list_cli_endpoints(*, probe: bool = False) -> List[Dict[str, Any]]:
    """List registered endpoints without probing every provider by default."""
    return get_default_endpoint_registry().list_endpoints(probe=probe)


def list_cli_endpoint_tools() -> List[Dict[str, Any]]:
    """List known tool types (metadata only; no factories, no probes)."""
    return [spec.to_dict() for spec in get_default_endpoint_factory().list_tool_specs()]


def execute_cli_inference(
    endpoint_id: str,
    prompt: str,
    task_type: str = "text_generation",
    timeout: int = DEFAULT_EXECUTE_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute via the canonical registry (bounds + nonzero-exit failure)."""
    return get_default_endpoint_registry().execute(
        endpoint_id,
        prompt,
        task_type=task_type,
        timeout=timeout,
        **kwargs,
    )


__all__ = [
    "MAX_ENDPOINT_ID_CHARS",
    "MAX_RESULT_CHARS",
    "MAX_STDERR_DIAGNOSTIC_CHARS",
    "DEFAULT_EXECUTE_TIMEOUT_SECONDS",
    "EndpointHealth",
    "EndpointLifecycleOp",
    "EndpointToolSpec",
    "EndpointStats",
    "EndpointRecord",
    "EndpointAdapterProtocol",
    "UnsupportedEndpointToolError",
    "EndpointNotFoundError",
    "EndpointUnavailableError",
    "CLIEndpointFactory",
    "CLIEndpointRegistry",
    "CLI_ADAPTER_REGISTRY",
    "sanitize_error_payload",
    "error_envelope",
    "bound_prompt",
    "bound_result_text",
    "validate_endpoint_id",
    "get_default_endpoint_factory",
    "get_default_endpoint_registry",
    "reset_default_endpoint_registry",
    "create_cli_endpoint",
    "register_cli_endpoint",
    "get_cli_endpoint",
    "list_cli_endpoints",
    "list_cli_endpoint_tools",
    "execute_cli_inference",
]
