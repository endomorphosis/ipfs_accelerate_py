"""Unified LLM call observations and failure classification.

Provider-neutral records used by DuckDB allocation. Never stores prompts,
credentials, raw endpoints, or generated text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from ..common.meta_model_api import (
    MetaModelApiErrorKind,
    classify_meta_diagnostic_text,
)


class CallProtocol(str, Enum):
    HTTP = "http"
    CLI = "cli"
    SDK = "sdk"
    LOCAL = "local"
    UNKNOWN = "unknown"


class CallErrorKind(str, Enum):
    SUCCESS = "success"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    SERVER = "server"
    TRANSPORT = "transport"
    POLICY = "policy"
    UNKNOWN = "unknown"


_META_KIND_MAP = {
    MetaModelApiErrorKind.AUTHENTICATION: CallErrorKind.AUTHENTICATION,
    MetaModelApiErrorKind.BILLING: CallErrorKind.BILLING,
    MetaModelApiErrorKind.RATE_LIMIT: CallErrorKind.RATE_LIMIT,
    MetaModelApiErrorKind.INVALID_REQUEST: CallErrorKind.INVALID_REQUEST,
    MetaModelApiErrorKind.NOT_FOUND: CallErrorKind.NOT_FOUND,
    MetaModelApiErrorKind.GATEWAY_TIMEOUT: CallErrorKind.TIMEOUT,
    MetaModelApiErrorKind.SERVER: CallErrorKind.SERVER,
    MetaModelApiErrorKind.SERVICE_UNAVAILABLE: CallErrorKind.SERVER,
    MetaModelApiErrorKind.CONTEXT_WINDOW: CallErrorKind.INVALID_REQUEST,
    MetaModelApiErrorKind.CONTENT_POLICY: CallErrorKind.POLICY,
    MetaModelApiErrorKind.FORBIDDEN: CallErrorKind.POLICY,
    MetaModelApiErrorKind.PAYLOAD_TOO_LARGE: CallErrorKind.INVALID_REQUEST,
    MetaModelApiErrorKind.TRANSPORT: CallErrorKind.TRANSPORT,
}

_PROVIDER_PROTOCOL: dict[str, CallProtocol] = {
    "meta_ai": CallProtocol.HTTP,
    "openai": CallProtocol.HTTP,
    "openrouter": CallProtocol.HTTP,
    "xai": CallProtocol.HTTP,
    "hf_inference_api": CallProtocol.HTTP,
    "muse_code": CallProtocol.CLI,
    "goose_cli": CallProtocol.CLI,
    "codex_cli": CallProtocol.CLI,
    "copilot_cli": CallProtocol.CLI,
    "grok_cli": CallProtocol.CLI,
    "claude_code": CallProtocol.CLI,
    "gemini_cli": CallProtocol.CLI,
    "mistral_vibe": CallProtocol.CLI,
    "claude_py": CallProtocol.SDK,
    "gemini_py": CallProtocol.SDK,
    "copilot_sdk": CallProtocol.SDK,
    "local_hf": CallProtocol.LOCAL,
    "llama_cpp": CallProtocol.LOCAL,
    "llama_cpp_native": CallProtocol.LOCAL,
    "mock": CallProtocol.LOCAL,
}

_RETRYABLE = frozenset(
    {CallErrorKind.RATE_LIMIT, CallErrorKind.SERVER, CallErrorKind.TRANSPORT, CallErrorKind.TIMEOUT}
)


@dataclass(frozen=True)
class CallFailure:
    kind: CallErrorKind
    retryable: bool
    message: str = ""
    status_code: Optional[int] = None


@dataclass
class CallObservation:
    """One bounded LLM call outcome for allocation."""

    provider: str
    protocol: str = CallProtocol.UNKNOWN.value
    model: str = ""
    success: bool = False
    latency_ms: float = 0.0
    error_kind: str = CallErrorKind.UNKNOWN.value
    retryable: bool = False
    status_code: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    remaining_requests: Optional[int] = None
    remaining_tokens: Optional[int] = None
    limit_requests: Optional[int] = None
    limit_tokens: Optional[int] = None
    session_id: str = ""
    path: str = ""
    tokens_per_second: float = 0.0
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def protocol_for_provider(provider: str) -> CallProtocol:
    key = str(provider or "").strip().lower().replace("-", "_")
    return _PROVIDER_PROTOCOL.get(key, CallProtocol.UNKNOWN)


def classify_provider_failure(
    provider: str,
    text: str = "",
    *,
    exc: Optional[BaseException] = None,
    status_code: Optional[int] = None,
) -> CallFailure:
    """Classify a CLI/API/SDK failure into a closed error vocabulary."""
    kind_attr = getattr(exc, "meta_error_kind", None) if exc is not None else None
    if isinstance(kind_attr, MetaModelApiErrorKind):
        mapped = _META_KIND_MAP.get(kind_attr, CallErrorKind.UNKNOWN)
        return CallFailure(
            kind=mapped,
            retryable=mapped in _RETRYABLE,
            message=str(exc or text)[:512],
            status_code=getattr(exc, "status_code", status_code),
        )

    message = str(text or exc or "")
    lowered = message.lower()
    key = str(provider or "").strip().lower().replace("-", "_")
    code = status_code
    if code is None and exc is not None:
        raw = getattr(exc, "status_code", None)
        if isinstance(raw, int):
            code = raw

    if key in {"meta_ai", "muse_code", "goose_cli"} or "meta ai http" in lowered:
        failure = classify_meta_diagnostic_text(message)
        mapped = _META_KIND_MAP.get(failure.kind, CallErrorKind.UNKNOWN)
        return CallFailure(
            kind=mapped,
            retryable=mapped in _RETRYABLE,
            message=failure.message[:512],
            status_code=failure.status_code or code,
        )

    if key in {"codex_cli", "codex"}:
        if any(
            token in lowered
            for token in (
                "insufficient_quota",
                "exceeded your current quota",
                "billing",
                "payment method",
            )
        ) and "usage limit" not in lowered:
            return CallFailure(CallErrorKind.BILLING, False, message[:512], code)
        if "usage_limit" in lowered or "usage limit" in lowered:
            return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)

    if key in {"grok_cli", "xai"}:
        if "not signed in" in lowered or "not authenticated" in lowered:
            return CallFailure(CallErrorKind.AUTHENTICATION, False, message[:512], code)
        if "rate limit" in lowered or "429" in lowered:
            return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)

    if key in {"mistral_vibe", "vibe"}:
        if "labs_not_enabled" in lowered:
            return CallFailure(CallErrorKind.POLICY, False, message[:512], code)
        if "rate_limit" in lowered or "retry after" in lowered:
            return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)

    if key in {"claude_code", "claude_py", "claude"}:
        if "credit" in lowered or "billing" in lowered:
            return CallFailure(CallErrorKind.BILLING, False, message[:512], code)
        if "rate limit" in lowered or "429" in lowered:
            return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)

    if key in {"gemini_cli", "gemini_py", "gemini"}:
        if "billing" in lowered or "disabled" in lowered:
            return CallFailure(CallErrorKind.BILLING, False, message[:512], code)
        if "resource exhausted" in lowered or "rate" in lowered:
            return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)

    if key in {"copilot_cli", "copilot_sdk", "copilot"}:
        if "subscription" in lowered or "upgrade" in lowered:
            return CallFailure(CallErrorKind.BILLING, False, message[:512], code)
        if "rate" in lowered or "usage" in lowered:
            return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)

    if code == 401 or "unauthorized" in lowered or "invalid api key" in lowered:
        return CallFailure(CallErrorKind.AUTHENTICATION, False, message[:512], code)
    if code == 402 or "insufficient_quota" in lowered or "payment required" in lowered:
        return CallFailure(CallErrorKind.BILLING, False, message[:512], code)
    if code == 429 or "rate limit" in lowered or "too many requests" in lowered:
        return CallFailure(CallErrorKind.RATE_LIMIT, True, message[:512], code)
    if code in {500, 503} or "overloaded" in lowered:
        return CallFailure(CallErrorKind.SERVER, True, message[:512], code)
    if code == 408 or "timeout" in lowered:
        return CallFailure(CallErrorKind.TIMEOUT, True, message[:512], code)
    if "not found" in lowered:
        return CallFailure(CallErrorKind.NOT_FOUND, False, message[:512], code)
    return CallFailure(CallErrorKind.UNKNOWN, False, message[:512], code)


def observation_from_exception(
    *,
    provider: str,
    model: str = "",
    latency_ms: float = 0.0,
    exc: BaseException,
) -> CallObservation:
    failure = classify_provider_failure(provider, str(exc), exc=exc)
    return CallObservation(
        provider=str(provider or "").strip().lower(),
        protocol=protocol_for_provider(provider).value,
        model=str(model or "")[:128],
        success=False,
        latency_ms=float(latency_ms),
        error_kind=failure.kind.value,
        retryable=failure.retryable,
        status_code=failure.status_code,
    )
