"""Shared Meta Model API configuration, catalog, usage, and error contracts.

Sources (operator-facing, no secrets)::

    https://dev.meta.ai/docs/pricing-rate-limits/
    https://dev.meta.ai/docs/error-handling/
    https://dev.meta.ai/docs/token-counting/
    https://dev.meta.ai/docs/models/

The hosted default model remains ``muse-spark-1.1`` for compatibility.
Muse Spark 1.3 is the current recommended model for new work.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

META_MODEL_API_BASE_URL = "https://api.meta.ai/v1"
META_MODEL_API_DEFAULT_MODEL = "muse-spark-1.1"
META_MODEL_API_RECOMMENDED_MODEL = "muse-spark-1.3"
META_MODEL_API_SECRET_NAME = "meta_ai_api_key"
META_MODEL_API_ENV_VARS = (
    "META_API_KEY",
    "MODEL_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
)

MUSE_SPARK_CONTEXT_WINDOW = 1_048_576
MUSE_SPARK_MAX_OUTPUT_TOKENS = 131_072
MUSE_IMAGE_RPM = 150
MUSE_VOICE_CONCURRENT_STREAMS = 128
MUSE_VOICE_STREAMS_PER_HOUR = 16_000
MUSE_VOICE_USD_PER_HOUR = 0.18
WEB_SEARCH_USD_PER_1000_QUERIES = 2.50
BACKGROUND_SUBMISSIONS_PER_MINUTE = 600
META_RETRY_BASE_SECONDS = 0.5
META_RETRY_MAX_ATTEMPTS = 3

_MODEL_ALIASES = {
    "meta-spark/Spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "meta/muse-spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "meta-ai/muse-spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "muse-spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "meta/muse-spark-1.2": "muse-spark-1.2",
    "meta-ai/muse-spark-1.2": "muse-spark-1.2",
    "muse-spark-1.2": "muse-spark-1.2",
    "meta/muse-spark-1.2-contributor": "muse-spark-1.2-contributor",
    "muse-spark-1.2-contributor": "muse-spark-1.2-contributor",
    "meta/muse-spark-1.3": "muse-spark-1.3",
    "meta-ai/muse-spark-1.3": "muse-spark-1.3",
    "muse-spark-1.3": "muse-spark-1.3",
    "meta/muse-spark-1.3-contributor": "muse-spark-1.3-contributor",
    "muse-spark-1.3-contributor": "muse-spark-1.3-contributor",
}

_LAST_OBSERVATION = threading.local()
_SleepFn = Callable[[float], None]


class MetaPricingTier(str, Enum):
    STANDARD = "standard"
    CONTRIBUTOR = "contributor"


class MetaModelApiErrorKind(str, Enum):
    """Closed vocabulary matching Meta Model API status/type/code."""

    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    SERVICE_UNAVAILABLE = "service_unavailable"
    GATEWAY_TIMEOUT = "gateway_timeout"
    CONTEXT_WINDOW = "context_window"
    CONTENT_POLICY = "content_policy"
    TRANSPORT = "transport"


_RETRYABLE_KINDS = frozenset(
    {
        MetaModelApiErrorKind.RATE_LIMIT,
        MetaModelApiErrorKind.SERVER,
        MetaModelApiErrorKind.SERVICE_UNAVAILABLE,
    }
)

# Standard / contributor token prices are USD per 1M tokens.
_STANDARD_PRICING = {
    "cached_input": 0.15,
    "input": 1.25,
    "output": 4.25,
}
_CONTRIBUTOR_PRICING = {
    "cached_input": 0.002,
    "input": 0.10,
    "output": 0.20,
}
_STANDARD_LIMITS = {"rpm": 3_000, "tpm": 4_000_000}
_CONTRIBUTOR_LIMITS = {"rpm": 100, "tpm": 3_000_000}

MUSE_SPARK_MODELS: dict[str, dict[str, Any]] = {
    "muse-spark-1.3": {
        "tier": MetaPricingTier.STANDARD.value,
        "context_window": MUSE_SPARK_CONTEXT_WINDOW,
        "max_output_tokens": MUSE_SPARK_MAX_OUTPUT_TOKENS,
        "modalities": ("text", "image", "video", "pdf"),
        "recommended": True,
    },
    "muse-spark-1.3-contributor": {
        "tier": MetaPricingTier.CONTRIBUTOR.value,
        "context_window": MUSE_SPARK_CONTEXT_WINDOW,
        "max_output_tokens": MUSE_SPARK_MAX_OUTPUT_TOKENS,
        "modalities": ("text", "image", "video", "pdf"),
        "training_eligible": True,
    },
    "muse-spark-1.2": {
        "tier": MetaPricingTier.STANDARD.value,
        "context_window": MUSE_SPARK_CONTEXT_WINDOW,
        "max_output_tokens": MUSE_SPARK_MAX_OUTPUT_TOKENS,
        "modalities": ("text", "image", "video", "audio", "pdf"),
    },
    "muse-spark-1.2-contributor": {
        "tier": MetaPricingTier.CONTRIBUTOR.value,
        "context_window": MUSE_SPARK_CONTEXT_WINDOW,
        "max_output_tokens": MUSE_SPARK_MAX_OUTPUT_TOKENS,
        "modalities": ("text", "image", "video", "audio", "pdf"),
        "training_eligible": True,
    },
    "muse-spark-1.1": {
        "tier": MetaPricingTier.STANDARD.value,
        "context_window": MUSE_SPARK_CONTEXT_WINDOW,
        "max_output_tokens": MUSE_SPARK_MAX_OUTPUT_TOKENS,
        "modalities": ("text", "image", "video", "audio", "pdf"),
    },
}

_RATE_LIMIT_HEADER_MAP = {
    "x-ratelimit-limit-tokens": "limit_tokens",
    "x-ratelimit-remaining-tokens": "remaining_tokens",
    "x-ratelimit-limit-requests": "limit_requests",
    "x-ratelimit-remaining-requests": "remaining_requests",
}

_RETRY_AFTER_RE = re.compile(r"retry(?:\s+|-)?after[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)
_CONTEXT_WINDOW_RE = re.compile(
    r"context (?:length|window)|input_tokens \+ max_output_tokens|maximum input length",
    re.IGNORECASE,
)


def normalize_meta_model_name(model_name: Optional[str]) -> str:
    value = str(model_name or "").strip()
    if not value:
        return META_MODEL_API_DEFAULT_MODEL
    return _MODEL_ALIASES.get(value, value)


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_meta_model_api_key(
    explicit: Optional[str] = None,
    *,
    secrets_manager: Any = None,
    use_secrets_manager: bool = True,
) -> Optional[str]:
    value = str(explicit or "").strip()
    if value:
        return value
    for variable in META_MODEL_API_ENV_VARS:
        value = str(os.environ.get(variable) or "").strip()
        if value:
            return value
    if not use_secrets_manager or _truthy(
        os.environ.get("IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER")
    ):
        return None
    manager = secrets_manager
    if manager is None:
        try:
            from .secrets_manager import get_global_secrets_manager

            manager = get_global_secrets_manager()
        except Exception:
            return None
    for credential_name in (META_MODEL_API_SECRET_NAME, "model_api_key"):
        try:
            value = str(manager.get_credential(credential_name) or "").strip()
        except Exception:
            continue
        if value:
            return value
    return None


def meta_model_api_key_fingerprint() -> str:
    value = resolve_meta_model_api_key()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def meta_model_tier(model_name: Optional[str]) -> MetaPricingTier:
    canonical = normalize_meta_model_name(model_name)
    info = MUSE_SPARK_MODELS.get(canonical)
    if info and str(info.get("tier")) == MetaPricingTier.CONTRIBUTOR.value:
        return MetaPricingTier.CONTRIBUTOR
    if canonical.endswith("-contributor"):
        return MetaPricingTier.CONTRIBUTOR
    return MetaPricingTier.STANDARD


def meta_model_context_window(model_name: Optional[str]) -> int:
    canonical = normalize_meta_model_name(model_name)
    info = MUSE_SPARK_MODELS.get(canonical) or {}
    try:
        return int(info.get("context_window") or MUSE_SPARK_CONTEXT_WINDOW)
    except (TypeError, ValueError):
        return MUSE_SPARK_CONTEXT_WINDOW


def meta_model_pricing(model_name: Optional[str]) -> dict[str, float]:
    if meta_model_tier(model_name) is MetaPricingTier.CONTRIBUTOR:
        return dict(_CONTRIBUTOR_PRICING)
    return dict(_STANDARD_PRICING)


def meta_model_rate_limits(model_name: Optional[str]) -> dict[str, int]:
    if meta_model_tier(model_name) is MetaPricingTier.CONTRIBUTOR:
        return dict(_CONTRIBUTOR_LIMITS)
    return dict(_STANDARD_LIMITS)


@dataclass(frozen=True)
class MetaTokenUsage:
    """Token accounting from a Chat Completions or Responses payload."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaRateLimitSnapshot:
    """Values from ``x-ratelimit-*`` headers (per-team window)."""

    limit_tokens: Optional[int] = None
    remaining_tokens: Optional[int] = None
    limit_requests: Optional[int] = None
    remaining_requests: Optional[int] = None
    retry_after_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaModelApiFailure:
    """Typed Meta Model API failure (no prompt or credential material)."""

    kind: MetaModelApiErrorKind
    message: str
    status_code: Optional[int] = None
    error_type: str = ""
    error_code: str = ""
    param: str = ""
    retryable: bool = False
    retry_after_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind.value,
            "message": self.message,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "param": self.param,
            "retryable": self.retryable,
        }
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        return payload


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_retry_after(value: Any) -> Optional[float]:
    """Parse ``Retry-After`` as a delay in seconds (integer/float only)."""
    parsed = _as_optional_float(value)
    if parsed is None:
        return None
    return max(0.0, parsed)


def parse_meta_rate_limit_headers(
    headers: Optional[Mapping[str, Any]] = None,
) -> MetaRateLimitSnapshot:
    if not headers:
        return MetaRateLimitSnapshot()
    lowered = {str(key).strip().lower(): value for key, value in headers.items()}
    fields: dict[str, Any] = {}
    for header, field_name in _RATE_LIMIT_HEADER_MAP.items():
        fields[field_name] = _as_optional_int(lowered.get(header))
    retry_after = parse_retry_after(lowered.get("retry-after"))
    return MetaRateLimitSnapshot(retry_after_seconds=retry_after, **fields)


def parse_meta_usage(payload: Optional[Mapping[str, Any]]) -> MetaTokenUsage:
    """Parse Chat Completions or Responses ``usage`` objects."""
    if not isinstance(payload, Mapping):
        return MetaTokenUsage()
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        usage = payload
    prompt = _as_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _as_int(usage.get("completion_tokens", usage.get("output_tokens")))
    total = _as_int(usage.get("total_tokens")) or (prompt + completion)
    cached = 0
    reasoning = 0
    source = ""
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        source = "chat_completions"
        details = usage.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            cached = _as_int(details.get("cached_tokens"))
        out_details = usage.get("completion_tokens_details")
        if isinstance(out_details, Mapping):
            reasoning = _as_int(out_details.get("reasoning_tokens"))
    elif "input_tokens" in usage or "output_tokens" in usage:
        source = "responses"
        details = usage.get("input_tokens_details")
        if isinstance(details, Mapping):
            cached = _as_int(details.get("cached_tokens"))
        out_details = usage.get("output_tokens_details")
        if isinstance(out_details, Mapping):
            reasoning = _as_int(out_details.get("reasoning_tokens"))
    cached = cached or _as_int(usage.get("cached_tokens"))
    reasoning = reasoning or _as_int(usage.get("reasoning_tokens"))
    return MetaTokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        source=source,
    )


def estimate_meta_cost_usd(
    usage: MetaTokenUsage,
    *,
    model_name: Optional[str] = None,
    web_search_queries: int = 0,
) -> float:
    """Estimate billed USD from reported usage (injected tokens already excluded)."""
    pricing = meta_model_pricing(model_name)
    billed_input = max(0, usage.prompt_tokens - usage.cached_tokens)
    cost = (
        (billed_input / 1_000_000.0) * pricing["input"]
        + (max(0, usage.cached_tokens) / 1_000_000.0) * pricing["cached_input"]
        + (max(0, usage.completion_tokens) / 1_000_000.0) * pricing["output"]
        + (max(0, int(web_search_queries)) / 1_000.0) * WEB_SEARCH_USD_PER_1000_QUERIES
    )
    return round(cost, 8)


def _parse_error_envelope(body: str) -> dict[str, Any]:
    text = str(body or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    error = payload.get("error")
    if isinstance(error, Mapping):
        return dict(error)
    return {}


def classify_meta_http_error(
    status_code: Optional[int],
    body: str = "",
    *,
    headers: Optional[Mapping[str, Any]] = None,
    reason: str = "",
) -> MetaModelApiFailure:
    """Classify a Meta Model API HTTP failure from status, envelope, and headers."""
    envelope = _parse_error_envelope(body)
    error_type = str(envelope.get("type") or "").strip()
    error_code = str(envelope.get("code") or "").strip()
    param = str(envelope.get("param") or "").strip()
    message = str(envelope.get("message") or body or reason or "Meta Model API request failed").strip()
    snapshot = parse_meta_rate_limit_headers(headers)
    retry_after = snapshot.retry_after_seconds
    if retry_after is None:
        match = _RETRY_AFTER_RE.search(message)
        if match:
            retry_after = parse_retry_after(match.group(1))

    kind = MetaModelApiErrorKind.TRANSPORT
    code = int(status_code) if status_code is not None else None
    lowered_type = error_type.lower()
    lowered_code = error_code.lower()
    lowered_message = message.lower()

    if code == 401 or lowered_type == "authentication_error" or lowered_code == "invalid_api_key":
        kind = MetaModelApiErrorKind.AUTHENTICATION
    elif code == 402 or lowered_type == "billing_error" or lowered_code == "billing_not_configured":
        kind = MetaModelApiErrorKind.BILLING
    elif code == 403:
        kind = MetaModelApiErrorKind.FORBIDDEN
    elif code == 404 or lowered_code in {"model_not_found", "file_not_found"}:
        kind = MetaModelApiErrorKind.NOT_FOUND
    elif code == 413 or lowered_code == "payload_too_large":
        kind = MetaModelApiErrorKind.PAYLOAD_TOO_LARGE
    elif (
        code == 429
        or lowered_type == "rate_limit_error"
        or lowered_code == "rate_limit_exceeded"
    ):
        kind = MetaModelApiErrorKind.RATE_LIMIT
    elif code == 504 or lowered_code == "gateway_timeout":
        kind = MetaModelApiErrorKind.GATEWAY_TIMEOUT
    elif code == 503 or lowered_code in {
        "server_shutting_down",
        "service_overloaded",
        "backend_unavailable",
    }:
        kind = MetaModelApiErrorKind.SERVICE_UNAVAILABLE
    elif code == 500 or lowered_type == "server_error":
        kind = MetaModelApiErrorKind.SERVER
    elif lowered_code == "content_policy_violation":
        kind = MetaModelApiErrorKind.CONTENT_POLICY
    elif code == 400 and _CONTEXT_WINDOW_RE.search(message):
        kind = MetaModelApiErrorKind.CONTEXT_WINDOW
    elif code == 400 or lowered_type == "invalid_request_error":
        kind = MetaModelApiErrorKind.INVALID_REQUEST
    elif "rate limit" in lowered_message or "too many requests" in lowered_message:
        kind = MetaModelApiErrorKind.RATE_LIMIT
    elif "insufficient" in lowered_message and "quota" in lowered_message:
        kind = MetaModelApiErrorKind.BILLING
    elif "unauthorized" in lowered_message or "invalid_api_key" in lowered_message:
        kind = MetaModelApiErrorKind.AUTHENTICATION

    retryable = kind in _RETRYABLE_KINDS
    return MetaModelApiFailure(
        kind=kind,
        message=message[:4096],
        status_code=code,
        error_type=error_type,
        error_code=error_code,
        param=param,
        retryable=retryable,
        retry_after_seconds=retry_after,
    )


def classify_meta_diagnostic_text(text: str) -> MetaModelApiFailure:
    """Classify CLI/text diagnostics when there is no HTTP envelope."""
    body = str(text or "")
    status = None
    match = re.search(r"\bHTTP\s+(\d{3})\b", body, re.IGNORECASE)
    if match:
        status = int(match.group(1))
    return classify_meta_http_error(status, body, reason=body or "Meta Model API failure")


def meta_error_is_retryable(kind: MetaModelApiErrorKind | str) -> bool:
    if isinstance(kind, MetaModelApiErrorKind):
        return kind in _RETRYABLE_KINDS
    try:
        return MetaModelApiErrorKind(str(kind)) in _RETRYABLE_KINDS
    except ValueError:
        return False


def meta_retry_delay_seconds(
    attempt: int,
    *,
    retry_after: Optional[float] = None,
    base: float = META_RETRY_BASE_SECONDS,
    rng: Optional[random.Random] = None,
) -> float:
    """Exponential backoff with jitter. ``Retry-After`` wins when present."""
    if retry_after is not None:
        return max(0.0, float(retry_after))
    jitter = (rng.random() if rng is not None else random.random())
    return (float(base) * (2 ** max(0, int(attempt)))) + jitter


def format_meta_http_error(failure: MetaModelApiFailure) -> str:
    """Stable message prefix consumed by CLI quota classifiers."""
    code = failure.status_code if failure.status_code is not None else "error"
    return f"Meta AI HTTP {code}: {failure.message}"


def record_meta_observation(
    *,
    model_name: Optional[str] = None,
    usage: Optional[MetaTokenUsage] = None,
    rate_limit: Optional[MetaRateLimitSnapshot] = None,
    failure: Optional[MetaModelApiFailure] = None,
    estimated_cost_usd: Optional[float] = None,
) -> dict[str, Any]:
    """Store a thread-local observation (no prompts, keys, or raw endpoints)."""
    payload: dict[str, Any] = {}
    if model_name:
        payload["model"] = normalize_meta_model_name(model_name)
        payload["tier"] = meta_model_tier(model_name).value
        payload["context_window"] = meta_model_context_window(model_name)
        payload["rate_limits"] = meta_model_rate_limits(model_name)
        payload["pricing_per_1m"] = meta_model_pricing(model_name)
    if usage is not None:
        payload["usage"] = usage.to_dict()
        if estimated_cost_usd is None and (usage.total_tokens or usage.prompt_tokens):
            estimated_cost_usd = estimate_meta_cost_usd(usage, model_name=model_name)
    if estimated_cost_usd is not None:
        payload["estimated_cost_usd"] = estimated_cost_usd
    if rate_limit is not None:
        payload["rate_limit"] = rate_limit.to_dict()
    if failure is not None:
        payload["failure"] = failure.to_dict()
    _LAST_OBSERVATION.payload = payload
    return dict(payload)


def get_last_meta_observation() -> dict[str, Any]:
    payload = getattr(_LAST_OBSERVATION, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def clear_last_meta_observation() -> None:
    _LAST_OBSERVATION.payload = {}


def raise_meta_http_error(
    status_code: Optional[int],
    body: str = "",
    *,
    headers: Optional[Mapping[str, Any]] = None,
    reason: str = "",
    model_name: Optional[str] = None,
) -> None:
    """Raise :class:`RuntimeError` with Meta classification attributes attached."""
    failure = classify_meta_http_error(
        status_code, body, headers=headers, reason=reason
    )
    record_meta_observation(model_name=model_name, failure=failure)
    error = RuntimeError(format_meta_http_error(failure))
    setattr(error, "meta_error_kind", failure.kind)
    setattr(error, "retryable", failure.retryable)
    setattr(error, "status_code", failure.status_code)
    setattr(error, "error_type", failure.error_type)
    setattr(error, "error_code", failure.error_code)
    setattr(error, "retry_after_seconds", failure.retry_after_seconds)
    raise error


def meta_http_should_retry(
    exc: BaseException,
    *,
    attempt: int,
    max_attempts: int = META_RETRY_MAX_ATTEMPTS,
) -> bool:
    if attempt >= max_attempts - 1:
        return False
    retryable = getattr(exc, "retryable", None)
    if retryable is True:
        return True
    kind = getattr(exc, "meta_error_kind", None)
    if isinstance(kind, MetaModelApiErrorKind):
        return meta_error_is_retryable(kind)
    text = str(exc)
    failure = classify_meta_diagnostic_text(text)
    return failure.retryable


def sleep_for_meta_retry(
    exc: BaseException,
    attempt: int,
    *,
    sleep_fn: Optional[_SleepFn] = None,
) -> None:
    retry_after = getattr(exc, "retry_after_seconds", None)
    delay = meta_retry_delay_seconds(attempt, retry_after=retry_after)
    sleeper = sleep_fn or time.sleep
    sleeper(delay)


__all__ = [
    "BACKGROUND_SUBMISSIONS_PER_MINUTE",
    "META_MODEL_API_BASE_URL",
    "META_MODEL_API_DEFAULT_MODEL",
    "META_MODEL_API_ENV_VARS",
    "META_MODEL_API_RECOMMENDED_MODEL",
    "META_MODEL_API_SECRET_NAME",
    "META_RETRY_BASE_SECONDS",
    "META_RETRY_MAX_ATTEMPTS",
    "MUSE_IMAGE_RPM",
    "MUSE_SPARK_CONTEXT_WINDOW",
    "MUSE_SPARK_MAX_OUTPUT_TOKENS",
    "MUSE_SPARK_MODELS",
    "MUSE_VOICE_CONCURRENT_STREAMS",
    "MUSE_VOICE_STREAMS_PER_HOUR",
    "MUSE_VOICE_USD_PER_HOUR",
    "WEB_SEARCH_USD_PER_1000_QUERIES",
    "MetaModelApiErrorKind",
    "MetaModelApiFailure",
    "MetaPricingTier",
    "MetaRateLimitSnapshot",
    "MetaTokenUsage",
    "classify_meta_diagnostic_text",
    "classify_meta_http_error",
    "clear_last_meta_observation",
    "estimate_meta_cost_usd",
    "format_meta_http_error",
    "get_last_meta_observation",
    "meta_error_is_retryable",
    "meta_http_should_retry",
    "meta_model_api_key_fingerprint",
    "meta_model_context_window",
    "meta_model_pricing",
    "meta_model_rate_limits",
    "meta_model_tier",
    "meta_retry_delay_seconds",
    "normalize_meta_model_name",
    "parse_meta_rate_limit_headers",
    "parse_meta_usage",
    "parse_retry_after",
    "raise_meta_http_error",
    "record_meta_observation",
    "resolve_meta_model_api_key",
    "sleep_for_meta_retry",
]
