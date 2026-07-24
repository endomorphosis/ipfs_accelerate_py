"""Abby IndexTTS and Hugging Face Whisper HTTP providers.

This module ports the remote-provider behavior formerly embedded in the Abby
wallet UI into reusable, synchronous ``VoiceProvider`` adapters.  It has no
optional model or UI imports and performs no work at import time.  All network
I/O is behind an injectable transport so routing, retry, timeout, and circuit
behavior can be tested offline.

The adapters deliberately retain the public router's small return contract:
TTS returns bytes and STT returns text.  ``last_receipt`` provides a
JSON-serializable, privacy-safe record of remote attempts; the voice router
copies it into stage traces and remains responsible for ordered remote/local
fallback and the final degraded turn receipt.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429})
_AUDIO_CONTENT_TYPES = (
    "audio/",
    "application/octet-stream",
    "binary/octet-stream",
)
_SECRET_PATTERN = re.compile(
    r"(?i)"
    r"(authorization\s*:\s*bearer\s+|bearer\s+|"
    r"(?:api[_-]?key|token|authorization|secret)\s*[=:]\s*)"
    r"[^\s,;\"']+"
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api[_-]?key|token|access_token|secret)=)[^&#\s]+"
)


def _safe_error_text(
    value: object,
    *,
    limit: int = 240,
    sensitive_values: Sequence[object] = (),
) -> str:
    """Return a bounded error string with common credential forms redacted."""
    message = " ".join(str(value or "").replace("\x00", "").split())
    message = _SECRET_PATTERN.sub(lambda match: match.group(1) + "[redacted]", message)
    message = _QUERY_SECRET_PATTERN.sub(r"\1[redacted]", message)
    for sensitive in sensitive_values:
        if isinstance(sensitive, bytes):
            sample = (
                sensitive
                if len(sensitive) <= 8192
                else sensitive[:4096] + sensitive[-4096:]
            )
            decoded = sample.decode("utf-8", errors="ignore")
            fragments = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{7,}", decoded)
        else:
            decoded = str(sensitive or "").strip()
            fragments = [decoded] if len(decoded) >= 8 else []
        for fragment in sorted(set(fragments), key=len, reverse=True):
            message = message.replace(fragment, "[redacted-input]")
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _normalized_urls(values: Sequence[str]) -> Tuple[str, ...]:
    urls = []
    for value in values:
        normalized = str(value or "").strip().rstrip("/")
        if normalized and normalized not in urls:
            urls.append(normalized)
    return tuple(urls)


def _split_urls(value: str) -> Tuple[str, ...]:
    return _normalized_urls(re.split(r"[\s,]+", str(value or "")))


def _env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class HTTPRequest:
    """A transport-neutral HTTP request."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None

    def __post_init__(self) -> None:
        method = str(self.method or "").strip().upper()
        url = str(self.url or "").strip()
        if not method:
            raise ValueError("HTTPRequest.method must be non-empty")
        if not url:
            raise ValueError("HTTPRequest.url must be non-empty")
        if self.body is not None and not isinstance(self.body, bytes):
            raise TypeError("HTTPRequest.body must be bytes or None")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "url", url)
        object.__setattr__(
            self,
            "headers",
            MappingProxyType(
                {str(key): str(value) for key, value in dict(self.headers).items()}
            ),
        )


@dataclass(frozen=True)
class HTTPResponse:
    """A transport-neutral HTTP response."""

    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.body, bytes):
            raise TypeError("HTTPResponse.body must be bytes")
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(
            self,
            "headers",
            MappingProxyType(
                {
                    str(key).lower(): str(value)
                    for key, value in dict(self.headers).items()
                }
            ),
        )


HTTPTransport = Callable[[HTTPRequest, float], HTTPResponse]
Sleeper = Callable[[float], None]
Clock = Callable[[], float]


def _urllib_transport(request: HTTPRequest, timeout_seconds: float) -> HTTPResponse:
    """Execute an :class:`HTTPRequest` with the Python standard library."""
    wire_request = urllib.request.Request(
        request.url,
        data=request.body,
        headers=dict(request.headers),
        method=request.method,
    )
    try:
        with urllib.request.urlopen(wire_request, timeout=timeout_seconds) as response:
            return HTTPResponse(
                status=int(getattr(response, "status", response.getcode())),
                body=response.read(),
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as error:
        return HTTPResponse(
            status=int(error.code),
            body=error.read(),
            headers=dict(error.headers.items()) if error.headers is not None else {},
        )


@dataclass(frozen=True)
class AbbyResiliencePolicy:
    """Bounded remote-call retry and circuit-breaker settings."""

    timeout_seconds: float = 45.0
    max_retries: int = 1
    backoff_seconds: float = 0.2
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 2.0
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        numeric = {
            "timeout_seconds": float(self.timeout_seconds),
            "backoff_seconds": float(self.backoff_seconds),
            "backoff_multiplier": float(self.backoff_multiplier),
            "max_backoff_seconds": float(self.max_backoff_seconds),
            "circuit_recovery_seconds": float(self.circuit_recovery_seconds),
        }
        if numeric["timeout_seconds"] <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= int(self.max_retries) <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        if numeric["backoff_seconds"] < 0 or numeric["max_backoff_seconds"] < 0:
            raise ValueError("backoff values must be non-negative")
        if numeric["backoff_multiplier"] < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if int(self.circuit_failure_threshold) < 1:
            raise ValueError("circuit_failure_threshold must be at least 1")
        if numeric["circuit_recovery_seconds"] < 0:
            raise ValueError("circuit_recovery_seconds must be non-negative")
        for name, value in numeric.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "max_retries", int(self.max_retries))
        object.__setattr__(
            self, "circuit_failure_threshold", int(self.circuit_failure_threshold)
        )

    @classmethod
    def from_environment(
        cls, *, operation: str, default_timeout: float
    ) -> "AbbyResiliencePolicy":
        """Build policy from common and operation-specific Abby settings."""
        prefix = f"IPFS_ACCELERATE_PY_ABBY_{operation.upper()}"

        def selected(suffix: str, default: str) -> str:
            return os.getenv(
                f"{prefix}_{suffix}",
                os.getenv(f"IPFS_ACCELERATE_PY_ABBY_{suffix}", default),
            )

        try:
            timeout = max(0.001, float(selected("TIMEOUT_SECONDS", str(default_timeout))))
        except ValueError:
            timeout = default_timeout
        try:
            retries = max(0, min(10, int(selected("MAX_RETRIES", "1"))))
        except ValueError:
            retries = 1
        return cls(
            timeout_seconds=timeout,
            max_retries=retries,
            backoff_seconds=_env_float(
                "IPFS_ACCELERATE_PY_ABBY_RETRY_BACKOFF_SECONDS", 0.2, minimum=0.0
            ),
            backoff_multiplier=_env_float(
                "IPFS_ACCELERATE_PY_ABBY_RETRY_BACKOFF_MULTIPLIER",
                2.0,
                minimum=1.0,
            ),
            max_backoff_seconds=_env_float(
                "IPFS_ACCELERATE_PY_ABBY_RETRY_MAX_BACKOFF_SECONDS",
                2.0,
                minimum=0.0,
            ),
            circuit_failure_threshold=_env_int(
                "IPFS_ACCELERATE_PY_ABBY_CIRCUIT_FAILURE_THRESHOLD",
                3,
                minimum=1,
                maximum=100,
            ),
            circuit_recovery_seconds=_env_float(
                "IPFS_ACCELERATE_PY_ABBY_CIRCUIT_RECOVERY_SECONDS",
                30.0,
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class AbbyProviderAttempt:
    """One privacy-safe remote attempt."""

    endpoint: str
    attempt: int
    status: str
    duration_ms: float
    http_status: Optional[int] = None
    retryable: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "attempt": self.attempt,
            "status": self.status,
            "duration_ms": round(max(0.0, self.duration_ms), 3),
            "http_status": self.http_status,
            "retryable": self.retryable,
            "error": self.error,
        }


@dataclass(frozen=True)
class AbbyProviderReceipt:
    """Structured receipt for the last adapter call."""

    provider: str
    operation: str
    status: str
    selected_endpoint: Optional[str] = None
    attempts: Tuple[AbbyProviderAttempt, ...] = ()
    error_code: Optional[str] = None
    retryable: bool = False

    @property
    def degraded(self) -> bool:
        return self.status != "completed"

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "status": self.status,
            "degraded": self.degraded,
            "selected_endpoint": self.selected_endpoint,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "error_code": self.error_code,
            "retryable": self.retryable,
        }


class AbbyProviderError(RuntimeError):
    """Normalized Abby provider failure safe to include in router traces."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "abby_provider_error",
        retryable: bool = False,
        http_status: Optional[int] = None,
        receipt: Optional[AbbyProviderReceipt] = None,
    ) -> None:
        super().__init__(_safe_error_text(message))
        self.code = str(code)
        self.retryable = bool(retryable)
        self.http_status = int(http_status) if http_status is not None else None
        self.receipt = receipt


class AbbyCircuitOpenError(AbbyProviderError):
    """Raised when an endpoint circuit rejects a call without network I/O."""

    def __init__(self, endpoint: str) -> None:
        super().__init__(
            f"circuit breaker is open for {_safe_endpoint(endpoint)}",
            code="circuit_open",
            retryable=True,
        )


def _safe_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(str(endpoint))
    if parsed.scheme and parsed.netloc:
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, "", "")
        )
    return _safe_error_text(endpoint, limit=160)


class _CircuitBreaker:
    def __init__(self, policy: AbbyResiliencePolicy, clock: Clock) -> None:
        self.policy = policy
        self.clock = clock
        self._state = "closed"
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def before_call(self, endpoint: str) -> None:
        with self._lock:
            if self._state == "open":
                elapsed = self.clock() - float(self._opened_at or 0.0)
                if elapsed < self.policy.circuit_recovery_seconds:
                    raise AbbyCircuitOpenError(endpoint)
                self._state = "half_open"
            if self._state == "half_open":
                if self._probe_in_flight:
                    raise AbbyCircuitOpenError(endpoint)
                self._probe_in_flight = True

    def success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def failure(self, *, retryable: bool) -> None:
        with self._lock:
            was_half_open = self._state == "half_open"
            self._probe_in_flight = False
            if not retryable and not was_half_open:
                return
            self._failures += 1
            if was_half_open or self._failures >= self.policy.circuit_failure_threshold:
                self._state = "open"
                self._opened_at = self.clock()

    def reset(self) -> None:
        self.success()


def _status_retryable(status: int) -> bool:
    return status in _TRANSIENT_HTTP_STATUSES or 500 <= status <= 599


def _exception_retryable(error: BaseException) -> bool:
    if isinstance(error, AbbyProviderError):
        return error.retryable
    return isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            OSError,
        ),
    )


def _transport_response(
    transport: HTTPTransport,
    request: HTTPRequest,
    timeout_seconds: float,
) -> HTTPResponse:
    response = transport(request, timeout_seconds)
    if hasattr(response, "__await__"):
        close = getattr(response, "close", None)
        if callable(close):
            close()
        raise TypeError("Abby HTTP transport must be synchronous")
    if not isinstance(response, HTTPResponse):
        raise TypeError("Abby HTTP transport must return HTTPResponse")
    return response


class _ResilientHTTPProvider:
    provider_name = "abby_http"

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        policy: AbbyResiliencePolicy,
        transport: Optional[HTTPTransport] = None,
        sleeper: Sleeper = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self.endpoints = _normalized_urls(endpoints)
        self.policy = policy
        self._transport = transport or _urllib_transport
        self._sleeper = sleeper
        self._clock = clock
        self._circuits = {
            endpoint: _CircuitBreaker(policy, clock) for endpoint in self.endpoints
        }
        self.last_receipt: Optional[AbbyProviderReceipt] = None

    @property
    def cache_identity(self) -> str:
        payload = (
            self.provider_name,
            self.endpoints,
            self.policy,
            self._configuration_identity(),
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def _configuration_identity(self) -> object:
        return None

    def circuit_state(self, endpoint: Optional[str] = None) -> str:
        selected = str(endpoint or (self.endpoints[0] if self.endpoints else ""))
        breaker = self._circuits.get(selected)
        return breaker.state if breaker is not None else "unconfigured"

    def reset_circuits(self) -> None:
        for breaker in self._circuits.values():
            breaker.reset()

    def _execute(
        self,
        operation: str,
        request_factory: Callable[[str], HTTPRequest],
        response_parser: Callable[[HTTPResponse, str], object],
        *,
        sensitive_values: Sequence[object] = (),
    ) -> object:
        attempts = []
        if not self.endpoints:
            receipt = AbbyProviderReceipt(
                provider=self.provider_name,
                operation=operation,
                status="failed",
                error_code="provider_not_configured",
            )
            self.last_receipt = receipt
            raise AbbyProviderError(
                f"{self.provider_name} has no configured endpoint",
                code="provider_not_configured",
                receipt=receipt,
            )

        last_error: Optional[AbbyProviderError] = None
        for endpoint in self.endpoints:
            breaker = self._circuits[endpoint]
            try:
                breaker.before_call(endpoint)
            except AbbyProviderError as error:
                attempts.append(
                    AbbyProviderAttempt(
                        endpoint=_safe_endpoint(endpoint),
                        attempt=0,
                        status="circuit_open",
                        duration_ms=0.0,
                        retryable=True,
                        error=str(error),
                    )
                )
                last_error = error
                continue

            endpoint_error: Optional[AbbyProviderError] = None
            for attempt_index in range(self.policy.max_retries + 1):
                started_at = self._clock()
                http_status: Optional[int] = None
                try:
                    response = _transport_response(
                        self._transport,
                        request_factory(endpoint),
                        self.policy.timeout_seconds,
                    )
                    http_status = response.status
                    if not 200 <= response.status <= 299:
                        retryable = _status_retryable(response.status)
                        raise AbbyProviderError(
                            f"{self.provider_name} returned HTTP {response.status}",
                            code="remote_http_error",
                            retryable=retryable,
                            http_status=response.status,
                        )
                    result = response_parser(response, endpoint)
                    attempts.append(
                        AbbyProviderAttempt(
                            endpoint=_safe_endpoint(endpoint),
                            attempt=attempt_index + 1,
                            status="succeeded",
                            duration_ms=(self._clock() - started_at) * 1000.0,
                            http_status=response.status,
                        )
                    )
                    breaker.success()
                    call_status = (
                        "degraded"
                        if any(item.status != "succeeded" for item in attempts[:-1])
                        else "completed"
                    )
                    self.last_receipt = AbbyProviderReceipt(
                        provider=self.provider_name,
                        operation=operation,
                        status=call_status,
                        selected_endpoint=_safe_endpoint(endpoint),
                        attempts=tuple(attempts),
                    )
                    return result
                except Exception as raw_error:
                    retryable = _exception_retryable(raw_error)
                    endpoint_error = (
                        raw_error
                        if isinstance(raw_error, AbbyProviderError)
                        else AbbyProviderError(
                            _safe_error_text(
                                str(raw_error) or raw_error.__class__.__name__,
                                sensitive_values=sensitive_values,
                            ),
                            code="remote_transport_error"
                            if retryable
                            else "invalid_remote_response",
                            retryable=retryable,
                        )
                    )
                    attempts.append(
                        AbbyProviderAttempt(
                            endpoint=_safe_endpoint(endpoint),
                            attempt=attempt_index + 1,
                            status="failed",
                            duration_ms=(self._clock() - started_at) * 1000.0,
                            http_status=http_status or endpoint_error.http_status,
                            retryable=retryable,
                            error=_safe_error_text(
                                endpoint_error,
                                sensitive_values=sensitive_values,
                            ),
                        )
                    )
                    if not retryable or attempt_index >= self.policy.max_retries:
                        break
                    delay = min(
                        self.policy.max_backoff_seconds,
                        self.policy.backoff_seconds
                        * (self.policy.backoff_multiplier ** attempt_index),
                    )
                    if delay > 0:
                        self._sleeper(delay)
            if endpoint_error is not None:
                breaker.failure(retryable=endpoint_error.retryable)
                last_error = endpoint_error

        code = last_error.code if last_error is not None else "provider_failed"
        retryable = bool(last_error and last_error.retryable)
        receipt = AbbyProviderReceipt(
            provider=self.provider_name,
            operation=operation,
            status="degraded",
            attempts=tuple(attempts),
            error_code=code,
            retryable=retryable,
        )
        self.last_receipt = receipt
        raise AbbyProviderError(
            f"{self.provider_name} failed across configured endpoints",
            code=code,
            retryable=retryable,
            receipt=receipt,
        ) from last_error


def _json_mapping(response: HTTPResponse, *, provider: str) -> Mapping[str, object]:
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AbbyProviderError(
            f"{provider} returned invalid JSON",
            code="invalid_remote_response",
        ) from error
    if not isinstance(value, Mapping):
        raise AbbyProviderError(
            f"{provider} returned a non-object JSON response",
            code="invalid_remote_response",
        )
    return value


def _nested_value(
    value: object, keys: Sequence[str], *, max_depth: int = 8
) -> Optional[object]:
    if max_depth < 0:
        return None
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if candidate is not None:
                return candidate
        for container in (
            "data",
            "result",
            "output",
            "response",
            "items",
            "results",
            "segments",
            "chunks",
        ):
            if container in value:
                candidate = _nested_value(
                    value[container], keys, max_depth=max_depth - 1
                )
                if candidate is not None:
                    return candidate
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            candidate = _nested_value(item, keys, max_depth=max_depth - 1)
            if candidate is not None:
                return candidate
    return None


def _looks_like_audio(audio: bytes) -> bool:
    return bool(audio) and (
        audio.startswith(b"RIFF")
        and audio[8:12] == b"WAVE"
        or audio.startswith(b"ID3")
        or audio[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}
        or audio.startswith(b"OggS")
        or audio.startswith(b"fLaC")
    )


class IndexTTSHTTPProvider(_ResilientHTTPProvider):
    """Synthesis-only adapter for Abby IndexTTS HTTP endpoints."""

    provider_name = "abby_indextts"

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        token: Optional[str] = None,
        bill_to: Optional[str] = None,
        default_model: str = "Publicus/IndexTTS-2-Demo",
        policy: Optional[AbbyResiliencePolicy] = None,
        transport: Optional[HTTPTransport] = None,
        sleeper: Sleeper = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        super().__init__(
            endpoints,
            policy=policy
            or AbbyResiliencePolicy.from_environment(
                operation="indextts", default_timeout=45.0
            ),
            transport=transport,
            sleeper=sleeper,
            clock=clock,
        )
        self._token = str(token or "").strip()
        self._bill_to = str(bill_to or "").strip()
        self.default_model = str(default_model or "").strip()

    @classmethod
    def from_environment(
        cls, **overrides: object
    ) -> "IndexTTSHTTPProvider":
        configured = _split_urls(
            os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URLS", "")
        )
        if not configured:
            configured = _normalized_urls(
                (
                    os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URL", ""),
                    os.getenv("WALLET_INDEXTTS_SPACE_URL", ""),
                    os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_FALLBACK_URL", ""),
                    os.getenv("WALLET_INDEXTTS_FALLBACK_SPACE_URL", ""),
                )
            )
        values: Dict[str, object] = {
            "endpoints": configured,
            "token": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TOKEN",
                os.getenv("HF_TOKEN", ""),
            ),
            "bill_to": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_HF_BILL_TO",
                os.getenv("IPFS_DATASETS_PY_HF_BILL_TO", ""),
            ),
            "default_model": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_MODEL",
                os.getenv("WALLET_INDEXTTS_MODEL_NAME", "Publicus/IndexTTS-2-Demo"),
            ),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def _configuration_identity(self) -> object:
        token_digest = (
            hashlib.sha256(self._token.encode("utf-8")).hexdigest()[:12]
            if self._token
            else ""
        )
        return (self.default_model, token_digest, self._bill_to)

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs: object,
    ) -> bytes:
        return _run_indextts_gradio_tts(
            text=text,
            provider=self,
            voice=voice,
            model_name=model_name,
            device=device,
            output_format=output_format,
            **kwargs,
        )

    def _synthesize(
        self,
        text: str,
        *,
        voice: Optional[str],
        model_name: Optional[str],
        output_format: Optional[str],
        options: Mapping[str, object],
    ) -> bytes:
        prompt = str(text or "").strip()
        if not prompt:
            raise ValueError("text is required")
        payload: Dict[str, object] = {
            "text": prompt,
            "model": model_name or self.default_model,
            "output_format": (output_format or "wav").lower().lstrip("."),
        }
        if voice:
            payload["voice"] = voice
        for key, value in options.items():
            if str(key) not in {"text", "model", "voice", "output_format"}:
                payload[str(key)] = value
        body = json.dumps(payload, separators=(",", ":"), default=repr).encode("utf-8")

        def request_factory(endpoint: str) -> HTTPRequest:
            headers = {
                "Accept": "audio/*, application/json",
                "Content-Type": "application/json",
            }
            if self._token:
                headers["Authorization"] = "Bearer " + self._token
            if self._bill_to:
                headers["X-HF-Bill-To"] = self._bill_to
            return HTTPRequest("POST", endpoint, headers, body)

        def response_parser(response: HTTPResponse, endpoint: str) -> bytes:
            content_type = response.headers.get("content-type", "").lower()
            if any(content_type.startswith(kind) for kind in _AUDIO_CONTENT_TYPES):
                if not response.body:
                    raise AbbyProviderError(
                        "IndexTTS returned empty audio",
                        code="invalid_remote_response",
                    )
                return response.body
            value = _json_mapping(response, provider="IndexTTS")
            encoded = _nested_value(
                value,
                (
                    "audioBase64",
                    "audio_base64",
                    "audio",
                    "bytes",
                    "content",
                ),
            )
            if isinstance(encoded, Mapping):
                encoded = _nested_value(encoded, ("base64", "data", "content"))
            if isinstance(encoded, str):
                if encoded.startswith("data:") and "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                try:
                    audio = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise AbbyProviderError(
                        "IndexTTS returned invalid base64 audio",
                        code="invalid_remote_response",
                    ) from error
                if not audio:
                    raise AbbyProviderError(
                        "IndexTTS returned empty audio",
                        code="invalid_remote_response",
                    )
                return audio
            audio_url = _nested_value(
                value, ("audioUrl", "audio_url", "download_url", "url")
            )
            if isinstance(audio_url, str) and audio_url.strip():
                resolved = urllib.parse.urljoin(endpoint + "/", audio_url.strip())
                source = urllib.parse.urlsplit(endpoint)
                target = urllib.parse.urlsplit(resolved)
                if target.scheme not in {"http", "https"} or target.netloc != source.netloc:
                    raise AbbyProviderError(
                        "IndexTTS returned an unsafe audio URL",
                        code="invalid_remote_response",
                    )
                download = _transport_response(
                    self._transport,
                    HTTPRequest(
                        "GET",
                        resolved,
                        {
                            "Accept": "audio/*, application/octet-stream",
                            **(
                                {"Authorization": "Bearer " + self._token}
                                if self._token
                                else {}
                            ),
                        },
                    ),
                    self.policy.timeout_seconds,
                )
                if not 200 <= download.status <= 299 or not download.body:
                    raise AbbyProviderError(
                        f"IndexTTS audio download returned HTTP {download.status}",
                        code="remote_http_error",
                        retryable=_status_retryable(download.status),
                        http_status=download.status,
                    )
                return download.body
            raise AbbyProviderError(
                "IndexTTS response did not contain audio",
                code="invalid_remote_response",
            )

        result = self._execute(
            "synthesis",
            request_factory,
            response_parser,
            sensitive_values=(prompt,),
        )
        if not isinstance(result, bytes) or not result:
            raise AbbyProviderError(
                "IndexTTS returned invalid audio",
                code="invalid_remote_response",
            )
        return result

    def transcribe(
        self,
        audio: Union[str, bytes],
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        raise NotImplementedError("IndexTTS does not support transcription")


def _run_indextts_gradio_tts(
    *,
    text: str,
    provider: IndexTTSHTTPProvider,
    voice: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    output_format: Optional[str] = None,
    **kwargs: object,
) -> bytes:
    """Run IndexTTS with ordered endpoint and resilience handling.

    The historical function name is retained as objective evidence and as a
    narrow migration seam for callers moving out of wallet-specific helpers.
    """
    _ = device
    return provider._synthesize(
        text,
        voice=voice,
        model_name=model_name,
        output_format=output_format,
        options=kwargs,
    )


def _read_audio(audio: Union[str, bytes]) -> Tuple[bytes, Optional[str]]:
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("audio is required")
        return audio, None
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError("audio is required")
    path = os.path.abspath(audio)
    try:
        with open(path, "rb") as input_file:
            data = input_file.read()
    except OSError as error:
        raise ValueError("audio path is not readable") from error
    if not data:
        raise ValueError("audio file is empty")
    return data, path


class HuggingFaceWhisperHTTPProvider(_ResilientHTTPProvider):
    """Transcription-only adapter for Hugging Face Whisper HTTP inference."""

    provider_name = "abby_whisper"

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        token: Optional[str] = None,
        bill_to: Optional[str] = None,
        default_model: str = "openai/whisper-large-v3-turbo",
        policy: Optional[AbbyResiliencePolicy] = None,
        transport: Optional[HTTPTransport] = None,
        sleeper: Sleeper = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        super().__init__(
            endpoints,
            policy=policy
            or AbbyResiliencePolicy.from_environment(
                operation="whisper", default_timeout=45.0
            ),
            transport=transport,
            sleeper=sleeper,
            clock=clock,
        )
        self._token = str(token or "").strip()
        self._bill_to = str(bill_to or "").strip()
        self.default_model = str(default_model or "").strip()

    @classmethod
    def from_environment(
        cls, **overrides: object
    ) -> "HuggingFaceWhisperHTTPProvider":
        configured = _split_urls(
            os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_URLS", "")
        )
        if not configured:
            base = os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_WHISPER_BASE_URL",
                os.getenv(
                    "WALLET_HF_WHISPER_BASE_URL",
                    "https://router.huggingface.co/hf-inference/models",
                ),
            )
            configured = _normalized_urls((base,))
        values: Dict[str, object] = {
            "endpoints": configured,
            "token": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_WHISPER_TOKEN",
                os.getenv("WALLET_HF_WHISPER_TOKEN", os.getenv("HF_TOKEN", "")),
            ),
            "bill_to": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_HF_BILL_TO",
                os.getenv("IPFS_DATASETS_PY_HF_BILL_TO", ""),
            ),
            "default_model": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_WHISPER_MODEL",
                os.getenv(
                    "WALLET_HF_WHISPER_MODEL_NAME",
                    "openai/whisper-large-v3-turbo",
                ),
            ),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def _configuration_identity(self) -> object:
        token_digest = (
            hashlib.sha256(self._token.encode("utf-8")).hexdigest()[:12]
            if self._token
            else ""
        )
        return (self.default_model, token_digest, self._bill_to)

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs: object,
    ) -> bytes:
        raise NotImplementedError("Whisper does not support synthesis")

    def transcribe(
        self,
        audio: Union[str, bytes],
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        _ = device
        return _run_hf_whisper_stt(
            audio,
            provider=self,
            model_name=model_name,
            language=language,
            **kwargs,
        )

    def _transcribe(
        self,
        audio: Union[str, bytes],
        *,
        model_name: Optional[str],
        language: Optional[str],
        content_type: Optional[str],
        options: Mapping[str, object],
    ) -> str:
        audio_bytes, path = _read_audio(audio)
        selected_model = str(model_name or self.default_model).strip()
        if not selected_model:
            raise ValueError("Whisper model name is required")
        selected_content_type = str(
            content_type
            or mimetypes.guess_type(path or "")[0]
            or "audio/wav"
        ).strip()

        def request_factory(endpoint: str) -> HTTPRequest:
            encoded_model = urllib.parse.quote(selected_model, safe="/")
            url = endpoint
            if not endpoint.rstrip("/").endswith(encoded_model):
                url = endpoint.rstrip("/") + "/" + encoded_model
            headers = {
                "Accept": "application/json",
                "Content-Type": selected_content_type,
            }
            if self._token:
                headers["Authorization"] = "Bearer " + self._token
            if self._bill_to:
                headers["X-HF-Bill-To"] = self._bill_to
            if language:
                headers["X-Wallet-STT-Language"] = str(language)
            for key, value in options.items():
                if str(key).lower().startswith("header_"):
                    headers[str(key)[7:].replace("_", "-")] = str(value)
            return HTTPRequest("POST", url, headers, audio_bytes)

        def response_parser(response: HTTPResponse, endpoint: str) -> str:
            _ = endpoint
            value = _json_mapping(response, provider="Whisper")
            text = _nested_value(
                value,
                (
                    "text",
                    "transcription",
                    "transcript",
                    "generated_text",
                    "output_text",
                ),
            )
            if isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
                parts = []
                for item in text:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                    elif isinstance(item, Mapping):
                        nested = _nested_value(
                            item,
                            (
                                "text",
                                "transcription",
                                "transcript",
                                "generated_text",
                            ),
                        )
                        if isinstance(nested, str) and nested.strip():
                            parts.append(nested.strip())
                text = " ".join(parts)
            if not isinstance(text, str) or not text.strip():
                raise AbbyProviderError(
                    "Whisper response did not contain transcription text",
                    code="invalid_remote_response",
                )
            return text.strip()

        result = self._execute(
            "transcription",
            request_factory,
            response_parser,
            sensitive_values=(audio_bytes,),
        )
        if not isinstance(result, str) or not result.strip():
            raise AbbyProviderError(
                "Whisper returned invalid transcription text",
                code="invalid_remote_response",
            )
        return result.strip()


def _run_hf_whisper_stt(
    audio: Union[str, bytes],
    *,
    provider: HuggingFaceWhisperHTTPProvider,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    content_type: Optional[str] = None,
    **kwargs: object,
) -> str:
    """Run Hugging Face Whisper HTTP with bounded resilient attempts."""
    return provider._transcribe(
        audio,
        model_name=model_name,
        language=language,
        content_type=content_type,
        options=kwargs,
    )


# Descriptive aliases retained for code that names adapters after the objective.
AbbyIndexTTSProvider = IndexTTSHTTPProvider
AbbyWhisperProvider = HuggingFaceWhisperHTTPProvider


__all__ = [
    "HTTPRequest",
    "HTTPResponse",
    "HTTPTransport",
    "AbbyResiliencePolicy",
    "AbbyProviderAttempt",
    "AbbyProviderReceipt",
    "AbbyProviderError",
    "AbbyCircuitOpenError",
    "IndexTTSHTTPProvider",
    "HuggingFaceWhisperHTTPProvider",
    "AbbyIndexTTSProvider",
    "AbbyWhisperProvider",
    "_run_indextts_gradio_tts",
    "_run_hf_whisper_stt",
]
