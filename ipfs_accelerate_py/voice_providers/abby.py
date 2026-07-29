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

PUBLICUS_INDEXTTS_SPACE_URL = "https://publicus-indextts-2-demo.hf.space"
PUBLICUS_INDEXTTS_MODEL = "Publicus/IndexTTS-2-Demo"
PUBLICUS_INDEXTTS_SINGLE_API_NAME = "/gen_single"
PUBLICUS_INDEXTTS_BATCH_API_NAME = "/gen_batch"
PUBLICUS_INDEXTTS_SINGLE_FN_INDEX = 6
PUBLICUS_INDEXTTS_BATCH_FN_INDEX = 7
PUBLICUS_INDEXTTS_INPUT_COUNT = 25
PUBLICUS_INDEXTTS_TIMEOUT_SECONDS = 900.0


def _cached_huggingface_token() -> str:
    """Return the locally cached Hub token without importing Hub at module load."""
    try:
        from huggingface_hub import get_token
    except (ImportError, AttributeError):
        try:
            from huggingface_hub.utils import get_token
        except (ImportError, AttributeError):
            return ""
    try:
        return str(get_token() or "").strip()
    except Exception:
        # A malformed or unreadable cache should behave like an absent token.
        return ""


def _huggingface_token_from_environment_or_cache() -> str:
    """Resolve an explicit Hub token first and consult the cache only as fallback."""
    for name in (
        "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "IPFS_DATASETS_PY_HF_API_TOKEN",
    ):
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return _cached_huggingface_token()


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
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{hostname}:{port}" if port is not None else hostname
        return urllib.parse.urlunsplit(
            (parsed.scheme, netloc, parsed.path, "", "")
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
    # Keep requests lazy for provider discovery while sharing the Space
    # transport and expired-FileData classifier with batch workers.
    from ..hf_space_inference import is_retryable_hf_space_error

    if is_retryable_hf_space_error(error):
        return True
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


@dataclass(frozen=True)
class _ProviderCallResult:
    value: object
    http_status: Optional[int] = None


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

    def _execute_callable(
        self,
        operation: str,
        endpoint_call: Callable[[str], _ProviderCallResult],
        *,
        sensitive_values: Sequence[object] = (),
    ) -> object:
        """Run a multi-request endpoint operation with the standard resilience policy.

        Gradio synthesis consists of upload, queue, event-stream, and file
        requests, so it cannot be represented by the single-request
        :meth:`_execute` callback.  This companion keeps those logical calls
        under the same endpoint ordering, retry, circuit, and receipt contract.
        """
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
                    call_result = endpoint_call(endpoint)
                    if not isinstance(call_result, _ProviderCallResult):
                        raise TypeError(
                            "Abby endpoint operation must return _ProviderCallResult"
                        )
                    http_status = call_result.http_status
                    attempts.append(
                        AbbyProviderAttempt(
                            endpoint=_safe_endpoint(endpoint),
                            attempt=attempt_index + 1,
                            status="succeeded",
                            duration_ms=(self._clock() - started_at) * 1000.0,
                            http_status=http_status,
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
                    return call_result.value
                except Exception as raw_error:
                    response = getattr(raw_error, "response", None)
                    raw_http_status = getattr(response, "status_code", None)
                    if isinstance(raw_http_status, int):
                        http_status = raw_http_status
                    retryable = (
                        _status_retryable(http_status)
                        if http_status is not None
                        else _exception_retryable(raw_error)
                    )
                    if isinstance(raw_error, AbbyProviderError):
                        endpoint_error = raw_error
                    else:
                        endpoint_error = AbbyProviderError(
                            _safe_error_text(
                                str(raw_error) or raw_error.__class__.__name__,
                                sensitive_values=sensitive_values,
                            ),
                            code=(
                                "remote_http_error"
                                if http_status is not None
                                else "remote_transport_error"
                                if retryable
                                else "invalid_remote_response"
                            ),
                            retryable=retryable,
                            http_status=http_status,
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


SpaceClientFactory = Callable[
    [str, float, Callable[[], Mapping[str, str]]],
    object,
]


def _default_space_client_factory(
    endpoint: str,
    timeout_seconds: float,
    headers_factory: Callable[[], Mapping[str, str]],
) -> object:
    # The shared client imports requests, so keep it behind the first actual
    # Publicus call rather than making voice-provider discovery import it.
    from ..hf_space_inference import HFSpaceClient

    return HFSpaceClient(
        endpoint,
        timeout_seconds=timeout_seconds,
        headers_factory=headers_factory,
    )


def _first_upload_path(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("path", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for candidate in value.values():
            found = _first_upload_path(candidate)
            if found:
                return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for candidate in value:
            found = _first_upload_path(candidate)
            if found:
                return found
    return ""


def _gradio_output_values(result: object) -> Tuple[object, ...]:
    if not isinstance(result, Mapping):
        return ()
    raw_values = result.get("data")
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        return ()
    values = []
    for value in raw_values:
        if isinstance(value, Mapping) and value.get("__type__") == "update":
            values.append(value.get("value"))
        else:
            values.append(value)
    return tuple(values)


def _is_audio_reference(value: object) -> bool:
    if isinstance(value, Mapping):
        mime_type = str(
            value.get("mime_type") or value.get("mimeType") or ""
        ).lower()
        path = str(
            value.get("path") or value.get("url") or value.get("name") or ""
        ).lower()
        if mime_type.startswith("audio/"):
            return True
        return path.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"))
    if isinstance(value, str):
        lowered = value.lower().split("?", 1)[0]
        return lowered.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"))
    return False


def _audio_references(value: object) -> Tuple[object, ...]:
    references = []
    seen = set()

    def visit(candidate: object) -> None:
        if _is_audio_reference(candidate):
            if isinstance(candidate, Mapping):
                identity = json.dumps(candidate, sort_keys=True, default=repr)
            else:
                identity = str(candidate)
            if identity not in seen:
                seen.add(identity)
                references.append(candidate)
            return
        if isinstance(candidate, Mapping):
            for child in candidate.values():
                visit(child)
        elif isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes)
        ):
            for child in candidate:
                visit(child)

    visit(value)
    return tuple(references)


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _bounded_int(
    value: object,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def _bounded_float(
    value: object,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return default


class IndexTTSHTTPProvider(_ResilientHTTPProvider):
    """Synthesis adapter for Publicus Gradio and compatible JSON endpoints."""

    provider_name = "abby_indextts"

    def __init__(
        self,
        endpoints: Sequence[str] = (PUBLICUS_INDEXTTS_SPACE_URL,),
        *,
        token: Optional[str] = None,
        bill_to: Optional[str] = None,
        default_model: str = PUBLICUS_INDEXTTS_MODEL,
        backend: str = "auto",
        reference_audio: Optional[object] = None,
        voice_description: str = "",
        single_api_name: str = PUBLICUS_INDEXTTS_SINGLE_API_NAME,
        batch_api_name: str = PUBLICUS_INDEXTTS_BATCH_API_NAME,
        single_fn_index: int = PUBLICUS_INDEXTTS_SINGLE_FN_INDEX,
        batch_fn_index: int = PUBLICUS_INDEXTTS_BATCH_FN_INDEX,
        contract_input_count: int = PUBLICUS_INDEXTTS_INPUT_COUNT,
        validate_contract: bool = True,
        space_client_factory: Optional[SpaceClientFactory] = None,
        policy: Optional[AbbyResiliencePolicy] = None,
        transport: Optional[HTTPTransport] = None,
        sleeper: Sleeper = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        normalized_backend = str(backend or "auto").strip().casefold().replace("-", "_")
        normalized_backend = {
            "generic": "http",
            "http_json": "http",
            "json": "http",
            "publicus": "gradio",
            "publicus_gradio": "gradio",
        }.get(normalized_backend, normalized_backend)
        if normalized_backend not in {"auto", "gradio", "http"}:
            raise ValueError("backend must be auto, gradio, or http")
        publicus_hostname = urllib.parse.urlsplit(
            PUBLICUS_INDEXTTS_SPACE_URL
        ).hostname
        publicus_default = normalized_backend == "gradio" or (
            normalized_backend == "auto"
            and any(
                urllib.parse.urlsplit(str(endpoint)).hostname == publicus_hostname
                for endpoint in endpoints
            )
        )
        super().__init__(
            endpoints,
            policy=policy
            or AbbyResiliencePolicy.from_environment(
                operation="indextts",
                default_timeout=(
                    PUBLICUS_INDEXTTS_TIMEOUT_SECONDS
                    if publicus_default
                    else 45.0
                ),
            ),
            transport=transport,
            sleeper=sleeper,
            clock=clock,
        )
        self._token = str(token or "").strip()
        self._bill_to = (
            "Publicus"
            if bill_to is None and publicus_default
            else str(bill_to or "").strip()
        )
        self.default_model = str(default_model or "").strip()
        self._backend_mode = normalized_backend
        self._reference_audio = reference_audio
        self.voice_description = str(voice_description or "").strip()
        self.single_api_name = "/" + str(single_api_name or "").strip().lstrip("/")
        self.batch_api_name = "/" + str(batch_api_name or "").strip().lstrip("/")
        self.single_fn_index = int(single_fn_index)
        self.batch_fn_index = int(batch_fn_index)
        self.contract_input_count = int(contract_input_count)
        if self.contract_input_count != PUBLICUS_INDEXTTS_INPUT_COUNT:
            raise ValueError("Publicus IndexTTS contract requires exactly 25 inputs")
        self.validate_contract = bool(validate_contract)
        self._space_client_factory = (
            space_client_factory or _default_space_client_factory
        )
        self._reference_cache: Dict[Tuple[str, str], Mapping[str, object]] = {}
        self._contract_cache: Dict[Tuple[str, str], int] = {}
        self._gradio_cache_lock = threading.Lock()

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
        if not configured:
            configured = (PUBLICUS_INDEXTTS_SPACE_URL,)
        local_reference = (
            os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_REFERENCE_AUDIO", "")
            or os.getenv("WALLET_INDEXTTS_REFERENCE_AUDIO_PATH", "")
        ).strip()
        remote_reference = (
            os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_REFERENCE_AUDIO_REMOTE_PATH",
                "",
            )
            or os.getenv("WALLET_INDEXTTS_REFERENCE_AUDIO_REMOTE_PATH", "")
        ).strip()
        reference_audio: object = local_reference or None
        if not local_reference and remote_reference:
            reference_audio = {
                "path": remote_reference,
                "meta": {"_type": "gradio.FileData"},
                "orig_name": os.path.basename(remote_reference) or "reference.wav",
            }
        resolved_token = (
            str(overrides["token"] or "").strip()
            if "token" in overrides
            else _huggingface_token_from_environment_or_cache()
        )
        values: Dict[str, object] = {
            "endpoints": configured,
            "token": resolved_token,
            "bill_to": (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_HF_BILL_TO", "")
                or os.getenv("IPFS_DATASETS_PY_HF_BILL_TO", "")
                or os.getenv("WALLET_INDEXTTS_HF_BILL_TO", "")
                or None
            ),
            "default_model": (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_MODEL", "")
                or os.getenv("WALLET_INDEXTTS_MODEL_NAME", "")
                or PUBLICUS_INDEXTTS_MODEL
            ),
            "backend": os.getenv(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_BACKEND", "auto"
            ),
            "reference_audio": reference_audio,
            "voice_description": (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_VOICE_DESCRIPTION", "")
                or os.getenv("WALLET_INDEXTTS_VOICE_DESCRIPTION", "")
            ),
            "single_api_name": (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_API_NAME", "")
                or os.getenv("WALLET_INDEXTTS_API_NAME", "")
                or PUBLICUS_INDEXTTS_SINGLE_API_NAME
            ),
            "batch_api_name": (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_BATCH_API_NAME", "")
                or os.getenv("WALLET_INDEXTTS_BATCH_API_NAME", "")
                or PUBLICUS_INDEXTTS_BATCH_API_NAME
            ),
            "single_fn_index": _env_int(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_FN_INDEX",
                _env_int(
                    "WALLET_INDEXTTS_FN_INDEX",
                    PUBLICUS_INDEXTTS_SINGLE_FN_INDEX,
                    minimum=0,
                    maximum=10000,
                ),
                minimum=0,
                maximum=10000,
            ),
            "batch_fn_index": _env_int(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_BATCH_FN_INDEX",
                _env_int(
                    "WALLET_INDEXTTS_BATCH_FN_INDEX",
                    PUBLICUS_INDEXTTS_BATCH_FN_INDEX,
                    minimum=0,
                    maximum=10000,
                ),
                minimum=0,
                maximum=10000,
            ),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @property
    def backend(self) -> str:
        modes = {self._endpoint_backend(endpoint) for endpoint in self.endpoints}
        if modes == {"gradio"}:
            return "publicus_gradio"
        if modes == {"http"}:
            return "http_json"
        return "mixed" if modes else self._backend_mode

    @property
    def authenticated(self) -> bool:
        return bool(self._token)

    @property
    def gradio_contract(self) -> Mapping[str, object]:
        """Return the credential-free Publicus endpoint contract."""
        return MappingProxyType(
            {
                "backend": "publicus_gradio",
                "space_url": PUBLICUS_INDEXTTS_SPACE_URL,
                "model": self.default_model,
                "input_count": self.contract_input_count,
                "single": MappingProxyType(
                    {
                        "api_name": self.single_api_name,
                        "fn_index": self.single_fn_index,
                    }
                ),
                "batch": MappingProxyType(
                    {
                        "api_name": self.batch_api_name,
                        "fn_index": self.batch_fn_index,
                    }
                ),
            }
        )

    def _configuration_identity(self) -> object:
        token_digest = (
            hashlib.sha256(self._token.encode("utf-8")).hexdigest()[:12]
            if self._token
            else ""
        )
        return (
            self.default_model,
            token_digest,
            self._bill_to,
            self._backend_mode,
            self.single_api_name,
            self.single_fn_index,
            self.batch_api_name,
            self.batch_fn_index,
            self.contract_input_count,
        )

    def _endpoint_backend(self, endpoint: str) -> str:
        if self._backend_mode != "auto":
            return self._backend_mode
        hostname = (urllib.parse.urlsplit(endpoint).hostname or "").casefold()
        return (
            "gradio"
            if hostname == urllib.parse.urlsplit(
                PUBLICUS_INDEXTTS_SPACE_URL
            ).hostname
            else "http"
        )

    def _authorization_headers(self) -> Mapping[str, str]:
        headers: Dict[str, str] = {}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        if self._bill_to:
            headers["X-HF-Bill-To"] = self._bill_to
        return headers

    @staticmethod
    def _reference_identity(reference: object) -> str:
        if isinstance(reference, bytes):
            return "bytes:" + hashlib.sha256(reference).hexdigest()
        if isinstance(reference, str):
            path = os.path.abspath(reference)
            if os.path.isfile(path):
                stat = os.stat(path)
                return f"path:{path}:{stat.st_mtime_ns}:{stat.st_size}"
            return "remote:" + reference
        return "mapping:" + hashlib.sha256(
            json.dumps(reference, sort_keys=True, default=repr).encode("utf-8")
        ).hexdigest()

    def _selected_reference(
        self,
        *,
        voice: Optional[str],
        options: Mapping[str, object],
    ) -> object:
        for key in (
            "reference_audio",
            "prompt_audio",
            "speaker_audio",
            "voice_reference",
        ):
            if options.get(key) is not None:
                return options[key]
        if voice and os.path.isfile(os.path.abspath(voice)):
            return voice
        if self._reference_audio is not None:
            return self._reference_audio
        raise AbbyProviderError(
            "Publicus IndexTTS requires reference_audio bytes, a local path, "
            "or a Gradio FileData mapping",
            code="provider_not_configured",
        )

    def _prepare_reference(
        self,
        client: object,
        endpoint: str,
        reference: object,
    ) -> Mapping[str, object]:
        if isinstance(reference, Mapping):
            return dict(reference)
        identity = self._reference_identity(reference)
        cache_key = (endpoint, identity)
        with self._gradio_cache_lock:
            cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if isinstance(reference, bytes):
            audio = reference
            file_name = "reference.wav"
        elif isinstance(reference, str):
            path = os.path.abspath(reference)
            if not os.path.isfile(path):
                if reference.startswith(("http://", "https://", "/gradio/")):
                    return {
                        "path": reference,
                        "meta": {"_type": "gradio.FileData"},
                        "orig_name": os.path.basename(reference) or "reference.wav",
                    }
                raise AbbyProviderError(
                    "Publicus reference audio path is not readable",
                    code="provider_not_configured",
                )
            try:
                with open(path, "rb") as input_file:
                    audio = input_file.read()
            except OSError as error:
                raise AbbyProviderError(
                    "Publicus reference audio path is not readable",
                    code="provider_not_configured",
                ) from error
            file_name = os.path.basename(path)
        else:
            raise AbbyProviderError(
                "Publicus reference audio must be bytes, a path, or FileData",
                code="provider_not_configured",
            )
        if not audio:
            raise AbbyProviderError(
                "Publicus reference audio is empty",
                code="provider_not_configured",
            )
        upload_file = getattr(client, "upload_file", None)
        if not callable(upload_file):
            raise AbbyProviderError(
                "Publicus Space client does not support file upload",
                code="invalid_remote_response",
            )
        upload = upload_file(
            file_name,
            audio,
            mimetypes.guess_type(file_name)[0] or "audio/wav",
        )
        upload_path = _first_upload_path(upload)
        if not upload_path:
            raise AbbyProviderError(
                "Publicus reference upload did not return a file path",
                code="invalid_remote_response",
            )
        prepared: Mapping[str, object] = {
            "path": upload_path,
            "meta": {"_type": "gradio.FileData"},
            "orig_name": file_name,
        }
        with self._gradio_cache_lock:
            self._reference_cache[cache_key] = dict(prepared)
        return prepared

    def _resolve_gradio_fn_index(
        self,
        client: object,
        endpoint: str,
        *,
        batch: bool,
    ) -> int:
        api_name = self.batch_api_name if batch else self.single_api_name
        expected_index = self.batch_fn_index if batch else self.single_fn_index
        if not self.validate_contract:
            return expected_index
        cache_key = (endpoint, api_name)
        with self._gradio_cache_lock:
            cached = self._contract_cache.get(cache_key)
        if cached is not None:
            return cached
        get_config = getattr(client, "get_config", None)
        resolve_fn_index = getattr(client, "resolve_fn_index", None)
        lookup_input_count = getattr(client, "lookup_dependency_input_count", None)
        if not all(
            callable(value)
            for value in (get_config, resolve_fn_index, lookup_input_count)
        ):
            raise AbbyProviderError(
                "Publicus Space client cannot validate the Gradio contract",
                code="publicus_contract_mismatch",
            )
        config = get_config()
        resolved_index = int(resolve_fn_index(api_name, config))
        input_count = lookup_input_count(resolved_index, config)
        if (
            resolved_index != expected_index
            or input_count != self.contract_input_count
        ):
            raise AbbyProviderError(
                f"Publicus {api_name} contract mismatch: expected fn "
                f"{expected_index} with {self.contract_input_count} inputs",
                code="publicus_contract_mismatch",
            )
        with self._gradio_cache_lock:
            self._contract_cache[cache_key] = resolved_index
        return resolved_index

    def _publicus_request_data(
        self,
        texts: Sequence[str],
        reference_audio: Mapping[str, object],
        *,
        voice: Optional[str],
        options: Mapping[str, object],
        batch: bool,
    ) -> Tuple[object, ...]:
        voice_description = str(
            options.get("voice_description")
            or options.get("emo_text")
            or self.voice_description
            or (voice if voice and not os.path.isfile(os.path.abspath(voice)) else "")
        )
        emotion_vectors = [
            _bounded_float(
                options.get(f"emotion_vector_{index}", options.get(f"vec{index}", 0.0)),
                0.0,
                minimum=0.0,
                maximum=1.0,
            )
            for index in range(1, 9)
        ]
        text_value = json.dumps(list(texts)) if batch else texts[0]
        default_bucket_size = len(texts) if batch and len(texts) > 1 else 0
        data: Tuple[object, ...] = (
            str(
                options.get("emotion_control_method")
                or options.get("emo_control_method")
                or "Same as the voice reference"
            ),
            reference_audio,
            text_value,
            options.get(
                "emotion_reference_audio",
                options.get("emo_ref_path"),
            ),
            _bounded_float(
                options.get("emotion_weight", options.get("emo_weight")),
                0.8,
                minimum=0.0,
                maximum=1.0,
            ),
            *emotion_vectors,
            voice_description,
            _coerce_bool(
                options.get("emotion_random", options.get("emo_random")),
                False,
            ),
            _bounded_int(
                options.get("max_text_tokens_per_segment"),
                120,
                minimum=1,
                maximum=4096,
            ),
            _bounded_int(
                options.get("segments_bucket_max_size"),
                default_bucket_size,
                minimum=0,
                maximum=4096,
            ),
            _coerce_bool(options.get("do_sample"), True),
            _bounded_float(options.get("top_p"), 0.8, minimum=0.0, maximum=1.0),
            _bounded_int(options.get("top_k"), 30, minimum=0, maximum=1000),
            _bounded_float(
                options.get("temperature"), 0.8, minimum=0.0, maximum=10.0
            ),
            _bounded_float(
                options.get("length_penalty"), 0.0, minimum=-10.0, maximum=10.0
            ),
            _bounded_int(options.get("num_beams"), 3, minimum=1, maximum=100),
            _bounded_float(
                options.get("repetition_penalty"),
                10.0,
                minimum=0.0,
                maximum=100.0,
            ),
            _bounded_int(
                options.get("max_mel_tokens"),
                1500,
                minimum=1,
                maximum=10000,
            ),
        )
        if len(data) != self.contract_input_count:
            raise AbbyProviderError(
                "Publicus request did not satisfy the 25-input Gradio contract",
                code="publicus_contract_mismatch",
            )
        return data

    def _call_publicus(
        self,
        endpoint: str,
        texts: Sequence[str],
        *,
        voice: Optional[str],
        options: Mapping[str, object],
        batch: bool,
    ) -> Tuple[bytes, ...]:
        reference = self._selected_reference(voice=voice, options=options)
        reference_identity = self._reference_identity(reference)
        client = self._space_client_factory(
            endpoint,
            self.policy.timeout_seconds,
            self._authorization_headers,
        )
        close = getattr(client, "close", None)
        try:
            prepared_reference = self._prepare_reference(
                client, endpoint, reference
            )
            fn_index = self._resolve_gradio_fn_index(
                client, endpoint, batch=batch
            )
            data = self._publicus_request_data(
                texts,
                prepared_reference,
                voice=voice,
                options=options,
                batch=batch,
            )
            queue_join = getattr(client, "queue_join", None)
            wait_for_result = getattr(client, "wait_for_queue_result", None)
            fetch_file = getattr(client, "fetch_file", None)
            if not all(
                callable(value)
                for value in (queue_join, wait_for_result, fetch_file)
            ):
                raise AbbyProviderError(
                    "Publicus Space client is missing queue or file methods",
                    code="invalid_remote_response",
                )
            session_hash = queue_join(fn_index, data)
            result = wait_for_result(
                session_hash,
                timeout_seconds=self.policy.timeout_seconds,
                poll_interval_seconds=0.5,
            )
            outputs = _gradio_output_values(result)
            if batch and len(outputs) >= 2:
                references = _audio_references(outputs[1])
            else:
                references = _audio_references(outputs or result)
            if len(references) < len(texts):
                all_references = _audio_references(result)
                combined = []
                seen_references = set()
                for audio_reference in (*references, *all_references):
                    identity = (
                        json.dumps(
                            audio_reference,
                            sort_keys=True,
                            default=repr,
                        )
                        if isinstance(audio_reference, Mapping)
                        else str(audio_reference)
                    )
                    if identity not in seen_references:
                        seen_references.add(identity)
                        combined.append(audio_reference)
                references = tuple(combined)
            if len(references) < len(texts):
                raise AbbyProviderError(
                    f"Publicus returned {len(references)} audio files for "
                    f"{len(texts)} texts",
                    code="invalid_remote_response",
                )
            audio_outputs = []
            for audio_reference in references[: len(texts)]:
                fetched = fetch_file(audio_reference)
                if (
                    not isinstance(fetched, Sequence)
                    or isinstance(fetched, (str, bytes))
                    or not fetched
                    or not isinstance(fetched[0], bytes)
                    or not fetched[0]
                ):
                    raise AbbyProviderError(
                        "Publicus returned invalid audio",
                        code="invalid_remote_response",
                    )
                audio_outputs.append(fetched[0])
            return tuple(audio_outputs)
        except Exception:
            # A Space restart can invalidate cached upload paths. Let a retry
            # re-upload the reference instead of repeating the stale FileData.
            with self._gradio_cache_lock:
                self._reference_cache.pop(
                    (endpoint, reference_identity), None
                )
            raise
        finally:
            if callable(close):
                close()

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

    def synthesize_batch(
        self,
        texts: Sequence[str],
        *,
        voice: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs: object,
    ) -> Tuple[bytes, ...]:
        """Synthesize multiple utterances through Publicus ``/gen_batch``."""
        _ = (model_name, device, output_format)
        prompts = tuple(str(text or "").strip() for text in texts)
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("texts must contain at least one non-empty item")

        def endpoint_call(endpoint: str) -> _ProviderCallResult:
            if self._endpoint_backend(endpoint) != "gradio":
                raise AbbyProviderError(
                    "Generic IndexTTS JSON endpoints do not advertise /gen_batch",
                    code="batch_not_supported",
                )
            return _ProviderCallResult(
                self._call_publicus(
                    endpoint,
                    prompts,
                    voice=voice,
                    options=kwargs,
                    batch=True,
                )
            )

        result = self._execute_callable(
            "synthesis_batch",
            endpoint_call,
            sensitive_values=(
                *prompts,
                self._token,
                self._reference_audio,
                kwargs.get("reference_audio"),
                kwargs.get("prompt_audio"),
                kwargs.get("speaker_audio"),
                kwargs.get("voice_reference"),
            ),
        )
        if not isinstance(result, tuple) or len(result) != len(prompts):
            raise AbbyProviderError(
                "Publicus returned an invalid batch",
                code="invalid_remote_response",
            )
        return result

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

        def endpoint_call(endpoint: str) -> _ProviderCallResult:
            if self._endpoint_backend(endpoint) == "gradio":
                return _ProviderCallResult(
                    self._call_publicus(
                        endpoint,
                        (prompt,),
                        voice=voice,
                        options=options,
                        batch=False,
                    )[0]
                )
            headers = {
                "Accept": "audio/*, application/json",
                "Content-Type": "application/json",
            }
            headers.update(self._authorization_headers())
            response = _transport_response(
                self._transport,
                HTTPRequest("POST", endpoint, headers, body),
                self.policy.timeout_seconds,
            )
            if not 200 <= response.status <= 299:
                raise AbbyProviderError(
                    f"{self.provider_name} returned HTTP {response.status}",
                    code="remote_http_error",
                    retryable=_status_retryable(response.status),
                    http_status=response.status,
                )
            return _ProviderCallResult(
                response_parser(response, endpoint),
                http_status=response.status,
            )

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

        result = self._execute_callable(
            "synthesis",
            endpoint_call,
            sensitive_values=(
                prompt,
                self._token,
                self._reference_audio,
                options.get("reference_audio"),
                options.get("prompt_audio"),
                options.get("speaker_audio"),
                options.get("voice_reference"),
            ),
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
PublicusIndexTTSProvider = IndexTTSHTTPProvider
AbbyWhisperProvider = HuggingFaceWhisperHTTPProvider


__all__ = [
    "HTTPRequest",
    "HTTPResponse",
    "HTTPTransport",
    "SpaceClientFactory",
    "PUBLICUS_INDEXTTS_SPACE_URL",
    "PUBLICUS_INDEXTTS_MODEL",
    "PUBLICUS_INDEXTTS_SINGLE_API_NAME",
    "PUBLICUS_INDEXTTS_BATCH_API_NAME",
    "PUBLICUS_INDEXTTS_SINGLE_FN_INDEX",
    "PUBLICUS_INDEXTTS_BATCH_FN_INDEX",
    "PUBLICUS_INDEXTTS_INPUT_COUNT",
    "PUBLICUS_INDEXTTS_TIMEOUT_SECONDS",
    "AbbyResiliencePolicy",
    "AbbyProviderAttempt",
    "AbbyProviderReceipt",
    "AbbyProviderError",
    "AbbyCircuitOpenError",
    "IndexTTSHTTPProvider",
    "HuggingFaceWhisperHTTPProvider",
    "AbbyIndexTTSProvider",
    "PublicusIndexTTSProvider",
    "AbbyWhisperProvider",
    "_run_indextts_gradio_tts",
    "_run_hf_whisper_stt",
]
