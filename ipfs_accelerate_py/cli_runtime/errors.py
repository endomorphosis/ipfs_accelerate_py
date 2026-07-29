"""CLI runtime error taxonomy and bounded failure records.

Errors intentionally omit prompts, credentials, and sensitive environment
values. Messages and diagnostic details are length-bounded so they remain safe
to log, serialize, and surface to operators.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

MAX_ERROR_MESSAGE_CHARS: int = 4096
MAX_ERROR_DETAIL_KEYS: int = 32
MAX_ERROR_DETAIL_KEY_CHARS: int = 128
MAX_ERROR_DETAIL_VALUE_CHARS: int = 1024
MAX_ERROR_DETAIL_BYTES: int = 8192

_SENSITIVE_DETAIL_MARKERS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "prompt",
    "stdin",
)


class CLIRuntimeErrorCode(str, Enum):
    """Stable machine-readable failure codes for CLI runtime operations."""

    INVALID_CONTRACT = "invalid_contract"
    INVALID_STATE = "invalid_state"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    REGISTRY_COLLISION = "registry_collision"
    PROVIDER_NOT_FOUND = "provider_not_found"
    PROVIDER_LOAD_FAILED = "provider_load_failed"
    POLICY_DENIED = "policy_denied"
    SPAWN_FAILED = "spawn_failed"
    NONZERO_EXIT = "nonzero_exit"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MALFORMED_OUTPUT = "malformed_output"
    OUTPUT_TRUNCATED = "output_truncated"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION_FAILED = "authentication_failed"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    INTERNAL = "internal"


def _clip_text(value: Any, maximum: int) -> str:
    text = str("" if value is None else value)
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def _bounded_details(details: Mapping[str, Any] | None) -> dict[str, str]:
    if details is None:
        return {}
    if not isinstance(details, Mapping):
        raise TypeError("error details must be a mapping")
    out: dict[str, str] = {}
    for index, (raw_key, raw_value) in enumerate(details.items()):
        if index >= MAX_ERROR_DETAIL_KEYS:
            break
        key = _clip_text(raw_key, MAX_ERROR_DETAIL_KEY_CHARS)
        if not key:
            continue
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_DETAIL_MARKERS):
            out[key] = "[redacted]"
        else:
            out[key] = _clip_text(raw_value, MAX_ERROR_DETAIL_VALUE_CHARS)
    return out


def _normalize_code(code: CLIRuntimeErrorCode | str) -> CLIRuntimeErrorCode:
    if isinstance(code, CLIRuntimeErrorCode):
        return code
    try:
        return CLIRuntimeErrorCode(str(code))
    except ValueError as exc:
        raise ValueError(f"unknown CLI runtime error code: {code!r}") from exc


@dataclass(frozen=True)
class CLIErrorRecord:
    """Serializable, bounded failure detail without prompts or secrets."""

    code: CLIRuntimeErrorCode
    message: str
    retryable: bool = False
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _normalize_code(self.code))
        if not isinstance(self.message, str):
            raise TypeError("error message must be a string")
        message = self.message.strip()
        if not message:
            raise ValueError("error message must not be empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("error retryable must be a boolean")
        object.__setattr__(
            self, "message", _clip_text(message, MAX_ERROR_MESSAGE_CHARS)
        )
        object.__setattr__(self, "details", _bounded_details(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CLIErrorRecord:
        if not isinstance(payload, Mapping):
            raise TypeError("error record payload must be a mapping")
        return cls(
            code=payload.get("code", CLIRuntimeErrorCode.INTERNAL.value),
            message=str(payload.get("message", "")),
            retryable=bool(payload.get("retryable", False)),
            details=payload.get("details") or {},
        )


class CLIRuntimeError(RuntimeError):
    """Base exception carrying a bounded :class:`CLIErrorRecord`."""

    def __init__(
        self,
        message: str,
        *,
        code: CLIRuntimeErrorCode | str = CLIRuntimeErrorCode.INTERNAL,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        record = CLIErrorRecord(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )
        super().__init__(record.message)
        self.record = record
        self.code = record.code
        self.retryable = record.retryable
        self.details = record.details

    def to_dict(self) -> dict[str, Any]:
        return self.record.to_dict()


class ContractValidationError(CLIRuntimeError):
    """Raised when a contract field is missing, mistyped, or malformed."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.INVALID_CONTRACT,
            retryable=False,
            details=details,
        )


class InvalidStateError(CLIRuntimeError):
    """Raised for incompatible capability or mode combinations."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.INVALID_STATE,
            retryable=False,
            details=details,
        )


class BoundsExceededError(CLIRuntimeError):
    """Raised when a serialized or in-memory field exceeds contract bounds."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.BOUNDS_EXCEEDED,
            retryable=False,
            details=details,
        )


class RegistryError(CLIRuntimeError):
    """Base class for registry failures."""


class RegistryCollisionError(RegistryError):
    """Raised when a provider name or alias collides (fail-closed)."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.REGISTRY_COLLISION,
            retryable=False,
            details=details,
        )


class ProviderNotFoundError(RegistryError):
    """Raised when a registry lookup cannot resolve a provider name or alias."""

    def __init__(
        self,
        name: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"name": name}
        if details:
            payload.update(dict(details))
        super().__init__(
            f"CLI provider not found: {name}",
            code=CLIRuntimeErrorCode.PROVIDER_NOT_FOUND,
            retryable=False,
            details=payload,
        )


class ProviderLoadError(RegistryError):
    """Raised when a registered factory fails during lazy instantiation."""

    def __init__(
        self,
        name: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"name": name}
        if details:
            payload.update(dict(details))
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.PROVIDER_LOAD_FAILED,
            retryable=False,
            details=payload,
        )


class PolicyDeniedError(CLIRuntimeError):
    """Raised when policy forbids an operation (e.g. agent without auth)."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.POLICY_DENIED,
            retryable=False,
            details=details,
        )


class ProcessSpawnError(CLIRuntimeError):
    """Raised when a subprocess cannot be started."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.SPAWN_FAILED,
            retryable=False,
            details=details,
        )


class ProcessTimeoutError(CLIRuntimeError):
    """Raised when a subprocess exceeds its deadline."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.TIMEOUT,
            retryable=False,
            details=details,
        )


class ProcessCancelledError(CLIRuntimeError):
    """Raised when a subprocess is cancelled cooperatively or by token."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.CANCELLED,
            retryable=False,
            details=details,
        )


class NonzeroExitError(CLIRuntimeError):
    """Raised when a subprocess exits with a non-zero status."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.NONZERO_EXIT,
            retryable=retryable,
            details=details,
        )


class MalformedOutputError(CLIRuntimeError):
    """Raised when process output cannot be parsed into a contract result."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=CLIRuntimeErrorCode.MALFORMED_OUTPUT,
            retryable=False,
            details=details,
        )


__all__ = [
    "MAX_ERROR_MESSAGE_CHARS",
    "MAX_ERROR_DETAIL_KEYS",
    "MAX_ERROR_DETAIL_KEY_CHARS",
    "MAX_ERROR_DETAIL_VALUE_CHARS",
    "MAX_ERROR_DETAIL_BYTES",
    "CLIRuntimeErrorCode",
    "CLIErrorRecord",
    "CLIRuntimeError",
    "ContractValidationError",
    "InvalidStateError",
    "BoundsExceededError",
    "RegistryError",
    "RegistryCollisionError",
    "ProviderNotFoundError",
    "ProviderLoadError",
    "PolicyDeniedError",
    "ProcessSpawnError",
    "ProcessTimeoutError",
    "ProcessCancelledError",
    "NonzeroExitError",
    "MalformedOutputError",
]
