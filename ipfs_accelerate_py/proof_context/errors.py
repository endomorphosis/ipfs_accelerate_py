"""Typed v0.1 proof-context errors (PCCE-023).

Projects the frozen MCP++ error taxonomy into typed exceptions with retry,
escalation, review, and repair dispositions. Provider exceptions are mapped
to closed codes; messages are bounded and redacted. Importing this module
performs no I/O, network, process, or filesystem mutation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
ERROR_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/error"
CONTRACT_VERSION: Final[str] = "0.1"
ERROR_TAXONOMY_CONTENT_ID: Final[str] = (
    "sha256:570d43769cd47207f7c5f77bb7434252e6cefb1b4cf10791ccb82208db216a38"
)
PCCE_006_CONTENT_ID: Final[str] = (
    "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
)

ERRORS: Final[tuple[str, ...]] = (
    "unknown_field",
    "malformed",
    "identity_inconsistent",
    "stale_root",
    "simulated_promoted",
    "pseudo_cid",
    "schema_mismatch",
    "boundary_violation",
    "unavailable_capability",
    "timeout",
    "cancelled",
    "verification_failed",
    "proof_failed",
    "assurance_failed",
    "context_insufficient",
    "infrastructure_failure",
    "partial_effect",
    "repair_required",
    "human_review_required",
)

DISPOSITIONS: Final[tuple[str, ...]] = (
    "retry",
    "escalation",
    "review",
    "repair",
    "reject",
)

MAX_MESSAGE_LENGTH: Final[int] = 240
MAX_DETAIL_ITEMS: Final[int] = 8
MAX_DETAIL_VALUE_LENGTH: Final[int] = 120
REDACTED: Final[str] = "[redacted]"

ALLOWED_DETAIL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "operation",
        "field",
        "artifact",
        "capability",
        "stage",
        "reason",
        "retry_count",
        "lease_id",
    }
)

SECRET_DETAIL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "token",
        "secret",
        "authorization",
        "api_key",
        "apikey",
        "credential",
        "credentials",
        "private_key",
        "privatekey",
        "cookie",
        "session",
        "bearer",
        "access_token",
        "refresh_token",
    }
)

_SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|"
    r"authorization|bearer|private[_-]?key|credential)s?\s*([:=]|%3[dD])\s*\S+"
)
_BEARER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(bearer|sk-|ghp_|xox[baprs]-)\s*[A-Za-z0-9_\-./+=]{8,}"
)
_PROVIDER_CODE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "TimeoutError": "timeout",
        "CancelledError": "cancelled",
        "ConnectionError": "infrastructure_failure",
        "ConnectionResetError": "infrastructure_failure",
        "BrokenPipeError": "infrastructure_failure",
        "OSError": "infrastructure_failure",
        "MemoryError": "infrastructure_failure",
        "PermissionError": "infrastructure_failure",
        "FileNotFoundError": "unavailable_capability",
        "NotImplementedError": "unavailable_capability",
        "ModuleNotFoundError": "unavailable_capability",
        "ImportError": "unavailable_capability",
    }
)

# code -> (disposition, default status). Disposition is never success.
_ERROR_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("unknown_field", "reject", "invalid"),
    ("malformed", "reject", "invalid"),
    ("identity_inconsistent", "reject", "invalid"),
    ("stale_root", "reject", "stale"),
    ("simulated_promoted", "reject", "simulated"),
    ("pseudo_cid", "reject", "invalid"),
    ("schema_mismatch", "reject", "invalid"),
    ("boundary_violation", "reject", "rejected"),
    ("unavailable_capability", "retry", "unavailable"),
    ("timeout", "retry", "timeout"),
    ("cancelled", "reject", "cancelled"),
    ("verification_failed", "reject", "verification_failed"),
    ("proof_failed", "reject", "proof_failed"),
    ("assurance_failed", "reject", "assurance_failed"),
    ("context_insufficient", "escalation", "context_insufficient"),
    ("infrastructure_failure", "retry", "infrastructure_failure"),
    ("partial_effect", "repair", "partial_effect"),
    ("repair_required", "repair", "repair_required"),
    ("human_review_required", "review", "human_review_required"),
)


def _row_map() -> Mapping[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for code, disposition, status in _ERROR_ROWS:
        rows[code] = MappingProxyType(
            {
                "code": code,
                "disposition": disposition,
                "status": status,
                "retryable": disposition == "retry",
                "escalation": disposition == "escalation",
                "human_review": disposition == "review",
                "repair": disposition == "repair",
                "accepted": False,
            }
        )
    return MappingProxyType(rows)


ERROR_SEMANTICS: Final[Mapping[str, Mapping[str, Any]]] = _row_map()


def redact_text(value: str) -> str:
    """Bound and redact secret-bearing text. Never returns raw provider dumps."""

    text = value.replace("\x00", "")
    text = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED}", text)
    text = _BEARER_PATTERN.sub(REDACTED, text)
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH] + "…"
    return text


def _redact_key(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in SECRET_DETAIL_KEYS or any(
        token in normalized for token in ("secret", "password", "token", "credential")
    )


def bound_details(details: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Admit a closed, redacted, bounded detail mapping."""

    if details is None:
        return MappingProxyType({})
    if not isinstance(details, Mapping):
        raise ValueError("error details must be a mapping")
    admitted: dict[str, Any] = {}
    for raw_key, raw_value in details.items():
        key = str(raw_key)
        if key not in ALLOWED_DETAIL_KEYS or _redact_key(key):
            continue
        if len(admitted) >= MAX_DETAIL_ITEMS:
            break
        if raw_value is None or isinstance(raw_value, (bool, int)):
            admitted[key] = raw_value
            continue
        text = redact_text(str(raw_value))
        if len(text) > MAX_DETAIL_VALUE_LENGTH:
            text = text[:MAX_DETAIL_VALUE_LENGTH] + "…"
        admitted[key] = text
    return MappingProxyType(admitted)


def admit_error(code: Any) -> str:
    if not isinstance(code, str) or code not in ERRORS:
        raise UnknownFieldError(f"unknown error {code!r}")
    return code


def error_semantics(code: str) -> Mapping[str, Any]:
    admitted = admit_error(code)
    return ERROR_SEMANTICS[admitted]


def classify_error(code: str) -> str:
    """Return retry, escalation, review, repair, or reject. Never success."""

    return str(error_semantics(code)["disposition"])


def error_status(code: str) -> str:
    return str(error_semantics(code)["status"])


class ProofContextError(RuntimeError):
    """Fail-closed typed error. Wire `code` is a frozen v0.1 error value."""

    code: str = "malformed"
    disposition: str = "reject"
    retryable: bool = False
    escalation: bool = False
    human_review: bool = False
    repair: bool = False
    status: str = "invalid"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        chosen = code or reason or type(self).code
        if chosen not in ERRORS:
            raise ValueError(f"error code {chosen!r} is not in the frozen taxonomy")
        spec = ERROR_SEMANTICS[chosen]
        object.__setattr__(self, "code", chosen)
        object.__setattr__(self, "reason", chosen)
        object.__setattr__(self, "disposition", spec["disposition"])
        object.__setattr__(self, "retryable", spec["retryable"])
        object.__setattr__(self, "escalation", spec["escalation"])
        object.__setattr__(self, "human_review", spec["human_review"])
        object.__setattr__(self, "repair", spec["repair"])
        object.__setattr__(self, "status", spec["status"])
        object.__setattr__(self, "details", bound_details(details))
        redacted = redact_text(message) if message else chosen
        super().__init__(redacted)

    @property
    def accepted(self) -> bool:
        return False

    @accepted.setter
    def accepted(self, value: bool) -> None:
        if value:
            raise BoundaryViolationError("errors cannot claim success")

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": ERROR_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "code": self.code,
                "reason": self.reason,
                "disposition": self.disposition,
                "retryable": self.retryable,
                "escalation": self.escalation,
                "human_review": self.human_review,
                "repair": self.repair,
                "status": self.status,
                "accepted": False,
                "message": redact_text(str(self)),
                "details": dict(self.details),
                "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProofContextError:
        if not isinstance(payload, Mapping):
            raise MalformedError("error payload must be a mapping")
        extra = set(payload) - {
            "schema",
            "contract_version",
            "code",
            "reason",
            "disposition",
            "retryable",
            "escalation",
            "human_review",
            "repair",
            "status",
            "accepted",
            "message",
            "details",
            "error_taxonomy_content_id",
        }
        if extra:
            raise UnknownFieldError(f"unknown error field {sorted(extra)[0]!r}")
        schema = payload.get("schema")
        if schema is not None and schema != ERROR_SCHEMA:
            raise SchemaMismatchError(f"error schema {schema!r} is not {ERROR_SCHEMA}")
        code = admit_error(payload.get("code") or payload.get("reason"))
        if payload.get("accepted") is True:
            raise BoundaryViolationError("errors cannot claim success")
        return error_for(
            code,
            str(payload.get("message") or code),
            details=payload.get("details") if isinstance(payload.get("details"), Mapping) else None,
        )


def _typed_error(name: str, code: str) -> type[ProofContextError]:
    spec = ERROR_SEMANTICS[code]
    return type(
        name,
        (ProofContextError,),
        {
            "__module__": __name__,
            "__doc__": f"Typed v0.1 error for frozen code {code!r}.",
            "code": code,
            "disposition": spec["disposition"],
            "retryable": spec["retryable"],
            "escalation": spec["escalation"],
            "human_review": spec["human_review"],
            "repair": spec["repair"],
            "status": spec["status"],
        },
    )


UnknownFieldError = _typed_error("UnknownFieldError", "unknown_field")
MalformedError = _typed_error("MalformedError", "malformed")
IdentityInconsistentError = _typed_error("IdentityInconsistentError", "identity_inconsistent")
StaleRootError = _typed_error("StaleRootError", "stale_root")
SimulatedPromotedError = _typed_error("SimulatedPromotedError", "simulated_promoted")
PseudoCidError = _typed_error("PseudoCidError", "pseudo_cid")
SchemaMismatchError = _typed_error("SchemaMismatchError", "schema_mismatch")
BoundaryViolationError = _typed_error("BoundaryViolationError", "boundary_violation")
UnavailableCapabilityError = _typed_error("UnavailableCapabilityError", "unavailable_capability")
ProofTimeoutError = _typed_error("ProofTimeoutError", "timeout")
ProofCancelledError = _typed_error("ProofCancelledError", "cancelled")
VerificationFailedError = _typed_error("VerificationFailedError", "verification_failed")
ProofFailedError = _typed_error("ProofFailedError", "proof_failed")
AssuranceFailedError = _typed_error("AssuranceFailedError", "assurance_failed")
ContextInsufficientError = _typed_error("ContextInsufficientError", "context_insufficient")
InfrastructureFailureError = _typed_error("InfrastructureFailureError", "infrastructure_failure")
PartialEffectError = _typed_error("PartialEffectError", "partial_effect")
RepairRequiredError = _typed_error("RepairRequiredError", "repair_required")
HumanReviewRequiredError = _typed_error("HumanReviewRequiredError", "human_review_required")

ERROR_TYPES: Final[Mapping[str, type[ProofContextError]]] = MappingProxyType(
    {
        "unknown_field": UnknownFieldError,
        "malformed": MalformedError,
        "identity_inconsistent": IdentityInconsistentError,
        "stale_root": StaleRootError,
        "simulated_promoted": SimulatedPromotedError,
        "pseudo_cid": PseudoCidError,
        "schema_mismatch": SchemaMismatchError,
        "boundary_violation": BoundaryViolationError,
        "unavailable_capability": UnavailableCapabilityError,
        "timeout": ProofTimeoutError,
        "cancelled": ProofCancelledError,
        "verification_failed": VerificationFailedError,
        "proof_failed": ProofFailedError,
        "assurance_failed": AssuranceFailedError,
        "context_insufficient": ContextInsufficientError,
        "infrastructure_failure": InfrastructureFailureError,
        "partial_effect": PartialEffectError,
        "repair_required": RepairRequiredError,
        "human_review_required": HumanReviewRequiredError,
    }
)


def error_for(
    code: str,
    message: str = "",
    *,
    details: Mapping[str, Any] | None = None,
) -> ProofContextError:
    admitted = admit_error(code)
    return ERROR_TYPES[admitted](message, details=details)


def from_provider_error(
    exc: BaseException,
    *,
    code: str | None = None,
) -> ProofContextError:
    """Map an arbitrary provider exception to a bounded typed error.

    The original message, arguments, and traceback are not copied.
    """

    if isinstance(exc, ProofContextError):
        return exc
    mapped = code or _PROVIDER_CODE.get(type(exc).__name__, "infrastructure_failure")
    admitted = admit_error(mapped)
    typename = type(exc).__name__
    return error_for(
        admitted,
        f"provider {typename} classified as {admitted}",
        details={"capability": typename},
    )


def frozen_error_taxonomy() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": ERROR_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "errors": ERRORS,
            "dispositions": DISPOSITIONS,
            "semantics": {code: dict(ERROR_SEMANTICS[code]) for code in ERRORS},
            "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
            "pcce_006_content_id": PCCE_006_CONTENT_ID,
        }
    )


__all__ = [
    "ALLOWED_DETAIL_KEYS",
    "AssuranceFailedError",
    "BoundaryViolationError",
    "CONTRACT_VERSION",
    "ContextInsufficientError",
    "DISPOSITIONS",
    "ERROR_SCHEMA",
    "ERROR_SEMANTICS",
    "ERROR_TAXONOMY_CONTENT_ID",
    "ERROR_TYPES",
    "ERRORS",
    "HumanReviewRequiredError",
    "IdentityInconsistentError",
    "InfrastructureFailureError",
    "MalformedError",
    "PCCE_006_CONTENT_ID",
    "PartialEffectError",
    "ProofCancelledError",
    "ProofContextError",
    "ProofFailedError",
    "ProofTimeoutError",
    "PseudoCidError",
    "REDACTED",
    "RepairRequiredError",
    "SCHEMA",
    "SchemaMismatchError",
    "SimulatedPromotedError",
    "StaleRootError",
    "UnavailableCapabilityError",
    "UnknownFieldError",
    "VerificationFailedError",
    "admit_error",
    "bound_details",
    "classify_error",
    "error_for",
    "error_semantics",
    "error_status",
    "from_provider_error",
    "frozen_error_taxonomy",
    "redact_text",
]
