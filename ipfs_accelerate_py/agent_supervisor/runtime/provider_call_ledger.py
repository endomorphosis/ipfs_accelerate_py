"""Persist provider calls, usage, failure signatures, and churn decisions.

DQP-027 / Interfaces: ``ProviderCallLedger@1``, ``FailureSignature@1``,
``ChurnDecision@1``
============================================================================

Records redacted provider-call metadata so unchanged unsuccessful proposals
are not re-dispatched after their retry / negative-cache policy is exhausted,
while still charging every rejected, abandoned, and retry attempt.

Ordinary ledger rows never carry raw prompts, completions, credentials, or
raw endpoints.  Large bodies may only appear as content digests or CAS
handles; secret-shaped input is rejected at admission.

Acceptance properties
---------------------
* Same idempotency / call key dispatches once (exact replay returns the prior
  terminal record without a second provider dispatch).
* An unchanged failed proposal after an exhausted policy is suppressed.
* Changed evidence permits a new call.
* All rejected / abandoned / retry usage is charged.
* Raw prompts, completions, and secrets are not stored as ordinary rows.

Cold import of this module performs no filesystem, database, network,
provider, or process action.  Opening a ledger is the first I/O boundary.

Conflict policy: this module owns the call ledger and churn decision surface.
Existing provider routers remain the authority for provider selection.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

PROVIDER_CALL_LEDGER_INTERFACE: Final[str] = "ProviderCallLedger@1"
FAILURE_SIGNATURE_INTERFACE: Final[str] = "FailureSignature@1"
CHURN_DECISION_INTERFACE: Final[str] = "ChurnDecision@1"

PROVIDER_CALL_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-call-ledger@1"
)
PROVIDER_CALL_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-call-record@1"
)
PROVIDER_CALL_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-call-proposal@1"
)
FAILURE_SIGNATURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-failure-signature@1"
)
CHURN_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-churn-decision@1"
)
CHURN_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-churn-policy@1"
)
USAGE_CHARGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-usage-charge@1"
)
PROVIDER_RESPONSE_META_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-response-meta@1"
)

DEFAULT_LEDGER_VERSION: Final[str] = "provider-call-ledger@1"
AUTHORITY_CLASS: Final[str] = "operational_evidence"
PRODUCER_ID: Final[str] = "provider-call-ledger@1"

# Authority bounds: the ledger never authorizes usage or proves completion.
LEDGER_AUTHORIZES_USAGE: Final[bool] = False
LEDGER_REWRITES_PROVIDER_SETTLEMENT: Final[bool] = False
LEDGER_IS_COMPLETION_EVIDENCE: Final[bool] = False
LEDGER_IS_CORRECTNESS_EVIDENCE: Final[bool] = False

MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_BYTES: Final[int] = 1_024
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_TOKENS: Final[int] = 10**12
MAX_LATENCY_MS: Final[int] = 86_400_000
MAX_ATTEMPTS: Final[int] = 10_000
MAX_IDENTICAL_FAILURES: Final[int] = 1_024
MAX_NEGATIVE_CACHE_TTL_MS: Final[int] = 30 * 24 * 60 * 60 * 1_000
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 3
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_NEGATIVE_CACHE_TTL_MS: Final[int] = 3_600_000

_TEXT_SAFE = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*$")
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/=@+-]{0,255}$")

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|password|passwd|token|"
    r"credential|private[_-]?key|auth|auth[_-]?header|authorization|bearer)"
    r"(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:bearer\s+\S{12,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
    re.IGNORECASE,
)
_RAW_ENDPOINT = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|"
    r"(?:^|[.@/])(?:localhost|(?:\d{1,3}\.){3}\d{1,3})(?::\d+)?(?:/|$))"
)

_FORBIDDEN_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prompt",
        "prompts",
        "messages",
        "message",
        "completion",
        "completions",
        "output",
        "output_text",
        "input_text",
        "source",
        "source_body",
        "source_text",
        "media",
        "image_data",
        "audio_data",
        "video_data",
        "raw_body",
        "raw_headers",
        "response_body",
        "payload",
        "api_key",
        "authorization",
        "password",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "token",
        "private_key",
        "endpoint",
        "url",
        "uri",
        "base_url",
    }
)

_SENSITIVE_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "access_token",
        "refresh_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "passphrase",
        "private_key",
        "secret",
        "secrets",
        "token",
        "bearer",
    }
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS provider_call_ledger_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_calls (
    call_id VARCHAR PRIMARY KEY,
    call_key VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL DEFAULT '',
    provider VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    endpoint_fingerprint VARCHAR NOT NULL DEFAULT '',
    context_cid VARCHAR NOT NULL,
    plan_id VARCHAR NOT NULL DEFAULT '',
    task_id VARCHAR NOT NULL,
    attempt BIGINT NOT NULL DEFAULT 0,
    proposal_digest VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL DEFAULT '',
    outcome_class VARCHAR NOT NULL DEFAULT '',
    failure_signature_id VARCHAR NOT NULL DEFAULT '',
    response_digest VARCHAR NOT NULL DEFAULT '',
    mutation_result VARCHAR NOT NULL DEFAULT '',
    validation_result VARCHAR NOT NULL DEFAULT '',
    input_tokens_estimated BIGINT NOT NULL DEFAULT 0,
    output_tokens_estimated BIGINT NOT NULL DEFAULT 0,
    input_tokens_actual BIGINT NOT NULL DEFAULT 0,
    output_tokens_actual BIGINT NOT NULL DEFAULT 0,
    latency_ms BIGINT NOT NULL DEFAULT 0,
    budget_requests BIGINT NOT NULL DEFAULT 0,
    budget_input_tokens BIGINT NOT NULL DEFAULT 0,
    budget_output_tokens BIGINT NOT NULL DEFAULT 0,
    dispatched INTEGER NOT NULL DEFAULT 0,
    charged INTEGER NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    completed_at VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS provider_calls_call_key_uidx
    ON provider_calls(call_key);
CREATE INDEX IF NOT EXISTS provider_calls_task_idx
    ON provider_calls(task_id, attempt);
CREATE INDEX IF NOT EXISTS provider_calls_failure_idx
    ON provider_calls(failure_signature_id, evidence_digest);
CREATE INDEX IF NOT EXISTS provider_calls_idempotency_idx
    ON provider_calls(idempotency_key);

CREATE TABLE IF NOT EXISTS provider_responses (
    response_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    call_key VARCHAR NOT NULL,
    response_digest VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    outcome_class VARCHAR NOT NULL DEFAULT '',
    latency_ms BIGINT NOT NULL DEFAULT 0,
    input_tokens_actual BIGINT NOT NULL DEFAULT 0,
    output_tokens_actual BIGINT NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS provider_responses_call_idx
    ON provider_responses(call_id);

CREATE TABLE IF NOT EXISTS failure_signatures (
    failure_signature_id VARCHAR PRIMARY KEY,
    outcome_class VARCHAR NOT NULL,
    failure_code VARCHAR NOT NULL,
    proposal_digest VARCHAR NOT NULL,
    context_cid VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    provider VARCHAR NOT NULL DEFAULT '',
    model VARCHAR NOT NULL DEFAULT '',
    occurrence_count BIGINT NOT NULL DEFAULT 0,
    identical_failures BIGINT NOT NULL DEFAULT 0,
    last_call_id VARCHAR NOT NULL DEFAULT '',
    first_observed_at VARCHAR NOT NULL,
    last_observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS failure_signatures_proposal_idx
    ON failure_signatures(proposal_digest, context_cid);

CREATE TABLE IF NOT EXISTS replay_suppressions (
    suppression_id VARCHAR PRIMARY KEY,
    call_key VARCHAR NOT NULL,
    failure_signature_id VARCHAR NOT NULL DEFAULT '',
    decision VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    expires_at_ms BIGINT NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS replay_suppressions_call_key_idx
    ON replay_suppressions(call_key, expires_at_ms);
CREATE INDEX IF NOT EXISTS replay_suppressions_signature_idx
    ON replay_suppressions(failure_signature_id);

CREATE TABLE IF NOT EXISTS usage_charges (
    charge_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    call_key VARCHAR NOT NULL,
    charge_kind VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    requests BIGINT NOT NULL DEFAULT 1,
    cost_micros BIGINT NOT NULL DEFAULT 0,
    currency VARCHAR NOT NULL DEFAULT 'USD',
    charged INTEGER NOT NULL DEFAULT 1,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_charges_call_idx
    ON usage_charges(call_id);
CREATE INDEX IF NOT EXISTS usage_charges_kind_idx
    ON usage_charges(charge_kind, disposition);

CREATE TABLE IF NOT EXISTS churn_decisions (
    decision_id VARCHAR PRIMARY KEY,
    call_key VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    should_dispatch INTEGER NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    prior_call_id VARCHAR NOT NULL DEFAULT '',
    failure_signature_id VARCHAR NOT NULL DEFAULT '',
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    duplicate_kind VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS churn_decisions_call_key_idx
    ON churn_decisions(call_key, recorded_at);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderCallLedgerError(RuntimeError):
    """Base error for provider call ledger failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "provider_call_ledger",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ProviderCallLedgerNotOpenError(ProviderCallLedgerError):
    """Operation requires an open ledger."""

    def __init__(
        self, message: str = "ProviderCallLedger is not open"
    ) -> None:
        super().__init__(message, reason_code="not_open")


class ProviderCallLedgerIntegrityError(ProviderCallLedgerError, ValueError):
    """Identity or payload integrity failure."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "integrity",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class ProviderCallLedgerBoundsError(ProviderCallLedgerError, ValueError):
    """A resource or payload bound was exceeded."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "bounds",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class ProviderCallLedgerSecretError(ProviderCallLedgerError, ValueError):
    """Secret, prompt, or completion material was presented as an ordinary row."""

    def __init__(
        self,
        message: str = "raw prompts, completions, or secrets cannot be stored",
        *,
        reason_code: str = "secret_or_raw_payload_rejected",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class ProviderCallLedgerConflictError(ProviderCallLedgerError):
    """Idempotent conflict or inconsistent terminal state."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "conflict",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class DuckDBUnavailableError(ProviderCallLedgerError):
    """Optional DuckDB dependency is not installed."""

    def __init__(
        self,
        message: str = "DuckDB is required for ProviderCallLedger",
    ) -> None:
        super().__init__(message, reason_code="duckdb_unavailable")


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class CallStatus(str, Enum):
    """Lifecycle status of a provider call record."""

    ADMITTED = "admitted"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    SUPPRESSED = "suppressed"
    REPLAYED = "replayed"


class CallOutcome(str, Enum):
    """Typed terminal or intermediate outcome of a provider call."""

    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    RETRY = "retry"
    HARD_QUOTA = "hard_quota"
    TRANSIENT_FAILURE = "transient_failure"
    RESPONSE_LOSS = "response_loss"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SUPPRESSED = "suppressed"
    REPLAYED = "replayed"
    UNKNOWN = "unknown"


class OutcomeClass(str, Enum):
    """Coarse failure / outcome class used for signatures and churn."""

    SUCCESS = "success"
    HARD_QUOTA = "hard_quota"
    TRANSIENT = "transient"
    RESPONSE_LOSS = "response_loss"
    VALIDATION = "validation"
    MUTATION = "mutation"
    POLICY = "policy"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMITED = "rate_limited"
    TRANSPORT = "transport"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


class DuplicateKind(str, Enum):
    """How a proposal relates to prior ledger state."""

    NONE = "none"
    EXACT = "exact"
    SEMANTIC = "semantic"
    IDEMPOTENCY = "idempotency"


class ChurnDisposition(str, Enum):
    """Decision about whether a proposal may hit a provider."""

    DISPATCH = "dispatch"
    REPLAY_EXACT = "replay_exact"
    SUPPRESS_EXHAUSTED = "suppress_exhausted"
    SUPPRESS_NEGATIVE_CACHE = "suppress_negative_cache"
    SUPPRESS_SEMANTIC_DUPLICATE = "suppress_semantic_duplicate"
    ALLOW_CHANGED_EVIDENCE = "allow_changed_evidence"


class ChargeKind(str, Enum):
    """Usage charge attribution kind."""

    SUCCESS = "success"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    RETRY = "retry"
    FAILED = "failed"
    TRANSIENT = "transient"
    HARD_QUOTA = "hard_quota"
    RESPONSE_LOSS = "response_loss"
    SUPPRESSED = "suppressed"
    REPLAY = "replay"


class MutationResult(str, Enum):
    NONE = "none"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    NOT_APPLICABLE = "not_applicable"


class ValidationResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    NOT_REQUIRED = "not_required"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderCallLedgerIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _split_sql_statements(sql_text: str) -> list[str]:
    return [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
    ]


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise ProviderCallLedgerIntegrityError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise ProviderCallLedgerIntegrityError(f"{name} must be text")
    text = value.strip()
    if required and not text:
        raise ProviderCallLedgerIntegrityError(f"{name} is required")
    if not required and not text:
        return ""
    if "\x00" in text or not _TEXT_SAFE.fullmatch(text):
        raise ProviderCallLedgerIntegrityError(
            f"{name} contains unsafe control characters"
        )
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ProviderCallLedgerBoundsError(
            f"{name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    if _looks_like_secret_value(text):
        raise ProviderCallLedgerSecretError(
            f"{name} contains credential-shaped data"
        )
    if _looks_like_raw_endpoint(text) and name not in {
        "endpoint_fingerprint",
        "response_digest",
        "proposal_digest",
        "evidence_digest",
        "prompt_digest",
        "completion_digest",
        "call_key",
        "call_id",
        "context_cid",
    }:
        # Fingerprints and digests are allowed; raw URLs in ordinary fields are not.
        if name in {"provider", "model", "task_id", "plan_id", "idempotency_key"}:
            pass
        elif "endpoint" in name and "fingerprint" not in name:
            raise ProviderCallLedgerSecretError(
                f"{name} must not embed a raw endpoint or URL"
            )
    return text


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required)
    if not text:
        return ""
    if not _NAME.fullmatch(text):
        # Digests and content IDs use a broader charset; re-check loosely.
        if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]{0,511}$", text):
            raise ProviderCallLedgerIntegrityError(
                f"{name} is not a safe identifier"
            )
    return text


def _nonneg_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_TOKENS,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderCallLedgerBoundsError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise ProviderCallLedgerBoundsError(
            f"{name} must be between 0 and {maximum}"
        )
    return value


def _positive_int(
    value: Any,
    name: str,
    *,
    maximum: int = MAX_TOKENS,
) -> int:
    number = _nonneg_int(value, name, maximum=maximum)
    if number <= 0:
        raise ProviderCallLedgerBoundsError(f"{name} must be positive")
    return number


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ProviderCallLedgerIntegrityError(
            f"{name} is not a supported {enum_type.__name__}"
        ) from exc


def _normalized_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _looks_like_secret_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _SENSITIVE_KEY_MARKERS or normalized in _FORBIDDEN_ROW_KEYS:
        return True
    if re.search(r"(?:pseudonym|fingerprint|digest|cid)$", normalized):
        return False
    return bool(_SECRET_KEY.search(key))


def _looks_like_secret_value(value: str) -> bool:
    return bool(_SECRET_VALUE.search(str(value).strip()))


def _looks_like_raw_endpoint(value: str) -> bool:
    return bool(_RAW_ENDPOINT.search(str(value)))


def _assert_no_forbidden_payload(payload: Mapping[str, Any], *, path: str = "") -> None:
    """Reject raw prompts/completions/secrets in ordinary row bodies."""

    for key, value in payload.items():
        key_text = str(key)
        location = f"{path}.{key_text}" if path else key_text
        normalized = _normalized_key(key_text)
        if normalized in _FORBIDDEN_ROW_KEYS or _looks_like_secret_key(key_text):
            raise ProviderCallLedgerSecretError(
                f"forbidden ordinary-row field: {location}"
            )
        if isinstance(value, str):
            if _looks_like_secret_value(value):
                raise ProviderCallLedgerSecretError(
                    f"credential-shaped value at {location}"
                )
            # Raw multi-line model bodies are not ordinary rows.
            if normalized.endswith("_prompt") or normalized.endswith("_completion"):
                raise ProviderCallLedgerSecretError(
                    f"raw prompt/completion field at {location}"
                )
        elif isinstance(value, Mapping):
            _assert_no_forbidden_payload(value, path=location)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    _assert_no_forbidden_payload(
                        item, path=f"{location}[{index}]"
                    )
                elif isinstance(item, str) and _looks_like_secret_value(item):
                    raise ProviderCallLedgerSecretError(
                        f"credential-shaped value at {location}[{index}]"
                    )


def _freeze_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def _outcome_to_class(outcome: CallOutcome) -> OutcomeClass:
    mapping = {
        CallOutcome.SUCCESS: OutcomeClass.SUCCESS,
        CallOutcome.HARD_QUOTA: OutcomeClass.HARD_QUOTA,
        CallOutcome.TRANSIENT_FAILURE: OutcomeClass.TRANSIENT,
        CallOutcome.RESPONSE_LOSS: OutcomeClass.RESPONSE_LOSS,
        CallOutcome.TIMEOUT: OutcomeClass.TRANSIENT,
        CallOutcome.REJECTED: OutcomeClass.REJECTED,
        CallOutcome.ABANDONED: OutcomeClass.ABANDONED,
        CallOutcome.FAILED: OutcomeClass.UNKNOWN,
        CallOutcome.RETRY: OutcomeClass.TRANSIENT,
        CallOutcome.CANCELLED: OutcomeClass.ABANDONED,
        CallOutcome.SUPPRESSED: OutcomeClass.POLICY,
        CallOutcome.REPLAYED: OutcomeClass.SUCCESS,
        CallOutcome.UNKNOWN: OutcomeClass.UNKNOWN,
    }
    return mapping.get(outcome, OutcomeClass.UNKNOWN)


def _outcome_to_charge_kind(outcome: CallOutcome) -> ChargeKind:
    mapping = {
        CallOutcome.SUCCESS: ChargeKind.SUCCESS,
        CallOutcome.REJECTED: ChargeKind.REJECTED,
        CallOutcome.ABANDONED: ChargeKind.ABANDONED,
        CallOutcome.RETRY: ChargeKind.RETRY,
        CallOutcome.FAILED: ChargeKind.FAILED,
        CallOutcome.TRANSIENT_FAILURE: ChargeKind.TRANSIENT,
        CallOutcome.HARD_QUOTA: ChargeKind.HARD_QUOTA,
        CallOutcome.RESPONSE_LOSS: ChargeKind.RESPONSE_LOSS,
        CallOutcome.TIMEOUT: ChargeKind.TRANSIENT,
        CallOutcome.CANCELLED: ChargeKind.ABANDONED,
        CallOutcome.SUPPRESSED: ChargeKind.SUPPRESSED,
        CallOutcome.REPLAYED: ChargeKind.REPLAY,
        CallOutcome.UNKNOWN: ChargeKind.FAILED,
    }
    return mapping.get(outcome, ChargeKind.FAILED)


def _is_terminal_status(status: CallStatus) -> bool:
    return status in {
        CallStatus.COMPLETED,
        CallStatus.FAILED,
        CallStatus.REJECTED,
        CallStatus.ABANDONED,
        CallStatus.SUPPRESSED,
        CallStatus.REPLAYED,
    }


def _is_failed_outcome(outcome: CallOutcome) -> bool:
    return outcome in {
        CallOutcome.FAILED,
        CallOutcome.REJECTED,
        CallOutcome.ABANDONED,
        CallOutcome.HARD_QUOTA,
        CallOutcome.TRANSIENT_FAILURE,
        CallOutcome.RESPONSE_LOSS,
        CallOutcome.TIMEOUT,
        CallOutcome.CANCELLED,
        CallOutcome.UNKNOWN,
    }


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChurnPolicy:
    """Finite retry, identical-failure, and negative-cache bounds."""

    max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES
    max_retries: int = DEFAULT_MAX_RETRIES
    negative_cache_ttl_ms: int = DEFAULT_NEGATIVE_CACHE_TTL_MS
    charge_rejected: bool = True
    charge_abandoned: bool = True
    charge_retry: bool = True
    suppress_semantic_duplicates: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_identical_failures",
            _positive_int(
                int(self.max_identical_failures),
                "max_identical_failures",
                maximum=MAX_IDENTICAL_FAILURES,
            ),
        )
        object.__setattr__(
            self,
            "max_retries",
            _positive_int(
                int(self.max_retries),
                "max_retries",
                maximum=MAX_ATTEMPTS,
            ),
        )
        object.__setattr__(
            self,
            "negative_cache_ttl_ms",
            _nonneg_int(
                int(self.negative_cache_ttl_ms),
                "negative_cache_ttl_ms",
                maximum=MAX_NEGATIVE_CACHE_TTL_MS,
            ),
        )
        object.__setattr__(self, "charge_rejected", bool(self.charge_rejected))
        object.__setattr__(self, "charge_abandoned", bool(self.charge_abandoned))
        object.__setattr__(self, "charge_retry", bool(self.charge_retry))
        object.__setattr__(
            self,
            "suppress_semantic_duplicates",
            bool(self.suppress_semantic_duplicates),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHURN_POLICY_SCHEMA,
            "max_identical_failures": self.max_identical_failures,
            "max_retries": self.max_retries,
            "negative_cache_ttl_ms": self.negative_cache_ttl_ms,
            "charge_rejected": self.charge_rejected,
            "charge_abandoned": self.charge_abandoned,
            "charge_retry": self.charge_retry,
            "suppress_semantic_duplicates": self.suppress_semantic_duplicates,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChurnPolicy":
        if not isinstance(payload, Mapping):
            raise ProviderCallLedgerIntegrityError(
                "churn policy must be an object"
            )
        data = dict(payload)
        data.pop("schema", None)
        return cls(**data)


@dataclass(frozen=True)
class ProviderCallBudget:
    """Token / request budget bound to a proposal."""

    requests: int = 1
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requests", _nonneg_int(int(self.requests), "requests")
        )
        object.__setattr__(
            self,
            "input_tokens",
            _nonneg_int(int(self.input_tokens), "input_tokens"),
        )
        object.__setattr__(
            self,
            "output_tokens",
            _nonneg_int(int(self.output_tokens), "output_tokens"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class ProviderTokenUsage:
    """Estimated and/or actual token observations for one call."""

    input_tokens_estimated: int = 0
    output_tokens_estimated: int = 0
    input_tokens_actual: int = 0
    output_tokens_actual: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _nonneg_int(int(getattr(self, name)), name),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }

    @property
    def charged_input_tokens(self) -> int:
        return (
            self.input_tokens_actual
            if self.input_tokens_actual
            else self.input_tokens_estimated
        )

    @property
    def charged_output_tokens(self) -> int:
        return (
            self.output_tokens_actual
            if self.output_tokens_actual
            else self.output_tokens_estimated
        )


@dataclass(frozen=True)
class ProviderCallProposal:
    """Redacted proposal presented for dispatch evaluation.

    Callers must supply digests for prompts / packets rather than raw bodies.
    ``idempotency_key``, when present, is the exact-duplicate key.
    """

    provider: str
    model: str
    context_cid: str
    task_id: str
    proposal_digest: str
    evidence_digest: str
    plan_id: str = ""
    attempt: int = 0
    idempotency_key: str = ""
    endpoint_fingerprint: str = ""
    prompt_digest: str = ""
    budget: ProviderCallBudget = field(default_factory=ProviderCallBudget)
    token_estimate: ProviderTokenUsage = field(
        default_factory=ProviderTokenUsage
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider", _identifier(self.provider, "provider")
        )
        object.__setattr__(self, "model", _identifier(self.model, "model"))
        object.__setattr__(
            self, "context_cid", _identifier(self.context_cid, "context_cid")
        )
        object.__setattr__(
            self, "task_id", _identifier(self.task_id, "task_id")
        )
        object.__setattr__(
            self,
            "proposal_digest",
            _identifier(self.proposal_digest, "proposal_digest"),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _identifier(self.evidence_digest, "evidence_digest"),
        )
        object.__setattr__(
            self, "plan_id", _identifier(self.plan_id, "plan_id", required=False)
        )
        object.__setattr__(
            self,
            "attempt",
            _nonneg_int(int(self.attempt), "attempt", maximum=MAX_ATTEMPTS),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _identifier(
                self.idempotency_key, "idempotency_key", required=False
            ),
        )
        object.__setattr__(
            self,
            "endpoint_fingerprint",
            _identifier(
                self.endpoint_fingerprint,
                "endpoint_fingerprint",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "prompt_digest",
            _identifier(
                self.prompt_digest, "prompt_digest", required=False
            ),
        )
        budget = self.budget
        if not isinstance(budget, ProviderCallBudget):
            if not isinstance(budget, Mapping):
                raise ProviderCallLedgerIntegrityError("budget is invalid")
            budget = ProviderCallBudget(**dict(budget))
        object.__setattr__(self, "budget", budget)
        tokens = self.token_estimate
        if not isinstance(tokens, ProviderTokenUsage):
            if not isinstance(tokens, Mapping):
                raise ProviderCallLedgerIntegrityError(
                    "token_estimate is invalid"
                )
            tokens = ProviderTokenUsage(**dict(tokens))
        object.__setattr__(self, "token_estimate", tokens)
        meta = dict(self.metadata or {})
        _assert_no_forbidden_payload(meta)
        object.__setattr__(self, "metadata", _freeze_mapping(meta))

    @property
    def call_key(self) -> str:
        """Redacted exact call key used for idempotent dispatch."""

        if self.idempotency_key:
            return _identity(
                "call",
                {
                    "kind": "idempotency",
                    "idempotency_key": self.idempotency_key,
                    "provider": self.provider,
                    "model": self.model,
                },
            )
        return _identity(
            "call",
            {
                "kind": "semantic",
                "provider": self.provider,
                "model": self.model,
                "endpoint_fingerprint": self.endpoint_fingerprint,
                "context_cid": self.context_cid,
                "plan_id": self.plan_id,
                "task_id": self.task_id,
                "attempt": self.attempt,
                "proposal_digest": self.proposal_digest,
                "evidence_digest": self.evidence_digest,
                "prompt_digest": self.prompt_digest,
            },
        )

    @property
    def semantic_key(self) -> str:
        """Semantic identity independent of attempt / idempotency framing."""

        return _identity(
            "semantic",
            {
                "provider": self.provider,
                "model": self.model,
                "endpoint_fingerprint": self.endpoint_fingerprint,
                "context_cid": self.context_cid,
                "plan_id": self.plan_id,
                "task_id": self.task_id,
                "proposal_digest": self.proposal_digest,
                "prompt_digest": self.prompt_digest,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CALL_PROPOSAL_SCHEMA,
            "provider": self.provider,
            "model": self.model,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "context_cid": self.context_cid,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "proposal_digest": self.proposal_digest,
            "evidence_digest": self.evidence_digest,
            "prompt_digest": self.prompt_digest,
            "budget": self.budget.to_dict(),
            "token_estimate": self.token_estimate.to_dict(),
            "metadata": dict(self.metadata),
            "call_key": self.call_key,
            "semantic_key": self.semantic_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderCallProposal":
        if not isinstance(payload, Mapping):
            raise ProviderCallLedgerIntegrityError(
                "provider call proposal must be an object"
            )
        data = {
            key: payload[key]
            for key in (
                "provider",
                "model",
                "context_cid",
                "task_id",
                "proposal_digest",
                "evidence_digest",
                "plan_id",
                "attempt",
                "idempotency_key",
                "endpoint_fingerprint",
                "prompt_digest",
                "budget",
                "token_estimate",
                "metadata",
            )
            if key in payload
        }
        return cls(**data)


@dataclass(frozen=True)
class FailureSignature:
    """Typed failure identity used for replay suppression.

    Interface: ``FailureSignature@1``.
    """

    INTERFACE: Final[str] = FAILURE_SIGNATURE_INTERFACE

    outcome_class: OutcomeClass | str
    failure_code: str
    proposal_digest: str
    context_cid: str
    evidence_digest: str = ""
    provider: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_class",
            _enum(self.outcome_class, OutcomeClass, "outcome_class"),
        )
        object.__setattr__(
            self,
            "failure_code",
            _identifier(self.failure_code, "failure_code"),
        )
        object.__setattr__(
            self,
            "proposal_digest",
            _identifier(self.proposal_digest, "proposal_digest"),
        )
        object.__setattr__(
            self, "context_cid", _identifier(self.context_cid, "context_cid")
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _identifier(
                self.evidence_digest, "evidence_digest", required=False
            ),
        )
        object.__setattr__(
            self,
            "provider",
            _identifier(self.provider, "provider", required=False),
        )
        object.__setattr__(
            self, "model", _identifier(self.model, "model", required=False)
        )

    @property
    def failure_signature_id(self) -> str:
        return _identity(
            "fsig",
            {
                "schema": FAILURE_SIGNATURE_SCHEMA,
                "outcome_class": self.outcome_class.value,  # type: ignore[union-attr]
                "failure_code": self.failure_code,
                "proposal_digest": self.proposal_digest,
                "context_cid": self.context_cid,
                "provider": self.provider,
                "model": self.model,
                # evidence_digest is intentionally excluded from the stable
                # signature identity so *changed* evidence can reopen a call
                # while the same signature tracks exhaustion across attempts.
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FAILURE_SIGNATURE_SCHEMA,
            "interface": FAILURE_SIGNATURE_INTERFACE,
            "failure_signature_id": self.failure_signature_id,
            "outcome_class": self.outcome_class.value,  # type: ignore[union-attr]
            "failure_code": self.failure_code,
            "proposal_digest": self.proposal_digest,
            "context_cid": self.context_cid,
            "evidence_digest": self.evidence_digest,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureSignature":
        if not isinstance(payload, Mapping):
            raise ProviderCallLedgerIntegrityError(
                "failure signature must be an object"
            )
        data = {
            key: payload[key]
            for key in (
                "outcome_class",
                "failure_code",
                "proposal_digest",
                "context_cid",
                "evidence_digest",
                "provider",
                "model",
            )
            if key in payload
        }
        result = cls(**data)
        claimed = str(payload.get("failure_signature_id") or "").strip()
        if claimed and claimed != result.failure_signature_id:
            raise ProviderCallLedgerIntegrityError(
                "failure signature id does not match content"
            )
        return result

    @classmethod
    def from_outcome(
        cls,
        *,
        proposal: ProviderCallProposal,
        outcome: CallOutcome | str,
        failure_code: str = "",
        outcome_class: OutcomeClass | str | None = None,
    ) -> "FailureSignature":
        resolved_outcome = _enum(outcome, CallOutcome, "outcome")
        resolved_class = (
            _enum(outcome_class, OutcomeClass, "outcome_class")
            if outcome_class is not None
            else _outcome_to_class(resolved_outcome)
        )
        code = failure_code or resolved_outcome.value
        return cls(
            outcome_class=resolved_class,
            failure_code=code,
            proposal_digest=proposal.proposal_digest,
            context_cid=proposal.context_cid,
            evidence_digest=proposal.evidence_digest,
            provider=proposal.provider,
            model=proposal.model,
        )


@dataclass(frozen=True)
class ChurnDecision:
    """Whether a proposal may be dispatched to a provider.

    Interface: ``ChurnDecision@1``.
    """

    INTERFACE: Final[str] = CHURN_DECISION_INTERFACE

    disposition: ChurnDisposition | str
    call_key: str
    should_dispatch: bool
    reason: str = ""
    prior_call_id: str = ""
    failure_signature_id: str = ""
    evidence_digest: str = ""
    duplicate_kind: DuplicateKind | str = DuplicateKind.NONE
    decision_id: str = ""
    recorded_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ChurnDisposition, "disposition"),
        )
        object.__setattr__(
            self, "call_key", _identifier(self.call_key, "call_key")
        )
        object.__setattr__(self, "should_dispatch", bool(self.should_dispatch))
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, "reason", required=False)
            if self.reason
            else "",
        )
        if self.reason and len(self.reason.encode("utf-8")) > MAX_REASON_BYTES:
            raise ProviderCallLedgerBoundsError("reason is too large")
        object.__setattr__(
            self,
            "prior_call_id",
            _identifier(
                self.prior_call_id, "prior_call_id", required=False
            ),
        )
        object.__setattr__(
            self,
            "failure_signature_id",
            _identifier(
                self.failure_signature_id,
                "failure_signature_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _identifier(
                self.evidence_digest, "evidence_digest", required=False
            ),
        )
        object.__setattr__(
            self,
            "duplicate_kind",
            _enum(self.duplicate_kind, DuplicateKind, "duplicate_kind"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at, "recorded_at", required=False)
            if self.recorded_at
            else "",
        )
        if not self.decision_id:
            object.__setattr__(
                self,
                "decision_id",
                _identity(
                    "churn",
                    {
                        "call_key": self.call_key,
                        "disposition": self.disposition.value,  # type: ignore[union-attr]
                        "should_dispatch": self.should_dispatch,
                        "prior_call_id": self.prior_call_id,
                        "failure_signature_id": self.failure_signature_id,
                        "evidence_digest": self.evidence_digest,
                        "duplicate_kind": self.duplicate_kind.value,  # type: ignore[union-attr]
                        "reason": self.reason,
                    },
                ),
            )
        else:
            object.__setattr__(
                self,
                "decision_id",
                _identifier(self.decision_id, "decision_id"),
            )
        expected_dispatch = self.disposition in {
            ChurnDisposition.DISPATCH,
            ChurnDisposition.ALLOW_CHANGED_EVIDENCE,
        }
        if bool(self.should_dispatch) != bool(expected_dispatch):
            raise ProviderCallLedgerIntegrityError(
                "should_dispatch is inconsistent with disposition"
            )

    @property
    def suppressed(self) -> bool:
        return self.disposition in {
            ChurnDisposition.SUPPRESS_EXHAUSTED,
            ChurnDisposition.SUPPRESS_NEGATIVE_CACHE,
            ChurnDisposition.SUPPRESS_SEMANTIC_DUPLICATE,
            ChurnDisposition.REPLAY_EXACT,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHURN_DECISION_SCHEMA,
            "interface": CHURN_DECISION_INTERFACE,
            "decision_id": self.decision_id,
            "disposition": self.disposition.value,  # type: ignore[union-attr]
            "call_key": self.call_key,
            "should_dispatch": self.should_dispatch,
            "reason": self.reason,
            "prior_call_id": self.prior_call_id,
            "failure_signature_id": self.failure_signature_id,
            "evidence_digest": self.evidence_digest,
            "duplicate_kind": self.duplicate_kind.value,  # type: ignore[union-attr]
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChurnDecision":
        if not isinstance(payload, Mapping):
            raise ProviderCallLedgerIntegrityError(
                "churn decision must be an object"
            )
        data = {
            key: payload[key]
            for key in (
                "disposition",
                "call_key",
                "should_dispatch",
                "reason",
                "prior_call_id",
                "failure_signature_id",
                "evidence_digest",
                "duplicate_kind",
                "decision_id",
                "recorded_at",
            )
            if key in payload
        }
        return cls(**data)


@dataclass(frozen=True)
class UsageCharge:
    """One charged usage attribution for a call attempt."""

    charge_id: str
    call_id: str
    call_key: str
    charge_kind: ChargeKind | str
    disposition: str
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1
    cost_micros: int = 0
    currency: str = "USD"
    charged: bool = True
    recorded_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "charge_kind",
            _enum(self.charge_kind, ChargeKind, "charge_kind"),
        )
        object.__setattr__(
            self, "call_id", _identifier(self.call_id, "call_id")
        )
        object.__setattr__(
            self, "call_key", _identifier(self.call_key, "call_key")
        )
        object.__setattr__(
            self,
            "disposition",
            _text(self.disposition, "disposition"),
        )
        for name in (
            "input_tokens",
            "output_tokens",
            "requests",
            "cost_micros",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(int(getattr(self, name)), name),
            )
        currency = _text(self.currency, "currency")
        if not re.fullmatch(r"^[A-Z]{3}$", currency):
            raise ProviderCallLedgerIntegrityError(
                "currency must be a three-letter code"
            )
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "charged", bool(self.charged))
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at, "recorded_at", required=False)
            if self.recorded_at
            else "",
        )
        if not self.charge_id:
            object.__setattr__(
                self,
                "charge_id",
                _identity(
                    "charge",
                    {
                        "call_id": self.call_id,
                        "charge_kind": self.charge_kind.value,  # type: ignore[union-attr]
                        "disposition": self.disposition,
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens,
                        "requests": self.requests,
                    },
                ),
            )
        else:
            object.__setattr__(
                self, "charge_id", _identifier(self.charge_id, "charge_id")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": USAGE_CHARGE_SCHEMA,
            "charge_id": self.charge_id,
            "call_id": self.call_id,
            "call_key": self.call_key,
            "charge_kind": self.charge_kind.value,  # type: ignore[union-attr]
            "disposition": self.disposition,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "cost_micros": self.cost_micros,
            "currency": self.currency,
            "charged": self.charged,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UsageCharge":
        if not isinstance(payload, Mapping):
            raise ProviderCallLedgerIntegrityError(
                "usage charge must be an object"
            )
        data = {
            key: payload[key]
            for key in (
                "charge_id",
                "call_id",
                "call_key",
                "charge_kind",
                "disposition",
                "input_tokens",
                "output_tokens",
                "requests",
                "cost_micros",
                "currency",
                "charged",
                "recorded_at",
            )
            if key in payload
        }
        return cls(**data)


@dataclass(frozen=True)
class ProviderCallRecord:
    """Redacted durable provider call row."""

    call_id: str
    call_key: str
    provider: str
    model: str
    context_cid: str
    task_id: str
    proposal_digest: str
    evidence_digest: str
    status: CallStatus | str
    plan_id: str = ""
    attempt: int = 0
    idempotency_key: str = ""
    endpoint_fingerprint: str = ""
    outcome: CallOutcome | str = CallOutcome.UNKNOWN
    outcome_class: OutcomeClass | str = OutcomeClass.UNKNOWN
    failure_signature_id: str = ""
    response_digest: str = ""
    mutation_result: MutationResult | str = MutationResult.NOT_APPLICABLE
    validation_result: ValidationResult | str = ValidationResult.NOT_APPLICABLE
    tokens: ProviderTokenUsage = field(default_factory=ProviderTokenUsage)
    budget: ProviderCallBudget = field(default_factory=ProviderCallBudget)
    latency_ms: int = 0
    dispatched: bool = False
    charged: bool = False
    recorded_at: str = ""
    completed_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "call_id", _identifier(self.call_id, "call_id")
        )
        object.__setattr__(
            self, "call_key", _identifier(self.call_key, "call_key")
        )
        object.__setattr__(
            self, "provider", _identifier(self.provider, "provider")
        )
        object.__setattr__(self, "model", _identifier(self.model, "model"))
        object.__setattr__(
            self, "context_cid", _identifier(self.context_cid, "context_cid")
        )
        object.__setattr__(
            self, "task_id", _identifier(self.task_id, "task_id")
        )
        object.__setattr__(
            self,
            "proposal_digest",
            _identifier(self.proposal_digest, "proposal_digest"),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _identifier(self.evidence_digest, "evidence_digest"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, CallStatus, "status")
        )
        object.__setattr__(
            self, "plan_id", _identifier(self.plan_id, "plan_id", required=False)
        )
        object.__setattr__(
            self,
            "attempt",
            _nonneg_int(int(self.attempt), "attempt", maximum=MAX_ATTEMPTS),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _identifier(
                self.idempotency_key, "idempotency_key", required=False
            ),
        )
        object.__setattr__(
            self,
            "endpoint_fingerprint",
            _identifier(
                self.endpoint_fingerprint,
                "endpoint_fingerprint",
                required=False,
            ),
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, CallOutcome, "outcome")
        )
        object.__setattr__(
            self,
            "outcome_class",
            _enum(self.outcome_class, OutcomeClass, "outcome_class"),
        )
        object.__setattr__(
            self,
            "failure_signature_id",
            _identifier(
                self.failure_signature_id,
                "failure_signature_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "response_digest",
            _identifier(
                self.response_digest, "response_digest", required=False
            ),
        )
        object.__setattr__(
            self,
            "mutation_result",
            _enum(self.mutation_result, MutationResult, "mutation_result"),
        )
        object.__setattr__(
            self,
            "validation_result",
            _enum(
                self.validation_result, ValidationResult, "validation_result"
            ),
        )
        tokens = self.tokens
        if not isinstance(tokens, ProviderTokenUsage):
            if not isinstance(tokens, Mapping):
                raise ProviderCallLedgerIntegrityError("tokens are invalid")
            tokens = ProviderTokenUsage(**dict(tokens))
        object.__setattr__(self, "tokens", tokens)
        budget = self.budget
        if not isinstance(budget, ProviderCallBudget):
            if not isinstance(budget, Mapping):
                raise ProviderCallLedgerIntegrityError("budget is invalid")
            budget = ProviderCallBudget(**dict(budget))
        object.__setattr__(self, "budget", budget)
        object.__setattr__(
            self,
            "latency_ms",
            _nonneg_int(
                int(self.latency_ms), "latency_ms", maximum=MAX_LATENCY_MS
            ),
        )
        object.__setattr__(self, "dispatched", bool(self.dispatched))
        object.__setattr__(self, "charged", bool(self.charged))
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at, "recorded_at", required=False)
            if self.recorded_at
            else "",
        )
        object.__setattr__(
            self,
            "completed_at",
            _text(self.completed_at, "completed_at", required=False)
            if self.completed_at
            else "",
        )
        meta = dict(self.metadata or {})
        _assert_no_forbidden_payload(meta)
        object.__setattr__(self, "metadata", _freeze_mapping(meta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CALL_RECORD_SCHEMA,
            "call_id": self.call_id,
            "call_key": self.call_key,
            "provider": self.provider,
            "model": self.model,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "context_cid": self.context_cid,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "proposal_digest": self.proposal_digest,
            "evidence_digest": self.evidence_digest,
            "status": self.status.value,  # type: ignore[union-attr]
            "outcome": self.outcome.value,  # type: ignore[union-attr]
            "outcome_class": self.outcome_class.value,  # type: ignore[union-attr]
            "failure_signature_id": self.failure_signature_id,
            "response_digest": self.response_digest,
            "mutation_result": self.mutation_result.value,  # type: ignore[union-attr]
            "validation_result": self.validation_result.value,  # type: ignore[union-attr]
            "tokens": self.tokens.to_dict(),
            "budget": self.budget.to_dict(),
            "latency_ms": self.latency_ms,
            "dispatched": self.dispatched,
            "charged": self.charged,
            "recorded_at": self.recorded_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderCallRecord":
        if not isinstance(payload, Mapping):
            raise ProviderCallLedgerIntegrityError(
                "provider call record must be an object"
            )
        data = dict(payload)
        data.pop("schema", None)
        return cls(**data)


@dataclass(frozen=True)
class ProviderCallCompletion:
    """Terminal observations recorded after a dispatched (or suppressed) call."""

    outcome: CallOutcome | str
    failure_code: str = ""
    outcome_class: OutcomeClass | str | None = None
    response_digest: str = ""
    latency_ms: int = 0
    tokens: ProviderTokenUsage = field(default_factory=ProviderTokenUsage)
    mutation_result: MutationResult | str = MutationResult.NOT_APPLICABLE
    validation_result: ValidationResult | str = ValidationResult.NOT_REQUIRED
    cost_micros: int = 0
    currency: str = "USD"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcome", _enum(self.outcome, CallOutcome, "outcome")
        )
        object.__setattr__(
            self,
            "failure_code",
            _identifier(self.failure_code, "failure_code", required=False),
        )
        if self.outcome_class is not None:
            object.__setattr__(
                self,
                "outcome_class",
                _enum(self.outcome_class, OutcomeClass, "outcome_class"),
            )
        object.__setattr__(
            self,
            "response_digest",
            _identifier(
                self.response_digest, "response_digest", required=False
            ),
        )
        object.__setattr__(
            self,
            "latency_ms",
            _nonneg_int(
                int(self.latency_ms), "latency_ms", maximum=MAX_LATENCY_MS
            ),
        )
        tokens = self.tokens
        if not isinstance(tokens, ProviderTokenUsage):
            if not isinstance(tokens, Mapping):
                raise ProviderCallLedgerIntegrityError("tokens are invalid")
            tokens = ProviderTokenUsage(**dict(tokens))
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(
            self,
            "mutation_result",
            _enum(self.mutation_result, MutationResult, "mutation_result"),
        )
        object.__setattr__(
            self,
            "validation_result",
            _enum(
                self.validation_result, ValidationResult, "validation_result"
            ),
        )
        object.__setattr__(
            self,
            "cost_micros",
            _nonneg_int(int(self.cost_micros), "cost_micros"),
        )
        currency = _text(self.currency, "currency")
        if not re.fullmatch(r"^[A-Z]{3}$", currency):
            raise ProviderCallLedgerIntegrityError(
                "currency must be a three-letter code"
            )
        object.__setattr__(self, "currency", currency)
        meta = dict(self.metadata or {})
        _assert_no_forbidden_payload(meta)
        object.__setattr__(self, "metadata", _freeze_mapping(meta))

    def resolved_outcome_class(self) -> OutcomeClass:
        if self.outcome_class is not None:
            return self.outcome_class  # type: ignore[return-value]
        return _outcome_to_class(self.outcome)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class ProviderCallLedger:
    """Persist provider calls, failure signatures, usage, and churn decisions.

    Interface: ``ProviderCallLedger@1``.
    """

    INTERFACE: Final[str] = PROVIDER_CALL_LEDGER_INTERFACE
    SCHEMA: Final[str] = PROVIDER_CALL_LEDGER_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        policy: ChurnPolicy | None = None,
        ledger_version: str = DEFAULT_LEDGER_VERSION,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for ProviderCallLedger; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._policy = policy or ChurnPolicy()
        self._ledger_version = _text(
            ledger_version or DEFAULT_LEDGER_VERSION, "ledger_version"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def policy(self) -> ChurnPolicy:
        return self._policy

    @property
    def ledger_version(self) -> str:
        return self._ledger_version

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "ProviderCallLedger":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", PROVIDER_CALL_LEDGER_INTERFACE),
                ("schema", PROVIDER_CALL_LEDGER_SCHEMA),
                ("ledger_version", self._ledger_version),
                ("authority", AUTHORITY_CLASS),
                ("producer", PRODUCER_ID),
                ("policy", _canonical_json(self._policy.to_dict())),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO provider_call_ledger_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "ProviderCallLedger":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise ProviderCallLedgerNotOpenError()
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def metadata(self) -> dict[str, Any]:
        connection = self._require()
        with self._lock:
            rows = connection.execute(
                "SELECT key, value FROM provider_call_ledger_metadata"
            ).fetchall()
            meta = {str(row[0]): str(row[1]) for row in rows}
            meta["database_path"] = str(self._path)
            meta["is_open"] = True
            return meta

    # -- evaluation ----------------------------------------------------------

    def evaluate_dispatch(
        self,
        proposal: ProviderCallProposal | Mapping[str, Any],
        *,
        now_ms: int | None = None,
        persist: bool = True,
    ) -> ChurnDecision:
        """Decide whether *proposal* may be dispatched to a provider."""

        connection = self._require()
        prop = (
            proposal
            if isinstance(proposal, ProviderCallProposal)
            else ProviderCallProposal.from_dict(proposal)
        )
        clock = (
            _nonneg_int(int(now_ms), "now_ms")
            if now_ms is not None
            else _now_ms()
        )
        with self._lock:
            decision = self._evaluate_locked(connection, prop, now_ms=clock)
            if persist:
                self._persist_decision(connection, decision)
                self._commit_if_idle(connection)
            return decision

    def _evaluate_locked(
        self,
        connection: Any,
        proposal: ProviderCallProposal,
        *,
        now_ms: int,
    ) -> ChurnDecision:
        call_key = proposal.call_key
        existing = self._get_call_by_key_locked(connection, call_key)
        if existing is not None:
            # Exact call-key match always collapses to a single dispatch,
            # whether the prior row is still in-flight or already terminal.
            return ChurnDecision(
                disposition=ChurnDisposition.REPLAY_EXACT,
                call_key=call_key,
                should_dispatch=False,
                reason=(
                    "exact call key already has a terminal record"
                    if _is_terminal_status(existing.status)  # type: ignore[arg-type]
                    else "exact call key already admitted or dispatched"
                ),
                prior_call_id=existing.call_id,
                failure_signature_id=existing.failure_signature_id,
                evidence_digest=existing.evidence_digest,
                duplicate_kind=(
                    DuplicateKind.IDEMPOTENCY
                    if proposal.idempotency_key
                    else DuplicateKind.EXACT
                ),
                recorded_at=_utc_iso(),
            )

        # Negative cache: active suppression for this call key.
        suppression = self._active_suppression_locked(
            connection, call_key=call_key, now_ms=now_ms
        )
        if suppression is not None:
            return ChurnDecision(
                disposition=ChurnDisposition.SUPPRESS_NEGATIVE_CACHE,
                call_key=call_key,
                should_dispatch=False,
                reason="negative cache TTL has not expired",
                prior_call_id=str(suppression.get("prior_call_id") or ""),
                failure_signature_id=str(
                    suppression.get("failure_signature_id") or ""
                ),
                evidence_digest=proposal.evidence_digest,
                duplicate_kind=DuplicateKind.EXACT,
                recorded_at=_utc_iso(),
            )

        # Semantic / failure-signature exhaustion against unchanged evidence.
        semantic_rows = connection.execute(
            """
            SELECT call_id, call_key, evidence_digest, failure_signature_id,
                   outcome, status, proposal_digest, context_cid, attempt
            FROM provider_calls
            WHERE provider = ?
              AND model = ?
              AND context_cid = ?
              AND task_id = ?
              AND proposal_digest = ?
              AND status IN (?, ?, ?, ?, ?)
            ORDER BY attempt ASC, recorded_at ASC
            """,
            [
                proposal.provider,
                proposal.model,
                proposal.context_cid,
                proposal.task_id,
                proposal.proposal_digest,
                CallStatus.FAILED.value,
                CallStatus.REJECTED.value,
                CallStatus.ABANDONED.value,
                CallStatus.COMPLETED.value,
                CallStatus.SUPPRESSED.value,
            ],
        ).fetchall()

        same_evidence_failures = 0
        last_failed_call_id = ""
        last_failure_signature_id = ""
        for row in semantic_rows:
            # DuckDBRow iterates keys, not values — index access only.
            call_id = row[0]
            evidence_digest = row[2]
            failure_signature_id = row[3]
            outcome = row[4]
            status = row[5]
            if str(evidence_digest) != proposal.evidence_digest:
                continue
            outcome_enum = (
                CallOutcome(str(outcome)) if outcome else CallOutcome.UNKNOWN
            )
            status_enum = CallStatus(str(status))
            if (
                status_enum == CallStatus.COMPLETED
                and outcome_enum == CallOutcome.SUCCESS
            ):
                continue
            if _is_failed_outcome(outcome_enum) or status_enum in {
                CallStatus.FAILED,
                CallStatus.REJECTED,
                CallStatus.ABANDONED,
                CallStatus.SUPPRESSED,
            }:
                same_evidence_failures += 1
                last_failed_call_id = str(call_id)
                last_failure_signature_id = str(failure_signature_id or "")

        if same_evidence_failures:
            # Retry budget counts prior failed attempts for the same proposal+evidence.
            if same_evidence_failures >= self._policy.max_retries:
                return ChurnDecision(
                    disposition=ChurnDisposition.SUPPRESS_EXHAUSTED,
                    call_key=call_key,
                    should_dispatch=False,
                    reason="retry budget exhausted for unchanged failed proposal",
                    prior_call_id=last_failed_call_id,
                    failure_signature_id=last_failure_signature_id,
                    evidence_digest=proposal.evidence_digest,
                    duplicate_kind=DuplicateKind.SEMANTIC,
                    recorded_at=_utc_iso(),
                )
            if (
                self._policy.suppress_semantic_duplicates
                and same_evidence_failures
                >= self._policy.max_identical_failures
            ):
                return ChurnDecision(
                    disposition=ChurnDisposition.SUPPRESS_SEMANTIC_DUPLICATE,
                    call_key=call_key,
                    should_dispatch=False,
                    reason="identical failed proposal suppressed by policy",
                    prior_call_id=last_failed_call_id,
                    failure_signature_id=last_failure_signature_id,
                    evidence_digest=proposal.evidence_digest,
                    duplicate_kind=DuplicateKind.SEMANTIC,
                    recorded_at=_utc_iso(),
                )

        # Different evidence after prior failures is an explicit reopen.
        prior_any = connection.execute(
            """
            SELECT call_id, failure_signature_id, evidence_digest
            FROM provider_calls
            WHERE provider = ?
              AND model = ?
              AND context_cid = ?
              AND task_id = ?
              AND proposal_digest = ?
              AND evidence_digest != ?
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            [
                proposal.provider,
                proposal.model,
                proposal.context_cid,
                proposal.task_id,
                proposal.proposal_digest,
                proposal.evidence_digest,
            ],
        ).fetchone()
        if prior_any is not None:
            return ChurnDecision(
                disposition=ChurnDisposition.ALLOW_CHANGED_EVIDENCE,
                call_key=call_key,
                should_dispatch=True,
                reason="changed evidence permits a new provider call",
                prior_call_id=str(prior_any[0]),
                failure_signature_id=str(prior_any[1] or ""),
                evidence_digest=proposal.evidence_digest,
                duplicate_kind=DuplicateKind.NONE,
                recorded_at=_utc_iso(),
            )

        return ChurnDecision(
            disposition=ChurnDisposition.DISPATCH,
            call_key=call_key,
            should_dispatch=True,
            reason="no suppressing prior call or active negative cache",
            evidence_digest=proposal.evidence_digest,
            duplicate_kind=DuplicateKind.NONE,
            recorded_at=_utc_iso(),
        )

    def _active_suppression_locked(
        self,
        connection: Any,
        *,
        call_key: str,
        now_ms: int,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT suppression_id, failure_signature_id, decision, reason,
                   evidence_digest, expires_at_ms, body_json
            FROM replay_suppressions
            WHERE call_key = ?
              AND (expires_at_ms = 0 OR expires_at_ms > ?)
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            [call_key, now_ms],
        ).fetchone()
        if row is None:
            return None
        body: dict[str, Any] = {}
        try:
            body = json.loads(str(row[6] or "{}"))
        except json.JSONDecodeError:
            body = {}
        return {
            "suppression_id": str(row[0]),
            "failure_signature_id": str(row[1] or ""),
            "decision": str(row[2]),
            "reason": str(row[3] or ""),
            "evidence_digest": str(row[4] or ""),
            "expires_at_ms": int(row[5] or 0),
            "prior_call_id": str(body.get("prior_call_id") or ""),
        }

    def _persist_decision(
        self, connection: Any, decision: ChurnDecision
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO churn_decisions(
                decision_id, call_key, disposition, should_dispatch, reason,
                prior_call_id, failure_signature_id, evidence_digest,
                duplicate_kind, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision.decision_id,
                decision.call_key,
                decision.disposition.value,  # type: ignore[union-attr]
                1 if decision.should_dispatch else 0,
                decision.reason,
                decision.prior_call_id,
                decision.failure_signature_id,
                decision.evidence_digest,
                decision.duplicate_kind.value,  # type: ignore[union-attr]
                decision.recorded_at or _utc_iso(),
                _canonical_json(decision.to_dict()),
            ],
        )

    # -- admit / complete ----------------------------------------------------

    def admit_and_dispatch(
        self,
        proposal: ProviderCallProposal | Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> tuple[ChurnDecision, ProviderCallRecord | None]:
        """Evaluate churn and, when allowed, admit a dispatched call row.

        Exact replays return the prior terminal record without creating a new
        dispatch.  Suppressions admit a non-dispatched suppressed row and still
        record a usage charge when policy requires it.
        """

        connection = self._require()
        prop = (
            proposal
            if isinstance(proposal, ProviderCallProposal)
            else ProviderCallProposal.from_dict(proposal)
        )
        clock = (
            _nonneg_int(int(now_ms), "now_ms")
            if now_ms is not None
            else _now_ms()
        )
        with self._lock:
            decision = self._evaluate_locked(connection, prop, now_ms=clock)
            self._persist_decision(connection, decision)

            if decision.disposition is ChurnDisposition.REPLAY_EXACT:
                prior = self._get_call_by_key_locked(connection, prop.call_key)
                self._commit_if_idle(connection)
                return decision, prior

            if not decision.should_dispatch:
                record = self._insert_call_locked(
                    connection,
                    prop,
                    status=CallStatus.SUPPRESSED,
                    outcome=CallOutcome.SUPPRESSED,
                    outcome_class=OutcomeClass.POLICY,
                    dispatched=False,
                    failure_signature_id=decision.failure_signature_id,
                )
                # Suppressed re-prompts still get a zero-or-estimated charge so
                # accounting never drops the attempt.
                self._charge_locked(
                    connection,
                    record,
                    charge_kind=ChargeKind.SUPPRESSED,
                    disposition=decision.disposition.value,  # type: ignore[union-attr]
                    tokens=prop.token_estimate,
                    cost_micros=0,
                )
                if self._policy.negative_cache_ttl_ms > 0:
                    self._write_suppression_locked(
                        connection,
                        call_key=prop.call_key,
                        decision=decision,
                        now_ms=clock,
                    )
                self._commit_if_idle(connection)
                return decision, record

            record = self._insert_call_locked(
                connection,
                prop,
                status=CallStatus.DISPATCHED,
                outcome=CallOutcome.UNKNOWN,
                outcome_class=OutcomeClass.UNKNOWN,
                dispatched=True,
            )
            self._commit_if_idle(connection)
            return decision, record

    def complete_call(
        self,
        call_id: str,
        completion: ProviderCallCompletion | Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> ProviderCallRecord:
        """Record terminal outcome, response meta, failure signature, and charge."""

        connection = self._require()
        comp = (
            completion
            if isinstance(completion, ProviderCallCompletion)
            else ProviderCallCompletion(
                **{
                    key: completion[key]
                    for key in completion  # type: ignore[union-attr]
                    if key
                    in {
                        "outcome",
                        "failure_code",
                        "outcome_class",
                        "response_digest",
                        "latency_ms",
                        "tokens",
                        "mutation_result",
                        "validation_result",
                        "cost_micros",
                        "currency",
                        "metadata",
                    }
                }
            )
        )
        clock = (
            _nonneg_int(int(now_ms), "now_ms")
            if now_ms is not None
            else _now_ms()
        )
        with self._lock:
            existing = self._get_call_by_id_locked(connection, call_id)
            if existing is None:
                raise ProviderCallLedgerIntegrityError(
                    f"unknown call_id: {call_id}"
                )
            if _is_terminal_status(existing.status) and existing.status not in {  # type: ignore[arg-type]
                CallStatus.DISPATCHED,
                CallStatus.ADMITTED,
            }:
                # Idempotent complete of an already terminal call returns it.
                if existing.outcome == comp.outcome:
                    return existing
                raise ProviderCallLedgerConflictError(
                    "call is already terminal with a different outcome"
                )

            outcome = comp.outcome  # type: ignore[assignment]
            outcome_class = comp.resolved_outcome_class()
            status = self._status_for_outcome(outcome)  # type: ignore[arg-type]
            failure_signature_id = ""
            signature: FailureSignature | None = None
            if _is_failed_outcome(outcome):  # type: ignore[arg-type]
                proposal_view = ProviderCallProposal(
                    provider=existing.provider,
                    model=existing.model,
                    context_cid=existing.context_cid,
                    task_id=existing.task_id,
                    proposal_digest=existing.proposal_digest,
                    evidence_digest=existing.evidence_digest,
                    plan_id=existing.plan_id,
                    attempt=existing.attempt,
                    idempotency_key=existing.idempotency_key,
                    endpoint_fingerprint=existing.endpoint_fingerprint,
                )
                signature = FailureSignature.from_outcome(
                    proposal=proposal_view,
                    outcome=outcome,  # type: ignore[arg-type]
                    failure_code=comp.failure_code
                    or outcome.value,  # type: ignore[union-attr]
                    outcome_class=outcome_class,
                )
                failure_signature_id = signature.failure_signature_id
                self._upsert_failure_signature_locked(
                    connection,
                    signature,
                    call_id=existing.call_id,
                    now_iso=_utc_iso(),
                )

            tokens = ProviderTokenUsage(
                input_tokens_estimated=existing.tokens.input_tokens_estimated,
                output_tokens_estimated=existing.tokens.output_tokens_estimated,
                input_tokens_actual=comp.tokens.input_tokens_actual
                or existing.tokens.input_tokens_actual,
                output_tokens_actual=comp.tokens.output_tokens_actual
                or existing.tokens.output_tokens_actual,
            )
            completed_at = _utc_iso()
            meta = dict(existing.metadata)
            meta.update(dict(comp.metadata))
            _assert_no_forbidden_payload(meta)
            record = ProviderCallRecord(
                call_id=existing.call_id,
                call_key=existing.call_key,
                provider=existing.provider,
                model=existing.model,
                context_cid=existing.context_cid,
                task_id=existing.task_id,
                proposal_digest=existing.proposal_digest,
                evidence_digest=existing.evidence_digest,
                status=status,
                plan_id=existing.plan_id,
                attempt=existing.attempt,
                idempotency_key=existing.idempotency_key,
                endpoint_fingerprint=existing.endpoint_fingerprint,
                outcome=outcome,
                outcome_class=outcome_class,
                failure_signature_id=failure_signature_id,
                response_digest=comp.response_digest,
                mutation_result=comp.mutation_result,
                validation_result=comp.validation_result,
                tokens=tokens,
                budget=existing.budget,
                latency_ms=comp.latency_ms,
                dispatched=existing.dispatched,
                charged=True,
                recorded_at=existing.recorded_at,
                completed_at=completed_at,
                metadata=meta,
            )
            self._update_call_locked(connection, record)
            if comp.response_digest:
                self._insert_response_meta_locked(connection, record)
            charge_kind = _outcome_to_charge_kind(outcome)  # type: ignore[arg-type]
            if self._should_charge(outcome):  # type: ignore[arg-type]
                self._charge_locked(
                    connection,
                    record,
                    charge_kind=charge_kind,
                    disposition=outcome.value,  # type: ignore[union-attr]
                    tokens=tokens,
                    cost_micros=comp.cost_micros,
                    currency=comp.currency,
                )
            if (
                signature is not None
                and self._policy.negative_cache_ttl_ms > 0
                and _is_failed_outcome(outcome)  # type: ignore[arg-type]
            ):
                # Seed negative cache only after policy exhaustion would apply
                # on the next identical proposal; always write when max_retries
                # is already reached by counting this completion.
                failures = self._count_same_evidence_failures_locked(
                    connection,
                    provider=record.provider,
                    model=record.model,
                    context_cid=record.context_cid,
                    task_id=record.task_id,
                    proposal_digest=record.proposal_digest,
                    evidence_digest=record.evidence_digest,
                )
                if failures >= self._policy.max_retries:
                    decision = ChurnDecision(
                        disposition=ChurnDisposition.SUPPRESS_EXHAUSTED,
                        call_key=record.call_key,
                        should_dispatch=False,
                        reason="retry budget exhausted after terminal failure",
                        prior_call_id=record.call_id,
                        failure_signature_id=failure_signature_id,
                        evidence_digest=record.evidence_digest,
                        duplicate_kind=DuplicateKind.SEMANTIC,
                        recorded_at=completed_at,
                    )
                    self._write_suppression_locked(
                        connection,
                        call_key=record.call_key,
                        decision=decision,
                        now_ms=clock,
                    )
            self._commit_if_idle(connection)
            return record

    def record_usage(
        self,
        *,
        call_id: str,
        charge_kind: ChargeKind | str,
        disposition: str,
        tokens: ProviderTokenUsage | Mapping[str, Any] | None = None,
        cost_micros: int = 0,
        currency: str = "USD",
        requests: int = 1,
    ) -> UsageCharge:
        """Explicitly charge usage for rejected / abandoned / retry paths."""

        connection = self._require()
        with self._lock:
            record = self._get_call_by_id_locked(connection, call_id)
            if record is None:
                raise ProviderCallLedgerIntegrityError(
                    f"unknown call_id: {call_id}"
                )
            resolved_tokens = (
                ProviderTokenUsage(**dict(tokens))
                if isinstance(tokens, Mapping)
                else tokens or record.tokens
            )
            charge = self._charge_locked(
                connection,
                record,
                charge_kind=_enum(charge_kind, ChargeKind, "charge_kind"),
                disposition=disposition,
                tokens=resolved_tokens,
                cost_micros=cost_micros,
                currency=currency,
                requests=requests,
            )
            connection.execute(
                "UPDATE provider_calls SET charged = 1 WHERE call_id = ?",
                [call_id],
            )
            self._commit_if_idle(connection)
            return charge

    # -- queries -------------------------------------------------------------

    def get_call(self, call_id: str) -> ProviderCallRecord | None:
        connection = self._require()
        with self._lock:
            return self._get_call_by_id_locked(connection, call_id)

    def get_call_by_key(self, call_key: str) -> ProviderCallRecord | None:
        connection = self._require()
        with self._lock:
            return self._get_call_by_key_locked(connection, call_key)

    def list_usage_charges(
        self, *, call_id: str | None = None
    ) -> tuple[UsageCharge, ...]:
        connection = self._require()
        with self._lock:
            if call_id:
                rows = connection.execute(
                    """
                    SELECT body_json FROM usage_charges
                    WHERE call_id = ?
                    ORDER BY recorded_at ASC
                    """,
                    [call_id],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT body_json FROM usage_charges
                    ORDER BY recorded_at ASC
                    """
                ).fetchall()
            return tuple(
                UsageCharge.from_dict(json.loads(str(row[0]))) for row in rows
            )

    def get_failure_signature(
        self, failure_signature_id: str
    ) -> dict[str, Any] | None:
        connection = self._require()
        with self._lock:
            row = connection.execute(
                """
                SELECT body_json FROM failure_signatures
                WHERE failure_signature_id = ?
                """,
                [failure_signature_id],
            ).fetchone()
            if row is None:
                return None
            return json.loads(str(row[0]))

    def list_churn_decisions(
        self, *, call_key: str | None = None
    ) -> tuple[ChurnDecision, ...]:
        connection = self._require()
        with self._lock:
            if call_key:
                rows = connection.execute(
                    """
                    SELECT body_json FROM churn_decisions
                    WHERE call_key = ?
                    ORDER BY recorded_at ASC
                    """,
                    [call_key],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT body_json FROM churn_decisions
                    ORDER BY recorded_at ASC
                    """
                ).fetchall()
            return tuple(
                ChurnDecision.from_dict(json.loads(str(row[0]))) for row in rows
            )

    def total_charged_tokens(
        self, *, include_kinds: Sequence[ChargeKind | str] | None = None
    ) -> dict[str, int]:
        """Sum charged input/output tokens, optionally filtered by charge kind."""

        charges = self.list_usage_charges()
        kinds: set[str] | None = None
        if include_kinds is not None:
            kinds = {
                _enum(item, ChargeKind, "charge_kind").value
                for item in include_kinds
            }
        input_tokens = 0
        output_tokens = 0
        requests = 0
        for charge in charges:
            if not charge.charged:
                continue
            if kinds is not None and charge.charge_kind.value not in kinds:  # type: ignore[union-attr]
                continue
            input_tokens += charge.input_tokens
            output_tokens += charge.output_tokens
            requests += charge.requests
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests": requests,
        }

    # -- internal persistence helpers ----------------------------------------

    def _status_for_outcome(self, outcome: CallOutcome) -> CallStatus:
        if outcome is CallOutcome.SUCCESS:
            return CallStatus.COMPLETED
        if outcome is CallOutcome.REJECTED:
            return CallStatus.REJECTED
        if outcome in {CallOutcome.ABANDONED, CallOutcome.CANCELLED}:
            return CallStatus.ABANDONED
        if outcome is CallOutcome.SUPPRESSED:
            return CallStatus.SUPPRESSED
        if outcome is CallOutcome.REPLAYED:
            return CallStatus.REPLAYED
        return CallStatus.FAILED

    def _should_charge(self, outcome: CallOutcome) -> bool:
        if outcome is CallOutcome.REJECTED:
            return bool(self._policy.charge_rejected)
        if outcome in {CallOutcome.ABANDONED, CallOutcome.CANCELLED}:
            return bool(self._policy.charge_abandoned)
        if outcome is CallOutcome.RETRY:
            return bool(self._policy.charge_retry)
        # Success, failed, transient, hard quota, response loss always charge.
        return True

    def _insert_call_locked(
        self,
        connection: Any,
        proposal: ProviderCallProposal,
        *,
        status: CallStatus,
        outcome: CallOutcome,
        outcome_class: OutcomeClass,
        dispatched: bool,
        failure_signature_id: str = "",
    ) -> ProviderCallRecord:
        recorded_at = _utc_iso()
        call_id = _identity(
            "pcall",
            {
                "call_key": proposal.call_key,
                "recorded_at": recorded_at,
                "status": status.value,
                "attempt": proposal.attempt,
            },
        )
        # Exact call_key uniqueness: if a non-terminal row already exists, reuse.
        existing = self._get_call_by_key_locked(connection, proposal.call_key)
        if existing is not None:
            if _is_terminal_status(existing.status):  # type: ignore[arg-type]
                return existing
            return existing

        record = ProviderCallRecord(
            call_id=call_id,
            call_key=proposal.call_key,
            provider=proposal.provider,
            model=proposal.model,
            context_cid=proposal.context_cid,
            task_id=proposal.task_id,
            proposal_digest=proposal.proposal_digest,
            evidence_digest=proposal.evidence_digest,
            status=status,
            plan_id=proposal.plan_id,
            attempt=proposal.attempt,
            idempotency_key=proposal.idempotency_key,
            endpoint_fingerprint=proposal.endpoint_fingerprint,
            outcome=outcome,
            outcome_class=outcome_class,
            failure_signature_id=failure_signature_id,
            tokens=proposal.token_estimate,
            budget=proposal.budget,
            dispatched=dispatched,
            charged=False,
            recorded_at=recorded_at,
            metadata=dict(proposal.metadata),
        )
        body = _canonical_json(record.to_dict())
        if len(body.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            raise ProviderCallLedgerBoundsError("call body exceeds bound")
        connection.execute(
            """
            INSERT INTO provider_calls(
                call_id, call_key, idempotency_key, provider, model,
                endpoint_fingerprint, context_cid, plan_id, task_id, attempt,
                proposal_digest, evidence_digest, status, outcome, outcome_class,
                failure_signature_id, response_digest, mutation_result,
                validation_result, input_tokens_estimated, output_tokens_estimated,
                input_tokens_actual, output_tokens_actual, latency_ms,
                budget_requests, budget_input_tokens, budget_output_tokens,
                dispatched, charged, recorded_at, completed_at, body_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                record.call_id,
                record.call_key,
                record.idempotency_key,
                record.provider,
                record.model,
                record.endpoint_fingerprint,
                record.context_cid,
                record.plan_id,
                record.task_id,
                record.attempt,
                record.proposal_digest,
                record.evidence_digest,
                record.status.value,  # type: ignore[union-attr]
                record.outcome.value,  # type: ignore[union-attr]
                record.outcome_class.value,  # type: ignore[union-attr]
                record.failure_signature_id,
                record.response_digest,
                record.mutation_result.value,  # type: ignore[union-attr]
                record.validation_result.value,  # type: ignore[union-attr]
                record.tokens.input_tokens_estimated,
                record.tokens.output_tokens_estimated,
                record.tokens.input_tokens_actual,
                record.tokens.output_tokens_actual,
                record.latency_ms,
                record.budget.requests,
                record.budget.input_tokens,
                record.budget.output_tokens,
                1 if record.dispatched else 0,
                1 if record.charged else 0,
                record.recorded_at,
                record.completed_at,
                body,
            ],
        )
        return record

    def _update_call_locked(
        self, connection: Any, record: ProviderCallRecord
    ) -> None:
        body = _canonical_json(record.to_dict())
        if len(body.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            raise ProviderCallLedgerBoundsError("call body exceeds bound")
        connection.execute(
            """
            UPDATE provider_calls SET
                status = ?,
                outcome = ?,
                outcome_class = ?,
                failure_signature_id = ?,
                response_digest = ?,
                mutation_result = ?,
                validation_result = ?,
                input_tokens_estimated = ?,
                output_tokens_estimated = ?,
                input_tokens_actual = ?,
                output_tokens_actual = ?,
                latency_ms = ?,
                dispatched = ?,
                charged = ?,
                completed_at = ?,
                body_json = ?
            WHERE call_id = ?
            """,
            [
                record.status.value,  # type: ignore[union-attr]
                record.outcome.value,  # type: ignore[union-attr]
                record.outcome_class.value,  # type: ignore[union-attr]
                record.failure_signature_id,
                record.response_digest,
                record.mutation_result.value,  # type: ignore[union-attr]
                record.validation_result.value,  # type: ignore[union-attr]
                record.tokens.input_tokens_estimated,
                record.tokens.output_tokens_estimated,
                record.tokens.input_tokens_actual,
                record.tokens.output_tokens_actual,
                record.latency_ms,
                1 if record.dispatched else 0,
                1 if record.charged else 0,
                record.completed_at,
                body,
                record.call_id,
            ],
        )

    def _insert_response_meta_locked(
        self, connection: Any, record: ProviderCallRecord
    ) -> None:
        response_id = _identity(
            "presp",
            {
                "call_id": record.call_id,
                "response_digest": record.response_digest,
            },
        )
        body = {
            "schema": PROVIDER_RESPONSE_META_SCHEMA,
            "response_id": response_id,
            "call_id": record.call_id,
            "call_key": record.call_key,
            "response_digest": record.response_digest,
            "outcome": record.outcome.value,  # type: ignore[union-attr]
            "outcome_class": record.outcome_class.value,  # type: ignore[union-attr]
            "latency_ms": record.latency_ms,
            "input_tokens_actual": record.tokens.input_tokens_actual,
            "output_tokens_actual": record.tokens.output_tokens_actual,
        }
        _assert_no_forbidden_payload(body)
        connection.execute(
            """
            INSERT OR REPLACE INTO provider_responses(
                response_id, call_id, call_key, response_digest, outcome,
                outcome_class, latency_ms, input_tokens_actual,
                output_tokens_actual, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                response_id,
                record.call_id,
                record.call_key,
                record.response_digest,
                record.outcome.value,  # type: ignore[union-attr]
                record.outcome_class.value,  # type: ignore[union-attr]
                record.latency_ms,
                record.tokens.input_tokens_actual,
                record.tokens.output_tokens_actual,
                record.completed_at or _utc_iso(),
                _canonical_json(body),
            ],
        )

    def _upsert_failure_signature_locked(
        self,
        connection: Any,
        signature: FailureSignature,
        *,
        call_id: str,
        now_iso: str,
    ) -> None:
        existing = connection.execute(
            """
            SELECT occurrence_count, identical_failures, first_observed_at,
                   body_json
            FROM failure_signatures
            WHERE failure_signature_id = ?
            """,
            [signature.failure_signature_id],
        ).fetchone()
        if existing is None:
            occurrence = 1
            identical = 0
            first = now_iso
        else:
            occurrence = int(existing[0] or 0) + 1
            identical = int(existing[1] or 0) + 1
            first = str(existing[2] or now_iso)
        body = signature.to_dict()
        body.update(
            {
                "occurrence_count": occurrence,
                "identical_failures": identical,
                "last_call_id": call_id,
                "first_observed_at": first,
                "last_observed_at": now_iso,
            }
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO failure_signatures(
                failure_signature_id, outcome_class, failure_code,
                proposal_digest, context_cid, evidence_digest, provider, model,
                occurrence_count, identical_failures, last_call_id,
                first_observed_at, last_observed_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                signature.failure_signature_id,
                signature.outcome_class.value,  # type: ignore[union-attr]
                signature.failure_code,
                signature.proposal_digest,
                signature.context_cid,
                signature.evidence_digest,
                signature.provider,
                signature.model,
                occurrence,
                identical,
                call_id,
                first,
                now_iso,
                _canonical_json(body),
            ],
        )

    def _charge_locked(
        self,
        connection: Any,
        record: ProviderCallRecord,
        *,
        charge_kind: ChargeKind,
        disposition: str,
        tokens: ProviderTokenUsage,
        cost_micros: int = 0,
        currency: str = "USD",
        requests: int = 1,
    ) -> UsageCharge:
        charge = UsageCharge(
            charge_id="",
            call_id=record.call_id,
            call_key=record.call_key,
            charge_kind=charge_kind,
            disposition=disposition,
            input_tokens=tokens.charged_input_tokens,
            output_tokens=tokens.charged_output_tokens,
            requests=requests,
            cost_micros=cost_micros,
            currency=currency,
            charged=True,
            recorded_at=_utc_iso(),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO usage_charges(
                charge_id, call_id, call_key, charge_kind, disposition,
                input_tokens, output_tokens, requests, cost_micros, currency,
                charged, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                charge.charge_id,
                charge.call_id,
                charge.call_key,
                charge.charge_kind.value,  # type: ignore[union-attr]
                charge.disposition,
                charge.input_tokens,
                charge.output_tokens,
                charge.requests,
                charge.cost_micros,
                charge.currency,
                1 if charge.charged else 0,
                charge.recorded_at,
                _canonical_json(charge.to_dict()),
            ],
        )
        return charge

    def _write_suppression_locked(
        self,
        connection: Any,
        *,
        call_key: str,
        decision: ChurnDecision,
        now_ms: int,
    ) -> None:
        expires = (
            now_ms + self._policy.negative_cache_ttl_ms
            if self._policy.negative_cache_ttl_ms > 0
            else 0
        )
        suppression_id = _identity(
            "suppress",
            {
                "call_key": call_key,
                "decision": decision.disposition.value,  # type: ignore[union-attr]
                "failure_signature_id": decision.failure_signature_id,
                "expires_at_ms": expires,
                "prior_call_id": decision.prior_call_id,
            },
        )
        body = {
            "suppression_id": suppression_id,
            "call_key": call_key,
            "decision": decision.disposition.value,  # type: ignore[union-attr]
            "reason": decision.reason,
            "failure_signature_id": decision.failure_signature_id,
            "evidence_digest": decision.evidence_digest,
            "prior_call_id": decision.prior_call_id,
            "expires_at_ms": expires,
        }
        connection.execute(
            """
            INSERT OR REPLACE INTO replay_suppressions(
                suppression_id, call_key, failure_signature_id, decision,
                reason, evidence_digest, expires_at_ms, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                suppression_id,
                call_key,
                decision.failure_signature_id,
                decision.disposition.value,  # type: ignore[union-attr]
                decision.reason,
                decision.evidence_digest,
                expires,
                _utc_iso(),
                _canonical_json(body),
            ],
        )

    def _count_same_evidence_failures_locked(
        self,
        connection: Any,
        *,
        provider: str,
        model: str,
        context_cid: str,
        task_id: str,
        proposal_digest: str,
        evidence_digest: str,
    ) -> int:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM provider_calls
            WHERE provider = ?
              AND model = ?
              AND context_cid = ?
              AND task_id = ?
              AND proposal_digest = ?
              AND evidence_digest = ?
              AND (
                    status IN (?, ?, ?, ?)
                    OR outcome IN (?, ?, ?, ?, ?, ?, ?)
                  )
            """,
            [
                provider,
                model,
                context_cid,
                task_id,
                proposal_digest,
                evidence_digest,
                CallStatus.FAILED.value,
                CallStatus.REJECTED.value,
                CallStatus.ABANDONED.value,
                CallStatus.SUPPRESSED.value,
                CallOutcome.FAILED.value,
                CallOutcome.REJECTED.value,
                CallOutcome.ABANDONED.value,
                CallOutcome.HARD_QUOTA.value,
                CallOutcome.TRANSIENT_FAILURE.value,
                CallOutcome.RESPONSE_LOSS.value,
                CallOutcome.TIMEOUT.value,
            ],
        ).fetchone()
        return int(row[0] if row else 0)

    def _get_call_by_id_locked(
        self, connection: Any, call_id: str
    ) -> ProviderCallRecord | None:
        row = connection.execute(
            "SELECT body_json FROM provider_calls WHERE call_id = ?",
            [call_id],
        ).fetchone()
        if row is None:
            return None
        return ProviderCallRecord.from_dict(json.loads(str(row[0])))

    def _get_call_by_key_locked(
        self, connection: Any, call_key: str
    ) -> ProviderCallRecord | None:
        row = connection.execute(
            "SELECT body_json FROM provider_calls WHERE call_key = ?",
            [call_key],
        ).fetchone()
        if row is None:
            return None
        return ProviderCallRecord.from_dict(json.loads(str(row[0])))


def open_provider_call_ledger(
    database_path: Path | str,
    *,
    policy: ChurnPolicy | None = None,
    ledger_version: str = DEFAULT_LEDGER_VERSION,
) -> ProviderCallLedger:
    """Open a durable provider call ledger at *database_path*."""

    return ProviderCallLedger(
        database_path,
        policy=policy,
        ledger_version=ledger_version,
    ).open()


def build_call_key(
    *,
    provider: str,
    model: str,
    context_cid: str,
    task_id: str,
    proposal_digest: str,
    evidence_digest: str,
    plan_id: str = "",
    attempt: int = 0,
    idempotency_key: str = "",
    endpoint_fingerprint: str = "",
    prompt_digest: str = "",
) -> str:
    """Compute the redacted call key for a proposal without opening a ledger."""

    return ProviderCallProposal(
        provider=provider,
        model=model,
        context_cid=context_cid,
        task_id=task_id,
        proposal_digest=proposal_digest,
        evidence_digest=evidence_digest,
        plan_id=plan_id,
        attempt=attempt,
        idempotency_key=idempotency_key,
        endpoint_fingerprint=endpoint_fingerprint,
        prompt_digest=prompt_digest,
    ).call_key


def digest_text(value: str) -> str:
    """Content digest for prompt/completion bodies that must not be stored raw."""

    return _sha256_text(str(value))


__all__ = (
    "AUTHORITY_CLASS",
    "CHURN_DECISION_INTERFACE",
    "CallOutcome",
    "CallStatus",
    "ChargeKind",
    "ChurnDecision",
    "ChurnDisposition",
    "ChurnPolicy",
    "DuckDBUnavailableError",
    "DuplicateKind",
    "FAILURE_SIGNATURE_INTERFACE",
    "FailureSignature",
    "LEDGER_AUTHORIZES_USAGE",
    "LEDGER_IS_COMPLETION_EVIDENCE",
    "LEDGER_IS_CORRECTNESS_EVIDENCE",
    "LEDGER_REWRITES_PROVIDER_SETTLEMENT",
    "MutationResult",
    "OutcomeClass",
    "PROVIDER_CALL_LEDGER_INTERFACE",
    "ProviderCallBudget",
    "ProviderCallCompletion",
    "ProviderCallLedger",
    "ProviderCallLedgerBoundsError",
    "ProviderCallLedgerConflictError",
    "ProviderCallLedgerError",
    "ProviderCallLedgerIntegrityError",
    "ProviderCallLedgerNotOpenError",
    "ProviderCallLedgerSecretError",
    "ProviderCallProposal",
    "ProviderCallRecord",
    "ProviderTokenUsage",
    "UsageCharge",
    "ValidationResult",
    "build_call_key",
    "digest_text",
    "duckdb_available",
    "open_provider_call_ledger",
)
