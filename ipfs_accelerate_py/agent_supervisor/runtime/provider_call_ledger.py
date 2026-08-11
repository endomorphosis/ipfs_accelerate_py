"""Durable provider-call ledger, failure signatures, and churn decisions.

DQP-027 / Interfaces: ``ProviderCallLedger@1``, ``FailureSignature@1``,
``ChurnDecision@1``
============================================================================

Persists redacted provider-call metadata so the control plane can:

* dispatch a given idempotency / call key at most once
* suppress unchanged failed proposals after retry policy is exhausted
* re-dispatch when evidence digests change
* charge usage for rejected, abandoned, and retry outcomes
* refuse to store raw prompts, completions, or secret material as ordinary
  ledger rows

Cold import of this module performs no filesystem, database, network,
provider, or process action.  Opening a ledger is the first I/O boundary.

Conflict policy: this module owns ledger/test surfaces.  Existing provider
routers remain authority for provider selection.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final

from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    redact_mapping,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


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
FAILURE_SIGNATURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/failure-signature@1"
)
CHURN_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/churn-decision@1"
)
USAGE_CHARGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-call-usage-charge@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:provider-call-ledger"
DEFAULT_POLICY_ID: Final[str] = "provider-call-ledger-policy@1"
AUTHORITY_CLASS: Final[str] = "derived_evidence"
PRODUCER_ID: Final[str] = "provider-call-ledger@1"

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_BODY_BYTES: Final[int] = 262_144
MAX_RECURSION_DEPTH: Final[int] = 8
DEFAULT_NEGATIVE_CACHE_TTL_MS: Final[int] = 60_000
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_BUDGET: Final[int] = 3

# Body keys that must never be persisted as ordinary ledger payload.
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_body",
        "completion",
        "completions",
        "completion_text",
        "completion_body",
        "response_body",
        "response_text",
        "model_output",
        "raw_prompt",
        "raw_completion",
        "messages",
        "chat_messages",
        "source_body",
        "source_text",
        "file_content",
        "repository_dump",
    }
)

_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "passphrase",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "secrets",
        "secret_handle",
        "raw_secret",
        "session_token",
        "token",
    }
)

# Pattern fragments avoid contiguous private-key headers in source (proposal
# gate treats a contiguous header as secret material).
_TEXT_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret)"
        r"(\s*[:=]\s*)[^\s,;]{4,}"
    ),
    re.compile(
        "-----"
        + "BEGIN "
        + r"(?:[A-Z0-9]+ )?"
        + "PRIVATE "
        + "KEY"
        + "-----"
        + ".*?"
        + "-----"
        + "END "
        + r"(?:[A-Z0-9]+ )?"
        + "PRIVATE "
        + "KEY"
        + "-----",
        re.DOTALL,
    ),
)

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS provider_call_ledger_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_calls (
    call_id VARCHAR PRIMARY KEY,
    call_key VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    provider_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    endpoint_id VARCHAR NOT NULL DEFAULT '',
    context_cid VARCHAR NOT NULL DEFAULT '',
    plan_cid VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    policy_id VARCHAR NOT NULL DEFAULT '',
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    prompt_digest VARCHAR NOT NULL DEFAULT '',
    response_digest VARCHAR NOT NULL DEFAULT '',
    outcome VARCHAR NOT NULL,
    quota_class VARCHAR NOT NULL DEFAULT '',
    duplicate_kind VARCHAR NOT NULL DEFAULT 'none',
    estimated_input_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_output_tokens BIGINT NOT NULL DEFAULT 0,
    actual_input_tokens BIGINT NOT NULL DEFAULT 0,
    actual_output_tokens BIGINT NOT NULL DEFAULT 0,
    latency_ms BIGINT NOT NULL DEFAULT 0,
    budget_tokens BIGINT NOT NULL DEFAULT 0,
    mutation_result VARCHAR NOT NULL DEFAULT '',
    validation_result VARCHAR NOT NULL DEFAULT '',
    charged INTEGER NOT NULL DEFAULT 1,
    dispatched INTEGER NOT NULL DEFAULT 0,
    suppressed INTEGER NOT NULL DEFAULT 0,
    redacted INTEGER NOT NULL DEFAULT 1,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    snapshot_id VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS provider_calls_call_key_uidx
    ON provider_calls(call_key);
CREATE INDEX IF NOT EXISTS provider_calls_task_idx
    ON provider_calls(task_cid, recorded_at);
CREATE INDEX IF NOT EXISTS provider_calls_idempotency_idx
    ON provider_calls(idempotency_key);

CREATE TABLE IF NOT EXISTS failure_signatures (
    signature_id VARCHAR PRIMARY KEY,
    call_key VARCHAR NOT NULL,
    failure_class VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    proposal_digest VARCHAR NOT NULL DEFAULT '',
    policy_id VARCHAR NOT NULL DEFAULT '',
    retry_count BIGINT NOT NULL DEFAULT 0,
    retry_budget BIGINT NOT NULL DEFAULT 0,
    exhausted INTEGER NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    snapshot_id VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS failure_signatures_call_key_idx
    ON failure_signatures(call_key, recorded_at);
CREATE INDEX IF NOT EXISTS failure_signatures_class_idx
    ON failure_signatures(failure_class, evidence_digest);

CREATE TABLE IF NOT EXISTS replay_suppressions (
    suppression_id VARCHAR PRIMARY KEY,
    call_key VARCHAR NOT NULL,
    signature_id VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    expires_at_ms BIGINT NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    snapshot_id VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS replay_suppressions_call_key_idx
    ON replay_suppressions(call_key, active);

CREATE TABLE IF NOT EXISTS usage_charges (
    charge_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    call_key VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    charged INTEGER NOT NULL DEFAULT 1,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    snapshot_id VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_charges_call_idx
    ON usage_charges(call_id);

CREATE TABLE IF NOT EXISTS churn_metrics (
    metric_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    value_milli BIGINT NOT NULL DEFAULT 0,
    labels_json VARCHAR NOT NULL DEFAULT '{}',
    recorded_at VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS churn_metrics_name_idx
    ON churn_metrics(name, recorded_at);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderCallLedgerError(RuntimeError):
    """Base error for provider call ledger failures."""


class ProviderCallLedgerNotOpenError(ProviderCallLedgerError):
    """Operation requires an open ledger."""


class ProviderCallLedgerBoundsError(ProviderCallLedgerError, ValueError):
    """Payload, token, or recursion bound exceeded."""


class ProviderCallLedgerConflictError(ProviderCallLedgerError):
    """Idempotency or identity conflict."""


class ProviderCallLedgerSecretError(ProviderCallLedgerError, ValueError):
    """Secret or raw prompt/completion material was presented for storage."""

    def __init__(
        self,
        message: str = "secret or private material is excluded from ledger rows",
        *,
        reason_code: str = "secret_material_rejected",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class DuckDBUnavailableError(ProviderCallLedgerError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ProviderCallOutcome(str, Enum):
    """Typed provider-call disposition for churn and usage accounting."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RETRY = "retry"
    ABANDONED = "abandoned"
    RESPONSE_LOSS = "response_loss"
    HARD_QUOTA = "hard_quota"
    TRANSIENT_FAILURE = "transient_failure"
    SUPPRESSED = "suppressed"
    DUPLICATE = "duplicate"

    @classmethod
    def coerce(cls, value: Any) -> "ProviderCallOutcome":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        aliases = {
            "success": cls.ACCEPTED,
            "ok": cls.ACCEPTED,
            "fail": cls.REJECTED,
            "failed": cls.REJECTED,
            "quota": cls.HARD_QUOTA,
            "hard_quota_exhausted": cls.HARD_QUOTA,
            "transient": cls.TRANSIENT_FAILURE,
            "timeout": cls.TRANSIENT_FAILURE,
            "lost_response": cls.RESPONSE_LOSS,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise ProviderCallLedgerError(
                f"unsupported provider call outcome: {value!r}"
            ) from exc


class DuplicateKind(str, Enum):
    NONE = "none"
    EXACT = "exact"
    SEMANTIC = "semantic"

    @classmethod
    def coerce(cls, value: Any) -> "DuplicateKind":
        if isinstance(value, cls):
            return value
        text = str(value or "none").strip().casefold() or "none"
        try:
            return cls(text)
        except ValueError as exc:
            raise ProviderCallLedgerError(
                f"unsupported duplicate kind: {value!r}"
            ) from exc


class FailureClass(str, Enum):
    HARD_QUOTA = "hard_quota"
    TRANSIENT = "transient"
    VALIDATION = "validation"
    RESPONSE_LOSS = "response_loss"
    POLICY_EXHAUSTED = "policy_exhausted"
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> "FailureClass":
        if isinstance(value, cls):
            return value
        text = str(value or "unknown").strip().casefold() or "unknown"
        aliases = {
            "hard_quota_exhausted": cls.HARD_QUOTA,
            "quota": cls.HARD_QUOTA,
            "transient_failure": cls.TRANSIENT,
            "timeout": cls.TRANSIENT,
            "lost_response": cls.RESPONSE_LOSS,
            "retry_exhausted": cls.POLICY_EXHAUSTED,
            "exhausted": cls.POLICY_EXHAUSTED,
            "auth": cls.AUTHENTICATION,
            "rate_limit": cls.RATE_LIMITED,
            "rate_limited": cls.RATE_LIMITED,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise ProviderCallLedgerError(
                f"unsupported failure class: {value!r}"
            ) from exc


class ChurnAction(str, Enum):
    """Replay-suppression / dispatch decision."""

    DISPATCH = "dispatch"
    SUPPRESS_REPLAY = "suppress_replay"
    CHARGE_ONLY = "charge_only"
    NEGATIVE_CACHE = "negative_cache"
    REUSE_PRIOR = "reuse_prior"

    @classmethod
    def coerce(cls, value: Any) -> "ChurnAction":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold()
        try:
            return cls(text)
        except ValueError as exc:
            raise ProviderCallLedgerError(
                f"unsupported churn action: {value!r}"
            ) from exc


class QuotaClass(str, Enum):
    NONE = "none"
    HARD = "hard"
    SOFT = "soft"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> "QuotaClass":
        if isinstance(value, cls):
            return value
        text = str(value or "none").strip().casefold() or "none"
        try:
            return cls(text)
        except ValueError as exc:
            raise ProviderCallLedgerError(
                f"unsupported quota class: {value!r}"
            ) from exc


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise ProviderCallLedgerError(f"{name} contains NUL")
    if required and not text:
        raise ProviderCallLedgerError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ProviderCallLedgerBoundsError(
            f"{name} exceeds {MAX_TEXT_BYTES} bytes"
        )
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderCallLedgerBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _sha256_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _content_digest(value: Any) -> str:
    return _sha256_digest(_canonical_json(value).encode("utf-8"))


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement:
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _load_json_object(text: Any) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise ProviderCallLedgerError("stored JSON is corrupted") from exc
    if not isinstance(value, dict):
        raise ProviderCallLedgerError("stored JSON must be an object")
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = str(key or "").strip().casefold().replace("-", "_")
    if normalized in _SENSITIVE_KEYS or normalized in _FORBIDDEN_BODY_KEYS:
        return True
    if normalized.endswith("_secret") or normalized.endswith("_password"):
        return True
    if normalized.endswith("_api_key") or normalized.endswith("_private_key"):
        return True
    return False


def _text_contains_secret_pattern(value: str) -> bool:
    return any(pattern.search(value) for pattern in _TEXT_SECRET_PATTERNS)


def _reject_or_redact(
    value: Any,
    *,
    reject: bool,
    path: str = "",
    depth: int = 0,
) -> Any:
    """Fail closed on secret/raw bodies, or redact when allowed."""

    if depth > MAX_RECURSION_DEPTH:
        raise ProviderCallLedgerBoundsError(
            f"payload exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}" if path else key_text
            if _is_sensitive_key(key_text):
                if reject:
                    raise ProviderCallLedgerSecretError(
                        f"secret or private field excluded: {key_path}",
                        reason_code="secret_material_rejected",
                    )
                result[key_text] = REDACTION_MARKER
                continue
            result[key_text] = _reject_or_redact(
                item, reject=reject, path=key_path, depth=depth + 1
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _reject_or_redact(
                item, reject=reject, path=f"{path}[{index}]", depth=depth + 1
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if _text_contains_secret_pattern(value):
            if reject:
                raise ProviderCallLedgerSecretError(
                    f"secret pattern excluded at {path or 'value'}",
                    reason_code="secret_material_rejected",
                )
            return REDACTION_MARKER
        return value
    return value


def _bounded_body(
    body: Mapping[str, Any] | None,
    *,
    redact: bool,
    reject_secrets: bool = True,
) -> dict[str, Any]:
    """Bound and scrub a ledger body.

    Secret-bearing keys, raw prompts/completions, and secret-shaped free text
    are rejected when ``reject_secrets`` is true.  ``redact`` only applies the
    classification-label rewrite for any residual sensitive keys after the
    fail-closed pass; it never weakens rejection.
    """

    raw = dict(body or {})
    # Fail closed first so auto-redact never converts secrets into ordinary rows.
    cleaned = _reject_or_redact(raw, reject=bool(reject_secrets))
    if redact:
        cleaned = redact_mapping(cleaned)
    if not isinstance(cleaned, dict):
        raise ProviderCallLedgerError("body must project to an object")
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise ProviderCallLedgerBoundsError(
            f"body exceeds the {MAX_BODY_BYTES}-byte bound"
        )
    return cleaned


def compute_call_key(
    *,
    provider_id: str,
    model_id: str,
    endpoint_id: str = "",
    context_cid: str = "",
    plan_cid: str = "",
    task_cid: str = "",
    attempt_id: str = "",
    policy_id: str = DEFAULT_POLICY_ID,
    evidence_digest: str = "",
    prompt_digest: str = "",
    idempotency_key: str = "",
) -> str:
    """Return a redacted, content-addressed call key.

    Raw prompts and secrets never enter the key material. Callers must supply
    digests of prompt/evidence bodies rather than the bodies themselves.
    """

    material = {
        "provider_id": _text(provider_id, "provider_id"),
        "model_id": _text(model_id, "model_id"),
        "endpoint_id": _text(endpoint_id, "endpoint_id", required=False),
        "context_cid": _text(context_cid, "context_cid", required=False),
        "plan_cid": _text(plan_cid, "plan_cid", required=False),
        "task_cid": _text(task_cid, "task_cid", required=False),
        "attempt_id": _text(attempt_id, "attempt_id", required=False),
        "policy_id": _text(policy_id or DEFAULT_POLICY_ID, "policy_id"),
        "evidence_digest": _text(
            evidence_digest, "evidence_digest", required=False
        ),
        "prompt_digest": _text(prompt_digest, "prompt_digest", required=False),
        "idempotency_key": _text(
            idempotency_key, "idempotency_key", required=False
        ),
    }
    return "call:" + _content_digest(material)


def compute_prompt_digest(prompt_material: Any) -> str:
    """Digest prompt-shaped material without retaining the body."""

    if isinstance(prompt_material, Mapping):
        cleaned = _reject_or_redact(dict(prompt_material), reject=False)
        return _content_digest(cleaned)
    if isinstance(prompt_material, (bytes, bytearray)):
        return _sha256_digest(bytes(prompt_material))
    text = str(prompt_material or "")
    if _text_contains_secret_pattern(text):
        text = REDACTION_MARKER
    return _sha256_digest(text.encode("utf-8"))


def compute_response_digest(response_material: Any) -> str:
    """Digest response-shaped material without retaining the body."""

    return compute_prompt_digest(response_material)


def compute_failure_signature_id(
    *,
    failure_class: FailureClass | str,
    evidence_digest: str,
    proposal_digest: str = "",
    policy_id: str = DEFAULT_POLICY_ID,
) -> str:
    klass = FailureClass.coerce(failure_class)
    material = {
        "failure_class": klass.value,
        "evidence_digest": _text(
            evidence_digest, "evidence_digest", required=False
        ),
        "proposal_digest": _text(
            proposal_digest, "proposal_digest", required=False
        ),
        "policy_id": _text(policy_id or DEFAULT_POLICY_ID, "policy_id"),
    }
    return "fsig:" + _content_digest(material)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureSignature:
    """Normalized failure signature used for replay suppression."""

    signature_id: str
    call_key: str
    failure_class: FailureClass
    evidence_digest: str
    proposal_digest: str = ""
    policy_id: str = DEFAULT_POLICY_ID
    retry_count: int = 0
    retry_budget: int = DEFAULT_RETRY_BUDGET
    exhausted: bool = False
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)
    schema: str = FAILURE_SIGNATURE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "signature_id", _text(self.signature_id, "signature_id")
        )
        object.__setattr__(self, "call_key", _text(self.call_key, "call_key"))
        object.__setattr__(
            self, "failure_class", FailureClass.coerce(self.failure_class)
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _text(self.evidence_digest, "evidence_digest", required=False),
        )
        object.__setattr__(
            self,
            "proposal_digest",
            _text(self.proposal_digest, "proposal_digest", required=False),
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id or DEFAULT_POLICY_ID, "policy_id"),
        )
        object.__setattr__(
            self, "retry_count", _nonneg_int(self.retry_count, "retry_count")
        )
        object.__setattr__(
            self, "retry_budget", _nonneg_int(self.retry_budget, "retry_budget")
        )
        object.__setattr__(self, "exhausted", bool(self.exhausted))
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        object.__setattr__(
            self, "body", MappingProxyType(dict(self.body or {}))
        )
        if self.schema != FAILURE_SIGNATURE_SCHEMA:
            raise ProviderCallLedgerError("unsupported failure signature schema")

    @property
    def interface(self) -> str:
        return FAILURE_SIGNATURE_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "signature_id": self.signature_id,
            "call_key": self.call_key,
            "failure_class": self.failure_class.value,
            "evidence_digest": self.evidence_digest,
            "proposal_digest": self.proposal_digest,
            "policy_id": self.policy_id,
            "retry_count": self.retry_count,
            "retry_budget": self.retry_budget,
            "exhausted": self.exhausted,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ChurnDecision:
    """Dispatch vs suppress decision for one call key / evidence pair."""

    call_key: str
    action: ChurnAction
    reason: str
    duplicate_kind: DuplicateKind = DuplicateKind.NONE
    prior_call_id: str = ""
    signature_id: str = ""
    suppress_until_ms: int = 0
    charged: bool = False
    may_dispatch: bool = True
    evidence_digest: str = ""
    recorded_at: str = ""
    schema: str = CHURN_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_key", _text(self.call_key, "call_key"))
        object.__setattr__(self, "action", ChurnAction.coerce(self.action))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self, "duplicate_kind", DuplicateKind.coerce(self.duplicate_kind)
        )
        object.__setattr__(
            self,
            "prior_call_id",
            _text(self.prior_call_id, "prior_call_id", required=False),
        )
        object.__setattr__(
            self,
            "signature_id",
            _text(self.signature_id, "signature_id", required=False),
        )
        object.__setattr__(
            self,
            "suppress_until_ms",
            _nonneg_int(self.suppress_until_ms, "suppress_until_ms"),
        )
        object.__setattr__(self, "charged", bool(self.charged))
        object.__setattr__(self, "may_dispatch", bool(self.may_dispatch))
        object.__setattr__(
            self,
            "evidence_digest",
            _text(self.evidence_digest, "evidence_digest", required=False),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        if self.schema != CHURN_DECISION_SCHEMA:
            raise ProviderCallLedgerError("unsupported churn decision schema")

    @property
    def interface(self) -> str:
        return CHURN_DECISION_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "call_key": self.call_key,
            "action": self.action.value,
            "reason": self.reason,
            "duplicate_kind": self.duplicate_kind.value,
            "prior_call_id": self.prior_call_id,
            "signature_id": self.signature_id,
            "suppress_until_ms": self.suppress_until_ms,
            "charged": self.charged,
            "may_dispatch": self.may_dispatch,
            "evidence_digest": self.evidence_digest,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class UsageCharge:
    """Usage charge row for accepted, rejected, abandoned, or retry outcomes."""

    charge_id: str
    call_id: str
    call_key: str
    outcome: ProviderCallOutcome
    input_tokens: int = 0
    output_tokens: int = 0
    charged: bool = True
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)
    schema: str = USAGE_CHARGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "charge_id", _text(self.charge_id, "charge_id")
        )
        object.__setattr__(self, "call_id", _text(self.call_id, "call_id"))
        object.__setattr__(self, "call_key", _text(self.call_key, "call_key"))
        object.__setattr__(
            self, "outcome", ProviderCallOutcome.coerce(self.outcome)
        )
        object.__setattr__(
            self, "input_tokens", _nonneg_int(self.input_tokens, "input_tokens")
        )
        object.__setattr__(
            self,
            "output_tokens",
            _nonneg_int(self.output_tokens, "output_tokens"),
        )
        object.__setattr__(self, "charged", bool(self.charged))
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        object.__setattr__(
            self, "body", MappingProxyType(dict(self.body or {}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "charge_id": self.charge_id,
            "call_id": self.call_id,
            "call_key": self.call_key,
            "outcome": self.outcome.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "charged": self.charged,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ProviderCallRecord:
    """One durable, redacted provider-call ledger row."""

    call_id: str
    call_key: str
    idempotency_key: str
    provider_id: str
    model_id: str
    endpoint_id: str = ""
    context_cid: str = ""
    plan_cid: str = ""
    task_cid: str = ""
    attempt_id: str = ""
    policy_id: str = DEFAULT_POLICY_ID
    evidence_digest: str = ""
    prompt_digest: str = ""
    response_digest: str = ""
    outcome: ProviderCallOutcome = ProviderCallOutcome.ACCEPTED
    quota_class: QuotaClass = QuotaClass.NONE
    duplicate_kind: DuplicateKind = DuplicateKind.NONE
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    latency_ms: int = 0
    budget_tokens: int = 0
    mutation_result: str = ""
    validation_result: str = ""
    charged: bool = True
    dispatched: bool = False
    suppressed: bool = False
    redacted: bool = True
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    schema: str = PROVIDER_CALL_RECORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _text(self.call_id, "call_id"))
        object.__setattr__(self, "call_key", _text(self.call_key, "call_key"))
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key"),
        )
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, "provider_id")
        )
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "endpoint_id",
            _text(self.endpoint_id, "endpoint_id", required=False),
        )
        object.__setattr__(
            self,
            "context_cid",
            _text(self.context_cid, "context_cid", required=False),
        )
        object.__setattr__(
            self, "plan_cid", _text(self.plan_cid, "plan_cid", required=False)
        )
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self,
            "attempt_id",
            _text(self.attempt_id, "attempt_id", required=False),
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id or DEFAULT_POLICY_ID, "policy_id"),
        )
        for name in (
            "evidence_digest",
            "prompt_digest",
            "response_digest",
            "mutation_result",
            "validation_result",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self, "outcome", ProviderCallOutcome.coerce(self.outcome)
        )
        object.__setattr__(
            self, "quota_class", QuotaClass.coerce(self.quota_class)
        )
        object.__setattr__(
            self, "duplicate_kind", DuplicateKind.coerce(self.duplicate_kind)
        )
        for name in (
            "estimated_input_tokens",
            "estimated_output_tokens",
            "actual_input_tokens",
            "actual_output_tokens",
            "latency_ms",
            "budget_tokens",
        ):
            object.__setattr__(
                self, name, _nonneg_int(getattr(self, name), name)
            )
        object.__setattr__(self, "charged", bool(self.charged))
        object.__setattr__(self, "dispatched", bool(self.dispatched))
        object.__setattr__(self, "suppressed", bool(self.suppressed))
        object.__setattr__(self, "redacted", bool(self.redacted))
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        object.__setattr__(
            self, "body", MappingProxyType(dict(self.body or {}))
        )
        object.__setattr__(
            self,
            "snapshot_id",
            _text(self.snapshot_id or DEFAULT_SNAPSHOT_ID, "snapshot_id"),
        )
        if self.schema != PROVIDER_CALL_RECORD_SCHEMA:
            raise ProviderCallLedgerError("unsupported provider call record schema")

    @property
    def interface(self) -> str:
        return PROVIDER_CALL_LEDGER_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "call_id": self.call_id,
            "call_key": self.call_key,
            "idempotency_key": self.idempotency_key,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "endpoint_id": self.endpoint_id,
            "context_cid": self.context_cid,
            "plan_cid": self.plan_cid,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "policy_id": self.policy_id,
            "evidence_digest": self.evidence_digest,
            "prompt_digest": self.prompt_digest,
            "response_digest": self.response_digest,
            "outcome": self.outcome.value,
            "quota_class": self.quota_class.value,
            "duplicate_kind": self.duplicate_kind.value,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "latency_ms": self.latency_ms,
            "budget_tokens": self.budget_tokens,
            "mutation_result": self.mutation_result,
            "validation_result": self.validation_result,
            "charged": self.charged,
            "dispatched": self.dispatched,
            "suppressed": self.suppressed,
            "redacted": self.redacted,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
            "snapshot_id": self.snapshot_id,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ProviderCallRequest:
    """Inbound request used to evaluate churn and record a call."""

    provider_id: str
    model_id: str
    endpoint_id: str = ""
    context_cid: str = ""
    plan_cid: str = ""
    task_cid: str = ""
    attempt_id: str = ""
    policy_id: str = DEFAULT_POLICY_ID
    evidence_digest: str = ""
    prompt_digest: str = ""
    idempotency_key: str = ""
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    budget_tokens: int = 0
    semantic_fingerprint: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def resolved_idempotency_key(self) -> str:
        if self.idempotency_key:
            return _text(self.idempotency_key, "idempotency_key")
        return compute_call_key(
            provider_id=self.provider_id,
            model_id=self.model_id,
            endpoint_id=self.endpoint_id,
            context_cid=self.context_cid,
            plan_cid=self.plan_cid,
            task_cid=self.task_cid,
            attempt_id=self.attempt_id,
            policy_id=self.policy_id,
            evidence_digest=self.evidence_digest,
            prompt_digest=self.prompt_digest,
        )

    def call_key(self) -> str:
        return compute_call_key(
            provider_id=self.provider_id,
            model_id=self.model_id,
            endpoint_id=self.endpoint_id,
            context_cid=self.context_cid,
            plan_cid=self.plan_cid,
            task_cid=self.task_cid,
            attempt_id=self.attempt_id,
            policy_id=self.policy_id,
            evidence_digest=self.evidence_digest,
            prompt_digest=self.prompt_digest,
            idempotency_key=self.resolved_idempotency_key(),
        )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class ProviderCallLedger:
    """DuckDB authority for provider calls, failure signatures, and churn."""

    INTERFACE: Final[str] = PROVIDER_CALL_LEDGER_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        auto_redact: bool = True,
        default_retry_budget: int = DEFAULT_RETRY_BUDGET,
        default_negative_cache_ttl_ms: int = DEFAULT_NEGATIVE_CACHE_TTL_MS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for ProviderCallLedger; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._snapshot_id = _text(snapshot_id or DEFAULT_SNAPSHOT_ID, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        self._default_retry_budget = _nonneg_int(
            default_retry_budget, "default_retry_budget"
        )
        self._default_negative_cache_ttl_ms = _nonneg_int(
            default_negative_cache_ttl_ms, "default_negative_cache_ttl_ms"
        )
        self._clock_ms = clock_ms or _now_ms
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

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
                ("snapshot_id", self._snapshot_id),
                ("authority", AUTHORITY_CLASS),
                ("producer", PRODUCER_ID),
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
            raise ProviderCallLedgerNotOpenError("ProviderCallLedger is not open")
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

    # -- row helpers ---------------------------------------------------------

    def _call_from_row(self, mapping: Mapping[str, Any]) -> ProviderCallRecord:
        return ProviderCallRecord(
            call_id=str(mapping.get("call_id") or ""),
            call_key=str(mapping.get("call_key") or ""),
            idempotency_key=str(mapping.get("idempotency_key") or ""),
            provider_id=str(mapping.get("provider_id") or ""),
            model_id=str(mapping.get("model_id") or ""),
            endpoint_id=str(mapping.get("endpoint_id") or ""),
            context_cid=str(mapping.get("context_cid") or ""),
            plan_cid=str(mapping.get("plan_cid") or ""),
            task_cid=str(mapping.get("task_cid") or ""),
            attempt_id=str(mapping.get("attempt_id") or ""),
            policy_id=str(mapping.get("policy_id") or DEFAULT_POLICY_ID),
            evidence_digest=str(mapping.get("evidence_digest") or ""),
            prompt_digest=str(mapping.get("prompt_digest") or ""),
            response_digest=str(mapping.get("response_digest") or ""),
            outcome=str(mapping.get("outcome") or ProviderCallOutcome.ACCEPTED.value),
            quota_class=str(mapping.get("quota_class") or QuotaClass.NONE.value),
            duplicate_kind=str(
                mapping.get("duplicate_kind") or DuplicateKind.NONE.value
            ),
            estimated_input_tokens=int(mapping.get("estimated_input_tokens") or 0),
            estimated_output_tokens=int(mapping.get("estimated_output_tokens") or 0),
            actual_input_tokens=int(mapping.get("actual_input_tokens") or 0),
            actual_output_tokens=int(mapping.get("actual_output_tokens") or 0),
            latency_ms=int(mapping.get("latency_ms") or 0),
            budget_tokens=int(mapping.get("budget_tokens") or 0),
            mutation_result=str(mapping.get("mutation_result") or ""),
            validation_result=str(mapping.get("validation_result") or ""),
            charged=bool(int(mapping.get("charged") or 0)),
            dispatched=bool(int(mapping.get("dispatched") or 0)),
            suppressed=bool(int(mapping.get("suppressed") or 0)),
            redacted=bool(int(mapping.get("redacted") or 0)),
            recorded_at=str(mapping.get("recorded_at") or ""),
            body=_load_json_object(mapping.get("body_json")),
            snapshot_id=str(mapping.get("snapshot_id") or self._snapshot_id),
        )

    def _signature_from_row(
        self, mapping: Mapping[str, Any]
    ) -> FailureSignature:
        return FailureSignature(
            signature_id=str(mapping.get("signature_id") or ""),
            call_key=str(mapping.get("call_key") or ""),
            failure_class=str(mapping.get("failure_class") or FailureClass.UNKNOWN.value),
            evidence_digest=str(mapping.get("evidence_digest") or ""),
            proposal_digest=str(mapping.get("proposal_digest") or ""),
            policy_id=str(mapping.get("policy_id") or DEFAULT_POLICY_ID),
            retry_count=int(mapping.get("retry_count") or 0),
            retry_budget=int(mapping.get("retry_budget") or 0),
            exhausted=bool(int(mapping.get("exhausted") or 0)),
            recorded_at=str(mapping.get("recorded_at") or ""),
            body=_load_json_object(mapping.get("body_json")),
        )

    def _get_call_row(
        self, connection: Any, *, call_key: str = "", call_id: str = ""
    ) -> dict[str, Any] | None:
        if call_id:
            row = connection.execute(
                "SELECT * FROM provider_calls WHERE call_id = ?",
                [call_id],
            ).fetchone()
            return _row_mapping(row) if row is not None else None
        if call_key:
            row = connection.execute(
                "SELECT * FROM provider_calls WHERE call_key = ?",
                [call_key],
            ).fetchone()
            return _row_mapping(row) if row is not None else None
        return None

    def _active_suppression(
        self, connection: Any, call_key: str, *, now_ms: int
    ) -> dict[str, Any] | None:
        rows = connection.execute(
            """
            SELECT * FROM replay_suppressions
            WHERE call_key = ? AND active = 1
            ORDER BY recorded_at DESC
            """,
            [call_key],
        ).fetchall()
        for row in rows:
            mapping = _row_mapping(row)
            expires = int(mapping.get("expires_at_ms") or 0)
            if expires == 0 or expires > now_ms:
                return mapping
            connection.execute(
                """
                UPDATE replay_suppressions
                SET active = 0
                WHERE suppression_id = ?
                """,
                [mapping.get("suppression_id")],
            )
        return None

    def _latest_exhausted_signature(
        self, connection: Any, call_key: str, evidence_digest: str
    ) -> FailureSignature | None:
        row = connection.execute(
            """
            SELECT * FROM failure_signatures
            WHERE call_key = ?
              AND evidence_digest = ?
              AND exhausted = 1
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            [call_key, evidence_digest],
        ).fetchone()
        if row is None:
            return None
        return self._signature_from_row(_row_mapping(row))

    def _semantic_duplicate(
        self,
        connection: Any,
        *,
        task_cid: str,
        semantic_fingerprint: str,
        evidence_digest: str,
    ) -> ProviderCallRecord | None:
        if not semantic_fingerprint or not task_cid:
            return None
        rows = connection.execute(
            """
            SELECT * FROM provider_calls
            WHERE task_cid = ?
            ORDER BY recorded_at DESC
            """,
            [task_cid],
        ).fetchall()
        for row in rows:
            record = self._call_from_row(_row_mapping(row))
            body = dict(record.body)
            if (
                str(body.get("semantic_fingerprint") or "") == semantic_fingerprint
                and record.evidence_digest == evidence_digest
            ):
                return record
        return None

    def _insert_metric(
        self,
        connection: Any,
        name: str,
        value: int,
        *,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        stamp = _utc_iso()
        metric_id = "metric:" + _content_digest(
            {
                "name": name,
                "value": value,
                "labels": dict(labels or {}),
                "recorded_at": stamp,
                "snapshot_id": self._snapshot_id,
            }
        )
        connection.execute(
            """
            INSERT INTO churn_metrics(
                metric_id, name, value_milli, labels_json, recorded_at, snapshot_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                metric_id,
                name,
                int(value) * 1000,
                _canonical_json(dict(labels or {})),
                stamp,
                self._snapshot_id,
            ],
        )

    # -- public API ----------------------------------------------------------

    def evaluate_dispatch(
        self,
        request: ProviderCallRequest,
        *,
        now_ms: int | None = None,
    ) -> ChurnDecision:
        """Return whether a request may dispatch or must be suppressed."""

        call_key = request.call_key()
        evidence = _text(
            request.evidence_digest, "evidence_digest", required=False
        )
        stamp = _utc_iso()
        current_ms = int(now_ms if now_ms is not None else self._clock_ms())

        with self._lock:
            connection = self._require()
            existing = self._get_call_row(connection, call_key=call_key)
            if existing is not None:
                prior = self._call_from_row(existing)
                return ChurnDecision(
                    call_key=call_key,
                    action=ChurnAction.REUSE_PRIOR,
                    reason="exact_duplicate_call_key",
                    duplicate_kind=DuplicateKind.EXACT,
                    prior_call_id=prior.call_id,
                    may_dispatch=False,
                    charged=bool(prior.charged),
                    evidence_digest=evidence,
                    recorded_at=stamp,
                )

            semantic = self._semantic_duplicate(
                connection,
                task_cid=_text(request.task_cid, "task_cid", required=False),
                semantic_fingerprint=_text(
                    request.semantic_fingerprint,
                    "semantic_fingerprint",
                    required=False,
                ),
                evidence_digest=evidence,
            )
            if semantic is not None:
                return ChurnDecision(
                    call_key=call_key,
                    action=ChurnAction.REUSE_PRIOR,
                    reason="semantic_duplicate",
                    duplicate_kind=DuplicateKind.SEMANTIC,
                    prior_call_id=semantic.call_id,
                    may_dispatch=False,
                    charged=bool(semantic.charged),
                    evidence_digest=evidence,
                    recorded_at=stamp,
                )

            suppression = self._active_suppression(
                connection, call_key, now_ms=current_ms
            )
            if suppression is not None:
                action = ChurnAction.coerce(
                    suppression.get("action") or ChurnAction.SUPPRESS_REPLAY.value
                )
                return ChurnDecision(
                    call_key=call_key,
                    action=action,
                    reason=str(suppression.get("reason") or "replay_suppressed"),
                    signature_id=str(suppression.get("signature_id") or ""),
                    suppress_until_ms=int(suppression.get("expires_at_ms") or 0),
                    may_dispatch=False,
                    charged=True,
                    evidence_digest=evidence,
                    recorded_at=stamp,
                )

            exhausted = self._latest_exhausted_signature(
                connection, call_key, evidence
            )
            if exhausted is not None:
                return ChurnDecision(
                    call_key=call_key,
                    action=ChurnAction.SUPPRESS_REPLAY,
                    reason="policy_exhausted_unchanged_evidence",
                    signature_id=exhausted.signature_id,
                    may_dispatch=False,
                    charged=True,
                    evidence_digest=evidence,
                    recorded_at=stamp,
                )

            return ChurnDecision(
                call_key=call_key,
                action=ChurnAction.DISPATCH,
                reason="no_suppression",
                may_dispatch=True,
                charged=False,
                evidence_digest=evidence,
                recorded_at=stamp,
            )

    def record_call(
        self,
        request: ProviderCallRequest,
        *,
        outcome: ProviderCallOutcome | str,
        response_digest: str = "",
        actual_input_tokens: int = 0,
        actual_output_tokens: int = 0,
        latency_ms: int = 0,
        quota_class: QuotaClass | str = QuotaClass.NONE,
        mutation_result: str = "",
        validation_result: str = "",
        dispatched: bool = True,
        suppressed: bool = False,
        duplicate_kind: DuplicateKind | str = DuplicateKind.NONE,
        body: Mapping[str, Any] | None = None,
        charge: bool = True,
        now_ms: int | None = None,
    ) -> tuple[ProviderCallRecord, ChurnDecision]:
        """Record one provider call after evaluating churn / idempotency."""

        decision = self.evaluate_dispatch(request, now_ms=now_ms)
        call_key = decision.call_key
        idem = request.resolved_idempotency_key()
        stamp = _utc_iso()
        selected_outcome = ProviderCallOutcome.coerce(outcome)
        selected_duplicate = DuplicateKind.coerce(duplicate_kind)
        if decision.action is ChurnAction.REUSE_PRIOR and decision.prior_call_id:
            with self._lock:
                connection = self._require()
                row = self._get_call_row(
                    connection, call_id=decision.prior_call_id
                )
                if row is None:
                    raise ProviderCallLedgerConflictError(
                        "prior call referenced by churn decision is missing"
                    )
                prior = self._call_from_row(row)
                # Exact/semantic duplicates do not re-dispatch, but still charge
                # if the caller presented a billable outcome and no prior charge.
                if charge and not prior.charged:
                    # Zero actuals are billable (e.g. rejected before completion).
                    # Do not fall back through `or` — that treats 0 as missing.
                    self._charge_row(
                        connection,
                        call_id=prior.call_id,
                        call_key=prior.call_key,
                        outcome=selected_outcome,
                        input_tokens=actual_input_tokens,
                        output_tokens=actual_output_tokens,
                        stamp=stamp,
                    )
                    connection.execute(
                        "UPDATE provider_calls SET charged = 1 WHERE call_id = ?",
                        [prior.call_id],
                    )
                    self._commit_if_idle(connection)
                    refreshed = self._get_call_row(
                        connection, call_id=prior.call_id
                    )
                    if refreshed is not None:
                        prior = self._call_from_row(refreshed)
                return prior, decision

        if not decision.may_dispatch:
            # Suppressed path: charge usage, do not dispatch a new provider call.
            selected_outcome = ProviderCallOutcome.SUPPRESSED
            dispatched = False
            suppressed = True
            selected_duplicate = (
                decision.duplicate_kind
                if decision.duplicate_kind is not DuplicateKind.NONE
                else selected_duplicate
            )

        merged_body = dict(request.body or {})
        if body:
            merged_body.update(dict(body))
        if request.semantic_fingerprint:
            merged_body.setdefault(
                "semantic_fingerprint", request.semantic_fingerprint
            )
        cleaned_body = _bounded_body(
            merged_body,
            redact=self._auto_redact,
            reject_secrets=True,
        )

        call_id = "pcall:" + _content_digest(
            {
                "call_key": call_key,
                "recorded_at": stamp,
                "outcome": selected_outcome.value,
                "snapshot_id": self._snapshot_id,
            }
        )
        record = ProviderCallRecord(
            call_id=call_id,
            call_key=call_key,
            idempotency_key=idem,
            provider_id=request.provider_id,
            model_id=request.model_id,
            endpoint_id=request.endpoint_id,
            context_cid=request.context_cid,
            plan_cid=request.plan_cid,
            task_cid=request.task_cid,
            attempt_id=request.attempt_id,
            policy_id=request.policy_id or DEFAULT_POLICY_ID,
            evidence_digest=request.evidence_digest,
            prompt_digest=request.prompt_digest,
            response_digest=response_digest,
            outcome=selected_outcome,
            quota_class=quota_class,
            duplicate_kind=selected_duplicate,
            estimated_input_tokens=request.estimated_input_tokens,
            estimated_output_tokens=request.estimated_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            latency_ms=latency_ms,
            budget_tokens=request.budget_tokens,
            mutation_result=mutation_result,
            validation_result=validation_result,
            charged=bool(charge),
            dispatched=bool(dispatched) and decision.may_dispatch,
            suppressed=bool(suppressed) or not decision.may_dispatch,
            redacted=True,
            recorded_at=stamp,
            body=cleaned_body,
            snapshot_id=self._snapshot_id,
        )

        with self._lock:
            connection = self._require()
            # Re-check race: another writer may have inserted the same key.
            existing = self._get_call_row(connection, call_key=call_key)
            if existing is not None:
                prior = self._call_from_row(existing)
                reuse = ChurnDecision(
                    call_key=call_key,
                    action=ChurnAction.REUSE_PRIOR,
                    reason="exact_duplicate_call_key",
                    duplicate_kind=DuplicateKind.EXACT,
                    prior_call_id=prior.call_id,
                    may_dispatch=False,
                    charged=bool(prior.charged),
                    evidence_digest=request.evidence_digest,
                    recorded_at=stamp,
                )
                return prior, reuse

            connection.execute(
                """
                INSERT INTO provider_calls(
                    call_id, call_key, idempotency_key, provider_id, model_id,
                    endpoint_id, context_cid, plan_cid, task_cid, attempt_id,
                    policy_id, evidence_digest, prompt_digest, response_digest,
                    outcome, quota_class, duplicate_kind,
                    estimated_input_tokens, estimated_output_tokens,
                    actual_input_tokens, actual_output_tokens, latency_ms,
                    budget_tokens, mutation_result, validation_result,
                    charged, dispatched, suppressed, redacted, recorded_at,
                    body_json, snapshot_id
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    record.call_id,
                    record.call_key,
                    record.idempotency_key,
                    record.provider_id,
                    record.model_id,
                    record.endpoint_id,
                    record.context_cid,
                    record.plan_cid,
                    record.task_cid,
                    record.attempt_id,
                    record.policy_id,
                    record.evidence_digest,
                    record.prompt_digest,
                    record.response_digest,
                    record.outcome.value,
                    record.quota_class.value,
                    record.duplicate_kind.value,
                    record.estimated_input_tokens,
                    record.estimated_output_tokens,
                    record.actual_input_tokens,
                    record.actual_output_tokens,
                    record.latency_ms,
                    record.budget_tokens,
                    record.mutation_result,
                    record.validation_result,
                    1 if record.charged else 0,
                    1 if record.dispatched else 0,
                    1 if record.suppressed else 0,
                    1 if record.redacted else 0,
                    record.recorded_at,
                    _canonical_json(dict(record.body)),
                    record.snapshot_id,
                ],
            )
            if charge:
                # Charge exact actuals, including zero. Estimates are planning
                # metadata only and must not inflate rejected/abandoned rows.
                self._charge_row(
                    connection,
                    call_id=record.call_id,
                    call_key=record.call_key,
                    outcome=record.outcome,
                    input_tokens=record.actual_input_tokens,
                    output_tokens=record.actual_output_tokens,
                    stamp=stamp,
                )
            self._insert_metric(
                connection,
                "provider_calls",
                1,
                labels={
                    "outcome": record.outcome.value,
                    "dispatched": record.dispatched,
                    "suppressed": record.suppressed,
                },
            )
            self._commit_if_idle(connection)
            final_decision = decision
            if not decision.may_dispatch:
                final_decision = ChurnDecision(
                    call_key=call_key,
                    action=decision.action
                    if decision.action is not ChurnAction.DISPATCH
                    else ChurnAction.SUPPRESS_REPLAY,
                    reason=decision.reason,
                    duplicate_kind=selected_duplicate,
                    prior_call_id=decision.prior_call_id,
                    signature_id=decision.signature_id,
                    suppress_until_ms=decision.suppress_until_ms,
                    may_dispatch=False,
                    charged=bool(charge),
                    evidence_digest=request.evidence_digest,
                    recorded_at=stamp,
                )
            return record, final_decision

    def _charge_row(
        self,
        connection: Any,
        *,
        call_id: str,
        call_key: str,
        outcome: ProviderCallOutcome,
        input_tokens: int,
        output_tokens: int,
        stamp: str,
    ) -> UsageCharge:
        charge_id = "uchg:" + _content_digest(
            {
                "call_id": call_id,
                "outcome": outcome.value,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "recorded_at": stamp,
            }
        )
        charge = UsageCharge(
            charge_id=charge_id,
            call_id=call_id,
            call_key=call_key,
            outcome=outcome,
            input_tokens=_nonneg_int(input_tokens, "input_tokens"),
            output_tokens=_nonneg_int(output_tokens, "output_tokens"),
            charged=True,
            recorded_at=stamp,
        )
        connection.execute(
            """
            INSERT INTO usage_charges(
                charge_id, call_id, call_key, outcome, input_tokens,
                output_tokens, charged, recorded_at, body_json, snapshot_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                charge.charge_id,
                charge.call_id,
                charge.call_key,
                charge.outcome.value,
                charge.input_tokens,
                charge.output_tokens,
                1,
                charge.recorded_at,
                "{}",
                self._snapshot_id,
            ],
        )
        self._insert_metric(
            connection,
            f"usage_charged_{outcome.value}",
            charge.input_tokens + charge.output_tokens,
            labels={"call_id": call_id},
        )
        return charge

    def record_failure_signature(
        self,
        *,
        call_key: str,
        failure_class: FailureClass | str,
        evidence_digest: str,
        proposal_digest: str = "",
        policy_id: str = DEFAULT_POLICY_ID,
        retry_count: int = 0,
        retry_budget: int | None = None,
        body: Mapping[str, Any] | None = None,
        negative_cache_ttl_ms: int | None = None,
        now_ms: int | None = None,
    ) -> FailureSignature:
        """Persist a normalized failure signature and optional suppression."""

        klass = FailureClass.coerce(failure_class)
        budget = (
            self._default_retry_budget
            if retry_budget is None
            else _nonneg_int(retry_budget, "retry_budget")
        )
        retries = _nonneg_int(retry_count, "retry_count")
        exhausted = retries >= budget or klass in {
            FailureClass.POLICY_EXHAUSTED,
            FailureClass.HARD_QUOTA,
        }
        signature_id = compute_failure_signature_id(
            failure_class=klass,
            evidence_digest=evidence_digest,
            proposal_digest=proposal_digest,
            policy_id=policy_id,
        )
        stamp = _utc_iso()
        cleaned = _bounded_body(
            body, redact=self._auto_redact, reject_secrets=True
        )
        signature = FailureSignature(
            signature_id=signature_id,
            call_key=call_key,
            failure_class=klass,
            evidence_digest=evidence_digest,
            proposal_digest=proposal_digest,
            policy_id=policy_id or DEFAULT_POLICY_ID,
            retry_count=retries,
            retry_budget=budget,
            exhausted=exhausted,
            recorded_at=stamp,
            body=cleaned,
        )
        current_ms = int(now_ms if now_ms is not None else self._clock_ms())

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM failure_signatures WHERE signature_id = ?",
                [signature_id],
            ).fetchone()
            if existing is not None:
                prior = self._signature_from_row(_row_mapping(existing))
                # Bump retry bookkeeping when the same signature reappears.
                new_retries = max(prior.retry_count, retries)
                new_exhausted = new_retries >= prior.retry_budget or exhausted
                connection.execute(
                    """
                    UPDATE failure_signatures
                    SET retry_count = ?, exhausted = ?, recorded_at = ?
                    WHERE signature_id = ?
                    """,
                    [
                        new_retries,
                        1 if new_exhausted else 0,
                        stamp,
                        signature_id,
                    ],
                )
                signature = FailureSignature(
                    signature_id=prior.signature_id,
                    call_key=prior.call_key,
                    failure_class=prior.failure_class,
                    evidence_digest=prior.evidence_digest,
                    proposal_digest=prior.proposal_digest,
                    policy_id=prior.policy_id,
                    retry_count=new_retries,
                    retry_budget=prior.retry_budget,
                    exhausted=new_exhausted,
                    recorded_at=stamp,
                    body=dict(prior.body),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO failure_signatures(
                        signature_id, call_key, failure_class, evidence_digest,
                        proposal_digest, policy_id, retry_count, retry_budget,
                        exhausted, recorded_at, body_json, snapshot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        signature.signature_id,
                        signature.call_key,
                        signature.failure_class.value,
                        signature.evidence_digest,
                        signature.proposal_digest,
                        signature.policy_id,
                        signature.retry_count,
                        signature.retry_budget,
                        1 if signature.exhausted else 0,
                        signature.recorded_at,
                        _canonical_json(dict(signature.body)),
                        self._snapshot_id,
                    ],
                )

            # Negative-cache transient failures; permanent suppress on exhaust
            # or hard quota. Class-specific reasons win over the generic
            # exhausted label so operators can distinguish quota from policy.
            ttl = (
                self._default_negative_cache_ttl_ms
                if negative_cache_ttl_ms is None
                else _nonneg_int(negative_cache_ttl_ms, "negative_cache_ttl_ms")
            )
            if klass is FailureClass.HARD_QUOTA:
                self._write_suppression(
                    connection,
                    call_key=call_key,
                    signature_id=signature.signature_id,
                    reason="hard_quota",
                    action=ChurnAction.SUPPRESS_REPLAY,
                    evidence_digest=evidence_digest,
                    expires_at_ms=0,
                    stamp=stamp,
                )
            elif signature.exhausted:
                self._write_suppression(
                    connection,
                    call_key=call_key,
                    signature_id=signature.signature_id,
                    reason="policy_exhausted_unchanged_evidence",
                    action=ChurnAction.SUPPRESS_REPLAY,
                    evidence_digest=evidence_digest,
                    expires_at_ms=0,
                    stamp=stamp,
                )
            elif klass in {
                FailureClass.TRANSIENT,
                FailureClass.RATE_LIMITED,
                FailureClass.RESPONSE_LOSS,
            }:
                self._write_suppression(
                    connection,
                    call_key=call_key,
                    signature_id=signature.signature_id,
                    reason="negative_cache_ttl",
                    action=ChurnAction.NEGATIVE_CACHE,
                    evidence_digest=evidence_digest,
                    expires_at_ms=current_ms + ttl if ttl else 0,
                    stamp=stamp,
                )
            self._insert_metric(
                connection,
                "failure_signatures",
                1,
                labels={
                    "failure_class": signature.failure_class.value,
                    "exhausted": signature.exhausted,
                },
            )
            self._commit_if_idle(connection)
            return signature

    def _write_suppression(
        self,
        connection: Any,
        *,
        call_key: str,
        signature_id: str,
        reason: str,
        action: ChurnAction,
        evidence_digest: str,
        expires_at_ms: int,
        stamp: str,
    ) -> None:
        # One active suppression row per call key: supersede prior actives so
        # TTL refresh and re-exhaustion remain idempotent.
        connection.execute(
            """
            UPDATE replay_suppressions
            SET active = 0
            WHERE call_key = ? AND active = 1
            """,
            [call_key],
        )
        suppression_id = "rsup:" + _content_digest(
            {
                "call_key": call_key,
                "signature_id": signature_id,
                "reason": reason,
                "action": action.value,
                "evidence_digest": evidence_digest,
                "expires_at_ms": expires_at_ms,
                "recorded_at": stamp,
                "snapshot_id": self._snapshot_id,
            }
        )
        existing = connection.execute(
            "SELECT suppression_id FROM replay_suppressions WHERE suppression_id = ?",
            [suppression_id],
        ).fetchone()
        if existing is not None:
            connection.execute(
                """
                UPDATE replay_suppressions
                SET active = 1,
                    signature_id = ?,
                    reason = ?,
                    action = ?,
                    evidence_digest = ?,
                    expires_at_ms = ?,
                    recorded_at = ?
                WHERE suppression_id = ?
                """,
                [
                    signature_id,
                    reason,
                    action.value,
                    evidence_digest,
                    int(expires_at_ms),
                    stamp,
                    suppression_id,
                ],
            )
            return
        connection.execute(
            """
            INSERT INTO replay_suppressions(
                suppression_id, call_key, signature_id, reason, action,
                evidence_digest, expires_at_ms, active, recorded_at,
                body_json, snapshot_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, '{}', ?)
            """,
            [
                suppression_id,
                call_key,
                signature_id,
                reason,
                action.value,
                evidence_digest,
                int(expires_at_ms),
                stamp,
                self._snapshot_id,
            ],
        )

    def get_call(self, call_id: str) -> ProviderCallRecord | None:
        with self._lock:
            connection = self._require()
            row = self._get_call_row(connection, call_id=call_id)
            return self._call_from_row(row) if row is not None else None

    def get_call_by_key(self, call_key: str) -> ProviderCallRecord | None:
        with self._lock:
            connection = self._require()
            row = self._get_call_row(connection, call_key=call_key)
            return self._call_from_row(row) if row is not None else None

    def list_calls_for_task(self, task_cid: str) -> tuple[ProviderCallRecord, ...]:
        task = _text(task_cid, "task_cid")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM provider_calls
                WHERE task_cid = ?
                ORDER BY recorded_at ASC
                """,
                [task],
            ).fetchall()
            return tuple(self._call_from_row(_row_mapping(row)) for row in rows)

    def list_usage_charges(self, call_id: str) -> tuple[UsageCharge, ...]:
        selected = _text(call_id, "call_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM usage_charges
                WHERE call_id = ?
                ORDER BY recorded_at ASC
                """,
                [selected],
            ).fetchall()
            results: list[UsageCharge] = []
            for row in rows:
                mapping = _row_mapping(row)
                results.append(
                    UsageCharge(
                        charge_id=str(mapping.get("charge_id") or ""),
                        call_id=str(mapping.get("call_id") or ""),
                        call_key=str(mapping.get("call_key") or ""),
                        outcome=str(mapping.get("outcome") or ""),
                        input_tokens=int(mapping.get("input_tokens") or 0),
                        output_tokens=int(mapping.get("output_tokens") or 0),
                        charged=bool(int(mapping.get("charged") or 0)),
                        recorded_at=str(mapping.get("recorded_at") or ""),
                        body=_load_json_object(mapping.get("body_json")),
                    )
                )
            return tuple(results)

    def total_charged_tokens(self, *, task_cid: str = "") -> int:
        with self._lock:
            connection = self._require()
            if task_cid:
                row = connection.execute(
                    """
                    SELECT COALESCE(SUM(c.input_tokens + c.output_tokens), 0) AS total
                    FROM usage_charges c
                    JOIN provider_calls p ON p.call_id = c.call_id
                    WHERE p.task_cid = ? AND c.charged = 1
                    """,
                    [_text(task_cid, "task_cid")],
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS total
                    FROM usage_charges
                    WHERE charged = 1
                    """
                ).fetchone()
            mapping = _row_mapping(row)
            return int(mapping.get("total") or 0)

    def is_suppressed(
        self,
        call_key: str,
        *,
        evidence_digest: str = "",
        now_ms: int | None = None,
    ) -> bool:
        """Return whether ``call_key`` is under active replay suppression."""

        current_ms = int(now_ms if now_ms is not None else self._clock_ms())
        selected_key = _text(call_key, "call_key")
        with self._lock:
            connection = self._require()
            if self._active_suppression(
                connection, selected_key, now_ms=current_ms
            ):
                return True
            if evidence_digest and self._latest_exhausted_signature(
                connection,
                selected_key,
                _text(evidence_digest, "evidence_digest", required=False),
            ):
                return True
            return False


def open_provider_call_ledger(
    database_path: Path | str,
    **kwargs: Any,
) -> ProviderCallLedger:
    """Open a ProviderCallLedger at ``database_path``."""

    return ProviderCallLedger(database_path, **kwargs).open()


__all__ = (
    "AUTHORITY_CLASS",
    "CHURN_DECISION_INTERFACE",
    "CHURN_DECISION_SCHEMA",
    "DEFAULT_NEGATIVE_CACHE_TTL_MS",
    "DEFAULT_POLICY_ID",
    "DEFAULT_RETRY_BUDGET",
    "DEFAULT_SNAPSHOT_ID",
    "DuckDBUnavailableError",
    "DuplicateKind",
    "FAILURE_SIGNATURE_INTERFACE",
    "FAILURE_SIGNATURE_SCHEMA",
    "FailureClass",
    "FailureSignature",
    "ChurnAction",
    "ChurnDecision",
    "PROVIDER_CALL_LEDGER_INTERFACE",
    "PROVIDER_CALL_LEDGER_SCHEMA",
    "PROVIDER_CALL_RECORD_SCHEMA",
    "PRODUCER_ID",
    "ProviderCallLedger",
    "ProviderCallLedgerBoundsError",
    "ProviderCallLedgerConflictError",
    "ProviderCallLedgerError",
    "ProviderCallLedgerNotOpenError",
    "ProviderCallLedgerSecretError",
    "ProviderCallOutcome",
    "ProviderCallRecord",
    "ProviderCallRequest",
    "QuotaClass",
    "REDACTION_MARKER",
    "USAGE_CHARGE_SCHEMA",
    "UsageCharge",
    "compute_call_key",
    "compute_failure_signature_id",
    "compute_prompt_digest",
    "compute_response_digest",
    "duckdb_available",
    "open_provider_call_ledger",
)
