"""Bounded Grok implementation and independent Codex review routing.

This module is the proposal-only provider boundary for ``CodeEditPacket@1``.
It intentionally does not know how to read a repository, apply a patch, prove
an obligation, or complete a task.  Those capabilities stay in supervisor
owned callbacks which are called only after proposal admission.

The route is strictly sequential:

``packet -> Grok proposal -> admission -> Codex review/repair -> admission``

An optional writer receives only the final admitted proposal and an explicit
writer lease ID.  Grok and Codex have independent quota latches, so exhaustion
of the review provider can safely fall back to the already-admitted Grok
proposal.  A caller may also configure a deterministic, no-model proposal
provider for local fallback.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import queue
import re
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


IMPLEMENTATION_PROVIDER_ROUTER_INTERFACE: Final = "ImplementationProviderRouter@1"
IMPLEMENTATION_PROVIDER_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/implementation-provider-request@1"
)
IMPLEMENTATION_PROVIDER_PROPOSAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/implementation-provider-proposal@1"
)
IMPLEMENTATION_PROVIDER_ROUTE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/implementation-provider-route@1"
)
PROVIDER_EXECUTION_RECEIPT_INTERFACE: Final = "ProviderExecutionReceipt@1"
PROVIDER_EXECUTION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-execution-receipt@1"
)
# SCA-615 production wiring: the only model-assisted implement/review route.
PRODUCTION_PROVIDER_ROUTE_INTERFACE: Final = "ProductionProviderRoute@1"
PRODUCTION_PROVIDER_ROUTE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-provider-route@1"
)
PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-provider-route-evaluation@1"
)
PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-review-chain-binding@1"
)
SCAEV615ROUTE: Final = "SCAEV615ROUTE"
SCAEV615ROUTE_COVERAGE: Final = (
    "typed-packet-route-only",
    "grok-cannot-self-review",
    "codex-bounded-proposal-evidence-slice",
    "apply-and-merge-bound-to-admitted-review-chain",
    "absent-degraded-stale-cross-task-receipts-remain-pending",
    "deterministic-only-invokes-no-model",
    "no-provider-receives-repository-corpus",
)

# These are protocol limits, not provider suggestions.  Size checks are over
# UTF-8 bytes and are inclusive at the boundary.
# Keep in lockstep with production_context_slice.MAX_PROVIDER_PROMPT_TOKENS:
# multi-file schema corrections (e.g. UIR-010) need ~24k+ context tokens under
# utf8-bytes/4, plus reserved envelope and correction feedback.
MAX_PROVIDER_PROMPT_TOKENS: Final = 65_536
MAX_PROVIDER_PROMPT_BYTES: Final = 512 * 1_024
MAX_PROVIDER_RESPONSE_BYTES: Final = 256 * 1_024
MAX_PROVIDER_TIMEOUT_SECONDS: Final = 600.0
MAX_PROVIDER_JSON_DEPTH: Final = 24
MAX_PROVIDER_JSON_ITEMS: Final = 8_192
REDACTION_MARKER: Final = "[REDACTED]"


class ProviderRole(str, Enum):
    GROK_IMPLEMENT = "grok-implement"
    CODEX_QUOTA_IMPLEMENT = "codex-quota-fallback-implement"
    CODEX_REVIEW = "codex-independent-review"
    NON_CODEX_REVIEW = "non-codex-independent-review"
    DETERMINISTIC_LOCAL = "deterministic-local"


class RouteStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FALLBACK = "fallback"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class ReviewPresence(str, Enum):
    """Explicit independent-review disposition recorded on every receipt."""

    INDEPENDENT = "independent_review"
    ABSENT = "review_absent"
    DEGRADED = "review_degraded"
    DECLINED = "review_declined"
    LOCAL_ONLY = "local_only"
    NOT_APPLICABLE = "not_applicable"


class ProviderReason(str, Enum):
    ROUTED = "bounded_provider_route"
    LOCAL_ONLY = "deterministic_local_route"
    GROK_QUOTA_EXHAUSTED = "grok_quota_exhausted"
    GROK_BUILD_QUOTA_EXHAUSTED = "grok_build_balance_exhausted"
    CODEX_QUOTA_IMPLEMENTED_REVIEW_PENDING = (
        "codex_quota_implemented_non_codex_review_pending"
    )
    NON_CODEX_REVIEW_REQUIRED = "non_codex_independent_review_required"
    CODEX_QUOTA_EXHAUSTED = "codex_quota_exhausted_grok_fallback"
    GROK_UNAVAILABLE = "grok_unavailable"
    CODEX_UNAVAILABLE = "codex_unavailable_grok_fallback"
    PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_FAILURE = "provider_failure"
    PROVIDER_RESPONSE_MALFORMED = "provider_response_malformed"
    PROVIDER_RESPONSE_TOO_LARGE = "provider_response_too_large"
    PROVIDER_AUTHORITY_CLAIM = "provider_authority_claim"
    PROMPT_TOO_LARGE = "provider_prompt_too_large"
    PROMPT_TOKEN_BUDGET = "provider_prompt_token_budget_exceeded"
    BROAD_CONTEXT_FORBIDDEN = "broad_repository_context_forbidden"
    PACKET_STALE = "packet_stale"
    PACKET_NOT_IMPLEMENTABLE = "packet_not_implementable"
    PACKET_MALFORMED = "packet_malformed"
    ADMISSION_REQUIRED = "proposal_admission_required"
    PROPOSAL_REJECTED = "proposal_rejected"
    REVIEW_REJECTED = "review_rejected"
    REVIEW_DECLINED = "review_declined"
    REVIEW_ABSENT = "review_absent"
    REVIEW_DEGRADED = "review_degraded"
    SELF_REVIEW_FORBIDDEN = "grok_self_review_forbidden"
    PROVIDERS_NOT_INDEPENDENT = "providers_not_independent"
    WRITER_LEASE_REQUIRED = "writer_lease_required"
    WRITER_NOT_CONFIGURED = "writer_not_configured"
    WRITE_FAILED = "admitted_write_failed"
    NO_FALLBACK = "no_deterministic_fallback"
    # Production admission dispositions (SCA-615): remain pending, never complete.
    RECEIPT_ABSENT = "provider_receipt_absent"
    RECEIPT_DEGRADED = "provider_receipt_degraded"
    RECEIPT_STALE = "provider_receipt_stale"
    RECEIPT_CROSS_TASK = "provider_receipt_cross_task"
    REVIEW_CHAIN_UNBOUND = "admitted_review_chain_unbound"
    RAW_MODEL_COMMAND_FORBIDDEN = "raw_model_command_forbidden_for_production_route"


class ProductionReceiptDisposition(str, Enum):
    """Disposition of a production provider receipt for completion/merge gates."""

    ADMITTED = "admitted"
    PENDING_ABSENT = "pending_absent"
    PENDING_DEGRADED = "pending_degraded"
    PENDING_STALE = "pending_stale"
    PENDING_CROSS_TASK = "pending_cross_task"
    PENDING_DECLINED = "pending_declined"
    PENDING_NOT_ADMITTED = "pending_not_admitted"
    REJECTED = "rejected"


class ProviderRoutingError(ValueError):
    """A typed fail-closed provider boundary error."""

    def __init__(self, message: str, *, reason_code: ProviderReason | str) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


class ProviderQuotaError(RuntimeError):
    """A provider reported quota or capacity exhaustion."""

    def __init__(
        self,
        message: str = "provider quota exhausted",
        *,
        reason_code: str = ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value)


class VerifiedGrokQuotaExhaustion(ProviderQuotaError):
    """Supervisor-observed, exact Grok Build balance-exhaustion signal.

    Only the native transport adapter may construct this signal after checking
    the CLI exit status and its structured transport event.  Provider/model
    response text and generic quota exceptions deliberately cannot authorize
    the Codex Terra implementation fallback.
    """

    def __init__(self) -> None:
        super().__init__(
            ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
            reason_code=ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
        )


@dataclass(frozen=True, slots=True)
class ProviderBounds:
    """Frozen prompt, response, and wall-clock bounds for every provider call."""

    max_prompt_tokens: int = MAX_PROVIDER_PROMPT_TOKENS
    max_prompt_bytes: int = MAX_PROVIDER_PROMPT_BYTES
    max_response_bytes: int = MAX_PROVIDER_RESPONSE_BYTES
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in ("max_prompt_tokens", "max_prompt_bytes", "max_response_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        timeout = self.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
            or float(timeout) > MAX_PROVIDER_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"timeout_seconds must be in (0, {MAX_PROVIDER_TIMEOUT_SECONDS:g}]"
            )
        object.__setattr__(self, "timeout_seconds", float(timeout))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_response_bytes": self.max_response_bytes,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(slots=True)
class ProviderQuotaLatch:
    """Small independent call quota/latch for one provider role.

    ``remaining_calls=None`` means that no local call-count quota is imposed.
    A provider quota exception latches only this instance until ``reset``.
    """

    remaining_calls: int | None = None
    exhausted: bool = False
    reason_code: str = ""
    attempts: int = 0
    _supervisor_observed_grok_build_quota_exhausted: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.remaining_calls is not None and (
            isinstance(self.remaining_calls, bool)
            or not isinstance(self.remaining_calls, int)
            or self.remaining_calls < 0
        ):
            raise ValueError("remaining_calls must be a non-negative integer or None")
        if self.remaining_calls == 0:
            self.exhausted = True
            if not self.reason_code:
                self.reason_code = ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value

    @property
    def available(self) -> bool:
        return not self.exhausted and self.remaining_calls != 0

    def acquire(self) -> bool:
        if not self.available:
            return False
        self.attempts += 1
        if self.remaining_calls is not None:
            self.remaining_calls -= 1
            if self.remaining_calls == 0:
                # The acquired invocation may run; future invocations may not.
                self.exhausted = True
                self._supervisor_observed_grok_build_quota_exhausted = False
                if not self.reason_code:
                    self.reason_code = ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value
        return True

    def latch(self, reason_code: str = "") -> None:
        self.exhausted = True
        self._supervisor_observed_grok_build_quota_exhausted = False
        self.reason_code = str(
            reason_code or ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value
        )

    def _latch_supervisor_observed_grok_build_quota_exhaustion(
        self,
        error: VerifiedGrokQuotaExhaustion,
    ) -> None:
        """Retain typed transport provenance across reuse of this latch."""

        self.exhausted = True
        self.reason_code = str(error.reason_code)
        self._supervisor_observed_grok_build_quota_exhausted = (
            self.reason_code
            == ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value
        )

    def reset(self, *, remaining_calls: int | None = None) -> None:
        if remaining_calls is not None and (
            isinstance(remaining_calls, bool)
            or not isinstance(remaining_calls, int)
            or remaining_calls < 0
        ):
            raise ValueError("remaining_calls must be a non-negative integer or None")
        self.remaining_calls = remaining_calls
        self.exhausted = remaining_calls == 0
        self.reason_code = (
            ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value
            if remaining_calls == 0
            else ""
        )
        self.attempts = 0
        self._supervisor_observed_grok_build_quota_exhausted = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "remaining_calls": self.remaining_calls,
            "exhausted": self.exhausted,
            "reason_code": self.reason_code,
            "attempts": self.attempts,
        }


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProviderRoutingError(
            "provider data must be canonical JSON",
            reason_code=ProviderReason.PACKET_MALFORMED,
        ) from exc


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _default_token_count(value: bytes) -> int:
    """Conservative deterministic token estimate used without a tokenizer."""

    return max(1, (len(value) + 3) // 4)


_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "client_secret",
        "credential",
        "credentials",
        "github_token",
        "access_token",
        "refresh_token",
        "token",
        "password",
        "passphrase",
        "private_key",
        "secret",
    }
)
_TEXT_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret|token)"
        r"(\s*[:=]\s*)[^\s,;]{4,}"
    ),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)


def _normalized_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _redact_text(value: str) -> str:
    result = value
    for pattern in _TEXT_SECRET_PATTERNS:
        if pattern.pattern.startswith(r"(?i)\b(bearer)"):
            result = pattern.sub(r"\1 " + REDACTION_MARKER, result)
        elif "PRIVATE KEY" in pattern.pattern:
            result = pattern.sub(REDACTION_MARKER, result)
        else:
            result = pattern.sub(r"\1\2" + REDACTION_MARKER, result)
    return result


def redact_provider_data(value: Any) -> Any:
    """Detach JSON data and recursively redact credential-bearing values."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProviderRoutingError(
                    "provider data keys must be strings",
                    reason_code=ProviderReason.PACKET_MALFORMED,
                )
            result[key] = (
                REDACTION_MARKER
                if _normalized_key(key) in _SENSITIVE_KEYS
                else redact_provider_data(item)
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [redact_provider_data(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        # Canonical encoding below rejects NaN and infinity.
        return value
    raise ProviderRoutingError(
        f"provider data contains unsupported {type(value).__name__}",
        reason_code=ProviderReason.PACKET_MALFORMED,
    )


_BROAD_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "all_files",
        "ast_body",
        "ast_nodes",
        "file_content",
        "file_contents",
        "full_repository",
        "proof_body",
        "receipt_body",
        "repository_body",
        "repository_content",
        "repository_corpus",
        "repository_files",
        "source_body",
        "source_code",
        "source_text",
        "workspace",
        "workspace_path",
    }
)


def _check_structure(
    value: Any,
    *,
    forbid_broad_context: bool,
    location: str = "payload",
    depth: int = 0,
    item_counter: list[int] | None = None,
) -> None:
    if depth > MAX_PROVIDER_JSON_DEPTH:
        raise ProviderRoutingError(
            "provider data exceeds its depth bound",
            reason_code=ProviderReason.PACKET_MALFORMED,
        )
    counter = item_counter if item_counter is not None else [0]
    counter[0] += 1
    if counter[0] > MAX_PROVIDER_JSON_ITEMS:
        raise ProviderRoutingError(
            "provider data exceeds its item bound",
            reason_code=ProviderReason.PACKET_MALFORMED,
        )
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ProviderRoutingError(
                    f"{location} keys must be strings",
                    reason_code=ProviderReason.PACKET_MALFORMED,
                )
            key = _normalized_key(raw_key)
            if forbid_broad_context and (
                key in _BROAD_CONTEXT_KEYS or key.endswith("_body")
            ):
                raise ProviderRoutingError(
                    f"{location}.{raw_key} would expose broad repository context",
                    reason_code=ProviderReason.BROAD_CONTEXT_FORBIDDEN,
                )
            _check_structure(
                item,
                forbid_broad_context=forbid_broad_context,
                location=f"{location}.{raw_key}",
                depth=depth + 1,
                item_counter=counter,
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            _check_structure(
                item,
                forbid_broad_context=forbid_broad_context,
                location=f"{location}[{index}]",
                depth=depth + 1,
                item_counter=counter,
            )


_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "code_proof_authoritative",
        "completion_authoritative",
        "mark_complete",
        "proof_authoritative",
        "semantic_authority",
    }
)
_STATUS_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {"completion_status", "proof_status", "task_status"}
)
_AUTHORITATIVE_STATUS_VALUES: Final[frozenset[str]] = frozenset(
    {"accepted", "complete", "completed", "done", "passed", "proved", "proven", "satisfied"}
)


def _reject_provider_authority(value: Any, *, location: str = "response") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalized_key(str(raw_key))
            if key in _AUTHORITY_KEYS and item not in (False, None, "", 0):
                raise ProviderRoutingError(
                    f"{location}.{raw_key} attempts to claim supervisor authority",
                    reason_code=ProviderReason.PROVIDER_AUTHORITY_CLAIM,
                )
            if (
                key in _STATUS_AUTHORITY_KEYS
                and str(item).strip().casefold() in _AUTHORITATIVE_STATUS_VALUES
            ):
                raise ProviderRoutingError(
                    f"{location}.{raw_key} attempts to change proof/completion state",
                    reason_code=ProviderReason.PROVIDER_AUTHORITY_CLAIM,
                )
            _reject_provider_authority(item, location=f"{location}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            _reject_provider_authority(item, location=f"{location}[{index}]")


@dataclass(frozen=True, slots=True)
class ProviderRequest(Mapping[str, Any]):
    """Canonical provider request.

    It implements ``Mapping`` so simple providers can treat the request as a
    dictionary.  ``prompt`` is the exact canonical JSON sent across the
    provider boundary.
    """

    role: ProviderRole
    packet_id: str
    snapshot_id: str
    task_id: str
    payload: Mapping[str, Any]
    bounds: ProviderBounds
    prompt: bytes
    prompt_tokens: int
    response_contract: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPLEMENTATION_PROVIDER_REQUEST_SCHEMA,
            "interface": IMPLEMENTATION_PROVIDER_ROUTER_INTERFACE,
            "role": self.role.value,
            "packet_id": self.packet_id,
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "provider_input": dict(self.payload),
            "bounds": self.bounds.to_dict(),
            "response_contract": dict(self.response_contract),
            "authority": {
                "provider_output_tier": "proposal",
                "repository_write_allowed": False,
                "proof_authoritative": False,
                "completion_authoritative": False,
            },
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProviderProposal:
    """One detached, redacted, non-authoritative provider proposal."""

    role: ProviderRole
    packet_id: str
    snapshot_id: str
    task_id: str
    payload: Mapping[str, Any]
    response_bytes: int
    response_digest: str
    admitted: bool = False
    admission_reason: str = ""

    @property
    def provider(self) -> str:
        return self.role.value

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def with_admission(self, accepted: bool, reason: str = "") -> "ProviderProposal":
        return ProviderProposal(
            role=self.role,
            packet_id=self.packet_id,
            snapshot_id=self.snapshot_id,
            task_id=self.task_id,
            payload=self.payload,
            response_bytes=self.response_bytes,
            response_digest=self.response_digest,
            admitted=bool(accepted),
            admission_reason=str(reason or ""),
        )

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": IMPLEMENTATION_PROVIDER_PROPOSAL_SCHEMA,
            "role": self.role.value,
            "packet_id": self.packet_id,
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "response_bytes": self.response_bytes,
            "response_digest": self.response_digest,
            "admitted": self.admitted,
            "admission_reason": self.admission_reason,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
        if include_payload:
            result["proposal"] = dict(self.payload)
        return result


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason_code: str = ""

    @classmethod
    def coerce(cls, value: Any) -> "AdmissionDecision":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(value, "" if value else ProviderReason.PROPOSAL_REJECTED.value)
        if isinstance(value, Mapping):
            accepted = value.get("accepted", value.get("admitted", False))
            return cls(
                bool(accepted),
                str(value.get("reason_code") or value.get("reason") or ""),
            )
        accepted = getattr(value, "accepted", getattr(value, "admitted", None))
        if accepted is not None:
            return cls(
                bool(accepted),
                str(
                    getattr(value, "reason_code", "")
                    or getattr(value, "reason", "")
                    or ""
                ),
            )
        raise ProviderRoutingError(
            "proposal admission gate returned an unsupported result",
            reason_code=ProviderReason.PROPOSAL_REJECTED,
        )


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    role: ProviderRole
    status: str
    reason_code: str
    prompt_bytes: int = 0
    prompt_tokens: int = 0
    response_bytes: int = 0
    prompt_digest: str = ""
    response_digest: str = ""
    execution_schema: str = ""
    execution_policy_id: str = ""
    execution_request_id: str = ""
    configured_provider: str = ""
    effective_provider: str = ""
    configured_model: str = ""
    child_result_schema: str = ""
    child_result_status: str = ""
    child_exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "status": self.status,
            "reason_code": self.reason_code,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens": self.prompt_tokens,
            "response_bytes": self.response_bytes,
            "prompt_digest": self.prompt_digest,
            "response_digest": self.response_digest,
            "execution_schema": self.execution_schema,
            "execution_policy_id": self.execution_policy_id,
            "execution_request_id": self.execution_request_id,
            "configured_provider": self.configured_provider,
            "effective_provider": self.effective_provider,
            "configured_model": self.configured_model,
            "child_result_schema": self.child_result_schema,
            "child_result_status": self.child_result_status,
            "child_exit_code": self.child_exit_code,
            "prompt_embedded": False,
            "response_embedded": False,
        }


@dataclass(frozen=True, slots=True)
class PacketIdentity:
    """Content-addressed identity of the bounded provider packet."""

    packet_id: str
    packet_cid: str
    packet_bytes: int
    snapshot_id: str = ""
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "packet_cid": self.packet_cid,
            "packet_bytes": self.packet_bytes,
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class ReviewChainStep:
    """One ordered step of the Grok implementation / Codex review chain."""

    role: str
    status: str
    reason_code: str
    admitted: bool = False
    response_digest: str = ""
    prompt_bytes: int = 0
    prompt_tokens: int = 0
    response_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status,
            "reason_code": self.reason_code,
            "admitted": self.admitted,
            "response_digest": self.response_digest,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens": self.prompt_tokens,
            "response_bytes": self.response_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProviderExecutionReceipt:
    """Receipt proving which provider ran, on which packet, with which review.

    A lane label is not a receipt.  Provider output never becomes completion or
    proof authority; independent Codex review is required for admission of a
    model-assisted result as reviewed.
    """

    receipt_id: str
    status: str
    reason_code: str
    provider: str
    packet: Mapping[str, Any]
    review_chain: tuple[ReviewChainStep, ...]
    review_presence: str
    admission: Mapping[str, Any]
    attempts: tuple[Mapping[str, Any], ...] = ()
    writer_lease_id: str = ""
    write_performed: bool = False
    fallback: bool = False
    selected_proposal_digest: str = ""
    implementation_proposal_digest: str = ""
    review_proposal_digest: str = ""

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def provider_result_admitted(self) -> bool:
        return bool(self.admission.get("provider_result_admitted"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_EXECUTION_RECEIPT_SCHEMA,
            "interface": PROVIDER_EXECUTION_RECEIPT_INTERFACE,
            "receipt_id": self.receipt_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "provider": self.provider,
            "packet": dict(self.packet),
            "review_chain": [step.to_dict() for step in self.review_chain],
            "review_presence": self.review_presence,
            "admission": dict(self.admission),
            "attempts": [dict(item) for item in self.attempts],
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "write_performed": self.write_performed,
            "fallback": self.fallback,
            "selected_proposal_digest": self.selected_proposal_digest,
            "implementation_proposal_digest": self.implementation_proposal_digest,
            "review_proposal_digest": self.review_proposal_digest,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }


def _packet_content_id(payload: Mapping[str, Any]) -> str:
    """Return a stable content identity for the redacted packet payload."""

    try:
        from ..proof.formal_verification_contracts import content_identity

        return str(content_identity(dict(payload)))
    except Exception:
        return _sha256(_canonical_bytes(payload))


def _bounded_evidence_slice(
    provider_input: Mapping[str, Any],
    *,
    packet_id: str,
    snapshot_id: str,
    task_id: str,
) -> dict[str, Any]:
    """Build the only packet slice Codex may see: proposal evidence, not corpus."""

    scope = provider_input.get("scope")
    acceptance = provider_input.get("acceptance")
    expansion: Any = []
    for key in (
        "expansion_handles",
        "expansion_references",
        "expansion_cids",
        "evidence_handles",
    ):
        if key in provider_input:
            expansion = provider_input[key]
            break
    goal = provider_input.get("goal")
    context_slice = provider_input.get("context_slice")
    correction_feedback = provider_input.get("correction_feedback")
    goal_ids: dict[str, Any] = {}
    if isinstance(goal, Mapping):
        for key in (
            "contract_ids",
            "obligation_ids",
            "acceptance_ids",
            "claim_ids",
            "property_ids",
        ):
            if key in goal:
                goal_ids[key] = goal[key]
    evidence = {
        "packet_id": packet_id,
        "snapshot_id": snapshot_id,
        "task_id": task_id,
        "scope": dict(scope) if isinstance(scope, Mapping) else scope,
        "acceptance": (
            dict(acceptance) if isinstance(acceptance, Mapping) else acceptance
        ),
        "goal_ids": goal_ids,
        "expansion_handles": expansion if expansion is not None else [],
        "authority": {
            "provider_output_tier": "proposal",
            "repository_write_allowed": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        },
    }
    # A reviewer cannot independently assess a patch against source it never
    # saw.  Production packets may attach the supervisor-built, CID-addressed
    # bounded context manifest; Codex receives that same exact manifest.  The
    # completed Codex envelope is still measured against the hard provider
    # prompt token/byte bounds in ``_request``.
    if isinstance(context_slice, Mapping):
        evidence["context_slice"] = dict(context_slice)
    if isinstance(correction_feedback, Mapping):
        # Correction feedback is already a bounded, content-addressed
        # projection of a verified strict denial event.  The independent
        # reviewer must receive the same complete finding set as the
        # implementer so approval cannot silently ignore an omitted finding.
        evidence["correction_feedback"] = dict(correction_feedback)
    return evidence


def _provider_response_contract(role: ProviderRole) -> dict[str, Any]:
    """Return the exact authority-free JSON shape requested from one role."""

    if role in {
        ProviderRole.GROK_IMPLEMENT,
        ProviderRole.CODEX_QUOTA_IMPLEMENT,
    }:
        shape: dict[str, Any] = {
            "proposal": {
                "declared_paths": ["repo/relative/path"],
                "files": [
                    {
                        "path": "repo/relative/path",
                        "content": "complete replacement text",
                    }
                ],
                "patch": "optional unified diff instead of files",
            }
        }
    elif role is ProviderRole.CODEX_REVIEW:
        shape = {
            "decision": "approve|reject",
            "findings": [],
            "proposal": "forbidden; Codex is an independent reviewer only",
        }
    else:
        shape = {"proposal": {}}
    return {
        "format": "canonical-json-object-only",
        "markdown_fences_allowed": False,
        "prose_outside_json_allowed": False,
        "authority_claims_allowed": False,
        "expected_shape": shape,
    }


def build_provider_execution_receipt(
    result: "ImplementationRoutingResult",
) -> ProviderExecutionReceipt:
    """Materialize a content-addressed ProviderExecutionReceipt@1 from a route."""

    review_chain = result.review_chain
    review_presence = result.review_presence
    provider = result.provider
    packet = result.packet.to_dict() if result.packet is not None else {
        "packet_id": result.packet_id,
        "packet_cid": "",
        "packet_bytes": 0,
        "snapshot_id": "",
        "task_id": "",
    }
    admission = {
        "proposal_only": True,
        "repository_write_allowed": bool(result.write_performed),
        "completion_authoritative": False,
        "proof_authoritative": False,
        "provider_result_admitted": result.provider_result_admitted,
        "independent_review": review_presence == ReviewPresence.INDEPENDENT.value,
        "review_presence": review_presence,
        "self_review": False,
        "writer_lease_bound": bool(result.write_performed and result.writer_lease_id),
    }
    body = {
        "schema": PROVIDER_EXECUTION_RECEIPT_SCHEMA,
        "interface": PROVIDER_EXECUTION_RECEIPT_INTERFACE,
        "status": result.status.value,
        "reason_code": result.reason_code,
        "provider": provider,
        "packet": packet,
        "review_chain": [step.to_dict() for step in review_chain],
        "review_presence": review_presence,
        "admission": admission,
        "attempts": [item.to_dict() for item in result.attempts],
        "writer_lease_id": result.writer_lease_id if result.write_performed else "",
        "write_performed": result.write_performed,
        "fallback": result.status is RouteStatus.FALLBACK,
        "selected_proposal_digest": (
            result.selected_proposal.response_digest
            if result.selected_proposal is not None
            else ""
        ),
        "implementation_proposal_digest": (
            result.implementation_proposal.response_digest
            if result.implementation_proposal is not None
            else ""
        ),
        "review_proposal_digest": (
            result.review_proposal.response_digest
            if result.review_proposal is not None
            else ""
        ),
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    receipt_id = _packet_content_id(body)
    return ProviderExecutionReceipt(
        receipt_id=receipt_id,
        status=result.status.value,
        reason_code=result.reason_code,
        provider=provider,
        packet=MappingProxyType(packet),
        review_chain=review_chain,
        review_presence=review_presence,
        admission=MappingProxyType(admission),
        attempts=tuple(MappingProxyType(item.to_dict()) for item in result.attempts),
        writer_lease_id=result.writer_lease_id if result.write_performed else "",
        write_performed=result.write_performed,
        fallback=result.status is RouteStatus.FALLBACK,
        selected_proposal_digest=body["selected_proposal_digest"],
        implementation_proposal_digest=body["implementation_proposal_digest"],
        review_proposal_digest=body["review_proposal_digest"],
    )


@dataclass(frozen=True, slots=True)
class ImplementationRoutingResult:
    status: RouteStatus
    reason_code: str
    packet_id: str = ""
    packet: PacketIdentity | None = None
    selected_proposal: ProviderProposal | None = None
    implementation_proposal: ProviderProposal | None = None
    review_proposal: ProviderProposal | None = None
    attempts: tuple[ProviderAttempt, ...] = ()
    write_performed: bool = False
    writer_lease_id: str = ""

    @property
    def admitted(self) -> bool:
        return bool(self.selected_proposal and self.selected_proposal.admitted)

    @property
    def deferred(self) -> bool:
        return self.status is RouteStatus.DEFERRED

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        # Provider routes never complete tasks.  Absent or degraded independent
        # review is explicit on the receipt and cannot satisfy authority.
        return False

    @property
    def provider(self) -> str:
        if self.selected_proposal is not None:
            return self.selected_proposal.role.value
        if self.implementation_proposal is not None:
            return self.implementation_proposal.role.value
        for attempt in self.attempts:
            return attempt.role.value
        return ""

    @property
    def review_presence(self) -> str:
        if self.status is RouteStatus.SUCCEEDED and self.review_proposal is not None:
            decision = self.review_proposal.payload.get("decision")
            if decision == "reject":
                return ReviewPresence.DECLINED.value
            if decision in {"approve", "repair", "replace"}:
                return ReviewPresence.INDEPENDENT.value
            return ReviewPresence.DEGRADED.value
        if self.reason_code in {
            ProviderReason.REVIEW_DECLINED.value,
            ProviderReason.REVIEW_REJECTED.value,
        }:
            return ReviewPresence.DECLINED.value
        if self.reason_code in {
            ProviderReason.LOCAL_ONLY.value,
        } or (
            self.selected_proposal is not None
            and self.selected_proposal.role is ProviderRole.DETERMINISTIC_LOCAL
            and self.implementation_proposal is not None
            and self.implementation_proposal.role is ProviderRole.DETERMINISTIC_LOCAL
        ):
            if self.reason_code == ProviderReason.LOCAL_ONLY.value:
                return ReviewPresence.LOCAL_ONLY.value
        if self.reason_code in {
            ProviderReason.CODEX_UNAVAILABLE.value,
            ProviderReason.CODEX_QUOTA_EXHAUSTED.value,
            ProviderReason.CODEX_QUOTA_IMPLEMENTED_REVIEW_PENDING.value,
            ProviderReason.NON_CODEX_REVIEW_REQUIRED.value,
            ProviderReason.REVIEW_ABSENT.value,
        }:
            return ReviewPresence.ABSENT.value
        if self.status is RouteStatus.FALLBACK and self.implementation_proposal is not None:
            return ReviewPresence.DEGRADED.value
        if self.review_proposal is None and self.implementation_proposal is not None:
            if self.status is RouteStatus.REJECTED and self.reason_code in {
                ProviderReason.PROPOSAL_REJECTED.value,
                ProviderReason.PROVIDER_AUTHORITY_CLAIM.value,
                ProviderReason.PROVIDER_RESPONSE_MALFORMED.value,
                ProviderReason.PROVIDER_RESPONSE_TOO_LARGE.value,
                ProviderReason.SELF_REVIEW_FORBIDDEN.value,
                ProviderReason.PROVIDERS_NOT_INDEPENDENT.value,
            }:
                return ReviewPresence.NOT_APPLICABLE.value
            return ReviewPresence.ABSENT.value
        if self.implementation_proposal is None:
            return ReviewPresence.NOT_APPLICABLE.value
        return ReviewPresence.DEGRADED.value

    @property
    def provider_result_admitted(self) -> bool:
        """True only when independent Codex review succeeded and was admitted."""

        return (
            self.status is RouteStatus.SUCCEEDED
            and self.review_proposal is not None
            and self.review_proposal.admitted
            and self.admitted
            and self.review_presence == ReviewPresence.INDEPENDENT.value
        )

    @property
    def review_chain(self) -> tuple[ReviewChainStep, ...]:
        steps: list[ReviewChainStep] = []
        attempt_by_role = {item.role: item for item in self.attempts}

        def _step_from_proposal(
            proposal: ProviderProposal | None,
            role: ProviderRole,
            *,
            default_status: str,
            default_reason: str,
        ) -> ReviewChainStep:
            attempt = attempt_by_role.get(role)
            if proposal is not None:
                if default_status in {
                    "absent",
                    "degraded",
                    "declined",
                    "not_applicable",
                    "failed",
                }:
                    step_status = default_status
                elif proposal.admitted:
                    step_status = "succeeded"
                else:
                    step_status = "failed"
                return ReviewChainStep(
                    role=role.value,
                    status=step_status,
                    reason_code=proposal.admission_reason or default_reason,
                    admitted=proposal.admitted,
                    response_digest=proposal.response_digest,
                    prompt_bytes=attempt.prompt_bytes if attempt else 0,
                    prompt_tokens=attempt.prompt_tokens if attempt else 0,
                    response_bytes=proposal.response_bytes,
                )
            if attempt is not None:
                # Prefer the explicit review-presence disposition over the raw
                # attempt status so absent/degraded review stays explicit.
                if default_status in {
                    "absent",
                    "degraded",
                    "declined",
                    "not_applicable",
                }:
                    step_status = default_status
                    step_reason = default_reason or attempt.reason_code
                else:
                    step_status = attempt.status
                    step_reason = attempt.reason_code or default_reason
                return ReviewChainStep(
                    role=role.value,
                    status=step_status,
                    reason_code=step_reason,
                    admitted=False,
                    response_digest=attempt.response_digest,
                    prompt_bytes=attempt.prompt_bytes,
                    prompt_tokens=attempt.prompt_tokens,
                    response_bytes=attempt.response_bytes,
                )
            return ReviewChainStep(
                role=role.value,
                status=default_status,
                reason_code=default_reason,
                admitted=False,
            )

        terra_assisted = any(
            item.role is ProviderRole.CODEX_QUOTA_IMPLEMENT
            for item in self.attempts
        ) or (
            self.implementation_proposal is not None
            and self.implementation_proposal.role
            is ProviderRole.CODEX_QUOTA_IMPLEMENT
        )
        if terra_assisted:
            steps.append(
                _step_from_proposal(
                    None,
                    ProviderRole.GROK_IMPLEMENT,
                    default_status="failed",
                    default_reason=ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
                )
            )
            terra_proposal = (
                self.implementation_proposal
                if self.implementation_proposal is not None
                and self.implementation_proposal.role
                is ProviderRole.CODEX_QUOTA_IMPLEMENT
                else None
            )
            terra_admitted = bool(terra_proposal and terra_proposal.admitted)
            steps.append(
                _step_from_proposal(
                    terra_proposal,
                    ProviderRole.CODEX_QUOTA_IMPLEMENT,
                    default_status="succeeded" if terra_admitted else "failed",
                    default_reason=(
                        ProviderReason.CODEX_QUOTA_IMPLEMENTED_REVIEW_PENDING.value
                        if terra_admitted
                        else self.reason_code
                    ),
                )
            )
            steps.append(
                _step_from_proposal(
                    None,
                    ProviderRole.NON_CODEX_REVIEW,
                    default_status="absent" if terra_admitted else "not_applicable",
                    default_reason=(
                        ProviderReason.NON_CODEX_REVIEW_REQUIRED.value
                        if terra_admitted
                        else self.reason_code
                    ),
                )
            )
            return tuple(steps)

        if (
            self.implementation_proposal is not None
            or any(item.role is ProviderRole.GROK_IMPLEMENT for item in self.attempts)
            or any(
                item.role is ProviderRole.DETERMINISTIC_LOCAL for item in self.attempts
            )
        ):
            if (
                self.implementation_proposal is not None
                and self.implementation_proposal.role is ProviderRole.DETERMINISTIC_LOCAL
            ) or any(
                item.role is ProviderRole.DETERMINISTIC_LOCAL for item in self.attempts
            ):
                steps.append(
                    _step_from_proposal(
                        self.implementation_proposal
                        if self.implementation_proposal is not None
                        and self.implementation_proposal.role
                        is ProviderRole.DETERMINISTIC_LOCAL
                        else None,
                        ProviderRole.DETERMINISTIC_LOCAL,
                        default_status=(
                            "succeeded"
                            if self.selected_proposal is not None
                            else "failed"
                        ),
                        default_reason=self.reason_code or ProviderReason.LOCAL_ONLY.value,
                    )
                )
            else:
                steps.append(
                    _step_from_proposal(
                        self.implementation_proposal
                        if self.implementation_proposal is not None
                        and self.implementation_proposal.role
                        is ProviderRole.GROK_IMPLEMENT
                        else (
                            self.implementation_proposal
                            if self.implementation_proposal is not None
                            else None
                        ),
                        ProviderRole.GROK_IMPLEMENT,
                        default_status=(
                            "succeeded"
                            if self.implementation_proposal is not None
                            and self.implementation_proposal.admitted
                            else (
                                "failed"
                                if any(
                                    item.role is ProviderRole.GROK_IMPLEMENT
                                    and item.status == "failed"
                                    for item in self.attempts
                                )
                                else "absent"
                            )
                        ),
                        default_reason=self.reason_code or ProviderReason.ROUTED.value,
                    )
                )

        # Independent review step is always explicit for model-assisted routes.
        model_assisted = any(
            item.role is ProviderRole.GROK_IMPLEMENT for item in self.attempts
        ) or (
            self.implementation_proposal is not None
            and self.implementation_proposal.role is ProviderRole.GROK_IMPLEMENT
        )
        if model_assisted or self.review_proposal is not None:
            presence = self.review_presence
            if presence == ReviewPresence.INDEPENDENT.value:
                default_status = "succeeded"
                default_reason = ProviderReason.ROUTED.value
            elif presence == ReviewPresence.DECLINED.value:
                default_status = "declined"
                default_reason = self.reason_code or ProviderReason.REVIEW_DECLINED.value
            elif presence == ReviewPresence.ABSENT.value:
                default_status = "absent"
                default_reason = self.reason_code or ProviderReason.REVIEW_ABSENT.value
            elif presence == ReviewPresence.DEGRADED.value:
                default_status = "degraded"
                default_reason = self.reason_code or ProviderReason.REVIEW_DEGRADED.value
            else:
                default_status = "not_applicable"
                default_reason = self.reason_code or ""
            steps.append(
                _step_from_proposal(
                    self.review_proposal,
                    ProviderRole.CODEX_REVIEW,
                    default_status=default_status,
                    default_reason=default_reason,
                )
            )
        return tuple(steps)

    @property
    def provider_receipt(self) -> ProviderExecutionReceipt:
        return build_provider_execution_receipt(self)

    def to_dict(self) -> dict[str, Any]:
        receipt = self.provider_receipt
        return {
            "schema": IMPLEMENTATION_PROVIDER_ROUTE_SCHEMA,
            "interface": IMPLEMENTATION_PROVIDER_ROUTER_INTERFACE,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "provider": self.provider,
            "packet_id": self.packet_id,
            "packet": self.packet.to_dict() if self.packet is not None else {
                "packet_id": self.packet_id,
                "packet_cid": "",
                "packet_bytes": 0,
                "snapshot_id": "",
                "task_id": "",
            },
            "review_chain": [step.to_dict() for step in self.review_chain],
            "review_presence": self.review_presence,
            "provider_receipt": receipt.to_dict(),
            "selected_proposal": (
                self.selected_proposal.to_dict()
                if self.selected_proposal is not None
                else None
            ),
            "implementation_proposal": (
                self.implementation_proposal.to_dict()
                if self.implementation_proposal is not None
                else None
            ),
            "review_proposal": (
                self.review_proposal.to_dict()
                if self.review_proposal is not None
                else None
            ),
            "attempts": [item.to_dict() for item in self.attempts],
            "write_performed": self.write_performed,
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "provider_result_admitted": self.provider_result_admitted,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }


ProviderCallable = Callable[[ProviderRequest], Any]
AdmissionCallable = Callable[..., Any]
WriterCallable = Callable[..., Any]
TokenCounter = Callable[[bytes], int]


def _call_with_supported_arguments(
    callback: Callable[..., Any],
    primary: Any,
    secondary: Any,
) -> Any:
    """Call a one- or two-positional-argument supervisor callback."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(primary)
    positional = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs or len(positional) >= 2:
        return callback(primary, secondary)
    return callback(primary)


def _json_no_duplicates(value: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    return json.loads(
        value,
        object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {item}")
        ),
    )


def _raw_response(raw: Any) -> tuple[Mapping[str, Any], bytes]:
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        raw = raw.to_dict()
    if isinstance(raw, bytes):
        encoded = raw
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProviderRoutingError(
                "provider response is not UTF-8",
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
            ) from exc
        try:
            raw = _json_no_duplicates(decoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRoutingError(
                "provider response is not valid JSON",
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
            ) from exc
    elif isinstance(raw, str):
        encoded = raw.encode("utf-8", errors="strict")
        try:
            raw = _json_no_duplicates(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRoutingError(
                "provider response is not valid JSON",
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
            ) from exc
    elif isinstance(raw, Mapping):
        encoded = _canonical_bytes(raw)
    else:
        raise ProviderRoutingError(
            "provider response must be a JSON object",
            reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
        )
    if not isinstance(raw, Mapping):
        raise ProviderRoutingError(
            "provider response JSON must contain an object",
            reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED,
        )
    return raw, encoded


def _declared_quota_failure(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return ""
    status = str(raw.get("status") or "").strip().casefold()
    reason = str(raw.get("reason_code") or raw.get("reason") or "").strip()
    combined = f"{status} {reason}".casefold()
    if status in {"quota_exhausted", "capacity_exhausted", "rate_limited"} or any(
        marker in combined
        for marker in ("quota_exhaust", "capacity_exhaust", "rate_limit")
    ):
        return reason or ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value
    return ""


def _invoke_with_timeout(
    provider: ProviderCallable,
    request: ProviderRequest,
) -> Any:
    outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            outcomes.put((True, provider(request)))
        except BaseException as exc:  # provider boundary; re-raised below
            outcomes.put((False, exc))

    thread = threading.Thread(
        target=target,
        name=f"contract-provider-{request.role.value}",
        daemon=True,
    )
    thread.start()
    thread.join(request.bounds.timeout_seconds)
    if thread.is_alive():
        raise ProviderRoutingError(
            f"{request.role.value} exceeded its timeout",
            reason_code=ProviderReason.PROVIDER_TIMEOUT,
        )
    succeeded, value = outcomes.get_nowait()
    if not succeeded:
        raise value
    return value


@dataclass(slots=True)
class ImplementationProviderRouter:
    """Route one current contract packet through bounded proposal providers."""

    grok_provider: ProviderCallable | None = None
    codex_provider: ProviderCallable | None = None
    deterministic_provider: ProviderCallable | None = None
    admission_gate: AdmissionCallable | None = None
    writer: WriterCallable | None = None
    bounds: ProviderBounds = field(default_factory=ProviderBounds)
    grok_quota: ProviderQuotaLatch = field(default_factory=ProviderQuotaLatch)
    codex_quota: ProviderQuotaLatch = field(default_factory=ProviderQuotaLatch)
    deterministic_quota: ProviderQuotaLatch = field(default_factory=ProviderQuotaLatch)
    token_counter: TokenCounter = _default_token_count
    codex_implementation_fallback_provider: ProviderCallable | None = None
    codex_implementation_fallback_quota: ProviderQuotaLatch = field(
        default_factory=ProviderQuotaLatch
    )
    enforce_provider_identity: bool = False
    _writer_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, ProviderBounds):
            if not isinstance(self.bounds, Mapping):
                raise TypeError("bounds must be ProviderBounds or a mapping")
            self.bounds = ProviderBounds(**dict(self.bounds))
        for name in (
            "grok_quota",
            "codex_quota",
            "deterministic_quota",
            "codex_implementation_fallback_quota",
        ):
            value = getattr(self, name)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, name, ProviderQuotaLatch(remaining_calls=value))
            elif not isinstance(value, ProviderQuotaLatch):
                raise TypeError(f"{name} must be ProviderQuotaLatch or an integer")
        if not callable(self.token_counter):
            raise TypeError("token_counter must be callable")
        if not isinstance(self.enforce_provider_identity, bool):
            raise TypeError("enforce_provider_identity must be a boolean")

    @property
    def quota_state(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                ProviderRole.GROK_IMPLEMENT.value: MappingProxyType(
                    self.grok_quota.to_dict()
                ),
                ProviderRole.CODEX_REVIEW.value: MappingProxyType(
                    self.codex_quota.to_dict()
                ),
                ProviderRole.CODEX_QUOTA_IMPLEMENT.value: MappingProxyType(
                    self.codex_implementation_fallback_quota.to_dict()
                ),
                ProviderRole.DETERMINISTIC_LOCAL.value: MappingProxyType(
                    self.deterministic_quota.to_dict()
                ),
            }
        )

    def _packet_fields(
        self,
        packet: Any,
        current_snapshot_id: str,
    ) -> tuple[str, str, str, Mapping[str, Any]]:
        if not current_snapshot_id or current_snapshot_id != current_snapshot_id.strip():
            raise ProviderRoutingError(
                "current_snapshot_id is required and canonical",
                reason_code=ProviderReason.PACKET_MALFORMED,
            )
        if getattr(packet, "implementable", True) is False:
            raise ProviderRoutingError(
                "contract packet is not implementable",
                reason_code=ProviderReason.PACKET_NOT_IMPLEMENTABLE,
            )
        assertion = getattr(packet, "assert_current", None)
        if callable(assertion):
            try:
                assertion(current_snapshot_id)
            except Exception as exc:
                raise ProviderRoutingError(
                    "contract packet is stale",
                    reason_code=ProviderReason.PACKET_STALE,
                ) from exc
        snapshot_id = str(
            getattr(packet, "snapshot_id", "")
            or getattr(packet, "repository_tree_id", "")
        )
        if snapshot_id != current_snapshot_id:
            raise ProviderRoutingError(
                "contract packet snapshot does not match current snapshot",
                reason_code=ProviderReason.PACKET_STALE,
            )
        packet_id = str(
            getattr(packet, "packet_id", "") or getattr(packet, "content_id", "")
        ).strip()
        task_id = str(getattr(packet, "task_id", "")).strip()
        raw_payload = getattr(packet, "provider_input_payload", None)
        if callable(raw_payload):
            raw_payload = raw_payload()
        if not packet_id or not task_id or not isinstance(raw_payload, Mapping):
            raise ProviderRoutingError(
                "packet_id, task_id, and provider_input_payload are required",
                reason_code=ProviderReason.PACKET_MALFORMED,
            )
        _check_structure(raw_payload, forbid_broad_context=True)
        payload = redact_provider_data(raw_payload)
        _canonical_bytes(payload)
        return packet_id, snapshot_id, task_id, payload

    def _request(
        self,
        *,
        role: ProviderRole,
        packet_id: str,
        snapshot_id: str,
        task_id: str,
        provider_input: Mapping[str, Any],
        admitted_proposal: ProviderProposal | None = None,
    ) -> ProviderRequest:
        # Grok (and local fallback) receive the bounded contract packet.
        # Independent Codex review receives only the admitted proposal plus a
        # narrow evidence slice — never the implementer's full goal corpus.
        if role is ProviderRole.CODEX_REVIEW:
            if admitted_proposal is None or not admitted_proposal.admitted:
                raise ProviderRoutingError(
                    "Codex may review only an admitted implementation proposal",
                    reason_code=ProviderReason.ADMISSION_REQUIRED,
                )
            if admitted_proposal.role is not ProviderRole.GROK_IMPLEMENT:
                raise ProviderRoutingError(
                    "independent Codex review requires an admitted Grok proposal",
                    reason_code=ProviderReason.PROVIDERS_NOT_INDEPENDENT,
                )
            payload = {
                "admitted_implementation_proposal": {
                    "role": admitted_proposal.role.value,
                    "response_digest": admitted_proposal.response_digest,
                    "proposal": dict(admitted_proposal.payload),
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                },
                "evidence_slice": _bounded_evidence_slice(
                    provider_input,
                    packet_id=packet_id,
                    snapshot_id=snapshot_id,
                    task_id=task_id,
                ),
            }
        else:
            payload = {"contract_packet": dict(provider_input)}
            if admitted_proposal is not None:
                if not admitted_proposal.admitted:
                    raise ProviderRoutingError(
                        "only an admitted implementation proposal may be attached",
                        reason_code=ProviderReason.ADMISSION_REQUIRED,
                    )
                payload["admitted_implementation_proposal"] = {
                    "role": admitted_proposal.role.value,
                    "response_digest": admitted_proposal.response_digest,
                    "proposal": dict(admitted_proposal.payload),
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                }
        response_contract = _provider_response_contract(role)
        envelope = {
            "schema": IMPLEMENTATION_PROVIDER_REQUEST_SCHEMA,
            "interface": IMPLEMENTATION_PROVIDER_ROUTER_INTERFACE,
            "role": role.value,
            "packet_id": packet_id,
            "snapshot_id": snapshot_id,
            "task_id": task_id,
            "provider_input": payload,
            "bounds": self.bounds.to_dict(),
            "response_contract": response_contract,
            "authority": {
                "provider_output_tier": "proposal",
                "repository_write_allowed": False,
                "proof_authoritative": False,
                "completion_authoritative": False,
            },
        }
        prompt = _canonical_bytes(envelope)
        try:
            prompt_tokens = self.token_counter(prompt)
        except Exception as exc:
            raise ProviderRoutingError(
                "token counter failed",
                reason_code=ProviderReason.PACKET_MALFORMED,
            ) from exc
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
        ):
            raise ProviderRoutingError(
                "token counter returned an invalid value",
                reason_code=ProviderReason.PACKET_MALFORMED,
            )
        if len(prompt) > self.bounds.max_prompt_bytes:
            raise ProviderRoutingError(
                "provider prompt exceeds its exact UTF-8 byte bound",
                reason_code=ProviderReason.PROMPT_TOO_LARGE,
            )
        if prompt_tokens > self.bounds.max_prompt_tokens:
            raise ProviderRoutingError(
                "provider prompt exceeds its token bound",
                reason_code=ProviderReason.PROMPT_TOKEN_BUDGET,
            )
        return ProviderRequest(
            role=role,
            packet_id=packet_id,
            snapshot_id=snapshot_id,
            task_id=task_id,
            payload=MappingProxyType(payload),
            bounds=self.bounds,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            response_contract=MappingProxyType(response_contract),
        )

    def _invoke(
        self,
        provider: ProviderCallable,
        latch: ProviderQuotaLatch,
        request: ProviderRequest,
    ) -> tuple[ProviderProposal, ProviderAttempt]:
        if not latch.acquire():
            reason_code = (
                latch.reason_code
                or ProviderReason.PROVIDER_QUOTA_EXHAUSTED.value
            )
            if (
                request.role is ProviderRole.GROK_IMPLEMENT
                and latch._supervisor_observed_grok_build_quota_exhausted
                and reason_code
                == ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value
            ):
                raise VerifiedGrokQuotaExhaustion()
            raise ProviderQuotaError(reason_code, reason_code=reason_code)
        try:
            raw = _invoke_with_timeout(provider, request)
        except VerifiedGrokQuotaExhaustion as exc:
            if request.role is ProviderRole.GROK_IMPLEMENT:
                latch._latch_supervisor_observed_grok_build_quota_exhaustion(
                    exc
                )
            else:
                latch.latch(exc.reason_code)
            raise
        except ProviderQuotaError as exc:
            latch.latch(exc.reason_code)
            raise
        except ProviderRoutingError:
            raise
        except Exception as exc:
            raise ProviderRoutingError(
                f"{request.role.value} failed: {type(exc).__name__}",
                reason_code=ProviderReason.PROVIDER_FAILURE,
            ) from exc
        quota_reason = _declared_quota_failure(raw)
        if quota_reason:
            latch.latch(quota_reason)
            raise ProviderQuotaError(quota_reason, reason_code=quota_reason)
        raw_payload, encoded = _raw_response(raw)
        if len(encoded) > self.bounds.max_response_bytes:
            raise ProviderRoutingError(
                "provider response exceeds its exact UTF-8 byte bound",
                reason_code=ProviderReason.PROVIDER_RESPONSE_TOO_LARGE,
            )
        _check_structure(raw_payload, forbid_broad_context=False)
        _reject_provider_authority(raw_payload)
        payload = redact_provider_data(raw_payload)
        execution = payload.get("supervisor_provider_execution")
        execution = dict(execution) if isinstance(execution, Mapping) else {}
        proposal = ProviderProposal(
            role=request.role,
            packet_id=request.packet_id,
            snapshot_id=request.snapshot_id,
            task_id=request.task_id,
            payload=MappingProxyType(payload),
            response_bytes=len(encoded),
            response_digest=_sha256(encoded),
        )
        attempt = ProviderAttempt(
            role=request.role,
            status="succeeded",
            reason_code=ProviderReason.ROUTED.value,
            prompt_bytes=len(request.prompt),
            prompt_tokens=request.prompt_tokens,
            response_bytes=len(encoded),
            prompt_digest=_sha256(request.prompt),
            response_digest=proposal.response_digest,
            execution_schema=str(execution.get("schema") or ""),
            execution_policy_id=str(execution.get("policy_id") or ""),
            execution_request_id=str(execution.get("request_id") or ""),
            configured_provider=str(execution.get("configured_provider") or ""),
            effective_provider=str(execution.get("effective_provider") or ""),
            configured_model=str(execution.get("configured_model") or ""),
            child_result_schema=str(execution.get("child_result_schema") or ""),
            child_result_status=str(execution.get("child_result_status") or ""),
            child_exit_code=(
                execution.get("child_exit_code")
                if isinstance(execution.get("child_exit_code"), int)
                and not isinstance(execution.get("child_exit_code"), bool)
                else None
            ),
        )
        return proposal, attempt

    def _admit(self, proposal: ProviderProposal) -> ProviderProposal:
        if self.admission_gate is None:
            raise ProviderRoutingError(
                "a supervisor-owned proposal admission gate is required",
                reason_code=ProviderReason.ADMISSION_REQUIRED,
            )
        try:
            raw_decision = _call_with_supported_arguments(
                self.admission_gate,
                proposal,
                proposal.role,
            )
            decision = AdmissionDecision.coerce(raw_decision)
        except ProviderRoutingError:
            raise
        except Exception as exc:
            raise ProviderRoutingError(
                "proposal admission gate failed closed",
                reason_code=ProviderReason.PROPOSAL_REJECTED,
            ) from exc
        return proposal.with_admission(decision.accepted, decision.reason_code)

    def _write(
        self,
        proposal: ProviderProposal,
        *,
        apply: bool,
        writer_lease_id: str,
    ) -> tuple[bool, str]:
        if not apply:
            return False, ""
        if not proposal.admitted:
            return False, ProviderReason.ADMISSION_REQUIRED.value
        if self.writer is None:
            return False, ProviderReason.WRITER_NOT_CONFIGURED.value
        if (
            not isinstance(writer_lease_id, str)
            or not writer_lease_id
            or writer_lease_id != writer_lease_id.strip()
        ):
            return False, ProviderReason.WRITER_LEASE_REQUIRED.value
        try:
            # This router never owns the durable lease, but it does preserve
            # its single-writer property inside one router instance.
            with self._writer_lock:
                _call_with_supported_arguments(
                    self.writer,
                    proposal,
                    writer_lease_id,
                )
        except Exception:
            return False, ProviderReason.WRITE_FAILED.value
        return True, ""

    @staticmethod
    def _error_attempt(
        role: ProviderRole,
        reason_code: str,
        request: ProviderRequest | None = None,
    ) -> ProviderAttempt:
        return ProviderAttempt(
            role=role,
            status="failed",
            reason_code=reason_code,
            prompt_bytes=len(request.prompt) if request else 0,
            prompt_tokens=request.prompt_tokens if request else 0,
            prompt_digest=_sha256(request.prompt) if request else "",
        )

    @staticmethod
    def _bound_provider_identity(provider: ProviderCallable | None) -> str:
        """Return a normalized supervisor-bound provider-family identity."""

        if provider is None:
            return ""
        return str(getattr(provider, "provider_name", "") or "").strip().casefold()

    def _verify_attempt_provider_identity(
        self,
        provider: ProviderCallable,
        attempt: ProviderAttempt,
    ) -> str:
        """Fail closed when production execution provenance is absent/aliased."""

        if not self.enforce_provider_identity:
            return str(
                attempt.effective_provider or attempt.configured_provider or ""
            ).strip().casefold()
        expected = self._bound_provider_identity(provider)
        configured = str(attempt.configured_provider or "").strip().casefold()
        effective = str(attempt.effective_provider or "").strip().casefold()
        if not expected or configured != expected or effective != expected:
            raise ProviderRoutingError(
                "production provider execution identity is absent or mismatched",
                reason_code=ProviderReason.PROVIDERS_NOT_INDEPENDENT,
            )
        return effective

    def _local_fallback(
        self,
        *,
        packet_id: str,
        snapshot_id: str,
        task_id: str,
        payload: Mapping[str, Any],
        packet: PacketIdentity | None = None,
        attempts: list[ProviderAttempt],
        fallback_reason: str,
        apply: bool,
        writer_lease_id: str,
    ) -> ImplementationRoutingResult:
        packet_identity = packet or self._packet_identity(
            packet_id=packet_id,
            snapshot_id=snapshot_id,
            task_id=task_id,
            payload=payload,
        )
        if self.deterministic_provider is None:
            return self._result(
                status=RouteStatus.DEFERRED,
                reason_code=fallback_reason or ProviderReason.NO_FALLBACK.value,
                packet_id=packet_id,
                packet=packet_identity,
                attempts=attempts,
            )
        try:
            request = self._request(
                role=ProviderRole.DETERMINISTIC_LOCAL,
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                provider_input=payload,
            )
            proposal, attempt = self._invoke(
                self.deterministic_provider,
                self.deterministic_quota,
                request,
            )
            attempts.append(attempt)
            admitted = self._admit(proposal)
            if not admitted.admitted:
                return self._result(
                    status=RouteStatus.REJECTED,
                    reason_code=admitted.admission_reason
                    or ProviderReason.PROPOSAL_REJECTED.value,
                    packet_id=packet_id,
                    packet=packet_identity,
                    implementation_proposal=admitted,
                    attempts=attempts,
                )
            wrote, write_reason = self._write(
                admitted,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )
            if apply and not wrote:
                return self._result(
                    status=RouteStatus.REJECTED,
                    reason_code=write_reason,
                    packet_id=packet_id,
                    packet=packet_identity,
                    selected_proposal=admitted,
                    implementation_proposal=admitted,
                    attempts=attempts,
                )
            return self._result(
                status=RouteStatus.FALLBACK,
                reason_code=fallback_reason or ProviderReason.LOCAL_ONLY.value,
                packet_id=packet_id,
                packet=packet_identity,
                selected_proposal=admitted,
                implementation_proposal=admitted,
                attempts=attempts,
                write_performed=wrote,
                writer_lease_id=writer_lease_id if wrote else "",
            )
        except ProviderQuotaError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.DETERMINISTIC_LOCAL,
                    exc.reason_code,
                )
            )
            return self._result(
                status=RouteStatus.DEFERRED,
                reason_code=exc.reason_code or ProviderReason.NO_FALLBACK.value,
                packet_id=packet_id,
                packet=packet_identity,
                attempts=attempts,
            )
        except ProviderRoutingError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.DETERMINISTIC_LOCAL,
                    exc.reason_code,
                    locals().get("request"),
                )
            )
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=exc.reason_code,
                packet_id=packet_id,
                packet=packet_identity,
                attempts=attempts,
            )

    def _codex_quota_implementation_fallback(
        self,
        *,
        packet_id: str,
        snapshot_id: str,
        task_id: str,
        payload: Mapping[str, Any],
        packet: PacketIdentity,
        attempts: list[ProviderAttempt],
    ) -> ImplementationRoutingResult:
        """Collect a Terra-authored proposal that still needs non-Codex review.

        The fallback shares the ``codex_cli`` provider family with the normal
        Codex reviewer.  Its output is therefore evidence only: this branch
        never invokes that reviewer, never calls the writer, and never creates
        a review-chain binding.
        """

        provider = self.codex_implementation_fallback_provider
        if provider is None:
            return self._result(
                status=RouteStatus.DEFERRED,
                reason_code=ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
                packet_id=packet_id,
                packet=packet,
                attempts=attempts,
            )
        request: ProviderRequest | None = None
        try:
            request = self._request(
                role=ProviderRole.CODEX_QUOTA_IMPLEMENT,
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                provider_input=payload,
            )
            proposal, attempt = self._invoke(
                provider,
                self.codex_implementation_fallback_quota,
                request,
            )
            self._verify_attempt_provider_identity(provider, attempt)
            attempts.append(attempt)
            proposal = self._admit(proposal)
        except ProviderQuotaError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.CODEX_QUOTA_IMPLEMENT,
                    exc.reason_code,
                    request,
                )
            )
            return self._result(
                status=RouteStatus.DEFERRED,
                reason_code=exc.reason_code,
                packet_id=packet_id,
                packet=packet,
                attempts=attempts,
            )
        except ProviderRoutingError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.CODEX_QUOTA_IMPLEMENT,
                    exc.reason_code,
                    request,
                )
            )
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=exc.reason_code,
                packet_id=packet_id,
                packet=packet,
                attempts=attempts,
            )
        if not proposal.admitted:
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=(
                    proposal.admission_reason
                    or ProviderReason.PROPOSAL_REJECTED.value
                ),
                packet_id=packet_id,
                packet=packet,
                implementation_proposal=proposal,
                attempts=attempts,
            )
        return self._result(
            status=RouteStatus.DEFERRED,
            reason_code=ProviderReason.NON_CODEX_REVIEW_REQUIRED.value,
            packet_id=packet_id,
            packet=packet,
            implementation_proposal=proposal,
            attempts=attempts,
            write_performed=False,
            writer_lease_id="",
        )

    def _packet_identity(
        self,
        *,
        packet_id: str,
        snapshot_id: str,
        task_id: str,
        payload: Mapping[str, Any],
    ) -> PacketIdentity:
        encoded = _canonical_bytes(payload)
        return PacketIdentity(
            packet_id=packet_id,
            packet_cid=_packet_content_id(payload),
            packet_bytes=len(encoded),
            snapshot_id=snapshot_id,
            task_id=task_id,
        )

    def _result(
        self,
        *,
        status: RouteStatus,
        reason_code: str,
        packet_id: str = "",
        packet: PacketIdentity | None = None,
        selected_proposal: ProviderProposal | None = None,
        implementation_proposal: ProviderProposal | None = None,
        review_proposal: ProviderProposal | None = None,
        attempts: Sequence[ProviderAttempt] = (),
        write_performed: bool = False,
        writer_lease_id: str = "",
    ) -> ImplementationRoutingResult:
        return ImplementationRoutingResult(
            status=status,
            reason_code=reason_code,
            packet_id=packet_id or (packet.packet_id if packet is not None else ""),
            packet=packet,
            selected_proposal=selected_proposal,
            implementation_proposal=implementation_proposal,
            review_proposal=review_proposal,
            attempts=tuple(attempts),
            write_performed=write_performed,
            writer_lease_id=writer_lease_id if write_performed else "",
        )

    def route(
        self,
        packet: Any,
        *,
        current_snapshot_id: str,
        local_only: bool = False,
        apply: bool = False,
        writer_lease_id: str = "",
    ) -> ImplementationRoutingResult:
        """Route a current packet and optionally apply its admitted proposal.

        ``apply=False`` is deliberately the default.  A write requires both a
        configured supervisor writer and a non-empty lease ID.
        """

        attempts: list[ProviderAttempt] = []
        try:
            packet_id, snapshot_id, task_id, payload = self._packet_fields(
                packet, current_snapshot_id
            )
        except ProviderRoutingError as exc:
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=exc.reason_code,
            )

        packet_identity = self._packet_identity(
            packet_id=packet_id,
            snapshot_id=snapshot_id,
            task_id=task_id,
            payload=payload,
        )

        if isinstance(payload.get("correction_feedback"), Mapping):
            # Correction feedback is mandatory evidence for every permitted
            # provider role.  Prove the complete implementer/Terra envelope
            # and a minimally useful independent-review envelope fit before
            # spending either provider's quota.  The actual admitted proposal
            # is measured again before Codex can be invoked.
            probe_body = {"declared_paths": [], "files": []}
            probe_payload = MappingProxyType(probe_body)
            probe_bytes = _canonical_bytes(probe_body)
            probe = ProviderProposal(
                role=ProviderRole.GROK_IMPLEMENT,
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                payload=probe_payload,
                response_bytes=len(probe_bytes),
                response_digest=_sha256(probe_bytes),
                admitted=True,
            )
            try:
                self._request(
                    role=ProviderRole.GROK_IMPLEMENT,
                    packet_id=packet_id,
                    snapshot_id=snapshot_id,
                    task_id=task_id,
                    provider_input=payload,
                )
                if self.codex_implementation_fallback_provider is not None:
                    self._request(
                        role=ProviderRole.CODEX_QUOTA_IMPLEMENT,
                        packet_id=packet_id,
                        snapshot_id=snapshot_id,
                        task_id=task_id,
                        provider_input=payload,
                    )
                if self.codex_provider is not None:
                    self._request(
                        role=ProviderRole.CODEX_REVIEW,
                        packet_id=packet_id,
                        snapshot_id=snapshot_id,
                        task_id=task_id,
                        provider_input=payload,
                        admitted_proposal=probe,
                    )
            except ProviderRoutingError as exc:
                return self._result(
                    status=RouteStatus.REJECTED,
                    reason_code=exc.reason_code,
                    packet_id=packet_id,
                    packet=packet_identity,
                )

        if local_only:
            return self._local_fallback(
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                payload=payload,
                packet=packet_identity,
                attempts=attempts,
                fallback_reason=ProviderReason.LOCAL_ONLY.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )
        if self.grok_provider is None:
            return self._local_fallback(
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                payload=payload,
                packet=packet_identity,
                attempts=attempts,
                fallback_reason=ProviderReason.GROK_UNAVAILABLE.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )

        # Grok cannot self-review: implementer and reviewer must be independent
        # callables.  A lane label is not a receipt of independence.
        if (
            self.codex_provider is not None
            and self.grok_provider is self.codex_provider
        ):
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=ProviderReason.SELF_REVIEW_FORBIDDEN.value,
                packet_id=packet_id,
                packet=packet_identity,
            )
        if self.enforce_provider_identity and self.codex_provider is not None:
            grok_identity = self._bound_provider_identity(self.grok_provider)
            codex_identity = self._bound_provider_identity(self.codex_provider)
            if (
                not grok_identity
                or not codex_identity
                or grok_identity == codex_identity
            ):
                return self._result(
                    status=RouteStatus.REJECTED,
                    reason_code=ProviderReason.PROVIDERS_NOT_INDEPENDENT.value,
                    packet_id=packet_id,
                    packet=packet_identity,
                )

        grok_request: ProviderRequest | None = None
        try:
            grok_request = self._request(
                role=ProviderRole.GROK_IMPLEMENT,
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                provider_input=payload,
            )
            grok, attempt = self._invoke(
                self.grok_provider, self.grok_quota, grok_request
            )
            grok_effective_identity = self._verify_attempt_provider_identity(
                self.grok_provider,
                attempt,
            )
            attempts.append(attempt)
            grok = self._admit(grok)
        except VerifiedGrokQuotaExhaustion as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.GROK_IMPLEMENT, exc.reason_code, grok_request
                )
            )
            if self.codex_implementation_fallback_provider is not None:
                return self._codex_quota_implementation_fallback(
                    packet_id=packet_id,
                    snapshot_id=snapshot_id,
                    task_id=task_id,
                    payload=payload,
                    packet=packet_identity,
                    attempts=attempts,
                )
            return self._local_fallback(
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                payload=payload,
                packet=packet_identity,
                attempts=attempts,
                fallback_reason=ProviderReason.GROK_BUILD_QUOTA_EXHAUSTED.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )
        except ProviderQuotaError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.GROK_IMPLEMENT, exc.reason_code, grok_request
                )
            )
            return self._local_fallback(
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                payload=payload,
                packet=packet_identity,
                attempts=attempts,
                fallback_reason=ProviderReason.GROK_QUOTA_EXHAUSTED.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )
        except ProviderRoutingError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.GROK_IMPLEMENT, exc.reason_code, grok_request
                )
            )
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=exc.reason_code,
                packet_id=packet_id,
                packet=packet_identity,
                attempts=attempts,
            )
        if not grok.admitted:
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=grok.admission_reason
                or ProviderReason.PROPOSAL_REJECTED.value,
                packet_id=packet_id,
                packet=packet_identity,
                implementation_proposal=grok,
                attempts=attempts,
            )

        if self.codex_provider is None:
            return self._finish_with_grok(
                grok,
                attempts,
                packet=packet_identity,
                reason_code=ProviderReason.CODEX_UNAVAILABLE.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )

        codex_request: ProviderRequest | None = None
        try:
            codex_request = self._request(
                role=ProviderRole.CODEX_REVIEW,
                packet_id=packet_id,
                snapshot_id=snapshot_id,
                task_id=task_id,
                provider_input=payload,
                admitted_proposal=grok,
            )
            review, attempt = self._invoke(
                self.codex_provider, self.codex_quota, codex_request
            )
            codex_effective_identity = self._verify_attempt_provider_identity(
                self.codex_provider,
                attempt,
            )
            if (
                self.enforce_provider_identity
                and codex_effective_identity == grok_effective_identity
            ):
                raise ProviderRoutingError(
                    "implementation and review effective providers match",
                    reason_code=ProviderReason.PROVIDERS_NOT_INDEPENDENT,
                )
            attempts.append(attempt)
            review = self._admit(review)
        except ProviderQuotaError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.CODEX_REVIEW, exc.reason_code, codex_request
                )
            )
            return self._finish_with_grok(
                grok,
                attempts,
                packet=packet_identity,
                reason_code=ProviderReason.CODEX_QUOTA_EXHAUSTED.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )
        except ProviderRoutingError as exc:
            attempts.append(
                self._error_attempt(
                    ProviderRole.CODEX_REVIEW, exc.reason_code, codex_request
                )
            )
            # Grok has already passed the supervisor gate.  Review degradation
            # does not invalidate that admission, but it remains explicit and
            # cannot satisfy authoritative completion.
            return self._finish_with_grok(
                grok,
                attempts,
                packet=packet_identity,
                reason_code=exc.reason_code,
                apply=apply,
                writer_lease_id=writer_lease_id,
            )
        if not review.admitted:
            return self._finish_with_grok(
                grok,
                attempts,
                packet=packet_identity,
                reason_code=review.admission_reason
                or ProviderReason.REVIEW_REJECTED.value,
                apply=apply,
                writer_lease_id=writer_lease_id,
                review=review,
            )

        decision = review.payload.get("decision")
        if not isinstance(decision, str) or decision not in {
            "approve",
            "repair",
            "replace",
            "reject",
        }:
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED.value,
                packet_id=packet_id,
                packet=packet_identity,
                implementation_proposal=grok,
                review_proposal=review,
                attempts=attempts,
            )
        if decision == "reject":
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=ProviderReason.REVIEW_DECLINED.value,
                packet_id=packet_id,
                packet=packet_identity,
                implementation_proposal=grok,
                review_proposal=review,
                attempts=attempts,
            )
        if decision in {"repair", "replace"}:
            # Codex-authored replacement bytes have not been independently
            # reviewed.  Preserve both proposals as non-authoritative evidence
            # and require a new implementation/review round before any write.
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=ProviderReason.REVIEW_REJECTED.value,
                packet_id=packet_id,
                packet=packet_identity,
                implementation_proposal=grok,
                review_proposal=review,
                attempts=attempts,
            )
        # An approving review cannot simultaneously report findings.  Keep
        # this defense at the writer boundary as well as in the production
        # provider adapter so an injected/custom provider cannot turn a
        # contradictory review into an admitted repository mutation.
        if decision == "approve" and review.payload.get("findings") != []:
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED.value,
                packet_id=packet_id,
                packet=packet_identity,
                implementation_proposal=grok,
                review_proposal=review,
                attempts=attempts,
            )
        selected = grok
        if review.payload.get("proposal") not in (None, {}):
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=ProviderReason.PROVIDER_RESPONSE_MALFORMED.value,
                packet_id=packet_id,
                packet=packet_identity,
                implementation_proposal=grok,
                review_proposal=review,
                attempts=attempts,
            )
        wrote, write_reason = self._write(
            selected,
            apply=apply,
            writer_lease_id=writer_lease_id,
        )
        if apply and not wrote:
            return self._result(
                status=RouteStatus.REJECTED,
                reason_code=write_reason,
                packet_id=packet_id,
                packet=packet_identity,
                selected_proposal=selected,
                implementation_proposal=grok,
                review_proposal=review,
                attempts=attempts,
            )
        return self._result(
            status=RouteStatus.SUCCEEDED,
            reason_code=ProviderReason.ROUTED.value,
            packet_id=packet_id,
            packet=packet_identity,
            selected_proposal=selected,
            implementation_proposal=grok,
            review_proposal=review,
            attempts=attempts,
            write_performed=wrote,
            writer_lease_id=writer_lease_id if wrote else "",
        )

    def _finish_with_grok(
        self,
        grok: ProviderProposal,
        attempts: list[ProviderAttempt],
        *,
        packet: PacketIdentity | None,
        reason_code: str,
        apply: bool,
        writer_lease_id: str,
        review: ProviderProposal | None = None,
    ) -> ImplementationRoutingResult:
        # A Grok proposal is deliberately evidence-only until an independent
        # Codex review is present, well formed, admitted, and approving.  In
        # particular, review quota/error/degradation must never turn the
        # implementation proposal into a write-capable fallback.
        return self._result(
            status=RouteStatus.FALLBACK,
            reason_code=reason_code,
            packet_id=grok.packet_id,
            packet=packet,
            selected_proposal=grok,
            implementation_proposal=grok,
            review_proposal=review,
            attempts=attempts,
            write_performed=False,
            writer_lease_id="",
        )


@dataclass(frozen=True, slots=True)
class ProductionContractPacket:
    """Bounded production contract packet for model-assisted implement/review.

    Never embeds repository corpus, full source, or expansion bodies.  Providers
    receive only :attr:`provider_input_payload` (Grok) or the admitted proposal
    plus evidence slice (Codex).
    """

    packet_id: str
    snapshot_id: str
    task_id: str
    implementable: bool = True
    payload: Mapping[str, Any] = field(default_factory=dict)

    def assert_current(self, current_snapshot_id: str) -> None:
        if str(current_snapshot_id or "") != self.snapshot_id:
            raise ValueError("production contract packet is stale")

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "implementable": self.implementable,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class ProductionReviewChainBinding:
    """Binds an applied patch and optional merge to one admitted review chain.

    A lane label, raw exit code, or admission boolean is not a binding.
    """

    receipt_id: str
    task_id: str
    packet_id: str
    packet_cid: str
    snapshot_id: str
    review_chain_digest: str
    selected_proposal_digest: str
    implementation_proposal_digest: str
    review_proposal_digest: str
    writer_lease_id: str
    write_performed: bool
    review_presence: str
    provider_result_admitted: bool
    implementation_commit: str = ""
    merge_commit: str = ""
    disposition: str = ProductionReceiptDisposition.ADMITTED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA,
            "interface": PRODUCTION_PROVIDER_ROUTE_INTERFACE,
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "packet_id": self.packet_id,
            "packet_cid": self.packet_cid,
            "snapshot_id": self.snapshot_id,
            "review_chain_digest": self.review_chain_digest,
            "selected_proposal_digest": self.selected_proposal_digest,
            "implementation_proposal_digest": self.implementation_proposal_digest,
            "review_proposal_digest": self.review_proposal_digest,
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "write_performed": self.write_performed,
            "review_presence": self.review_presence,
            "provider_result_admitted": self.provider_result_admitted,
            "implementation_commit": self.implementation_commit,
            "merge_commit": self.merge_commit,
            "disposition": self.disposition,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }


def review_chain_content_digest(
    review_chain: Sequence[ReviewChainStep | Mapping[str, Any]],
) -> str:
    """Content identity of an ordered review chain (roles, digests, admission)."""

    steps: list[dict[str, Any]] = []
    for step in review_chain:
        if isinstance(step, ReviewChainStep):
            steps.append(step.to_dict())
        elif isinstance(step, Mapping):
            steps.append(dict(step))
        else:
            raise TypeError("review chain steps must be ReviewChainStep or mapping")
    return _packet_content_id({"review_chain": steps})


def evaluate_production_provider_receipt(
    receipt: ProviderExecutionReceipt | Mapping[str, Any] | None,
    *,
    expected_task_id: str,
    expected_snapshot_id: str,
    current_snapshot_id: str = "",
) -> tuple[ProductionReceiptDisposition, str]:
    """Fail-closed production admission for apply/merge/completion gates.

    Absent, degraded, stale, and cross-task receipts remain *pending* and never
    satisfy authoritative completion.  Independent admitted review is required
    for the ``ADMITTED`` disposition.
    """

    if receipt is None:
        return (
            ProductionReceiptDisposition.PENDING_ABSENT,
            ProviderReason.RECEIPT_ABSENT.value,
        )
    if isinstance(receipt, ProviderExecutionReceipt):
        payload = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        payload = dict(receipt)
    else:
        return (
            ProductionReceiptDisposition.REJECTED,
            ProviderReason.PACKET_MALFORMED.value,
        )

    task_id = str(expected_task_id or "").strip()
    if not task_id:
        return (
            ProductionReceiptDisposition.REJECTED,
            ProviderReason.PACKET_MALFORMED.value,
        )

    packet = payload.get("packet")
    packet_map = dict(packet) if isinstance(packet, Mapping) else {}
    receipt_task = str(
        packet_map.get("task_id") or payload.get("task_id") or ""
    ).strip()
    if receipt_task and receipt_task != task_id:
        return (
            ProductionReceiptDisposition.PENDING_CROSS_TASK,
            ProviderReason.RECEIPT_CROSS_TASK.value,
        )
    if not receipt_task:
        return (
            ProductionReceiptDisposition.PENDING_CROSS_TASK,
            ProviderReason.RECEIPT_CROSS_TASK.value,
        )

    receipt_snapshot = str(
        packet_map.get("snapshot_id") or payload.get("snapshot_id") or ""
    ).strip()
    expected_snapshot = str(expected_snapshot_id or "").strip()
    current = str(current_snapshot_id or expected_snapshot).strip()
    if not receipt_snapshot or not expected_snapshot:
        return (
            ProductionReceiptDisposition.PENDING_STALE,
            ProviderReason.RECEIPT_STALE.value,
        )
    if receipt_snapshot != expected_snapshot or (
        current and receipt_snapshot != current
    ):
        return (
            ProductionReceiptDisposition.PENDING_STALE,
            ProviderReason.RECEIPT_STALE.value,
        )

    presence = str(
        payload.get("review_presence")
        or (payload.get("admission") or {}).get("review_presence")
        or ""
    )
    if presence == ReviewPresence.ABSENT.value:
        return (
            ProductionReceiptDisposition.PENDING_ABSENT,
            ProviderReason.RECEIPT_ABSENT.value,
        )
    if presence == ReviewPresence.DEGRADED.value:
        return (
            ProductionReceiptDisposition.PENDING_DEGRADED,
            ProviderReason.RECEIPT_DEGRADED.value,
        )
    if presence == ReviewPresence.DECLINED.value:
        return (
            ProductionReceiptDisposition.PENDING_DECLINED,
            ProviderReason.REVIEW_DECLINED.value,
        )
    if presence != ReviewPresence.INDEPENDENT.value:
        return (
            ProductionReceiptDisposition.PENDING_NOT_ADMITTED,
            ProviderReason.REVIEW_CHAIN_UNBOUND.value,
        )

    admitted = bool(
        payload.get("provider_result_admitted")
        if "provider_result_admitted" in payload
        else (payload.get("admission") or {}).get("provider_result_admitted")
    )
    if not admitted:
        return (
            ProductionReceiptDisposition.PENDING_NOT_ADMITTED,
            ProviderReason.ADMISSION_REQUIRED.value,
        )
    if payload.get("completion_authoritative") is True:
        return (
            ProductionReceiptDisposition.REJECTED,
            ProviderReason.PROVIDER_AUTHORITY_CLAIM.value,
        )
    return (
        ProductionReceiptDisposition.ADMITTED,
        ProviderReason.ROUTED.value,
    )


def bind_applied_patch_to_review_chain(
    route_result: "ImplementationRoutingResult",
    *,
    writer_lease_id: str = "",
    implementation_commit: str = "",
    merge_commit: str = "",
) -> ProductionReviewChainBinding | None:
    """Bind apply/merge identity to the admitted independent review chain.

    Returns ``None`` when the route did not produce an independent admitted
    review chain.  Callers must leave merge/completion pending in that case.
    """

    if route_result is None:
        return None
    if not route_result.provider_result_admitted:
        return None
    if route_result.review_presence != ReviewPresence.INDEPENDENT.value:
        return None
    receipt = route_result.provider_receipt
    packet = route_result.packet
    packet_id = packet.packet_id if packet is not None else route_result.packet_id
    packet_cid = packet.packet_cid if packet is not None else ""
    snapshot_id = packet.snapshot_id if packet is not None else ""
    task_id = packet.task_id if packet is not None else ""
    chain_digest = review_chain_content_digest(route_result.review_chain)
    lease = writer_lease_id or (
        route_result.writer_lease_id if route_result.write_performed else ""
    )
    return ProductionReviewChainBinding(
        receipt_id=receipt.receipt_id,
        task_id=task_id,
        packet_id=packet_id,
        packet_cid=packet_cid,
        snapshot_id=snapshot_id,
        review_chain_digest=chain_digest,
        selected_proposal_digest=receipt.selected_proposal_digest,
        implementation_proposal_digest=receipt.implementation_proposal_digest,
        review_proposal_digest=receipt.review_proposal_digest,
        writer_lease_id=lease if route_result.write_performed else "",
        write_performed=bool(route_result.write_performed),
        review_presence=route_result.review_presence,
        provider_result_admitted=True,
        implementation_commit=str(implementation_commit or ""),
        merge_commit=str(merge_commit or ""),
        disposition=ProductionReceiptDisposition.ADMITTED.value,
    )


def build_production_contract_packet(
    *,
    task_id: str,
    snapshot_id: str,
    write_paths: Sequence[str],
    read_paths: Sequence[str] | None = None,
    validation_commands: Sequence[str] = (),
    acceptance_criteria: str = "",
    contract_ids: Sequence[str] = (),
    obligation_ids: Sequence[str] = (),
    expansion_handles: Sequence[Any] = (),
    packet_id: str = "",
    extra_goal: Mapping[str, Any] | None = None,
) -> ProductionContractPacket:
    """Build a bounded production packet that never embeds repository corpus."""

    tid = str(task_id or "").strip()
    snap = str(snapshot_id or "").strip()
    if not tid or not snap:
        raise ProviderRoutingError(
            "task_id and snapshot_id are required for a production packet",
            reason_code=ProviderReason.PACKET_MALFORMED,
        )
    writes = [str(path).strip() for path in write_paths if str(path).strip()]
    reads = [
        str(path).strip()
        for path in (read_paths if read_paths is not None else writes)
        if str(path).strip()
    ]
    if not writes:
        raise ProviderRoutingError(
            "production packet requires at least one write path",
            reason_code=ProviderReason.PACKET_MALFORMED,
        )
    goal: dict[str, Any] = {
        "contract_ids": list(contract_ids),
        "obligation_ids": list(obligation_ids),
        "task_id": tid,
    }
    if extra_goal:
        for key, value in extra_goal.items():
            normalized = _normalized_key(str(key))
            if normalized in _BROAD_CONTEXT_KEYS or normalized.endswith("_body"):
                raise ProviderRoutingError(
                    f"goal.{key} would expose broad repository context",
                    reason_code=ProviderReason.BROAD_CONTEXT_FORBIDDEN,
                )
            if key not in goal:
                goal[key] = value
    payload: dict[str, Any] = {
        "goal": goal,
        "authority": {
            "provider_semantic_authority": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        },
        "scope": {
            "read_paths": reads,
            "write_paths": writes,
        },
        "acceptance": {
            "validation_commands": [
                str(command) for command in validation_commands if str(command)
            ],
            "criteria": str(acceptance_criteria or ""),
        },
        "expansion_handles": list(expansion_handles),
    }
    _check_structure(payload, forbid_broad_context=True)
    pid = str(packet_id or "").strip() or f"packet:production:{tid}"
    return ProductionContractPacket(
        packet_id=pid,
        snapshot_id=snap,
        task_id=tid,
        implementable=True,
        payload=MappingProxyType(payload),
    )


def build_production_provider_route_evaluation(
    *,
    route_result: ImplementationRoutingResult | None = None,
    binding: ProductionReviewChainBinding | None = None,
    receipt_disposition: ProductionReceiptDisposition | str | None = None,
    deterministic_only_model_calls: int = 0,
    raw_model_command_invoked: bool = False,
    corpus_exposed_to_provider: bool = False,
    cases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the SCA-615 evaluation artifact for the production provider route."""

    disposition = (
        str(getattr(receipt_disposition, "value", receipt_disposition) or "")
        if receipt_disposition is not None
        else ""
    )
    if not disposition and binding is not None:
        disposition = binding.disposition
    if not disposition and route_result is not None:
        if route_result.provider_result_admitted:
            disposition = ProductionReceiptDisposition.ADMITTED.value
        elif route_result.review_presence == ReviewPresence.ABSENT.value:
            disposition = ProductionReceiptDisposition.PENDING_ABSENT.value
        elif route_result.review_presence == ReviewPresence.DEGRADED.value:
            disposition = ProductionReceiptDisposition.PENDING_DEGRADED.value
        else:
            disposition = ProductionReceiptDisposition.PENDING_NOT_ADMITTED.value

    route_payload = route_result.to_dict() if route_result is not None else {}
    binding_payload = binding.to_dict() if binding is not None else {}
    body = {
        "schema": PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA,
        "interface": PRODUCTION_PROVIDER_ROUTE_INTERFACE,
        "evidence": {
            "requirement_ids": [SCAEV615ROUTE],
            "coverage": list(SCAEV615ROUTE_COVERAGE),
            "objective_id": "SCA-615",
            "goal_id": "SCA-G177",
        },
        "production_route": {
            "typed_packet_route_only": not raw_model_command_invoked,
            "raw_model_command_invoked": bool(raw_model_command_invoked),
            "raw_model_command_forbidden": True,
            "router_interface": IMPLEMENTATION_PROVIDER_ROUTER_INTERFACE,
            "route_schema": IMPLEMENTATION_PROVIDER_ROUTE_SCHEMA,
        },
        "independence": {
            "grok_self_review_forbidden": True,
            "codex_receives_only_bounded_proposal_evidence_slice": True,
            "providers_must_be_distinct_callables": True,
        },
        "apply_merge_binding": {
            "requires_admitted_review_chain": True,
            "binding": binding_payload,
            "bound": bool(binding_payload),
        },
        "receipt_policy": {
            "absent_degraded_stale_cross_task_remain_pending": True,
            "disposition": disposition,
            "completion_authoritative": False,
        },
        "deterministic_only": {
            "invokes_no_model": deterministic_only_model_calls == 0,
            "model_call_count": int(deterministic_only_model_calls),
        },
        "corpus_isolation": {
            "provider_receives_repository_corpus": bool(corpus_exposed_to_provider),
            "forbidden": True,
            "broad_context_keys": sorted(_BROAD_CONTEXT_KEYS),
        },
        "route_result": {
            "status": route_payload.get("status", ""),
            "reason_code": route_payload.get("reason_code", ""),
            "provider": route_payload.get("provider", ""),
            "review_presence": route_payload.get("review_presence", ""),
            "provider_result_admitted": bool(
                route_payload.get("provider_result_admitted", False)
            ),
            "write_performed": bool(route_payload.get("write_performed", False)),
            "completion_authoritative": False,
            "proof_authoritative": False,
        },
        "cases": [dict(item) for item in cases],
        "acceptance": {
            "typed_packet_route_only": not raw_model_command_invoked,
            "grok_cannot_self_review": True,
            "codex_bounded_slice_only": True,
            "apply_merge_bound_to_review_chain": bool(binding_payload)
            or disposition
            in {
                ProductionReceiptDisposition.PENDING_ABSENT.value,
                ProductionReceiptDisposition.PENDING_DEGRADED.value,
                ProductionReceiptDisposition.PENDING_STALE.value,
                ProductionReceiptDisposition.PENDING_CROSS_TASK.value,
                ProductionReceiptDisposition.PENDING_DECLINED.value,
                ProductionReceiptDisposition.PENDING_NOT_ADMITTED.value,
            },
            "pending_receipts_remain_pending": disposition
            != ProductionReceiptDisposition.ADMITTED.value
            or bool(binding_payload),
            "deterministic_only_no_model": deterministic_only_model_calls == 0,
            "no_repository_corpus": not corpus_exposed_to_provider,
        },
    }
    body["evaluation_id"] = _packet_content_id(body)
    return body


def route_contract_packet(
    packet: Any,
    *,
    current_snapshot_id: str,
    grok_provider: ProviderCallable | None = None,
    codex_provider: ProviderCallable | None = None,
    deterministic_provider: ProviderCallable | None = None,
    admission_gate: AdmissionCallable | None = None,
    writer: WriterCallable | None = None,
    apply: bool = False,
    writer_lease_id: str = "",
    local_only: bool = False,
    bounds: ProviderBounds | Mapping[str, Any] | None = None,
    grok_quota: ProviderQuotaLatch | int | None = None,
    codex_quota: ProviderQuotaLatch | int | None = None,
    codex_implementation_fallback_provider: ProviderCallable | None = None,
    codex_implementation_fallback_quota: ProviderQuotaLatch | int | None = None,
    enforce_provider_identity: bool = False,
) -> ImplementationRoutingResult:
    """Functional facade for one bounded packet route."""

    router = ImplementationProviderRouter(
        grok_provider=grok_provider,
        codex_provider=codex_provider,
        deterministic_provider=deterministic_provider,
        admission_gate=admission_gate,
        writer=writer,
        bounds=bounds or ProviderBounds(),
        grok_quota=(
            grok_quota if grok_quota is not None else ProviderQuotaLatch()
        ),
        codex_quota=(
            codex_quota if codex_quota is not None else ProviderQuotaLatch()
        ),
        codex_implementation_fallback_provider=(
            codex_implementation_fallback_provider
        ),
        codex_implementation_fallback_quota=(
            codex_implementation_fallback_quota
            if codex_implementation_fallback_quota is not None
            else ProviderQuotaLatch()
        ),
        enforce_provider_identity=enforce_provider_identity,
    )
    return router.route(
        packet,
        current_snapshot_id=current_snapshot_id,
        local_only=local_only,
        apply=apply,
        writer_lease_id=writer_lease_id,
    )


# Compatibility aliases for callers using shorter provider-routing names.
ContractPacketProviderRouter = ImplementationProviderRouter
ImplementationRouteResult = ImplementationRoutingResult
ProviderRouteResult = ImplementationRoutingResult
QuotaLatch = ProviderQuotaLatch


__all__ = [
    "AdmissionDecision",
    "ContractPacketProviderRouter",
    "IMPLEMENTATION_PROVIDER_PROPOSAL_SCHEMA",
    "IMPLEMENTATION_PROVIDER_REQUEST_SCHEMA",
    "IMPLEMENTATION_PROVIDER_ROUTE_SCHEMA",
    "IMPLEMENTATION_PROVIDER_ROUTER_INTERFACE",
    "ImplementationProviderRouter",
    "ImplementationRouteResult",
    "ImplementationRoutingResult",
    "MAX_PROVIDER_JSON_DEPTH",
    "MAX_PROVIDER_JSON_ITEMS",
    "MAX_PROVIDER_PROMPT_BYTES",
    "MAX_PROVIDER_PROMPT_TOKENS",
    "MAX_PROVIDER_RESPONSE_BYTES",
    "MAX_PROVIDER_TIMEOUT_SECONDS",
    "PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA",
    "PRODUCTION_PROVIDER_ROUTE_INTERFACE",
    "PRODUCTION_PROVIDER_ROUTE_SCHEMA",
    "PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA",
    "ProductionContractPacket",
    "ProductionReceiptDisposition",
    "ProductionReviewChainBinding",
    "PROVIDER_EXECUTION_RECEIPT_INTERFACE",
    "PROVIDER_EXECUTION_RECEIPT_SCHEMA",
    "PacketIdentity",
    "ProviderAttempt",
    "ProviderBounds",
    "ProviderCallable",
    "ProviderExecutionReceipt",
    "ProviderProposal",
    "ProviderQuotaError",
    "ProviderQuotaLatch",
    "ProviderReason",
    "ProviderRequest",
    "ProviderRole",
    "ProviderRouteResult",
    "ProviderRoutingError",
    "QuotaLatch",
    "REDACTION_MARKER",
    "ReviewChainStep",
    "ReviewPresence",
    "RouteStatus",
    "SCAEV615ROUTE",
    "SCAEV615ROUTE_COVERAGE",
    "VerifiedGrokQuotaExhaustion",
    "bind_applied_patch_to_review_chain",
    "build_production_contract_packet",
    "build_production_provider_route_evaluation",
    "build_provider_execution_receipt",
    "evaluate_production_provider_receipt",
    "redact_provider_data",
    "review_chain_content_digest",
    "route_contract_packet",
]
