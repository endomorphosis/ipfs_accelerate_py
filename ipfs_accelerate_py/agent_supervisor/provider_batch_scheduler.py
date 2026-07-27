"""Provider-aware batching with member-local cancellation and evidence.

The supervisor has several producers of model work, but a provider should see
one bounded stream rather than one model server or request loop per producer.
This module is the small coordination boundary for that stream:

* only requests with an identical :class:`ProviderBatchKey` share a call;
* identical in-flight requests are single-flighted;
* deadlines, cancellation, budgets, provenance, and results remain local to
  each submitted request (existing sibling isolation and single-flight receipts);
* provider health and capacity are checked immediately before dispatch; and
* every completed batch has a content-addressed receipt.

IndexTTS/Whisper batch-size-one policy: IndexTTS and Whisper adapter aliases in
``_SINGLE_MEMBER_AUDIO_PROVIDERS`` remain physical batch size one until those
adapters prove real multi-member batching.

Provider callbacks receive a tuple of :class:`ProviderBatchRequest` objects and
may return either a sequence in request order or a mapping keyed by request id.
An exception returned as one member's value fails only that member.  An
exception raised by the callback fails the call; an optional member fallback
can still recover each request independently.

No floating point values are serialized.  Durations and ratios in metrics are
integer milliseconds and millionths so receipts remain suitable for strict
supervisor artifacts.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any, Final


PARTIAL_CANCELLATION_REQUIREMENT_ID: Final = (
    "124037811551945145648172208272779822741"
)
PROVIDER_BATCH_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-batch-receipt@1"
)
PROVIDER_BATCH_METRICS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-batch-metrics@2"
)
_PROVIDER_BATCH_RECEIPT_SEAL: Final = object()
_SINGLE_MEMBER_AUDIO_PROVIDERS: Final = frozenset(
    {
        "abby_hf_whisper",
        "abby_index_tts",
        "abby_indextts",
        "abby_whisper",
        "hf_whisper",
        "huggingface_whisper",
        "huggingface_whisper_http",
        "huggingfacewhisperhttp",
        "index_tts",
        "index_tts_http",
        "indextts",
        "indexttshttp",
        "whisper",
    }
)


def _canonical(value: Any) -> Any:
    """Return a deterministic JSON-safe value without exposing object reprs."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Provider settings occasionally contain temperatures.  Preserve their
        # canonical textual value rather than putting floats in evidence.
        return format(value, ".17g")
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    # Unknown opaque objects must never all collapse to the same single-flight
    # identity merely because they share a class.  The process-local identity
    # is intentionally not durable evidence, but it safely prevents a false
    # cache/single-flight hit while retaining a JSON-safe diagnostic value.
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "process_identity": id(value),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _cancelled(token: Any) -> bool:
    """Understand the cancellation conventions used by supervisor providers."""

    if token is None:
        return False
    value = getattr(token, "cancelled", None)
    if callable(value):
        try:
            return bool(value())
        except TypeError:
            pass
    if value is not None:
        return bool(value)
    method = getattr(token, "is_cancelled", None)
    if callable(method):
        return bool(method())
    method = getattr(token, "is_set", None)
    if callable(method):
        return bool(method())
    return False


def _requires_single_member_batch(provider_id: str) -> bool:
    """Enforce the IndexTTS/Whisper batch-size-one policy for audio adapters.

    Returns whether the provider's current adapter lacks a batch wire API and
    therefore must launch with at most one physical member per provider call.
    """

    normalized = str(provider_id).strip().lower().replace("-", "_")
    return normalized in _SINGLE_MEMBER_AUDIO_PROVIDERS


class ProviderBatchStatus(str, Enum):
    """Terminal and observable states of one submitted member."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FALLBACK = "fallback"

    @property
    def successful(self) -> bool:
        return self in {ProviderBatchStatus.SUCCEEDED, ProviderBatchStatus.FALLBACK}

    @property
    def terminal(self) -> bool:
        return self not in {ProviderBatchStatus.QUEUED, ProviderBatchStatus.RUNNING}


@dataclass(frozen=True)
class ProviderBatchKey:
    """All provider inputs which must match before two requests may batch."""

    provider_id: str
    route: str
    model: str
    operation: str
    context_limit: int
    policy_digest: str
    generation_digest: str
    voice: str = ""
    locale: str = ""
    reference_hash: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    tenant_policy_digest: str = field(default_factory=lambda: _digest({}))

    def __post_init__(self) -> None:
        for name in ("provider_id", "route", "model", "operation"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        for name in ("voice", "locale", "codec"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        reference_hash = str(self.reference_hash).strip().lower()
        if reference_hash and (
            len(reference_hash) != 64
            or any(character not in "0123456789abcdef" for character in reference_hash)
        ):
            raise ValueError("reference_hash must be an empty value or sha256 digest")
        object.__setattr__(self, "reference_hash", reference_hash)
        for name in ("context_limit", "sample_rate", "channels"):
            _positive_integer(getattr(self, name), name, allow_zero=True)
        for name in (
            "policy_digest",
            "tenant_policy_digest",
            "generation_digest",
        ):
            value = str(getattr(self, name)).lower()
            if len(value) != 64:
                raise ValueError(f"{name} must be a sha256 digest")
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a sha256 digest")
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "route": self.route,
            "model": self.model,
            "operation": self.operation,
            "context_limit": self.context_limit,
            "voice": self.voice,
            "locale": self.locale,
            "reference_hash": self.reference_hash,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "policy_digest": self.policy_digest,
            "tenant_policy_digest": self.tenant_policy_digest,
            "generation_digest": self.generation_digest,
        }


@dataclass(frozen=True)
class ProviderBatchRequest:
    """One independently budgeted unit of otherwise batch-compatible work."""

    request_id: str
    payload: Any
    provider_id: str = "default"
    route: str = "default"
    model: str = "default"
    operation: str = "generate"
    context_limit: int = 0
    voice: str = ""
    locale: str = ""
    reference_hash: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    policy: Mapping[str, Any] = field(default_factory=dict)
    tenant_policy: Mapping[str, Any] = field(default_factory=dict)
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    token_budget: int = 0
    timeout_ms: int = 0
    priority: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    cancellation_token: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ValueError("request_id must not be empty")
        object.__setattr__(self, "request_id", request_id)
        for name in ("provider_id", "route", "model", "operation"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        for name in ("voice", "locale", "codec"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        reference_hash = str(self.reference_hash).strip().lower()
        if reference_hash and (
            len(reference_hash) != 64
            or any(character not in "0123456789abcdef" for character in reference_hash)
        ):
            raise ValueError("reference_hash must be an empty value or sha256 digest")
        object.__setattr__(self, "reference_hash", reference_hash)
        for name in (
            "context_limit",
            "sample_rate",
            "channels",
            "token_budget",
            "timeout_ms",
        ):
            _positive_integer(getattr(self, name), name, allow_zero=True)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")
        for name in (
            "policy",
            "tenant_policy",
            "generation_settings",
            "provenance",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            # Copy to prevent caller mutation from changing compatibility or
            # evidence after submission.
            object.__setattr__(self, name, _canonical(value))

    @classmethod
    def from_value(
        cls, value: "ProviderBatchRequest | Mapping[str, Any]"
    ) -> "ProviderBatchRequest":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("provider batch request must be a request or mapping")
        return cls(**dict(value))

    @property
    def batch_key(self) -> ProviderBatchKey:
        return ProviderBatchKey(
            provider_id=self.provider_id,
            route=self.route,
            model=self.model,
            operation=self.operation,
            context_limit=self.context_limit,
            voice=self.voice,
            locale=self.locale,
            reference_hash=self.reference_hash,
            codec=self.codec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            policy_digest=_digest(self.policy),
            tenant_policy_digest=_digest(self.tenant_policy),
            generation_digest=_digest(self.generation_settings),
        )

    @property
    def execution_fingerprint(self) -> str:
        """Identity for single-flight work, excluding member-local metadata."""

        return _digest(
            {
                "batch_key": self.batch_key.to_dict(),
                "payload": self.payload,
                "token_budget": self.token_budget,
            }
        )

    def dispatch_copy(self) -> "ProviderBatchRequest":
        """Return a provider view without a member token that can kill siblings."""

        return replace(self, cancellation_token=None)


@dataclass(frozen=True)
class ProviderBatchCapacity:
    """Live provider limits sampled immediately before a batch dispatch."""

    provider_id: str
    healthy: bool = True
    max_batch_size: int = 0
    max_concurrent_batches: int = 1
    available_concurrent_batches: int = 1
    token_budget_remaining: int = -1
    retry_after_ms: int = 0

    def __post_init__(self) -> None:
        if not str(self.provider_id).strip():
            raise ValueError("provider_id must not be empty")
        object.__setattr__(self, "provider_id", str(self.provider_id).strip())
        for name in (
            "max_batch_size",
            "max_concurrent_batches",
            "available_concurrent_batches",
            "retry_after_ms",
        ):
            _positive_integer(getattr(self, name), name, allow_zero=True)
        if (
            isinstance(self.token_budget_remaining, bool)
            or not isinstance(self.token_budget_remaining, int)
            or self.token_budget_remaining < -1
        ):
            raise ValueError("token_budget_remaining must be -1 or non-negative")

    @classmethod
    def from_value(
        cls, provider_id: str, value: Any
    ) -> "ProviderBatchCapacity":
        if isinstance(value, cls):
            if value.provider_id != provider_id:
                raise ValueError("capacity provider does not match requested provider")
            return value
        if value is None:
            return cls(provider_id=provider_id)
        if isinstance(value, Mapping):
            fields = dict(value)
        else:
            fields = {
                name: getattr(value, name)
                for name in (
                    "healthy",
                    "max_batch_size",
                    "max_concurrent_batches",
                    "available_concurrent_batches",
                    "token_budget_remaining",
                    "retry_after_ms",
                )
                if hasattr(value, name)
            }
            # ResourceScheduler.ProviderCapacity uses request concurrency.
            if hasattr(value, "max_concurrency"):
                maximum = int(getattr(value, "max_concurrency"))
                active = int(getattr(value, "active_requests", 0))
                fields.setdefault("max_concurrent_batches", maximum)
                fields.setdefault(
                    "available_concurrent_batches", max(0, maximum - active)
                )
        fields["provider_id"] = provider_id
        return cls(**fields)


@dataclass(frozen=True)
class ProviderBatchSchedulerConfig:
    """Queue, batching, capacity, and adaptation policy."""

    max_batch_size: int = 8
    min_batch_size: int = 1
    batch_window_ms: int = 5
    max_queue_size: int = 1024
    max_parallel_batches: int = 4
    provider_limits: Mapping[str, int] = field(default_factory=dict)
    target_batch_latency_ms: int = 2_000
    admission_retry_ms: int = 10
    receipt_history: int = 256
    fallback_on_dispatch_error: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_batch_size",
            "min_batch_size",
            "max_queue_size",
            "max_parallel_batches",
            "target_batch_latency_ms",
            "admission_retry_ms",
            "receipt_history",
        ):
            _positive_integer(getattr(self, name), name)
        _positive_integer(self.batch_window_ms, "batch_window_ms", allow_zero=True)
        if self.min_batch_size > self.max_batch_size:
            raise ValueError("min_batch_size must not exceed max_batch_size")
        normalized: dict[str, int] = {}
        for provider_id, raw_limit in self.provider_limits.items():
            provider = str(provider_id).strip()
            if not provider:
                raise ValueError("provider limit id must not be empty")
            normalized[provider] = _positive_integer(
                raw_limit, "provider limit"
            )
        object.__setattr__(self, "provider_limits", normalized)


@dataclass(frozen=True)
class ProviderBatchResult(Mapping[str, Any]):
    """Member-local result, even when execution was shared."""

    request_id: str
    status: ProviderBatchStatus
    output: Any = None
    error: str = ""
    batch_id: str = ""
    provider_id: str = ""
    execution_id: str = ""
    receipt_id: str = ""
    token_budget: int = 0
    timeout_ms: int = 0
    queue_wait_ms: int = 0
    execution_ms: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    singleflight_shared: bool = False

    @property
    def successful(self) -> bool:
        return self.status.successful

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "output": _canonical(self.output),
            "error": self.error,
            "batch_id": self.batch_id,
            "provider_id": self.provider_id,
            "execution_id": self.execution_id,
            "receipt_id": self.receipt_id,
            "token_budget": self.token_budget,
            "timeout_ms": self.timeout_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "execution_ms": self.execution_ms,
            "provenance": _canonical(self.provenance),
            "singleflight_shared": self.singleflight_shared,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderBatchResult":
        if not isinstance(value, Mapping):
            raise TypeError("provider batch result must be a mapping")
        allowed = {
            "request_id",
            "status",
            "output",
            "error",
            "batch_id",
            "provider_id",
            "execution_id",
            "receipt_id",
            "token_budget",
            "timeout_ms",
            "queue_wait_ms",
            "execution_ms",
            "provenance",
            "singleflight_shared",
        }
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(
                "unknown provider batch result fields: " + ", ".join(unknown)
            )
        return cls(
            request_id=str(value.get("request_id") or ""),
            status=ProviderBatchStatus(str(value.get("status") or "")),
            output=value.get("output"),
            error=str(value.get("error") or ""),
            batch_id=str(value.get("batch_id") or ""),
            provider_id=str(value.get("provider_id") or ""),
            execution_id=str(value.get("execution_id") or ""),
            receipt_id=str(value.get("receipt_id") or ""),
            token_budget=int(value.get("token_budget") or 0),
            timeout_ms=int(value.get("timeout_ms") or 0),
            queue_wait_ms=int(value.get("queue_wait_ms") or 0),
            execution_ms=int(value.get("execution_ms") or 0),
            provenance=(
                value.get("provenance")
                if isinstance(value.get("provenance"), Mapping)
                else {}
            ),
            singleflight_shared=bool(value.get("singleflight_shared", False)),
        )


@dataclass
class ProviderBatchAdmissionGrant:
    """One pre-dispatch admission decision with an optional resource lease.

    ``release`` is invoked exactly once after the physical provider call (or
    after a launch failure).  This is deliberately opaque: callers can bind a
    :class:`resource_scheduler.ResourceAdmissionLease`, a provider semaphore,
    or another reclaimable capacity grant without coupling this module to its
    implementation.
    """

    admitted: bool
    release: Callable[[], Any] | None = field(default=None, repr=False)
    reason: str = ""
    lease: Any = field(default=None, repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _release_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @classmethod
    def from_value(cls, value: Any) -> "ProviderBatchAdmissionGrant":
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            decision, lease = value
            return cls(
                admitted=bool(getattr(decision, "admitted", decision)),
                reason=str(getattr(decision, "reason", "") or ""),
                lease=lease,
            )
        return cls(
            admitted=bool(getattr(value, "admitted", value)),
            reason=str(getattr(value, "reason", "") or ""),
            lease=getattr(value, "lease", None),
        )

    def release_once(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        if self.release is not None:
            self.release()


@dataclass(frozen=True)
class ProviderBatchMemberEvidence:
    """Immutable member projection included in a batch evidence receipt."""

    request_id: str
    execution_id: str
    status: ProviderBatchStatus
    token_budget: int
    timeout_ms: int
    provenance_digest: str
    result_digest: str
    singleflight_shared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "token_budget": self.token_budget,
            "timeout_ms": self.timeout_ms,
            "provenance_digest": self.provenance_digest,
            "result_digest": self.result_digest,
            "singleflight_shared": self.singleflight_shared,
        }


@dataclass(frozen=True)
class ProviderBatchEvidenceReceipt:
    """Content-addressed proof of one batch's independent member outcomes."""

    batch_id: str
    provider_id: str
    compatibility_digest: str
    started_at_ms: int
    completed_at_ms: int
    members: tuple[ProviderBatchMemberEvidence, ...]
    content_digest: str = ""
    schema: str = PROVIDER_BATCH_RECEIPT_SCHEMA
    _producer_seal: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        members = tuple(sorted(self.members, key=lambda item: item.request_id))
        if not self.batch_id or not self.provider_id:
            raise ValueError("batch_id and provider_id must not be empty")
        if len({item.request_id for item in members}) != len(members):
            raise ValueError("batch receipt request ids must be unique")
        if self.completed_at_ms < self.started_at_ms:
            raise ValueError("batch receipt completion precedes start")
        object.__setattr__(self, "members", members)
        expected = _digest(self._unsigned_dict())
        if self.content_digest and self.content_digest != expected:
            raise ValueError("provider batch receipt content digest mismatch")
        object.__setattr__(self, "content_digest", expected)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "batch_id": self.batch_id,
            "provider_id": self.provider_id,
            "compatibility_digest": self.compatibility_digest,
            "started_at_ms": self.started_at_ms,
            "completed_at_ms": self.completed_at_ms,
            "members": [item.to_dict() for item in self.members],
        }

    @property
    def evidence_id(self) -> str:
        return f"sha256:{self.content_digest}"

    @property
    def proves_partial_cancellation(self) -> bool:
        statuses = {item.status for item in self.members}
        return (
            ProviderBatchStatus.CANCELLED in statuses
            and bool(
                statuses
                & {ProviderBatchStatus.SUCCEEDED, ProviderBatchStatus.FALLBACK}
            )
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        if (
            self._producer_seal is _PROVIDER_BATCH_RECEIPT_SEAL
            and self.verify_integrity()
            and self.proves_partial_cancellation
        ):
            return (PARTIAL_CANCELLATION_REQUIREMENT_ID,)
        return ()

    def verify_integrity(self) -> bool:
        return (
            self.schema == PROVIDER_BATCH_RECEIPT_SCHEMA
            and self.content_digest == _digest(self._unsigned_dict())
            and len({item.request_id for item in self.members}) == len(self.members)
            and self.completed_at_ms >= self.started_at_ms
        )

    def to_dict(self) -> dict[str, Any]:
        result = self._unsigned_dict()
        result.update(
            {
                "content_digest": self.content_digest,
                "evidence_id": self.evidence_id,
                "proved_requirement_ids": list(self.proved_requirement_ids),
            }
        )
        return result


@dataclass(frozen=True)
class ProviderBatchMetrics:
    """Separately benchmarkable snapshot of the provider-batching lane."""

    submitted_requests: int
    completed_requests: int
    succeeded_requests: int
    failed_requests: int
    cancelled_requests: int
    timed_out_requests: int
    queued_requests: int
    active_batches: int
    provider_calls: int
    physical_executions: int
    duplicate_executions: int
    completed_batches: int
    batched_executions: int
    singleflight_hits: int
    provider_calls_avoided: int
    fallback_requests: int
    admission_deferrals: int
    capacity_errors: int
    admission_errors: int
    max_queue_depth: int
    max_observed_batch_size: int
    peak_active_batches: int
    total_queue_wait_ms: int
    total_execution_ms: int
    elapsed_ms: int
    adaptive_batch_sizes: Mapping[str, int]
    provider_calls_by_id: Mapping[str, int]
    schema: str = PROVIDER_BATCH_METRICS_SCHEMA

    @property
    def completion_throughput_millionths_per_second(self) -> int:
        if self.elapsed_ms <= 0:
            return 0
        return self.completed_requests * 1_000_000_000 // self.elapsed_ms

    @property
    def average_members_per_call_millionths(self) -> int:
        if self.provider_calls <= 0:
            return 0
        return self.completed_requests * 1_000_000 // self.provider_calls

    @property
    def duplicate_compute_percent_millionths(self) -> int:
        if self.physical_executions <= 0:
            return 0
        return (
            self.duplicate_executions
            * 100_000_000
            // self.physical_executions
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            name: getattr(self, name)
            for name in (
                "submitted_requests",
                "completed_requests",
                "succeeded_requests",
                "failed_requests",
                "cancelled_requests",
                "timed_out_requests",
                "queued_requests",
                "active_batches",
                "provider_calls",
                "physical_executions",
                "duplicate_executions",
                "completed_batches",
                "batched_executions",
                "singleflight_hits",
                "provider_calls_avoided",
                "fallback_requests",
                "admission_deferrals",
                "capacity_errors",
                "admission_errors",
                "max_queue_depth",
                "max_observed_batch_size",
                "peak_active_batches",
                "total_queue_wait_ms",
                "total_execution_ms",
                "elapsed_ms",
            )
        }
        result.update(
            {
                "schema": self.schema,
                "adaptive_batch_sizes": dict(self.adaptive_batch_sizes),
                "provider_calls_by_id": dict(self.provider_calls_by_id),
                "completion_throughput_millionths_per_second": (
                    self.completion_throughput_millionths_per_second
                ),
                "average_members_per_call_millionths": (
                    self.average_members_per_call_millionths
                ),
                "duplicate_compute_percent_millionths": (
                    self.duplicate_compute_percent_millionths
                ),
            }
        )
        return result


@dataclass
class _Subscriber:
    request: ProviderBatchRequest
    future: Future[ProviderBatchResult]
    submitted_at_ms: int
    deadline_ms: int


@dataclass
class _ExecutionGroup:
    fingerprint: str
    key: ProviderBatchKey
    representative: ProviderBatchRequest
    execution_id: str
    subscribers: list[_Subscriber]
    queued_at_ms: int
    running: bool = False
    accepting_subscribers: bool = True
    completed: bool = False


ProviderBatchDispatch = Callable[
    [Sequence[ProviderBatchRequest]], Sequence[Any] | Mapping[str, Any] | Any
]
ProviderMemberFallback = Callable[[ProviderBatchRequest], Any]
ProviderCapacitySupplier = Callable[[str], ProviderBatchCapacity | Mapping[str, Any] | Any]
ProviderAdmission = Callable[
    [ProviderBatchKey, Sequence[ProviderBatchRequest], ProviderBatchCapacity], Any
]


class ResourceSchedulerBatchAdmission:
    """Adapt the supervisor resource scheduler to physical provider batches.

    One lease represents one provider call, not every logical subscriber.
    Token demand is summed across the unique execution groups while context and
    model-memory requirements are maxima/shared load costs.  The adapter is
    intentionally callable so it can be passed directly as ``admission=``.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        host_supplier: Any = None,
        provider_supplier: Any = None,
        budget: Any = None,
        path: Any = ".",
        resource_class: str = "llm-proof-draft",
        memory_bytes: int = 0,
        gpu_memory_bytes: int = 0,
        required_capabilities: Sequence[str] = (),
    ) -> None:
        acquire = getattr(scheduler, "acquire", None)
        release = getattr(scheduler, "release", None)
        if not callable(acquire) or not callable(release):
            raise TypeError("resource scheduler must provide acquire and release")
        self.scheduler = scheduler
        self.host_supplier = host_supplier
        self.provider_supplier = provider_supplier
        self.budget = budget
        self.path = path
        self.resource_class = str(resource_class or "llm-proof-draft")
        self.memory_bytes = _positive_integer(
            memory_bytes, "memory_bytes", allow_zero=True
        )
        self.gpu_memory_bytes = _positive_integer(
            gpu_memory_bytes, "gpu_memory_bytes", allow_zero=True
        )
        self.required_capabilities = tuple(
            str(item).strip() for item in required_capabilities if str(item).strip()
        )

    @staticmethod
    def _sample(source: Any, provider_id: str = "") -> Any:
        if not callable(source):
            return source
        try:
            return source(provider_id) if provider_id else source()
        except TypeError:
            return source()

    def __call__(
        self,
        key: ProviderBatchKey,
        requests: Sequence[ProviderBatchRequest],
        _capacity: ProviderBatchCapacity,
    ) -> ProviderBatchAdmissionGrant:
        from .resource_scheduler import LaneResourceRequirements

        requirement = LaneResourceRequirements(
            lane_id=f"provider-batch:{uuid.uuid4().hex}",
            stage="inference",
            resource_class=self.resource_class,
            required_capabilities=self.required_capabilities,
            provider_id=key.provider_id,
            requires_provider=True,
            context_tokens=max(
                (request.context_limit for request in requests), default=0
            ),
            token_budget=sum(request.token_budget for request in requests),
            quota_units=1,
            memory_bytes=self.memory_bytes,
            gpu_memory_bytes=self.gpu_memory_bytes,
            process_slots=1,
            fairness_key=key.provider_id,
        )
        decision, lease = self.scheduler.acquire(
            requirement,
            budget=self.budget,
            host=self._sample(self.host_supplier),
            providers=self._sample(self.provider_supplier, key.provider_id),
            path=self.path,
        )
        if lease is None:
            return ProviderBatchAdmissionGrant(
                admitted=False,
                reason=str(getattr(decision, "reason", "") or ""),
            )
        return ProviderBatchAdmissionGrant(
            admitted=bool(getattr(decision, "admitted", False)),
            reason=str(getattr(decision, "reason", "") or ""),
            lease=lease,
            release=lambda: self.scheduler.release(
                lease, reason="provider_batch_completed"
            ),
        )


class ProviderBatchScheduler:
    """Fair shared provider queue with adaptive compatible batching."""

    def __init__(
        self,
        dispatch: ProviderBatchDispatch | None = None,
        *,
        providers: Mapping[str, ProviderBatchDispatch] | None = None,
        config: ProviderBatchSchedulerConfig | None = None,
        capacity_supplier: ProviderCapacitySupplier | None = None,
        admission: ProviderAdmission | None = None,
        fallback: ProviderMemberFallback | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if dispatch is None and not providers:
            raise ValueError("a default dispatch or provider dispatch is required")
        self.config = config or ProviderBatchSchedulerConfig()
        self._default_dispatch = dispatch
        self._providers = dict(providers or {})
        self._capacity_supplier = capacity_supplier
        self._admission = admission
        self._fallback = fallback
        self._clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._condition = threading.Condition(threading.RLock())
        self._queues: "OrderedDict[ProviderBatchKey, deque[_ExecutionGroup]]" = (
            OrderedDict()
        )
        self._singleflight: dict[str, _ExecutionGroup] = {}
        self._running_fingerprints: set[str] = set()
        self._request_subscribers: dict[str, _Subscriber] = {}
        self._active_by_provider: dict[str, int] = {}
        self._active_batches = 0
        self._closed = False
        self._started_at_ms = self._clock_ms()
        self._receipts: deque[ProviderBatchEvidenceReceipt] = deque(
            maxlen=self.config.receipt_history
        )
        self._adaptive_sizes: dict[str, int] = {}
        self._provider_calls_by_id: dict[str, int] = {}
        self._counters: dict[str, int] = {
            "submitted_requests": 0,
            "completed_requests": 0,
            "succeeded_requests": 0,
            "failed_requests": 0,
            "cancelled_requests": 0,
            "timed_out_requests": 0,
            "provider_calls": 0,
            "physical_executions": 0,
            "duplicate_executions": 0,
            "completed_batches": 0,
            "batched_executions": 0,
            "singleflight_hits": 0,
            "provider_calls_avoided": 0,
            "fallback_requests": 0,
            "admission_deferrals": 0,
            "capacity_errors": 0,
            "admission_errors": 0,
            "max_queue_depth": 0,
            "max_observed_batch_size": 0,
            "peak_active_batches": 0,
            "total_queue_wait_ms": 0,
            "total_execution_ms": 0,
        }
        self._worker = threading.Thread(
            target=self._coordinator,
            name="provider-batch-scheduler",
            daemon=True,
        )
        self._worker.start()

    def __enter__(self) -> "ProviderBatchScheduler":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.shutdown(wait=True)

    def register_provider(
        self, provider_id: str, dispatch: ProviderBatchDispatch
    ) -> None:
        provider = str(provider_id).strip()
        if not provider or not callable(dispatch):
            raise ValueError("provider id and callable dispatch are required")
        with self._condition:
            if self._closed:
                raise RuntimeError("provider batch scheduler is closed")
            self._providers[provider] = dispatch

    def submit(
        self, request: ProviderBatchRequest | Mapping[str, Any]
    ) -> Future[ProviderBatchResult]:
        normalized = ProviderBatchRequest.from_value(request)
        now = self._clock_ms()
        future: Future[ProviderBatchResult] = Future()
        deadline = now + normalized.timeout_ms if normalized.timeout_ms else 0
        subscriber = _Subscriber(normalized, future, now, deadline)
        with self._condition:
            if self._closed:
                raise RuntimeError("provider batch scheduler is closed")
            if normalized.request_id in self._request_subscribers:
                raise ValueError(
                    f"request_id is already active: {normalized.request_id}"
                )
            if len(self._request_subscribers) >= self.config.max_queue_size:
                raise RuntimeError("provider batch queue is full")
            fingerprint = normalized.execution_fingerprint
            group = self._singleflight.get(fingerprint)
            if group is not None and group.accepting_subscribers:
                group.subscribers.append(subscriber)
                self._counters["singleflight_hits"] += 1
                self._counters["provider_calls_avoided"] += 1
            else:
                group = _ExecutionGroup(
                    fingerprint=fingerprint,
                    key=normalized.batch_key,
                    representative=normalized,
                    execution_id=f"execution:{uuid.uuid4().hex}",
                    subscribers=[subscriber],
                    queued_at_ms=now,
                )
                self._singleflight[fingerprint] = group
                self._queues.setdefault(group.key, deque()).append(group)
            self._request_subscribers[normalized.request_id] = subscriber
            self._counters["submitted_requests"] += 1
            self._counters["max_queue_depth"] = max(
                self._counters["max_queue_depth"], len(self._request_subscribers)
            )
            self._condition.notify_all()
        return future

    def execute(
        self,
        request: ProviderBatchRequest | Mapping[str, Any],
        *,
        wait_timeout: float | None = None,
    ) -> ProviderBatchResult:
        return self.submit(request).result(timeout=wait_timeout)

    def execute_many(
        self,
        requests: Sequence[ProviderBatchRequest | Mapping[str, Any]],
        *,
        wait_timeout: float | None = None,
    ) -> tuple[ProviderBatchResult, ...]:
        futures = [self.submit(item) for item in requests]
        return tuple(future.result(timeout=wait_timeout) for future in futures)

    # Familiar spellings for callers which treat this as a provider facade.
    run = execute_many
    dispatch = execute_many

    def cancel(self, request_id: str) -> bool:
        """Cancel one subscriber without forwarding cancellation to siblings."""

        with self._condition:
            subscriber = self._request_subscribers.get(str(request_id))
            if subscriber is None or subscriber.future.done():
                return False
            cancelled = subscriber.future.cancel()
            self._condition.notify_all()
            return cancelled

    def flush(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._request_subscribers or self._active_batches:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                else:
                    remaining = None
                self._condition.wait(remaining)
            return True

    def shutdown(self, wait: bool = True, *, cancel_pending: bool = False) -> None:
        with self._condition:
            if cancel_pending:
                for subscriber in self._request_subscribers.values():
                    subscriber.future.cancel()
            self._closed = True
            self._condition.notify_all()
        if wait and threading.current_thread() is not self._worker:
            self._worker.join()

    close = shutdown

    def metrics(self) -> ProviderBatchMetrics:
        with self._condition:
            counters = dict(self._counters)
            completed = (
                counters["succeeded_requests"]
                + counters["failed_requests"]
                + counters["cancelled_requests"]
                + counters["timed_out_requests"]
            )
            queued = sum(
                1
                for queue in self._queues.values()
                for group in queue
                for subscriber in group.subscribers
                if subscriber.request.request_id in self._request_subscribers
            )
            return ProviderBatchMetrics(
                submitted_requests=counters["submitted_requests"],
                completed_requests=completed,
                succeeded_requests=counters["succeeded_requests"],
                failed_requests=counters["failed_requests"],
                cancelled_requests=counters["cancelled_requests"],
                timed_out_requests=counters["timed_out_requests"],
                queued_requests=queued,
                active_batches=self._active_batches,
                provider_calls=counters["provider_calls"],
                physical_executions=counters["physical_executions"],
                duplicate_executions=counters["duplicate_executions"],
                completed_batches=counters["completed_batches"],
                batched_executions=counters["batched_executions"],
                singleflight_hits=counters["singleflight_hits"],
                provider_calls_avoided=counters["provider_calls_avoided"],
                fallback_requests=counters["fallback_requests"],
                admission_deferrals=counters["admission_deferrals"],
                capacity_errors=counters["capacity_errors"],
                admission_errors=counters["admission_errors"],
                max_queue_depth=counters["max_queue_depth"],
                max_observed_batch_size=counters["max_observed_batch_size"],
                peak_active_batches=counters["peak_active_batches"],
                total_queue_wait_ms=counters["total_queue_wait_ms"],
                total_execution_ms=counters["total_execution_ms"],
                elapsed_ms=max(0, self._clock_ms() - self._started_at_ms),
                adaptive_batch_sizes=dict(sorted(self._adaptive_sizes.items())),
                provider_calls_by_id=dict(
                    sorted(self._provider_calls_by_id.items())
                ),
            )

    snapshot = metrics
    benchmark_snapshot = metrics

    def evidence_receipts(self) -> tuple[ProviderBatchEvidenceReceipt, ...]:
        with self._condition:
            return tuple(self._receipts)

    def partial_cancellation_evidence(
        self,
    ) -> tuple[ProviderBatchEvidenceReceipt, ...]:
        return tuple(
            receipt
            for receipt in self.evidence_receipts()
            if receipt.proved_requirement_ids
            == (PARTIAL_CANCELLATION_REQUIREMENT_ID,)
        )

    def _capacity(self, provider_id: str) -> ProviderBatchCapacity:
        if self._capacity_supplier is None:
            limit = self.config.provider_limits.get(
                provider_id, self.config.max_parallel_batches
            )
            return ProviderBatchCapacity(
                provider_id=provider_id,
                max_concurrent_batches=limit,
                available_concurrent_batches=max(
                    0, limit - self._active_by_provider.get(provider_id, 0)
                ),
            )
        return ProviderBatchCapacity.from_value(
            provider_id, self._capacity_supplier(provider_id)
        )

    def _effective_size(
        self, provider_id: str, capacity: ProviderBatchCapacity
    ) -> int:
        # The current remote Abby adapters expose one-request HTTP APIs.  Keep
        # physical calls at one member until an adapter explicitly implements
        # a real batch wire contract.  Identical logical subscribers can still
        # share that member through single-flight.
        if _requires_single_member_batch(provider_id):
            self._adaptive_sizes[provider_id] = 1
            return 1
        adaptive = self._adaptive_sizes.setdefault(
            provider_id, self.config.max_batch_size
        )
        limits = [self.config.max_batch_size, adaptive]
        if capacity.max_batch_size:
            limits.append(capacity.max_batch_size)
        return max(self.config.min_batch_size, min(limits))

    def _provider_has_slot(
        self, provider_id: str, capacity: ProviderBatchCapacity
    ) -> bool:
        configured = self.config.provider_limits.get(
            provider_id, self.config.max_parallel_batches
        )
        active = self._active_by_provider.get(provider_id, 0)
        live_limit = capacity.max_concurrent_batches or configured
        return (
            capacity.healthy
            and capacity.retry_after_ms == 0
            and capacity.available_concurrent_batches > 0
            and active < min(configured, live_limit)
        )

    def _release_admission(
        self, grant: ProviderBatchAdmissionGrant
    ) -> None:
        """Release a provider lease without destabilizing coordination."""

        try:
            grant.release_once()
        except Exception:
            # A faulty external lease manager is operationally significant,
            # but must not terminate the sole coordinator or obscure member
            # results which have already been materialized.
            with self._condition:
                self._counters["admission_errors"] += 1

    def _expire_subscribers(self, now: int) -> None:
        for subscriber in tuple(self._request_subscribers.values()):
            if subscriber.future.done():
                status = ProviderBatchStatus.CANCELLED
            elif _cancelled(subscriber.request.cancellation_token):
                status = ProviderBatchStatus.CANCELLED
            elif subscriber.deadline_ms and now >= subscriber.deadline_ms:
                status = ProviderBatchStatus.TIMED_OUT
            else:
                continue
            if not subscriber.future.done():
                self._set_member_result(
                    subscriber,
                    ProviderBatchResult(
                        request_id=subscriber.request.request_id,
                        status=status,
                        error=status.value,
                        provider_id=subscriber.request.provider_id,
                        token_budget=subscriber.request.token_budget,
                        timeout_ms=subscriber.request.timeout_ms,
                        queue_wait_ms=max(0, now - subscriber.submitted_at_ms),
                        provenance=subscriber.request.provenance,
                    ),
                )
            else:
                self._record_done(subscriber, status)

    def _remove_empty_groups(self) -> None:
        for key, queue in tuple(self._queues.items()):
            retained: deque[_ExecutionGroup] = deque()
            for group in queue:
                group.subscribers[:] = [
                    item
                    for item in group.subscribers
                    if item.request.request_id in self._request_subscribers
                ]
                if group.subscribers:
                    retained.append(group)
                else:
                    self._singleflight.pop(group.fingerprint, None)
            if retained:
                self._queues[key] = retained
            else:
                self._queues.pop(key, None)

    def _coordinator(self) -> None:
        while True:
            with self._condition:
                now = self._clock_ms()
                self._expire_subscribers(now)
                self._remove_empty_groups()
                if (
                    self._closed
                    and not self._request_subscribers
                    and self._active_batches == 0
                ):
                    return
                launched = self._launch_ready(now)
                if launched:
                    continue
                self._condition.wait(self.config.admission_retry_ms / 1_000)

    def _launch_ready(self, now: int) -> bool:
        if self._active_batches >= self.config.max_parallel_batches:
            return False
        # OrderedDict rotation gives compatibility classes round-robin fairness.
        for _ in range(len(self._queues)):
            key, queue = self._queues.popitem(last=False)
            self._queues[key] = queue
            if not queue:
                continue
            try:
                capacity = self._capacity(key.provider_id)
            except Exception:
                # Telemetry is live operational input.  A transient monitor
                # failure applies backpressure; it must not kill the one
                # coordinator responsible for every provider route.
                self._counters["capacity_errors"] += 1
                self._counters["admission_deferrals"] += 1
                continue
            if not self._provider_has_slot(key.provider_id, capacity):
                self._counters["admission_deferrals"] += 1
                continue
            max_size = self._effective_size(key.provider_id, capacity)
            oldest = queue[0].queued_at_ms
            if len(queue) < max_size and now - oldest < self.config.batch_window_ms:
                continue
            groups: list[_ExecutionGroup] = []
            token_total = 0
            while queue and len(groups) < max_size:
                candidate = queue[0]
                candidate_budget = candidate.representative.token_budget
                if (
                    capacity.token_budget_remaining >= 0
                    and groups
                    and token_total + candidate_budget
                    > capacity.token_budget_remaining
                ):
                    break
                if (
                    capacity.token_budget_remaining >= 0
                    and not groups
                    and candidate_budget > capacity.token_budget_remaining
                ):
                    break
                groups.append(queue.popleft())
                token_total += candidate_budget
            if not groups:
                self._counters["admission_deferrals"] += 1
                continue
            requests = tuple(group.representative.dispatch_copy() for group in groups)
            admission_grant = ProviderBatchAdmissionGrant(admitted=True)
            if self._admission is not None:
                try:
                    admission_grant = ProviderBatchAdmissionGrant.from_value(
                        self._admission(key, requests, capacity)
                    )
                except Exception:
                    self._counters["admission_errors"] += 1
                    admission_grant = ProviderBatchAdmissionGrant(
                        admitted=False,
                        reason="admission_error",
                    )
                if not admission_grant.admitted:
                    self._release_admission(admission_grant)
                    for group in reversed(groups):
                        queue.appendleft(group)
                    self._counters["admission_deferrals"] += 1
                    continue
            if not queue:
                self._queues.pop(key, None)
            for group in groups:
                if group.fingerprint in self._running_fingerprints:
                    # This should be unreachable because running groups retain
                    # their single-flight handle.  Keep the metric fail-visible
                    # if an extension violates that invariant.
                    self._counters["duplicate_executions"] += 1
                self._running_fingerprints.add(group.fingerprint)
                group.running = True
            self._active_batches += 1
            self._counters["peak_active_batches"] = max(
                self._counters["peak_active_batches"],
                self._active_batches,
            )
            self._active_by_provider[key.provider_id] = (
                self._active_by_provider.get(key.provider_id, 0) + 1
            )
            batch_id = f"provider-batch:{uuid.uuid4().hex}"
            thread = threading.Thread(
                target=self._run_batch,
                args=(
                    batch_id,
                    key,
                    groups,
                    requests,
                    now,
                    admission_grant,
                ),
                name=f"provider-batch-{key.provider_id}",
                daemon=True,
            )
            try:
                thread.start()
            except Exception:
                self._release_admission(admission_grant)
                for group in reversed(groups):
                    self._running_fingerprints.discard(group.fingerprint)
                    group.running = False
                    queue.appendleft(group)
                self._queues.setdefault(key, queue)
                self._active_batches -= 1
                self._active_by_provider[key.provider_id] -= 1
                self._counters["admission_errors"] += 1
                continue
            return True
        return False

    def _resolve_dispatch(self, provider_id: str) -> ProviderBatchDispatch:
        dispatch = self._providers.get(provider_id, self._default_dispatch)
        if dispatch is None:
            raise RuntimeError(f"no dispatch registered for provider {provider_id}")
        return dispatch

    def _normalize_outputs(
        self, requests: Sequence[ProviderBatchRequest], raw: Any
    ) -> tuple[Any, ...]:
        if isinstance(raw, Mapping):
            missing = [
                request.request_id
                for request in requests
                if request.request_id not in raw
            ]
            if missing:
                raise ValueError(
                    "provider omitted batch members: " + ", ".join(missing)
                )
            return tuple(raw[request.request_id] for request in requests)
        if isinstance(raw, Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            if len(raw) != len(requests):
                raise ValueError(
                    "provider result count does not match batch request count"
                )
            return tuple(raw)
        if len(requests) == 1:
            return (raw,)
        raise ValueError("batch provider must return a mapping or sequence")

    def _run_batch(
        self,
        batch_id: str,
        key: ProviderBatchKey,
        groups: list[_ExecutionGroup],
        requests: tuple[ProviderBatchRequest, ...],
        started_at_ms: int,
        admission_grant: ProviderBatchAdmissionGrant,
    ) -> None:
        try:
            self._run_admitted_batch(
                batch_id,
                key,
                groups,
                requests,
                started_at_ms,
            )
        except BaseException as exc:
            # The normal path materializes provider errors per member.  This
            # guard covers scheduler-internal failures (for example a faulty
            # receipt extension) so subscribers and capacity cannot leak.
            now = self._clock_ms()
            with self._condition:
                incomplete = [group for group in groups if not group.completed]
                for group in incomplete:
                    self._running_fingerprints.discard(group.fingerprint)
                    group.accepting_subscribers = False
                    if self._singleflight.get(group.fingerprint) is group:
                        self._singleflight.pop(group.fingerprint, None)
                    for subscriber in tuple(group.subscribers):
                        if (
                            subscriber.request.request_id
                            not in self._request_subscribers
                        ):
                            continue
                        self._set_member_result(
                            subscriber,
                            ProviderBatchResult(
                                request_id=subscriber.request.request_id,
                                status=ProviderBatchStatus.FAILED,
                                error=(
                                    "provider batch scheduler internal failure: "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                                batch_id=batch_id,
                                provider_id=key.provider_id,
                                execution_id=group.execution_id,
                                token_budget=subscriber.request.token_budget,
                                timeout_ms=subscriber.request.timeout_ms,
                                queue_wait_ms=max(
                                    0, now - subscriber.submitted_at_ms
                                ),
                                provenance=subscriber.request.provenance,
                                singleflight_shared=(
                                    len(group.subscribers) > 1
                                ),
                            ),
                        )
                    group.completed = True
                if incomplete:
                    self._active_batches = max(0, self._active_batches - 1)
                    self._active_by_provider[key.provider_id] = max(
                        0,
                        self._active_by_provider.get(key.provider_id, 0) - 1,
                    )
                    self._counters["completed_batches"] += 1
                    self._condition.notify_all()
        finally:
            # A provider exception, malformed batch result, cancellation, or
            # receipt failure must never leak GPU/provider capacity.
            self._release_admission(admission_grant)

    def _run_admitted_batch(
        self,
        batch_id: str,
        key: ProviderBatchKey,
        groups: list[_ExecutionGroup],
        requests: tuple[ProviderBatchRequest, ...],
        _selected_at_ms: int,
    ) -> None:
        outputs: tuple[Any, ...]
        dispatch_error: BaseException | None = None
        call_started = self._clock_ms()
        try:
            with self._condition:
                self._counters["provider_calls"] += 1
                self._counters["physical_executions"] += len(groups)
                self._provider_calls_by_id[key.provider_id] = (
                    self._provider_calls_by_id.get(key.provider_id, 0) + 1
                )
                self._counters["max_observed_batch_size"] = max(
                    self._counters["max_observed_batch_size"], len(groups)
                )
                if len(groups) > 1:
                    self._counters["batched_executions"] += len(groups)
                    self._counters["provider_calls_avoided"] += len(groups) - 1
            raw = self._resolve_dispatch(key.provider_id)(requests)
            outputs = self._normalize_outputs(requests, raw)
        except BaseException as exc:  # provider boundary; materialize per member
            dispatch_error = exc
            outputs = tuple(exc for _ in requests)
        completed_at_ms = self._clock_ms()
        duration = max(0, completed_at_ms - call_started)
        # Seal the groups and remove their single-flight handles before taking
        # a subscriber snapshot.  A late identical submit now creates a fresh
        # queued group; every subscriber which joined the completed execution
        # is guaranteed to appear in this immutable snapshot.
        with self._condition:
            subscriber_groups: list[tuple[_Subscriber, ...]] = []
            for group in groups:
                group.accepting_subscribers = False
                if self._singleflight.get(group.fingerprint) is group:
                    self._singleflight.pop(group.fingerprint, None)
                subscriber_groups.append(tuple(group.subscribers))
        result_entries: list[tuple[_Subscriber, ProviderBatchResult]] = []
        for group, subscribers, output in zip(
            groups, subscriber_groups, outputs
        ):
            used_fallback = False
            if (
                isinstance(output, BaseException)
                and self._fallback is not None
                and self.config.fallback_on_dispatch_error
            ):
                try:
                    output = self._fallback(group.representative)
                    used_fallback = True
                except BaseException as fallback_error:
                    output = fallback_error
            for subscriber in subscribers:
                now = self._clock_ms()
                if (
                    subscriber.future.cancelled()
                    or _cancelled(subscriber.request.cancellation_token)
                ):
                    status = ProviderBatchStatus.CANCELLED
                    value = None
                    error = "cancelled"
                elif subscriber.deadline_ms and now >= subscriber.deadline_ms:
                    status = ProviderBatchStatus.TIMED_OUT
                    value = None
                    error = "timed_out"
                elif isinstance(output, BaseException):
                    status = ProviderBatchStatus.FAILED
                    value = None
                    error = f"{type(output).__name__}: {output}"
                else:
                    status = (
                        ProviderBatchStatus.FALLBACK
                        if used_fallback
                        else ProviderBatchStatus.SUCCEEDED
                    )
                    value = output
                    error = ""
                result_entries.append(
                    (
                        subscriber,
                        ProviderBatchResult(
                            request_id=subscriber.request.request_id,
                            status=status,
                            output=value,
                            error=error,
                            batch_id=batch_id,
                            provider_id=key.provider_id,
                            execution_id=group.execution_id,
                            token_budget=subscriber.request.token_budget,
                            timeout_ms=subscriber.request.timeout_ms,
                            queue_wait_ms=max(
                                0, call_started - subscriber.submitted_at_ms
                            ),
                            execution_ms=duration,
                            provenance=subscriber.request.provenance,
                            singleflight_shared=len(group.subscribers) > 1,
                        ),
                    )
                )
        receipt = self._build_receipt(
            batch_id, key, result_entries, call_started, completed_at_ms
        )
        with self._condition:
            for subscriber, result in result_entries:
                self._set_member_result(
                    subscriber, replace(result, receipt_id=receipt.evidence_id)
                )
            for group in groups:
                self._running_fingerprints.discard(group.fingerprint)
                group.completed = True
            self._receipts.append(receipt)
            self._active_batches -= 1
            self._active_by_provider[key.provider_id] -= 1
            self._counters["completed_batches"] += 1
            self._counters["total_execution_ms"] += duration
            current = self._adaptive_sizes.get(
                key.provider_id, self.config.max_batch_size
            )
            if _requires_single_member_batch(key.provider_id):
                current = 1
            elif (
                dispatch_error is not None
                or duration > self.config.target_batch_latency_ms
            ):
                current = max(self.config.min_batch_size, current // 2)
            elif len(groups) >= current and current < self.config.max_batch_size:
                current += 1
            self._adaptive_sizes[key.provider_id] = current
            self._condition.notify_all()

    def _set_member_result(
        self, subscriber: _Subscriber, result: ProviderBatchResult
    ) -> None:
        if not subscriber.future.done():
            subscriber.future.set_result(result)
        self._record_done(subscriber, result.status)
        self._counters["total_queue_wait_ms"] += result.queue_wait_ms

    def _record_done(
        self, subscriber: _Subscriber, status: ProviderBatchStatus
    ) -> None:
        removed = self._request_subscribers.pop(
            subscriber.request.request_id, None
        )
        if removed is None:
            return
        if status.successful:
            self._counters["succeeded_requests"] += 1
            if status is ProviderBatchStatus.FALLBACK:
                self._counters["fallback_requests"] += 1
        elif status is ProviderBatchStatus.CANCELLED:
            self._counters["cancelled_requests"] += 1
        elif status is ProviderBatchStatus.TIMED_OUT:
            self._counters["timed_out_requests"] += 1
        else:
            self._counters["failed_requests"] += 1

    def _build_receipt(
        self,
        batch_id: str,
        key: ProviderBatchKey,
        entries: Sequence[tuple[_Subscriber, ProviderBatchResult]],
        started_at_ms: int,
        completed_at_ms: int,
    ) -> ProviderBatchEvidenceReceipt:
        members = tuple(
            ProviderBatchMemberEvidence(
                request_id=subscriber.request.request_id,
                execution_id=result.execution_id,
                status=result.status,
                token_budget=subscriber.request.token_budget,
                timeout_ms=subscriber.request.timeout_ms,
                provenance_digest=_digest(subscriber.request.provenance),
                result_digest=_digest(
                    {
                        "status": result.status.value,
                        "output": result.output,
                        "error": result.error,
                    }
                ),
                singleflight_shared=result.singleflight_shared,
            )
            for subscriber, result in entries
        )
        return ProviderBatchEvidenceReceipt(
            batch_id=batch_id,
            provider_id=key.provider_id,
            compatibility_digest=key.digest,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            members=members,
            _producer_seal=_PROVIDER_BATCH_RECEIPT_SEAL,
        )


# Compatibility aliases keep the public surface readable for model and
# inference callers while retaining one implementation.
BatchCompatibilityKey = ProviderBatchKey
BatchRequest = ProviderBatchRequest
BatchResult = ProviderBatchResult
BatchMetrics = ProviderBatchMetrics


__all__ = [
    "PARTIAL_CANCELLATION_REQUIREMENT_ID",
    "PROVIDER_BATCH_METRICS_SCHEMA",
    "PROVIDER_BATCH_RECEIPT_SCHEMA",
    "BatchCompatibilityKey",
    "BatchMetrics",
    "BatchRequest",
    "BatchResult",
    "ProviderBatchCapacity",
    "ProviderBatchAdmissionGrant",
    "ProviderBatchEvidenceReceipt",
    "ProviderBatchKey",
    "ProviderBatchMemberEvidence",
    "ProviderBatchMetrics",
    "ProviderBatchRequest",
    "ProviderBatchResult",
    "ProviderBatchScheduler",
    "ProviderBatchSchedulerConfig",
    "ProviderBatchStatus",
    "ResourceSchedulerBatchAdmission",
]
