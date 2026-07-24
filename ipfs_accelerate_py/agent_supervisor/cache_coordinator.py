"""In-process single-flight coordination for the analysis cache.

``AnalysisCache`` owns persistence, integrity checking, expiry, and exact-key
invalidation.  This module adds only execution coordination: concurrent work
for one :class:`~analysis_cache.AnalysisCacheKey` is represented by one shared
future while unrelated keys remain fully parallel.

The cache is always consulted with ``require_completion_evidence=True``.
Consequently a stale, partial, failed, timed-out, inconclusive, or corrupt
entry can be observed in the returned diagnostics but can never bypass the
producer as an authoritative cache hit.  Followers receive a shared execution
result only after the leader has published it, and the result's completion
authority remains derived from a fresh exact-key cache lookup.

Coordination is deliberately in-process.  The future used as the rendezvous
point is safe for both threads and asyncio event loops, so synchronous and
asynchronous callers in the same process also collapse onto one producer.
Cross-process coordination belongs in a durable lease implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import secrets
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Awaitable, Callable, Final, Mapping, TypeVar, Union

from .analysis_cache import (
    AnalysisCache,
    AnalysisCacheEntry,
    AnalysisCacheKey,
    AnalysisCacheLookupResult,
    AnalysisCacheLookupStatus,
    AnalysisCacheStoreResult,
    AnalysisOutcome,
    AnalysisReceipt,
    canonical_analysis_json,
)


SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID: Final = (
    "206259342916458424196977899134352826879"
)
CONCURRENT_IDENTICAL_MISS_COLLAPSE_REQUIREMENT_ID: Final = (
    SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID
)
SINGLE_FLIGHT_COLLAPSE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/single-flight-collapse-evidence@1"
)


T = TypeVar("T")
CompletionValidator = Callable[[AnalysisCacheLookupResult], bool]
ProducerValue = Union[
    Mapping[str, Any],
    AnalysisReceipt,
    AnalysisCacheEntry,
    AnalysisCacheLookupResult,
    AnalysisCacheStoreResult,
]
SyncProducer = Callable[[], Union[ProducerValue, "CachePublication"]]
AsyncProducer = Callable[
    [],
    Union[
        ProducerValue,
        "CachePublication",
        Awaitable[Union[ProducerValue, "CachePublication"]],
    ],
]


class CacheCoordinationError(RuntimeError):
    """Base class for analysis cache coordination failures."""


class CacheCoordinationTimeout(CacheCoordinationError, TimeoutError):
    """A follower did not observe the leader's result within its wait bound."""


class CacheProducerResultError(CacheCoordinationError, TypeError):
    """A producer returned a value that cannot be persisted or validated."""


class CacheCoordinationStatus(str, Enum):
    """How a coordinated analysis result was obtained."""

    CACHE_HIT = "cache_hit"
    PRODUCED = "produced"
    SHARED = "shared"

    # Readable compatibility spellings.
    HIT = "cache_hit"
    LEADER = "produced"
    MISS_PRODUCED = "produced"
    FOLLOWER = "shared"
    COALESCED = "shared"


CoordinatorStatus = CacheCoordinationStatus
CacheCoordinatorStatus = CacheCoordinationStatus


@dataclass(frozen=True)
class CachePublication:
    """Per-production persistence decision used inside one shared flight.

    This lets a producer coalesce expensive work while applying a result-aware
    cache policy: conclusive receipts can be durable, negative receipts can
    have a bounded TTL, and explicitly non-cacheable receipts can still be
    delivered to all in-process followers.  An omitted publication TTL
    inherits the coordination call's TTL; an explicit publication TTL
    overrides it.
    """

    value: Any
    store: bool = True
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, bool):
            raise ValueError("store must be a boolean")
        if self.ttl_seconds is not None and (
            isinstance(self.ttl_seconds, bool)
            or not isinstance(self.ttl_seconds, int)
            or self.ttl_seconds < 1
        ):
            raise ValueError("ttl_seconds must be a positive integer or None")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CacheCoordinationError(f"{name} is required")
    result = value.strip()
    if "\x00" in result:
        raise CacheCoordinationError(f"{name} must not contain NUL bytes")
    return result


@dataclass(frozen=True)
class _SingleFlightAttestation:
    """Opaque coordinator-issued authority retained only on typed results."""

    cache_key_id: str
    flight_id: str
    publication_entry_digest: str
    receipt_id: str
    producer_invocation_count: int
    participant_count: int
    follower_count: int
    seal: object


@dataclass(frozen=True)
class SingleFlightCollapseEvidence:
    """Content-addressed proof that one active keyed miss had one producer.

    The witness is deliberately created from the coordinator's private flight
    state, never from global metrics or caller-supplied participant counts.
    It binds the complete seven-dimension cache key, the active flight, and
    the exact durable publication shared by at least one follower.
    """

    cache_key: AnalysisCacheKey
    flight_id: str
    publication_entry_digest: str
    receipt_id: str
    producer_invocation_count: int
    participant_count: int
    follower_count: int
    requirement_id: str = SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID

    def __post_init__(self) -> None:
        if not isinstance(self.cache_key, AnalysisCacheKey):
            raise CacheCoordinationError(
                "single-flight evidence requires an AnalysisCacheKey"
            )
        for name in ("flight_id", "publication_entry_digest", "receipt_id"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        if self.requirement_id != SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID:
            raise CacheCoordinationError(
                "unexpected single-flight collapse requirement ID"
            )
        for name in (
            "producer_invocation_count",
            "participant_count",
            "follower_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CacheCoordinationError(f"{name} must be an integer")
        if self.producer_invocation_count != 1:
            raise CacheCoordinationError(
                "single-flight evidence requires exactly one producer"
            )
        if self.follower_count < 1:
            raise CacheCoordinationError(
                "single-flight evidence requires at least one follower"
            )
        if self.participant_count != self.follower_count + 1:
            raise CacheCoordinationError(
                "single-flight participant count must equal leader plus followers"
            )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": SINGLE_FLIGHT_COLLAPSE_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "cache_key": self.cache_key.to_dict(),
            "cache_key_id": self.cache_key.key_id,
            "flight_id": self.flight_id,
            "publication_entry_digest": self.publication_entry_digest,
            "receipt_id": self.receipt_id,
            "producer_invocation_count": self.producer_invocation_count,
            "participant_count": self.participant_count,
            "follower_count": self.follower_count,
        }

    @property
    def evidence_id(self) -> str:
        digest = hashlib.sha256(
            canonical_analysis_json(self._content()).encode("utf-8")
        ).hexdigest()
        return f"single-flight-collapse:sha256:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SingleFlightCollapseEvidence":
        if not isinstance(value, Mapping):
            raise CacheCoordinationError(
                "single-flight evidence must be an object"
            )
        allowed = {
            "schema",
            "evidence_id",
            "requirement_id",
            "cache_key",
            "cache_key_id",
            "flight_id",
            "publication_entry_digest",
            "receipt_id",
            "producer_invocation_count",
            "participant_count",
            "follower_count",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise CacheCoordinationError(
                "single-flight evidence has unknown fields: "
                + ", ".join(unknown)
            )
        if value.get("schema") != SINGLE_FLIGHT_COLLAPSE_EVIDENCE_SCHEMA:
            raise CacheCoordinationError(
                "unsupported single-flight evidence schema"
            )
        key_value = value.get("cache_key")
        if not isinstance(key_value, Mapping):
            raise CacheCoordinationError(
                "single-flight evidence cache_key must be an object"
            )
        try:
            cache_key = AnalysisCacheKey.from_dict(key_value)
            restored = cls(
                cache_key=cache_key,
                flight_id=value.get("flight_id", ""),
                publication_entry_digest=value.get(
                    "publication_entry_digest", ""
                ),
                receipt_id=value.get("receipt_id", ""),
                producer_invocation_count=value.get(
                    "producer_invocation_count", 0
                ),
                participant_count=value.get("participant_count", 0),
                follower_count=value.get("follower_count", 0),
                requirement_id=value.get("requirement_id", ""),
            )
        except (CacheCoordinationError, TypeError, ValueError) as exc:
            if isinstance(exc, CacheCoordinationError):
                raise
            raise CacheCoordinationError(
                "malformed single-flight evidence"
            ) from exc
        if value.get("cache_key_id") != cache_key.key_id:
            raise CacheCoordinationError(
                "single-flight cache key identity does not match its content"
            )
        if value.get("evidence_id") != restored.evidence_id:
            raise CacheCoordinationError(
                "single-flight evidence identity does not match its content"
            )
        return restored

    @classmethod
    def from_result(
        cls,
        result: "CacheCoordinationResult",
        *,
        follower_count: int,
        _attestation: _SingleFlightAttestation | None = None,
    ) -> "SingleFlightCollapseEvidence":
        """Create evidence from a sealed leader publication."""

        entry = result.entry
        receipt = result.receipt
        if (
            result.status is not CacheCoordinationStatus.PRODUCED
            or not result.leader
            or result.waited
            or not result.is_completion_evidence
            or entry is None
            or not isinstance(receipt, Mapping)
            or not isinstance(_attestation, _SingleFlightAttestation)
            or _attestation.cache_key_id != result.key.key_id
            or _attestation.flight_id != result.flight_id
            or _attestation.publication_entry_digest != entry.entry_digest
            or _attestation.receipt_id != receipt.get("receipt_id")
            or _attestation.producer_invocation_count
            != result.producer_invocation_count
            or _attestation.follower_count != follower_count
            or _attestation.participant_count != follower_count + 1
        ):
            raise CacheCoordinationError(
                "single-flight evidence requires a coordinator-attested "
                "completed leader publication"
            )
        witness = cls(
            cache_key=result.key,
            flight_id=result.flight_id,
            publication_entry_digest=entry.entry_digest,
            receipt_id=receipt.get("receipt_id", ""),
            producer_invocation_count=result.producer_invocation_count,
            participant_count=follower_count + 1,
            follower_count=follower_count,
        )
        return witness

    def proves_for(
        self,
        active_key: AnalysisCacheKey,
        result: "CacheCoordinationResult",
    ) -> bool:
        """Revalidate the witness against an active key and typed result."""

        if (
            not isinstance(active_key, AnalysisCacheKey)
            or not isinstance(result, CacheCoordinationResult)
            or active_key != self.cache_key
            or result.key != active_key
            or result.status
            not in (
                CacheCoordinationStatus.PRODUCED,
                CacheCoordinationStatus.SHARED,
            )
            or result.flight_id != self.flight_id
            or result.producer_invocation_count
            != self.producer_invocation_count
            or not result.is_completion_evidence
        ):
            return False
        if result.status is CacheCoordinationStatus.PRODUCED:
            if not result.leader or result.waited:
                return False
        elif result.leader or not result.waited:
            return False
        entry = result.entry
        receipt = result.receipt
        attestation = result._single_flight_attestation
        return bool(
            entry is not None
            and entry.key == active_key
            and entry.entry_digest
            and entry.entry_digest == entry.computed_digest
            and entry.entry_digest == self.publication_entry_digest
            and isinstance(receipt, Mapping)
            and receipt.get("receipt_id") == self.receipt_id
            and isinstance(attestation, _SingleFlightAttestation)
            and attestation.cache_key_id == active_key.key_id
            and attestation.flight_id == self.flight_id
            and attestation.publication_entry_digest
            == self.publication_entry_digest
            and attestation.receipt_id == self.receipt_id
            and attestation.producer_invocation_count
            == self.producer_invocation_count
            and attestation.participant_count == self.participant_count
            and attestation.follower_count == self.follower_count
            and type(attestation.seal) is object
        )


@dataclass(frozen=True)
class CacheCoordinationResult:
    """Typed outcome of one coordinated cache operation.

    ``producer_value`` is retained for local callers that need a typed value
    even when it was not completion-eligible.  It is intentionally omitted
    from :meth:`to_dict`: analysis bodies and other large producer values must
    not accidentally cross an audit or scheduler boundary.
    """

    status: CacheCoordinationStatus
    key: AnalysisCacheKey
    lookup: AnalysisCacheLookupResult
    store_result: AnalysisCacheStoreResult | None = None
    producer_value: Any = None
    leader: bool = False
    waited: bool = False
    flight_id: str = ""
    producer_invocation_count: int = 0
    single_flight_collapse_evidence: SingleFlightCollapseEvidence | None = None
    _single_flight_attestation: _SingleFlightAttestation | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.status, CacheCoordinationStatus):
            object.__setattr__(
                self, "status", CacheCoordinationStatus(str(self.status))
            )
        if not isinstance(self.key, AnalysisCacheKey):
            raise CacheCoordinationError(
                "coordination result requires an AnalysisCacheKey"
            )
        if not isinstance(self.lookup, AnalysisCacheLookupResult):
            raise CacheCoordinationError(
                "coordination result requires a typed cache lookup"
            )
        if self.lookup.key != self.key:
            raise CacheCoordinationError(
                "coordination result lookup is bound to another cache key"
            )
        if isinstance(self.producer_invocation_count, bool) or not isinstance(
            self.producer_invocation_count, int
        ):
            raise CacheCoordinationError(
                "producer_invocation_count must be an integer"
            )
        if self.status is CacheCoordinationStatus.CACHE_HIT:
            if (
                self.leader
                or self.waited
                or self.flight_id
                or self.producer_invocation_count
                or self.single_flight_collapse_evidence is not None
                or self._single_flight_attestation is not None
            ):
                raise CacheCoordinationError(
                    "cache hits cannot carry single-flight execution state"
                )
        else:
            _required_text(self.flight_id, "flight_id")
            if self.producer_invocation_count != 1:
                raise CacheCoordinationError(
                    "produced/shared results require one producer invocation"
                )
            if self.status is CacheCoordinationStatus.PRODUCED and (
                not self.leader or self.waited
            ):
                raise CacheCoordinationError(
                    "produced results must carry the leader role"
                )
            if self.status is CacheCoordinationStatus.SHARED and (
                self.leader or not self.waited
            ):
                raise CacheCoordinationError(
                    "shared results must carry the follower role"
                )
        witness = self.single_flight_collapse_evidence
        if (
            self._single_flight_attestation is not None
            and witness is None
        ):
            raise CacheCoordinationError(
                "single-flight attestation cannot be detached from its witness"
            )
        if witness is not None and not witness.proves_for(self.key, self):
            raise CacheCoordinationError(
                "single-flight evidence is detached from coordination result"
            )

    @property
    def cache_hit(self) -> bool:
        return self.status is CacheCoordinationStatus.CACHE_HIT

    @property
    def produced(self) -> bool:
        return self.status is CacheCoordinationStatus.PRODUCED

    @property
    def shared(self) -> bool:
        return self.status is CacheCoordinationStatus.SHARED

    @property
    def coalesced(self) -> bool:
        return self.shared

    @property
    def follower(self) -> bool:
        return self.shared

    @property
    def entry(self) -> AnalysisCacheEntry | None:
        if self.lookup.entry is not None:
            return self.lookup.entry
        if self.store_result is not None:
            return self.store_result.entry
        return None

    @property
    def receipt(self) -> Mapping[str, Any] | None:
        """Return the compact receipt without implying completion authority."""

        entry = self.entry
        return entry.receipt if entry is not None else None

    @property
    def outcome(self) -> AnalysisOutcome | None:
        entry = self.entry
        return entry.status if entry is not None else self.lookup.outcome

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.lookup.reason_codes:
            return self.lookup.reason_codes
        if self.store_result is not None:
            return self.store_result.reason_codes
        return ()

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def is_completion_evidence(self) -> bool:
        """Whether a fresh exact-key lookup grants completion authority."""

        return bool(
            self.lookup.status is AnalysisCacheLookupStatus.HIT
            and self.lookup.key == self.key
            and self.lookup.is_completion_evidence
        )

    @property
    def completion_evidence(self) -> bool:
        return self.is_completion_evidence

    @property
    def authoritative(self) -> bool:
        return self.is_completion_evidence

    def proved_requirement_ids_for(
        self, active_key: AnalysisCacheKey | Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Return operational proof IDs only after active-key rebinding."""

        try:
            key = (
                active_key
                if isinstance(active_key, AnalysisCacheKey)
                else AnalysisCacheKey.from_dict(active_key)
            )
        except (TypeError, ValueError):
            return ()
        witness = self.single_flight_collapse_evidence
        if witness is None or not witness.proves_for(key, self):
            return ()
        return (witness.requirement_id,)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        """Fail closed without the caller's active cache-key context."""

        return ()

    @property
    def operational_evidence_claim_references(self) -> tuple[str, ...]:
        return self.proved_requirement_ids_for(self.key)

    @property
    def value(self) -> Any:
        """Return the producer value, falling back to the compact receipt."""

        return (
            self.producer_value
            if self.producer_value is not None
            else self.receipt
        )

    def require_completion_evidence(self) -> Mapping[str, Any]:
        if not self.is_completion_evidence or self.receipt is None:
            raise CacheCoordinationError(
                "coordinated analysis result is not completion evidence"
            )
        return self.receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "key_id": self.key.key_id,
            "cache_lookup_status": self.lookup.status.value,
            "cache_hit": self.cache_hit,
            "produced": self.produced,
            "shared": self.shared,
            "leader": self.leader,
            "waited": self.waited,
            "flight_id": self.flight_id,
            "producer_invocation_count": self.producer_invocation_count,
            "single_flight_collapse_evidence": (
                self.single_flight_collapse_evidence.to_dict()
                if self.single_flight_collapse_evidence is not None
                else None
            ),
            "operational_evidence_claim_references": list(
                self.operational_evidence_claim_references
            ),
            "is_completion_evidence": self.is_completion_evidence,
            "reason_codes": list(self.reason_codes),
            "outcome": self.outcome.value if self.outcome is not None else "",
            "stored": bool(self.store_result and self.store_result.stored),
        }


CoordinatorResult = CacheCoordinationResult
CacheCoordinatorResult = CacheCoordinationResult


@dataclass(frozen=True)
class CacheCoordinatorMetrics:
    """Atomic snapshot of coordinator activity."""

    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_validation_rejections: int = 0
    leaders: int = 0
    followers: int = 0
    produced: int = 0
    shared: int = 0
    completion_results: int = 0
    non_authoritative_results: int = 0
    producer_failures: int = 0
    wait_timeouts: int = 0
    active_flights: int = 0

    @property
    def collapsed_count(self) -> int:
        return self.followers

    @property
    def single_flight_savings(self) -> int:
        return self.followers

    @property
    def hit_ratio(self) -> float:
        return self.cache_hits / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_validation_rejections": self.cache_validation_rejections,
            "leaders": self.leaders,
            "followers": self.followers,
            "produced": self.produced,
            "shared": self.shared,
            "completion_results": self.completion_results,
            "non_authoritative_results": self.non_authoritative_results,
            "producer_failures": self.producer_failures,
            "wait_timeouts": self.wait_timeouts,
            "active_flights": self.active_flights,
            "collapsed_count": self.collapsed_count,
            "single_flight_savings": self.single_flight_savings,
            "hit_ratio": self.hit_ratio,
        }


CoordinatorMetrics = CacheCoordinatorMetrics


@dataclass
class _Flight:
    future: Future[CacheCoordinationResult]
    flight_id: str
    follower_count: int = 0
    producer_invocation_count: int = 0


class AnalysisCacheCoordinator:
    """Collapse identical in-process analysis misses to one producer."""

    def __init__(
        self,
        cache: AnalysisCache,
        *,
        wait_timeout_seconds: float | None = 30.0,
    ) -> None:
        if not isinstance(cache, AnalysisCache):
            raise ValueError("cache must be an AnalysisCache")
        if wait_timeout_seconds is not None and (
            isinstance(wait_timeout_seconds, bool)
            or not isinstance(wait_timeout_seconds, (int, float))
            or wait_timeout_seconds <= 0
        ):
            raise ValueError(
                "wait_timeout_seconds must be positive or None"
            )
        self.cache = cache
        self.wait_timeout_seconds = (
            None
            if wait_timeout_seconds is None
            else float(wait_timeout_seconds)
        )
        self._lock = threading.RLock()
        self._flights: dict[str, _Flight] = {}
        self._metric_values: dict[str, int] = {
            name: 0
            for name in CacheCoordinatorMetrics.__dataclass_fields__
            if name != "active_flights"
        }

    def _coerce_key(
        self, key: AnalysisCacheKey | Mapping[str, Any]
    ) -> AnalysisCacheKey:
        return (
            key
            if isinstance(key, AnalysisCacheKey)
            else AnalysisCacheKey.from_dict(key)
        )

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._metric_values[name] += amount

    def metrics(self) -> CacheCoordinatorMetrics:
        with self._lock:
            return CacheCoordinatorMetrics(
                **self._metric_values,
                active_flights=len(self._flights),
            )

    stats = metrics
    metrics_snapshot = metrics

    def reset_metrics(self) -> CacheCoordinatorMetrics:
        """Reset counters and return the snapshot that was replaced."""

        with self._lock:
            previous = CacheCoordinatorMetrics(
                **self._metric_values,
                active_flights=len(self._flights),
            )
            for name in self._metric_values:
                self._metric_values[name] = 0
            return previous

    def _completion_lookup(
        self, key: AnalysisCacheKey
    ) -> AnalysisCacheLookupResult:
        return self.cache.lookup(key, require_completion_evidence=True)

    @staticmethod
    def _is_exact_completion_hit(
        lookup: AnalysisCacheLookupResult, key: AnalysisCacheKey
    ) -> bool:
        return bool(
            lookup.status is AnalysisCacheLookupStatus.HIT
            and lookup.key == key
            and lookup.is_completion_evidence
        )

    def _is_accepted_completion_hit(
        self,
        lookup: AnalysisCacheLookupResult,
        key: AnalysisCacheKey,
        completion_validator: CompletionValidator | None,
    ) -> bool:
        """Apply a caller's outer-artifact gate to an exact cache hit.

        ``AnalysisCache`` proves the compact entry.  Some consumers also keep
        content-addressed bodies outside that entry and must independently
        validate those bodies before the hit can bypass production.  Running
        that validation inside both lookup phases preserves the
        lookup-to-flight race closure: an invalid outer artifact becomes a
        keyed miss, while a valid repair published by a preceding leader can
        still be reused.
        """

        if not self._is_exact_completion_hit(lookup, key):
            return False
        if completion_validator is None:
            return True
        accepted = completion_validator(lookup)
        if not isinstance(accepted, bool):
            raise CacheCoordinationError(
                "completion_validator must return a boolean"
            )
        if not accepted:
            self._increment("cache_validation_rejections")
        return accepted

    def _cache_hit_result(
        self,
        key: AnalysisCacheKey,
        lookup: AnalysisCacheLookupResult,
    ) -> CacheCoordinationResult:
        return CacheCoordinationResult(
            status=CacheCoordinationStatus.CACHE_HIT,
            key=key,
            lookup=lookup,
        )

    def _begin(
        self,
        key: AnalysisCacheKey,
        completion_validator: CompletionValidator | None = None,
    ) -> tuple[
        CacheCoordinationResult | None,
        _Flight | None,
        bool,
    ]:
        """Return ``(cached, flight, leader)`` without running user code."""

        self._increment("requests")
        lookup = self._completion_lookup(key)
        if self._is_accepted_completion_hit(
            lookup, key, completion_validator
        ):
            self._increment("cache_hits")
            self._increment("completion_results")
            return self._cache_hit_result(key, lookup), None, False

        with self._lock:
            # Close the lookup-to-registration race.  A preceding leader may
            # have populated the exact cache immediately before this lock.
            lookup = self._completion_lookup(key)
            if self._is_accepted_completion_hit(
                lookup, key, completion_validator
            ):
                self._metric_values["cache_hits"] += 1
                self._metric_values["completion_results"] += 1
                return self._cache_hit_result(key, lookup), None, False
            self._metric_values["cache_misses"] += 1
            existing = self._flights.get(key.key_id)
            if existing is not None:
                self._metric_values["followers"] += 1
                existing.follower_count += 1
                return None, existing, False
            flight = _Flight(
                Future(),
                flight_id=(
                    "analysis-single-flight:"
                    + secrets.token_hex(24)
                ),
            )
            self._flights[key.key_id] = flight
            self._metric_values["leaders"] += 1
            return None, flight, True

    def _finish_flight(
        self, key: AnalysisCacheKey, flight: _Flight
    ) -> None:
        with self._lock:
            if self._flights.get(key.key_id) is flight:
                del self._flights[key.key_id]

    @staticmethod
    def _validate_result_key(
        expected: AnalysisCacheKey, actual: AnalysisCacheKey
    ) -> None:
        if actual != expected:
            raise CacheProducerResultError(
                "producer result is bound to a different analysis cache key"
            )

    def _publish_producer_value(
        self,
        key: AnalysisCacheKey,
        value: ProducerValue | CachePublication,
        *,
        ttl_seconds: int | None,
        flight: _Flight,
    ) -> CacheCoordinationResult:
        store_result: AnalysisCacheStoreResult | None = None
        should_store = True
        if isinstance(value, CachePublication):
            should_store = value.store
            if value.ttl_seconds is not None:
                ttl_seconds = value.ttl_seconds
            value = value.value

        if not should_store:
            if not isinstance(
                value,
                (
                    Mapping,
                    AnalysisReceipt,
                    AnalysisCacheEntry,
                    AnalysisCacheLookupResult,
                    AnalysisCacheStoreResult,
                ),
            ):
                raise CacheProducerResultError(
                    "non-stored publication has an unsupported value"
                )
            lookup = self._completion_lookup(key)
            result = CacheCoordinationResult(
                status=CacheCoordinationStatus.PRODUCED,
                key=key,
                lookup=lookup,
                producer_value=value,
                leader=True,
                flight_id=flight.flight_id,
                producer_invocation_count=flight.producer_invocation_count,
            )
            self._increment("produced")
            self._increment("non_authoritative_results")
            return result

        if isinstance(value, AnalysisCacheLookupResult):
            self._validate_result_key(key, value.key)
            lookup = self._completion_lookup(key)
        elif isinstance(value, AnalysisCacheStoreResult):
            self._validate_result_key(key, value.key)
            store_result = value
            lookup = self._completion_lookup(key)
        elif isinstance(value, AnalysisCacheEntry):
            self._validate_result_key(key, value.key)
            store_result = self.cache.put(
                key,
                value.receipt,
                status=value.status,
                ttl_seconds=ttl_seconds,
            )
            lookup = self._completion_lookup(key)
        elif isinstance(value, AnalysisReceipt):
            store_result = self.cache.put(
                key, value, ttl_seconds=ttl_seconds
            )
            lookup = self._completion_lookup(key)
        elif isinstance(value, Mapping):
            store_result = self.cache.put(
                key, value, ttl_seconds=ttl_seconds
            )
            lookup = self._completion_lookup(key)
        else:
            raise CacheProducerResultError(
                "producer must return a compact receipt, cache entry, "
                "lookup result, or store result"
            )

        result = CacheCoordinationResult(
            status=CacheCoordinationStatus.PRODUCED,
            key=key,
            lookup=lookup,
            store_result=store_result,
            producer_value=value,
            leader=True,
            flight_id=flight.flight_id,
            producer_invocation_count=flight.producer_invocation_count,
        )
        self._increment("produced")
        self._increment(
            "completion_results"
            if result.is_completion_evidence
            else "non_authoritative_results"
        )
        return result

    def _seal_publication(
        self,
        result: CacheCoordinationResult,
        flight: _Flight,
    ) -> CacheCoordinationResult:
        """Snapshot the cohort and attach a proof before publishing its future."""

        with self._lock:
            follower_count = flight.follower_count
        if follower_count < 1 or not result.is_completion_evidence:
            return result
        entry = result.entry
        receipt = result.receipt
        if entry is None or not isinstance(receipt, Mapping):
            raise CacheCoordinationError(
                "completion publication lacks attestation bindings"
            )
        attestation = _SingleFlightAttestation(
            cache_key_id=result.key.key_id,
            flight_id=result.flight_id,
            publication_entry_digest=entry.entry_digest,
            receipt_id=_required_text(receipt.get("receipt_id"), "receipt_id"),
            producer_invocation_count=result.producer_invocation_count,
            participant_count=follower_count + 1,
            follower_count=follower_count,
            seal=object(),
        )
        evidence = SingleFlightCollapseEvidence.from_result(
            result,
            follower_count=follower_count,
            _attestation=attestation,
        )
        return replace(
            result,
            single_flight_collapse_evidence=evidence,
            _single_flight_attestation=attestation,
        )

    def _shared_result(
        self, result: CacheCoordinationResult
    ) -> CacheCoordinationResult:
        # Authority remains whatever the leader established with an exact
        # post-store cache lookup.  The status only describes coordination.
        shared = replace(
            result,
            status=CacheCoordinationStatus.SHARED,
            leader=False,
            waited=True,
        )
        self._increment("shared")
        self._increment(
            "completion_results"
            if shared.is_completion_evidence
            else "non_authoritative_results"
        )
        return shared

    def _validated_shared_result(
        self,
        result: CacheCoordinationResult,
        key: AnalysisCacheKey,
        completion_validator: CompletionValidator | None,
    ) -> CacheCoordinationResult:
        """Reapply a waiter's artifact gate before sharing authority.

        Validators may close over caller-local artifact stores or other
        request-bound state.  A leader accepting its compact publication does
        not prove that a follower can load the same external artifact.
        Followers therefore revalidate the exact published lookup with their
        own validator.  Rejection fails closed instead of returning a shared
        result whose ``is_completion_evidence`` flag would grant authority.
        """

        if (
            completion_validator is not None
            and result.is_completion_evidence
            and not self._is_accepted_completion_hit(
                result.lookup, key, completion_validator
            )
        ):
            raise CacheCoordinationError(
                "shared completion result rejected by caller artifact validator"
            )
        return self._shared_result(result)

    def _timeout(
        self, key: AnalysisCacheKey, timeout: float | None
    ) -> CacheCoordinationTimeout:
        self._increment("wait_timeouts")
        detail = "without a deadline" if timeout is None else f"after {timeout:g}s"
        return CacheCoordinationTimeout(
            f"timed out waiting for analysis single flight {key.key_id} {detail}"
        )

    def get_or_compute(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        producer: SyncProducer,
        *,
        ttl_seconds: int | None = None,
        wait_timeout_seconds: float | None = None,
        completion_validator: CompletionValidator | None = None,
    ) -> CacheCoordinationResult:
        """Return an exact completion hit or run ``producer`` once.

        ``producer`` is called with no arguments.  ``completion_validator``
        may add a fail-closed gate for content-addressed bodies kept outside
        the compact cache entry.  Use :meth:`async_get_or_compute` for
        coroutine producers.
        """

        if not callable(producer):
            raise ValueError("producer must be callable")
        if completion_validator is not None and not callable(
            completion_validator
        ):
            raise ValueError("completion_validator must be callable or None")
        cache_key = self._coerce_key(key)
        if completion_validator is None:
            cached, flight, leader = self._begin(cache_key)
        else:
            cached, flight, leader = self._begin(
                cache_key, completion_validator
            )
        if cached is not None:
            return cached
        assert flight is not None

        if leader:
            try:
                flight.producer_invocation_count += 1
                value = producer()
                if inspect.isawaitable(value):
                    close = getattr(value, "close", None)
                    if callable(close):
                        close()
                    raise CacheProducerResultError(
                        "synchronous producer returned an awaitable; use "
                        "async_get_or_compute"
                    )
                result = self._publish_producer_value(
                    cache_key,
                    value,
                    ttl_seconds=ttl_seconds,
                    flight=flight,
                )
                result = self._seal_publication(result, flight)
                flight.future.set_result(result)
                return result
            except BaseException as exc:
                self._increment("producer_failures")
                flight.future.set_exception(exc)
                raise
            finally:
                self._finish_flight(cache_key, flight)

        timeout = (
            self.wait_timeout_seconds
            if wait_timeout_seconds is None
            else self._validate_timeout(wait_timeout_seconds)
        )
        try:
            result = flight.future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise self._timeout(cache_key, timeout) from exc
        return self._validated_shared_result(
            result, cache_key, completion_validator
        )

    def _validate_timeout(self, value: float | None) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError("wait_timeout_seconds must be positive or None")
        return float(value)

    async def async_get_or_compute(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        producer: AsyncProducer,
        *,
        ttl_seconds: int | None = None,
        wait_timeout_seconds: float | None = None,
        completion_validator: CompletionValidator | None = None,
    ) -> CacheCoordinationResult:
        """Async counterpart that shares flights with synchronous callers."""

        if not callable(producer):
            raise ValueError("producer must be callable")
        if completion_validator is not None and not callable(
            completion_validator
        ):
            raise ValueError("completion_validator must be callable or None")
        cache_key = self._coerce_key(key)
        if completion_validator is None:
            cached, flight, leader = self._begin(cache_key)
        else:
            cached, flight, leader = self._begin(
                cache_key, completion_validator
            )
        if cached is not None:
            return cached
        assert flight is not None

        if leader:
            try:
                flight.producer_invocation_count += 1
                value = producer()
                if inspect.isawaitable(value):
                    value = await value
                result = self._publish_producer_value(
                    cache_key,
                    value,
                    ttl_seconds=ttl_seconds,
                    flight=flight,
                )
                result = self._seal_publication(result, flight)
                flight.future.set_result(result)
                return result
            except BaseException as exc:
                self._increment("producer_failures")
                flight.future.set_exception(exc)
                raise
            finally:
                self._finish_flight(cache_key, flight)

        timeout = (
            self.wait_timeout_seconds
            if wait_timeout_seconds is None
            else self._validate_timeout(wait_timeout_seconds)
        )
        wrapped = asyncio.wrap_future(flight.future)
        try:
            if timeout is None:
                result = await asyncio.shield(wrapped)
            else:
                result = await asyncio.wait_for(
                    asyncio.shield(wrapped), timeout=timeout
                )
        except asyncio.TimeoutError as exc:
            raise self._timeout(cache_key, timeout) from exc
        return self._validated_shared_result(
            result, cache_key, completion_validator
        )

    # Conventional synchronous spellings.
    coordinate = get_or_compute
    run = get_or_compute
    execute = get_or_compute
    single_flight = get_or_compute
    execute_single_flight = get_or_compute
    run_single_flight = get_or_compute

    # Conventional asynchronous spellings.
    aget_or_compute = async_get_or_compute
    acoordinate = async_get_or_compute
    arun = async_get_or_compute
    aexecute = async_get_or_compute
    async_single_flight = async_get_or_compute
    execute_async_single_flight = async_get_or_compute


CacheCoordinator = AnalysisCacheCoordinator
SingleFlightCacheCoordinator = AnalysisCacheCoordinator


__all__ = [
    "AnalysisCacheCoordinator",
    "CONCURRENT_IDENTICAL_MISS_COLLAPSE_REQUIREMENT_ID",
    "CacheCoordinationError",
    "CacheCoordinationResult",
    "CacheCoordinationStatus",
    "CacheCoordinationTimeout",
    "CachePublication",
    "CacheCoordinator",
    "CacheCoordinatorMetrics",
    "CacheCoordinatorResult",
    "CacheCoordinatorStatus",
    "CacheProducerResultError",
    "CompletionValidator",
    "CoordinatorMetrics",
    "CoordinatorResult",
    "CoordinatorStatus",
    "SINGLE_FLIGHT_COLLAPSE_EVIDENCE_SCHEMA",
    "SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID",
    "SingleFlightCollapseEvidence",
    "SingleFlightCacheCoordinator",
]
