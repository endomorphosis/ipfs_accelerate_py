"""Trust-aware coordination for supervisor caches.

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

The analysis compatibility facade uses a future as its in-process rendezvous
point.  :class:`NamespaceCacheCoordinator` is the common persistent layer for
analysis, context, planning, proof, validation, and merge classifications.  It
adds canonical namespace metadata, semantic keys, bounded artifact references,
integrity checking, quotas, garbage collection, lookup metrics, and
cross-process keyed leases.  Namespace-specific payload schemas and validators
remain authoritative; the common envelope never upgrades a draft, negative,
or inconclusive record into completion evidence.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import json
import os
import secrets
import tempfile
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Final,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

from .analysis_cache import (
    ANALYSIS_CACHE_ENTRY_SCHEMA,
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
from ..merge.lease_coordination import (
    DistributedSingleFlightCancelled,
    DistributedSingleFlightCoordinator,
    DistributedSingleFlightResult,
    DistributedSingleFlightTimeout,
    SingleFlightAttestation,
    SingleFlightOutcome,
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
INTEGRATED_ANALYSIS_CACHE_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    "expensive identical misses collapse across lanes",
    "stale or negative records never become completion evidence",
    (
        "repeated fixtures achieve at least 70 percent cache reuse with "
        "zero stale authoritative hits."
    ),
)
_SINGLE_FLIGHT_ATTESTATION_SEAL: Final = object()


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
            or _attestation.seal is not _SINGLE_FLIGHT_ATTESTATION_SEAL
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
            and attestation.seal is _SINGLE_FLIGHT_ATTESTATION_SEAL
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
    """Collapse identical analysis misses across threads and processes."""

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
        self._lease_dir = cache.path / ".analysis-single-flight"
        self._lease_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._metric_values: dict[str, int] = {
            name: 0
            for name in CacheCoordinatorMetrics.__dataclass_fields__
            if name != "active_flights"
        }

    @contextmanager
    def _process_flight_lease(
        self, key: AnalysisCacheKey, timeout: float | None
    ) -> Iterator[None]:
        """Acquire the durable per-key lease used by process leaders.

        The in-memory flight still handles rich result/failure fan-out inside
        this process.  The file lease makes one such leader globally active;
        after acquiring it the leader always repeats the authoritative cache
        lookup before invoking user code.
        """

        descriptor = self._acquire_process_flight_lease(key, timeout)
        try:
            yield
        finally:
            self._release_process_flight_lease(descriptor)

    def _acquire_process_flight_lease(
        self, key: AnalysisCacheKey, timeout: float | None
    ) -> int:
        path = self._lease_dir / f"{key.digest}.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(
                        descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    break
                except BlockingIOError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise self._timeout(key, timeout)
                    time.sleep(0.01)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _release_process_flight_lease(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _cross_process_hit(
        self,
        key: AnalysisCacheKey,
        completion_validator: CompletionValidator | None,
    ) -> CacheCoordinationResult | None:
        lookup = self._completion_lookup(key)
        if not self._is_accepted_completion_hit(
            lookup,
            key,
            completion_validator,
            count_rejection=False,
        ):
            return None
        self._increment("cache_hits")
        self._increment("completion_results")
        # This is a normal exact-key hit.  A process lease/outcome is never
        # promoted into operational completion evidence of its own.
        return self._cache_hit_result(key, lookup)

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
        *,
        count_rejection: bool = True,
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
        if not accepted and count_rejection:
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
            seal=_SINGLE_FLIGHT_ATTESTATION_SEAL,
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

        if result.status is CacheCoordinationStatus.CACHE_HIT:
            if not self._is_accepted_completion_hit(
                result.lookup, key, completion_validator
            ):
                raise CacheCoordinationError(
                    "cross-process completion result rejected by caller "
                    "artifact validator"
                )
            return result
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
                timeout = (
                    self.wait_timeout_seconds
                    if wait_timeout_seconds is None
                    else self._validate_timeout(wait_timeout_seconds)
                )
                with self._process_flight_lease(cache_key, timeout):
                    cross_process_hit = self._cross_process_hit(
                        cache_key, completion_validator
                    )
                    if cross_process_hit is not None:
                        result = cross_process_hit
                    else:
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
                timeout = (
                    self.wait_timeout_seconds
                    if wait_timeout_seconds is None
                    else self._validate_timeout(wait_timeout_seconds)
                )
                # Acquire through a worker thread so a leader in another
                # process cannot block this event loop while it finishes.
                descriptor = await asyncio.to_thread(
                    self._acquire_process_flight_lease,
                    cache_key,
                    timeout,
                )
                try:
                    cross_process_hit = self._cross_process_hit(
                        cache_key, completion_validator
                    )
                    if cross_process_hit is not None:
                        result = cross_process_hit
                    else:
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
                finally:
                    self._release_process_flight_lease(descriptor)
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


COMMON_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/common-cache-key@1"
)
COMMON_CACHE_ENTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/common-cache-entry@1"
)
COMMON_CACHE_NAMESPACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cache-namespace@1"
)
CACHE_CAS_ADAPTER_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cache-cas-adapter-binding@1"
)
CACHE_CAS_ADAPTER_PAYLOAD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cache-cas-adapter-payload@1"
)
NAMESPACE_SINGLE_FLIGHT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/namespace-single-flight-result@1"
)
DEFAULT_NAMESPACE_MAX_ENTRIES: Final = 512
DEFAULT_NAMESPACE_MAX_BYTES: Final = 32 * 1024 * 1024
DEFAULT_NAMESPACE_MAX_ENTRY_BYTES: Final = 256 * 1024
DEFAULT_NAMESPACE_NEGATIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_NAMESPACE_MAX_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_MAX_ARTIFACT_REFERENCES: Final = 32
DEFAULT_MAX_ARTIFACT_REFERENCE_BYTES: Final = 16 * 1024


class CacheNamespace(str, Enum):
    """Stable cache classifications; each retains its native schema."""

    ANALYSIS = "analysis"
    CONTEXT = "context"
    PLANNING = "planning"
    PROVIDER = "provider"
    PROOF = "proof"
    PROOF_DRAFT = "proof_draft"
    VALIDATION = "validation"
    MERGE = "merge"

    # Common singular spelling used by plan-oriented callers.
    PLAN = "planning"
    PROVIDER_CALL = "provider"
    INFERENCE = "provider"
    PROOF_RECEIPT = "proof"
    VALIDATION_COMMAND = "validation"
    MERGE_CLASSIFICATION = "merge"


class CacheAuthority(str, Enum):
    """Trust class of a persisted record."""

    AUTHORITATIVE = "authoritative"
    DIAGNOSTIC = "diagnostic"
    DRAFT = "draft"


class CacheRecordOutcome(str, Enum):
    """Outcome class independent of a namespace's native status spelling."""

    SUCCESSFUL = "successful"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def coerce(cls, value: Any) -> "CacheRecordOutcome":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "success": cls.SUCCESSFUL,
            "succeeded": cls.SUCCESSFUL,
            "complete": cls.SUCCESSFUL,
            "completed": cls.SUCCESSFUL,
            "ok": cls.SUCCESSFUL,
            "failed": cls.NEGATIVE,
            "failure": cls.NEGATIVE,
            "error": cls.NEGATIVE,
            "timed_out": cls.NEGATIVE,
            "timeout": cls.NEGATIVE,
            "partial": cls.INCONCLUSIVE,
            "unknown": cls.INCONCLUSIVE,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            raise ValueError(
                "outcome must be successful, negative, or inconclusive"
            ) from exc

    @property
    def can_complete(self) -> bool:
        return self is CacheRecordOutcome.SUCCESSFUL


class NamespaceLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CacheQuotaPolicy:
    """Per-namespace persistence and artifact bounds."""

    max_entries: int = DEFAULT_NAMESPACE_MAX_ENTRIES
    max_bytes: int = DEFAULT_NAMESPACE_MAX_BYTES
    max_entry_bytes: int = DEFAULT_NAMESPACE_MAX_ENTRY_BYTES
    max_artifact_references: int = DEFAULT_MAX_ARTIFACT_REFERENCES
    max_artifact_reference_bytes: int = DEFAULT_MAX_ARTIFACT_REFERENCE_BYTES
    negative_ttl_seconds: int = DEFAULT_NAMESPACE_NEGATIVE_TTL_SECONDS
    max_ttl_seconds: int = DEFAULT_NAMESPACE_MAX_TTL_SECONDS

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_entry_bytes > self.max_bytes:
            raise ValueError("max_entry_bytes cannot exceed max_bytes")
        if self.negative_ttl_seconds > self.max_ttl_seconds:
            raise ValueError("negative_ttl_seconds cannot exceed max_ttl_seconds")

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


_NAMESPACE_NATIVE_SCHEMAS: Final[dict[CacheNamespace, tuple[str, str]]] = {
    CacheNamespace.ANALYSIS: (
        "ipfs_accelerate_py/agent-supervisor/analysis-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/analysis-cache-entry@1",
    ),
    CacheNamespace.CONTEXT: (
        "ipfs_accelerate_py/agent-supervisor/context-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/context-cache-entry@1",
    ),
    CacheNamespace.PLANNING: (
        "ipfs_accelerate_py/agent-supervisor/planning-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/planning-cache-entry@1",
    ),
    CacheNamespace.PROVIDER: (
        "ipfs_accelerate_py/agent-supervisor/provider-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/provider-cache-entry@1",
    ),
    CacheNamespace.PROOF: (
        "ipfs_accelerate_py/agent-supervisor/formal-verification-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/formal-verification-cache-entry@1",
    ),
    CacheNamespace.PROOF_DRAFT: (
        "ipfs_accelerate_py/agent-supervisor/formal-verification-draft-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/formal-verification-draft-cache-entry@1",
    ),
    CacheNamespace.VALIDATION: (
        "ipfs_accelerate_py/agent-supervisor/validation-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/validation-cache@1",
    ),
    CacheNamespace.MERGE: (
        "ipfs_accelerate_py/agent-supervisor/merge-classification-cache-key@1",
        "ipfs_accelerate_py/agent-supervisor/merge-classification-cache-entry@1",
    ),
}

_NAMESPACE_REQUIRED_DIMENSIONS: Final[
    dict[CacheNamespace, tuple[str, ...]]
] = {
    CacheNamespace.ANALYSIS: (
        "repository_tree_identity",
        "objective_revision",
        "analyzer_version",
        "schema_version",
        "configuration_digest",
        "query_digest",
        "policy_digest",
    ),
    CacheNamespace.CONTEXT: (
        "repository_tree_identity",
        "objective_revision",
        "compiler_version",
        "schema_version",
        "configuration_digest",
        "request_digest",
        "policy_digest",
    ),
    CacheNamespace.PLANNING: (
        "repository_tree_identity",
        "objective_revision",
        "planner_version",
        "schema_version",
        "context_digest",
        "configuration_digest",
        "policy_digest",
        "capability_digest",
    ),
    CacheNamespace.PROVIDER: (
        "operation",
        "request_digest",
        "provider_id",
        "provider_version",
        "capability_revision",
        "protocol_version",
        "configuration_digest",
        "policy_digest",
        "resource_budget_digest",
    ),
    CacheNamespace.PROOF: (
        "obligation",
        "premises",
        "translator",
        "solver",
        "kernel",
        "toolchain",
        "theorem_registry",
        "policy",
        "resource_budget",
        "candidate_tree",
    ),
    CacheNamespace.PROOF_DRAFT: (
        "goal_digest",
        "repository_tree_digest",
        "vocabulary_digest",
        "compiler_digest",
        "model_route_digest",
        "model_version",
        "assumptions_digest",
        "bounds_digest",
        "policy_digest",
    ),
    CacheNamespace.VALIDATION: (
        "target_commit",
        "candidate_tree",
        "command",
        "environment",
        "dependency_state",
        "toolchain",
        "policy",
        "schema_version",
    ),
    CacheNamespace.MERGE: (
        "candidate_tree",
        "target_branch",
        "merge_base",
        "classifier_version",
        "validation_digest",
        "policy_digest",
    ),
}


@dataclass(frozen=True)
class CacheNamespaceMetadata:
    """Common metadata without replacing a namespace's native contracts."""

    namespace: CacheNamespace
    key_schema: str
    entry_schema: str
    authority: CacheAuthority = CacheAuthority.DIAGNOSTIC
    required_dimensions: tuple[str, ...] = ()
    common_schema: str = COMMON_CACHE_NAMESPACE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, CacheNamespace):
            object.__setattr__(
                self, "namespace", CacheNamespace(str(self.namespace))
            )
        if not isinstance(self.authority, CacheAuthority):
            object.__setattr__(
                self, "authority", CacheAuthority(str(self.authority))
            )
        _required_text(self.key_schema, "key_schema")
        _required_text(self.entry_schema, "entry_schema")
        object.__setattr__(
            self,
            "required_dimensions",
            tuple(
                _required_text(item, "required dimension")
                for item in self.required_dimensions
            ),
        )
        if self.common_schema != COMMON_CACHE_NAMESPACE_SCHEMA:
            raise ValueError("unsupported common cache namespace schema")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.common_schema,
            "namespace": self.namespace.value,
            "key_schema": self.key_schema,
            "entry_schema": self.entry_schema,
            "authority": self.authority.value,
            "required_dimensions": list(self.required_dimensions),
        }

    @classmethod
    def for_namespace(
        cls,
        namespace: CacheNamespace | str,
        *,
        authority: CacheAuthority | str = CacheAuthority.DIAGNOSTIC,
        key_schema: str | None = None,
        entry_schema: str | None = None,
    ) -> "CacheNamespaceMetadata":
        kind = (
            namespace
            if isinstance(namespace, CacheNamespace)
            else CacheNamespace(str(namespace))
        )
        native_key, native_entry = _NAMESPACE_NATIVE_SCHEMAS[kind]
        return cls(
            namespace=kind,
            key_schema=key_schema or native_key,
            entry_schema=entry_schema or native_entry,
            authority=(
                authority
                if isinstance(authority, CacheAuthority)
                else CacheAuthority(str(authority))
            ),
            required_dimensions=_NAMESPACE_REQUIRED_DIMENSIONS[kind],
        )


def namespace_metadata(
    namespace: CacheNamespace | str,
    *,
    authority: CacheAuthority | str = CacheAuthority.DIAGNOSTIC,
    key_schema: str | None = None,
    entry_schema: str | None = None,
) -> CacheNamespaceMetadata:
    """Return canonical common metadata for one native cache namespace."""

    return CacheNamespaceMetadata.for_namespace(
        namespace,
        authority=authority,
        key_schema=key_schema,
        entry_schema=entry_schema,
    )


def _common_json_value(value: Any, *, name: str) -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain canonical JSON values") from exc


def _common_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class SemanticCacheKey:
    """Canonical namespace plus every named semantic invalidation dimension."""

    namespace: CacheNamespace
    dimensions: Mapping[str, Any]
    schema: str = COMMON_CACHE_KEY_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, CacheNamespace):
            object.__setattr__(
                self, "namespace", CacheNamespace(str(self.namespace))
            )
        if self.schema != COMMON_CACHE_KEY_SCHEMA:
            raise ValueError("unsupported common cache key schema")
        if not isinstance(self.dimensions, Mapping) or not self.dimensions:
            raise ValueError("semantic cache key requires named dimensions")
        normalized: dict[str, Any] = {}
        for raw_name, raw_value in self.dimensions.items():
            name = _required_text(raw_name, "semantic dimension name")
            if name in normalized:
                raise ValueError(f"duplicate semantic dimension: {name}")
            if raw_value is None or (
                isinstance(raw_value, str) and not raw_value.strip()
            ):
                raise ValueError(f"semantic dimension {name} is required")
            normalized[name] = _common_json_value(
                raw_value, name=f"semantic dimension {name}"
            )
        object.__setattr__(
            self, "dimensions", {name: normalized[name] for name in sorted(normalized)}
        )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "namespace": self.namespace.value,
            "dimensions": dict(self.dimensions),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_common_json_bytes(self._content())).hexdigest()

    @property
    def key_id(self) -> str:
        return f"supervisor-cache:{self.namespace.value}:sha256:{self.digest}"

    @property
    def semantic_key(self) -> str:
        return self.key_id

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "key_id": self.key_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticCacheKey":
        if not isinstance(value, Mapping):
            raise ValueError("semantic cache key must be an object")
        if value.get("schema") != COMMON_CACHE_KEY_SCHEMA:
            raise ValueError("unsupported common cache key schema")
        restored = cls(
            namespace=CacheNamespace(str(value.get("namespace") or "")),
            dimensions=value.get("dimensions"),
        )
        supplied = value.get("key_id")
        if supplied is not None and supplied != restored.key_id:
            raise ValueError("semantic cache key identity mismatch")
        return restored


def build_semantic_cache_key(
    namespace: CacheNamespace | str,
    dimensions: Mapping[str, Any] | None = None,
    **semantic_dimensions: Any,
) -> SemanticCacheKey:
    """Build an exact semantic key without dropping caller-named dimensions."""

    combined = dict(dimensions or {})
    overlap = set(combined).intersection(semantic_dimensions)
    if overlap:
        raise ValueError(
            "duplicate semantic dimensions: " + ", ".join(sorted(overlap))
        )
    combined.update(semantic_dimensions)
    return SemanticCacheKey(
        namespace=(
            namespace
            if isinstance(namespace, CacheNamespace)
            else CacheNamespace(str(namespace))
        ),
        dimensions=combined,
    )


# Short spelling expected by embedding callers.
build_semantic_key = build_semantic_cache_key


def build_namespace_semantic_key(
    namespace: CacheNamespace | str,
    dimensions: Mapping[str, Any] | None = None,
    **semantic_dimensions: Any,
) -> SemanticCacheKey:
    """Build a key after checking the namespace's complete dimension contract."""

    kind = (
        namespace
        if isinstance(namespace, CacheNamespace)
        else CacheNamespace(str(namespace))
    )
    combined = dict(dimensions or {})
    overlap = set(combined).intersection(semantic_dimensions)
    if overlap:
        raise ValueError(
            "duplicate semantic dimensions: " + ", ".join(sorted(overlap))
        )
    combined.update(semantic_dimensions)
    required = set(_NAMESPACE_REQUIRED_DIMENSIONS[kind])
    missing = sorted(required.difference(combined))
    if missing:
        raise ValueError(
            f"{kind.value} semantic key is missing dimensions: "
            + ", ".join(missing)
        )
    return SemanticCacheKey(kind, combined)


_ARTIFACT_IDENTITY_FIELDS: Final = frozenset(
    {"artifact_id", "cid", "digest", "uri", "path", "ref", "schema", "kind", "size_bytes"}
)
_ARTIFACT_BODY_FIELDS: Final = frozenset(
    {"body", "content", "contents", "data", "payload", "source", "text", "bytes"}
)


@dataclass(frozen=True)
class BoundedArtifactReference:
    """A shallow artifact identity; artifact bodies never enter common entries."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping) or not self.value:
            raise ValueError("artifact reference must be a nonempty object")
        normalized: dict[str, Any] = {}
        for raw_name, raw_value in self.value.items():
            name = _required_text(raw_name, "artifact reference field")
            if name in _ARTIFACT_BODY_FIELDS:
                raise ValueError(
                    f"artifact reference {name} cannot embed artifact content"
                )
            if name not in _ARTIFACT_IDENTITY_FIELDS:
                raise ValueError(f"unsupported artifact reference field: {name}")
            if isinstance(raw_value, (Mapping, Sequence)) and not isinstance(
                raw_value, (str, bytes, bytearray)
            ):
                raise ValueError("artifact reference values must be scalar")
            if isinstance(raw_value, (bytes, bytearray)):
                raise ValueError("artifact reference values cannot contain bytes")
            normalized[name] = _common_json_value(
                raw_value, name=f"artifact reference {name}"
            )
        if not set(normalized).intersection(
            {"artifact_id", "cid", "digest", "uri", "path", "ref"}
        ):
            raise ValueError("artifact reference requires a stable identity")
        object.__setattr__(self, "value", normalized)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.value)


# Compatibility-friendly singular spelling.
ArtifactReference = BoundedArtifactReference


@dataclass(frozen=True)
class CacheWrite:
    """A producer's native payload and trust-aware persistence decision."""

    payload: Any
    outcome: CacheRecordOutcome = CacheRecordOutcome.SUCCESSFUL
    authority: CacheAuthority = CacheAuthority.DIAGNOSTIC
    ttl_seconds: int | None = None
    artifact_references: tuple[BoundedArtifactReference, ...] = ()
    payload_schema: str = ""
    store: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, CacheRecordOutcome):
            object.__setattr__(
                self, "outcome", CacheRecordOutcome.coerce(self.outcome)
            )
        if not isinstance(self.authority, CacheAuthority):
            object.__setattr__(
                self, "authority", CacheAuthority(str(self.authority))
            )
        if not isinstance(self.store, bool):
            raise ValueError("store must be a boolean")
        if self.ttl_seconds is not None and (
            isinstance(self.ttl_seconds, bool)
            or not isinstance(self.ttl_seconds, int)
            or self.ttl_seconds < 1
        ):
            raise ValueError("ttl_seconds must be a positive integer or None")
        references = tuple(
            item
            if isinstance(item, BoundedArtifactReference)
            else BoundedArtifactReference(item)
            for item in self.artifact_references
        )
        object.__setattr__(self, "artifact_references", references)


@dataclass(frozen=True)
class NamespaceCacheEntry:
    metadata: CacheNamespaceMetadata
    key: SemanticCacheKey
    payload: Any
    outcome: CacheRecordOutcome
    authority: CacheAuthority
    created_at_ms: int
    expires_at_ms: int | None
    artifact_references: tuple[BoundedArtifactReference, ...] = ()
    payload_schema: str = ""
    entry_digest: str = ""

    def _content(self) -> dict[str, Any]:
        return {
            "schema": COMMON_CACHE_ENTRY_SCHEMA,
            "metadata": self.metadata.to_dict(),
            "key": self.key.to_dict(),
            "key_id": self.key.key_id,
            "payload_schema": self.payload_schema or self.metadata.entry_schema,
            "payload": self.payload,
            "outcome": self.outcome.value,
            "authority": self.authority.value,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "artifact_references": [
                item.to_dict() for item in self.artifact_references
            ],
        }

    @property
    def computed_digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            _common_json_bytes(self._content())
        ).hexdigest()

    @property
    def is_completion_evidence(self) -> bool:
        return bool(
            self.outcome.can_complete
            and self.authority is CacheAuthority.AUTHORITATIVE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content(),
            "entry_digest": self.entry_digest or self.computed_digest,
            # Caller-provided flags are never accepted on decode; this is a
            # projection of outcome plus authority, not stored trust input.
            "is_completion_evidence": self.is_completion_evidence,
        }


@dataclass(frozen=True)
class NamespaceCacheLookup:
    status: NamespaceLookupStatus
    key: SemanticCacheKey
    entry: NamespaceCacheEntry | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.status is NamespaceLookupStatus.HIT

    @property
    def is_completion_evidence(self) -> bool:
        return bool(
            self.hit
            and self.entry is not None
            and self.entry.is_completion_evidence
        )

    @property
    def payload(self) -> Any:
        return self.entry.payload if self.entry is not None else None


@dataclass(frozen=True)
class NamespaceCacheMetrics:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    rejected: int = 0
    stale_rejections: int = 0
    corruption_recoveries: int = 0
    poisoned_rejections: int = 0
    writes: int = 0
    write_rejections: int = 0
    leaders: int = 0
    followers: int = 0
    evictions: int = 0
    gc_runs: int = 0
    wait_timeouts: int = 0
    bytes_reused: int = 0
    stale_authoritative_hits: int = 0
    active_flights: int = 0
    entries: int = 0
    bytes: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def single_flight_savings(self) -> int:
        return self.followers

    @property
    def requests(self) -> int:
        return self.lookups

    @property
    def cache_hits(self) -> int:
        return self.hits

    @property
    def cache_misses(self) -> int:
        return self.misses

    @property
    def invalidated(self) -> int:
        return self.rejected

    def to_dict(self) -> dict[str, Any]:
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        values["hit_ratio"] = self.hit_ratio
        values["single_flight_savings"] = self.single_flight_savings
        values["requests"] = self.requests
        values["cache_hits"] = self.cache_hits
        values["cache_misses"] = self.cache_misses
        values["invalidated"] = self.invalidated
        return values


@dataclass(frozen=True)
class NamespaceCoordinationResult:
    lookup: NamespaceCacheLookup
    produced: bool = False
    shared: bool = False
    producer_value: Any = None
    fencing_token: int = 0
    flight_owner_id: str = ""
    outcome_digest: str = ""
    attestation: SingleFlightAttestation | None = None
    single_flight_outcome: SingleFlightOutcome | None = None

    @property
    def cache_hit(self) -> bool:
        return self.lookup.hit and not self.produced

    @property
    def value(self) -> Any:
        return (
            self.producer_value
            if self.producer_value is not None
            else self.lookup.payload
        )

    @property
    def is_completion_evidence(self) -> bool:
        return self.lookup.is_completion_evidence

    @property
    def attested(self) -> bool:
        return bool(
            self.attestation is not None
            and self.single_flight_outcome is not None
            and self.attestation == self.single_flight_outcome.attestation
        )


_COMMON_THREAD_LOCKS: dict[str, threading.RLock] = {}
_COMMON_THREAD_LOCKS_GUARD = threading.Lock()


def _common_thread_lock(path: Path) -> threading.RLock:
    identity = str(path.absolute())
    with _COMMON_THREAD_LOCKS_GUARD:
        return _COMMON_THREAD_LOCKS.setdefault(identity, threading.RLock())


class NamespaceCacheCoordinator:
    """Persistent common cache envelope with keyed cross-process single-flight.

    The lock is a coordination primitive, not an authority token.  Every
    follower performs a fresh exact-key read and the namespace payload
    validator is rerun before a hit is returned.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        quotas: CacheQuotaPolicy | Mapping[CacheNamespace | str, CacheQuotaPolicy] | None = None,
        wait_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
        single_flight_coordinator: DistributedSingleFlightCoordinator | None = None,
        coordination_path: str | os.PathLike[str] | None = None,
        lease_seconds: float = 30.0,
        outcome_ttl_seconds: float = 60.0,
    ) -> None:
        if path is None:
            path = tempfile.mkdtemp(prefix="supervisor-cache-")
        if (
            isinstance(wait_timeout_seconds, bool)
            or not isinstance(wait_timeout_seconds, (int, float))
            or wait_timeout_seconds <= 0
        ):
            raise ValueError("wait_timeout_seconds must be positive")
        self.path = Path(path)
        self.entries_path = self.path / "entries"
        self.locks_path = self.path / "locks"
        self.entries_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.locks_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self._clock = clock
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be positive")
        if (
            isinstance(outcome_ttl_seconds, bool)
            or not isinstance(outcome_ttl_seconds, (int, float))
            or outcome_ttl_seconds <= 0
        ):
            raise ValueError("outcome_ttl_seconds must be positive")
        if single_flight_coordinator is not None and coordination_path is not None:
            raise ValueError(
                "provide single_flight_coordinator or coordination_path, not both"
            )
        if single_flight_coordinator is not None and not isinstance(
            single_flight_coordinator, DistributedSingleFlightCoordinator
        ):
            raise ValueError(
                "single_flight_coordinator must be a "
                "DistributedSingleFlightCoordinator"
            )
        self.lease_seconds = float(lease_seconds)
        self.outcome_ttl_seconds = float(outcome_ttl_seconds)
        self.single_flight_coordinator = (
            single_flight_coordinator
            or DistributedSingleFlightCoordinator(
                coordination_path or (self.path / "single-flight.sqlite3"),
                lease_seconds=self.lease_seconds,
                outcome_ttl_seconds=self.outcome_ttl_seconds,
                clock_ms=self._now_ms,
            )
        )
        self._metrics_lock = threading.Lock()
        self._metric_values = {
            name: 0
            for name in NamespaceCacheMetrics.__dataclass_fields__
            if name not in {"entries", "bytes", "active_flights"}
        }
        self._active_flights = 0
        if quotas is None:
            self._quotas = {
                namespace: CacheQuotaPolicy() for namespace in CacheNamespace
            }
        elif isinstance(quotas, CacheQuotaPolicy):
            self._quotas = {namespace: quotas for namespace in CacheNamespace}
        else:
            normalized = {
                (
                    namespace
                    if isinstance(namespace, CacheNamespace)
                    else CacheNamespace(str(namespace))
                ): policy
                for namespace, policy in quotas.items()
            }
            self._quotas = {
                namespace: normalized.get(namespace, CacheQuotaPolicy())
                for namespace in CacheNamespace
            }

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            self._metric_values[name] += amount

    def _coerce_key(
        self, key: SemanticCacheKey | Mapping[str, Any]
    ) -> SemanticCacheKey:
        return (
            key
            if isinstance(key, SemanticCacheKey)
            else SemanticCacheKey.from_dict(key)
        )

    def _entry_path(self, key: SemanticCacheKey) -> Path:
        return (
            self.entries_path
            / key.namespace.value
            / key.digest[:2]
            / f"{key.digest}.json"
        )

    def _lease_path(self, key: SemanticCacheKey) -> Path:
        return self.locks_path / key.namespace.value / f"{key.digest}.lock"

    def _quota(self, namespace: CacheNamespace) -> CacheQuotaPolicy:
        return self._quotas[namespace]

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    @contextmanager
    def _process_lease(
        self, key: SemanticCacheKey, timeout_seconds: float | None = None
    ) -> Iterator[None]:
        path = self._lease_path(key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timeout = (
            self.wait_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        deadline = time.monotonic() + timeout
        thread_lock = _common_thread_lock(path)
        if not thread_lock.acquire(timeout=timeout):
            raise CacheCoordinationTimeout(
                f"timed out waiting for cache flight {key.key_id}"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            while True:
                try:
                    fcntl.flock(
                        descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self._increment("wait_timeouts")
                        raise CacheCoordinationTimeout(
                            f"timed out waiting for cache flight {key.key_id}"
                        )
                    time.sleep(0.01)
            with self._metrics_lock:
                self._active_flights += 1
            yield
        finally:
            if descriptor is not None:
                with self._metrics_lock:
                    self._active_flights = max(0, self._active_flights - 1)
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            thread_lock.release()

    @contextmanager
    def lease(
        self,
        key: SemanticCacheKey | Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[None]:
        """Public keyed lease for native caches that retain their own codec."""

        with self._process_lease(self._coerce_key(key), timeout_seconds):
            yield

    def _remove_invalid(self, path: Path, *, reason: str) -> None:
        try:
            path.unlink()
            self._increment("corruption_recoveries")
        except FileNotFoundError:
            pass
        if reason == "poisoned_entry":
            self._increment("poisoned_rejections")

    def _decode(
        self,
        raw: bytes,
        expected: SemanticCacheKey,
        *,
        payload_validator: Callable[[Any], bool] | None,
    ) -> NamespaceCacheEntry:
        quota = self._quota(expected.namespace)
        if len(raw) > quota.max_entry_bytes:
            raise ValueError("entry exceeds namespace max_entry_bytes")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("cache entry must be an object")
        if payload.get("schema") != COMMON_CACHE_ENTRY_SCHEMA:
            raise ValueError("unsupported common cache entry schema")
        if "is_completion_evidence" in payload:
            claimed = payload.get("is_completion_evidence")
            if not isinstance(claimed, bool):
                raise ValueError("poisoned completion claim")
        key = SemanticCacheKey.from_dict(payload.get("key"))
        if key != expected or payload.get("key_id") != expected.key_id:
            raise ValueError("poisoned semantic key binding")
        metadata_value = payload.get("metadata")
        if not isinstance(metadata_value, Mapping):
            raise ValueError("missing namespace metadata")
        metadata = CacheNamespaceMetadata(
            namespace=CacheNamespace(str(metadata_value.get("namespace") or "")),
            key_schema=str(metadata_value.get("key_schema") or ""),
            entry_schema=str(metadata_value.get("entry_schema") or ""),
            authority=CacheAuthority(str(metadata_value.get("authority") or "")),
            required_dimensions=tuple(
                metadata_value.get("required_dimensions") or ()
            ),
            common_schema=str(metadata_value.get("schema") or ""),
        )
        if metadata.namespace is not expected.namespace:
            raise ValueError("poisoned namespace binding")
        outcome = CacheRecordOutcome.coerce(payload.get("outcome"))
        authority = CacheAuthority(str(payload.get("authority") or ""))
        if authority is not metadata.authority:
            raise ValueError("poisoned authority binding")
        created_at_ms = payload.get("created_at_ms")
        expires_at_ms = payload.get("expires_at_ms")
        if (
            isinstance(created_at_ms, bool)
            or not isinstance(created_at_ms, int)
            or created_at_ms < 0
            or created_at_ms > self._now_ms() + 60_000
        ):
            raise ValueError("poisoned creation timestamp")
        if expires_at_ms is not None and (
            isinstance(expires_at_ms, bool)
            or not isinstance(expires_at_ms, int)
            or expires_at_ms <= created_at_ms
        ):
            raise ValueError("invalid expiry timestamp")
        if not outcome.can_complete and expires_at_ms is None:
            raise ValueError("negative and inconclusive entries require TTL")
        refs_value = payload.get("artifact_references") or ()
        if (
            isinstance(refs_value, (str, bytes, bytearray))
            or not isinstance(refs_value, Sequence)
            or len(refs_value) > quota.max_artifact_references
        ):
            raise ValueError("artifact references exceed namespace bounds")
        refs = tuple(BoundedArtifactReference(item) for item in refs_value)
        if (
            len(_common_json_bytes([item.to_dict() for item in refs]))
            > quota.max_artifact_reference_bytes
        ):
            raise ValueError("artifact references exceed byte bound")
        native_payload = _common_json_value(
            payload.get("payload"), name="cache payload"
        )
        if payload_validator is not None:
            valid = payload_validator(native_payload)
            if not isinstance(valid, bool):
                raise ValueError("payload_validator must return a boolean")
            if not valid:
                raise ValueError("poisoned native payload")
        entry = NamespaceCacheEntry(
            metadata=metadata,
            key=key,
            payload=native_payload,
            outcome=outcome,
            authority=authority,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
            artifact_references=refs,
            payload_schema=str(payload.get("payload_schema") or ""),
            entry_digest=str(payload.get("entry_digest") or ""),
        )
        if entry.entry_digest != entry.computed_digest:
            raise ValueError("cache entry integrity mismatch")
        claimed = payload.get("is_completion_evidence")
        if claimed is not None and claimed != entry.is_completion_evidence:
            raise ValueError("poisoned completion authority")
        return entry

    def lookup(
        self,
        key: SemanticCacheKey | Mapping[str, Any],
        *,
        require_completion_evidence: bool = False,
        payload_validator: Callable[[Any], bool] | None = None,
    ) -> NamespaceCacheLookup:
        semantic_key = self._coerce_key(key)
        self._increment("lookups")
        path = self._entry_path(semantic_key)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            self._increment("misses")
            return NamespaceCacheLookup(
                NamespaceLookupStatus.MISS,
                semantic_key,
                reason_codes=("cache_miss",),
            )
        except OSError:
            self._increment("rejected")
            return NamespaceCacheLookup(
                NamespaceLookupStatus.REJECTED,
                semantic_key,
                reason_codes=("cache_read_error",),
            )
        try:
            entry = self._decode(
                raw, semantic_key, payload_validator=payload_validator
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            self._remove_invalid(path, reason="poisoned_entry")
            self._increment("rejected")
            return NamespaceCacheLookup(
                NamespaceLookupStatus.REJECTED,
                semantic_key,
                reason_codes=("poisoned_or_corrupt_entry",),
            )
        if entry.expires_at_ms is not None and self._now_ms() >= entry.expires_at_ms:
            self._remove_invalid(path, reason="stale_entry")
            self._increment("rejected")
            self._increment("stale_rejections")
            return NamespaceCacheLookup(
                NamespaceLookupStatus.REJECTED,
                semantic_key,
                reason_codes=("stale_entry",),
            )
        if require_completion_evidence and not entry.is_completion_evidence:
            self._increment("rejected")
            return NamespaceCacheLookup(
                NamespaceLookupStatus.REJECTED,
                semantic_key,
                entry=entry,
                reason_codes=("not_completion_evidence",),
            )
        self._increment("hits")
        self._increment("bytes_reused", len(raw))
        return NamespaceCacheLookup(
            NamespaceLookupStatus.HIT,
            semantic_key,
            entry=entry,
            reason_codes=("exact_key_hit",),
        )

    def put(
        self,
        key: SemanticCacheKey | Mapping[str, Any],
        payload: Any,
        *,
        outcome: CacheRecordOutcome | str = CacheRecordOutcome.SUCCESSFUL,
        authority: CacheAuthority | str = CacheAuthority.DIAGNOSTIC,
        ttl_seconds: int | None = None,
        artifact_references: Sequence[
            BoundedArtifactReference | Mapping[str, Any]
        ] = (),
        payload_schema: str | None = None,
        key_schema: str | None = None,
        entry_schema: str | None = None,
        payload_validator: Callable[[Any], bool] | None = None,
    ) -> NamespaceCacheEntry | None:
        semantic_key = self._coerce_key(key)
        record_outcome = CacheRecordOutcome.coerce(outcome)
        record_authority = (
            authority
            if isinstance(authority, CacheAuthority)
            else CacheAuthority(str(authority))
        )
        if (
            semantic_key.namespace is CacheNamespace.PROOF_DRAFT
            and record_authority is not CacheAuthority.DRAFT
        ):
            raise ValueError(
                "proof drafts must use the isolated draft authority namespace"
            )
        quota = self._quota(semantic_key.namespace)
        if not record_outcome.can_complete:
            ttl_seconds = ttl_seconds or quota.negative_ttl_seconds
        if ttl_seconds is not None:
            if (
                isinstance(ttl_seconds, bool)
                or not isinstance(ttl_seconds, int)
                or ttl_seconds < 1
            ):
                raise ValueError("ttl_seconds must be a positive integer or None")
            ttl_seconds = min(ttl_seconds, quota.max_ttl_seconds)
        refs = tuple(
            item
            if isinstance(item, BoundedArtifactReference)
            else BoundedArtifactReference(item)
            for item in artifact_references
        )
        if len(refs) > quota.max_artifact_references:
            self._increment("write_rejections")
            return None
        if (
            len(_common_json_bytes([item.to_dict() for item in refs]))
            > quota.max_artifact_reference_bytes
        ):
            self._increment("write_rejections")
            return None
        native_payload = _common_json_value(payload, name="cache payload")
        if payload_validator is not None:
            valid = payload_validator(native_payload)
            if not isinstance(valid, bool):
                raise ValueError("payload_validator must return a boolean")
            if not valid:
                self._increment("write_rejections")
                return None
        metadata = namespace_metadata(
            semantic_key.namespace,
            authority=record_authority,
            key_schema=key_schema,
            entry_schema=entry_schema,
        )
        now_ms = self._now_ms()
        entry = NamespaceCacheEntry(
            metadata=metadata,
            key=semantic_key,
            payload=native_payload,
            outcome=record_outcome,
            authority=record_authority,
            created_at_ms=now_ms,
            expires_at_ms=(
                now_ms + ttl_seconds * 1000
                if ttl_seconds is not None
                else None
            ),
            artifact_references=refs,
            payload_schema=payload_schema or metadata.entry_schema,
        )
        entry = replace(entry, entry_digest=entry.computed_digest)
        encoded = _common_json_bytes(entry.to_dict()) + b"\n"
        if len(encoded) > quota.max_entry_bytes:
            self._increment("write_rejections")
            return None
        path = self._entry_path(semantic_key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        self._increment("writes")
        self.gc(semantic_key.namespace)
        return entry

    def get_or_compute(
        self,
        key: SemanticCacheKey | Mapping[str, Any],
        producer: Callable[[], Any],
        *,
        outcome: CacheRecordOutcome | str = CacheRecordOutcome.SUCCESSFUL,
        authority: CacheAuthority | str = CacheAuthority.DIAGNOSTIC,
        ttl_seconds: int | None = None,
        artifact_references: Sequence[
            BoundedArtifactReference | Mapping[str, Any]
        ] = (),
        payload_schema: str | None = None,
        key_schema: str | None = None,
        entry_schema: str | None = None,
        require_completion_evidence: bool = False,
        payload_validator: Callable[[Any], bool] | None = None,
        wait_timeout_seconds: float | None = None,
        owner_id: str | None = None,
        lease_seconds: float | None = None,
        outcome_ttl_seconds: float | None = None,
        deadline_monotonic: float | None = None,
        cancel_event: Any = None,
    ) -> NamespaceCoordinationResult:
        if not callable(producer):
            raise ValueError("producer must be callable")
        semantic_key = self._coerce_key(key)
        initial = self.lookup(
            semantic_key,
            require_completion_evidence=require_completion_evidence,
            payload_validator=payload_validator,
        )
        if initial.hit:
            return NamespaceCoordinationResult(initial)
        timeout = (
            self.wait_timeout_seconds
            if wait_timeout_seconds is None
            else float(wait_timeout_seconds)
        )
        member_deadline = time.monotonic() + timeout
        if deadline_monotonic is not None:
            member_deadline = min(member_deadline, float(deadline_monotonic))
        local_state: dict[str, Any] = {}

        def execute_owner() -> dict[str, Any]:
            refreshed = self.lookup(
                semantic_key,
                require_completion_evidence=require_completion_evidence,
                payload_validator=payload_validator,
            )
            if refreshed.hit:
                return {
                    "schema": NAMESPACE_SINGLE_FLIGHT_RESULT_SCHEMA,
                    "key_id": semantic_key.key_id,
                    "produced": False,
                    "entry_digest": (
                        refreshed.entry.entry_digest
                        if refreshed.entry is not None
                        else ""
                    ),
                    "direct_value": None,
                }
            self._increment("leaders")
            produced = producer()
            publication = (
                produced
                if isinstance(produced, CacheWrite)
                else CacheWrite(
                    produced,
                    outcome=CacheRecordOutcome.coerce(outcome),
                    authority=(
                        authority
                        if isinstance(authority, CacheAuthority)
                        else CacheAuthority(str(authority))
                    ),
                    ttl_seconds=ttl_seconds,
                    artifact_references=tuple(
                        item
                        if isinstance(item, BoundedArtifactReference)
                        else BoundedArtifactReference(item)
                        for item in artifact_references
                    ),
                    payload_schema=payload_schema or "",
                )
            )
            local_state["producer_value"] = publication.payload
            if publication.store:
                self.put(
                    semantic_key,
                    publication.payload,
                    outcome=publication.outcome,
                    authority=publication.authority,
                    ttl_seconds=publication.ttl_seconds,
                    artifact_references=publication.artifact_references,
                    payload_schema=publication.payload_schema or payload_schema,
                    key_schema=key_schema,
                    entry_schema=entry_schema,
                    payload_validator=payload_validator,
                )
            final = self.lookup(
                semantic_key,
                require_completion_evidence=require_completion_evidence,
                payload_validator=payload_validator,
            )
            return {
                "schema": NAMESPACE_SINGLE_FLIGHT_RESULT_SCHEMA,
                "key_id": semantic_key.key_id,
                "produced": True,
                "entry_digest": (
                    final.entry.entry_digest if final.entry is not None else ""
                ),
                # A successful exact entry is the rendezvous value.  Direct
                # values are used only for deliberately non-persisted or
                # rejected writes and remain subject to the flight byte bound.
                "direct_value": None if final.hit else publication.payload,
            }

        try:
            coordinated: DistributedSingleFlightResult = (
                self.single_flight_coordinator.coordinate(
                    semantic_key,
                    execute_owner,
                    owner_id=owner_id,
                    lease_seconds=(
                        self.lease_seconds
                        if lease_seconds is None
                        else lease_seconds
                    ),
                    timeout_seconds=timeout,
                    deadline_monotonic=member_deadline,
                    cancel_event=cancel_event,
                    outcome_ttl_seconds=(
                        self.outcome_ttl_seconds
                        if outcome_ttl_seconds is None
                        else outcome_ttl_seconds
                    ),
                )
            )
        except DistributedSingleFlightTimeout as exc:
            self._increment("wait_timeouts")
            raise CacheCoordinationTimeout(str(exc)) from exc
        except DistributedSingleFlightCancelled:
            # Cancellation is intentionally not translated to a timeout and
            # never mutates another member's lease or published outcome.
            raise

        flight_value = coordinated.value
        if (
            not isinstance(flight_value, Mapping)
            or flight_value.get("schema")
            != NAMESPACE_SINGLE_FLIGHT_RESULT_SCHEMA
            or flight_value.get("key_id") != semantic_key.key_id
            or not isinstance(flight_value.get("produced"), bool)
        ):
            raise CacheCoordinationError(
                "single-flight outcome is not bound to the semantic cache key"
            )
        direct_value = flight_value.get("direct_value")
        if direct_value is not None and payload_validator is not None:
            accepted = payload_validator(direct_value)
            if not isinstance(accepted, bool):
                raise ValueError("payload_validator must return a boolean")
            if not accepted:
                raise CacheCoordinationError(
                    "shared producer value failed this member's validator"
                )
        final = self.lookup(
            semantic_key,
            require_completion_evidence=require_completion_evidence,
            payload_validator=payload_validator,
        )
        if (
            not final.hit
            and direct_value is None
            and bool(flight_value.get("entry_digest"))
            and set(final.reason_codes).intersection(
                {
                    "cache_miss",
                    "stale_entry",
                    "poisoned_or_corrupt_entry",
                    "cache_read_error",
                }
            )
        ):
            # The outcome is only a rendezvous receipt, not cache authority.
            # If its referenced record is stale or invalid, retire exactly
            # that fence and compete for a fresh generation.  Never project
            # the old authoritative payload as a hit.
            self.single_flight_coordinator.discard_outcome(
                semantic_key,
                fencing_token=coordinated.fencing_token,
            )
            remaining = member_deadline - time.monotonic()
            if remaining <= 0:
                self._increment("wait_timeouts")
                raise CacheCoordinationTimeout(
                    f"timed out refreshing cache flight {semantic_key.key_id}"
                )
            return self.get_or_compute(
                semantic_key,
                producer,
                outcome=outcome,
                authority=authority,
                ttl_seconds=ttl_seconds,
                artifact_references=artifact_references,
                payload_schema=payload_schema,
                key_schema=key_schema,
                entry_schema=entry_schema,
                require_completion_evidence=require_completion_evidence,
                payload_validator=payload_validator,
                wait_timeout_seconds=remaining,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
                outcome_ttl_seconds=outcome_ttl_seconds,
                deadline_monotonic=member_deadline,
                cancel_event=cancel_event,
            )
        produced_here = bool(
            coordinated.owner and flight_value.get("produced")
        )
        shared = not produced_here
        if shared:
            self._increment("followers")
        producer_value = (
            local_state.get("producer_value")
            if produced_here
            else direct_value
        )
        return NamespaceCoordinationResult(
            final,
            produced=produced_here,
            shared=shared,
            producer_value=producer_value,
            fencing_token=coordinated.fencing_token,
            flight_owner_id=coordinated.outcome.owner_id,
            outcome_digest=coordinated.outcome.outcome_digest,
            attestation=coordinated.attestation,
            single_flight_outcome=coordinated.outcome,
        )

    # Conventional coordination spellings.
    coordinate = get_or_compute
    run = get_or_compute
    single_flight = get_or_compute

    def gc(
        self, namespace: CacheNamespace | str | None = None
    ) -> dict[str, int]:
        """Remove invalid/expired records then enforce count and byte quotas."""

        namespaces = (
            tuple(CacheNamespace)
            if namespace is None
            else (
                namespace
                if isinstance(namespace, CacheNamespace)
                else CacheNamespace(str(namespace)),
            )
        )
        removed = 0
        reclaimed = 0
        now_ms = self._now_ms()
        for kind in namespaces:
            root = self.entries_path / kind.value
            candidates: list[tuple[int, bool, Path, int]] = []
            quota = self._quota(kind)
            for path in root.glob("*/*.json") if root.exists() else ():
                try:
                    raw = path.read_bytes()
                    payload = json.loads(raw)
                    key = SemanticCacheKey.from_dict(payload.get("key"))
                    entry = self._decode(raw, key, payload_validator=None)
                    if (
                        key.namespace is not kind
                        or path.stem != key.digest
                        or (
                            entry.expires_at_ms is not None
                            and now_ms >= entry.expires_at_ms
                        )
                    ):
                        raise ValueError("invalid, misplaced, or expired entry")
                    candidates.append(
                        (
                            entry.created_at_ms,
                            entry.is_completion_evidence,
                            path,
                            len(raw),
                        )
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    try:
                        size = path.stat().st_size
                        path.unlink()
                        removed += 1
                        reclaimed += size
                    except OSError:
                        pass
            total_bytes = sum(item[3] for item in candidates)
            # Evict oldest non-authoritative records first, then authoritative.
            candidates.sort(key=lambda item: (item[1], item[0], str(item[2])))
            while (
                len(candidates) > quota.max_entries
                or total_bytes > quota.max_bytes
            ):
                _, _, path, size = candidates.pop(0)
                try:
                    path.unlink()
                    removed += 1
                    reclaimed += size
                    total_bytes -= size
                except OSError:
                    pass
        self._increment("gc_runs")
        self._increment("evictions", removed)
        return {"removed": removed, "reclaimed_bytes": reclaimed}

    prune = gc

    def clear(self, namespace: CacheNamespace | str | None = None) -> int:
        kinds = (
            tuple(CacheNamespace)
            if namespace is None
            else (
                namespace
                if isinstance(namespace, CacheNamespace)
                else CacheNamespace(str(namespace)),
            )
        )
        removed = 0
        for kind in kinds:
            root = self.entries_path / kind.value
            for path in root.glob("*/*.json") if root.exists() else ():
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def metrics(
        self, namespace: CacheNamespace | str | None = None
    ) -> NamespaceCacheMetrics:
        roots = (
            tuple(CacheNamespace)
            if namespace is None
            else (
                namespace
                if isinstance(namespace, CacheNamespace)
                else CacheNamespace(str(namespace)),
            )
        )
        entries = 0
        byte_count = 0
        for kind in roots:
            root = self.entries_path / kind.value
            for path in root.glob("*/*.json") if root.exists() else ():
                try:
                    byte_count += path.stat().st_size
                    entries += 1
                except OSError:
                    pass
        distributed_active = (
            self.single_flight_coordinator.active_lease_count(
                namespace=(
                    None
                    if namespace is None
                    else (
                        namespace.value
                        if isinstance(namespace, CacheNamespace)
                        else str(namespace)
                    )
                )
            )
        )
        with self._metrics_lock:
            return NamespaceCacheMetrics(
                **self._metric_values,
                active_flights=self._active_flights + distributed_active,
                entries=entries,
                bytes=byte_count,
            )

    stats = metrics
    metrics_snapshot = metrics

    def reset_metrics(self) -> NamespaceCacheMetrics:
        """Atomically reset counters and return the replaced snapshot."""

        previous = self.metrics()
        with self._metrics_lock:
            for name in self._metric_values:
                self._metric_values[name] = 0
        return previous


CacheCASBindingFactory = Callable[[Mapping[str, Any]], Any]


class _CacheCASAdapter:
    """Shared, deliberately narrow bridge from native caches into RuntimeCAS.

    Runtime CAS imports stay local to the methods that need their enum types.
    This keeps the historic cache modules usable when the optional tiered
    runtime store is not imported and avoids a module-level dependency cycle.
    """

    def __init__(
        self,
        cache: Any,
        runtime_cas: Any,
        *,
        producer_version: str,
        policy_version: str,
        capability_version: str,
        binding_factory: CacheCASBindingFactory | None = None,
    ) -> None:
        if not callable(getattr(runtime_cas, "put", None)):
            raise ValueError("runtime_cas must provide a callable put method")
        if not callable(getattr(runtime_cas, "get", None)):
            raise ValueError("runtime_cas must provide a callable get method")
        if binding_factory is not None and not callable(binding_factory):
            raise ValueError("binding_factory must be callable or None")
        self.cache = cache
        self.runtime_cas = runtime_cas
        self.producer_version = _required_text(
            producer_version, "producer_version"
        )
        self.policy_version = _required_text(
            policy_version, "policy_version"
        )
        self.capability_version = _required_text(
            capability_version, "capability_version"
        )
        self.binding_factory = binding_factory

    def _binding(
        self,
        *,
        namespace: str,
        semantic_key_id: str,
        semantic_dimensions: Mapping[str, Any],
        binding: Any,
    ) -> Any:
        descriptor = {
            "schema": CACHE_CAS_ADAPTER_BINDING_SCHEMA,
            "namespace": namespace,
            "semantic_key_id": semantic_key_id,
            "semantic_dimensions": _common_json_value(
                semantic_dimensions, name="semantic dimensions"
            ),
            "producer_version": self.producer_version,
            "policy_version": self.policy_version,
            "capability_version": self.capability_version,
        }
        if binding is None:
            if self.binding_factory is None:
                raise ValueError(
                    "binding is required when no binding_factory is configured"
                )
            binding = self.binding_factory(descriptor)
            if binding is None:
                raise ValueError(
                    "binding_factory must return a ResultBinding"
                )
        from ..self_improvement.supervisor_v2_contracts import ResultBinding

        if isinstance(binding, ResultBinding):
            result = binding
        elif isinstance(binding, Mapping):
            result = ResultBinding.from_dict(binding)
        else:
            raise ValueError(
                "binding must be a ResultBinding or canonical mapping"
            )
        expected_revisions = (
            (
                "producer_revision",
                result.producer_revision,
                self.producer_version,
            ),
            (
                "policy_revision",
                result.policy_revision,
                self.policy_version,
            ),
            (
                "capability_revision",
                result.capability_revision,
                self.capability_version,
            ),
        )
        mismatches = [
            name
            for name, actual, expected in expected_revisions
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "runtime binding revision mismatch: "
                + ", ".join(mismatches)
            )
        return result

    def _remaining_ttl_seconds(self, expires_at_ms: int | None) -> int | None:
        if expires_at_ms is None:
            return None
        now = getattr(self.cache, "_now_ms", None)
        now_ms = int(now()) if callable(now) else int(time.time() * 1000)
        remaining_ms = expires_at_ms - now_ms
        if remaining_ms <= 0:
            return 0
        # Round down so importing a native entry can only preserve or shorten
        # its lifetime; promotion must never refresh freshness.
        return remaining_ms // 1000

    @staticmethod
    def _runtime_authority(authority: CacheAuthority) -> Any:
        from ..runtime.runtime_cas import RuntimeAuthority

        return RuntimeAuthority(authority.value)

    @staticmethod
    def _freshness(value: Any) -> Any:
        if value is not None:
            return value
        from ..runtime.runtime_cas import EvidenceFreshness

        return EvidenceFreshness.FRESH

    def get(
        self,
        artifact_id: str,
        *,
        expected_namespace: str | None = None,
        expected_authority: Any = None,
        require_fresh: bool = False,
    ) -> Any:
        """Read one imported record through RuntimeCAS's verification path."""

        if isinstance(expected_authority, CacheAuthority):
            expected_authority = self._runtime_authority(expected_authority)
        return self.runtime_cas.get(
            artifact_id,
            expected_namespace=expected_namespace,
            expected_authority=expected_authority,
            require_fresh=require_fresh,
        )

    def _put_runtime_record(
        self,
        payload: Mapping[str, Any],
        *,
        binding: Any,
        namespace: str,
        authority: CacheAuthority,
        dependencies: Sequence[Any],
        freshness: Any,
        ttl_seconds: int | None,
        tiers: Sequence[Any] | None,
        projection_key: str | None,
        artifact_kind: str,
    ) -> Any:
        from ..runtime.runtime_cas import RuntimeArtifactRecord

        result = self.runtime_cas.put(
            payload,
            binding=binding,
            namespace=namespace,
            artifact_kind=artifact_kind,
            authority=self._runtime_authority(authority),
            dependencies=tuple(dependencies),
            freshness=self._freshness(freshness),
            ttl_seconds=ttl_seconds,
            tiers=tiers,
            payload_schema=CACHE_CAS_ADAPTER_PAYLOAD_SCHEMA,
            projection_key=projection_key,
        )
        if not isinstance(result, RuntimeArtifactRecord):
            raise CacheCoordinationError(
                "runtime_cas.put must return a RuntimeArtifactRecord"
            )
        return result


class NamespaceCacheCASAdapter(_CacheCASAdapter):
    """Import verified exact NamespaceCacheCoordinator entries into RuntimeCAS.

    The adapter never changes the existing coordinator's lookup, persistence,
    metrics, or single-flight behavior.  It is an explicit migration/reuse
    path: a runtime caller first attempts its CAS identity and may invoke
    :meth:`import_entry` on a miss to reuse a native exact-key record.
    """

    def __init__(
        self,
        cache: NamespaceCacheCoordinator,
        runtime_cas: Any,
        *,
        producer_version: str,
        policy_version: str,
        capability_version: str,
        binding_factory: CacheCASBindingFactory | None = None,
    ) -> None:
        if not isinstance(cache, NamespaceCacheCoordinator):
            raise ValueError(
                "cache must be a NamespaceCacheCoordinator"
            )
        super().__init__(
            cache,
            runtime_cas,
            producer_version=producer_version,
            policy_version=policy_version,
            capability_version=capability_version,
            binding_factory=binding_factory,
        )

    def import_entry(
        self,
        key: SemanticCacheKey | Mapping[str, Any],
        *,
        require_completion_evidence: bool = False,
        payload_validator: Callable[[Any], bool] | None = None,
        dependencies: Sequence[Any] = (),
        freshness: Any = None,
        tiers: Sequence[Any] | None = None,
        projection_key: str | None = None,
        project_authoritative: bool = True,
        artifact_kind: str = "namespace_cache_entry",
        binding: Any = None,
    ) -> Any:
        """Import a current exact native hit, returning its runtime record.

        Misses, stale/corrupt entries, failed native validators, and
        non-completion records requested as completion evidence return
        ``None``.  RuntimeCAS remains responsible for dependency verification,
        cycle rejection, tier publication, and immutable identity validation.
        """

        semantic_key = self.cache._coerce_key(key)
        lookup = self.cache.lookup(
            semantic_key,
            require_completion_evidence=require_completion_evidence,
            payload_validator=payload_validator,
        )
        if (
            lookup.status is not NamespaceLookupStatus.HIT
            or lookup.entry is None
            or lookup.key != semantic_key
            or lookup.entry.key != semantic_key
        ):
            return None
        entry = lookup.entry
        metadata = entry.metadata
        canonical_metadata = namespace_metadata(
            semantic_key.namespace,
            authority=entry.authority,
            key_schema=metadata.key_schema,
            entry_schema=metadata.entry_schema,
        )
        if (
            metadata.namespace is not semantic_key.namespace
            or metadata.authority is not entry.authority
            or metadata.required_dimensions
            != canonical_metadata.required_dimensions
            or not set(canonical_metadata.required_dimensions).issubset(
                semantic_key.dimensions
            )
        ):
            return None
        if semantic_key.namespace is CacheNamespace.PROOF_DRAFT:
            if entry.authority is not CacheAuthority.DRAFT:
                return None
            if projection_key is not None:
                raise ValueError(
                    "proof drafts cannot create authoritative projections"
                )
        remaining_ttl = self._remaining_ttl_seconds(entry.expires_at_ms)
        if remaining_ttl == 0:
            return None
        if not entry.is_completion_evidence:
            if projection_key is not None:
                raise ValueError(
                    "non-completion cache entries cannot create projections"
                )
            project_authoritative = False
        if project_authoritative and projection_key is None:
            projection_key = semantic_key.key_id
        runtime_binding = self._binding(
            namespace=semantic_key.namespace.value,
            semantic_key_id=semantic_key.key_id,
            semantic_dimensions=semantic_key.dimensions,
            binding=binding,
        )
        payload = {
            "schema": CACHE_CAS_ADAPTER_PAYLOAD_SCHEMA,
            "native_schema": metadata.entry_schema,
            "payload_schema": entry.payload_schema or metadata.entry_schema,
            "semantic_key": semantic_key.to_dict(),
            "legacy_entry_digest": entry.entry_digest,
            "outcome": entry.outcome.value,
            "authority": entry.authority.value,
            "created_at_ms": entry.created_at_ms,
            "expires_at_ms": entry.expires_at_ms,
            "artifact_references": [
                item.to_dict() for item in entry.artifact_references
            ],
            "payload": entry.payload,
        }
        return self._put_runtime_record(
            payload,
            binding=runtime_binding,
            namespace=semantic_key.namespace.value,
            authority=entry.authority,
            dependencies=dependencies,
            freshness=freshness,
            ttl_seconds=remaining_ttl,
            tiers=tiers,
            projection_key=projection_key,
            artifact_kind=artifact_kind,
        )

    import_exact_entry = import_entry
    reuse_exact = import_entry


class AnalysisCacheCASAdapter(_CacheCASAdapter):
    """Import exact native AnalysisCache receipts without changing its API."""

    def __init__(
        self,
        cache: AnalysisCache,
        runtime_cas: Any,
        *,
        producer_version: str,
        policy_version: str,
        capability_version: str,
        binding_factory: CacheCASBindingFactory | None = None,
    ) -> None:
        if not isinstance(cache, AnalysisCache):
            raise ValueError("cache must be an AnalysisCache")
        super().__init__(
            cache,
            runtime_cas,
            producer_version=producer_version,
            policy_version=policy_version,
            capability_version=capability_version,
            binding_factory=binding_factory,
        )

    def import_entry(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        *,
        require_completion_evidence: bool = False,
        completion_validator: CompletionValidator | None = None,
        dependencies: Sequence[Any] = (),
        freshness: Any = None,
        tiers: Sequence[Any] | None = None,
        projection_key: str | None = None,
        project_authoritative: bool = True,
        artifact_kind: str = "analysis_cache_entry",
        binding: Any = None,
    ) -> Any:
        """Import only an exact, fresh analysis receipt accepted by its gate."""

        if completion_validator is not None and not callable(
            completion_validator
        ):
            raise ValueError("completion_validator must be callable or None")
        cache_key = self.cache._coerce_key(key)
        lookup = self.cache.lookup(
            cache_key,
            require_completion_evidence=require_completion_evidence,
        )
        if (
            lookup.status is not AnalysisCacheLookupStatus.HIT
            or lookup.entry is None
            or lookup.key != cache_key
            or lookup.entry.key != cache_key
        ):
            return None
        if completion_validator is not None and lookup.is_completion_evidence:
            accepted = completion_validator(lookup)
            if not isinstance(accepted, bool):
                raise CacheCoordinationError(
                    "completion_validator must return a boolean"
                )
            if not accepted:
                return None
        entry = lookup.entry
        authority = (
            CacheAuthority.AUTHORITATIVE
            if entry.is_completion_evidence
            else CacheAuthority.DIAGNOSTIC
        )
        remaining_ttl = self._remaining_ttl_seconds(entry.expires_at_ms)
        if remaining_ttl == 0:
            return None
        if not entry.is_completion_evidence:
            if projection_key is not None:
                raise ValueError(
                    "non-completion analysis entries cannot create projections"
                )
            project_authoritative = False
        if project_authoritative and projection_key is None:
            projection_key = cache_key.key_id
        dimensions = entry.key.to_dict()
        dimensions.pop("key_id", None)
        runtime_binding = self._binding(
            namespace=CacheNamespace.ANALYSIS.value,
            semantic_key_id=cache_key.key_id,
            semantic_dimensions=dimensions,
            binding=binding,
        )
        payload = {
            "schema": CACHE_CAS_ADAPTER_PAYLOAD_SCHEMA,
            "native_schema": ANALYSIS_CACHE_ENTRY_SCHEMA,
            "payload_schema": ANALYSIS_CACHE_ENTRY_SCHEMA,
            "semantic_key": cache_key.to_dict(),
            "legacy_entry_digest": entry.entry_digest,
            "outcome": entry.status.value,
            "authority": authority.value,
            "created_at_ms": entry.created_at_ms,
            "expires_at_ms": entry.expires_at_ms,
            "receipt": entry.receipt,
        }
        return self._put_runtime_record(
            payload,
            binding=runtime_binding,
            namespace=CacheNamespace.ANALYSIS.value,
            authority=authority,
            dependencies=dependencies,
            freshness=freshness,
            ttl_seconds=remaining_ttl,
            tiers=tiers,
            projection_key=projection_key,
            artifact_kind=artifact_kind,
        )

    import_exact_entry = import_entry
    reuse_exact = import_entry


class CacheCoordinator:
    """Compatibility facade for the common and legacy analysis coordinators.

    Passing an :class:`AnalysisCache` retains the historic analysis API.
    Passing a directory (or ``None``) selects the common namespace API.
    """

    def __init__(self, cache_or_path: Any = None, **kwargs: Any) -> None:
        if cache_or_path is None and "path" in kwargs:
            cache_or_path = kwargs.pop("path")
        if isinstance(cache_or_path, AnalysisCache):
            self._delegate: Any = AnalysisCacheCoordinator(
                cache_or_path, **kwargs
            )
        else:
            self._delegate = NamespaceCacheCoordinator(
                cache_or_path, **kwargs
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


SingleFlightCacheCoordinator = AnalysisCacheCoordinator
CommonCacheCoordinator = NamespaceCacheCoordinator
CommonCacheEntry = NamespaceCacheEntry
CommonCacheKey = SemanticCacheKey
CommonCacheLookup = NamespaceCacheLookup
CommonCacheMetrics = NamespaceCacheMetrics
NamespaceRuntimeCASAdapter = NamespaceCacheCASAdapter
RuntimeCASNamespaceCacheAdapter = NamespaceCacheCASAdapter
AnalysisRuntimeCASAdapter = AnalysisCacheCASAdapter
RuntimeCASAnalysisCacheAdapter = AnalysisCacheCASAdapter
CacheOutcome = CacheRecordOutcome
CacheLookupStatus = NamespaceLookupStatus


__all__ = [
    "AnalysisCacheCASAdapter",
    "AnalysisCacheCoordinator",
    "AnalysisRuntimeCASAdapter",
    "ArtifactReference",
    "BoundedArtifactReference",
    "CACHE_CAS_ADAPTER_BINDING_SCHEMA",
    "CACHE_CAS_ADAPTER_PAYLOAD_SCHEMA",
    "COMMON_CACHE_ENTRY_SCHEMA",
    "COMMON_CACHE_KEY_SCHEMA",
    "COMMON_CACHE_NAMESPACE_SCHEMA",
    "CONCURRENT_IDENTICAL_MISS_COLLAPSE_REQUIREMENT_ID",
    "CacheAuthority",
    "CacheCASBindingFactory",
    "CacheLookupStatus",
    "CacheOutcome",
    "CacheCoordinationError",
    "CacheCoordinationResult",
    "CacheCoordinationStatus",
    "CacheCoordinationTimeout",
    "CacheNamespace",
    "CacheNamespaceMetadata",
    "CachePublication",
    "CacheQuotaPolicy",
    "CacheRecordOutcome",
    "CacheWrite",
    "CacheCoordinator",
    "CacheCoordinatorMetrics",
    "CacheCoordinatorResult",
    "CacheCoordinatorStatus",
    "CacheProducerResultError",
    "CompletionValidator",
    "CommonCacheCoordinator",
    "CommonCacheEntry",
    "CommonCacheKey",
    "CommonCacheLookup",
    "CommonCacheMetrics",
    "CoordinatorMetrics",
    "CoordinatorResult",
    "CoordinatorStatus",
    "INTEGRATED_ANALYSIS_CACHE_ACCEPTANCE_CRITERIA",
    "NamespaceCacheCoordinator",
    "NamespaceCacheCASAdapter",
    "NamespaceCacheEntry",
    "NamespaceCacheLookup",
    "NamespaceCacheMetrics",
    "NamespaceCoordinationResult",
    "NamespaceLookupStatus",
    "NamespaceRuntimeCASAdapter",
    "RuntimeCASAnalysisCacheAdapter",
    "RuntimeCASNamespaceCacheAdapter",
    "SINGLE_FLIGHT_COLLAPSE_EVIDENCE_SCHEMA",
    "SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID",
    "SemanticCacheKey",
    "SingleFlightCollapseEvidence",
    "SingleFlightCacheCoordinator",
    "build_semantic_cache_key",
    "build_semantic_key",
    "build_namespace_semantic_key",
    "namespace_metadata",
]
