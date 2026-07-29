"""Durable, fenced usage-ledger store backends.

``UsageLedgerStore`` is the sole real-time compare-and-set authority for
endpoint usage reservations. Content-addressed/IPFS records may mirror events
for audit and recovery evidence, but eventual IPFS replication alone cannot
authorize admission.

Backends:

* :class:`InMemoryUsageLedgerStore` — process-local with an injectable
  :class:`FakeClock` for deterministic tests;
* :class:`DurableUsageLedgerStore` — single-host transactional file store
  with exclusive ``fcntl`` locking and atomic replace;
* :class:`PartitionedUsageLedgerStore` — explicit conservative per-node
  capacity partitions when a global fenced backend is unavailable.

Distributed multi-host admission requires a strongly consistent fenced backend
or explicit partitions; split writers and stale fences fail closed.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from .identity import canonical_json, content_cid, stable_id
from .schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


ATOMIC_USAGE_LEDGER_REQUIREMENT_ID = "requirement:atomic-usage-ledger.v1"
STORE_SCHEMA = "ipfs_accelerate_py.endpoint_usage.ledger-store@1"
STORE_SCHEMA_VERSION = "1.0"
SUPPORTED_STORE_SCHEMA_VERSIONS = frozenset((STORE_SCHEMA_VERSION,))

DEFAULT_MAX_EVENTS = 100_000
DEFAULT_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_RESERVATIONS = 10_000
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Errors (fail closed)
# ---------------------------------------------------------------------------


class LedgerStoreError(RuntimeError):
    """Base error for usage ledger store failures."""

    code = "ledger_store_error"


class CorruptionError(LedgerStoreError):
    """Persisted state failed integrity or parse checks."""

    code = "ledger_corruption"


class SchemaDriftError(LedgerStoreError):
    """Unsupported or drifted store/document schema."""

    code = "ledger_schema_drift"


class SplitWriterError(LedgerStoreError):
    """Another writer holds the exclusive fence/token."""

    code = "ledger_split_writer"


class StaleFenceError(LedgerStoreError):
    """Caller fence is older than the store fence."""

    code = "ledger_stale_fence"


class StoreExhaustedError(LedgerStoreError):
    """Store retention or size limits would be exceeded."""

    code = "ledger_store_exhausted"


class MigrationError(LedgerStoreError):
    """Schema migration is unsupported or unsafe."""

    code = "ledger_migration_failed"


class CompareAndSetConflict(LedgerStoreError):
    """Optimistic concurrency conflict on revision."""

    code = "ledger_cas_conflict"


class AdmissionAuthorityError(LedgerStoreError):
    """Backend is not authorized to grant capacity (e.g. IPFS-only)."""

    code = "ledger_admission_authority"


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class Clock(Protocol):
    """Clock used by ledger operations for deterministic windows/TTL."""

    def now(self) -> datetime:
        """Return the current UTC-aware instant."""


class SystemClock:
    """Wall-clock UTC."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FakeClock:
    """Injectable monotonic-capable clock for deterministic tests.

    Clock changes (set/advance) are explicit and deterministic: they never
    invent jumps. Negative advances fail closed.
    """

    def __init__(self, start: Optional[datetime] = None) -> None:
        if start is None:
            start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("FakeClock start must be timezone-aware")
        self._now = start.astimezone(timezone.utc)
        self._lock = threading.RLock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def set(self, when: datetime) -> datetime:
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("FakeClock.set requires timezone-aware datetime")
        with self._lock:
            self._now = when.astimezone(timezone.utc)
            return self._now

    def advance(self, *, milliseconds: int = 0, seconds: float = 0.0) -> datetime:
        if milliseconds < 0 or seconds < 0:
            raise ValueError("FakeClock.advance cannot move backwards; use set()")
        delta = timedelta(milliseconds=milliseconds, seconds=seconds)
        with self._lock:
            self._now = self._now + delta
            return self._now

    def to_rfc3339(self) -> str:
        return _to_rfc3339(self.now())


def _to_rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_rfc3339(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def datetime_to_ms(value: datetime) -> int:
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def rfc3339_to_ms(value: str) -> int:
    return datetime_to_ms(parse_rfc3339(value))


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------


def empty_ledger_document(
    *,
    writer_id: Optional[str] = None,
    fence: int = 0,
) -> Dict[str, Any]:
    """Return a fresh, empty ledger document."""

    return {
        "schema": STORE_SCHEMA,
        "schema_version": STORE_SCHEMA_VERSION,
        "contract_schema_version": SCHEMA_VERSION,
        "revision": 0,
        "fence": int(fence),
        "writer_id": writer_id,
        "next_sequence": 1,
        "compacted_through": 0,
        "checkpoint": None,
        "events": [],
        "limits": {},  # scope_id -> list[limit dict]
        "caller_budgets": {},  # scope_id -> list[budget vector entry dicts] or limit-like
        "reservations": {},  # reservation_id -> record
        "idempotency": {},  # key -> decision record
        "stream_settled": {},  # reservation_id -> {dim_key: amount}
        "batch_charges": {},  # batch_id -> {overhead_charged: bool, members: {id: bool}}
        "corrections": {},  # event_id -> correction event_id
        "cooldown_until": {},  # scope_id -> rfc3339
        "disabled_scopes": {},  # scope_id -> reason
        "partition": None,  # optional partition config
        "metadata": {},
    }


def _deep_copy_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(document))


def validate_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a ledger document; fail closed on drift/corruption."""

    if not isinstance(document, Mapping):
        raise CorruptionError("ledger document must be an object")
    schema = document.get("schema")
    if schema != STORE_SCHEMA:
        raise SchemaDriftError(
            "unsupported ledger store schema: %r (expected %s)" % (schema, STORE_SCHEMA)
        )
    version = document.get("schema_version")
    if version not in SUPPORTED_STORE_SCHEMA_VERSIONS:
        raise SchemaDriftError(
            "unsupported ledger store schema_version: %r" % (version,)
        )
    contract = document.get("contract_schema_version", SCHEMA_VERSION)
    if contract not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaDriftError(
            "unsupported endpoint usage contract schema_version: %r" % (contract,)
        )
    for key in (
        "revision",
        "fence",
        "next_sequence",
        "compacted_through",
    ):
        value = document.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CorruptionError("%s must be a non-negative integer" % key)
    if not isinstance(document.get("events", []), list):
        raise CorruptionError("events must be an array")
    for map_key in (
        "limits",
        "caller_budgets",
        "reservations",
        "idempotency",
        "stream_settled",
        "batch_charges",
        "corrections",
        "cooldown_until",
        "disabled_scopes",
        "metadata",
    ):
        value = document.get(map_key, {})
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise CorruptionError("%s must be an object" % map_key)
    # Round-trip through canonical JSON to reject non-JSON-safe values.
    try:
        encoded = canonical_json(dict(document))
        decoded = json.loads(encoded)
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise CorruptionError("ledger document is not canonically serializable") from exc
    if not isinstance(decoded, dict):
        raise CorruptionError("ledger document round-trip lost object shape")
    return decoded


def document_cid(document: Mapping[str, Any]) -> str:
    return content_cid(dict(document))


def idempotency_index_key(
    *,
    scope_id: str,
    request_id: str,
    attempt_id: str,
    idempotency_key: str,
) -> str:
    """Stable key for replay of the same request/attempt/idempotency tuple."""

    return stable_id(
        "uidem",
        {
            "scope_id": scope_id,
            "request_id": request_id,
            "attempt_id": attempt_id,
            "idempotency_key": idempotency_key,
        },
    )


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class UsageLedgerStore(Protocol):
    """Compare-and-set authority for the usage ledger.

    Implementations must serialize mutations so one CAS transaction observes a
    consistent view of limits, windows, reservations, and fences.
    """

    @property
    def clock(self) -> Clock:
        """Clock bound to this store."""

    @property
    def authorizes_admission(self) -> bool:
        """Whether this store may grant capacity."""

    def read(self) -> Dict[str, Any]:
        """Return a deep copy of the current document."""

    def compare_and_set(
        self,
        expected_revision: int,
        document: Mapping[str, Any],
        *,
        writer_id: Optional[str] = None,
        fence: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Atomically replace the document when revision matches.

        Returns the committed document (deep copy). Raises
        :class:`CompareAndSetConflict` on revision mismatch,
        :class:`StaleFenceError` / :class:`SplitWriterError` on fence/writer
        conflicts, and other store errors fail closed.
        """

    def checkpoint(self) -> Dict[str, Any]:
        """Return a durable checkpoint receipt for the current revision."""

    def close(self) -> None:
        """Release resources."""


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryUsageLedgerStore:
    """Process-local transactional store with optional max capacity bounds."""

    authorizes_admission = True

    def __init__(
        self,
        *,
        clock: Optional[Clock] = None,
        writer_id: Optional[str] = None,
        fence: int = 1,
        max_events: int = DEFAULT_MAX_EVENTS,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        max_reservations: int = DEFAULT_MAX_RESERVATIONS,
    ) -> None:
        self._clock: Clock = clock if clock is not None else FakeClock()
        self._lock = threading.RLock()
        self._max_events = int(max_events)
        self._max_document_bytes = int(max_document_bytes)
        self._max_reservations = int(max_reservations)
        self._document = empty_ledger_document(writer_id=writer_id, fence=fence)
        if writer_id is not None:
            self._document["writer_id"] = writer_id
            self._document["fence"] = int(fence)

    @property
    def clock(self) -> Clock:
        return self._clock

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return _deep_copy_document(self._document)

    def compare_and_set(
        self,
        expected_revision: int,
        document: Mapping[str, Any],
        *,
        writer_id: Optional[str] = None,
        fence: Optional[int] = None,
    ) -> Dict[str, Any]:
        validated = validate_document(document)
        with self._lock:
            current = self._document
            if int(expected_revision) != int(current["revision"]):
                raise CompareAndSetConflict(
                    "expected revision %s but store is at %s"
                    % (expected_revision, current["revision"])
                )
            self._check_writer_fence(current, writer_id=writer_id, fence=fence)
            self._check_capacity(validated)
            if writer_id is not None:
                validated["writer_id"] = writer_id
            if fence is not None:
                if int(fence) < int(current["fence"]):
                    raise StaleFenceError(
                        "caller fence %s is stale (store fence %s)"
                        % (fence, current["fence"])
                    )
                validated["fence"] = int(fence)
            else:
                validated["fence"] = int(current["fence"])
            validated["revision"] = int(current["revision"]) + 1
            # Preserve schema markers.
            validated["schema"] = STORE_SCHEMA
            validated["schema_version"] = STORE_SCHEMA_VERSION
            validated = validate_document(validated)
            self._document = validated
            return _deep_copy_document(self._document)

    def bump_fence(self, *, writer_id: str) -> int:
        """Advance the writer fence (e.g. after crash recovery takeover)."""

        with self._lock:
            doc = _deep_copy_document(self._document)
            doc["fence"] = int(doc["fence"]) + 1
            doc["writer_id"] = writer_id
            doc["revision"] = int(doc["revision"]) + 1
            self._document = validate_document(doc)
            return int(self._document["fence"])

    def checkpoint(self) -> Dict[str, Any]:
        with self._lock:
            doc = self._document
            return {
                "schema": "ipfs_accelerate_py.endpoint_usage.checkpoint@1",
                "revision": doc["revision"],
                "fence": doc["fence"],
                "writer_id": doc.get("writer_id"),
                "next_sequence": doc["next_sequence"],
                "compacted_through": doc["compacted_through"],
                "document_cid": document_cid(doc),
                "observed_at": _to_rfc3339(self._clock.now()),
            }

    def close(self) -> None:
        return None

    def _check_writer_fence(
        self,
        current: Mapping[str, Any],
        *,
        writer_id: Optional[str],
        fence: Optional[int],
    ) -> None:
        current_writer = current.get("writer_id")
        current_fence = int(current.get("fence") or 0)
        if fence is not None and int(fence) < current_fence:
            raise StaleFenceError(
                "caller fence %s is stale (store fence %s)" % (fence, current_fence)
            )
        if (
            writer_id is not None
            and current_writer is not None
            and writer_id != current_writer
            and fence is not None
            and int(fence) <= current_fence
        ):
            # A different writer may only proceed with a strictly greater fence
            # (takeover). Equal fence + different writer is a split-writer.
            raise SplitWriterError(
                "writer %r conflicts with active writer %r at fence %s"
                % (writer_id, current_writer, current_fence)
            )

    def _check_capacity(self, document: Mapping[str, Any]) -> None:
        events = document.get("events") or []
        if len(events) > self._max_events:
            raise StoreExhaustedError(
                "event log exceeds max_events=%d" % self._max_events
            )
        reservations = document.get("reservations") or {}
        if len(reservations) > self._max_reservations:
            raise StoreExhaustedError(
                "reservations exceed max_reservations=%d" % self._max_reservations
            )
        encoded = canonical_json(dict(document)).encode("utf-8")
        if len(encoded) > self._max_document_bytes:
            raise StoreExhaustedError(
                "document exceeds max_document_bytes=%d" % self._max_document_bytes
            )


# ---------------------------------------------------------------------------
# Durable file backend
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class DurableUsageLedgerStore:
    """Single-host durable store: exclusive flock + atomic document replace.

    Multiple local processes coordinate through the lock file. The document is
    the CAS unit; partial writes are never visible.
    """

    authorizes_admission = True

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        clock: Optional[Clock] = None,
        writer_id: Optional[str] = None,
        fence: int = 1,
        max_events: int = DEFAULT_MAX_EVENTS,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        max_reservations: int = DEFAULT_MAX_RESERVATIONS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._writer_id = writer_id
        self._initial_fence = int(fence)
        self._max_events = int(max_events)
        self._max_document_bytes = int(max_document_bytes)
        self._max_reservations = int(max_reservations)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._thread_lock = threading.RLock()
        self._ensure_initialized()

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_initialized(self) -> None:
        with self._exclusive():
            if not self._path.exists():
                doc = empty_ledger_document(
                    writer_id=self._writer_id, fence=self._initial_fence
                )
                self._write_unlocked(doc)
            else:
                # Touch-validate existing document.
                self._read_unlocked()

    def read(self) -> Dict[str, Any]:
        with self._thread_lock:
            with self._exclusive():
                return self._read_unlocked()

    def compare_and_set(
        self,
        expected_revision: int,
        document: Mapping[str, Any],
        *,
        writer_id: Optional[str] = None,
        fence: Optional[int] = None,
    ) -> Dict[str, Any]:
        validated = validate_document(document)
        with self._thread_lock:
            with self._exclusive():
                current = self._read_unlocked()
                if int(expected_revision) != int(current["revision"]):
                    raise CompareAndSetConflict(
                        "expected revision %s but store is at %s"
                        % (expected_revision, current["revision"])
                    )
                self._check_writer_fence(current, writer_id=writer_id, fence=fence)
                self._check_capacity(validated)
                if writer_id is not None:
                    validated["writer_id"] = writer_id
                if fence is not None:
                    if int(fence) < int(current["fence"]):
                        raise StaleFenceError(
                            "caller fence %s is stale (store fence %s)"
                            % (fence, current["fence"])
                        )
                    validated["fence"] = int(fence)
                else:
                    validated["fence"] = int(current["fence"])
                validated["revision"] = int(current["revision"]) + 1
                validated["schema"] = STORE_SCHEMA
                validated["schema_version"] = STORE_SCHEMA_VERSION
                validated = validate_document(validated)
                self._write_unlocked(validated)
                return _deep_copy_document(validated)

    def bump_fence(self, *, writer_id: str) -> int:
        with self._thread_lock:
            with self._exclusive():
                doc = self._read_unlocked()
                doc["fence"] = int(doc["fence"]) + 1
                doc["writer_id"] = writer_id
                doc["revision"] = int(doc["revision"]) + 1
                doc = validate_document(doc)
                self._write_unlocked(doc)
                return int(doc["fence"])

    def checkpoint(self) -> Dict[str, Any]:
        with self._thread_lock:
            with self._exclusive():
                doc = self._read_unlocked()
                receipt = {
                    "schema": "ipfs_accelerate_py.endpoint_usage.checkpoint@1",
                    "revision": doc["revision"],
                    "fence": doc["fence"],
                    "writer_id": doc.get("writer_id"),
                    "next_sequence": doc["next_sequence"],
                    "compacted_through": doc["compacted_through"],
                    "document_cid": document_cid(doc),
                    "path": str(self._path),
                    "observed_at": _to_rfc3339(self._clock.now()),
                }
                checkpoint_path = self._path.with_suffix(
                    self._path.suffix + ".checkpoint.json"
                )
                payload = canonical_json(receipt).encode("utf-8")
                _atomic_write_bytes(checkpoint_path, payload)
                return receipt

    def close(self) -> None:
        return None

    def _check_writer_fence(
        self,
        current: Mapping[str, Any],
        *,
        writer_id: Optional[str],
        fence: Optional[int],
    ) -> None:
        current_writer = current.get("writer_id")
        current_fence = int(current.get("fence") or 0)
        if fence is not None and int(fence) < current_fence:
            raise StaleFenceError(
                "caller fence %s is stale (store fence %s)" % (fence, current_fence)
            )
        if (
            writer_id is not None
            and current_writer is not None
            and writer_id != current_writer
            and fence is not None
            and int(fence) <= current_fence
        ):
            raise SplitWriterError(
                "writer %r conflicts with active writer %r at fence %s"
                % (writer_id, current_writer, current_fence)
            )

    def _check_capacity(self, document: Mapping[str, Any]) -> None:
        events = document.get("events") or []
        if len(events) > self._max_events:
            raise StoreExhaustedError(
                "event log exceeds max_events=%d" % self._max_events
            )
        reservations = document.get("reservations") or {}
        if len(reservations) > self._max_reservations:
            raise StoreExhaustedError(
                "reservations exceed max_reservations=%d" % self._max_reservations
            )
        encoded = canonical_json(dict(document)).encode("utf-8")
        if len(encoded) > self._max_document_bytes:
            raise StoreExhaustedError(
                "document exceeds max_document_bytes=%d" % self._max_document_bytes
            )

    def _read_unlocked(self) -> Dict[str, Any]:
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError as exc:
            raise CorruptionError("ledger document missing") from exc
        except OSError as exc:
            raise LedgerStoreError("failed to read ledger document") from exc
        if not raw:
            raise CorruptionError("ledger document is empty")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptionError("ledger document is not valid JSON") from exc
        return validate_document(payload)

    def _write_unlocked(self, document: Mapping[str, Any]) -> None:
        validated = validate_document(document)
        payload = canonical_json(validated).encode("utf-8")
        if len(payload) > self._max_document_bytes:
            raise StoreExhaustedError(
                "document exceeds max_document_bytes=%d" % self._max_document_bytes
            )
        _atomic_write_bytes(self._path, payload)

    def _exclusive(self):
        return _FileLock(self._lock_path, timeout=self._lock_timeout_seconds)


class _FileLock:
    def __init__(self, path: Path, *, timeout: float) -> None:
        self._path = path
        self._timeout = timeout
        self._handle: Optional[Any] = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+", encoding="utf-8")
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise LedgerStoreError(
                        "timed out acquiring ledger lock at %s" % self._path
                    )
                time.sleep(0.01)

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


# ---------------------------------------------------------------------------
# Partitioned / distributed / IPFS helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityPartition:
    """Explicit conservative per-node capacity partition.

    When a global strongly consistent backend is unavailable, each node may
    own a disjoint fraction of the configured ceiling. Fractions must sum to
    at most 1.0 across the deployment; this type does not coordinate peers.
    """

    node_id: str
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not self.node_id or not isinstance(self.node_id, str):
            raise ValueError("node_id is required")
        if (
            isinstance(self.numerator, bool)
            or not isinstance(self.numerator, int)
            or self.numerator < 0
        ):
            raise ValueError("numerator must be a non-negative integer")
        if (
            isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator <= 0
        ):
            raise ValueError("denominator must be a positive integer")
        if self.numerator > self.denominator:
            raise ValueError("numerator cannot exceed denominator")

    def scale_ceiling(self, ceiling: int) -> int:
        """Return the integer partition share (floor, conservative)."""

        if ceiling < 0:
            raise ValueError("ceiling must be non-negative")
        return (ceiling * self.numerator) // self.denominator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapacityPartition":
        return cls(
            node_id=str(data["node_id"]),
            numerator=int(data["numerator"]),
            denominator=int(data["denominator"]),
        )


class PartitionedUsageLedgerStore:
    """Wrap a local store with an explicit capacity partition marker.

    Admission remains local to the wrapped store; ceilings must be scaled by
    the partition before limits are installed. This adapter records the
    partition in the document metadata so recovery can prove the conservative
    split was intentional.
    """

    def __init__(
        self,
        inner: UsageLedgerStore,
        partition: CapacityPartition,
    ) -> None:
        self._inner = inner
        self._partition = partition

    @property
    def clock(self) -> Clock:
        return self._inner.clock

    @property
    def authorizes_admission(self) -> bool:
        return bool(self._inner.authorizes_admission)

    @property
    def partition(self) -> CapacityPartition:
        return self._partition

    def read(self) -> Dict[str, Any]:
        return self._inner.read()

    def compare_and_set(
        self,
        expected_revision: int,
        document: Mapping[str, Any],
        *,
        writer_id: Optional[str] = None,
        fence: Optional[int] = None,
    ) -> Dict[str, Any]:
        doc = _deep_copy_document(document)
        doc["partition"] = self._partition.to_dict()
        return self._inner.compare_and_set(
            expected_revision, doc, writer_id=writer_id, fence=fence
        )

    def checkpoint(self) -> Dict[str, Any]:
        return self._inner.checkpoint()

    def close(self) -> None:
        self._inner.close()


class IPFSAuditMirror:
    """Audit/recovery mirror over content-addressed records.

    This type deliberately does **not** implement admission authority.
    Eventual IPFS replication alone cannot authorize capacity grants.
    """

    authorizes_admission = False

    def __init__(self, *, records: Optional[MutableMapping[str, Mapping[str, Any]]] = None):
        self._records: MutableMapping[str, Mapping[str, Any]] = (
            records if records is not None else {}
        )

    def put_event(self, event: Mapping[str, Any]) -> str:
        cid = content_cid(dict(event))
        self._records[cid] = dict(event)
        return cid

    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        item = self._records.get(cid)
        return dict(item) if item is not None else None

    def authorize_admission(self, *_args: Any, **_kwargs: Any) -> None:
        raise AdmissionAuthorityError(
            "eventual IPFS replication alone cannot authorize admission; "
            "use a strongly consistent fenced backend or explicit per-node partitions"
        )

    def compare_and_set(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        self.authorize_admission()
        raise AssertionError("unreachable")


def migrate_document(
    document: Mapping[str, Any],
    *,
    target_schema_version: str = STORE_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Migrate a ledger document; unknown versions fail closed."""

    if target_schema_version not in SUPPORTED_STORE_SCHEMA_VERSIONS:
        raise MigrationError(
            "unsupported target store schema_version: %r" % (target_schema_version,)
        )
    if not isinstance(document, Mapping):
        raise MigrationError("document must be an object")
    schema = document.get("schema")
    version = document.get("schema_version")
    if schema is None and version is None:
        raise MigrationError("document lacks schema markers")
    if schema not in (None, STORE_SCHEMA):
        raise MigrationError("cannot migrate unknown schema %r" % (schema,))
    if version in SUPPORTED_STORE_SCHEMA_VERSIONS:
        # Identity migration: re-validate.
        return validate_document(document)
    raise MigrationError(
        "no migration path from schema_version %r to %r"
        % (version, target_schema_version)
    )


def read_only_recovery_view(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a redacted read-only recovery projection (no mutation handles)."""

    validated = validate_document(document)
    return {
        "schema": "ipfs_accelerate_py.endpoint_usage.recovery-view@1",
        "revision": validated["revision"],
        "fence": validated["fence"],
        "writer_id": validated.get("writer_id"),
        "next_sequence": validated["next_sequence"],
        "compacted_through": validated["compacted_through"],
        "event_count": len(validated.get("events") or []),
        "reservation_count": len(validated.get("reservations") or {}),
        "document_cid": document_cid(validated),
        "partition": validated.get("partition"),
        "read_only": True,
    }


__all__ = [
    "ATOMIC_USAGE_LEDGER_REQUIREMENT_ID",
    "AdmissionAuthorityError",
    "CapacityPartition",
    "Clock",
    "CompareAndSetConflict",
    "CorruptionError",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_MAX_DOCUMENT_BYTES",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_MAX_RESERVATIONS",
    "DurableUsageLedgerStore",
    "FakeClock",
    "IPFSAuditMirror",
    "InMemoryUsageLedgerStore",
    "LedgerStoreError",
    "MigrationError",
    "PartitionedUsageLedgerStore",
    "STORE_SCHEMA",
    "STORE_SCHEMA_VERSION",
    "SUPPORTED_STORE_SCHEMA_VERSIONS",
    "SchemaDriftError",
    "SplitWriterError",
    "StaleFenceError",
    "StoreExhaustedError",
    "SystemClock",
    "UsageLedgerStore",
    "datetime_to_ms",
    "document_cid",
    "empty_ledger_document",
    "idempotency_index_key",
    "migrate_document",
    "ms_to_datetime",
    "parse_rfc3339",
    "read_only_recovery_view",
    "rfc3339_to_ms",
    "validate_document",
]
