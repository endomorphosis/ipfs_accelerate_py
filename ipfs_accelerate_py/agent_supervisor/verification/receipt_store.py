"""Immutable verification receipt storage and generation-CAS indexes.

``VerificationReceiptStore@1`` is the narrow storage protocol for the
incremental verification planner:

* immutable receipt / artifact bytes under exact CID recomputation;
* Mapping envelopes for DurableCoordinationStore adaptation;
* a generation-bound local head compare-and-swap protocol;
* history replay, corruption rejection, tombstones, and GC metadata;
* optional, lazy ``ipfs_kit_py`` adaptation with exact leaf-symbol probing.

``HermeticVerificationReceiptStore`` is the production-local backend used by
unit and concurrency tests.  ``IpfsKitVerificationReceiptStore`` adapts public
``DurableCoordinationStore`` Mapping put/get/recover with an *explicit*
``storage_dir`` (no home defaults).  Head CAS is always the local generation
protocol; an optional injected async Iroh manifest bridge is used only when
that exact capability is operational.  Mock IPFS fallbacks and private SQLite
CAS tables are forbidden.  Corruption is never silently downgraded.
"""

from __future__ import annotations

import base64
import fcntl
import importlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Final, Protocol, runtime_checkable

from typing_extensions import Self

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    MultiformatsIdentityError,
    cid_for_bytes,
    validate_cid,
)

# ---------------------------------------------------------------------------
# Evidence / interface constants
# ---------------------------------------------------------------------------

VERIFICATION_RECEIPT_STORE_INTERFACE: Final[str] = "VerificationReceiptStore@1"
STORE_PROTOCOL_EVIDENCE: Final[str] = "ivp/store-protocol@1"
CONCURRENT_STORE_CAS_EVIDENCE: Final[str] = "ivp/concurrent-store-cas@1"

RECEIPT_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-receipt-envelope@1"
)
INDEX_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-index-snapshot@1"
)
INDEX_HEAD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-index-head@1"
)
TOMBSTONE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-tombstone@1"
)
GC_METADATA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-gc-metadata@1"
)
RAW_BYTES_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-raw-bytes-envelope@1"
)

DURABLE_COORDINATION_LEAF_MODULE: Final[str] = (
    "ipfs_kit_py.mcp_server.mcplusplus.coordination_storage"
)
DURABLE_COORDINATION_SYMBOL: Final[str] = "DurableCoordinationStore"
REQUIRED_STORE_METHODS: Final[tuple[str, ...]] = ("put", "get", "recover", "get_bytes")

DEFAULT_LOCK_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_BLOB_BYTES: Final[int] = 8 * 1_048_576
EMPTY_INDEX_ROOT_TOKEN: Final[str] = "empty"

_TMP_PREFIX: Final[str] = ".tmp."
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


# ---------------------------------------------------------------------------
# Errors and typed unavailable
# ---------------------------------------------------------------------------


class ReceiptStoreError(RuntimeError):
    """Base operational failure for verification receipt stores."""


class ReceiptStoreIntegrityError(ReceiptStoreError, ValueError):
    """Immutable block or index failed integrity verification."""


class ReceiptStoreConflictError(ReceiptStoreError):
    """Generation CAS observed a peer write; never overwrites peer state."""


class ReceiptStoreUnavailableError(ReceiptStoreError):
    """Typed unavailable: missing backend, revision, or CAS capability."""

    def __init__(self, unavailable: StoreUnavailable) -> None:
        super().__init__(unavailable.reason)
        self.unavailable = unavailable


class StoreUnavailableCode(str, Enum):
    """Machine-readable reasons for typed unavailability."""

    NAMESPACE_ONLY = "namespace_only"
    ABSENT_BACKEND = "absent_backend"
    ABSENT_REVISION = "absent_revision"
    ABSENT_SYMBOL = "absent_symbol"
    INCOMPATIBLE_API = "incompatible_api"
    CAS_UNAVAILABLE = "cas_unavailable"
    STORAGE_ROOT_REQUIRED = "storage_root_required"
    BRIDGE_UNAVAILABLE = "bridge_unavailable"
    NOT_OPERATIONAL = "not_operational"


@dataclass(frozen=True, slots=True)
class StoreUnavailable:
    """Fail-closed capability result; never invents storage authority."""

    reason: str
    code: StoreUnavailableCode
    detail: Mapping[str, Any] = field(default_factory=dict)
    status: str = "unavailable"

    def __post_init__(self) -> None:
        if self.status != "unavailable":
            raise ValueError("StoreUnavailable.status must be 'unavailable'")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("StoreUnavailable.reason is required")
        object.__setattr__(self, "reason", self.reason.strip())
        if not isinstance(self.code, StoreUnavailableCode):
            object.__setattr__(self, "code", StoreUnavailableCode(str(self.code)))
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail or {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "code": self.code.value,
            "detail": dict(self.detail),
        }

    def raise_error(self) -> None:
        raise ReceiptStoreUnavailableError(self)


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------


def _now_ms(clock: Callable[[], int] | None = None) -> int:
    if clock is not None:
        value = int(clock())
        if value < 0:
            raise ValueError("clock must return non-negative milliseconds")
        return value
    return int(time.time() * 1000)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptStoreIntegrityError(f"value is not canonical JSON: {exc}") from exc


def _loads_mapping(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptStoreIntegrityError("bytes are not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReceiptStoreIntegrityError("JSON root must be an object")
    if _canonical_json_bytes(value) != data:
        raise ReceiptStoreIntegrityError("bytes are not canonical DAG-JSON form")
    return value


def mapping_cid(value: Mapping[str, Any]) -> str:
    """Return the dag-json CIDv1 for a Mapping envelope."""

    data = _canonical_json_bytes(dict(value))
    try:
        return cid_for_bytes(data, codec="dag-json")
    except MultiformatsIdentityError as exc:
        raise ReceiptStoreIntegrityError(str(exc)) from exc


def raw_cid(data: bytes) -> str:
    """Return the raw CIDv1 for exact bytes."""

    if type(data) is not bytes:
        raise ReceiptStoreIntegrityError("raw payload must be exact bytes")
    try:
        return cid_for_bytes(data, codec="raw")
    except MultiformatsIdentityError as exc:
        raise ReceiptStoreIntegrityError(str(exc)) from exc


def _require_cid(value: Any, *, field_name: str) -> str:
    try:
        return validate_cid(value, codecs=("raw", "dag-json"))
    except MultiformatsIdentityError as exc:
        raise ReceiptStoreIntegrityError(f"{field_name} is not a valid CID: {exc}") from exc


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve()) if path.exists() or path.parent.exists() else str(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive_lock(path: Path, *, timeout_seconds: float) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = _thread_lock(path)
    deadline = time.monotonic() + timeout_seconds
    if not lock.acquire(timeout=max(0.0, timeout_seconds)):
        raise TimeoutError(f"timed out acquiring thread lock: {path}")
    handle = path.open("a+b")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring process lock: {path}")
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        lock.release()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` via temp file → fsync → replace → directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / (
        f"{_TMP_PREFIX}{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> bytes:
    data = _canonical_json_bytes(dict(value))
    _atomic_write_bytes(path, data)
    return data


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ReceiptStoreIntegrityError(f"symlink rejected: {path}")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PutResult:
    """Outcome of an immutable put."""

    cid: str
    created: bool
    codec: str
    byte_length: int
    durable: bool = True
    replicated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "created": self.created,
            "codec": self.codec,
            "byte_length": self.byte_length,
            "durable": self.durable,
            "replicated": self.replicated,
        }


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One exact-key index binding from receipt key id to receipt CID."""

    key_id: str
    receipt_cid: str
    kind: str = "receipt"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise ReceiptStoreIntegrityError("IndexEntry.key_id is required")
        object.__setattr__(self, "key_id", self.key_id.strip())
        object.__setattr__(self, "receipt_cid", _require_cid(self.receipt_cid, field_name="receipt_cid"))
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ReceiptStoreIntegrityError("IndexEntry.kind is required")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key_id": self.key_id,
            "receipt_cid": self.receipt_cid,
            "kind": self.kind,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IndexEntry:
        if not isinstance(value, Mapping):
            raise ReceiptStoreIntegrityError("IndexEntry must be an object")
        return cls(
            key_id=str(value.get("key_id") or ""),
            receipt_cid=str(value.get("receipt_cid") or ""),
            kind=str(value.get("kind") or "receipt"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class TombstoneRecord:
    """Immutable audit tombstone; history is retained, never rewritten."""

    key_id: str
    prior_receipt_cid: str
    reason: str
    tombstoned_at_ms: int
    tombstone_cid: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise ReceiptStoreIntegrityError("TombstoneRecord.key_id is required")
        object.__setattr__(self, "key_id", self.key_id.strip())
        object.__setattr__(
            self,
            "prior_receipt_cid",
            _require_cid(self.prior_receipt_cid, field_name="prior_receipt_cid"),
        )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ReceiptStoreIntegrityError("TombstoneRecord.reason is required")
        object.__setattr__(self, "reason", self.reason.strip())
        if (
            isinstance(self.tombstoned_at_ms, bool)
            or not isinstance(self.tombstoned_at_ms, int)
            or self.tombstoned_at_ms < 0
        ):
            raise ReceiptStoreIntegrityError("tombstoned_at_ms must be a non-negative int")
        if self.tombstone_cid is not None:
            object.__setattr__(
                self,
                "tombstone_cid",
                _require_cid(self.tombstone_cid, field_name="tombstone_cid"),
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def to_envelope(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TOMBSTONE_SCHEMA,
            "interface": VERIFICATION_RECEIPT_STORE_INTERFACE,
            "kind": "VerificationTombstone",
            "key_id": self.key_id,
            "prior_receipt_cid": self.prior_receipt_cid,
            "reason": self.reason,
            "tombstoned_at_ms": self.tombstoned_at_ms,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "key_id": self.key_id,
            "prior_receipt_cid": self.prior_receipt_cid,
            "reason": self.reason,
            "tombstoned_at_ms": self.tombstoned_at_ms,
        }
        if self.tombstone_cid is not None:
            payload["tombstone_cid"] = self.tombstone_cid
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TombstoneRecord:
        if not isinstance(value, Mapping):
            raise ReceiptStoreIntegrityError("TombstoneRecord must be an object")
        tombstone_cid = value.get("tombstone_cid")
        return cls(
            key_id=str(value.get("key_id") or ""),
            prior_receipt_cid=str(value.get("prior_receipt_cid") or ""),
            reason=str(value.get("reason") or ""),
            tombstoned_at_ms=int(value.get("tombstoned_at_ms") or 0),
            tombstone_cid=str(tombstone_cid) if tombstone_cid else None,
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """Immutable verification-index generation body."""

    generation: int
    entries: tuple[IndexEntry, ...] = ()
    tombstones: tuple[TombstoneRecord, ...] = ()
    previous_root_cid: str | None = None
    created_at_ms: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ReceiptStoreIntegrityError("generation must be a non-negative int")
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        tombstones = tuple(self.tombstones)
        object.__setattr__(self, "tombstones", tombstones)
        if self.previous_root_cid is not None:
            object.__setattr__(
                self,
                "previous_root_cid",
                _require_cid(self.previous_root_cid, field_name="previous_root_cid"),
            )
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms < 0
        ):
            raise ReceiptStoreIntegrityError("created_at_ms must be a non-negative int")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
        # Exact key uniqueness within a generation.
        seen: set[str] = set()
        for entry in entries:
            if entry.key_id in seen:
                raise ReceiptStoreIntegrityError(
                    f"duplicate index key_id in generation {self.generation}: {entry.key_id}"
                )
            seen.add(entry.key_id)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": INDEX_SNAPSHOT_SCHEMA,
            "interface": VERIFICATION_RECEIPT_STORE_INTERFACE,
            "kind": "VerificationIndexSnapshot",
            "generation": self.generation,
            "entries": [entry.to_dict() for entry in self.entries],
            "tombstones": [item.to_dict() for item in self.tombstones],
            "created_at_ms": self.created_at_ms,
        }
        if self.previous_root_cid is not None:
            payload["previous_root_cid"] = self.previous_root_cid
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @property
    def root_cid(self) -> str:
        return mapping_cid(self.to_dict())

    def entry_map(self) -> dict[str, IndexEntry]:
        return {entry.key_id: entry for entry in self.entries}

    def with_entries(
        self,
        entries: Iterable[IndexEntry],
        *,
        generation: int | None = None,
        previous_root_cid: str | None = None,
        created_at_ms: int | None = None,
        tombstones: Iterable[TombstoneRecord] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> IndexSnapshot:
        return IndexSnapshot(
            generation=self.generation if generation is None else generation,
            entries=tuple(entries),
            tombstones=self.tombstones if tombstones is None else tuple(tombstones),
            previous_root_cid=(
                self.previous_root_cid if previous_root_cid is None else previous_root_cid
            ),
            created_at_ms=self.created_at_ms if created_at_ms is None else created_at_ms,
            metadata=self.metadata if metadata is None else metadata,
        )

    @classmethod
    def empty(cls, *, created_at_ms: int = 0) -> IndexSnapshot:
        return cls(generation=0, created_at_ms=created_at_ms)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IndexSnapshot:
        if not isinstance(value, Mapping):
            raise ReceiptStoreIntegrityError("IndexSnapshot must be an object")
        schema = value.get("schema")
        if schema not in {None, INDEX_SNAPSHOT_SCHEMA}:
            raise ReceiptStoreIntegrityError("IndexSnapshot schema mismatch")
        raw_entries = value.get("entries") or []
        raw_tombstones = value.get("tombstones") or []
        if not isinstance(raw_entries, list) or not isinstance(raw_tombstones, list):
            raise ReceiptStoreIntegrityError("entries/tombstones must be arrays")
        previous = value.get("previous_root_cid")
        return cls(
            generation=int(value.get("generation") or 0),
            entries=tuple(IndexEntry.from_dict(item) for item in raw_entries),
            tombstones=tuple(TombstoneRecord.from_dict(item) for item in raw_tombstones),
            previous_root_cid=str(previous) if previous else None,
            created_at_ms=int(value.get("created_at_ms") or 0),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class CompareAndSwapResult:
    """Generation-CAS publication outcome; conflicts never overwrite peers."""

    success: bool
    generation: int
    root_cid: str
    expected_generation: int | None = None
    expected_root_cid: str | None = None
    conflict: bool = False
    unavailable: StoreUnavailable | None = None
    snapshot: IndexSnapshot | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.success,
            "generation": self.generation,
            "root_cid": self.root_cid,
            "conflict": self.conflict,
            "reason": self.reason,
        }
        if self.expected_generation is not None:
            payload["expected_generation"] = self.expected_generation
        if self.expected_root_cid is not None:
            payload["expected_root_cid"] = self.expected_root_cid
        if self.unavailable is not None:
            payload["unavailable"] = self.unavailable.to_dict()
        if self.snapshot is not None:
            payload["snapshot"] = self.snapshot.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class GCMetadata:
    """Reachability / last-access metadata for later GC (never deletes history)."""

    cid: str
    last_access_ms: int
    reachable: bool = True
    refcount: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": GC_METADATA_SCHEMA,
            "cid": self.cid,
            "last_access_ms": self.last_access_ms,
            "reachable": self.reachable,
            "refcount": self.refcount,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GCMetadata:
        if not isinstance(value, Mapping):
            raise ReceiptStoreIntegrityError("GCMetadata must be an object")
        return cls(
            cid=_require_cid(value.get("cid"), field_name="cid"),
            last_access_ms=int(value.get("last_access_ms") or 0),
            reachable=bool(value.get("reachable", True)),
            refcount=int(value.get("refcount") or 1),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class RecoverResult:
    """Outcome of verifying immutable blocks and replaying local state."""

    verified_blocks: int
    rebuilt: bool
    errors: tuple[Mapping[str, str], ...] = ()
    history_generations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_blocks": self.verified_blocks,
            "rebuilt": self.rebuilt,
            "errors": [dict(item) for item in self.errors],
            "history_generations": self.history_generations,
        }


@dataclass(frozen=True, slots=True)
class DurableCoordinationProbe:
    """Exact leaf-module / symbol probe for DurableCoordinationStore."""

    available: bool
    module_name: str = DURABLE_COORDINATION_LEAF_MODULE
    symbol_name: str = DURABLE_COORDINATION_SYMBOL
    unavailable: StoreUnavailable | None = None
    methods: tuple[str, ...] = ()
    module_origin: str | None = None
    revision_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": self.available,
            "module_name": self.module_name,
            "symbol_name": self.symbol_name,
            "methods": list(self.methods),
        }
        if self.unavailable is not None:
            payload["unavailable"] = self.unavailable.to_dict()
        if self.module_origin is not None:
            payload["module_origin"] = self.module_origin
        if self.revision_hint is not None:
            payload["revision_hint"] = self.revision_hint
        return payload


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VerificationReceiptStore(Protocol):
    """Narrow immutable store + generation-CAS protocol."""

    @property
    def storage_dir(self) -> Path: ...

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_cid: str | None = None,
        codec: str = "raw",
    ) -> PutResult: ...

    def get_bytes(self, cid: str) -> bytes: ...

    def put_mapping(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
    ) -> PutResult: ...

    def get_mapping(self, cid: str) -> dict[str, Any]: ...

    def put_receipt_envelope(
        self,
        body: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
        stored_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PutResult: ...

    def get_receipt_envelope(self, cid: str) -> dict[str, Any]: ...

    def current_index(self) -> IndexSnapshot: ...

    def compare_and_swap_index(
        self,
        snapshot: IndexSnapshot,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult: ...

    def replay_history(self) -> tuple[IndexSnapshot, ...]: ...

    def publish_tombstone(
        self,
        tombstone: TombstoneRecord,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult: ...

    def record_access(self, cid: str, *, at_ms: int | None = None) -> GCMetadata: ...

    def collect_gc_metadata(self) -> tuple[GCMetadata, ...]: ...

    def recover(self, *, rebuild: bool = True) -> RecoverResult: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Envelope builders
# ---------------------------------------------------------------------------


def build_receipt_envelope(
    body: Mapping[str, Any],
    *,
    stored_at_ms: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an immutable Mapping envelope for a receipt body."""

    if not isinstance(body, Mapping):
        raise ReceiptStoreIntegrityError("receipt body must be a Mapping")
    body_dict = dict(body)
    body_cid = mapping_cid(body_dict)
    envelope: dict[str, Any] = {
        "schema": RECEIPT_ENVELOPE_SCHEMA,
        "interface": VERIFICATION_RECEIPT_STORE_INTERFACE,
        "kind": "VerificationReceiptEnvelope",
        "body": body_dict,
        "body_cid": body_cid,
        "stored_at_ms": _now_ms() if stored_at_ms is None else int(stored_at_ms),
    }
    if metadata:
        envelope["metadata"] = dict(metadata)
    return envelope


def build_raw_bytes_envelope(data: bytes, *, stored_at_ms: int | None = None) -> dict[str, Any]:
    """Wrap raw bytes for Mapping-only backends (CID is still the raw CID)."""

    if type(data) is not bytes:
        raise ReceiptStoreIntegrityError("raw payload must be exact bytes")
    return {
        "schema": RAW_BYTES_ENVELOPE_SCHEMA,
        "interface": VERIFICATION_RECEIPT_STORE_INTERFACE,
        "kind": "VerificationRawBytesEnvelope",
        "codec": "raw",
        "byte_length": len(data),
        "data_b64": base64.b64encode(data).decode("ascii"),
        "content_cid": raw_cid(data),
        "stored_at_ms": _now_ms() if stored_at_ms is None else int(stored_at_ms),
    }


# ---------------------------------------------------------------------------
# DurableCoordinationStore leaf probe
# ---------------------------------------------------------------------------


def probe_durable_coordination_store(
    *,
    importer: Callable[[str], ModuleType] | None = None,
    namespace_only_module: ModuleType | None = None,
) -> DurableCoordinationProbe:
    """Probe the exact DurableCoordinationStore leaf module and symbol.

    A top-level ``ipfs_kit_py`` namespace import alone is *not* evidence of
    availability.  The leaf module must expose ``DurableCoordinationStore``
    with put/get/recover/get_bytes.
    """

    load = importer or importlib.import_module

    # Explicit namespace-only rejection path (tests inject a bare module).
    if namespace_only_module is not None:
        return DurableCoordinationProbe(
            available=False,
            unavailable=StoreUnavailable(
                reason="top-level ipfs_kit_py namespace import is not operational evidence",
                code=StoreUnavailableCode.NAMESPACE_ONLY,
                detail={"module": getattr(namespace_only_module, "__name__", "ipfs_kit_py")},
            ),
        )

    try:
        module = load(DURABLE_COORDINATION_LEAF_MODULE)
    except Exception as exc:  # noqa: BLE001 - probe must never raise for absence
        # Distinguish bare top-level presence vs missing leaf.
        top_level_present = False
        try:
            top = load("ipfs_kit_py")
            top_level_present = top is not None
        except Exception:  # noqa: BLE001
            top_level_present = False
        if top_level_present:
            return DurableCoordinationProbe(
                available=False,
                unavailable=StoreUnavailable(
                    reason=(
                        "ipfs_kit_py top-level import succeeded but exact leaf "
                        f"{DURABLE_COORDINATION_LEAF_MODULE} is unavailable"
                    ),
                    code=StoreUnavailableCode.NAMESPACE_ONLY,
                    detail={"error": str(exc), "leaf": DURABLE_COORDINATION_LEAF_MODULE},
                ),
            )
        return DurableCoordinationProbe(
            available=False,
            unavailable=StoreUnavailable(
                reason=f"DurableCoordinationStore backend absent: {exc}",
                code=StoreUnavailableCode.ABSENT_BACKEND,
                detail={"error": str(exc), "leaf": DURABLE_COORDINATION_LEAF_MODULE},
            ),
        )

    origin = getattr(module, "__file__", None)
    store_cls = getattr(module, DURABLE_COORDINATION_SYMBOL, None)
    if store_cls is None:
        return DurableCoordinationProbe(
            available=False,
            module_origin=origin,
            unavailable=StoreUnavailable(
                reason=f"symbol {DURABLE_COORDINATION_SYMBOL} missing from leaf module",
                code=StoreUnavailableCode.ABSENT_SYMBOL,
                detail={"leaf": DURABLE_COORDINATION_LEAF_MODULE},
            ),
        )

    methods = tuple(
        name for name in REQUIRED_STORE_METHODS if callable(getattr(store_cls, name, None))
    )
    missing = [name for name in REQUIRED_STORE_METHODS if name not in methods]
    if missing:
        return DurableCoordinationProbe(
            available=False,
            module_origin=origin,
            methods=methods,
            unavailable=StoreUnavailable(
                reason=f"DurableCoordinationStore API missing methods: {', '.join(missing)}",
                code=StoreUnavailableCode.INCOMPATIBLE_API,
                detail={"missing": missing},
            ),
        )

    # Revision / identity helpers from the same module.
    revision_hint = None
    for attr in ("cid_for_bytes", "cid_for_artifact"):
        if not callable(getattr(module, attr, None)):
            return DurableCoordinationProbe(
                available=False,
                module_origin=origin,
                methods=methods,
                unavailable=StoreUnavailable(
                    reason=f"leaf module missing required revision helper {attr}",
                    code=StoreUnavailableCode.ABSENT_REVISION,
                    detail={"missing": attr},
                ),
            )
    revision_hint = getattr(module, "__name__", None)

    return DurableCoordinationProbe(
        available=True,
        module_origin=str(origin) if origin else None,
        methods=methods,
        revision_hint=str(revision_hint) if revision_hint else None,
    )


# ---------------------------------------------------------------------------
# Shared local generation-CAS index
# ---------------------------------------------------------------------------


class _LocalGenerationCAS:
    """File-backed generation-CAS head protocol with history and fsync."""

    def __init__(
        self,
        root: Path,
        *,
        put_mapping: Callable[..., PutResult],
        get_mapping: Callable[[str], dict[str, Any]],
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.root = root
        self.index_dir = root / "index"
        self.history_dir = self.index_dir / "history"
        self.head_path = self.index_dir / "HEAD.json"
        self.lock_path = self.index_dir / "HEAD.lock"
        self.gc_path = root / "gc" / "metadata.json"
        self.gc_lock_path = root / "gc" / "metadata.lock"
        self._put_mapping = put_mapping
        self._get_mapping = get_mapping
        self.lock_timeout_seconds = lock_timeout_seconds
        self._clock = clock
        self.index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.history_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        (root / "gc").mkdir(parents=True, exist_ok=True, mode=0o700)

    def _read_head_unlocked(self) -> tuple[int, str | None, IndexSnapshot]:
        if not self.head_path.exists():
            empty = IndexSnapshot.empty(created_at_ms=_now_ms(self._clock))
            return 0, None, empty
        _reject_symlink(self.head_path)
        data = self.head_path.read_bytes()
        head = _loads_mapping(data)
        if head.get("schema") not in {None, INDEX_HEAD_SCHEMA}:
            raise ReceiptStoreIntegrityError("index HEAD schema mismatch")
        generation = int(head.get("generation") or 0)
        root_cid = head.get("root_cid")
        root_token = str(root_cid) if root_cid else None
        if root_token in {None, EMPTY_INDEX_ROOT_TOKEN}:
            return generation, None, IndexSnapshot.empty(
                created_at_ms=int(head.get("updated_at_ms") or 0)
            )
        snapshot_payload = self._get_mapping(str(root_token))
        snapshot = IndexSnapshot.from_dict(snapshot_payload)
        if snapshot.generation != generation:
            raise ReceiptStoreIntegrityError(
                "index HEAD generation does not match snapshot body"
            )
        if snapshot.root_cid != root_token:
            raise ReceiptStoreIntegrityError(
                "index HEAD root_cid does not match snapshot body CID"
            )
        return generation, str(root_token), snapshot

    def current_index(self) -> IndexSnapshot:
        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            _, _, snapshot = self._read_head_unlocked()
            return snapshot

    def compare_and_swap_index(
        self,
        snapshot: IndexSnapshot,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult:
        if not isinstance(snapshot, IndexSnapshot):
            raise ReceiptStoreIntegrityError("snapshot must be an IndexSnapshot")
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            raise ReceiptStoreIntegrityError("expected_generation must be a non-negative int")
        if expected_root_cid is not None:
            expected_root_cid = _require_cid(expected_root_cid, field_name="expected_root_cid")

        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            current_generation, current_root, current_snapshot = self._read_head_unlocked()
            expected_norm = expected_root_cid
            current_norm = current_root
            if current_generation != expected_generation or current_norm != expected_norm:
                return CompareAndSwapResult(
                    success=False,
                    conflict=True,
                    generation=current_generation,
                    root_cid=current_norm or EMPTY_INDEX_ROOT_TOKEN,
                    expected_generation=expected_generation,
                    expected_root_cid=expected_norm,
                    snapshot=current_snapshot,
                    reason="generation_cas_conflict",
                )

            if snapshot.generation != expected_generation + 1:
                return CompareAndSwapResult(
                    success=False,
                    conflict=False,
                    generation=current_generation,
                    root_cid=current_norm or EMPTY_INDEX_ROOT_TOKEN,
                    expected_generation=expected_generation,
                    expected_root_cid=expected_norm,
                    snapshot=current_snapshot,
                    reason=(
                        "snapshot.generation must be expected_generation + 1 "
                        f"(got {snapshot.generation}, expected {expected_generation + 1})"
                    ),
                )

            # Preserve peer entries: new snapshot must not drop keys that
            # existed at the expected generation unless explicitly tombstoned.
            # Callers that intentionally replace must include the merged set;
            # CAS itself never rewrites HEAD on conflict (handled above).

            body = snapshot.to_dict()
            if snapshot.previous_root_cid is None and current_norm is not None:
                # Auto-link history when caller omitted previous pointer.
                body = dict(body)
                body["previous_root_cid"] = current_norm
                snapshot = IndexSnapshot.from_dict(body)

            put = self._put_mapping(snapshot.to_dict(), expected_cid=snapshot.root_cid)
            root_cid = put.cid
            now = _now_ms(self._clock)
            head = {
                "schema": INDEX_HEAD_SCHEMA,
                "interface": VERIFICATION_RECEIPT_STORE_INTERFACE,
                "generation": snapshot.generation,
                "root_cid": root_cid,
                "updated_at_ms": now,
                "previous_root_cid": current_norm,
            }
            _atomic_write_json(self.head_path, head)
            history_name = f"{snapshot.generation:020d}-{root_cid}.json"
            history_path = self.history_dir / history_name
            if not history_path.exists():
                _atomic_write_json(history_path, snapshot.to_dict())
            return CompareAndSwapResult(
                success=True,
                conflict=False,
                generation=snapshot.generation,
                root_cid=root_cid,
                expected_generation=expected_generation,
                expected_root_cid=expected_norm,
                snapshot=snapshot,
                reason="ok",
            )

    def replay_history(self) -> tuple[IndexSnapshot, ...]:
        snapshots: list[IndexSnapshot] = []
        if not self.history_dir.exists():
            return ()
        for path in sorted(self.history_dir.glob("*.json")):
            _reject_symlink(path)
            data = path.read_bytes()
            payload = _loads_mapping(data)
            snapshot = IndexSnapshot.from_dict(payload)
            recomputed = mapping_cid(snapshot.to_dict())
            # Filename embeds root cid after the generation prefix.
            embedded = path.stem.split("-", 1)
            if len(embedded) == 2 and embedded[1] != recomputed:
                raise ReceiptStoreIntegrityError(
                    f"history file CID mismatch for {path.name}"
                )
            snapshots.append(snapshot)
        return tuple(snapshots)

    def record_access(self, cid: str, *, at_ms: int | None = None) -> GCMetadata:
        cid = _require_cid(cid, field_name="cid")
        when = _now_ms(self._clock) if at_ms is None else int(at_ms)
        # GC metadata uses its own lock so index CAS never deadlocks on access.
        with _exclusive_lock(self.gc_lock_path, timeout_seconds=self.lock_timeout_seconds):
            table = self._load_gc_table()
            existing = table.get(cid)
            if existing is None:
                meta = GCMetadata(cid=cid, last_access_ms=when, reachable=True, refcount=1)
            else:
                meta = GCMetadata(
                    cid=cid,
                    last_access_ms=when,
                    reachable=True,
                    refcount=int(existing.get("refcount") or 1),
                    metadata=dict(existing.get("metadata") or {}),
                )
            table[cid] = meta.to_dict()
            _atomic_write_json(
                self.gc_path,
                {
                    "schema": GC_METADATA_SCHEMA,
                    "interface": VERIFICATION_RECEIPT_STORE_INTERFACE,
                    "entries": table,
                    "updated_at_ms": when,
                },
            )
            return meta

    def collect_gc_metadata(self) -> tuple[GCMetadata, ...]:
        with _exclusive_lock(self.gc_lock_path, timeout_seconds=self.lock_timeout_seconds):
            table = self._load_gc_table()
            return tuple(GCMetadata.from_dict(item) for item in table.values())

    def _load_gc_table(self) -> dict[str, dict[str, Any]]:
        if not self.gc_path.exists():
            return {}
        _reject_symlink(self.gc_path)
        payload = _loads_mapping(self.gc_path.read_bytes())
        entries = payload.get("entries") or {}
        if not isinstance(entries, dict):
            raise ReceiptStoreIntegrityError("gc metadata entries must be an object")
        return {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Hermetic backend
# ---------------------------------------------------------------------------


class HermeticVerificationReceiptStore:
    """Local-file immutable store with locks, atomic replace/fsync, and CAS."""

    def __init__(
        self,
        storage_dir: os.PathLike[str] | str,
        *,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if storage_dir is None or str(storage_dir).strip() == "":
            raise ReceiptStoreUnavailableError(
                StoreUnavailable(
                    reason="HermeticVerificationReceiptStore requires an explicit storage_dir",
                    code=StoreUnavailableCode.STORAGE_ROOT_REQUIRED,
                )
            )
        self._storage_dir = Path(storage_dir).expanduser().resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blocks_dir = self._storage_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.max_blob_bytes = int(max_blob_bytes)
        self._clock = clock
        self._closed = False
        self._cas = _LocalGenerationCAS(
            self._storage_dir,
            put_mapping=self.put_mapping,
            get_mapping=self.get_mapping,
            lock_timeout_seconds=self.lock_timeout_seconds,
            clock=self._clock,
        )

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def _ensure_open(self) -> None:
        if self._closed:
            raise ReceiptStoreError("store is closed")

    def _block_path(self, cid: str) -> Path:
        cid = _require_cid(cid, field_name="cid")
        directory = self.blocks_dir / cid[1:3]
        return directory / f"{cid}.block"

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_cid: str | None = None,
        codec: str = "raw",
    ) -> PutResult:
        self._ensure_open()
        if type(data) is not bytes:
            raise ReceiptStoreIntegrityError("put_bytes requires exact bytes")
        if len(data) > self.max_blob_bytes:
            raise ReceiptStoreIntegrityError(
                f"payload exceeds max_blob_bytes ({self.max_blob_bytes})"
            )
        if codec not in {"raw", "dag-json"}:
            raise ReceiptStoreIntegrityError(f"unsupported codec: {codec}")
        cid = raw_cid(data) if codec == "raw" else cid_for_bytes(data, codec="dag-json")
        if expected_cid is not None and expected_cid != cid:
            raise ReceiptStoreIntegrityError(
                f"CID mismatch: computed {cid}, expected {expected_cid}"
            )
        path = self._block_path(cid)
        if path.exists():
            _reject_symlink(path)
            existing = path.read_bytes()
            if existing != data:
                raise ReceiptStoreIntegrityError(f"immutable block collision for {cid}")
            # Verify CID still holds.
            recomputed = raw_cid(existing) if codec == "raw" else cid_for_bytes(
                existing, codec="dag-json"
            )
            if recomputed != cid:
                raise ReceiptStoreIntegrityError(f"stored bytes do not match {cid}")
            return PutResult(
                cid=cid, created=False, codec=codec, byte_length=len(data), durable=True
            )
        _atomic_write_bytes(path, data)
        # Read-back verify.
        verified = path.read_bytes()
        if verified != data:
            raise ReceiptStoreIntegrityError(f"read-back mismatch for {cid}")
        recomputed = raw_cid(verified) if codec == "raw" else cid_for_bytes(
            verified, codec="dag-json"
        )
        if recomputed != cid:
            raise ReceiptStoreIntegrityError(f"read-back CID mismatch for {cid}")
        return PutResult(
            cid=cid, created=True, codec=codec, byte_length=len(data), durable=True
        )

    def get_bytes(self, cid: str, *, touch_access: bool = False) -> bytes:
        self._ensure_open()
        cid = _require_cid(cid, field_name="cid")
        path = self._block_path(cid)
        if not path.exists():
            raise ReceiptStoreIntegrityError(f"artifact not found: {cid}")
        _reject_symlink(path)
        data = path.read_bytes()
        # Accept either raw or dag-json codec based on CID verification.
        matched = False
        try:
            if cid_for_bytes(data, codec="raw") == cid:
                matched = True
        except MultiformatsIdentityError:
            pass
        if not matched:
            try:
                if cid_for_bytes(data, codec="dag-json") == cid:
                    matched = True
            except MultiformatsIdentityError:
                pass
        if not matched:
            raise ReceiptStoreIntegrityError(f"local bytes do not match {cid}")
        # Access tracking is explicit to avoid nested HEAD-lock deadlocks when
        # CAS reloads snapshot bodies under the generation lock.
        if touch_access:
            self.record_access(cid)
        return data

    def put_mapping(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
    ) -> PutResult:
        self._ensure_open()
        if not isinstance(artifact, Mapping):
            raise ReceiptStoreIntegrityError("artifact must be a Mapping")
        payload = dict(artifact)
        data = _canonical_json_bytes(payload)
        cid = mapping_cid(payload)
        if expected_cid is not None and expected_cid != cid:
            raise ReceiptStoreIntegrityError(
                f"artifact CID {cid} does not match expected {expected_cid}"
            )
        result = self.put_bytes(data, expected_cid=cid, codec="dag-json")
        return PutResult(
            cid=result.cid,
            created=result.created,
            codec="dag-json",
            byte_length=result.byte_length,
            durable=True,
        )

    def get_mapping(self, cid: str, *, touch_access: bool = False) -> dict[str, Any]:
        data = self.get_bytes(cid, touch_access=touch_access)
        return _loads_mapping(data)

    def put_receipt_envelope(
        self,
        body: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
        stored_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PutResult:
        envelope = build_receipt_envelope(
            body, stored_at_ms=stored_at_ms, metadata=metadata
        )
        return self.put_mapping(envelope, expected_cid=expected_cid)

    def get_receipt_envelope(self, cid: str) -> dict[str, Any]:
        payload = self.get_mapping(cid)
        if payload.get("schema") not in {None, RECEIPT_ENVELOPE_SCHEMA}:
            raise ReceiptStoreIntegrityError("receipt envelope schema mismatch")
        return payload

    def current_index(self) -> IndexSnapshot:
        self._ensure_open()
        return self._cas.current_index()

    def compare_and_swap_index(
        self,
        snapshot: IndexSnapshot,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult:
        self._ensure_open()
        return self._cas.compare_and_swap_index(
            snapshot,
            expected_generation=expected_generation,
            expected_root_cid=expected_root_cid,
        )

    def replay_history(self) -> tuple[IndexSnapshot, ...]:
        self._ensure_open()
        return self._cas.replay_history()

    def publish_tombstone(
        self,
        tombstone: TombstoneRecord,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult:
        """Append a tombstone and drop the live index entry for key_id.

        The prior receipt bytes remain immutable in the block store; the
        tombstone envelope is also stored immutably for audit.
        """

        self._ensure_open()
        if not isinstance(tombstone, TombstoneRecord):
            raise ReceiptStoreIntegrityError("tombstone must be a TombstoneRecord")
        envelope = tombstone.to_envelope()
        put = self.put_mapping(envelope)
        sealed = TombstoneRecord(
            key_id=tombstone.key_id,
            prior_receipt_cid=tombstone.prior_receipt_cid,
            reason=tombstone.reason,
            tombstoned_at_ms=tombstone.tombstoned_at_ms,
            tombstone_cid=put.cid,
            metadata=tombstone.metadata,
        )
        current = self.current_index()
        live_entries = tuple(
            entry for entry in current.entries if entry.key_id != sealed.key_id
        )
        new_tombstones = tuple(current.tombstones) + (sealed,)
        next_snapshot = IndexSnapshot(
            generation=expected_generation + 1,
            entries=live_entries,
            tombstones=new_tombstones,
            previous_root_cid=expected_root_cid,
            created_at_ms=_now_ms(self._clock),
        )
        return self.compare_and_swap_index(
            next_snapshot,
            expected_generation=expected_generation,
            expected_root_cid=expected_root_cid,
        )

    def record_access(self, cid: str, *, at_ms: int | None = None) -> GCMetadata:
        self._ensure_open()
        return self._cas.record_access(cid, at_ms=at_ms)

    def collect_gc_metadata(self) -> tuple[GCMetadata, ...]:
        self._ensure_open()
        return self._cas.collect_gc_metadata()

    def recover(self, *, rebuild: bool = True) -> RecoverResult:
        self._ensure_open()
        errors: list[dict[str, str]] = []
        verified = 0
        for path in sorted(self.blocks_dir.glob("*/*.block")):
            try:
                _reject_symlink(path)
                data = path.read_bytes()
                cid = path.stem
                matched_codec: str | None = None
                for codec_name in ("raw", "dag-json"):
                    try:
                        if cid_for_bytes(data, codec=codec_name) == cid:
                            matched_codec = codec_name
                            break
                    except MultiformatsIdentityError:
                        continue
                if matched_codec is None:
                    raise ReceiptStoreIntegrityError("CID mismatch")
                if matched_codec == "dag-json":
                    _loads_mapping(data)
                verified += 1
            except Exception as exc:  # noqa: BLE001 - collect all corruption
                errors.append({"cid": path.stem, "error": str(exc)})
        if errors:
            # Fail closed: never silently downgrade corruption.
            raise ReceiptStoreIntegrityError(
                f"receipt store recovery found corrupt blocks: {errors}"
            )
        history = self.replay_history() if rebuild else ()
        # Re-validate HEAD against history when present.
        if rebuild and self._cas.head_path.exists():
            current = self.current_index()
            if current.generation > 0:
                # Ensure history contains the current generation.
                by_gen = {item.generation: item for item in history}
                if (
                    current.generation in by_gen
                    and by_gen[current.generation].root_cid != current.root_cid
                ):
                    raise ReceiptStoreIntegrityError(
                        "HEAD root diverges from history snapshot"
                    )
        return RecoverResult(
            verified_blocks=verified,
            rebuilt=rebuild,
            errors=(),
            history_generations=len(history),
        )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Ipfs-kit adapter
# ---------------------------------------------------------------------------


class IpfsKitVerificationReceiptStore:
    """Optional adapter over DurableCoordinationStore + local generation CAS.

    Construction probes the exact leaf module/symbol.  When the backend is not
    operational the instance remains constructible only through
    :meth:`try_open`, which returns :class:`StoreUnavailable`.

    Head CAS is the local generation protocol under ``storage_dir``.  An
    optional injected ``iroh_manifest_bridge`` may mirror CAS when that exact
    async capability is operational; it is never mocked and never required.
    """

    def __init__(
        self,
        storage_dir: os.PathLike[str] | str,
        *,
        coordination_store: Any | None = None,
        iroh_manifest_bridge: Any | None = None,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        clock: Callable[[], int] | None = None,
        _probe: DurableCoordinationProbe | None = None,
        _allow_unavailable: bool = False,
    ) -> None:
        if storage_dir is None or str(storage_dir).strip() == "":
            unavailable = StoreUnavailable(
                reason="IpfsKitVerificationReceiptStore requires an explicit storage_dir",
                code=StoreUnavailableCode.STORAGE_ROOT_REQUIRED,
            )
            if _allow_unavailable:
                self._unavailable = unavailable
                self._storage_dir = Path(".")
                self._coord = None
                self._cas = None
                self._iroh = None
                self._closed = True
                self._local_blocks: HermeticVerificationReceiptStore | None = None
                return
            raise ReceiptStoreUnavailableError(unavailable)

        self._storage_dir = Path(storage_dir).expanduser().resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.max_blob_bytes = int(max_blob_bytes)
        self._clock = clock
        self._iroh = iroh_manifest_bridge
        self._closed = False
        self._unavailable: StoreUnavailable | None = None

        probe = _probe if _probe is not None else probe_durable_coordination_store()
        self.probe_result = probe

        if coordination_store is not None:
            self._coord = coordination_store
        elif probe.available:
            module = importlib.import_module(DURABLE_COORDINATION_LEAF_MODULE)
            store_cls = getattr(module, DURABLE_COORDINATION_SYMBOL)
            # Explicit storage root only — never home defaults.
            self._coord = store_cls(storage_dir=str(self._storage_dir / "coordination"))
        else:
            self._unavailable = probe.unavailable or StoreUnavailable(
                reason="DurableCoordinationStore unavailable",
                code=StoreUnavailableCode.ABSENT_BACKEND,
            )
            if not _allow_unavailable:
                raise ReceiptStoreUnavailableError(self._unavailable)
            self._coord = None

        # Local raw-byte blocks + generation CAS (not kit SQLite CAS).
        self._local_blocks = HermeticVerificationReceiptStore(
            self._storage_dir / "local",
            lock_timeout_seconds=lock_timeout_seconds,
            max_blob_bytes=max_blob_bytes,
            clock=clock,
        )
        self._cas = _LocalGenerationCAS(
            self._storage_dir,
            put_mapping=self.put_mapping,
            get_mapping=self.get_mapping,
            lock_timeout_seconds=lock_timeout_seconds,
            clock=clock,
        )

    @classmethod
    def try_open(
        cls,
        storage_dir: os.PathLike[str] | str,
        **kwargs: Any,
    ) -> IpfsKitVerificationReceiptStore | StoreUnavailable:
        """Open the adapter or return typed unavailable without raising."""

        probe = probe_durable_coordination_store()
        if not probe.available and kwargs.get("coordination_store") is None:
            return probe.unavailable or StoreUnavailable(
                reason="DurableCoordinationStore unavailable",
                code=StoreUnavailableCode.ABSENT_BACKEND,
            )
        try:
            return cls(storage_dir, _probe=probe, **kwargs)
        except ReceiptStoreUnavailableError as exc:
            return exc.unavailable

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @property
    def available(self) -> bool:
        return self._unavailable is None and self._coord is not None

    def _ensure_available(self) -> None:
        if self._closed:
            raise ReceiptStoreError("store is closed")
        if self._unavailable is not None:
            self._unavailable.raise_error()
        if self._coord is None:
            raise ReceiptStoreUnavailableError(
                StoreUnavailable(
                    reason="DurableCoordinationStore is not operational",
                    code=StoreUnavailableCode.NOT_OPERATIONAL,
                )
            )

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_cid: str | None = None,
        codec: str = "raw",
    ) -> PutResult:
        self._ensure_available()
        assert self._local_blocks is not None
        # Raw bytes live in the hermetic local tree; Mapping adapters use kit.
        return self._local_blocks.put_bytes(
            data, expected_cid=expected_cid, codec=codec
        )

    def get_bytes(self, cid: str) -> bytes:
        self._ensure_available()
        assert self._local_blocks is not None
        try:
            return self._local_blocks.get_bytes(cid)
        except ReceiptStoreIntegrityError:
            # Fall back to coordination Mapping envelope body.
            payload = self.get_mapping(cid)
            if payload.get("schema") == RAW_BYTES_ENVELOPE_SCHEMA:
                raw = base64.b64decode(str(payload["data_b64"]).encode("ascii"))
                if raw_cid(raw) != payload.get("content_cid"):
                    raise ReceiptStoreIntegrityError("raw envelope content_cid mismatch")
                return raw
            # Return canonical JSON bytes of the Mapping.
            data = _canonical_json_bytes(payload)
            if mapping_cid(payload) != cid:
                raise ReceiptStoreIntegrityError(f"mapping bytes do not match {cid}")
            return data

    def put_mapping(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
    ) -> PutResult:
        self._ensure_available()
        if not isinstance(artifact, Mapping):
            raise ReceiptStoreIntegrityError("artifact must be a Mapping")
        payload = dict(artifact)
        cid = mapping_cid(payload)
        if expected_cid is not None and expected_cid != cid:
            raise ReceiptStoreIntegrityError(
                f"artifact CID {cid} does not match expected {expected_cid}"
            )
        # DurableCoordinationStore Mapping put (immutable CID artifacts).
        result = self._coord.put(payload, expected_cid=cid, codec="dag-json")
        result_cid = str(result.get("cid") or cid)
        if result_cid != cid:
            raise ReceiptStoreIntegrityError(
                f"coordination store returned {result_cid}, expected {cid}"
            )
        # Mirror into local blocks so hermetic reopen/CAS share vectors.
        assert self._local_blocks is not None
        self._local_blocks.put_mapping(payload, expected_cid=cid)
        return PutResult(
            cid=cid,
            created=bool(result.get("created", True)),
            codec="dag-json",
            byte_length=int(result.get("byte_length") or len(_canonical_json_bytes(payload))),
            durable=bool(result.get("durable", True)),
            replicated=bool(result.get("replicated", False)),
        )

    def get_mapping(self, cid: str) -> dict[str, Any]:
        self._ensure_available()
        cid = _require_cid(cid, field_name="cid")
        # Prefer coordination store; fall back to local mirror.
        try:
            value = self._coord.get(cid)
            if not isinstance(value, dict):
                raise ReceiptStoreIntegrityError("coordination get returned non-mapping")
            if mapping_cid(value) != cid:
                raise ReceiptStoreIntegrityError(f"coordination mapping CID mismatch for {cid}")
            return value
        except (
            AttributeError,
            KeyError,
            OSError,
            ReceiptStoreError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            assert self._local_blocks is not None
            try:
                return self._local_blocks.get_mapping(cid)
            except ReceiptStoreIntegrityError:
                raise ReceiptStoreIntegrityError(
                    f"artifact not found or corrupt: {cid}"
                ) from exc

    def put_receipt_envelope(
        self,
        body: Mapping[str, Any],
        *,
        expected_cid: str | None = None,
        stored_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PutResult:
        envelope = build_receipt_envelope(
            body, stored_at_ms=stored_at_ms, metadata=metadata
        )
        return self.put_mapping(envelope, expected_cid=expected_cid)

    def get_receipt_envelope(self, cid: str) -> dict[str, Any]:
        payload = self.get_mapping(cid)
        if payload.get("schema") not in {None, RECEIPT_ENVELOPE_SCHEMA}:
            raise ReceiptStoreIntegrityError("receipt envelope schema mismatch")
        return payload

    def current_index(self) -> IndexSnapshot:
        self._ensure_available()
        assert self._cas is not None
        return self._cas.current_index()

    def compare_and_swap_index(
        self,
        snapshot: IndexSnapshot,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult:
        self._ensure_available()
        assert self._cas is not None
        result = self._cas.compare_and_swap_index(
            snapshot,
            expected_generation=expected_generation,
            expected_root_cid=expected_root_cid,
        )
        # Optional operational Iroh bridge — never mock; typed unavailable if asked.
        if (
            result.success
            and self._iroh is not None
            and getattr(self._iroh, "mirror_cas", None) is not None
        ):
            try:
                mirror = self._iroh.mirror_cas
                if callable(mirror):
                    mirror_result = mirror(
                        generation=result.generation,
                        root_cid=result.root_cid,
                        expected_generation=expected_generation,
                        expected_root_cid=expected_root_cid,
                    )
                    # Async bridge: caller may inject a coroutine function; we
                    # only accept already-resolved results here.
                    if hasattr(mirror_result, "cr_running"):
                        return CompareAndSwapResult(
                            success=False,
                            conflict=False,
                            generation=result.generation,
                            root_cid=result.root_cid,
                            expected_generation=expected_generation,
                            expected_root_cid=expected_root_cid,
                            snapshot=result.snapshot,
                            unavailable=StoreUnavailable(
                                reason="Iroh manifest bridge returned an unresolved coroutine",
                                code=StoreUnavailableCode.CAS_UNAVAILABLE,
                            ),
                            reason="iroh_bridge_unresolved",
                        )
            except Exception as exc:  # noqa: BLE001
                return CompareAndSwapResult(
                    success=False,
                    conflict=False,
                    generation=result.generation,
                    root_cid=result.root_cid,
                    expected_generation=expected_generation,
                    expected_root_cid=expected_root_cid,
                    snapshot=result.snapshot,
                    unavailable=StoreUnavailable(
                        reason=f"Iroh manifest bridge CAS unavailable: {exc}",
                        code=StoreUnavailableCode.BRIDGE_UNAVAILABLE,
                        detail={"error": str(exc)},
                    ),
                    reason="iroh_bridge_unavailable",
                )
        return result

    def iroh_cas_or_unavailable(self) -> CompareAndSwapResult | StoreUnavailable:
        """Explicit path: request CAS via bridge only when operational."""

        if self._iroh is None or not callable(getattr(self._iroh, "mirror_cas", None)):
            return StoreUnavailable(
                reason="Iroh manifest bridge is not injected or not operational",
                code=StoreUnavailableCode.CAS_UNAVAILABLE,
            )
        return StoreUnavailable(
            reason="Iroh bridge present but CAS must be invoked via compare_and_swap_index",
            code=StoreUnavailableCode.CAS_UNAVAILABLE,
        )

    def replay_history(self) -> tuple[IndexSnapshot, ...]:
        self._ensure_available()
        assert self._cas is not None
        return self._cas.replay_history()

    def publish_tombstone(
        self,
        tombstone: TombstoneRecord,
        *,
        expected_generation: int,
        expected_root_cid: str | None,
    ) -> CompareAndSwapResult:
        self._ensure_available()
        assert self._local_blocks is not None
        # Reuse hermetic tombstone logic against shared CAS via local helpers.
        envelope = tombstone.to_envelope()
        put = self.put_mapping(envelope)
        sealed = TombstoneRecord(
            key_id=tombstone.key_id,
            prior_receipt_cid=tombstone.prior_receipt_cid,
            reason=tombstone.reason,
            tombstoned_at_ms=tombstone.tombstoned_at_ms,
            tombstone_cid=put.cid,
            metadata=tombstone.metadata,
        )
        current = self.current_index()
        live_entries = tuple(
            entry for entry in current.entries if entry.key_id != sealed.key_id
        )
        next_snapshot = IndexSnapshot(
            generation=expected_generation + 1,
            entries=live_entries,
            tombstones=tuple(current.tombstones) + (sealed,),
            previous_root_cid=expected_root_cid,
            created_at_ms=_now_ms(self._clock),
        )
        return self.compare_and_swap_index(
            next_snapshot,
            expected_generation=expected_generation,
            expected_root_cid=expected_root_cid,
        )

    def record_access(self, cid: str, *, at_ms: int | None = None) -> GCMetadata:
        self._ensure_available()
        assert self._cas is not None
        return self._cas.record_access(cid, at_ms=at_ms)

    def collect_gc_metadata(self) -> tuple[GCMetadata, ...]:
        self._ensure_available()
        assert self._cas is not None
        return self._cas.collect_gc_metadata()

    def recover(self, *, rebuild: bool = True) -> RecoverResult:
        self._ensure_available()
        # Coordination recover verifies Mapping blocks.
        coord_report = self._coord.recover(rebuild=rebuild)
        errors = list(coord_report.get("errors") or [])
        if errors:
            raise ReceiptStoreIntegrityError(
                f"coordination recovery found corrupt blocks: {errors}"
            )
        assert self._local_blocks is not None
        local_report = self._local_blocks.recover(rebuild=rebuild)
        history = self.replay_history() if rebuild else ()
        return RecoverResult(
            verified_blocks=int(coord_report.get("verified_blocks") or 0)
            + local_report.verified_blocks,
            rebuilt=rebuild,
            errors=(),
            history_generations=len(history),
        )

    def close(self) -> None:
        self._closed = True
        if self._local_blocks is not None:
            self._local_blocks.close()
        close = getattr(self._coord, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CAS helper for concurrent writers (merge-retry without overwriting peers)
# ---------------------------------------------------------------------------


def cas_publish_entry(
    store: VerificationReceiptStore,
    entry: IndexEntry,
    *,
    max_retries: int = 32,
    clock: Callable[[], int] | None = None,
) -> CompareAndSwapResult:
    """Publish one index entry with CAS retry that preserves peer entries."""

    last: CompareAndSwapResult | None = None
    for _ in range(max_retries):
        current = store.current_index()
        expected_generation = current.generation
        expected_root = current.root_cid if current.generation > 0 else None
        merged = {item.key_id: item for item in current.entries}
        merged[entry.key_id] = entry
        snapshot = IndexSnapshot(
            generation=expected_generation + 1,
            entries=tuple(merged[key] for key in sorted(merged)),
            tombstones=current.tombstones,
            previous_root_cid=expected_root,
            created_at_ms=_now_ms(clock),
        )
        last = store.compare_and_swap_index(
            snapshot,
            expected_generation=expected_generation,
            expected_root_cid=expected_root,
        )
        if last.success:
            return last
        if not last.conflict:
            return last
        # Conflict: retry from freshly read root; never overwrite peer.
    assert last is not None
    return last


__all__ = [
    "CONCURRENT_STORE_CAS_EVIDENCE",
    "DURABLE_COORDINATION_LEAF_MODULE",
    "DURABLE_COORDINATION_SYMBOL",
    "INDEX_SNAPSHOT_SCHEMA",
    "RECEIPT_ENVELOPE_SCHEMA",
    "STORE_PROTOCOL_EVIDENCE",
    "TOMBSTONE_SCHEMA",
    "VERIFICATION_RECEIPT_STORE_INTERFACE",
    "CompareAndSwapResult",
    "DurableCoordinationProbe",
    "GCMetadata",
    "HermeticVerificationReceiptStore",
    "IndexEntry",
    "IndexSnapshot",
    "IpfsKitVerificationReceiptStore",
    "PutResult",
    "ReceiptStoreConflictError",
    "ReceiptStoreError",
    "ReceiptStoreIntegrityError",
    "ReceiptStoreUnavailableError",
    "RecoverResult",
    "StoreUnavailable",
    "StoreUnavailableCode",
    "TombstoneRecord",
    "VerificationReceiptStore",
    "build_raw_bytes_envelope",
    "build_receipt_envelope",
    "cas_publish_entry",
    "mapping_cid",
    "probe_durable_coordination_store",
    "raw_cid",
]
