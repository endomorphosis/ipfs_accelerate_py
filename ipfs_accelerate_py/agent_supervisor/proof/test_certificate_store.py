"""Immutable certificate CAS, fenced locator indexes, and atomic publication.

``TestCertificateStore@1`` owns physical persistence for proof-backed test
reuse.  Authority still lives in :mod:`test_proof_cache`: this module never
treats index metadata, provider labels, or serialized ``trusted`` flags as
pass authority.

Write protocol (fail-closed):

1. Bound and retain exact canonical DAG-JSON bytes.
2. Write them to a temporary file under the store root, fsync, and
   ``os.replace`` into the CID path.
3. Read the published path back and rehash; only then may the locator index
   be updated.
4. Index publication is controller-fenced so parallel writers cannot mix
   receipt and certificate identity from different authorities.

Reads reject symlinks, path escapes, oversized or partial files, and
rehash mismatches.  Corrupt artifacts are quarantined where safe and surface
as typed misses.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Iterator

from .formal_verification_contracts import canonical_json_bytes, content_identity
from .test_execution_contracts import (
    TestExecutionContractError,
    TestPassReceipt,
    TestProofCertificate,
)

TEST_CERTIFICATE_STORE_INTERFACE: Final = "TestCertificateStore@1"
TEST_CERTIFICATE_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-certificate-store@1"
)
TEST_CERTIFICATE_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-certificate-index@1"
)
CERTIFICATE_WRITE_FENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/certificate-write-fence@1"
)

DEFAULT_MAX_BLOB_BYTES: Final = 1_048_576
DEFAULT_MAX_CANDIDATES: Final = 32
DEFAULT_MAX_INDEX_BYTES: Final = 256 * 1024
DEFAULT_INDEX_TTL_MS: Final = 7 * 24 * 60 * 60 * 1000
DEFAULT_FENCE_TTL_MS: Final = 30_000
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 30.0

_CID_SAFE_RE: Final = re.compile(r"^[a-z0-9]+$")
_BLOB_SUFFIX: Final = ".blob"
_INDEX_SUFFIX: Final = ".json"
_TMP_PREFIX: Final = ".tmp."
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class CertificateStoreError(RuntimeError):
    """Base class for certificate-store operational failures."""


class CertificateStoreIntegrityError(CertificateStoreError, ValueError):
    """Blob or index failed integrity / path safety checks."""


class CertificateStoreFenceError(CertificateStoreError):
    """Write fence ownership was lost or mismatched."""


class CertificateStoreStatus(StrEnum):
    """Closed store / lookup result states."""

    STORED = "stored"
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"
    ERROR = "error"

    __test__ = False


class CertificateStoreReason(StrEnum):
    """Typed reasons for store admissions and safe misses."""

    OK = "ok"
    OVER_BUDGET = "over_budget"
    MALFORMED = "malformed"
    INTEGRITY_FAILED = "integrity_failed"
    PATH_ESCAPE = "path_escape"
    SYMLINK_REJECTED = "symlink_rejected"
    CORRUPT = "corrupt"
    PARTIAL = "partial"
    QUARANTINED = "quarantined"
    FENCED = "fenced"
    FENCE_MISMATCH = "fence_mismatch"
    FENCE_EXPIRED = "fence_expired"
    REVOKED = "revoked"
    EXPIRED = "expired"
    UNAVAILABLE = "unavailable"
    CANDIDATE_MISSING = "candidate_missing"
    CID_MISMATCH = "cid_mismatch"
    ALREADY_EXISTS = "already_exists"

    __test__ = False


def _now_ms(clock: Callable[[], float] | Callable[[], int] | None = None) -> int:
    """Return milliseconds since epoch.

    ``clock`` matches other supervisor stores: ``time.time``-style seconds.
    """

    if clock is None:
        return int(time.time() * 1000)
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CertificateStoreError("clock must return a numeric timestamp")
    return int(float(value) * 1000)


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_lock(
    path: Path, *, timeout_seconds: float
) -> Iterator[None]:
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
                    raise TimeoutError(
                        f"timed out acquiring process lock: {path}"
                    )
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        lock.release()


def _validate_cid_token(cid: str, *, field_name: str = "cid") -> str:
    if not isinstance(cid, str) or not cid:
        raise CertificateStoreIntegrityError(f"{field_name} must be a nonempty string")
    token = cid.strip()
    if token != cid or token != token.lower():
        raise CertificateStoreIntegrityError(
            f"{field_name} must be lowercase canonical form"
        )
    if ".." in token or "/" in token or "\\" in token or "\x00" in token:
        raise CertificateStoreIntegrityError(
            f"{field_name} path-escape characters are rejected"
        )
    if not _CID_SAFE_RE.match(token):
        raise CertificateStoreIntegrityError(
            f"{field_name} contains disallowed characters"
        )
    if len(token) < 8 or len(token) > 128:
        raise CertificateStoreIntegrityError(f"{field_name} length is out of bounds")
    return token


def _cid_for_canonical_bytes(data: bytes) -> str:
    """Derive the DAG-JSON CIDv1 for retained exact canonical bytes."""

    if type(data) is not bytes:
        raise CertificateStoreIntegrityError("canonical bytes must be exact bytes")
    if not data:
        raise CertificateStoreIntegrityError("canonical bytes must be nonempty")
    try:
        text = data.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificateStoreIntegrityError(
            "canonical bytes are not valid UTF-8 DAG-JSON"
        ) from exc
    try:
        recomputed = canonical_json_bytes(payload)
    except Exception as exc:
        raise CertificateStoreIntegrityError(
            "canonical bytes failed re-canonicalization"
        ) from exc
    if recomputed != data:
        raise CertificateStoreIntegrityError(
            "retained bytes are not canonical DAG-JSON form"
        )
    try:
        return content_identity(payload)
    except Exception as exc:
        raise CertificateStoreIntegrityError(
            "failed to derive content identity for retained bytes"
        ) from exc


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


@dataclass(frozen=True)
class CasPutResult:
    """Outcome of an immutable CAS put."""

    __test__: ClassVar[bool] = False

    stored: bool
    cid: str
    reason_code: CertificateStoreReason
    byte_length: int = 0
    path: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.stored


@dataclass(frozen=True)
class CasGetResult:
    """Outcome of an immutable CAS get; never promotes authority."""

    __test__: ClassVar[bool] = False

    status: CertificateStoreStatus
    cid: str
    reason_code: CertificateStoreReason
    data: bytes | None = None
    byte_length: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.status is CertificateStoreStatus.HIT and self.data is not None

    @property
    def hit(self) -> bool:
        return bool(self)


@dataclass(frozen=True)
class IndexCandidate:
    """One locator-index hint; never pass authority by itself."""

    __test__: ClassVar[bool] = False

    certificate_cid: str
    receipt_cid: str
    created_at_ms: int
    expires_at_ms: int | None = None
    revoked: bool = False
    quarantined: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "certificate_cid": self.certificate_cid,
            "receipt_cid": self.receipt_cid,
            "created_at_ms": self.created_at_ms,
            "revoked": bool(self.revoked),
            "quarantined": bool(self.quarantined),
        }
        if self.expires_at_ms is not None:
            payload["expires_at_ms"] = self.expires_at_ms
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndexCandidate":
        if not isinstance(payload, Mapping):
            raise CertificateStoreIntegrityError("index candidate must be an object")
        certificate_cid = _validate_cid_token(
            str(payload.get("certificate_cid", "")), field_name="certificate_cid"
        )
        receipt_cid = _validate_cid_token(
            str(payload.get("receipt_cid", "")), field_name="receipt_cid"
        )
        created = payload.get("created_at_ms")
        if isinstance(created, bool) or not isinstance(created, int) or created < 0:
            raise CertificateStoreIntegrityError("created_at_ms is invalid")
        expires = payload.get("expires_at_ms")
        if expires is not None and (
            isinstance(expires, bool) or not isinstance(expires, int) or expires < 0
        ):
            raise CertificateStoreIntegrityError("expires_at_ms is invalid")
        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise CertificateStoreIntegrityError("candidate metadata must be a mapping")
        return cls(
            certificate_cid=certificate_cid,
            receipt_cid=receipt_cid,
            created_at_ms=created,
            expires_at_ms=expires,
            revoked=bool(payload.get("revoked", False)),
            quarantined=bool(payload.get("quarantined", False)),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class IndexPublishResult:
    __test__: ClassVar[bool] = False

    published: bool
    locator_cid: str
    reason_code: CertificateStoreReason
    candidate_count: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.published


@dataclass(frozen=True)
class IndexLookupResult:
    __test__: ClassVar[bool] = False

    status: CertificateStoreStatus
    locator_cid: str
    reason_code: CertificateStoreReason
    candidates: tuple[IndexCandidate, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.status is CertificateStoreStatus.HIT and bool(self.candidates)


@dataclass(frozen=True)
class FenceLease:
    """Fenced single-flight lease for one publication key."""

    __test__: ClassVar[bool] = False

    key: str
    owner_id: str
    token: str
    fencing_token: int
    acquired_at_ms: int
    expires_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CERTIFICATE_WRITE_FENCE_SCHEMA,
            "key": self.key,
            "owner_id": self.owner_id,
            "token": self.token,
            "fencing_token": self.fencing_token,
            "acquired_at_ms": self.acquired_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }


@dataclass(frozen=True)
class StorePutResult:
    """Facade put outcome after CAS + optional fenced index publication."""

    __test__: ClassVar[bool] = False

    stored: bool
    reason_code: CertificateStoreReason
    receipt_cid: str = ""
    certificate_cid: str = ""
    locator_cid: str = ""
    indexed: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.stored


@dataclass(frozen=True)
class StoreLookupResult:
    """Facade candidate materialization for trust-aware admission."""

    __test__: ClassVar[bool] = False

    status: CertificateStoreStatus
    reason_code: CertificateStoreReason
    candidates: tuple[Mapping[str, Any], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.status is CertificateStoreStatus.HIT and bool(self.candidates)


class ImmutableCertificateCAS:
    """Local immutable content-addressed store for canonical proof bytes.

    Paths are always derived from validated CID tokens under ``root/cas``.
    Symlinks and path escapes never resolve into authority inputs.
    """

    __test__ = False

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], float] | Callable[[], int] | None = None,
    ) -> None:
        if (
            isinstance(max_blob_bytes, bool)
            or not isinstance(max_blob_bytes, int)
            or max_blob_bytes <= 0
        ):
            raise ValueError("max_blob_bytes must be a positive integer")
        self.root = Path(root).resolve()
        self.cas_root = self.root / "cas"
        self.quarantine_root = self.root / "quarantine"
        self.lock_path = self.root / "locks" / "cas.lock"
        self.max_blob_bytes = int(max_blob_bytes)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._clock = clock
        self.cas_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.quarantine_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._scrub_partial_temporaries()

    def _shard_dir(self, base: Path, cid: str) -> Path:
        # Two-character shard keeps directory fan-out bounded without nesting.
        return base / cid[:2]

    def blob_path(self, cid: str) -> Path:
        token = _validate_cid_token(cid)
        return self._shard_dir(self.cas_root, token) / f"{token}{_BLOB_SUFFIX}"

    def quarantine_path(self, cid: str) -> Path:
        token = _validate_cid_token(cid)
        return self._shard_dir(self.quarantine_root, token) / f"{token}{_BLOB_SUFFIX}"

    def _scrub_partial_temporaries(self) -> None:
        """Drop abandoned ``.tmp.*`` files left by interrupted writers."""

        for base in (self.cas_root, self.quarantine_root):
            if not base.is_dir():
                continue
            for path in base.rglob(f"{_TMP_PREFIX}*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        path.unlink()
                except OSError:
                    continue

    def _ensure_inside(self, path: Path, *, base: Path) -> Path:
        """Resolve ``path`` without following a final symlink; reject escapes."""

        base_resolved = base.resolve()
        # Walk parents and reject any symlink component before final open.
        try:
            relative = path.relative_to(base_resolved)
        except ValueError as exc:
            # path may still be unresolved; build under base only.
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.PATH_ESCAPE.value
            ) from exc
        current = base_resolved
        parts = relative.parts
        if not parts:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.PATH_ESCAPE.value
            )
        for part in parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.SYMLINK_REJECTED.value
                )
            if not current.exists():
                break
            try:
                current.resolve().relative_to(base_resolved)
            except ValueError as exc:
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.PATH_ESCAPE.value
                ) from exc
        final = current / parts[-1] if parts[:-1] else base_resolved / parts[-1]
        # Rebuild from base + relative to avoid relying on string path alone.
        final = base_resolved.joinpath(*parts)
        if final.is_symlink():
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.SYMLINK_REJECTED.value
            )
        if final.exists():
            try:
                final.resolve(strict=True).relative_to(base_resolved)
            except ValueError as exc:
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.PATH_ESCAPE.value
                ) from exc
            except OSError as exc:
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.CORRUPT.value
                ) from exc
        else:
            try:
                final.parent.resolve().relative_to(base_resolved)
            except ValueError as exc:
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.PATH_ESCAPE.value
                ) from exc
        return final

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{_TMP_PREFIX}{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            if temporary.is_symlink():
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.SYMLINK_REJECTED.value
                )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def put_bytes(
        self,
        data: bytes,
        *,
        claimed_cid: str | None = None,
        allow_existing: bool = True,
    ) -> CasPutResult:
        """Atomically publish bounded canonical bytes under their content CID."""

        if type(data) is not bytes:
            return CasPutResult(
                False, claimed_cid or "", CertificateStoreReason.MALFORMED
            )
        if not data:
            return CasPutResult(
                False, claimed_cid or "", CertificateStoreReason.MALFORMED
            )
        if len(data) > self.max_blob_bytes:
            return CasPutResult(
                False,
                claimed_cid or "",
                CertificateStoreReason.OVER_BUDGET,
                byte_length=len(data),
            )
        try:
            derived_cid = _cid_for_canonical_bytes(data)
        except CertificateStoreIntegrityError:
            return CasPutResult(
                False,
                claimed_cid or "",
                CertificateStoreReason.MALFORMED,
                byte_length=len(data),
            )
        if claimed_cid is not None:
            try:
                claimed = _validate_cid_token(claimed_cid, field_name="claimed_cid")
            except CertificateStoreIntegrityError:
                return CasPutResult(
                    False, claimed_cid, CertificateStoreReason.PATH_ESCAPE
                )
            if claimed != derived_cid:
                return CasPutResult(
                    False,
                    claimed,
                    CertificateStoreReason.CID_MISMATCH,
                    byte_length=len(data),
                )
        cid = derived_cid
        try:
            path = self._ensure_inside(self.blob_path(cid), base=self.cas_root)
        except CertificateStoreIntegrityError as exc:
            reason = CertificateStoreReason.PATH_ESCAPE
            message = str(exc)
            if message == CertificateStoreReason.SYMLINK_REJECTED.value:
                reason = CertificateStoreReason.SYMLINK_REJECTED
            return CasPutResult(False, cid, reason, byte_length=len(data))

        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            if path.exists():
                existing = self._read_and_rehash(path, expected_cid=cid)
                if existing.status is CertificateStoreStatus.HIT:
                    if allow_existing:
                        return CasPutResult(
                            True,
                            cid,
                            CertificateStoreReason.ALREADY_EXISTS,
                            byte_length=existing.byte_length,
                            path=str(path),
                        )
                    return CasPutResult(
                        False,
                        cid,
                        CertificateStoreReason.ALREADY_EXISTS,
                        byte_length=existing.byte_length,
                        path=str(path),
                    )
                # Corrupt existing blob: replace after quarantine.
                self._quarantine_locked(cid, path, reason="replace-corrupt")

            try:
                self._atomic_write(path, data)
            except CertificateStoreIntegrityError as exc:
                reason = CertificateStoreReason.CORRUPT
                if str(exc) == CertificateStoreReason.SYMLINK_REJECTED.value:
                    reason = CertificateStoreReason.SYMLINK_REJECTED
                return CasPutResult(False, cid, reason, byte_length=len(data))
            except OSError:
                return CasPutResult(
                    False, cid, CertificateStoreReason.UNAVAILABLE, byte_length=len(data)
                )

            readback = self._read_and_rehash(path, expected_cid=cid)
            if not readback.hit or readback.data != data:
                self._quarantine_locked(cid, path, reason="readback-mismatch")
                return CasPutResult(
                    False,
                    cid,
                    CertificateStoreReason.INTEGRITY_FAILED,
                    byte_length=len(data),
                    diagnostics={"stage": "readback_rehash"},
                )
            return CasPutResult(
                True,
                cid,
                CertificateStoreReason.OK,
                byte_length=len(data),
                path=str(path),
            )

    def get_bytes(self, cid: str) -> CasGetResult:
        """Read and rehash a CID blob; miss safely on any fault class."""

        try:
            token = _validate_cid_token(cid)
        except CertificateStoreIntegrityError:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                str(cid) if isinstance(cid, str) else "",
                CertificateStoreReason.PATH_ESCAPE,
            )
        try:
            path = self._ensure_inside(self.blob_path(token), base=self.cas_root)
        except CertificateStoreIntegrityError as exc:
            reason = CertificateStoreReason.PATH_ESCAPE
            if str(exc) == CertificateStoreReason.SYMLINK_REJECTED.value:
                reason = CertificateStoreReason.SYMLINK_REJECTED
            return CasGetResult(CertificateStoreStatus.MISS, token, reason)

        if not path.exists():
            # Quarantined entries are deliberate misses.
            try:
                qpath = self._ensure_inside(
                    self.quarantine_path(token), base=self.quarantine_root
                )
                if qpath.exists():
                    return CasGetResult(
                        CertificateStoreStatus.MISS,
                        token,
                        CertificateStoreReason.QUARANTINED,
                    )
            except CertificateStoreIntegrityError:
                pass
            return CasGetResult(
                CertificateStoreStatus.MISS,
                token,
                CertificateStoreReason.CANDIDATE_MISSING,
            )

        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            result = self._read_and_rehash(path, expected_cid=token)
            if result.status is CertificateStoreStatus.MISS and result.reason_code in {
                CertificateStoreReason.CORRUPT,
                CertificateStoreReason.INTEGRITY_FAILED,
                CertificateStoreReason.PARTIAL,
                CertificateStoreReason.OVER_BUDGET,
            }:
                self._quarantine_locked(token, path, reason=result.reason_code.value)
            return result

    def has(self, cid: str) -> bool:
        return bool(self.get_bytes(cid))

    def quarantine(self, cid: str, *, reason: str = "manual") -> bool:
        try:
            token = _validate_cid_token(cid)
            path = self._ensure_inside(self.blob_path(token), base=self.cas_root)
        except CertificateStoreIntegrityError:
            return False
        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            if not path.exists():
                return False
            self._quarantine_locked(token, path, reason=reason)
            return True

    def _quarantine_locked(self, cid: str, path: Path, *, reason: str) -> None:
        try:
            destination = self._ensure_inside(
                self.quarantine_path(cid), base=self.quarantine_root
            )
        except CertificateStoreIntegrityError:
            try:
                path.unlink()
            except OSError:
                pass
            return
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            if destination.exists():
                destination.unlink()
            os.replace(path, destination)
            marker = destination.with_suffix(".reason.json")
            payload = canonical_json_bytes(
                {
                    "schema": TEST_CERTIFICATE_STORE_SCHEMA,
                    "cid": cid,
                    "reason": reason,
                    "quarantined_at_ms": _now_ms(self._clock),
                }
            )
            self._atomic_write(marker, payload)
        except OSError:
            try:
                path.unlink()
            except OSError:
                pass

    def _read_and_rehash(self, path: Path, *, expected_cid: str) -> CasGetResult:
        if path.is_symlink():
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.SYMLINK_REJECTED,
            )
        try:
            stat = path.stat(follow_symlinks=False)
        except OSError:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.UNAVAILABLE,
            )
        if stat.st_size <= 0:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.PARTIAL,
            )
        if stat.st_size > self.max_blob_bytes:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.OVER_BUDGET,
                byte_length=stat.st_size,
            )
        try:
            data = path.read_bytes()
        except OSError:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.UNAVAILABLE,
            )
        if len(data) != stat.st_size:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.PARTIAL,
                byte_length=len(data),
            )
        if len(data) > self.max_blob_bytes:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.OVER_BUDGET,
                byte_length=len(data),
            )
        try:
            actual_cid = _cid_for_canonical_bytes(data)
        except CertificateStoreIntegrityError:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.CORRUPT,
                byte_length=len(data),
            )
        if actual_cid != expected_cid:
            return CasGetResult(
                CertificateStoreStatus.MISS,
                expected_cid,
                CertificateStoreReason.INTEGRITY_FAILED,
                byte_length=len(data),
                diagnostics={"actual_cid": actual_cid},
            )
        return CasGetResult(
            CertificateStoreStatus.HIT,
            expected_cid,
            CertificateStoreReason.OK,
            data=data,
            byte_length=len(data),
        )


class TestCertificateIndex:
    """Mutable locator → bounded certificate candidate hints.

    Index entries are never authority.  TTL, revocation, and quarantine only
    affect whether a CID is *suggested* for re-admission.
    """

    __test__ = False

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_index_bytes: int = DEFAULT_MAX_INDEX_BYTES,
        default_ttl_ms: int = DEFAULT_INDEX_TTL_MS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], float] | Callable[[], int] | None = None,
    ) -> None:
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")
        if (
            isinstance(max_index_bytes, bool)
            or not isinstance(max_index_bytes, int)
            or max_index_bytes <= 0
        ):
            raise ValueError("max_index_bytes must be a positive integer")
        self.root = Path(root).resolve()
        self.index_root = self.root / "index"
        self.lock_root = self.root / "locks" / "index"
        self.max_candidates = int(max_candidates)
        self.max_index_bytes = int(max_index_bytes)
        self.default_ttl_ms = int(default_ttl_ms)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._clock = clock
        self.index_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _index_path(self, locator_cid: str) -> Path:
        token = _validate_cid_token(locator_cid, field_name="locator_cid")
        return self.index_root / f"{token}{_INDEX_SUFFIX}"

    def _lock_path(self, locator_cid: str) -> Path:
        token = _validate_cid_token(locator_cid, field_name="locator_cid")
        return self.lock_root / f"{token}.lock"

    def _ensure_index_path(self, locator_cid: str) -> Path:
        path = self._index_path(locator_cid)
        base = self.index_root.resolve()
        if path.is_symlink():
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.SYMLINK_REJECTED.value
            )
        try:
            if path.exists():
                path.resolve(strict=True).relative_to(base)
            else:
                path.parent.resolve().relative_to(base)
        except ValueError as exc:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.PATH_ESCAPE.value
            ) from exc
        return path

    def _read_document(
        self, path: Path, *, locator_cid: str
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink():
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.SYMLINK_REJECTED.value
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CertificateStoreError("index unavailable") from exc
        if not raw:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.PARTIAL.value
            )
        if len(raw) > self.max_index_bytes:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.OVER_BUDGET.value
            )
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.CORRUPT.value
            ) from exc
        if not isinstance(payload, Mapping):
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.CORRUPT.value
            )
        if payload.get("schema") != TEST_CERTIFICATE_INDEX_SCHEMA:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.CORRUPT.value
            )
        if payload.get("locator_cid") != locator_cid:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.INTEGRITY_FAILED.value
            )
        return dict(payload)

    def _write_document(self, path: Path, document: Mapping[str, Any]) -> None:
        encoded = canonical_json_bytes(document)
        if len(encoded) > self.max_index_bytes:
            raise CertificateStoreIntegrityError(
                CertificateStoreReason.OVER_BUDGET.value
            )
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{_TMP_PREFIX}{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            if temporary.is_symlink():
                raise CertificateStoreIntegrityError(
                    CertificateStoreReason.SYMLINK_REJECTED.value
                )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def publish(
        self,
        locator_cid: str,
        *,
        certificate_cid: str,
        receipt_cid: str,
        created_at_ms: int | None = None,
        expires_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        fencing_token: int | None = None,
        expected_fencing_token: int | None = None,
    ) -> IndexPublishResult:
        """Atomically append/replace one candidate hint for ``locator_cid``."""

        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            certificate = _validate_cid_token(
                certificate_cid, field_name="certificate_cid"
            )
            receipt = _validate_cid_token(receipt_cid, field_name="receipt_cid")
            path = self._ensure_index_path(locator)
        except CertificateStoreIntegrityError as exc:
            reason = CertificateStoreReason.PATH_ESCAPE
            if str(exc) == CertificateStoreReason.SYMLINK_REJECTED.value:
                reason = CertificateStoreReason.SYMLINK_REJECTED
            return IndexPublishResult(False, str(locator_cid), reason)

        now = _now_ms(self._clock)
        created = now if created_at_ms is None else created_at_ms
        if isinstance(created, bool) or not isinstance(created, int) or created < 0:
            return IndexPublishResult(
                False, locator, CertificateStoreReason.MALFORMED
            )
        expires = expires_at_ms
        if expires is None and self.default_ttl_ms > 0:
            expires = created + self.default_ttl_ms
        if expires is not None and (
            isinstance(expires, bool) or not isinstance(expires, int) or expires < 0
        ):
            return IndexPublishResult(
                False, locator, CertificateStoreReason.MALFORMED
            )
        if expires is not None and expires <= created:
            return IndexPublishResult(
                False, locator, CertificateStoreReason.MALFORMED
            )

        candidate = IndexCandidate(
            certificate_cid=certificate,
            receipt_cid=receipt,
            created_at_ms=created,
            expires_at_ms=expires,
            metadata=dict(metadata or {}),
        )

        lock_path = self._lock_path(locator)
        try:
            with _exclusive_lock(
                lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                if (
                    expected_fencing_token is not None
                    and fencing_token is not None
                    and fencing_token != expected_fencing_token
                ):
                    return IndexPublishResult(
                        False, locator, CertificateStoreReason.FENCE_MISMATCH
                    )
                try:
                    document = self._read_document(path, locator_cid=locator)
                except CertificateStoreIntegrityError as exc:
                    reason_value = str(exc)
                    try:
                        reason = CertificateStoreReason(reason_value)
                    except ValueError:
                        reason = CertificateStoreReason.CORRUPT
                    # Corrupt index: start a fresh document rather than mix.
                    document = None
                    if reason is CertificateStoreReason.SYMLINK_REJECTED:
                        return IndexPublishResult(False, locator, reason)
                existing: list[IndexCandidate] = []
                if document is not None:
                    raw_candidates = document.get("candidates", [])
                    if not isinstance(raw_candidates, list):
                        existing = []
                    else:
                        for item in raw_candidates:
                            try:
                                existing.append(IndexCandidate.from_dict(item))
                            except CertificateStoreIntegrityError:
                                continue
                # Replace any prior entry for the same certificate CID so a
                # single authority mapping remains authoritative as a *hint*.
                merged = [
                    item
                    for item in existing
                    if item.certificate_cid != candidate.certificate_cid
                ]
                merged.insert(0, candidate)
                merged = merged[: self.max_candidates]
                payload = {
                    "schema": TEST_CERTIFICATE_INDEX_SCHEMA,
                    "interface": TEST_CERTIFICATE_STORE_INTERFACE,
                    "locator_cid": locator,
                    "updated_at_ms": now,
                    "candidates": [item.to_dict() for item in merged],
                }
                if fencing_token is not None:
                    payload["fencing_token"] = int(fencing_token)
                self._write_document(path, payload)
                return IndexPublishResult(
                    True,
                    locator,
                    CertificateStoreReason.OK,
                    candidate_count=len(merged),
                )
        except TimeoutError:
            return IndexPublishResult(
                False, locator, CertificateStoreReason.UNAVAILABLE
            )
        except CertificateStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                reason = CertificateStoreReason(reason_value)
            except ValueError:
                reason = CertificateStoreReason.CORRUPT
            return IndexPublishResult(False, locator, reason)
        except OSError:
            return IndexPublishResult(
                False, locator, CertificateStoreReason.UNAVAILABLE
            )

    def candidates(
        self,
        locator_cid: str,
        *,
        max_candidates: int | None = None,
        now_ms: int | None = None,
        include_revoked: bool = False,
        include_quarantined: bool = False,
    ) -> IndexLookupResult:
        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            path = self._ensure_index_path(locator)
        except CertificateStoreIntegrityError as exc:
            reason = CertificateStoreReason.PATH_ESCAPE
            if str(exc) == CertificateStoreReason.SYMLINK_REJECTED.value:
                reason = CertificateStoreReason.SYMLINK_REJECTED
            return IndexLookupResult(
                CertificateStoreStatus.MISS, str(locator_cid), reason
            )

        limit = self.max_candidates if max_candidates is None else max_candidates
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return IndexLookupResult(
                CertificateStoreStatus.MISS,
                locator,
                CertificateStoreReason.MALFORMED,
            )
        current = _now_ms(self._clock) if now_ms is None else now_ms
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            return IndexLookupResult(
                CertificateStoreStatus.MISS,
                locator,
                CertificateStoreReason.MALFORMED,
            )

        lock_path = self._lock_path(locator)
        try:
            with _exclusive_lock(
                lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                try:
                    document = self._read_document(path, locator_cid=locator)
                except CertificateStoreIntegrityError as exc:
                    reason_value = str(exc)
                    try:
                        reason = CertificateStoreReason(reason_value)
                    except ValueError:
                        reason = CertificateStoreReason.CORRUPT
                    return IndexLookupResult(
                        CertificateStoreStatus.MISS, locator, reason
                    )
                if document is None:
                    return IndexLookupResult(
                        CertificateStoreStatus.MISS,
                        locator,
                        CertificateStoreReason.CANDIDATE_MISSING,
                    )
                raw_candidates = document.get("candidates", [])
                if not isinstance(raw_candidates, list):
                    return IndexLookupResult(
                        CertificateStoreStatus.MISS,
                        locator,
                        CertificateStoreReason.CORRUPT,
                    )
                admitted: list[IndexCandidate] = []
                for item in raw_candidates:
                    if len(admitted) >= limit:
                        break
                    try:
                        candidate = IndexCandidate.from_dict(item)
                    except CertificateStoreIntegrityError:
                        continue
                    if candidate.revoked and not include_revoked:
                        continue
                    if candidate.quarantined and not include_quarantined:
                        continue
                    if (
                        candidate.expires_at_ms is not None
                        and candidate.expires_at_ms <= current
                    ):
                        continue
                    admitted.append(candidate)
                if not admitted:
                    return IndexLookupResult(
                        CertificateStoreStatus.MISS,
                        locator,
                        CertificateStoreReason.CANDIDATE_MISSING,
                    )
                return IndexLookupResult(
                    CertificateStoreStatus.HIT,
                    locator,
                    CertificateStoreReason.OK,
                    candidates=tuple(admitted),
                )
        except TimeoutError:
            return IndexLookupResult(
                CertificateStoreStatus.MISS,
                locator,
                CertificateStoreReason.UNAVAILABLE,
            )
        except OSError:
            return IndexLookupResult(
                CertificateStoreStatus.MISS,
                locator,
                CertificateStoreReason.UNAVAILABLE,
            )

    def revoke_certificate(
        self, locator_cid: str, certificate_cid: str
    ) -> IndexPublishResult:
        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            certificate = _validate_cid_token(
                certificate_cid, field_name="certificate_cid"
            )
            path = self._ensure_index_path(locator)
        except CertificateStoreIntegrityError as exc:
            reason = CertificateStoreReason.PATH_ESCAPE
            if str(exc) == CertificateStoreReason.SYMLINK_REJECTED.value:
                reason = CertificateStoreReason.SYMLINK_REJECTED
            return IndexPublishResult(False, str(locator_cid), reason)

        lock_path = self._lock_path(locator)
        try:
            with _exclusive_lock(
                lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                document = self._read_document(path, locator_cid=locator)
                if document is None:
                    return IndexPublishResult(
                        False, locator, CertificateStoreReason.CANDIDATE_MISSING
                    )
                changed = False
                candidates: list[dict[str, Any]] = []
                for item in document.get("candidates", []):
                    if not isinstance(item, Mapping):
                        continue
                    entry = dict(item)
                    if entry.get("certificate_cid") == certificate:
                        entry["revoked"] = True
                        changed = True
                    candidates.append(entry)
                if not changed:
                    return IndexPublishResult(
                        False, locator, CertificateStoreReason.CANDIDATE_MISSING
                    )
                document = dict(document)
                document["candidates"] = candidates
                document["updated_at_ms"] = _now_ms(self._clock)
                self._write_document(path, document)
                return IndexPublishResult(
                    True,
                    locator,
                    CertificateStoreReason.REVOKED,
                    candidate_count=len(candidates),
                )
        except CertificateStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                reason = CertificateStoreReason(reason_value)
            except ValueError:
                reason = CertificateStoreReason.CORRUPT
            return IndexPublishResult(False, locator, reason)
        except (TimeoutError, OSError):
            return IndexPublishResult(
                False, locator, CertificateStoreReason.UNAVAILABLE
            )

    def quarantine_certificate(
        self, locator_cid: str, certificate_cid: str
    ) -> IndexPublishResult:
        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            certificate = _validate_cid_token(
                certificate_cid, field_name="certificate_cid"
            )
            path = self._ensure_index_path(locator)
        except CertificateStoreIntegrityError as exc:
            reason = CertificateStoreReason.PATH_ESCAPE
            if str(exc) == CertificateStoreReason.SYMLINK_REJECTED.value:
                reason = CertificateStoreReason.SYMLINK_REJECTED
            return IndexPublishResult(False, str(locator_cid), reason)

        lock_path = self._lock_path(locator)
        try:
            with _exclusive_lock(
                lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                document = self._read_document(path, locator_cid=locator)
                if document is None:
                    return IndexPublishResult(
                        False, locator, CertificateStoreReason.CANDIDATE_MISSING
                    )
                changed = False
                candidates: list[dict[str, Any]] = []
                for item in document.get("candidates", []):
                    if not isinstance(item, Mapping):
                        continue
                    entry = dict(item)
                    if entry.get("certificate_cid") == certificate:
                        entry["quarantined"] = True
                        changed = True
                    candidates.append(entry)
                if not changed:
                    return IndexPublishResult(
                        False, locator, CertificateStoreReason.CANDIDATE_MISSING
                    )
                document = dict(document)
                document["candidates"] = candidates
                document["updated_at_ms"] = _now_ms(self._clock)
                self._write_document(path, document)
                return IndexPublishResult(
                    True,
                    locator,
                    CertificateStoreReason.QUARANTINED,
                    candidate_count=len(candidates),
                )
        except CertificateStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                reason = CertificateStoreReason(reason_value)
            except ValueError:
                reason = CertificateStoreReason.CORRUPT
            return IndexPublishResult(False, locator, reason)
        except (TimeoutError, OSError):
            return IndexPublishResult(
                False, locator, CertificateStoreReason.UNAVAILABLE
            )


class CertificateWriteFence:
    """Cross-process publication fence with monotonic fencing tokens.

    Only the current fence owner may publish mixed receipt/certificate
    authority for a key.  Stale writers that lose the fence cannot complete
    index publication even if their CAS puts already finished.
    """

    __test__ = False

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        default_ttl_ms: int = DEFAULT_FENCE_TTL_MS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], float] | Callable[[], int] | None = None,
    ) -> None:
        if (
            isinstance(default_ttl_ms, bool)
            or not isinstance(default_ttl_ms, int)
            or default_ttl_ms <= 0
        ):
            raise ValueError("default_ttl_ms must be a positive integer")
        self.root = Path(root).resolve()
        self.fence_root = self.root / "fence"
        self.lock_path = self.root / "locks" / "fence.lock"
        self.default_ttl_ms = int(default_ttl_ms)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._clock = clock
        self.fence_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _fence_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.fence_root / f"{digest}.json"

    def _read_lease(self, path: Path) -> FenceLease | None:
        if not path.exists() or path.is_symlink():
            return None
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return FenceLease(
                key=str(payload["key"]),
                owner_id=str(payload["owner_id"]),
                token=str(payload["token"]),
                fencing_token=int(payload["fencing_token"]),
                acquired_at_ms=int(payload["acquired_at_ms"]),
                expires_at_ms=int(payload["expires_at_ms"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _write_lease(self, path: Path, lease: FenceLease) -> None:
        encoded = canonical_json_bytes(lease.to_dict())
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{_TMP_PREFIX}fence.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def acquire(
        self,
        key: str,
        *,
        owner_id: str | None = None,
        ttl_ms: int | None = None,
    ) -> FenceLease:
        if not isinstance(key, str) or not key.strip():
            raise CertificateStoreFenceError("fence key is required")
        owner = owner_id or f"owner:{uuid.uuid4().hex}"
        ttl = self.default_ttl_ms if ttl_ms is None else ttl_ms
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
            raise ValueError("ttl_ms must be a positive integer")
        path = self._fence_path(key)
        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            now = _now_ms(self._clock)
            current = self._read_lease(path)
            next_token = 1
            if current is not None and current.expires_at_ms > now:
                if current.owner_id == owner and current.key == key:
                    renewed = FenceLease(
                        key=key,
                        owner_id=owner,
                        token=current.token,
                        fencing_token=current.fencing_token,
                        acquired_at_ms=current.acquired_at_ms,
                        expires_at_ms=now + ttl,
                    )
                    self._write_lease(path, renewed)
                    return renewed
                raise CertificateStoreFenceError(
                    CertificateStoreReason.FENCED.value
                )
            if current is not None:
                next_token = int(current.fencing_token) + 1
            lease = FenceLease(
                key=key,
                owner_id=owner,
                token=uuid.uuid4().hex,
                fencing_token=next_token,
                acquired_at_ms=now,
                expires_at_ms=now + ttl,
            )
            self._write_lease(path, lease)
            return lease

    def validate(self, lease: FenceLease) -> bool:
        path = self._fence_path(lease.key)
        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            current = self._read_lease(path)
            now = _now_ms(self._clock)
            if current is None:
                return False
            if current.expires_at_ms <= now:
                return False
            return (
                current.key == lease.key
                and current.owner_id == lease.owner_id
                and current.token == lease.token
                and current.fencing_token == lease.fencing_token
            )

    def release(self, lease: FenceLease) -> bool:
        """Drop ownership while retaining the fencing-token high-water mark.

        Keeping an expired tombstone ensures the next owner receives a strictly
        greater fencing token, so stale writers cannot re-publish under the old
        token after a clean release.
        """

        path = self._fence_path(lease.key)
        with _exclusive_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            current = self._read_lease(path)
            if current is None:
                return False
            if (
                current.owner_id != lease.owner_id
                or current.token != lease.token
                or current.fencing_token != lease.fencing_token
            ):
                return False
            now = _now_ms(self._clock)
            tombstone = FenceLease(
                key=current.key,
                owner_id="",
                token="",
                fencing_token=current.fencing_token,
                acquired_at_ms=current.acquired_at_ms,
                expires_at_ms=min(current.expires_at_ms, now),
            )
            try:
                self._write_lease(path, tombstone)
            except OSError:
                return False
            return True

    def require_valid(self, lease: FenceLease) -> None:
        if not self.validate(lease):
            raise CertificateStoreFenceError(
                CertificateStoreReason.FENCE_MISMATCH.value
            )


class TestCertificateStore:
    """Facade combining immutable CAS, locator index, and write fencing.

    ``put_candidate`` is the single publication entrypoint that enforces
    temporary-file write → atomic replace → readback rehash → fenced index
    publication.  Parallel putters for the same locator cannot publish a mixed
    receipt/certificate pair: each publication is one fenced authority unit.
    """

    __test__ = False
    interface = TEST_CERTIFICATE_STORE_INTERFACE

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_index_bytes: int = DEFAULT_MAX_INDEX_BYTES,
        index_ttl_ms: int = DEFAULT_INDEX_TTL_MS,
        fence_ttl_ms: int = DEFAULT_FENCE_TTL_MS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], float] | Callable[[], int] | None = None,
        owner_id: str | None = None,
    ) -> None:
        if root is None:
            self.root = Path(
                tempfile.mkdtemp(prefix="test-certificate-store-")
            ).resolve()
            self._ephemeral = True
        else:
            self.root = Path(root).resolve()
            self._ephemeral = False
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_blob_bytes = int(max_blob_bytes)
        self.max_candidates = int(max_candidates)
        self._clock = clock
        self.owner_id = owner_id or f"store:{uuid.uuid4().hex}"
        self.cas = ImmutableCertificateCAS(
            self.root,
            max_blob_bytes=max_blob_bytes,
            lock_timeout_seconds=lock_timeout_seconds,
            clock=clock,
        )
        self.index = TestCertificateIndex(
            self.root,
            max_candidates=max_candidates,
            max_index_bytes=max_index_bytes,
            default_ttl_ms=index_ttl_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            clock=clock,
        )
        self.fence = CertificateWriteFence(
            self.root,
            default_ttl_ms=fence_ttl_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            clock=clock,
        )

    @property
    def schema(self) -> str:
        return TEST_CERTIFICATE_STORE_SCHEMA

    def put_receipt(
        self,
        receipt: TestPassReceipt | Mapping[str, Any] | bytes,
    ) -> CasPutResult:
        try:
            data, claimed = self._coerce_receipt_bytes(receipt)
        except (CertificateStoreIntegrityError, TestExecutionContractError, TypeError, ValueError):
            return CasPutResult(False, "", CertificateStoreReason.MALFORMED)
        return self.cas.put_bytes(data, claimed_cid=claimed)

    def put_certificate(
        self,
        certificate: TestProofCertificate | Mapping[str, Any] | bytes,
    ) -> CasPutResult:
        try:
            data, claimed = self._coerce_certificate_bytes(certificate)
        except (CertificateStoreIntegrityError, TestExecutionContractError, TypeError, ValueError):
            return CasPutResult(False, "", CertificateStoreReason.MALFORMED)
        return self.cas.put_bytes(data, claimed_cid=claimed)

    def put_canonical_bytes(
        self, data: bytes, *, claimed_cid: str | None = None
    ) -> CasPutResult:
        return self.cas.put_bytes(data, claimed_cid=claimed_cid)

    def get_bytes(self, cid: str) -> CasGetResult:
        return self.cas.get_bytes(cid)

    def put_candidate(
        self,
        receipt: TestPassReceipt | Mapping[str, Any] | bytes,
        certificate: TestProofCertificate | Mapping[str, Any] | bytes,
        *,
        locator_cid: str | None = None,
        created_at_ms: int | None = None,
        expires_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        publish_index: bool = True,
        owner_id: str | None = None,
    ) -> StorePutResult:
        """Persist receipt+certificate as one fenced authority unit."""

        try:
            receipt_bytes, receipt_cid = self._coerce_receipt_bytes(receipt)
            certificate_bytes, certificate_cid = self._coerce_certificate_bytes(
                certificate
            )
        except (CertificateStoreIntegrityError, TestExecutionContractError, TypeError, ValueError):
            return StorePutResult(False, CertificateStoreReason.MALFORMED)

        if len(receipt_bytes) > self.max_blob_bytes or len(certificate_bytes) > self.max_blob_bytes:
            return StorePutResult(
                False,
                CertificateStoreReason.OVER_BUDGET,
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
            )

        # Bind locator from the receipt when not supplied.
        resolved_locator = locator_cid
        if resolved_locator is None:
            try:
                typed_receipt = TestPassReceipt.from_dict(
                    json.loads(receipt_bytes.decode("utf-8"))
                )
                resolved_locator = typed_receipt.locator_cid
            except Exception:
                return StorePutResult(
                    False,
                    CertificateStoreReason.MALFORMED,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                )
        try:
            locator = _validate_cid_token(
                resolved_locator, field_name="locator_cid"
            )
        except CertificateStoreIntegrityError:
            return StorePutResult(
                False,
                CertificateStoreReason.PATH_ESCAPE,
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                locator_cid=str(resolved_locator),
            )

        fence_key = f"locator:{locator}"
        owner = owner_id or self.owner_id
        try:
            lease = self.fence.acquire(fence_key, owner_id=owner)
        except CertificateStoreFenceError:
            return StorePutResult(
                False,
                CertificateStoreReason.FENCED,
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                locator_cid=locator,
            )
        except (TimeoutError, OSError):
            return StorePutResult(
                False,
                CertificateStoreReason.UNAVAILABLE,
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                locator_cid=locator,
            )

        try:
            # CAS writes first (immutable).  Index only after both rehash.
            receipt_put = self.cas.put_bytes(
                receipt_bytes, claimed_cid=receipt_cid
            )
            if not receipt_put.stored:
                return StorePutResult(
                    False,
                    receipt_put.reason_code,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                    locator_cid=locator,
                    diagnostics={"stage": "receipt_cas"},
                )
            certificate_put = self.cas.put_bytes(
                certificate_bytes, claimed_cid=certificate_cid
            )
            if not certificate_put.stored:
                return StorePutResult(
                    False,
                    certificate_put.reason_code,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                    locator_cid=locator,
                    diagnostics={"stage": "certificate_cas"},
                )

            # Explicit readback rehash before index publication.
            receipt_get = self.cas.get_bytes(receipt_cid)
            certificate_get = self.cas.get_bytes(certificate_cid)
            if (
                not receipt_get.hit
                or not certificate_get.hit
                or receipt_get.data != receipt_bytes
                or certificate_get.data != certificate_bytes
            ):
                return StorePutResult(
                    False,
                    CertificateStoreReason.INTEGRITY_FAILED,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                    locator_cid=locator,
                    diagnostics={"stage": "readback_rehash"},
                )

            if not publish_index:
                return StorePutResult(
                    True,
                    CertificateStoreReason.OK,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                    locator_cid=locator,
                    indexed=False,
                )

            # Fence must still be held: prevents mixed authority publication.
            try:
                self.fence.require_valid(lease)
            except CertificateStoreFenceError:
                return StorePutResult(
                    False,
                    CertificateStoreReason.FENCE_MISMATCH,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                    locator_cid=locator,
                    diagnostics={"stage": "pre_index_fence"},
                )

            publish = self.index.publish(
                locator,
                certificate_cid=certificate_cid,
                receipt_cid=receipt_cid,
                created_at_ms=created_at_ms,
                expires_at_ms=expires_at_ms,
                metadata=metadata,
                fencing_token=lease.fencing_token,
                expected_fencing_token=lease.fencing_token,
            )
            if not publish.published:
                return StorePutResult(
                    False,
                    publish.reason_code,
                    receipt_cid=receipt_cid,
                    certificate_cid=certificate_cid,
                    locator_cid=locator,
                    diagnostics={"stage": "index_publish"},
                )
            return StorePutResult(
                True,
                CertificateStoreReason.OK,
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                locator_cid=locator,
                indexed=True,
                diagnostics={"candidate_count": publish.candidate_count},
            )
        finally:
            try:
                self.fence.release(lease)
            except Exception:
                pass

    def lookup(
        self,
        locator_cid: str,
        *,
        max_candidates: int | None = None,
        now_ms: int | None = None,
    ) -> StoreLookupResult:
        """Materialize bounded candidates for :class:`TestProofCache`."""

        index_result = self.index.candidates(
            locator_cid,
            max_candidates=max_candidates or self.max_candidates,
            now_ms=now_ms,
        )
        if index_result.status is not CertificateStoreStatus.HIT:
            return StoreLookupResult(
                CertificateStoreStatus.MISS,
                index_result.reason_code,
                diagnostics=dict(index_result.diagnostics),
            )

        materialised: list[dict[str, Any]] = []
        for hint in index_result.candidates:
            receipt_get = self.cas.get_bytes(hint.receipt_cid)
            certificate_get = self.cas.get_bytes(hint.certificate_cid)
            if not receipt_get.hit or not certificate_get.hit:
                # Missing/corrupt CAS blob: drop the hint, never promote.
                if (
                    receipt_get.reason_code
                    in {
                        CertificateStoreReason.CORRUPT,
                        CertificateStoreReason.INTEGRITY_FAILED,
                        CertificateStoreReason.PARTIAL,
                        CertificateStoreReason.OVER_BUDGET,
                        CertificateStoreReason.SYMLINK_REJECTED,
                    }
                    or certificate_get.reason_code
                    in {
                        CertificateStoreReason.CORRUPT,
                        CertificateStoreReason.INTEGRITY_FAILED,
                        CertificateStoreReason.PARTIAL,
                        CertificateStoreReason.OVER_BUDGET,
                        CertificateStoreReason.SYMLINK_REJECTED,
                    }
                ):
                    self.index.quarantine_certificate(
                        locator_cid, hint.certificate_cid
                    )
                continue
            assert receipt_get.data is not None
            assert certificate_get.data is not None
            candidate: dict[str, Any] = {
                "receipt_bytes": receipt_get.data,
                "certificate_bytes": certificate_get.data,
                "receipt_cid": hint.receipt_cid,
                "certificate_cid": hint.certificate_cid,
                "created_at_ms": hint.created_at_ms,
                "metadata": dict(hint.metadata),
            }
            if hint.expires_at_ms is not None:
                candidate["expires_at_ms"] = hint.expires_at_ms
            materialised.append(candidate)

        if not materialised:
            return StoreLookupResult(
                CertificateStoreStatus.MISS,
                CertificateStoreReason.CANDIDATE_MISSING,
            )
        return StoreLookupResult(
            CertificateStoreStatus.HIT,
            CertificateStoreReason.OK,
            candidates=tuple(materialised),
        )

    def candidate_provider(
        self,
        locator: Any,
        execution_key: Any = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Adapter usable as ``TestProofCache(candidate_provider=...)``."""

        del execution_key
        locator_cid = self._locator_cid_of(locator)
        if not locator_cid:
            return []
        result = self.lookup(locator_cid)
        if not result:
            return []
        return [dict(item) for item in result.candidates]

    def _locator_cid_of(self, locator: Any) -> str:
        if isinstance(locator, str):
            return locator
        if isinstance(locator, Mapping):
            for key in ("locator_cid", "locator_id", "content_id", "cid"):
                value = locator.get(key)
                if isinstance(value, str) and value:
                    return value
            return ""
        for attr in ("locator_id", "locator_cid", "content_id", "cid"):
            value = getattr(locator, attr, None)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _coerce_receipt_bytes(
        receipt: TestPassReceipt | Mapping[str, Any] | bytes,
    ) -> tuple[bytes, str]:
        if isinstance(receipt, (bytes, bytearray)):
            data = bytes(receipt)
            cid = _cid_for_canonical_bytes(data)
            return data, cid
        if isinstance(receipt, TestPassReceipt):
            data = receipt.canonical_bytes()
            return data, receipt.receipt_id
        if isinstance(receipt, Mapping):
            typed = TestPassReceipt.from_dict(receipt)
            data = typed.canonical_bytes()
            return data, typed.receipt_id
        raise TypeError("receipt must be TestPassReceipt, mapping, or bytes")

    @staticmethod
    def _coerce_certificate_bytes(
        certificate: TestProofCertificate | Mapping[str, Any] | bytes,
    ) -> tuple[bytes, str]:
        if isinstance(certificate, (bytes, bytearray)):
            data = bytes(certificate)
            cid = _cid_for_canonical_bytes(data)
            return data, cid
        if isinstance(certificate, TestProofCertificate):
            data = certificate.canonical_bytes()
            return data, certificate.certificate_id
        if isinstance(certificate, Mapping):
            typed = TestProofCertificate.from_dict(certificate)
            data = typed.canonical_bytes()
            return data, typed.certificate_id
        raise TypeError(
            "certificate must be TestProofCertificate, mapping, or bytes"
        )


__all__ = [
    "CERTIFICATE_WRITE_FENCE_SCHEMA",
    "CertificateStoreError",
    "CertificateStoreFenceError",
    "CertificateStoreIntegrityError",
    "CertificateStoreReason",
    "CertificateStoreStatus",
    "CertificateWriteFence",
    "CasGetResult",
    "CasPutResult",
    "DEFAULT_FENCE_TTL_MS",
    "DEFAULT_INDEX_TTL_MS",
    "DEFAULT_MAX_BLOB_BYTES",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_INDEX_BYTES",
    "FenceLease",
    "ImmutableCertificateCAS",
    "IndexCandidate",
    "IndexLookupResult",
    "IndexPublishResult",
    "StoreLookupResult",
    "StorePutResult",
    "TEST_CERTIFICATE_INDEX_SCHEMA",
    "TEST_CERTIFICATE_STORE_INTERFACE",
    "TEST_CERTIFICATE_STORE_SCHEMA",
    "TestCertificateIndex",
    "TestCertificateStore",
]
