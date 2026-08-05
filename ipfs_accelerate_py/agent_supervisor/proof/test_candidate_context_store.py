"""Immutable candidate execution context CAS, fenced index, and atomic publication.

``TestCandidateContextStore@1`` retains exact pass-time canonical bytes so a
later warm run can reconstruct what a certificate attests.  Authority still
lives in the proof cache / certificate path: this module never treats index
metadata, cache presence, historical execution keys, or the candidate
descriptor itself as skip authority.

Write protocol (fail-closed):

1. Bound and rehash every retained component; confirm internal field CIDs and
   external claimed CIDs agree.
2. Write each component and the envelope to the immutable CAS via temporary
   file, fsync, and ``os.replace``.
3. Read published paths back and rehash; only then may the locator index be
   updated under a single-flight write fence.
4. Index publication is controller-fenced so parallel writers cannot mix
   components from different generations.

Reads reject symlinks, path escapes, oversized or partial files, poisoned
indexes, stale generations, remote failures, and transport absence.  Corrupt
artifacts are quarantined where safe and surface as typed misses.
``may_authorize_skip`` is always false for every store surface.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from .formal_verification_contracts import canonical_json_bytes, content_identity
from .test_certificate_store import (
    CertificateStoreFenceError,
    CertificateStoreIntegrityError,
    CertificateStoreReason as _CasReason,
    CertificateStoreStatus as _CasStatus,
    CertificateWriteFence,
    ImmutableCertificateCAS,
    _cid_for_canonical_bytes,
    _exclusive_lock,
    _now_ms,
    _object_without_duplicate_keys,
    _validate_cid_token,
)

try:
    from ...testing.proof_reuse.activation_contracts import (
        ArtifactRole,
        CandidateExecutionContext,
        admit_content_addressed_boundary,
    )
except ImportError:  # pragma: no cover - package layout fallback
    from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
        ArtifactRole,
        CandidateExecutionContext,
        admit_content_addressed_boundary,
    )

TEST_CANDIDATE_CONTEXT_STORE_INTERFACE: Final = "TestCandidateContextStore@1"
TEST_CANDIDATE_CONTEXT_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-candidate-context-store@1"
)
TEST_CANDIDATE_CONTEXT_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-candidate-context-index@1"
)
CANDIDATE_CONTEXT_ENVELOPE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/candidate-context-envelope@1"
)
CANDIDATE_CONTEXT_ENVELOPE_INTERFACE: Final = "CandidateContextEnvelope@1"
CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE: Final = (
    "CanonicalArtifactStoreTransport@1"
)
CANDIDATE_EXECUTION_CONTEXT_INTERFACE: Final = "CandidateExecutionContext@1"

DEFAULT_MAX_BLOB_BYTES: Final = 1_048_576
DEFAULT_MAX_COMPONENT_BYTES: Final = 1_048_576
DEFAULT_MAX_CANDIDATES: Final = 32
DEFAULT_MAX_INDEX_BYTES: Final = 256 * 1024
DEFAULT_INDEX_TTL_MS: Final = 7 * 24 * 60 * 60 * 1000
DEFAULT_FENCE_TTL_MS: Final = 30_000
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 30.0
ENVELOPE_VERSION: Final = 1

_INDEX_SUFFIX: Final = ".json"
_TMP_PREFIX: Final = ".tmp."

# Retained pass-time components that must rehash on every admission.
REQUIRED_COMPONENT_KEYS: Final[tuple[str, ...]] = (
    "execution_key",
    "static_trace",
    "runtime_trace",
    "repository_forest",
    "environment",
    "policy",
    "pass_receipt",
)

# Map store component keys → CandidateExecutionContext field names.
COMPONENT_FIELD_MAP: Final[Mapping[str, str]] = {
    "execution_key": "execution_key_cid",
    "static_trace": "static_trace_root_cid",
    "runtime_trace": "runtime_trace_root_cid",
    "repository_forest": "repository_forest_cid",
    "environment": "environment_cid",
    "policy": "policy_cid",
    "pass_receipt": "pass_receipt_cid",
    "test_ast": "test_ast_cid",
    "dependency_lock": "dependency_lock_cid",
    "installed_distributions": "installed_distributions_cid",
    "platform": "platform_cid",
    "capability_root": "capability_root_cid",
}


class CandidateContextStoreError(RuntimeError):
    """Base class for candidate-context store operational failures."""

    __test__ = False


class CandidateContextStoreIntegrityError(CandidateContextStoreError, ValueError):
    """Blob, envelope, or index failed integrity / path safety checks."""

    __test__ = False


class CandidateContextStoreStatus(StrEnum):
    """Closed store / lookup result states."""

    STORED = "stored"
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"
    ERROR = "error"

    __test__ = False


class CandidateContextStoreReason(StrEnum):
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
    INDEX_POISONED = "index_poisoned"
    STALE_GENERATION = "stale_generation"
    TRANSPORT_ABSENT = "transport_absent"
    REMOTE_FAILURE = "remote_failure"
    VERSION_MISMATCH = "version_mismatch"
    SIZE_EXCEEDED = "size_exceeded"
    COMPONENT_MISSING = "component_missing"
    INTERNAL_EXTERNAL_CID_MISMATCH = "internal_external_cid_mismatch"

    __test__ = False


def _map_cas_reason(reason: _CasReason | str) -> CandidateContextStoreReason:
    value = reason.value if isinstance(reason, _CasReason) else str(reason)
    try:
        return CandidateContextStoreReason(value)
    except ValueError:
        return CandidateContextStoreReason.UNAVAILABLE


@runtime_checkable
class CanonicalArtifactStoreTransport(Protocol):
    """Optional remote/local byte transport; never skip authority.

    Implementations must rehash retained bytes before returning a hit.
    Absence or failure of this transport is a typed miss, never an exception
    that suppresses test execution.
    """

    @property
    def interface(self) -> str:  # pragma: no cover - protocol
        ...

    def get_bytes(self, cid: str) -> Any:  # pragma: no cover - protocol
        ...

    def put_bytes(self, data: bytes, **kwargs: Any) -> Any:  # pragma: no cover
        ...


@dataclass(frozen=True)
class CandidateContextEnvelope:
    """Immutable publication unit binding a descriptor to retained components.

    The envelope is content-addressed.  Index entries that point at an envelope
    are retrieval hints only and never authorize ``SKIP``.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = CANDIDATE_CONTEXT_ENVELOPE_SCHEMA

    candidate_context_cid: str
    locator_cid: str
    execution_key_cid: str
    pass_receipt_cid: str
    component_cids: Mapping[str, str]
    version: int = ENVELOPE_VERSION
    generation: int = 1
    created_at_ms: int = 0
    expires_at_ms: int | None = None
    fencing_token: int | None = None
    byte_length: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_context_cid",
            _validate_cid_token(
                self.candidate_context_cid, field_name="candidate_context_cid"
            ),
        )
        object.__setattr__(
            self,
            "locator_cid",
            _validate_cid_token(self.locator_cid, field_name="locator_cid"),
        )
        object.__setattr__(
            self,
            "execution_key_cid",
            _validate_cid_token(
                self.execution_key_cid, field_name="execution_key_cid"
            ),
        )
        object.__setattr__(
            self,
            "pass_receipt_cid",
            _validate_cid_token(
                self.pass_receipt_cid, field_name="pass_receipt_cid"
            ),
        )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise CandidateContextStoreIntegrityError("version must be a positive int")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise CandidateContextStoreIntegrityError(
                "generation must be a positive int"
            )
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms < 0
        ):
            raise CandidateContextStoreIntegrityError("created_at_ms is invalid")
        if self.expires_at_ms is not None and (
            isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms < 0
        ):
            raise CandidateContextStoreIntegrityError("expires_at_ms is invalid")
        if self.fencing_token is not None and (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token < 0
        ):
            raise CandidateContextStoreIntegrityError("fencing_token is invalid")
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise CandidateContextStoreIntegrityError("byte_length is invalid")
        if not isinstance(self.component_cids, Mapping) or not self.component_cids:
            raise CandidateContextStoreIntegrityError(
                "component_cids must be a nonempty mapping"
            )
        normalized: dict[str, str] = {}
        for key, value in self.component_cids.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise CandidateContextStoreIntegrityError(
                    "component_cids keys are invalid"
                )
            normalized[key] = _validate_cid_token(
                str(value), field_name=f"component_cids.{key}"
            )
        for required in REQUIRED_COMPONENT_KEYS:
            if required not in normalized:
                raise CandidateContextStoreIntegrityError(
                    f"missing required component {required}"
                )
        object.__setattr__(self, "component_cids", dict(normalized))
        metadata = self.metadata or {}
        if not isinstance(metadata, Mapping):
            raise CandidateContextStoreIntegrityError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(metadata))

    @property
    def interface(self) -> str:
        return CANDIDATE_CONTEXT_ENVELOPE_INTERFACE

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def envelope_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": CANDIDATE_CONTEXT_ENVELOPE_INTERFACE,
            "version": self.version,
            "generation": self.generation,
            "candidate_context_cid": self.candidate_context_cid,
            "locator_cid": self.locator_cid,
            "execution_key_cid": self.execution_key_cid,
            "pass_receipt_cid": self.pass_receipt_cid,
            "component_cids": dict(self.component_cids),
            "created_at_ms": self.created_at_ms,
            "byte_length": self.byte_length,
            "may_authorize_skip": False,
        }
        if self.expires_at_ms is not None:
            payload["expires_at_ms"] = self.expires_at_ms
        if self.fencing_token is not None:
            payload["fencing_token"] = self.fencing_token
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateContextEnvelope":
        if not isinstance(payload, Mapping):
            raise CandidateContextStoreIntegrityError("envelope must be an object")
        if payload.get("schema") != cls.SCHEMA:
            raise CandidateContextStoreIntegrityError("envelope schema mismatch")
        if payload.get("interface") not in (
            None,
            "",
            CANDIDATE_CONTEXT_ENVELOPE_INTERFACE,
        ):
            raise CandidateContextStoreIntegrityError("envelope interface mismatch")
        if payload.get("may_authorize_skip") is True:
            raise CandidateContextStoreIntegrityError(
                "envelope must not claim skip authority"
            )
        return cls(
            candidate_context_cid=str(payload.get("candidate_context_cid", "")),
            locator_cid=str(payload.get("locator_cid", "")),
            execution_key_cid=str(payload.get("execution_key_cid", "")),
            pass_receipt_cid=str(payload.get("pass_receipt_cid", "")),
            component_cids=dict(payload.get("component_cids") or {}),
            version=int(payload.get("version", ENVELOPE_VERSION)),
            generation=int(payload.get("generation", 1)),
            created_at_ms=int(payload.get("created_at_ms", 0)),
            expires_at_ms=(
                None
                if payload.get("expires_at_ms") is None
                else int(payload["expires_at_ms"])
            ),
            fencing_token=(
                None
                if payload.get("fencing_token") is None
                else int(payload["fencing_token"])
            ),
            byte_length=int(payload.get("byte_length", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CandidateContextEnvelope":
        if type(data) is not bytes or not data:
            raise CandidateContextStoreIntegrityError("envelope bytes are required")
        try:
            payload = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.CORRUPT.value
            ) from exc
        envelope = cls.from_dict(payload)
        recomputed = envelope.canonical_bytes()
        if recomputed != data:
            raise CandidateContextStoreIntegrityError(
                "envelope bytes are not canonical DAG-JSON form"
            )
        return envelope


@dataclass(frozen=True)
class CandidateContextIndexHint:
    """One locator-index hint; never pass authority by itself."""

    __test__: ClassVar[bool] = False

    envelope_cid: str
    candidate_context_cid: str
    execution_key_cid: str
    generation: int
    created_at_ms: int
    expires_at_ms: int | None = None
    revoked: bool = False
    quarantined: bool = False
    fencing_token: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "envelope_cid": self.envelope_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "execution_key_cid": self.execution_key_cid,
            "generation": self.generation,
            "created_at_ms": self.created_at_ms,
            "revoked": bool(self.revoked),
            "quarantined": bool(self.quarantined),
            "may_authorize_skip": False,
        }
        if self.expires_at_ms is not None:
            payload["expires_at_ms"] = self.expires_at_ms
        if self.fencing_token is not None:
            payload["fencing_token"] = self.fencing_token
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateContextIndexHint":
        if not isinstance(payload, Mapping):
            raise CandidateContextStoreIntegrityError("index hint must be an object")
        if payload.get("may_authorize_skip") is True:
            raise CandidateContextStoreIntegrityError(
                "index hint must not claim skip authority"
            )
        generation = payload.get("generation", 1)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise CandidateContextStoreIntegrityError("generation is invalid")
        created = payload.get("created_at_ms")
        if isinstance(created, bool) or not isinstance(created, int) or created < 0:
            raise CandidateContextStoreIntegrityError("created_at_ms is invalid")
        expires = payload.get("expires_at_ms")
        if expires is not None and (
            isinstance(expires, bool) or not isinstance(expires, int) or expires < 0
        ):
            raise CandidateContextStoreIntegrityError("expires_at_ms is invalid")
        fencing = payload.get("fencing_token")
        if fencing is not None and (
            isinstance(fencing, bool) or not isinstance(fencing, int) or fencing < 0
        ):
            raise CandidateContextStoreIntegrityError("fencing_token is invalid")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise CandidateContextStoreIntegrityError("hint metadata must be a mapping")
        return cls(
            envelope_cid=_validate_cid_token(
                str(payload.get("envelope_cid", "")), field_name="envelope_cid"
            ),
            candidate_context_cid=_validate_cid_token(
                str(payload.get("candidate_context_cid", "")),
                field_name="candidate_context_cid",
            ),
            execution_key_cid=_validate_cid_token(
                str(payload.get("execution_key_cid", "")),
                field_name="execution_key_cid",
            ),
            generation=generation,
            created_at_ms=created,
            expires_at_ms=expires,
            revoked=bool(payload.get("revoked", False)),
            quarantined=bool(payload.get("quarantined", False)),
            fencing_token=fencing,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class CandidateContextPutResult:
    """Outcome of an atomic candidate-context publication."""

    __test__: ClassVar[bool] = False

    stored: bool
    reason_code: CandidateContextStoreReason
    envelope_cid: str = ""
    candidate_context_cid: str = ""
    locator_cid: str = ""
    generation: int = 0
    indexed: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.stored

    @property
    def may_authorize_skip(self) -> bool:
        return False


@dataclass(frozen=True)
class CandidateContextAdmission:
    """Result of rehashing every retained component for one candidate.

    Admission never elevates cache presence or historical keys into ``SKIP``.
    """

    __test__: ClassVar[bool] = False

    admitted: bool
    reason_code: CandidateContextStoreReason
    envelope_cid: str = ""
    candidate_context_cid: str = ""
    component_cids: Mapping[str, str] = field(default_factory=dict)
    byte_length: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.admitted

    @property
    def may_authorize_skip(self) -> bool:
        return False


@dataclass(frozen=True)
class CandidateContextLookupResult:
    """Lookup outcome: exact bytes plus a non-authoritative descriptor.

    ``descriptor`` is the decoded :class:`CandidateExecutionContext` and is
    never skip authority.  ``envelope_bytes`` / ``descriptor_bytes`` are the
    retained canonical forms after rehash admission.
    """

    __test__: ClassVar[bool] = False

    status: CandidateContextStoreStatus
    reason_code: CandidateContextStoreReason
    envelope_cid: str = ""
    candidate_context_cid: str = ""
    envelope_bytes: bytes | None = None
    descriptor_bytes: bytes | None = None
    descriptor: CandidateExecutionContext | None = None
    component_bytes: Mapping[str, bytes] = field(default_factory=dict)
    admission: CandidateContextAdmission | None = None
    generation: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return (
            self.status is CandidateContextStoreStatus.HIT
            and self.descriptor is not None
            and (
                self.envelope_bytes is not None or self.descriptor_bytes is not None
            )
        )

    @property
    def hit(self) -> bool:
        return bool(self)

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def bytes(self) -> bytes | None:
        """Primary retained bytes returned by lookup (envelope, else descriptor)."""

        if self.envelope_bytes is not None:
            return self.envelope_bytes
        return self.descriptor_bytes


@dataclass(frozen=True)
class IndexPublishResult:
    __test__: ClassVar[bool] = False

    published: bool
    locator_cid: str
    reason_code: CandidateContextStoreReason
    candidate_count: int = 0
    generation: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.published


@dataclass(frozen=True)
class IndexLookupResult:
    __test__: ClassVar[bool] = False

    status: CandidateContextStoreStatus
    locator_cid: str
    reason_code: CandidateContextStoreReason
    candidates: tuple[CandidateContextIndexHint, ...] = ()
    generation: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.status is CandidateContextStoreStatus.HIT and bool(self.candidates)


class CandidateContextIndex:
    """Mutable locator → bounded candidate-context envelope hints.

    Index entries are never authority.  TTL, revocation, quarantine, and
    generation fencing only affect whether a CID is *suggested* for re-admission.
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
        self.index_root = self.root / "candidate_index"
        self.lock_root = self.root / "locks" / "candidate_index"
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
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.SYMLINK_REJECTED.value
            )
        try:
            if path.exists():
                path.resolve(strict=True).relative_to(base)
            else:
                path.parent.resolve().relative_to(base)
        except ValueError as exc:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.PATH_ESCAPE.value
            ) from exc
        return path

    def _read_document(
        self, path: Path, *, locator_cid: str
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink():
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.SYMLINK_REJECTED.value
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CandidateContextStoreError("index unavailable") from exc
        if not raw:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.PARTIAL.value
            )
        if len(raw) > self.max_index_bytes:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.OVER_BUDGET.value
            )
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.INDEX_POISONED.value
            ) from exc
        if not isinstance(payload, Mapping):
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.INDEX_POISONED.value
            )
        if payload.get("schema") != TEST_CANDIDATE_CONTEXT_INDEX_SCHEMA:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.INDEX_POISONED.value
            )
        if payload.get("locator_cid") != locator_cid:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.INDEX_POISONED.value
            )
        if payload.get("may_authorize_skip") is True:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.INDEX_POISONED.value
            )
        generation = payload.get("generation", 1)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.INDEX_POISONED.value
            )
        return dict(payload)

    def _write_document(self, path: Path, document: Mapping[str, Any]) -> None:
        encoded = canonical_json_bytes(document)
        if len(encoded) > self.max_index_bytes:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.OVER_BUDGET.value
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
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.SYMLINK_REJECTED.value
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
        envelope_cid: str,
        candidate_context_cid: str,
        execution_key_cid: str,
        generation: int | None = None,
        created_at_ms: int | None = None,
        expires_at_ms: int | None = None,
        fencing_token: int | None = None,
        expected_fencing_token: int | None = None,
        expected_generation: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> IndexPublishResult:
        """Atomically append/replace one envelope hint for ``locator_cid``."""

        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            envelope = _validate_cid_token(envelope_cid, field_name="envelope_cid")
            context = _validate_cid_token(
                candidate_context_cid, field_name="candidate_context_cid"
            )
            execution = _validate_cid_token(
                execution_key_cid, field_name="execution_key_cid"
            )
            path = self._ensure_index_path(locator)
        except (CertificateStoreIntegrityError, CandidateContextStoreIntegrityError) as exc:
            reason = CandidateContextStoreReason.PATH_ESCAPE
            if str(exc) == CandidateContextStoreReason.SYMLINK_REJECTED.value:
                reason = CandidateContextStoreReason.SYMLINK_REJECTED
            return IndexPublishResult(False, str(locator_cid), reason)

        now = _now_ms(self._clock)
        created = now if created_at_ms is None else created_at_ms
        if isinstance(created, bool) or not isinstance(created, int) or created < 0:
            return IndexPublishResult(
                False, locator, CandidateContextStoreReason.MALFORMED
            )
        expires = expires_at_ms
        if expires is None and self.default_ttl_ms > 0:
            expires = created + self.default_ttl_ms
        if expires is not None and (
            isinstance(expires, bool) or not isinstance(expires, int) or expires < 0
        ):
            return IndexPublishResult(
                False, locator, CandidateContextStoreReason.MALFORMED
            )
        if expires is not None and expires <= created:
            return IndexPublishResult(
                False, locator, CandidateContextStoreReason.MALFORMED
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
                        False, locator, CandidateContextStoreReason.FENCE_MISMATCH
                    )
                try:
                    document = self._read_document(path, locator_cid=locator)
                except CandidateContextStoreIntegrityError as exc:
                    reason_value = str(exc)
                    try:
                        reason = CandidateContextStoreReason(reason_value)
                    except ValueError:
                        reason = CandidateContextStoreReason.INDEX_POISONED
                    if reason is CandidateContextStoreReason.SYMLINK_REJECTED:
                        return IndexPublishResult(False, locator, reason)
                    # Poisoned index: start a fresh document rather than mix.
                    document = None
                existing: list[CandidateContextIndexHint] = []
                current_generation = 0
                if document is not None:
                    current_generation = int(document.get("generation", 0) or 0)
                    raw_candidates = document.get("candidates", [])
                    if not isinstance(raw_candidates, list):
                        existing = []
                    else:
                        for item in raw_candidates:
                            try:
                                existing.append(
                                    CandidateContextIndexHint.from_dict(item)
                                )
                            except (
                                CandidateContextStoreIntegrityError,
                                CertificateStoreIntegrityError,
                            ):
                                continue
                if expected_generation is not None:
                    if current_generation != expected_generation:
                        return IndexPublishResult(
                            False,
                            locator,
                            CandidateContextStoreReason.STALE_GENERATION,
                            generation=current_generation,
                            diagnostics={
                                "expected_generation": expected_generation,
                                "current_generation": current_generation,
                            },
                        )
                next_generation = (
                    int(generation)
                    if generation is not None
                    else current_generation + 1
                )
                if next_generation <= current_generation:
                    return IndexPublishResult(
                        False,
                        locator,
                        CandidateContextStoreReason.STALE_GENERATION,
                        generation=current_generation,
                    )
                hint = CandidateContextIndexHint(
                    envelope_cid=envelope,
                    candidate_context_cid=context,
                    execution_key_cid=execution,
                    generation=next_generation,
                    created_at_ms=created,
                    expires_at_ms=expires,
                    fencing_token=fencing_token,
                    metadata=dict(metadata or {}),
                )
                merged = [
                    item
                    for item in existing
                    if item.envelope_cid != hint.envelope_cid
                    and item.candidate_context_cid != hint.candidate_context_cid
                ]
                merged.insert(0, hint)
                merged = merged[: self.max_candidates]
                payload = {
                    "schema": TEST_CANDIDATE_CONTEXT_INDEX_SCHEMA,
                    "interface": TEST_CANDIDATE_CONTEXT_STORE_INTERFACE,
                    "locator_cid": locator,
                    "generation": next_generation,
                    "updated_at_ms": now,
                    "may_authorize_skip": False,
                    "candidates": [item.to_dict() for item in merged],
                }
                if fencing_token is not None:
                    payload["fencing_token"] = int(fencing_token)
                self._write_document(path, payload)
                return IndexPublishResult(
                    True,
                    locator,
                    CandidateContextStoreReason.OK,
                    candidate_count=len(merged),
                    generation=next_generation,
                )
        except TimeoutError:
            return IndexPublishResult(
                False, locator, CandidateContextStoreReason.UNAVAILABLE
            )
        except CandidateContextStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                reason = CandidateContextStoreReason(reason_value)
            except ValueError:
                reason = CandidateContextStoreReason.CORRUPT
            return IndexPublishResult(False, locator, reason)
        except OSError:
            return IndexPublishResult(
                False, locator, CandidateContextStoreReason.UNAVAILABLE
            )

    def candidates(
        self,
        locator_cid: str,
        *,
        max_candidates: int | None = None,
        now_ms: int | None = None,
        include_revoked: bool = False,
        include_quarantined: bool = False,
        min_generation: int | None = None,
    ) -> IndexLookupResult:
        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            path = self._ensure_index_path(locator)
        except (CertificateStoreIntegrityError, CandidateContextStoreIntegrityError) as exc:
            reason = CandidateContextStoreReason.PATH_ESCAPE
            if str(exc) == CandidateContextStoreReason.SYMLINK_REJECTED.value:
                reason = CandidateContextStoreReason.SYMLINK_REJECTED
            return IndexLookupResult(
                CandidateContextStoreStatus.MISS, str(locator_cid), reason
            )

        limit = self.max_candidates if max_candidates is None else max_candidates
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return IndexLookupResult(
                CandidateContextStoreStatus.MISS,
                locator,
                CandidateContextStoreReason.MALFORMED,
            )
        current = _now_ms(self._clock) if now_ms is None else now_ms
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            return IndexLookupResult(
                CandidateContextStoreStatus.MISS,
                locator,
                CandidateContextStoreReason.MALFORMED,
            )

        lock_path = self._lock_path(locator)
        try:
            with _exclusive_lock(
                lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                try:
                    document = self._read_document(path, locator_cid=locator)
                except CandidateContextStoreIntegrityError as exc:
                    reason_value = str(exc)
                    try:
                        reason = CandidateContextStoreReason(reason_value)
                    except ValueError:
                        reason = CandidateContextStoreReason.INDEX_POISONED
                    return IndexLookupResult(
                        CandidateContextStoreStatus.MISS, locator, reason
                    )
                if document is None:
                    return IndexLookupResult(
                        CandidateContextStoreStatus.MISS,
                        locator,
                        CandidateContextStoreReason.CANDIDATE_MISSING,
                    )
                generation = int(document.get("generation", 0) or 0)
                raw_candidates = document.get("candidates", [])
                if not isinstance(raw_candidates, list):
                    return IndexLookupResult(
                        CandidateContextStoreStatus.MISS,
                        locator,
                        CandidateContextStoreReason.INDEX_POISONED,
                        generation=generation,
                    )
                admitted: list[CandidateContextIndexHint] = []
                for item in raw_candidates:
                    if len(admitted) >= limit:
                        break
                    try:
                        candidate = CandidateContextIndexHint.from_dict(item)
                    except (
                        CandidateContextStoreIntegrityError,
                        CertificateStoreIntegrityError,
                    ):
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
                    if (
                        min_generation is not None
                        and candidate.generation < min_generation
                    ):
                        continue
                    admitted.append(candidate)
                if not admitted:
                    return IndexLookupResult(
                        CandidateContextStoreStatus.MISS,
                        locator,
                        CandidateContextStoreReason.CANDIDATE_MISSING,
                        generation=generation,
                    )
                return IndexLookupResult(
                    CandidateContextStoreStatus.HIT,
                    locator,
                    CandidateContextStoreReason.OK,
                    candidates=tuple(admitted),
                    generation=generation,
                )
        except TimeoutError:
            return IndexLookupResult(
                CandidateContextStoreStatus.MISS,
                locator,
                CandidateContextStoreReason.UNAVAILABLE,
            )
        except OSError:
            return IndexLookupResult(
                CandidateContextStoreStatus.MISS,
                locator,
                CandidateContextStoreReason.UNAVAILABLE,
            )

    def revoke(
        self, locator_cid: str, candidate_context_cid: str
    ) -> IndexPublishResult:
        return self._flag(
            locator_cid,
            candidate_context_cid,
            field_name="revoked",
            reason=CandidateContextStoreReason.REVOKED,
        )

    def quarantine(
        self, locator_cid: str, candidate_context_cid: str
    ) -> IndexPublishResult:
        return self._flag(
            locator_cid,
            candidate_context_cid,
            field_name="quarantined",
            reason=CandidateContextStoreReason.QUARANTINED,
        )

    def _flag(
        self,
        locator_cid: str,
        candidate_context_cid: str,
        *,
        field_name: str,
        reason: CandidateContextStoreReason,
    ) -> IndexPublishResult:
        try:
            locator = _validate_cid_token(locator_cid, field_name="locator_cid")
            context = _validate_cid_token(
                candidate_context_cid, field_name="candidate_context_cid"
            )
            path = self._ensure_index_path(locator)
        except (CertificateStoreIntegrityError, CandidateContextStoreIntegrityError) as exc:
            mapped = CandidateContextStoreReason.PATH_ESCAPE
            if str(exc) == CandidateContextStoreReason.SYMLINK_REJECTED.value:
                mapped = CandidateContextStoreReason.SYMLINK_REJECTED
            return IndexPublishResult(False, str(locator_cid), mapped)

        lock_path = self._lock_path(locator)
        try:
            with _exclusive_lock(
                lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                document = self._read_document(path, locator_cid=locator)
                if document is None:
                    return IndexPublishResult(
                        False, locator, CandidateContextStoreReason.CANDIDATE_MISSING
                    )
                changed = False
                candidates: list[dict[str, Any]] = []
                for item in document.get("candidates", []):
                    if not isinstance(item, Mapping):
                        continue
                    entry = dict(item)
                    if entry.get("candidate_context_cid") == context:
                        entry[field_name] = True
                        changed = True
                    candidates.append(entry)
                if not changed:
                    return IndexPublishResult(
                        False, locator, CandidateContextStoreReason.CANDIDATE_MISSING
                    )
                document = dict(document)
                document["candidates"] = candidates
                document["updated_at_ms"] = _now_ms(self._clock)
                document["may_authorize_skip"] = False
                self._write_document(path, document)
                return IndexPublishResult(
                    True,
                    locator,
                    reason,
                    candidate_count=len(candidates),
                    generation=int(document.get("generation", 0) or 0),
                )
        except CandidateContextStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                mapped = CandidateContextStoreReason(reason_value)
            except ValueError:
                mapped = CandidateContextStoreReason.CORRUPT
            return IndexPublishResult(False, locator, mapped)
        except (TimeoutError, OSError):
            return IndexPublishResult(
                False, locator, CandidateContextStoreReason.UNAVAILABLE
            )


class TestCandidateContextStore:
    """Facade combining immutable CAS, locator index, and write fencing.

    ``publish`` is the single publication entrypoint that enforces temporary
    file write → atomic replace → readback rehash of every retained component →
    fenced index publication.  Parallel putters for the same locator cannot
    publish a mixed multi-component unit: each publication is one fenced
    generation.

    Lookup returns retained envelope bytes plus a non-authoritative
    :class:`CandidateExecutionContext` descriptor after admission rehash.
    Nothing exposed by this store may authorize ``SKIP``.
    """

    __test__ = False
    interface = TEST_CANDIDATE_CONTEXT_STORE_INTERFACE

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_component_bytes: int = DEFAULT_MAX_COMPONENT_BYTES,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_index_bytes: int = DEFAULT_MAX_INDEX_BYTES,
        index_ttl_ms: int = DEFAULT_INDEX_TTL_MS,
        fence_ttl_ms: int = DEFAULT_FENCE_TTL_MS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        clock: Callable[[], float] | Callable[[], int] | None = None,
        owner_id: str | None = None,
        remote_transport: CanonicalArtifactStoreTransport | None = None,
    ) -> None:
        if root is None:
            self.root = Path(
                tempfile.mkdtemp(prefix="test-candidate-context-store-")
            ).resolve()
            self._ephemeral = True
        else:
            self.root = Path(root).resolve()
            self._ephemeral = False
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_blob_bytes = int(max_blob_bytes)
        self.max_component_bytes = int(max_component_bytes)
        self.max_candidates = int(max_candidates)
        self._clock = clock
        self.owner_id = owner_id or f"candidate-store:{uuid.uuid4().hex}"
        self.remote_transport = remote_transport
        self.cas = ImmutableCertificateCAS(
            self.root / "cas",
            max_blob_bytes=max_blob_bytes,
            lock_timeout_seconds=lock_timeout_seconds,
            clock=clock,
        )
        self.index = CandidateContextIndex(
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
        self._publish_lock = threading.RLock()

    @property
    def schema(self) -> str:
        return TEST_CANDIDATE_CONTEXT_STORE_SCHEMA

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def put_canonical_bytes(
        self, data: bytes, *, claimed_cid: str | None = None
    ) -> Any:
        return self.cas.put_bytes(data, claimed_cid=claimed_cid)

    def get_bytes(self, cid: str) -> Any:
        local = self.cas.get_bytes(cid)
        if local.hit:
            return local
        if self.remote_transport is None:
            if local.reason_code is _CasReason.CANDIDATE_MISSING:
                # No remote backend configured: absence is a typed miss.
                return local
            return local
        return self._remote_get(cid, local_miss=local)

    def _remote_get(self, cid: str, *, local_miss: Any) -> Any:
        transport = self.remote_transport
        if transport is None:
            # Surface transport absence explicitly when callers force remote.
            class _Miss:
                status = _CasStatus.MISS
                reason_code = _CasReason.UNAVAILABLE
                data = None
                cid = cid
                hit = False
                diagnostics = {
                    "reason": CandidateContextStoreReason.TRANSPORT_ABSENT.value
                }

            return _Miss()
        try:
            result = transport.get_bytes(cid)
        except Exception as exc:  # noqa: BLE001 - typed miss boundary
            class _RemoteFail:
                status = _CasStatus.MISS
                reason_code = _CasReason.UNAVAILABLE
                data = None
                hit = False
                diagnostics = {
                    "reason": CandidateContextStoreReason.REMOTE_FAILURE.value,
                    "error": type(exc).__name__,
                }

            _RemoteFail.cid = cid  # type: ignore[attr-defined]
            return _RemoteFail()
        data = getattr(result, "data", None)
        if data is None and isinstance(result, Mapping):
            data = result.get("data")
        if not isinstance(data, (bytes, bytearray)):
            hit = bool(getattr(result, "hit", False) or getattr(result, "stored", False))
            if not hit:
                class _RemoteMiss:
                    status = _CasStatus.MISS
                    reason_code = _CasReason.CANDIDATE_MISSING
                    data = None
                    hit = False
                    diagnostics = {
                        "reason": CandidateContextStoreReason.REMOTE_FAILURE.value
                    }

                _RemoteMiss.cid = cid  # type: ignore[attr-defined]
                return _RemoteMiss()
            class _RemoteBad:
                status = _CasStatus.MISS
                reason_code = _CasReason.CORRUPT
                data = None
                hit = False
                diagnostics = {
                    "reason": CandidateContextStoreReason.REMOTE_FAILURE.value
                }

            _RemoteBad.cid = cid  # type: ignore[attr-defined]
            return _RemoteBad()
        data_bytes = bytes(data)
        put = self.cas.put_bytes(data_bytes, claimed_cid=cid)
        if not put.stored and put.reason_code not in {
            _CasReason.OK,
            _CasReason.ALREADY_EXISTS,
        }:
            class _RemoteReject:
                status = _CasStatus.MISS
                reason_code = put.reason_code
                data = None
                hit = False
                diagnostics = {
                    "reason": CandidateContextStoreReason.REMOTE_FAILURE.value,
                    "stage": "remote_rehash",
                }

            _RemoteReject.cid = cid  # type: ignore[attr-defined]
            return _RemoteReject()
        return self.cas.get_bytes(cid)

    def publish(
        self,
        descriptor: CandidateExecutionContext | Mapping[str, Any] | bytes,
        components: Mapping[str, bytes],
        *,
        locator_cid: str | None = None,
        created_at_ms: int | None = None,
        expires_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        publish_index: bool = True,
        owner_id: str | None = None,
        expected_generation: int | None = None,
    ) -> CandidateContextPutResult:
        """Persist descriptor + retained components as one fenced generation."""

        try:
            typed, descriptor_bytes = self._coerce_descriptor(descriptor)
        except (
            CandidateContextStoreIntegrityError,
            CertificateStoreIntegrityError,
            TypeError,
            ValueError,
        ):
            return CandidateContextPutResult(
                False, CandidateContextStoreReason.MALFORMED
            )

        if typed.may_authorize_skip:
            return CandidateContextPutResult(
                False,
                CandidateContextStoreReason.MALFORMED,
                diagnostics={"stage": "descriptor_claims_skip"},
            )

        try:
            component_plan = self._plan_components(typed, components)
        except CandidateContextStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                reason = CandidateContextStoreReason(reason_value)
            except ValueError:
                reason = CandidateContextStoreReason.MALFORMED
            return CandidateContextPutResult(
                False,
                reason,
                candidate_context_cid=typed.candidate_context_id,
                diagnostics={"stage": "component_plan", "detail": reason_value},
            )

        resolved_locator = locator_cid or typed.locator_cid
        try:
            locator = _validate_cid_token(
                resolved_locator, field_name="locator_cid"
            )
        except CertificateStoreIntegrityError:
            return CandidateContextPutResult(
                False,
                CandidateContextStoreReason.PATH_ESCAPE,
                candidate_context_cid=typed.candidate_context_id,
                locator_cid=str(resolved_locator),
            )

        if typed.locator_cid != locator:
            return CandidateContextPutResult(
                False,
                CandidateContextStoreReason.CID_MISMATCH,
                candidate_context_cid=typed.candidate_context_id,
                locator_cid=locator,
                diagnostics={"stage": "locator_binding"},
            )

        total_bytes = len(descriptor_bytes) + sum(
            len(item) for item in component_plan.values()
        )
        if (
            len(descriptor_bytes) > self.max_blob_bytes
            or any(len(item) > self.max_component_bytes for item in component_plan.values())
            or total_bytes > self.max_blob_bytes * max(4, len(component_plan) + 1)
        ):
            return CandidateContextPutResult(
                False,
                CandidateContextStoreReason.OVER_BUDGET
                if total_bytes > self.max_blob_bytes
                else CandidateContextStoreReason.SIZE_EXCEEDED,
                candidate_context_cid=typed.candidate_context_id,
                locator_cid=locator,
            )

        fence_key = f"candidate-locator:{locator}"
        owner = owner_id or self.owner_id
        try:
            lease = self.fence.acquire(fence_key, owner_id=owner)
        except CertificateStoreFenceError:
            return CandidateContextPutResult(
                False,
                CandidateContextStoreReason.FENCED,
                candidate_context_cid=typed.candidate_context_id,
                locator_cid=locator,
            )
        except (TimeoutError, OSError):
            return CandidateContextPutResult(
                False,
                CandidateContextStoreReason.UNAVAILABLE,
                candidate_context_cid=typed.candidate_context_id,
                locator_cid=locator,
            )

        with self._publish_lock:
            try:
                # 1. CAS-publish every retained component.
                for name, data in component_plan.items():
                    claimed = _cid_for_canonical_bytes(data)
                    put = self.cas.put_bytes(data, claimed_cid=claimed)
                    if not put.stored and put.reason_code not in {
                        _CasReason.OK,
                        _CasReason.ALREADY_EXISTS,
                    }:
                        return CandidateContextPutResult(
                            False,
                            _map_cas_reason(put.reason_code),
                            candidate_context_cid=typed.candidate_context_id,
                            locator_cid=locator,
                            diagnostics={"stage": f"component_cas:{name}"},
                        )
                    readback = self.cas.get_bytes(claimed)
                    if (
                        not readback.hit
                        or readback.data != data
                        or readback.cid != claimed
                    ):
                        return CandidateContextPutResult(
                            False,
                            CandidateContextStoreReason.INTEGRITY_FAILED,
                            candidate_context_cid=typed.candidate_context_id,
                            locator_cid=locator,
                            diagnostics={"stage": f"component_readback:{name}"},
                        )

                # 2. CAS-publish descriptor bytes under candidate_context_cid.
                desc_put = self.cas.put_bytes(
                    descriptor_bytes, claimed_cid=typed.candidate_context_id
                )
                if not desc_put.stored and desc_put.reason_code not in {
                    _CasReason.OK,
                    _CasReason.ALREADY_EXISTS,
                }:
                    return CandidateContextPutResult(
                        False,
                        _map_cas_reason(desc_put.reason_code),
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        diagnostics={"stage": "descriptor_cas"},
                    )
                desc_get = self.cas.get_bytes(typed.candidate_context_id)
                if (
                    not desc_get.hit
                    or desc_get.data != descriptor_bytes
                    or desc_get.cid != typed.candidate_context_id
                ):
                    return CandidateContextPutResult(
                        False,
                        CandidateContextStoreReason.INTEGRITY_FAILED,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        diagnostics={"stage": "descriptor_readback"},
                    )

                now = _now_ms(self._clock)
                created = now if created_at_ms is None else created_at_ms
                component_cids = {
                    name: _cid_for_canonical_bytes(data)
                    for name, data in component_plan.items()
                }
                envelope = CandidateContextEnvelope(
                    candidate_context_cid=typed.candidate_context_id,
                    locator_cid=locator,
                    execution_key_cid=typed.execution_key_cid,
                    pass_receipt_cid=typed.pass_receipt_cid,
                    component_cids=component_cids,
                    version=ENVELOPE_VERSION,
                    generation=1,  # concrete generation assigned at index publish
                    created_at_ms=created,
                    expires_at_ms=expires_at_ms,
                    fencing_token=lease.fencing_token,
                    byte_length=total_bytes,
                    metadata=dict(metadata or {}),
                )
                # Generation is finalized after index publish; envelope generation
                # field is rewritten once the index assigns the next generation.
                # Provisional envelope is stored after generation is known.

                try:
                    self.fence.require_valid(lease)
                except CertificateStoreFenceError:
                    return CandidateContextPutResult(
                        False,
                        CandidateContextStoreReason.FENCE_MISMATCH,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        diagnostics={"stage": "pre_index_fence"},
                    )

                if not publish_index:
                    # Store envelope with generation=1 as a pure CAS object.
                    envelope_bytes = envelope.canonical_bytes()
                    env_put = self.cas.put_bytes(envelope_bytes)
                    if not env_put.stored and env_put.reason_code not in {
                        _CasReason.OK,
                        _CasReason.ALREADY_EXISTS,
                    }:
                        return CandidateContextPutResult(
                            False,
                            _map_cas_reason(env_put.reason_code),
                            candidate_context_cid=typed.candidate_context_id,
                            locator_cid=locator,
                            diagnostics={"stage": "envelope_cas"},
                        )
                    return CandidateContextPutResult(
                        True,
                        CandidateContextStoreReason.OK,
                        envelope_cid=env_put.cid,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        generation=1,
                        indexed=False,
                    )

                # Reserve generation by publishing a provisional index entry after
                # the final envelope is written.  First compute next generation
                # from current index state under the fence.
                current = self.index.candidates(locator, max_candidates=1)
                next_generation = (current.generation or 0) + 1
                if (
                    expected_generation is not None
                    and current.generation != expected_generation
                ):
                    return CandidateContextPutResult(
                        False,
                        CandidateContextStoreReason.STALE_GENERATION,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        generation=current.generation,
                        diagnostics={
                            "expected_generation": expected_generation,
                            "current_generation": current.generation,
                        },
                    )

                final_envelope = CandidateContextEnvelope(
                    candidate_context_cid=typed.candidate_context_id,
                    locator_cid=locator,
                    execution_key_cid=typed.execution_key_cid,
                    pass_receipt_cid=typed.pass_receipt_cid,
                    component_cids=component_cids,
                    version=ENVELOPE_VERSION,
                    generation=next_generation,
                    created_at_ms=created,
                    expires_at_ms=expires_at_ms,
                    fencing_token=lease.fencing_token,
                    byte_length=total_bytes,
                    metadata=dict(metadata or {}),
                )
                envelope_bytes = final_envelope.canonical_bytes()
                if len(envelope_bytes) > self.max_blob_bytes:
                    return CandidateContextPutResult(
                        False,
                        CandidateContextStoreReason.OVER_BUDGET,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                    )
                env_put = self.cas.put_bytes(envelope_bytes)
                if not env_put.stored and env_put.reason_code not in {
                    _CasReason.OK,
                    _CasReason.ALREADY_EXISTS,
                }:
                    return CandidateContextPutResult(
                        False,
                        _map_cas_reason(env_put.reason_code),
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        diagnostics={"stage": "envelope_cas"},
                    )
                env_get = self.cas.get_bytes(env_put.cid)
                if (
                    not env_get.hit
                    or env_get.data != envelope_bytes
                    or env_get.cid != env_put.cid
                ):
                    return CandidateContextPutResult(
                        False,
                        CandidateContextStoreReason.INTEGRITY_FAILED,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        diagnostics={"stage": "envelope_readback"},
                    )

                # Optional remote mirror (best-effort; local remains authority for reads).
                if self.remote_transport is not None:
                    try:
                        self.remote_transport.put_bytes(
                            envelope_bytes, claimed_cid=env_put.cid
                        )
                    except Exception:
                        # Remote failure never rolls back local success; later
                        # remote-only reads surface REMOTE_FAILURE.
                        pass

                try:
                    self.fence.require_valid(lease)
                except CertificateStoreFenceError:
                    return CandidateContextPutResult(
                        False,
                        CandidateContextStoreReason.FENCE_MISMATCH,
                        envelope_cid=env_put.cid,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        diagnostics={"stage": "pre_index_fence_final"},
                    )

                publish = self.index.publish(
                    locator,
                    envelope_cid=env_put.cid,
                    candidate_context_cid=typed.candidate_context_id,
                    execution_key_cid=typed.execution_key_cid,
                    generation=next_generation,
                    created_at_ms=created,
                    expires_at_ms=expires_at_ms,
                    fencing_token=lease.fencing_token,
                    expected_fencing_token=lease.fencing_token,
                    expected_generation=expected_generation,
                    metadata=metadata,
                )
                if not publish.published:
                    return CandidateContextPutResult(
                        False,
                        publish.reason_code,
                        envelope_cid=env_put.cid,
                        candidate_context_cid=typed.candidate_context_id,
                        locator_cid=locator,
                        generation=publish.generation,
                        diagnostics={"stage": "index_publish"},
                    )
                return CandidateContextPutResult(
                    True,
                    CandidateContextStoreReason.OK,
                    envelope_cid=env_put.cid,
                    candidate_context_cid=typed.candidate_context_id,
                    locator_cid=locator,
                    generation=publish.generation,
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
        min_generation: int | None = None,
        candidate_context_cid: str | None = None,
    ) -> CandidateContextLookupResult:
        """Materialize one admitted candidate: bytes + non-authoritative descriptor."""

        index_result = self.index.candidates(
            locator_cid,
            max_candidates=max_candidates or self.max_candidates,
            now_ms=now_ms,
            min_generation=min_generation,
        )
        if index_result.status is not CandidateContextStoreStatus.HIT:
            return CandidateContextLookupResult(
                CandidateContextStoreStatus.MISS,
                index_result.reason_code,
                generation=index_result.generation,
                diagnostics=dict(index_result.diagnostics),
            )

        last_reason = CandidateContextStoreReason.CANDIDATE_MISSING
        for hint in index_result.candidates:
            if (
                candidate_context_cid is not None
                and hint.candidate_context_cid != candidate_context_cid
            ):
                continue
            admitted = self.admit(
                hint.envelope_cid,
                expected_generation=hint.generation,
                now_ms=now_ms,
                index_generation=index_result.generation,
            )
            if not admitted.admitted:
                last_reason = admitted.reason_code
                if admitted.reason_code in {
                    CandidateContextStoreReason.CORRUPT,
                    CandidateContextStoreReason.INTEGRITY_FAILED,
                    CandidateContextStoreReason.PARTIAL,
                    CandidateContextStoreReason.SYMLINK_REJECTED,
                    CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                    CandidateContextStoreReason.VERSION_MISMATCH,
                }:
                    self.index.quarantine(
                        locator_cid, hint.candidate_context_cid
                    )
                continue

            envelope_get = self.get_bytes(hint.envelope_cid)
            if not getattr(envelope_get, "hit", False) or envelope_get.data is None:
                last_reason = _map_cas_reason(
                    getattr(
                        envelope_get,
                        "reason_code",
                        CandidateContextStoreReason.CANDIDATE_MISSING,
                    )
                )
                continue
            try:
                envelope = CandidateContextEnvelope.from_bytes(envelope_get.data)
            except CandidateContextStoreIntegrityError:
                last_reason = CandidateContextStoreReason.CORRUPT
                self.index.quarantine(locator_cid, hint.candidate_context_cid)
                continue

            desc_get = self.get_bytes(envelope.candidate_context_cid)
            if not getattr(desc_get, "hit", False) or desc_get.data is None:
                last_reason = CandidateContextStoreReason.COMPONENT_MISSING
                continue
            try:
                descriptor = CandidateExecutionContext.from_dict(
                    json.loads(desc_get.data.decode("utf-8"))
                )
            except Exception:
                last_reason = CandidateContextStoreReason.CORRUPT
                self.index.quarantine(locator_cid, hint.candidate_context_cid)
                continue

            if descriptor.may_authorize_skip:
                last_reason = CandidateContextStoreReason.MALFORMED
                continue

            component_bytes: dict[str, bytes] = {}
            for name, cid in envelope.component_cids.items():
                blob = self.get_bytes(cid)
                if not getattr(blob, "hit", False) or blob.data is None:
                    last_reason = CandidateContextStoreReason.COMPONENT_MISSING
                    component_bytes = {}
                    break
                component_bytes[name] = blob.data
            if not component_bytes:
                continue

            return CandidateContextLookupResult(
                CandidateContextStoreStatus.HIT,
                CandidateContextStoreReason.OK,
                envelope_cid=hint.envelope_cid,
                candidate_context_cid=envelope.candidate_context_cid,
                envelope_bytes=envelope_get.data,
                descriptor_bytes=desc_get.data,
                descriptor=descriptor,
                component_bytes=component_bytes,
                admission=admitted,
                generation=envelope.generation,
                diagnostics={
                    "locator_cid": locator_cid,
                    "index_generation": index_result.generation,
                },
            )

        return CandidateContextLookupResult(
            CandidateContextStoreStatus.MISS,
            last_reason,
            generation=index_result.generation,
        )

    def admit(
        self,
        envelope_cid: str,
        *,
        expected_generation: int | None = None,
        index_generation: int | None = None,
        now_ms: int | None = None,
    ) -> CandidateContextAdmission:
        """Rehash every retained component and confirm CID agreement."""

        try:
            token = _validate_cid_token(envelope_cid, field_name="envelope_cid")
        except CertificateStoreIntegrityError:
            return CandidateContextAdmission(
                False, CandidateContextStoreReason.PATH_ESCAPE, envelope_cid=str(envelope_cid)
            )

        envelope_get = self.get_bytes(token)
        if not getattr(envelope_get, "hit", False) or envelope_get.data is None:
            remote_diag = getattr(envelope_get, "diagnostics", {}) or {}
            if (
                isinstance(remote_diag, Mapping)
                and remote_diag.get("reason")
                == CandidateContextStoreReason.TRANSPORT_ABSENT.value
            ):
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.TRANSPORT_ABSENT,
                    envelope_cid=token,
                )
            if (
                isinstance(remote_diag, Mapping)
                and remote_diag.get("reason")
                == CandidateContextStoreReason.REMOTE_FAILURE.value
            ):
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.REMOTE_FAILURE,
                    envelope_cid=token,
                )
            return CandidateContextAdmission(
                False,
                _map_cas_reason(
                    getattr(
                        envelope_get,
                        "reason_code",
                        CandidateContextStoreReason.CANDIDATE_MISSING,
                    )
                ),
                envelope_cid=token,
            )

        data = envelope_get.data
        if len(data) > self.max_blob_bytes:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.SIZE_EXCEEDED,
                envelope_cid=token,
                byte_length=len(data),
            )

        try:
            external_cid = _cid_for_canonical_bytes(data)
        except CertificateStoreIntegrityError:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.CORRUPT,
                envelope_cid=token,
                byte_length=len(data),
            )
        if external_cid != token:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                envelope_cid=token,
                diagnostics={"actual_cid": external_cid},
                byte_length=len(data),
            )

        try:
            envelope = CandidateContextEnvelope.from_bytes(data)
        except CandidateContextStoreIntegrityError as exc:
            reason_value = str(exc)
            try:
                reason = CandidateContextStoreReason(reason_value)
            except ValueError:
                reason = CandidateContextStoreReason.CORRUPT
            return CandidateContextAdmission(
                False, reason, envelope_cid=token, byte_length=len(data)
            )

        if envelope.version != ENVELOPE_VERSION:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.VERSION_MISMATCH,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={"version": envelope.version},
            )

        current = _now_ms(self._clock) if now_ms is None else now_ms
        if (
            envelope.expires_at_ms is not None
            and envelope.expires_at_ms <= current
        ):
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.EXPIRED,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
            )

        if expected_generation is not None and envelope.generation != expected_generation:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.STALE_GENERATION,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={
                    "envelope_generation": envelope.generation,
                    "expected_generation": expected_generation,
                },
            )
        if (
            index_generation is not None
            and envelope.generation > index_generation
        ):
            # Envelope from a future generation relative to the index document
            # is inconsistent (poison / partial write).
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.STALE_GENERATION,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={
                    "envelope_generation": envelope.generation,
                    "index_generation": index_generation,
                },
            )

        # Descriptor internal identity.
        desc_get = self.get_bytes(envelope.candidate_context_cid)
        if not getattr(desc_get, "hit", False) or desc_get.data is None:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.COMPONENT_MISSING,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={"stage": "descriptor_missing"},
            )
        boundary = admit_content_addressed_boundary(
            role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
            claimed_cid=envelope.candidate_context_cid,
            canonical_bytes=desc_get.data,
        )
        if not boundary.admitted:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.INTEGRITY_FAILED,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={"stage": "descriptor_rehash"},
            )
        try:
            descriptor = CandidateExecutionContext.from_dict(
                json.loads(desc_get.data.decode("utf-8"))
            )
        except Exception:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.CORRUPT,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
            )
        if descriptor.candidate_context_id != envelope.candidate_context_cid:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={"stage": "descriptor_identity"},
            )
        if descriptor.may_authorize_skip:
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.MALFORMED,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={"stage": "descriptor_claims_skip"},
            )

        # Cross-check envelope binding fields against descriptor.
        if (
            descriptor.locator_cid != envelope.locator_cid
            or descriptor.execution_key_cid != envelope.execution_key_cid
            or descriptor.pass_receipt_cid != envelope.pass_receipt_cid
        ):
            return CandidateContextAdmission(
                False,
                CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                envelope_cid=token,
                candidate_context_cid=envelope.candidate_context_cid,
                diagnostics={"stage": "envelope_descriptor_binding"},
            )

        verified_components: dict[str, str] = {}
        for name, claimed_cid in envelope.component_cids.items():
            blob = self.get_bytes(claimed_cid)
            if not getattr(blob, "hit", False) or blob.data is None:
                remote_diag = getattr(blob, "diagnostics", {}) or {}
                if (
                    isinstance(remote_diag, Mapping)
                    and remote_diag.get("reason")
                    == CandidateContextStoreReason.TRANSPORT_ABSENT.value
                ):
                    return CandidateContextAdmission(
                        False,
                        CandidateContextStoreReason.TRANSPORT_ABSENT,
                        envelope_cid=token,
                        candidate_context_cid=envelope.candidate_context_cid,
                        diagnostics={"component": name},
                    )
                if (
                    isinstance(remote_diag, Mapping)
                    and remote_diag.get("reason")
                    == CandidateContextStoreReason.REMOTE_FAILURE.value
                ):
                    return CandidateContextAdmission(
                        False,
                        CandidateContextStoreReason.REMOTE_FAILURE,
                        envelope_cid=token,
                        candidate_context_cid=envelope.candidate_context_cid,
                        diagnostics={"component": name},
                    )
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.COMPONENT_MISSING
                    if _map_cas_reason(
                        getattr(
                            blob,
                            "reason_code",
                            CandidateContextStoreReason.CANDIDATE_MISSING,
                        )
                    )
                    is CandidateContextStoreReason.CANDIDATE_MISSING
                    else _map_cas_reason(
                        getattr(
                            blob,
                            "reason_code",
                            CandidateContextStoreReason.CANDIDATE_MISSING,
                        )
                    ),
                    envelope_cid=token,
                    candidate_context_cid=envelope.candidate_context_cid,
                    diagnostics={"component": name},
                )
            if len(blob.data) > self.max_component_bytes:
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.SIZE_EXCEEDED,
                    envelope_cid=token,
                    candidate_context_cid=envelope.candidate_context_cid,
                    diagnostics={"component": name, "byte_length": len(blob.data)},
                )
            try:
                actual = _cid_for_canonical_bytes(blob.data)
            except CertificateStoreIntegrityError:
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.CORRUPT,
                    envelope_cid=token,
                    candidate_context_cid=envelope.candidate_context_cid,
                    diagnostics={"component": name},
                )
            if actual != claimed_cid:
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                    envelope_cid=token,
                    candidate_context_cid=envelope.candidate_context_cid,
                    diagnostics={
                        "component": name,
                        "claimed_cid": claimed_cid,
                        "actual_cid": actual,
                    },
                )
            # Internal field agreement for known components.
            field_name = COMPONENT_FIELD_MAP.get(name)
            if field_name is not None:
                expected_field = getattr(descriptor, field_name, "")
                if expected_field and expected_field != claimed_cid:
                    return CandidateContextAdmission(
                        False,
                        CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                        envelope_cid=token,
                        candidate_context_cid=envelope.candidate_context_cid,
                        diagnostics={
                            "component": name,
                            "field": field_name,
                            "field_cid": expected_field,
                            "component_cid": claimed_cid,
                        },
                    )
            verified_components[name] = claimed_cid

        for required in REQUIRED_COMPONENT_KEYS:
            if required not in verified_components:
                return CandidateContextAdmission(
                    False,
                    CandidateContextStoreReason.COMPONENT_MISSING,
                    envelope_cid=token,
                    candidate_context_cid=envelope.candidate_context_cid,
                    diagnostics={"missing": required},
                )

        return CandidateContextAdmission(
            True,
            CandidateContextStoreReason.OK,
            envelope_cid=token,
            candidate_context_cid=envelope.candidate_context_cid,
            component_cids=verified_components,
            byte_length=len(data),
            diagnostics={"generation": envelope.generation},
        )

    def lookup_by_context_cid(
        self, candidate_context_cid: str
    ) -> CandidateContextLookupResult:
        """Direct CAS lookup of a descriptor by its content CID (no index)."""

        try:
            token = _validate_cid_token(
                candidate_context_cid, field_name="candidate_context_cid"
            )
        except CertificateStoreIntegrityError:
            return CandidateContextLookupResult(
                CandidateContextStoreStatus.MISS,
                CandidateContextStoreReason.PATH_ESCAPE,
            )
        desc_get = self.get_bytes(token)
        if not getattr(desc_get, "hit", False) or desc_get.data is None:
            return CandidateContextLookupResult(
                CandidateContextStoreStatus.MISS,
                _map_cas_reason(
                    getattr(
                        desc_get,
                        "reason_code",
                        CandidateContextStoreReason.CANDIDATE_MISSING,
                    )
                ),
            )
        try:
            descriptor = CandidateExecutionContext.from_dict(
                json.loads(desc_get.data.decode("utf-8"))
            )
        except Exception:
            return CandidateContextLookupResult(
                CandidateContextStoreStatus.MISS,
                CandidateContextStoreReason.CORRUPT,
                candidate_context_cid=token,
            )
        if descriptor.candidate_context_id != token:
            return CandidateContextLookupResult(
                CandidateContextStoreStatus.MISS,
                CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH,
                candidate_context_cid=token,
            )
        # Direct descriptor presence is never SKIP authority.
        return CandidateContextLookupResult(
            CandidateContextStoreStatus.HIT,
            CandidateContextStoreReason.OK,
            candidate_context_cid=token,
            descriptor_bytes=desc_get.data,
            descriptor=descriptor,
            diagnostics={"path": "direct_descriptor", "may_authorize_skip": False},
        )

    @staticmethod
    def _coerce_descriptor(
        descriptor: CandidateExecutionContext | Mapping[str, Any] | bytes,
    ) -> tuple[CandidateExecutionContext, bytes]:
        if isinstance(descriptor, (bytes, bytearray)):
            data = bytes(descriptor)
            payload = json.loads(data.decode("utf-8"))
            typed = CandidateExecutionContext.from_dict(payload)
            recomputed = typed.canonical_bytes()
            if recomputed != data:
                raise CandidateContextStoreIntegrityError(
                    "descriptor bytes are not canonical"
                )
            if typed.candidate_context_id != _cid_for_canonical_bytes(data):
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.CID_MISMATCH.value
                )
            return typed, data
        if isinstance(descriptor, CandidateExecutionContext):
            data = descriptor.canonical_bytes()
            return descriptor, data
        if isinstance(descriptor, Mapping):
            typed = CandidateExecutionContext.from_dict(descriptor)
            return typed, typed.canonical_bytes()
        raise TypeError(
            "descriptor must be CandidateExecutionContext, mapping, or bytes"
        )

    def _plan_components(
        self,
        descriptor: CandidateExecutionContext,
        components: Mapping[str, bytes],
    ) -> dict[str, bytes]:
        if not isinstance(components, Mapping) or not components:
            raise CandidateContextStoreIntegrityError(
                CandidateContextStoreReason.COMPONENT_MISSING.value
            )
        plan: dict[str, bytes] = {}
        for name, data in components.items():
            if not isinstance(name, str) or not name:
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.MALFORMED.value
                )
            if type(data) is not bytes:
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.MALFORMED.value
                )
            if not data:
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.PARTIAL.value
                )
            if len(data) > self.max_component_bytes:
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.SIZE_EXCEEDED.value
                )
            try:
                actual = _cid_for_canonical_bytes(data)
            except CertificateStoreIntegrityError as exc:
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.CORRUPT.value
                ) from exc
            field_name = COMPONENT_FIELD_MAP.get(name)
            if field_name is not None:
                expected = getattr(descriptor, field_name, "")
                if expected and expected != actual:
                    raise CandidateContextStoreIntegrityError(
                        CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH.value
                    )
            # Also accept components listed in descriptor.component_cids.
            if name in descriptor.component_cids:
                expected = descriptor.component_cids[name]
                if expected and expected != actual:
                    raise CandidateContextStoreIntegrityError(
                        CandidateContextStoreReason.INTERNAL_EXTERNAL_CID_MISMATCH.value
                    )
            plan[name] = data
        for required in REQUIRED_COMPONENT_KEYS:
            if required not in plan:
                raise CandidateContextStoreIntegrityError(
                    CandidateContextStoreReason.COMPONENT_MISSING.value
                )
        # Optional test_ast agreement when provided.
        if "test_ast" not in plan and descriptor.test_ast_cid:
            # Not required as a retained blob when only the identity is pinned,
            # but callers may still supply it.  Required set above is enough.
            pass
        return plan


__all__ = [
    "CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE",
    "CANDIDATE_CONTEXT_ENVELOPE_INTERFACE",
    "CANDIDATE_CONTEXT_ENVELOPE_SCHEMA",
    "CANDIDATE_EXECUTION_CONTEXT_INTERFACE",
    "COMPONENT_FIELD_MAP",
    "CandidateContextAdmission",
    "CandidateContextEnvelope",
    "CandidateContextIndex",
    "CandidateContextIndexHint",
    "CandidateContextLookupResult",
    "CandidateContextPutResult",
    "CandidateContextStoreError",
    "CandidateContextStoreIntegrityError",
    "CandidateContextStoreReason",
    "CandidateContextStoreStatus",
    "CanonicalArtifactStoreTransport",
    "DEFAULT_FENCE_TTL_MS",
    "DEFAULT_INDEX_TTL_MS",
    "DEFAULT_MAX_BLOB_BYTES",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_COMPONENT_BYTES",
    "DEFAULT_MAX_INDEX_BYTES",
    "ENVELOPE_VERSION",
    "IndexLookupResult",
    "IndexPublishResult",
    "REQUIRED_COMPONENT_KEYS",
    "TEST_CANDIDATE_CONTEXT_INDEX_SCHEMA",
    "TEST_CANDIDATE_CONTEXT_STORE_INTERFACE",
    "TEST_CANDIDATE_CONTEXT_STORE_SCHEMA",
    "TestCandidateContextStore",
]
