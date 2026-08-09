"""Content-addressed DCR proof-evidence cache with exact invalidation roots.

DCR-034 seals two closed interfaces:

* ``ProofCache@1`` — content-addressed lookup of independently reconstructed
  proof evidence (kernel receipts and minimized counterexamples).  The cache
  is not a trust root: every hit re-parses the stored payload and re-runs
  reconstruction so a warm path equals a cold run.
* ``ProofInvalidation@1`` — reverse-dependency invalidation over declared
  :class:`ProofDependencyRoot` values.  Any change to input, policy, solver,
  schema, source, runtime, capability, epoch, kernel, graph, tree, or
  toolchain roots tombstones descendants so stale and cross-epoch evidence
  cannot be selected.

Only reconstructed evidence may be admitted.  Provider ``verified`` flags,
partial receipts, and non-current epochs are never promoted to cache hits.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .kernel_reconstruction import (
    Counterexample,
    DEFAULT_KERNEL_VERSION,
    KernelReconstructionError,
    ProofClaim,
    ProofKernelReceipt,
    ReconstructionStatus,
    reconstruct_proof,
)


PROOF_CACHE_INTERFACE: Final = "ProofCache@1"
PROOF_INVALIDATION_INTERFACE: Final = "ProofInvalidation@1"
DCR_PROOF_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-cache-key@1"
)
DCR_PROOF_CACHE_ENTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-cache-entry@1"
)
DCR_PROOF_DEPENDENCY_ROOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-dependency-root@1"
)
DCR_PROOF_INVALIDATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-invalidation@1"
)
DCR_PROOF_CACHE_INDEX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-proof-cache-index@1"
)
DCR_PROOF_CACHE_VERSION: Final = 1
DEFAULT_MAX_ENTRIES: Final = 2048
DEFAULT_MAX_BYTES: Final = 64 * 1024 * 1024
DEFAULT_TTL_SECONDS: Final = 24 * 60 * 60

# Closed dependency dimensions that, when changed, invalidate descendants.
_REQUIRED_ROOT_KINDS: Final[tuple[str, ...]] = (
    "input",
    "policy",
    "solver",
    "schema",
    "source",
    "runtime",
    "capability",
    "epoch",
    "kernel",
    "graph",
    "tree",
    "toolchain",
)


class DcrProofCacheError(ContractValidationError):
    """Raised when a DCR proof-cache contract is malformed or unsafe."""


class DcrCacheDisposition(str, Enum):
    """Closed outcomes for cache consultation."""

    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"
    STORED = "stored"
    NOT_STORED = "not_stored"


class DcrCacheReason(str, Enum):
    """Stable audit reason codes for the DCR proof cache."""

    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    STORED = "cache_entry_stored"
    NOT_RECONSTRUCTED = "evidence_not_reconstructed"
    STALE = "stale_cache_entry"
    CROSS_EPOCH = "cross_epoch_evidence"
    BINDING_MISMATCH = "cache_binding_mismatch"
    ROOT_MISMATCH = "dependency_root_mismatch"
    TOMBSTONED = "tombstoned_by_dependency_root"
    EXPIRED = "expired_cache_entry"
    POISONED = "poisoned_cache_entry"
    MALFORMED = "malformed_cache_entry"
    RECONSTRUCTION_MISMATCH = "reconstruction_mismatch"
    RECONSTRUCTION_FAILED = "reconstruction_failed"
    INVALID_EVIDENCE = "invalid_evidence"
    COUNTEREXAMPLE_ONLY = "counterexample_evidence"
    PRIVATE_MATERIAL = "private_material"
    CAPACITY = "cache_capacity_exceeded"
    IDENTITY_INVALID = "identity_invalid"


class DcrEvidenceKind(str, Enum):
    """Kinds of reconstructed evidence the cache may retain."""

    PROOF_KERNEL_RECEIPT = "proof_kernel_receipt"
    COUNTEREXAMPLE = "counterexample"


class ProofDependencyRootKind(str, Enum):
    """Closed dependency dimensions for exact invalidation."""

    INPUT = "input"
    POLICY = "policy"
    SOLVER = "solver"
    SCHEMA = "schema"
    SOURCE = "source"
    RUNTIME = "runtime"
    CAPABILITY = "capability"
    EPOCH = "epoch"
    KERNEL = "kernel"
    GRAPH = "graph"
    TREE = "tree"
    TOOLCHAIN = "toolchain"


_PRIVATE_FIELDS: Final = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_of(value: Any) -> str:
    return _sha256_hex(_canonical_bytes(value))


def _text(value: Any, name: str, *, required: bool = False, maximum: int = 4096) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise DcrProofCacheError(f"{name} must be a string")
    if required and not text:
        raise DcrProofCacheError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise DcrProofCacheError(f"{name} exceeds {maximum} bytes")
    if "\x00" in text:
        raise DcrProofCacheError(f"{name} contains a NUL byte")
    return text


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DcrProofCacheError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DcrProofCacheError(f"{name} must be a non-negative integer")
    return value


def _contains_private_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name).strip().casefold().replace("-", "_")
            if any(
                name == marker
                or name.endswith("_" + marker)
                or marker in name
                for marker in _PRIVATE_FIELDS
            ):
                # Counterexample witness is an explicit public, minimized field
                # and is exempt from the private-material gate by name alone.
                if name == "witness" and isinstance(item, Mapping):
                    if _contains_private_material(item):
                        return True
                    continue
                if name == "witness":
                    continue
                return True
            if _contains_private_material(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_private_material(item) for item in value)
    return False


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(payload)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class ProofDependencyRoot:
    """One exact invalidation dependency root.

    Interface surface: part of ``ProofInvalidation@1``.
    """

    kind: ProofDependencyRootKind | str
    digest: str
    label: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.kind, ProofDependencyRootKind):
            kind = self.kind
        else:
            try:
                kind = ProofDependencyRootKind(str(self.kind).strip().casefold())
            except ValueError as exc:
                raise DcrProofCacheError(
                    f"unsupported dependency root kind: {self.kind!r}"
                ) from exc
        object.__setattr__(self, "kind", kind)
        digest = _text(self.digest, "digest", required=True, maximum=512)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "label", _text(self.label, "label", maximum=512))

    @property
    def root_id(self) -> str:
        return f"{self.kind.value}:{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_PROOF_DEPENDENCY_ROOT_SCHEMA,
            "kind": self.kind.value if isinstance(self.kind, ProofDependencyRootKind) else str(self.kind),
            "digest": self.digest,
            "label": self.label,
            "root_id": self.root_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofDependencyRoot":
        if not isinstance(value, Mapping):
            raise DcrProofCacheError("dependency root must be an object")
        schema = value.get("schema")
        if schema not in (None, DCR_PROOF_DEPENDENCY_ROOT_SCHEMA):
            raise DcrProofCacheError("unsupported dependency-root schema")
        return cls(
            kind=value.get("kind", ""),
            digest=value.get("digest", ""),
            label=value.get("label", ""),
        )

    @classmethod
    def from_value(
        cls,
        kind: ProofDependencyRootKind | str,
        value: Any,
        *,
        label: str = "",
    ) -> "ProofDependencyRoot":
        """Build a root from a raw identity value (CID, digest, or mapping)."""

        if isinstance(value, ProofDependencyRoot):
            if value.kind != ProofDependencyRootKind(str(kind).strip().casefold()):
                raise DcrProofCacheError("dependency root kind disagrees with factory kind")
            return value
        if isinstance(value, str) and value.strip():
            digest = value.strip()
        else:
            digest = _digest_of(value)
        return cls(kind=kind, digest=digest, label=label)


def build_dependency_roots(
    *,
    input: Any,
    policy: Any,
    solver: Any,
    schema: Any,
    source: Any,
    runtime: Any,
    capability: Any,
    epoch: Any,
    kernel: Any,
    graph: Any,
    tree: Any,
    toolchain: Any,
    labels: Mapping[str, str] | None = None,
) -> tuple[ProofDependencyRoot, ...]:
    """Construct the complete required dependency-root set for one cache key."""

    labels = labels or {}
    values = {
        "input": input,
        "policy": policy,
        "solver": solver,
        "schema": schema,
        "source": source,
        "runtime": runtime,
        "capability": capability,
        "epoch": epoch,
        "kernel": kernel,
        "graph": graph,
        "tree": tree,
        "toolchain": toolchain,
    }
    roots: list[ProofDependencyRoot] = []
    for kind_name in _REQUIRED_ROOT_KINDS:
        roots.append(
            ProofDependencyRoot.from_value(
                kind_name,
                values[kind_name],
                label=labels.get(kind_name, ""),
            )
        )
    return tuple(roots)


@dataclass(frozen=True)
class DcrProofCacheKey(CanonicalContract):
    """Content-addressed identity of every input that can change a DCR proof."""

    SCHEMA: ClassVar[str] = DCR_PROOF_CACHE_KEY_SCHEMA
    INTERFACE: ClassVar[str] = PROOF_CACHE_INTERFACE

    obligation_id: str
    dependency_roots: tuple[ProofDependencyRoot, ...]
    evidence_kind: DcrEvidenceKind | str = DcrEvidenceKind.PROOF_KERNEL_RECEIPT
    kernel_version: str = DEFAULT_KERNEL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", required=True),
        )
        if isinstance(self.evidence_kind, DcrEvidenceKind):
            kind = self.evidence_kind
        else:
            try:
                kind = DcrEvidenceKind(str(self.evidence_kind).strip().casefold())
            except ValueError as exc:
                raise DcrProofCacheError(
                    f"unsupported evidence kind: {self.evidence_kind!r}"
                ) from exc
        object.__setattr__(self, "evidence_kind", kind)
        object.__setattr__(
            self,
            "kernel_version",
            _text(self.kernel_version, "kernel_version", required=True),
        )
        roots = self._normalize_roots(self.dependency_roots)
        object.__setattr__(self, "dependency_roots", roots)

    @staticmethod
    def _normalize_roots(
        roots: Sequence[ProofDependencyRoot | Mapping[str, Any]],
    ) -> tuple[ProofDependencyRoot, ...]:
        if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes, bytearray)):
            raise DcrProofCacheError("dependency_roots must be a sequence")
        normalized: list[ProofDependencyRoot] = []
        seen_kinds: set[str] = set()
        for item in roots:
            root = (
                item
                if isinstance(item, ProofDependencyRoot)
                else ProofDependencyRoot.from_dict(item)
            )
            kind_name = (
                root.kind.value
                if isinstance(root.kind, ProofDependencyRootKind)
                else str(root.kind)
            )
            if kind_name in seen_kinds:
                raise DcrProofCacheError(
                    f"duplicate dependency root kind: {kind_name}"
                )
            seen_kinds.add(kind_name)
            normalized.append(root)
        missing = [name for name in _REQUIRED_ROOT_KINDS if name not in seen_kinds]
        if missing:
            raise DcrProofCacheError(
                "dependency_roots missing required kinds: " + ", ".join(missing)
            )
        # Stable order by required kind sequence for deterministic key identity.
        by_kind = {
            (
                root.kind.value
                if isinstance(root.kind, ProofDependencyRootKind)
                else str(root.kind)
            ): root
            for root in normalized
        }
        return tuple(by_kind[name] for name in _REQUIRED_ROOT_KINDS)

    @property
    def epoch(self) -> str:
        return self.root_digest(ProofDependencyRootKind.EPOCH)

    def root_digest(self, kind: ProofDependencyRootKind | str) -> str:
        kind_name = (
            kind.value
            if isinstance(kind, ProofDependencyRootKind)
            else str(kind).strip().casefold()
        )
        for root in self.dependency_roots:
            root_kind = (
                root.kind.value
                if isinstance(root.kind, ProofDependencyRootKind)
                else str(root.kind)
            )
            if root_kind == kind_name:
                return root.digest
        raise DcrProofCacheError(f"dependency root not present: {kind_name}")

    def root_map(self) -> dict[str, str]:
        return {
            (
                root.kind.value
                if isinstance(root.kind, ProofDependencyRootKind)
                else str(root.kind)
            ): root.digest
            for root in self.dependency_roots
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "contract_version": DCR_PROOF_CACHE_VERSION,
            "obligation_id": self.obligation_id,
            "evidence_kind": (
                self.evidence_kind.value
                if isinstance(self.evidence_kind, DcrEvidenceKind)
                else str(self.evidence_kind)
            ),
            "kernel_version": self.kernel_version,
            "dependency_roots": [root.to_dict() for root in self.dependency_roots],
            "epoch": self.epoch,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DcrProofCacheKey":
        if not isinstance(value, Mapping):
            raise DcrProofCacheError("cache key must be an object")
        schema = value.get("schema")
        if schema not in (None, DCR_PROOF_CACHE_KEY_SCHEMA):
            raise DcrProofCacheError("unsupported DCR proof-cache key schema")
        return cls(
            obligation_id=value.get("obligation_id", ""),
            dependency_roots=tuple(value.get("dependency_roots") or ()),
            evidence_kind=value.get(
                "evidence_kind", DcrEvidenceKind.PROOF_KERNEL_RECEIPT
            ),
            kernel_version=value.get("kernel_version", DEFAULT_KERNEL_VERSION),
        )

    @property
    def key_id(self) -> str:
        return f"dcr-proof-cache-key:{self.content_id}"

    cache_key = key_id
    digest = key_id


@dataclass(frozen=True)
class DcrProofCacheEntry(CanonicalContract):
    """One durable cache entry for reconstructed DCR proof evidence."""

    SCHEMA: ClassVar[str] = DCR_PROOF_CACHE_ENTRY_SCHEMA

    key: DcrProofCacheKey
    receipt_cid: str
    evidence: Mapping[str, Any]
    claim: Mapping[str, Any] | None = None
    created_at_ms: int = 0
    expires_at_ms: int = 0
    entry_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, DcrProofCacheKey):
            raise DcrProofCacheError("entry.key must be a DcrProofCacheKey")
        object.__setattr__(
            self, "receipt_cid", _text(self.receipt_cid, "receipt_cid", required=True)
        )
        if not isinstance(self.evidence, Mapping):
            raise DcrProofCacheError("evidence must be an object")
        evidence = dict(self.evidence)
        if _contains_private_material(evidence):
            raise DcrProofCacheError(DcrCacheReason.PRIVATE_MATERIAL.value)
        object.__setattr__(self, "evidence", evidence)
        if self.claim is not None:
            if not isinstance(self.claim, Mapping):
                raise DcrProofCacheError("claim must be an object or null")
            claim = dict(self.claim)
            if _contains_private_material(claim):
                raise DcrProofCacheError(DcrCacheReason.PRIVATE_MATERIAL.value)
            object.__setattr__(self, "claim", claim)
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _nonnegative_int(self.expires_at_ms, "expires_at_ms")
        )
        computed = self.computed_digest
        if self.entry_digest and self.entry_digest != computed:
            raise DcrProofCacheError(DcrCacheReason.POISONED.value)
        object.__setattr__(self, "entry_digest", computed)

    @property
    def computed_digest(self) -> str:
        return _digest_of(self._unsigned_payload())

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": DCR_PROOF_CACHE_VERSION,
            "key_id": self.key.key_id,
            "key": self.key.to_dict(),
            "receipt_cid": self.receipt_cid,
            "evidence": dict(self.evidence),
            "claim": dict(self.claim) if self.claim is not None else None,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            **self._unsigned_payload(),
            "entry_digest": self.entry_digest or self.computed_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DcrProofCacheEntry":
        if not isinstance(value, Mapping):
            raise DcrProofCacheError("cache entry must be an object")
        schema = value.get("schema")
        if schema not in (None, DCR_PROOF_CACHE_ENTRY_SCHEMA):
            raise DcrProofCacheError("unsupported DCR proof-cache entry schema")
        key_payload = value.get("key")
        if not isinstance(key_payload, Mapping):
            raise DcrProofCacheError("cache entry requires key object")
        return cls(
            key=DcrProofCacheKey.from_dict(key_payload),
            receipt_cid=value.get("receipt_cid", ""),
            evidence=value.get("evidence") or {},
            claim=value.get("claim"),
            created_at_ms=value.get("created_at_ms", 0),
            expires_at_ms=value.get("expires_at_ms", 0),
            entry_digest=value.get("entry_digest", ""),
        )

    @classmethod
    def create(
        cls,
        *,
        key: DcrProofCacheKey,
        receipt_cid: str,
        evidence: Mapping[str, Any],
        claim: Mapping[str, Any] | None,
        created_at_ms: int,
        expires_at_ms: int,
    ) -> "DcrProofCacheEntry":
        return cls(
            key=key,
            receipt_cid=receipt_cid,
            evidence=evidence,
            claim=claim,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
        )


@dataclass(frozen=True)
class DcrCacheLookupResult:
    """Outcome of a cache lookup with full audit reasons."""

    disposition: DcrCacheDisposition
    key: DcrProofCacheKey
    receipt: ProofKernelReceipt | None = None
    counterexample: Counterexample | None = None
    reason_codes: tuple[str, ...] = ()
    reconstructed: bool = False
    receipt_cid: str = ""

    @property
    def hit(self) -> bool:
        return self.disposition is DcrCacheDisposition.HIT

    @property
    def authoritative(self) -> bool:
        return self.hit and self.reconstructed

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "key_id": self.key.key_id,
            "reason_codes": list(self.reason_codes),
            "reconstructed": self.reconstructed,
            "receipt_cid": self.receipt_cid,
            "hit": self.hit,
            "authoritative": self.authoritative,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "counterexample": (
                self.counterexample.to_dict()
                if self.counterexample is not None
                else None
            ),
        }


@dataclass(frozen=True)
class DcrCacheStoreResult:
    """Outcome of attempting to store reconstructed evidence."""

    stored: bool
    key: DcrProofCacheKey
    reason_codes: tuple[str, ...] = ()
    receipt_cid: str = ""
    entry: DcrProofCacheEntry | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stored": self.stored,
            "key_id": self.key.key_id,
            "reason_codes": list(self.reason_codes),
            "receipt_cid": self.receipt_cid,
            "entry_digest": self.entry.entry_digest if self.entry is not None else "",
        }


@dataclass(frozen=True)
class ProofInvalidationReceipt(CanonicalContract):
    """Receipt for one reverse-dependency invalidation action.

    Interface: ``ProofInvalidation@1``.
    """

    SCHEMA: ClassVar[str] = DCR_PROOF_INVALIDATION_SCHEMA
    INTERFACE: ClassVar[str] = PROOF_INVALIDATION_INTERFACE

    root: ProofDependencyRoot
    invalidated_key_ids: tuple[str, ...] = ()
    invalidated_at_ms: int = 0
    reason: str = DcrCacheReason.TOMBSTONED.value

    def __post_init__(self) -> None:
        if not isinstance(self.root, ProofDependencyRoot):
            raise DcrProofCacheError("invalidation root must be ProofDependencyRoot")
        object.__setattr__(
            self,
            "invalidated_key_ids",
            tuple(
                _text(item, "invalidated_key_ids", required=True)
                for item in self.invalidated_key_ids
            ),
        )
        object.__setattr__(
            self,
            "invalidated_at_ms",
            _nonnegative_int(self.invalidated_at_ms, "invalidated_at_ms"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=True, maximum=256)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "contract_version": DCR_PROOF_CACHE_VERSION,
            "root": self.root.to_dict(),
            "invalidated_key_ids": list(self.invalidated_key_ids),
            "invalidated_count": len(self.invalidated_key_ids),
            "invalidated_at_ms": self.invalidated_at_ms,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofInvalidationReceipt":
        if not isinstance(value, Mapping):
            raise DcrProofCacheError("invalidation receipt must be an object")
        schema = value.get("schema")
        if schema not in (None, DCR_PROOF_INVALIDATION_SCHEMA):
            raise DcrProofCacheError("unsupported invalidation receipt schema")
        root_payload = value.get("root")
        if not isinstance(root_payload, Mapping):
            raise DcrProofCacheError("invalidation receipt requires root object")
        return cls(
            root=ProofDependencyRoot.from_dict(root_payload),
            invalidated_key_ids=tuple(value.get("invalidated_key_ids") or ()),
            invalidated_at_ms=value.get("invalidated_at_ms", 0),
            reason=value.get("reason", DcrCacheReason.TOMBSTONED.value),
        )


@dataclass
class _Tombstone:
    root_id: str
    key_id: str
    invalidated_at_ms: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "key_id": self.key_id,
            "invalidated_at_ms": self.invalidated_at_ms,
            "reason": self.reason,
        }


class DcrProofCache:
    """Content-addressed DCR proof-evidence cache with reverse invalidation.

    Interfaces: ``ProofCache@1``, ``ProofInvalidation@1``.

    Durable authority is always re-derived: a hit reloads the stored evidence
    payload, re-runs :func:`reconstruct_proof` when a claim is present, and
    compares the reconstructed receipt CID against the stored receipt CID.
    """

    INTERFACE: ClassVar[str] = PROOF_CACHE_INTERFACE
    INVALIDATION_INTERFACE: ClassVar[str] = PROOF_INVALIDATION_INTERFACE

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_bytes = _positive_int(max_bytes, "max_bytes")
        self.default_ttl_seconds = _positive_int(
            default_ttl_seconds, "default_ttl_seconds"
        )
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._entries: dict[str, DcrProofCacheEntry] = {}
        self._root_index: dict[str, set[str]] = {}
        self._tombstones: dict[str, _Tombstone] = {}
        self._bytes_used = 0
        self._path: Path | None = Path(path) if path is not None else None
        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)
            self._load_durable()

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def invalidation_interface(self) -> str:
        return self.INVALIDATION_INTERFACE

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _coerce_key(
        self, key: DcrProofCacheKey | Mapping[str, Any]
    ) -> DcrProofCacheKey:
        return key if isinstance(key, DcrProofCacheKey) else DcrProofCacheKey.from_dict(key)

    def _entry_path(self, key_id: str) -> Path:
        assert self._path is not None
        digest = hashlib.sha256(key_id.encode("utf-8")).hexdigest()
        return self._path / "entries" / f"{digest}.json"

    def _index_path(self) -> Path:
        assert self._path is not None
        return self._path / "cache-index.json"

    def _load_durable(self) -> None:
        if self._path is None:
            return
        index_path = self._index_path()
        if not index_path.is_file():
            return
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, Mapping):
            return
        for raw in payload.get("entries") or ():
            if not isinstance(raw, Mapping):
                continue
            try:
                entry = DcrProofCacheEntry.from_dict(raw)
            except (DcrProofCacheError, ContractValidationError, TypeError, ValueError):
                continue
            self._entries[entry.key.key_id] = entry
            self._index_roots(entry.key)
            self._bytes_used += len(_canonical_bytes(entry.to_dict()))
        for raw in payload.get("tombstones") or ():
            if not isinstance(raw, Mapping):
                continue
            key_id = str(raw.get("key_id") or "").strip()
            if not key_id:
                continue
            self._tombstones[key_id] = _Tombstone(
                root_id=str(raw.get("root_id") or ""),
                key_id=key_id,
                invalidated_at_ms=int(raw.get("invalidated_at_ms") or 0),
                reason=str(raw.get("reason") or DcrCacheReason.TOMBSTONED.value),
            )

    def _persist(self) -> None:
        if self._path is None:
            return
        with self._lock:
            entries = [entry.to_dict() for entry in self._entries.values()]
            tombstones = [item.to_dict() for item in self._tombstones.values()]
            root_index = {
                root_id: sorted(key_ids)
                for root_id, key_ids in sorted(self._root_index.items())
            }
            payload = {
                "schema": DCR_PROOF_CACHE_INDEX_SCHEMA,
                "interface": PROOF_CACHE_INTERFACE,
                "invalidation_interface": PROOF_INVALIDATION_INTERFACE,
                "contract_version": DCR_PROOF_CACHE_VERSION,
                "entries": entries,
                "tombstones": tombstones,
                "root_index": root_index,
                "entry_count": len(entries),
                "bytes_used": self._bytes_used,
            }
            _atomic_write_json(self._index_path(), payload)
            for entry in self._entries.values():
                _atomic_write_json(self._entry_path(entry.key.key_id), entry.to_dict())

    def write_cache_index(self, path: str | Path) -> Path:
        """Materialize the reverse-dependency cache index at ``path``."""

        target = Path(path)
        with self._lock:
            payload = {
                "schema": DCR_PROOF_CACHE_INDEX_SCHEMA,
                "interface": PROOF_CACHE_INTERFACE,
                "invalidation_interface": PROOF_INVALIDATION_INTERFACE,
                "contract_version": DCR_PROOF_CACHE_VERSION,
                "entries": [entry.to_dict() for entry in self._entries.values()],
                "tombstones": [item.to_dict() for item in self._tombstones.values()],
                "root_index": {
                    root_id: sorted(key_ids)
                    for root_id, key_ids in sorted(self._root_index.items())
                },
                "entry_count": len(self._entries),
                "bytes_used": self._bytes_used,
            }
        _atomic_write_json(target, payload)
        return target

    def _index_roots(self, key: DcrProofCacheKey) -> None:
        for root in key.dependency_roots:
            self._root_index.setdefault(root.root_id, set()).add(key.key_id)

    def _unindex_roots(self, key: DcrProofCacheKey) -> None:
        for root in key.dependency_roots:
            indexed = self._root_index.get(root.root_id)
            if not indexed:
                continue
            indexed.discard(key.key_id)
            if not indexed:
                self._root_index.pop(root.root_id, None)

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        while (
            len(self._entries) >= self.max_entries
            or self._bytes_used + incoming_bytes > self.max_bytes
        ) and self._entries:
            # Evict oldest by created_at_ms.
            oldest_id = min(
                self._entries,
                key=lambda key_id: self._entries[key_id].created_at_ms,
            )
            removed = self._entries.pop(oldest_id)
            self._unindex_roots(removed.key)
            self._bytes_used = max(
                0, self._bytes_used - len(_canonical_bytes(removed.to_dict()))
            )

    @staticmethod
    def _receipt_cid(receipt: ProofKernelReceipt | Counterexample) -> str:
        return receipt.content_id

    def _admit_kernel_receipt(
        self, receipt: ProofKernelReceipt
    ) -> tuple[bool, tuple[str, ...]]:
        if not receipt.reconstructed or receipt.status is not ReconstructionStatus.RECONSTRUCTED:
            return False, (DcrCacheReason.NOT_RECONSTRUCTED.value,)
        if not receipt.independent:
            return False, (DcrCacheReason.NOT_RECONSTRUCTED.value,)
        if not receipt.valid:
            return False, (DcrCacheReason.INVALID_EVIDENCE.value,)
        return True, ()

    def _admit_counterexample(
        self, counterexample: Counterexample
    ) -> tuple[bool, tuple[str, ...]]:
        if not counterexample.minimized:
            return False, (DcrCacheReason.INVALID_EVIDENCE.value,)
        if counterexample.inferred_observations:
            return False, (DcrCacheReason.INVALID_EVIDENCE.value,)
        if not counterexample.graph_edge_ids and not counterexample.transcript_receipt_ids:
            return False, (DcrCacheReason.INVALID_EVIDENCE.value,)
        return True, ()

    def put(
        self,
        key: DcrProofCacheKey | Mapping[str, Any],
        evidence: ProofKernelReceipt | Counterexample | Mapping[str, Any],
        *,
        claim: ProofClaim | Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> DcrCacheStoreResult:
        """Store reconstructed evidence under an exact dependency-root key."""

        cache_key = self._coerce_key(key)
        ttl = (
            self.default_ttl_seconds
            if ttl_seconds is None
            else _positive_int(ttl_seconds, "ttl_seconds")
        )

        with self._lock:
            if cache_key.key_id in self._tombstones:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.TOMBSTONED.value,),
                )

        receipt: ProofKernelReceipt | None = None
        counterexample: Counterexample | None = None
        claim_payload: dict[str, Any] | None = None

        if isinstance(evidence, ProofKernelReceipt):
            receipt = evidence
        elif isinstance(evidence, Counterexample):
            counterexample = evidence
        elif isinstance(evidence, Mapping):
            interface = str(evidence.get("interface") or "")
            if interface == Counterexample.INTERFACE or "witness" in evidence:
                try:
                    counterexample = Counterexample.from_dict(evidence)
                except (KernelReconstructionError, ContractValidationError, TypeError, ValueError):
                    return DcrCacheStoreResult(
                        False,
                        cache_key,
                        reason_codes=(DcrCacheReason.MALFORMED.value,),
                    )
            else:
                try:
                    receipt = ProofKernelReceipt.from_dict(evidence)
                except (KernelReconstructionError, ContractValidationError, TypeError, ValueError):
                    return DcrCacheStoreResult(
                        False,
                        cache_key,
                        reason_codes=(DcrCacheReason.MALFORMED.value,),
                    )
        else:
            return DcrCacheStoreResult(
                False,
                cache_key,
                reason_codes=(DcrCacheReason.INVALID_EVIDENCE.value,),
            )

        if claim is not None:
            if isinstance(claim, ProofClaim):
                claim_payload = claim.to_dict()
            elif isinstance(claim, Mapping):
                claim_payload = dict(claim)
            else:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.MALFORMED.value,),
                )

        evidence_kind = (
            cache_key.evidence_kind
            if isinstance(cache_key.evidence_kind, DcrEvidenceKind)
            else DcrEvidenceKind(str(cache_key.evidence_kind))
        )

        if evidence_kind is DcrEvidenceKind.PROOF_KERNEL_RECEIPT:
            if receipt is None:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.INVALID_EVIDENCE.value,),
                )
            if receipt.obligation_id != cache_key.obligation_id:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.BINDING_MISMATCH.value,),
                )
            if receipt.kernel_version != cache_key.kernel_version:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.BINDING_MISMATCH.value,),
                )
            ok, reasons = self._admit_kernel_receipt(receipt)
            if not ok:
                return DcrCacheStoreResult(False, cache_key, reason_codes=reasons)
            # When a claim is supplied, cold reconstruction must equal the receipt.
            if claim_payload is not None:
                cold = reconstruct_proof(
                    claim_payload,
                    expected_root_ids=receipt.root_ids or None,
                    expected_tree_id=receipt.tree_id,
                    expected_graph_root=receipt.graph_root,
                    kernel_version=cache_key.kernel_version,
                )
                if not cold.valid or cold.content_id != receipt.content_id:
                    return DcrCacheStoreResult(
                        False,
                        cache_key,
                        reason_codes=(DcrCacheReason.RECONSTRUCTION_MISMATCH.value,),
                    )
            evidence_payload = receipt.to_dict()
            receipt_cid = self._receipt_cid(receipt)
        else:
            if counterexample is None:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.INVALID_EVIDENCE.value,),
                )
            if counterexample.obligation_id != cache_key.obligation_id:
                return DcrCacheStoreResult(
                    False,
                    cache_key,
                    reason_codes=(DcrCacheReason.BINDING_MISMATCH.value,),
                )
            ok, reasons = self._admit_counterexample(counterexample)
            if not ok:
                return DcrCacheStoreResult(False, cache_key, reason_codes=reasons)
            evidence_payload = counterexample.to_dict()
            receipt_cid = self._receipt_cid(counterexample)

        if _contains_private_material(evidence_payload):
            return DcrCacheStoreResult(
                False,
                cache_key,
                reason_codes=(DcrCacheReason.PRIVATE_MATERIAL.value,),
            )

        now = self._now_ms()
        try:
            entry = DcrProofCacheEntry.create(
                key=cache_key,
                receipt_cid=receipt_cid,
                evidence=evidence_payload,
                claim=claim_payload,
                created_at_ms=now,
                expires_at_ms=now + ttl * 1000,
            )
        except DcrProofCacheError as exc:
            reason = str(exc) or DcrCacheReason.MALFORMED.value
            mapped = (
                DcrCacheReason.PRIVATE_MATERIAL.value
                if reason == DcrCacheReason.PRIVATE_MATERIAL.value
                else DcrCacheReason.MALFORMED.value
            )
            return DcrCacheStoreResult(False, cache_key, reason_codes=(mapped,))

        entry_bytes = len(_canonical_bytes(entry.to_dict()))
        if entry_bytes > self.max_bytes:
            return DcrCacheStoreResult(
                False,
                cache_key,
                reason_codes=(DcrCacheReason.CAPACITY.value,),
            )

        with self._lock:
            if cache_key.key_id in self._entries:
                previous = self._entries[cache_key.key_id]
                if previous.receipt_cid != receipt_cid:
                    # Equivocation: same key, different reconstructed evidence.
                    self._tombstones[cache_key.key_id] = _Tombstone(
                        root_id="equivocation",
                        key_id=cache_key.key_id,
                        invalidated_at_ms=now,
                        reason=DcrCacheReason.POISONED.value,
                    )
                    self._entries.pop(cache_key.key_id, None)
                    self._unindex_roots(previous.key)
                    self._bytes_used = max(
                        0,
                        self._bytes_used - len(_canonical_bytes(previous.to_dict())),
                    )
                    self._persist()
                    return DcrCacheStoreResult(
                        False,
                        cache_key,
                        reason_codes=(DcrCacheReason.POISONED.value,),
                    )
                # Idempotent re-store of the same receipt.
                return DcrCacheStoreResult(
                    True,
                    cache_key,
                    reason_codes=(DcrCacheReason.STORED.value,),
                    receipt_cid=receipt_cid,
                    entry=previous,
                )

            self._evict_if_needed(entry_bytes)
            self._entries[cache_key.key_id] = entry
            self._index_roots(cache_key)
            self._bytes_used += entry_bytes
            self._persist()

        return DcrCacheStoreResult(
            True,
            cache_key,
            reason_codes=(DcrCacheReason.STORED.value,),
            receipt_cid=receipt_cid,
            entry=entry,
        )

    def lookup(
        self,
        key: DcrProofCacheKey | Mapping[str, Any],
        *,
        current_roots: Sequence[ProofDependencyRoot | Mapping[str, Any]] | None = None,
        current_epoch: str | None = None,
    ) -> DcrCacheLookupResult:
        """Lookup reconstructed evidence; hits always re-validate and re-reconstruct."""

        cache_key = self._coerce_key(key)

        with self._lock:
            tombstone = self._tombstones.get(cache_key.key_id)
            entry = self._entries.get(cache_key.key_id)

        if tombstone is not None:
            return DcrCacheLookupResult(
                DcrCacheDisposition.INVALIDATED,
                cache_key,
                reason_codes=(DcrCacheReason.TOMBSTONED.value,),
            )

        if entry is None:
            return DcrCacheLookupResult(
                DcrCacheDisposition.MISS,
                cache_key,
                reason_codes=(DcrCacheReason.CACHE_MISS.value,),
            )

        now = self._now_ms()
        if entry.expires_at_ms and entry.expires_at_ms < now:
            with self._lock:
                self._entries.pop(cache_key.key_id, None)
                self._unindex_roots(cache_key)
                self._bytes_used = max(
                    0, self._bytes_used - len(_canonical_bytes(entry.to_dict()))
                )
                self._persist()
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(DcrCacheReason.EXPIRED.value, DcrCacheReason.STALE.value),
            )

        # Exact dependency-root revalidation against the lookup key and optional
        # live current roots / epoch.
        if entry.key.key_id != cache_key.key_id:
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(DcrCacheReason.BINDING_MISMATCH.value,),
            )
        if entry.key.root_map() != cache_key.root_map():
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(
                    DcrCacheReason.ROOT_MISMATCH.value,
                    DcrCacheReason.STALE.value,
                ),
            )

        if current_epoch is not None:
            live_epoch = _text(current_epoch, "current_epoch", required=True)
            if live_epoch != cache_key.epoch:
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=(
                        DcrCacheReason.CROSS_EPOCH.value,
                        DcrCacheReason.STALE.value,
                    ),
                )

        if current_roots is not None:
            try:
                live_key = DcrProofCacheKey(
                    obligation_id=cache_key.obligation_id,
                    dependency_roots=tuple(current_roots),
                    evidence_kind=cache_key.evidence_kind,
                    kernel_version=cache_key.kernel_version,
                )
            except DcrProofCacheError:
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=(DcrCacheReason.IDENTITY_INVALID.value,),
                )
            if live_key.root_map() != cache_key.root_map():
                mismatched = [
                    kind
                    for kind, digest in cache_key.root_map().items()
                    if live_key.root_map().get(kind) != digest
                ]
                reasons = [DcrCacheReason.ROOT_MISMATCH.value, DcrCacheReason.STALE.value]
                if "epoch" in mismatched:
                    reasons.insert(0, DcrCacheReason.CROSS_EPOCH.value)
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=tuple(reasons),
                )

        # Integrity: rehash stored entry.
        try:
            reloaded = DcrProofCacheEntry.from_dict(entry.to_dict())
        except DcrProofCacheError:
            with self._lock:
                self._entries.pop(cache_key.key_id, None)
                self._unindex_roots(cache_key)
                self._persist()
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(DcrCacheReason.POISONED.value,),
            )
        if reloaded.entry_digest != entry.entry_digest:
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(DcrCacheReason.POISONED.value,),
            )

        evidence_kind = (
            cache_key.evidence_kind
            if isinstance(cache_key.evidence_kind, DcrEvidenceKind)
            else DcrEvidenceKind(str(cache_key.evidence_kind))
        )

        if evidence_kind is DcrEvidenceKind.PROOF_KERNEL_RECEIPT:
            try:
                stored_receipt = ProofKernelReceipt.from_dict(reloaded.evidence)
            except (KernelReconstructionError, ContractValidationError):
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=(DcrCacheReason.MALFORMED.value,),
                )
            if stored_receipt.content_id != reloaded.receipt_cid:
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=(DcrCacheReason.POISONED.value,),
                )
            if not stored_receipt.valid:
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=(DcrCacheReason.NOT_RECONSTRUCTED.value,),
                )

            # Cache-hit reconstruction equals a cold run when the claim is retained.
            if reloaded.claim is not None:
                cold = reconstruct_proof(
                    reloaded.claim,
                    expected_root_ids=stored_receipt.root_ids or None,
                    expected_tree_id=stored_receipt.tree_id,
                    expected_graph_root=stored_receipt.graph_root,
                    kernel_version=cache_key.kernel_version,
                )
                if not cold.valid:
                    return DcrCacheLookupResult(
                        DcrCacheDisposition.REJECTED,
                        cache_key,
                        reason_codes=(DcrCacheReason.RECONSTRUCTION_FAILED.value,),
                        receipt=cold,
                        reconstructed=False,
                        receipt_cid=reloaded.receipt_cid,
                    )
                if cold.content_id != stored_receipt.content_id:
                    return DcrCacheLookupResult(
                        DcrCacheDisposition.REJECTED,
                        cache_key,
                        reason_codes=(DcrCacheReason.RECONSTRUCTION_MISMATCH.value,),
                        receipt=cold,
                        reconstructed=False,
                        receipt_cid=reloaded.receipt_cid,
                    )
                # Prefer the freshly reconstructed receipt (equals cold run).
                return DcrCacheLookupResult(
                    DcrCacheDisposition.HIT,
                    cache_key,
                    receipt=cold,
                    reason_codes=(DcrCacheReason.CACHE_HIT.value,),
                    reconstructed=True,
                    receipt_cid=cold.content_id,
                )

            # No claim retained: re-admit the stored reconstructed receipt only.
            ok, reasons = self._admit_kernel_receipt(stored_receipt)
            if not ok:
                return DcrCacheLookupResult(
                    DcrCacheDisposition.REJECTED,
                    cache_key,
                    reason_codes=reasons,
                    receipt=stored_receipt,
                )
            return DcrCacheLookupResult(
                DcrCacheDisposition.HIT,
                cache_key,
                receipt=stored_receipt,
                reason_codes=(DcrCacheReason.CACHE_HIT.value,),
                reconstructed=True,
                receipt_cid=stored_receipt.content_id,
            )

        # Counterexample path: re-parse and re-validate minimization contract.
        try:
            stored_cx = Counterexample.from_dict(reloaded.evidence)
        except (KernelReconstructionError, ContractValidationError):
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(DcrCacheReason.MALFORMED.value,),
            )
        if stored_cx.content_id != reloaded.receipt_cid:
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=(DcrCacheReason.POISONED.value,),
            )
        ok, reasons = self._admit_counterexample(stored_cx)
        if not ok:
            return DcrCacheLookupResult(
                DcrCacheDisposition.REJECTED,
                cache_key,
                reason_codes=reasons,
                counterexample=stored_cx,
            )
        return DcrCacheLookupResult(
            DcrCacheDisposition.HIT,
            cache_key,
            counterexample=stored_cx,
            reason_codes=(DcrCacheReason.CACHE_HIT.value,),
            reconstructed=True,
            receipt_cid=stored_cx.content_id,
        )

    def get(
        self,
        key: DcrProofCacheKey | Mapping[str, Any],
        **options: Any,
    ) -> ProofKernelReceipt | Counterexample | None:
        result = self.lookup(key, **options)
        if not result.hit:
            return None
        return result.receipt if result.receipt is not None else result.counterexample

    def invalidate(
        self,
        root: ProofDependencyRoot | Mapping[str, Any],
        *,
        reason: str = DcrCacheReason.TOMBSTONED.value,
    ) -> ProofInvalidationReceipt:
        """Invalidate every cache key indexed under a changed dependency root."""

        dependency = (
            root
            if isinstance(root, ProofDependencyRoot)
            else ProofDependencyRoot.from_dict(root)
        )
        now = self._now_ms()
        invalidated: list[str] = []
        with self._lock:
            key_ids = set(self._root_index.get(dependency.root_id, set()))
            for key_id in sorted(key_ids):
                self._tombstones[key_id] = _Tombstone(
                    root_id=dependency.root_id,
                    key_id=key_id,
                    invalidated_at_ms=now,
                    reason=reason,
                )
                entry = self._entries.pop(key_id, None)
                if entry is not None:
                    self._unindex_roots(entry.key)
                    self._bytes_used = max(
                        0,
                        self._bytes_used - len(_canonical_bytes(entry.to_dict())),
                    )
                invalidated.append(key_id)
            self._root_index.pop(dependency.root_id, None)
            self._persist()
        return ProofInvalidationReceipt(
            root=dependency,
            invalidated_key_ids=tuple(invalidated),
            invalidated_at_ms=now,
            reason=reason,
        )

    # Alias matching the ProofInvalidation@1 surface.
    invalidate_dependency_root = invalidate

    def is_tombstoned(self, key: DcrProofCacheKey | Mapping[str, Any] | str) -> bool:
        if isinstance(key, str):
            key_id = key.strip()
        else:
            key_id = self._coerce_key(key).key_id
        with self._lock:
            return key_id in self._tombstones

    def contains(self, key: DcrProofCacheKey | Mapping[str, Any]) -> bool:
        result = self.lookup(key)
        return result.hit

    def __contains__(self, key: object) -> bool:
        if isinstance(key, (DcrProofCacheKey, Mapping)):
            return self.contains(key)  # type: ignore[arg-type]
        return False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "interface": self.INTERFACE,
                "invalidation_interface": self.INVALIDATION_INTERFACE,
                "entry_count": len(self._entries),
                "tombstone_count": len(self._tombstones),
                "bytes_used": self._bytes_used,
                "root_index_size": len(self._root_index),
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
            }


# Public aliases matching predicted AST symbols and interface names.
ProofCache = DcrProofCache
ProofInvalidation = DcrProofCache


def build_dcr_proof_cache_key(
    *,
    obligation_id: str,
    dependency_roots: Sequence[ProofDependencyRoot | Mapping[str, Any]] | None = None,
    evidence_kind: DcrEvidenceKind | str = DcrEvidenceKind.PROOF_KERNEL_RECEIPT,
    kernel_version: str = DEFAULT_KERNEL_VERSION,
    **root_values: Any,
) -> DcrProofCacheKey:
    """Build a complete DCR proof-cache key from roots or explicit root values."""

    if dependency_roots is None:
        missing = [name for name in _REQUIRED_ROOT_KINDS if name not in root_values]
        if missing:
            raise DcrProofCacheError(
                "build_dcr_proof_cache_key missing roots: " + ", ".join(missing)
            )
        dependency_roots = build_dependency_roots(
            **{name: root_values[name] for name in _REQUIRED_ROOT_KINDS}
        )
    return DcrProofCacheKey(
        obligation_id=obligation_id,
        dependency_roots=tuple(dependency_roots),
        evidence_kind=evidence_kind,
        kernel_version=kernel_version,
    )


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_TTL_SECONDS",
    "DCR_PROOF_CACHE_ENTRY_SCHEMA",
    "DCR_PROOF_CACHE_INDEX_SCHEMA",
    "DCR_PROOF_CACHE_KEY_SCHEMA",
    "DCR_PROOF_CACHE_VERSION",
    "DCR_PROOF_DEPENDENCY_ROOT_SCHEMA",
    "DCR_PROOF_INVALIDATION_SCHEMA",
    "DcrCacheDisposition",
    "DcrCacheLookupResult",
    "DcrCacheReason",
    "DcrCacheStoreResult",
    "DcrEvidenceKind",
    "DcrProofCache",
    "DcrProofCacheEntry",
    "DcrProofCacheError",
    "DcrProofCacheKey",
    "PROOF_CACHE_INTERFACE",
    "PROOF_INVALIDATION_INTERFACE",
    "ProofCache",
    "ProofDependencyRoot",
    "ProofDependencyRootKind",
    "ProofInvalidation",
    "ProofInvalidationReceipt",
    "build_dcr_proof_cache_key",
    "build_dependency_roots",
]
