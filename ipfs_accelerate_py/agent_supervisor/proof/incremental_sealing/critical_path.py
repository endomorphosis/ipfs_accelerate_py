"""Cold/warm hashing-critical-path instrumentation (PCTDD-004).

Observes the current sequential hashing/seal path with measured timing and
byte counters.  Instrumentation never changes identity, proof, storage,
execution, scheduler, or publication authority.  Native batch hashing,
production ZK, key ceremony, and direct-execution profiles remain typed
unavailable and cannot self-approve or upgrade a claim.

A verified exact-byte memo may reduce warm hashing.  Filesystem metadata is
candidate-only and never authorizes reuse.

Interfaces: ``StageCounters``, ``CriticalPathRun``, ``TypedUnavailable``,
``VerifiedByteMemo``, ``instrument_critical_path``, ``compare_cold_warm``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterator

EVIDENCE_SUBSET: Final[str] = "pctdd/cold-warm-critical-path@1"
RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "critical-path-run@1"
)
COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "cold-warm-comparison@1"
)
UNAVAILABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "typed-unavailable@1"
)

# Inventory labels from pctdd/hashing-critical-path@1.  Stage identifiers are
# the stable machine names; labels remain the inventory strings.
CURRENT_PATH_LABELS: Final[tuple[str, ...]] = (
    "source discovery",
    "canonical serialization",
    "SHA/CID",
    "proof verification",
    "Merkle",
    "immutable store",
    "serial WAL/CAS",
)
CURRENT_PATH_STAGES: Final[tuple[str, ...]] = (
    "source_discovery",
    "canonical_serialization",
    "sha_cid",
    "proof_verification",
    "merkle",
    "immutable_store",
    "serial_wal_cas",
)
_LABEL_BY_STAGE: Final[dict[str, str]] = dict(
    zip(CURRENT_PATH_STAGES, CURRENT_PATH_LABELS, strict=True)
)

CANDIDATE_METHODS: Final[tuple[str, ...]] = (
    "hashlib.file_digest",
    "readinto reusable buffer",
    "safe mmap",
    "threaded independent SHA",
    "process canonicalization",
    "optional qualified native batch",
)

NATIVE_BATCH_CANDIDATES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing._native_hash",
    "pctdd_native_hash",
)

UNKNOWN: Final[str] = "unknown"
MERKLE_LEAF_TAG: Final[bytes] = b"pctdd.critical-path.leaf.v1\n"
MERKLE_NODE_TAG: Final[bytes] = b"pctdd.critical-path.node.v1\n"
MERKLE_UNARY_TAG: Final[bytes] = b"pctdd.critical-path.unary.v1\n"
MERKLE_EMPTY_TAG: Final[bytes] = b"pctdd.critical-path.empty.v1\n"
ENVELOPE_DOMAIN: Final[bytes] = b"pctdd.critical-path.envelope.v1\n"


class CriticalPathError(ValueError):
    """Fail-closed critical-path instrumentation contract violation."""


class PathMode(str, Enum):
    COLD = "cold"
    WARM = "warm"


class CounterProvenance(str, Enum):
    MEASURED = "measured"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TypedUnavailable:
    """A capability that is absent without changing claim meaning."""

    capability: str
    reason_code: str
    message: str
    production_admitted: bool = False
    claim_unchanged: bool = True
    self_approved: bool = False

    def __post_init__(self) -> None:
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise CriticalPathError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": UNAVAILABLE_SCHEMA,
            "status": "typed_unavailable",
            "capability": self.capability,
            "reason_code": self.reason_code,
            "message": self.message,
            "production_admitted": False,
            "claim_unchanged": True,
            "self_approved": False,
        }


@dataclass(frozen=True, slots=True)
class SourceObject:
    """One discoverable hashing-path input.  Metadata is candidate-only."""

    source_id: str
    payload: bytes | Mapping[str, Any]
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not isinstance(self.source_id, str):
            raise CriticalPathError("source_id must be a non-empty string")
        if self.path is not None and not str(self.path).strip():
            raise CriticalPathError("source path must be a non-empty string")
        if not isinstance(self.payload, (bytes, bytearray, memoryview, Mapping)):
            raise CriticalPathError("payload must be bytes or a JSON-object mapping")


@dataclass(frozen=True, slots=True)
class StageCounters:
    """Measured timing and byte counters for one critical-path stage."""

    stage: str
    label: str
    wall_time_ns: int
    cpu_time_ns: int
    bytes_read: int
    bytes_hashed: int
    files_opened: int
    memo_hits: int
    memo_rejections: int
    provenance: CounterProvenance = CounterProvenance.MEASURED
    peak_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in CURRENT_PATH_STAGES:
            raise CriticalPathError(f"unknown stage {self.stage!r}")
        if self.label != _LABEL_BY_STAGE[self.stage]:
            raise CriticalPathError(f"stage label drift for {self.stage}")
        for name in (
            "wall_time_ns",
            "cpu_time_ns",
            "bytes_read",
            "bytes_hashed",
            "files_opened",
            "memo_hits",
            "memo_rejections",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CriticalPathError(f"{name} must be a non-negative int")
        if self.peak_rss_bytes is not None and (
            type(self.peak_rss_bytes) is not int or self.peak_rss_bytes < 0
        ):
            raise CriticalPathError("peak_rss_bytes must be a non-negative int or None")
        if self.provenance is CounterProvenance.UNKNOWN:
            raise CriticalPathError("recorded stage counters cannot be unknown")

    def to_canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "label": self.label,
            "wall_time_ns": self.wall_time_ns,
            "cpu_time_ns": self.cpu_time_ns,
            "bytes_read": self.bytes_read,
            "bytes_hashed": self.bytes_hashed,
            "files_opened": self.files_opened,
            "memo_hits": self.memo_hits,
            "memo_rejections": self.memo_rejections,
            "provenance": self.provenance.value,
            "peak_rss_bytes": (
                UNKNOWN if self.peak_rss_bytes is None else self.peak_rss_bytes
            ),
        }
        return payload


@dataclass(frozen=True, slots=True)
class LeafIdentity:
    """Exact-byte identity for one source.  CID is byte identity, not semantics."""

    source_id: str
    digest: str
    cid: str | None
    byte_length: int
    memo_hit: bool

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "digest": self.digest,
            "cid": self.cid,
            "byte_length": self.byte_length,
        }

    def to_canonical(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["memo_hit"] = self.memo_hit
        return payload


@dataclass(frozen=True, slots=True)
class CriticalPathRun:
    """Closed observation of one cold or warm pass over the current path."""

    schema: str
    evidence_subset: str
    mode: PathMode
    stages: tuple[StageCounters, ...]
    leaves: tuple[LeafIdentity, ...]
    merkle_root: str
    identity_digest: str
    publication_invoked: bool
    unavailable: tuple[TypedUnavailable, ...]
    candidate_methods: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != RUN_SCHEMA:
            raise CriticalPathError("run schema differs")
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise CriticalPathError("run evidence subset differs")
        if tuple(item.stage for item in self.stages) != CURRENT_PATH_STAGES:
            raise CriticalPathError("run must report every current-path stage in order")
        if self.publication_invoked:
            raise CriticalPathError(
                "critical-path instrumentation cannot invoke publication authority"
            )
        leaf_ids = tuple(item.source_id for item in self.leaves)
        if len(leaf_ids) != len(set(leaf_ids)):
            raise CriticalPathError("duplicate source_id")

    def stage(self, name: str) -> StageCounters:
        for item in self.stages:
            if item.stage == name:
                return item
        raise CriticalPathError(f"stage {name!r} was not recorded")

    def total_bytes_hashed(self) -> int:
        return sum(item.bytes_hashed for item in self.stages)

    def total_bytes_read(self) -> int:
        return sum(item.bytes_read for item in self.stages)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "mode": self.mode.value,
            "stages": [item.to_canonical() for item in self.stages],
            "leaves": [item.to_canonical() for item in self.leaves],
            "merkle_root": self.merkle_root,
            "identity_digest": self.identity_digest,
            "publication_invoked": False,
            "unavailable": [item.to_canonical() for item in self.unavailable],
            "candidate_methods": list(self.candidate_methods),
            "self_approved": False,
        }


@dataclass(frozen=True, slots=True)
class ColdWarmComparison:
    """Identity-preserving comparison of matched cold and warm runs."""

    schema: str
    evidence_subset: str
    cold: CriticalPathRun
    warm: CriticalPathRun
    identity_equal: bool
    merkle_equal: bool
    behavior_unchanged: bool

    def __post_init__(self) -> None:
        if self.schema != COMPARISON_SCHEMA:
            raise CriticalPathError("comparison schema differs")
        if self.cold.mode is not PathMode.COLD or self.warm.mode is not PathMode.WARM:
            raise CriticalPathError("comparison requires cold then warm runs")
        if not self.identity_equal or not self.merkle_equal or not self.behavior_unchanged:
            raise CriticalPathError("cold/warm instrumentation changed hashing behavior")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "cold": self.cold.to_canonical(),
            "warm": self.warm.to_canonical(),
            "identity_equal": True,
            "merkle_equal": True,
            "behavior_unchanged": True,
            "self_approved": False,
            "performance_target_claimed": False,
        }


@dataclass
class _Acc:
    stage: str
    wall_time_ns: int = 0
    cpu_time_ns: int = 0
    bytes_read: int = 0
    bytes_hashed: int = 0
    files_opened: int = 0
    memo_hits: int = 0
    memo_rejections: int = 0
    peak_rss_bytes: int | None = None
    _saw_rss: bool = False

    def snapshot(self) -> StageCounters:
        return StageCounters(
            stage=self.stage,
            label=_LABEL_BY_STAGE[self.stage],
            wall_time_ns=self.wall_time_ns,
            cpu_time_ns=self.cpu_time_ns,
            bytes_read=self.bytes_read,
            bytes_hashed=self.bytes_hashed,
            files_opened=self.files_opened,
            memo_hits=self.memo_hits,
            memo_rejections=self.memo_rejections,
            provenance=CounterProvenance.MEASURED,
            peak_rss_bytes=self.peak_rss_bytes if self._saw_rss else None,
        )


class VerifiedByteMemo:
    """Verified exact-byte memo.  Metadata never authorizes reuse."""

    def __init__(self) -> None:
        self._by_digest: dict[str, tuple[bytes, str | None]] = {}

    def remember(self, digest: str, canonical: bytes, cid: str | None) -> None:
        stored = self._by_digest.get(digest)
        if stored is not None and stored[0] != canonical:
            raise CriticalPathError("memo digest collision with unequal bytes")
        self._by_digest[digest] = (canonical, cid)

    def lookup_verified(self, canonical: bytes) -> tuple[str, str | None] | None:
        digest = _digest_label(canonical)
        stored = self._by_digest.get(digest)
        if stored is None:
            return None
        stored_bytes, cid = stored
        if not hmac.compare_digest(stored_bytes, canonical):
            return None
        return digest, cid

    def lookup_metadata(self, metadata: Mapping[str, Any]) -> None:
        del metadata
        return None

    def __len__(self) -> int:
        return len(self._by_digest)


class CandidateImmutableStore:
    """In-memory candidate store.  Rehashes before reference.  Cannot publish."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(self, digest: str, payload: bytes) -> str:
        recomputed = _digest_label(payload)
        if recomputed != digest:
            raise CriticalPathError("immutable put rehash mismatch")
        self._objects[digest] = payload
        return digest

    def publish_root(self) -> TypedUnavailable:
        return TypedUnavailable(
            capability="serial_wal_cas_publication",
            reason_code="instrumentation_has_no_publication_authority",
            message=(
                "critical-path instrumentation stores immutable candidates only "
                "and cannot publish a current root"
            ),
        )


def _digest_label(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonicalize(payload: bytes | Mapping[str, Any]) -> bytes:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    if not isinstance(payload, Mapping):
        raise CriticalPathError("canonical payload must be bytes or a mapping")
    from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

    return canonical_dag_json_bytes(dict(payload))


def _cid_for_bytes(data: bytes) -> str | TypedUnavailable:
    try:
        from ipfs_datasets_py.utils.cid_utils import cid_for_bytes

        cid = cid_for_bytes(
            data,
            base="base32",
            codec="raw",
            mh_type="sha2-256",
            version=1,
        )
    except Exception as exc:
        return TypedUnavailable(
            capability="cid_for_bytes",
            reason_code="cid_provider_unavailable",
            message=f"datasets cid_utils.cid_for_bytes unavailable: {exc!r}",
        )
    if not isinstance(cid, str) or not cid:
        return TypedUnavailable(
            capability="cid_for_bytes",
            reason_code="cid_provider_returned_empty",
            message="datasets cid_utils.cid_for_bytes returned an empty identity",
        )
    return cid


def _merkle_root(digests: Sequence[bytes]) -> bytes:
    if not digests:
        return hashlib.sha256(MERKLE_EMPTY_TAG).digest()
    level = [hashlib.sha256(MERKLE_LEAF_TAG + item).digest() for item in digests]
    while len(level) > 1:
        nxt: list[bytes] = []
        index = 0
        while index < len(level):
            if index + 1 < len(level):
                nxt.append(
                    hashlib.sha256(
                        MERKLE_NODE_TAG + level[index] + level[index + 1]
                    ).digest()
                )
                index += 2
            else:
                nxt.append(hashlib.sha256(MERKLE_UNARY_TAG + level[index]).digest())
                index += 1
        level = nxt
    return level[0]


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if rss <= 0:
        return None
    # Linux reports KiB; macOS reports bytes.  Record the raw observed value
    # without guessing the unit into a false measurement.
    return rss


def _observe_rss(acc: _Acc) -> None:
    rss = _peak_rss_bytes()
    if rss is None:
        return
    acc.peak_rss_bytes = rss
    acc._saw_rss = True


@contextmanager
def _measure(acc: _Acc) -> Iterator[None]:
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()
    try:
        yield
    finally:
        acc.wall_time_ns += time.perf_counter_ns() - wall0
        acc.cpu_time_ns += time.process_time_ns() - cpu0
        _observe_rss(acc)


def _coerce_source(raw: SourceObject | Mapping[str, Any]) -> SourceObject:
    if isinstance(raw, SourceObject):
        return raw
    if not isinstance(raw, Mapping):
        raise CriticalPathError("source must be SourceObject or mapping")
    payload = raw.get("payload")
    if payload is None:
        raise CriticalPathError("source mapping requires payload")
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise CriticalPathError("metadata must be a mapping")
    path = raw.get("path")
    return SourceObject(
        source_id=str(raw.get("source_id") or raw.get("id") or ""),
        payload=payload,  # type: ignore[arg-type]
        path=None if path in (None, "") else str(path),
        metadata=dict(metadata),
    )


def _read_source(source: SourceObject, acc: _Acc) -> bytes | Mapping[str, Any]:
    if source.path:
        path = Path(source.path)
        handle = path.open("rb")
        acc.files_opened += 1
        try:
            data = handle.read()
        finally:
            handle.close()
        acc.bytes_read += len(data)
        return data
    if isinstance(source.payload, (bytes, bytearray, memoryview)):
        data = bytes(source.payload)
        acc.bytes_read += len(data)
        return data
    encoded = json.dumps(
        dict(source.payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    acc.bytes_read += len(encoded)
    return dict(source.payload)


def probe_unavailable_capabilities() -> tuple[TypedUnavailable, ...]:
    """Record optional capabilities without claiming usability."""

    records: list[TypedUnavailable] = []
    for module_name in NATIVE_BATCH_CANDIDATES:
        try:
            __import__(module_name)
        except ImportError:
            records.append(
                TypedUnavailable(
                    capability="optional_qualified_native_batch",
                    reason_code="native_batch_hasher_not_installed",
                    message=(
                        f"{module_name} is not installed; hashlib.sha256 remains "
                        "the current hashing behavior and native batch is not admitted"
                    ),
                )
            )
            break
        records.append(
            TypedUnavailable(
                capability="optional_qualified_native_batch",
                reason_code="native_batch_hasher_unqualified",
                message=(
                    f"{module_name} imported but is not digest-bound or qualified; "
                    "presence is not production admission"
                ),
            )
        )
        break
    records.extend(
        (
            TypedUnavailable(
                capability="production_zk_proving",
                reason_code="production_zk_key_ceremony_unavailable",
                message=(
                    "production ZK proving and key ceremony remain typed unavailable; "
                    "instrumentation does not prove execution or admit simulated proof"
                ),
            ),
            TypedUnavailable(
                capability="direct_execution_profile",
                reason_code="direct_execution_profile_optional",
                message=(
                    "direct CPython execution profiles remain optional and unadmitted; "
                    "hashing instrumentation does not establish execution"
                ),
            ),
            TypedUnavailable(
                capability="serial_wal_cas_publication",
                reason_code="instrumentation_has_no_publication_authority",
                message=(
                    "serial WAL/CAS publication remains controller-owned; this "
                    "instrumentation cannot publish or advance a current root"
                ),
            ),
        )
    )
    return tuple(records)


def probe_candidate_methods(sample: bytes) -> dict[str, Any]:
    """Exercise candidate hash methods without replacing hashlib.sha256."""

    if not isinstance(sample, (bytes, bytearray, memoryview)):
        raise CriticalPathError("candidate probe requires bytes")
    payload = bytes(sample)
    expected = hashlib.sha256(payload).digest()
    file_digest_agrees = False
    try:
        file_digest = getattr(hashlib, "file_digest", None)
        if callable(file_digest):
            import io

            observed = file_digest(io.BytesIO(payload), "sha256").digest()
            file_digest_agrees = hmac.compare_digest(observed, expected)
    except Exception:
        file_digest_agrees = False
    return {
        "hashlib_sha256": "sha256:" + expected.hex(),
        "hashlib.file_digest": {
            "available": callable(getattr(hashlib, "file_digest", None)),
            "agrees_with_sha256": file_digest_agrees,
            "replaces_current_hasher": False,
        },
        "optional_qualified_native_batch": probe_unavailable_capabilities()[0].to_canonical(),
        "claim_changed": False,
    }


def instrument_critical_path(
    sources: Sequence[SourceObject | Mapping[str, Any]],
    *,
    mode: PathMode | str = PathMode.COLD,
    memo: VerifiedByteMemo | None = None,
) -> CriticalPathRun:
    """Run the current hashing path once and report stage counters.

    Identity is SHA-256 of canonical bytes (and CID when the datasets provider
    is available).  Warm mode may reuse a verified exact-byte memo.  Metadata
    lookups are rejected.  Publication is never invoked.
    """

    resolved_mode = PathMode(mode) if not isinstance(mode, PathMode) else mode
    if resolved_mode is PathMode.WARM and memo is None:
        raise CriticalPathError("warm mode requires a verified byte memo")
    if memo is None:
        memo = VerifiedByteMemo()

    parsed = [_coerce_source(item) for item in sources]
    parsed.sort(key=lambda item: item.source_id.encode("utf-8"))
    if len({item.source_id for item in parsed}) != len(parsed):
        raise CriticalPathError("duplicate source_id")

    unavailable = probe_unavailable_capabilities()
    store = CandidateImmutableStore()
    accs = {name: _Acc(stage=name) for name in CURRENT_PATH_STAGES}
    discovered: list[tuple[SourceObject, bytes | Mapping[str, Any]]] = []
    canonical_by_id: dict[str, bytes] = {}
    leaves: list[LeafIdentity] = []
    cid_unavailable: TypedUnavailable | None = None

    with _measure(accs["source_discovery"]):
        for source in parsed:
            if source.metadata and memo.lookup_metadata(source.metadata) is None:
                accs["source_discovery"].memo_rejections += 1
            discovered.append((source, _read_source(source, accs["source_discovery"])))

    with _measure(accs["canonical_serialization"]):
        for source, raw in discovered:
            canonical = _canonicalize(raw)
            accs["canonical_serialization"].bytes_read += len(canonical)
            canonical_by_id[source.source_id] = canonical

    digest_bytes: list[bytes] = []
    with _measure(accs["sha_cid"]):
        for source, _raw in discovered:
            canonical = canonical_by_id[source.source_id]
            hit = memo.lookup_verified(canonical)
            if hit is not None:
                digest, cid = hit
                accs["sha_cid"].memo_hits += 1
                memo_hit = True
            else:
                accs["sha_cid"].bytes_hashed += len(canonical)
                digest = _digest_label(canonical)
                cid_or_gap = _cid_for_bytes(canonical)
                if isinstance(cid_or_gap, TypedUnavailable):
                    cid = None
                    cid_unavailable = cid_or_gap
                else:
                    cid = cid_or_gap
                memo.remember(digest, canonical, cid)
                memo_hit = False
            digest_bytes.append(bytes.fromhex(digest.split(":", 1)[1]))
            leaves.append(
                LeafIdentity(
                    source_id=source.source_id,
                    digest=digest,
                    cid=cid,
                    byte_length=len(canonical),
                    memo_hit=memo_hit,
                )
            )

    with _measure(accs["proof_verification"]):
        for leaf in leaves:
            canonical = canonical_by_id[leaf.source_id]
            accs["proof_verification"].bytes_hashed += len(canonical)
            recomputed = _digest_label(canonical)
            if recomputed != leaf.digest:
                raise CriticalPathError("proof verification digest mismatch")
            if leaf.cid is not None:
                cid_or_gap = _cid_for_bytes(canonical)
                if isinstance(cid_or_gap, str) and cid_or_gap != leaf.cid:
                    raise CriticalPathError("proof verification CID mismatch")

    merkle = b""
    with _measure(accs["merkle"]):
        merkle = _merkle_root(digest_bytes)
        accs["merkle"].bytes_hashed += sum(len(item) for item in digest_bytes) + len(merkle)

    with _measure(accs["immutable_store"]):
        for leaf in leaves:
            canonical = canonical_by_id[leaf.source_id]
            accs["immutable_store"].bytes_hashed += len(canonical)
            store.put(leaf.digest, canonical)

    envelope = b""
    with _measure(accs["serial_wal_cas"]):
        publication = store.publish_root()
        envelope = (
            ENVELOPE_DOMAIN
            + merkle
            + json.dumps(
                {
                    "leaves": [item.to_canonical() for item in leaves],
                    "publication": publication.to_canonical(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        accs["serial_wal_cas"].bytes_hashed += len(envelope)
        _digest_label(envelope)

    extra_unavailable = list(unavailable)
    if cid_unavailable is not None:
        extra_unavailable.append(cid_unavailable)

    identity = _digest_label(
        json.dumps(
            {
                "leaves": [item.identity_payload() for item in leaves],
                "merkle_root": "sha256:" + merkle.hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return CriticalPathRun(
        schema=RUN_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        mode=resolved_mode,
        stages=tuple(accs[name].snapshot() for name in CURRENT_PATH_STAGES),
        leaves=tuple(leaves),
        merkle_root="sha256:" + merkle.hex(),
        identity_digest=identity,
        publication_invoked=False,
        unavailable=tuple(extra_unavailable),
        candidate_methods=CANDIDATE_METHODS,
    )


def compare_cold_warm(
    sources: Sequence[SourceObject | Mapping[str, Any]],
    *,
    memo: VerifiedByteMemo | None = None,
) -> ColdWarmComparison:
    """Run cold then warm over the same sources and require identical identity."""

    shared = memo if memo is not None else VerifiedByteMemo()
    cold = instrument_critical_path(sources, mode=PathMode.COLD, memo=shared)
    warm = instrument_critical_path(sources, mode=PathMode.WARM, memo=shared)
    identity_equal = (
        cold.identity_digest == warm.identity_digest
        and tuple(item.digest for item in cold.leaves)
        == tuple(item.digest for item in warm.leaves)
        and tuple(item.cid for item in cold.leaves)
        == tuple(item.cid for item in warm.leaves)
    )
    merkle_equal = cold.merkle_root == warm.merkle_root
    return ColdWarmComparison(
        schema=COMPARISON_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        cold=cold,
        warm=warm,
        identity_equal=identity_equal,
        merkle_equal=merkle_equal,
        behavior_unchanged=identity_equal and merkle_equal,
    )


__all__ = (
    "CANDIDATE_METHODS",
    "COMPARISON_SCHEMA",
    "CURRENT_PATH_LABELS",
    "CURRENT_PATH_STAGES",
    "EVIDENCE_SUBSET",
    "RUN_SCHEMA",
    "UNKNOWN",
    "CandidateImmutableStore",
    "ColdWarmComparison",
    "CounterProvenance",
    "CriticalPathError",
    "CriticalPathRun",
    "LeafIdentity",
    "PathMode",
    "SourceObject",
    "StageCounters",
    "TypedUnavailable",
    "VerifiedByteMemo",
    "compare_cold_warm",
    "instrument_critical_path",
    "probe_candidate_methods",
    "probe_unavailable_capabilities",
)
