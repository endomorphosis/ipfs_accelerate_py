"""PCTDD-018: bounded parallel proof/certificate verification.

Independently verify proof, signature, receipt, and integrity checks under
closed resource, timeout, and size bounds, then deterministically aggregate
accepted units.  Parallel workers never publish, never self-approve, and never
change claim meaning.

Hermetic HMAC and SHA-256 digest checks are the admitted local path.
Production ZK, key ceremony, unqualified Groth16, provekit, recursive
child verification, direct-execution, aggregate selected-test ZK, and serial
WAL/CAS publication remain typed unavailable.

Import is cold-safe: no pytest, network, package installer, or prover
subprocess.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures import as_completed
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.aggregation import (
    AGGREGATION_LABEL_MANIFEST,
    DEFAULT_FAN_IN,
    AggregationMode,
    ManifestAggregationResult,
    ProofAggregator,
    VerifiedUnit,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    CancellationToken,
    ProcessControlError,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.scheduling import (
    ProofResourcePolicy,
    ProofWorkItem,
    WorkClass,
    build_proof_schedule,
)
from ipfs_accelerate_py.agent_supervisor.runtime.hash_pressure import (
    HashingResourceTimeout,
    hash_worker_limit,
    hashing_worker_slot,
)

# This module is accelerate integration, not a pytest test module.
__test__ = False

EVIDENCE_SUBSET: Final[str] = "pctdd/parallel-proof-verification@1"
RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "parallel-proof-verification-result@1"
)
UNIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "parallel-proof-verification-unit@1"
)
BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "parallel-proof-verification-bounds@1"
)
UNAVAILABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "typed-unavailable@1"
)

PARALLEL_PROOF_VERIFICATION_INTERFACE: Final[str] = "ParallelProofVerification@1"
PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE: Final[str] = (
    "ParallelProofVerificationResult@1"
)
PARALLEL_PROOF_VERIFICATION_BOUNDS_INTERFACE: Final[str] = (
    "ParallelProofVerificationBounds@1"
)
CLAIM_CLASS: Final[str] = "IntegrityCommitment"
PREDECESSOR_INTERFACES: Final[tuple[str, ...]] = (
    "SealVerification@1",
    "ProofAggregator@1",
    "IncrementalProofBackendAdapter@1",
    "ProofWorkScheduler@1",
)

VERIFICATION_ESTABLISHES: Final[str] = (
    "proof, signature, receipt and integrity checks are bounded and "
    "deterministically aggregated"
)
VERIFICATION_DOES_NOT: Final[str] = (
    "execution or semantics; current-root publication; production ZK; "
    "recursive child verification; underlying test execution; self-approval"
)

_PROOF_DOMAIN: Final[bytes] = b"pctdd-018/parallel-proof-hmac@1\n"
_SIGNATURE_DOMAIN: Final[bytes] = b"pctdd-018/parallel-signature-hmac@1\n"
_DIGEST_PREFIX: Final[str] = "sha256:"

DEFAULT_MAX_PARALLEL: Final[int] = 4
DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
DEFAULT_MAX_UNIT_BYTES: Final[int] = 65_536
MAX_PARALLEL_CAP: Final[int] = 16
WORKER_BOUNDS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)

CLOSED_CHECK_KINDS: Final[frozenset[str]] = frozenset(
    {"proof", "signature", "receipt", "integrity"}
)
ADMITTED_BACKEND_IDS: Final[frozenset[str]] = frozenset(
    {
        "hermetic_hmac",
        "integrity",
        "signed_receipt",
        "merkle_manifest",
        "receipt_aggregation",
    }
)
UNAVAILABLE_BACKEND_IDS: Final[frozenset[str]] = frozenset(
    {"groth16", "provekit", "simulated"}
)
KNOWN_BACKEND_IDS: Final[frozenset[str]] = (
    ADMITTED_BACKEND_IDS | UNAVAILABLE_BACKEND_IDS
)

_SENSITIVE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "witness",
        "witness_bytes",
        "witness_material",
        "proving_key",
        "proving_key_bytes",
        "private_key",
        "private_key_bytes",
        "secret",
        "trapdoor",
        "raw_key",
        "key_bytes",
    }
)

_WORK_CLASS_BY_KIND: Final[dict[str, WorkClass]] = {
    "integrity": WorkClass.CACHE_VERIFICATION,
    "signature": WorkClass.SMALL_INDEPENDENT,
    "receipt": WorkClass.SMALL_INDEPENDENT,
    "proof": WorkClass.CRITICAL_PATH,
}

_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; parallel "
        "proof/certificate verification cannot admit simulated, structural, "
        "or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by parallel "
        "proof/certificate verification",
    ),
    (
        "groth16_unqualified",
        "groth16_backend_present_unqualified",
        "packaged Groth16 backend bytes, when present, do not establish "
        "ceremony, allowlist, or current public-input binding",
    ),
    (
        "provekit",
        "provekit_adapter_source_present_unadmitted",
        "provekit remains an unadmitted optional adapter; absence or failed "
        "capability probe is typed unavailable, never simulated success",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade integrity, signature, receipt, or hermetic "
        "proof checks",
    ),
    (
        "recursive_verification",
        "recursion_not_admitted",
        "recursive child-proof verification is selectable only after a live "
        "backend capability probe; the default path is Merkleized manifest "
        "aggregation and does not recursively verify children",
    ),
    (
        "aggregate_selected_test_zk",
        "aggregate_selected_test_zk_missing",
        "aggregate selected-test ZK remains a versioned successor; bounded "
        "manifest aggregation cannot exceed leaf evidence or prove CPython "
        "execution",
    ),
    (
        "serial_wal_cas_publication",
        "verification_has_no_publication_authority",
        "parallel proof/certificate verification prepares immutable "
        "verification outcomes only; final ordered WAL and current-root CAS "
        "remain serial, controller-owned, and fail closed",
    ),
)


class ParallelVerificationError(ValueError):
    """Fail-closed parallel proof/certificate verification contract violation."""


class CheckKind(str, Enum):
    PROOF = "proof"
    SIGNATURE = "signature"
    RECEIPT = "receipt"
    INTEGRITY = "integrity"


class UnitVerificationReason(str, Enum):
    VERIFIED = "verified"
    DIGEST_MISMATCH = "digest_mismatch"
    SIGNATURE_FAILURE = "signature_failure"
    PROOF_FAILURE = "proof_failure"
    UNALLOWLISTED_SIGNER = "unallowlisted_signer"
    UNALLOWLISTED_VERIFICATION_KEY = "unallowlisted_verification_key"
    UNKNOWN_BACKEND = "unknown_backend"
    UNKNOWN_KIND = "unknown_kind"
    BOUND_EXCEEDED = "bound_exceeded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"
    DUPLICATE_UNIT = "duplicate_unit"
    FAILED_CHILD = "failed_child"


class BatchVerificationReason(str, Enum):
    AGGREGATED = "aggregated"
    UNIT_REJECTED = "unit_rejected"
    MISSING_CHILD = "missing_child"
    DUPLICATE_CHILD = "duplicate_child"
    REORDERED_CHILDREN = "reordered_children"
    FAILED_CHILD = "failed_child"
    BOUND_EXCEEDED = "bound_exceeded"
    CANCELLED = "cancelled"
    EMPTY = "empty"


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParallelVerificationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ParallelVerificationError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ParallelVerificationError(f"{field_name} must be a positive int")
    return value


def _require_positive_float(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) <= 0
    ):
        raise ParallelVerificationError(f"{field_name} must be a positive number")
    return float(value)


def _require_bytes(value: Any, field_name: str, *, allow_empty: bool = False) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ParallelVerificationError(f"{field_name} must be bytes")
    data = bytes(value)
    if not allow_empty and not data:
        raise ParallelVerificationError(f"{field_name} must be non-empty")
    return data


def digest_bytes(payload: bytes) -> str:
    """Return ``sha256:<hex>`` of exact payload bytes."""

    data = _require_bytes(payload, "payload", allow_empty=True)
    return _DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def _digests_equal(left: str, right: str) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _bytes_equal(left: bytes, right: bytes) -> bool:
    if len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _cid(payload: Mapping[str, Any]) -> str:
    return digest_bytes(_canonical_json(payload).encode("utf-8"))


def assert_no_sensitive_material(payload: Mapping[str, Any]) -> None:
    """Reject public surfaces that appear to carry witness or key material."""

    if not isinstance(payload, Mapping):
        raise ParallelVerificationError("payload must be a mapping")
    for key, value in payload.items():
        if str(key).casefold() in _SENSITIVE_FIELD_NAMES:
            raise ParallelVerificationError(
                f"sensitive field {key!r} must not appear on public receipts"
            )
        if isinstance(value, Mapping):
            assert_no_sensitive_material(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    assert_no_sensitive_material(item)


def _hmac_key(domain: bytes, identity: str) -> bytes:
    return hashlib.sha256(domain + identity.encode("utf-8")).digest()


def hermetic_proof_tag(
    *,
    unit_id: str,
    public_input: bytes,
    verification_key_id: str,
) -> bytes:
    """Build the hermetic HMAC tag for one proof check.

    The tag binds unit identity, public input, and verification-key id.  It is
    an integrity-class hermetic adapter, not production ZK.
    """

    key = _hmac_key(_PROOF_DOMAIN, verification_key_id)
    return hmac.new(
        key,
        b"proof\0"
        + unit_id.encode("utf-8")
        + b"\0"
        + public_input
        + b"\0"
        + verification_key_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def hermetic_signature_tag(*, signer_id: str, payload: bytes) -> bytes:
    """Build the hermetic HMAC tag for one signature check."""

    key = _hmac_key(_SIGNATURE_DOMAIN, signer_id)
    return hmac.new(key, b"signature\0" + payload, hashlib.sha256).digest()


def integrity_proof_tag(payload: bytes) -> bytes:
    """Return the SHA-256 digest bytes used as an integrity proof object."""

    return hashlib.sha256(_require_bytes(payload, "payload", allow_empty=True)).digest()


def record_typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    """Record a typed unavailable case without changing claim meaning."""

    record = {
        "schema": UNAVAILABLE_SCHEMA,
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if (
        record["production_admitted"]
        or record["self_approved"]
        or not record["claim_unchanged"]
    ):
        raise ParallelVerificationError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-018 typed unavailable capabilities."""

    return tuple(
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    )


VERIFICATION_POLICY: Final[Mapping[str, Any]] = {
    "interface": PARALLEL_PROOF_VERIFICATION_INTERFACE,
    "result_interface": PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE,
    "predecessor_interfaces": list(PREDECESSOR_INTERFACES),
    "claim_class": CLAIM_CLASS,
    "establishes": VERIFICATION_ESTABLISHES,
    "does_not": VERIFICATION_DOES_NOT,
    "may_authorize_skip": False,
    "production_admitted": False,
    "self_approved": False,
    "publication_authority_invoked": False,
    "recursively_verifies_children": False,
    "claims_test_execution": False,
    "aggregation_label": AGGREGATION_LABEL_MANIFEST,
    "check_kinds": sorted(CLOSED_CHECK_KINDS),
    "worker_bounds": list(WORKER_BOUNDS),
}


@dataclass(frozen=True, slots=True)
class VerificationBounds:
    """Closed resource, timeout, size, and fan-in envelope."""

    max_parallel: int = DEFAULT_MAX_PARALLEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_unit_bytes: int = DEFAULT_MAX_UNIT_BYTES
    fan_in: int = DEFAULT_FAN_IN
    max_cpu: int = 4
    max_memory_mb: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_parallel",
            _require_positive_int(self.max_parallel, "max_parallel"),
        )
        if self.max_parallel > MAX_PARALLEL_CAP:
            raise ParallelVerificationError(
                f"max_parallel must be <= {MAX_PARALLEL_CAP}"
            )
        object.__setattr__(
            self,
            "timeout_seconds",
            _require_positive_float(self.timeout_seconds, "timeout_seconds"),
        )
        object.__setattr__(
            self,
            "max_unit_bytes",
            _require_positive_int(self.max_unit_bytes, "max_unit_bytes"),
        )
        object.__setattr__(
            self, "fan_in", _require_positive_int(self.fan_in, "fan_in")
        )
        if self.fan_in < 2:
            raise ParallelVerificationError("fan_in must be >= 2")
        object.__setattr__(
            self, "max_cpu", _require_positive_int(self.max_cpu, "max_cpu")
        )
        object.__setattr__(
            self,
            "max_memory_mb",
            _require_positive_int(self.max_memory_mb, "max_memory_mb"),
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": BOUNDS_SCHEMA,
            "interface": PARALLEL_PROOF_VERIFICATION_BOUNDS_INTERFACE,
            "max_parallel": self.max_parallel,
            "timeout_seconds": self.timeout_seconds,
            "max_unit_bytes": self.max_unit_bytes,
            "fan_in": self.fan_in,
            "max_cpu": self.max_cpu,
            "max_memory_mb": self.max_memory_mb,
            "max_parallel_cap": MAX_PARALLEL_CAP,
            "worker_bounds": list(WORKER_BOUNDS),
        }

    def resource_policy(self) -> ProofResourcePolicy:
        return ProofResourcePolicy(
            max_cpu=self.max_cpu,
            max_memory_mb=self.max_memory_mb,
            max_gpu=0,
            max_parallel=self.max_parallel,
            max_fan_in=self.fan_in,
            reject_simulated_gpu=True,
        )


@dataclass(frozen=True, slots=True)
class VerificationUnit:
    """One independent proof, signature, receipt, or integrity check."""

    unit_id: str
    kind: CheckKind
    payload: bytes
    expected_digest: str
    proof_bytes: bytes = b""
    signer_id: str = ""
    verification_key_id: str = ""
    backend_id: str = "integrity"
    category: str = "unit_test"
    production: bool = False
    cpu: int = 1
    memory_mb: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _require_nonempty_str(self.unit_id, "unit_id")
        )
        kind = self.kind
        if isinstance(kind, str):
            try:
                kind = CheckKind(kind)
            except ValueError as exc:
                raise ParallelVerificationError(
                    f"unknown check kind {self.kind!r}"
                ) from exc
            object.__setattr__(self, "kind", kind)
        if not isinstance(self.kind, CheckKind):
            raise ParallelVerificationError("kind must be CheckKind")
        object.__setattr__(
            self, "payload", _require_bytes(self.payload, "payload", allow_empty=True)
        )
        object.__setattr__(
            self,
            "expected_digest",
            _require_nonempty_str(self.expected_digest, "expected_digest"),
        )
        if not self.expected_digest.startswith(_DIGEST_PREFIX):
            raise ParallelVerificationError(
                "expected_digest must be a sha256:<hex> label"
            )
        object.__setattr__(
            self,
            "proof_bytes",
            _require_bytes(self.proof_bytes, "proof_bytes", allow_empty=True),
        )
        object.__setattr__(self, "signer_id", str(self.signer_id or ""))
        object.__setattr__(
            self, "verification_key_id", str(self.verification_key_id or "")
        )
        object.__setattr__(
            self,
            "backend_id",
            _require_nonempty_str(self.backend_id, "backend_id"),
        )
        object.__setattr__(
            self, "category", _require_nonempty_str(self.category, "category")
        )
        object.__setattr__(
            self, "production", _require_bool(self.production, "production")
        )
        object.__setattr__(self, "cpu", _require_positive_int(self.cpu, "cpu"))
        object.__setattr__(
            self, "memory_mb", _require_positive_int(self.memory_mb, "memory_mb")
        )

    @property
    def byte_length(self) -> int:
        return len(self.payload) + len(self.proof_bytes)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": UNIT_SCHEMA,
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "expected_digest": self.expected_digest,
            "payload_byte_length": len(self.payload),
            "proof_byte_length": len(self.proof_bytes),
            "signer_id": self.signer_id,
            "verification_key_id": self.verification_key_id,
            "backend_id": self.backend_id,
            "category": self.category,
            "production": self.production,
            "proof_bytes_exported": False,
            "witness_exported": False,
        }


@dataclass(frozen=True, slots=True)
class UnitVerificationRecord:
    """Per-unit verification disposition.  Never carries proof/witness bytes."""

    unit_id: str
    kind: CheckKind
    accepted: bool
    reason: UnitVerificationReason
    digest: str
    backend_id: str
    bounded: bool
    production_admitted: bool = False
    claim_unchanged: bool = True
    self_approved: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        if self.production_admitted:
            raise ParallelVerificationError(
                "unit verification must not admit production"
            )
        if self.self_approved:
            raise ParallelVerificationError("unit verification must not self-approve")
        if not self.claim_unchanged:
            raise ParallelVerificationError(
                "unit verification must not change claim meaning"
            )
        if self.accepted and self.reason is not UnitVerificationReason.VERIFIED:
            raise ParallelVerificationError("accepted units require reason VERIFIED")
        if not self.accepted and self.reason is UnitVerificationReason.VERIFIED:
            raise ParallelVerificationError("rejected units cannot use reason VERIFIED")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "accepted": self.accepted,
            "reason": self.reason.value,
            "digest": self.digest,
            "backend_id": self.backend_id,
            "bounded": self.bounded,
            "production_admitted": False,
            "claim_unchanged": True,
            "self_approved": False,
            "message": self.message,
            "proof_bytes_exported": False,
            "witness_exported": False,
        }


@dataclass(frozen=True, slots=True)
class ParallelVerificationHooks:
    """Optional cooperative hooks.  Must not publish or mutate claim meaning."""

    before_unit: Callable[[VerificationUnit], None] | None = None
    after_unit: Callable[[VerificationUnit, UnitVerificationRecord], None] | None = None


@dataclass(frozen=True, slots=True)
class ParallelProofVerificationResult:
    """Deterministic aggregate of independently verified units."""

    schema: str
    evidence_subset: str
    interface: str
    accepted: bool
    reason: BatchVerificationReason
    declared_unit_ids: tuple[str, ...]
    units: tuple[UnitVerificationRecord, ...]
    aggregate_root: str
    aggregation_label: str
    recursively_verifies_children: bool
    claims_test_execution: bool
    bounds: VerificationBounds
    schedule_work_ids: tuple[str, ...]
    production_admitted: bool = False
    publication_authority_invoked: bool = False
    self_approved: bool = False
    claim_class: str = CLAIM_CLASS
    establishes: str = VERIFICATION_ESTABLISHES
    does_not: str = VERIFICATION_DOES_NOT
    bounded: bool = True
    may_authorize_skip: bool = False
    completion_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise ParallelVerificationError(f"schema must be {RESULT_SCHEMA}")
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise ParallelVerificationError(
                f"evidence_subset must be {EVIDENCE_SUBSET}"
            )
        if self.interface != PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE:
            raise ParallelVerificationError(
                f"interface must be {PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE}"
            )
        if self.production_admitted:
            raise ParallelVerificationError(
                "parallel verification must not admit production"
            )
        if self.publication_authority_invoked:
            raise ParallelVerificationError(
                "parallel verification has no publication authority"
            )
        if self.self_approved:
            raise ParallelVerificationError("parallel verification must not self-approve")
        if self.may_authorize_skip:
            raise ParallelVerificationError(
                "parallel verification must not authorize skip"
            )
        if self.recursively_verifies_children:
            raise ParallelVerificationError(
                "parallel verification must not recursively verify children"
            )
        if self.claims_test_execution:
            raise ParallelVerificationError(
                "parallel verification must not claim test execution"
            )
        if self.aggregation_label != AGGREGATION_LABEL_MANIFEST:
            raise ParallelVerificationError(
                "aggregation_label must be manifest_aggregation"
            )
        if self.claim_class != CLAIM_CLASS:
            raise ParallelVerificationError("claim_class must remain IntegrityCommitment")
        if self.accepted and self.reason is not BatchVerificationReason.AGGREGATED:
            raise ParallelVerificationError(
                "accepted batches require reason AGGREGATED"
            )
        if not self.accepted and self.reason is BatchVerificationReason.AGGREGATED:
            raise ParallelVerificationError(
                "rejected batches cannot use reason AGGREGATED"
            )
        declared = tuple(item.unit_id for item in self.units)
        if declared != self.declared_unit_ids:
            raise ParallelVerificationError(
                "unit records must follow declared unit order"
            )
        assert_no_sensitive_material(self.to_canonical())

    @property
    def rejected(self) -> bool:
        return not self.accepted

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "interface": self.interface,
            "accepted": self.accepted,
            "reason": self.reason.value,
            "declared_unit_ids": list(self.declared_unit_ids),
            "units": [item.to_canonical() for item in self.units],
            "aggregate_root": self.aggregate_root,
            "aggregation_label": self.aggregation_label,
            "aggregation_mode": AggregationMode.MANIFEST.value,
            "recursively_verifies_children": False,
            "claims_test_execution": False,
            "bounds": self.bounds.to_canonical(),
            "schedule_work_ids": list(self.schedule_work_ids),
            "production_admitted": False,
            "publication_authority_invoked": False,
            "self_approved": False,
            "claim_class": self.claim_class,
            "establishes": self.establishes,
            "does_not": self.does_not,
            "bounded": self.bounded,
            "may_authorize_skip": False,
            "proof_bytes_exported": False,
            "witness_exported": False,
        }

    def result_cid(self) -> str:
        return _cid(self.to_canonical())


def _reject_unit(
    unit: VerificationUnit,
    reason: UnitVerificationReason,
    message: str,
    *,
    bounded: bool = True,
    digest: str = "",
) -> UnitVerificationRecord:
    return UnitVerificationRecord(
        unit_id=unit.unit_id,
        kind=unit.kind,
        accepted=False,
        reason=reason,
        digest=digest or unit.expected_digest,
        backend_id=unit.backend_id,
        bounded=bounded,
        message=message,
    )


def _accept_unit(
    unit: VerificationUnit,
    digest: str,
    *,
    bounded: bool = True,
) -> UnitVerificationRecord:
    return UnitVerificationRecord(
        unit_id=unit.unit_id,
        kind=unit.kind,
        accepted=True,
        reason=UnitVerificationReason.VERIFIED,
        digest=digest,
        backend_id=unit.backend_id,
        bounded=bounded,
        message="verified under current bounds and allowlists",
    )


def _verify_one(
    unit: VerificationUnit,
    *,
    bounds: VerificationBounds,
    allowlisted_signers: frozenset[str],
    allowlisted_verification_keys: frozenset[str],
    cancellation: CancellationToken | None,
    hooks: ParallelVerificationHooks | None,
    deadline: float | None = None,
) -> UnitVerificationRecord:
    verification_deadline = (
        time.monotonic() + bounds.timeout_seconds if deadline is None else deadline
    )
    if cancellation is not None:
        try:
            cancellation.check()
        except ProcessControlError:
            return _reject_unit(
                unit,
                UnitVerificationReason.CANCELLED,
                "verification cancelled before unit admission",
            )
    if hooks is not None and hooks.before_unit is not None:
        hooks.before_unit(unit)
        if cancellation is not None:
            try:
                cancellation.check()
            except ProcessControlError:
                return _reject_unit(
                    unit,
                    UnitVerificationReason.CANCELLED,
                    "verification cancelled during unit hook",
                )

    if unit.byte_length > bounds.max_unit_bytes:
        return _reject_unit(
            unit,
            UnitVerificationReason.BOUND_EXCEEDED,
            (
                f"unit {unit.unit_id!r} exceeds max_unit_bytes "
                f"{bounds.max_unit_bytes}"
            ),
            bounded=True,
        )

    if unit.backend_id not in KNOWN_BACKEND_IDS:
        return _reject_unit(
            unit,
            UnitVerificationReason.UNKNOWN_BACKEND,
            f"unknown backend {unit.backend_id!r}",
        )
    if unit.backend_id in UNAVAILABLE_BACKEND_IDS:
        return _reject_unit(
            unit,
            UnitVerificationReason.UNAVAILABLE,
            (
                f"backend {unit.backend_id!r} is typed unavailable and "
                "cannot be admitted as verified"
            ),
        )

    remaining = verification_deadline - time.monotonic()
    if remaining <= 0:
        return _reject_unit(
            unit, UnitVerificationReason.TIMEOUT,
            "verification deadline expired before hash admission",
        )
    try:
        # Only pure, size-bounded SHA/HMAC work occupies a shared worker slot.
        # Cooperative hooks may themselves hash; run them outside admission
        # so they never try to upgrade a shared slot to an exclusive batch.
        with hashing_worker_slot(timeout=remaining):
            record = _verify_payload(
                unit,
                allowlisted_signers=allowlisted_signers,
                allowlisted_verification_keys=allowlisted_verification_keys,
            )
    except HashingResourceTimeout:
        return _reject_unit(
            unit, UnitVerificationReason.TIMEOUT,
            "verification deadline expired waiting for hash admission",
        )
    if not record.accepted:
        return record
    if cancellation is not None:
        try:
            cancellation.check()
        except ProcessControlError:
            return _reject_unit(
                unit,
                UnitVerificationReason.CANCELLED,
                "verification cancelled after unit check",
                digest=record.digest,
            )
    if time.monotonic() >= verification_deadline:
        return _reject_unit(
            unit, UnitVerificationReason.TIMEOUT,
            "verification deadline expired during unit check", digest=record.digest,
        )
    if hooks is not None and hooks.after_unit is not None:
        hooks.after_unit(unit, record)
    return record


def _verify_payload(
    unit: VerificationUnit,
    *,
    allowlisted_signers: frozenset[str],
    allowlisted_verification_keys: frozenset[str],
) -> UnitVerificationRecord:
    """Pure checks for one already bounded unit; no hooks or owner requests."""
    observed = digest_bytes(unit.payload)
    if not _digests_equal(observed, unit.expected_digest):
        return _reject_unit(
            unit,
            UnitVerificationReason.DIGEST_MISMATCH,
            (
                f"unit {unit.unit_id!r} payload digest does not match "
                "expected_digest"
            ),
            digest=observed,
        )

    if unit.kind is CheckKind.INTEGRITY:
        record = _accept_unit(unit, observed)
    elif unit.kind is CheckKind.RECEIPT:
        if unit.backend_id not in {"merkle_manifest", "receipt_aggregation", "integrity"}:
            return _reject_unit(
                unit,
                UnitVerificationReason.UNKNOWN_BACKEND,
                (
                    f"receipt checks require merkle_manifest, "
                    f"receipt_aggregation, or integrity, not {unit.backend_id!r}"
                ),
                digest=observed,
            )
        record = _accept_unit(unit, observed)
    elif unit.kind is CheckKind.SIGNATURE:
        if not unit.signer_id:
            return _reject_unit(
                unit,
                UnitVerificationReason.SIGNATURE_FAILURE,
                f"unit {unit.unit_id!r} signature check requires signer_id",
                digest=observed,
            )
        if unit.signer_id not in allowlisted_signers:
            return _reject_unit(
                unit,
                UnitVerificationReason.UNALLOWLISTED_SIGNER,
                f"signer {unit.signer_id!r} is not allowlisted",
                digest=observed,
            )
        expected_tag = hermetic_signature_tag(
            signer_id=unit.signer_id, payload=unit.payload
        )
        if not unit.proof_bytes or not _bytes_equal(unit.proof_bytes, expected_tag):
            return _reject_unit(
                unit,
                UnitVerificationReason.SIGNATURE_FAILURE,
                f"unit {unit.unit_id!r} signature does not match payload",
                digest=observed,
            )
        record = _accept_unit(unit, observed)
    elif unit.kind is CheckKind.PROOF:
        vk = unit.verification_key_id
        if vk and allowlisted_verification_keys and vk not in allowlisted_verification_keys:
            return _reject_unit(
                unit,
                UnitVerificationReason.UNALLOWLISTED_VERIFICATION_KEY,
                f"verification key {vk!r} is not allowlisted",
                digest=observed,
            )
        if unit.backend_id == "hermetic_hmac":
            if not vk:
                return _reject_unit(
                    unit,
                    UnitVerificationReason.PROOF_FAILURE,
                    "hermetic proof checks require verification_key_id",
                    digest=observed,
                )
            expected_tag = hermetic_proof_tag(
                unit_id=unit.unit_id,
                public_input=unit.payload,
                verification_key_id=vk,
            )
            if not unit.proof_bytes or not _bytes_equal(unit.proof_bytes, expected_tag):
                return _reject_unit(
                    unit,
                    UnitVerificationReason.PROOF_FAILURE,
                    f"unit {unit.unit_id!r} proof bytes fail hermetic HMAC check",
                    digest=observed,
                )
            record = _accept_unit(unit, observed)
        elif unit.backend_id == "integrity":
            expected_proof = hashlib.sha256(unit.payload).digest()
            if not unit.proof_bytes or not _bytes_equal(
                unit.proof_bytes, expected_proof
            ):
                return _reject_unit(
                    unit,
                    UnitVerificationReason.PROOF_FAILURE,
                    (
                        f"unit {unit.unit_id!r} integrity proof bytes must be "
                        "the SHA-256 digest of the committed payload"
                    ),
                    digest=observed,
                )
            record = _accept_unit(unit, observed)
        else:
            return _reject_unit(
                unit,
                UnitVerificationReason.UNKNOWN_BACKEND,
                f"proof checks do not admit backend {unit.backend_id!r}",
                digest=observed,
            )
    else:
        return _reject_unit(
            unit,
            UnitVerificationReason.UNKNOWN_KIND,
            f"unknown check kind {unit.kind!r}",
        )

    return record


def _work_item(unit: VerificationUnit, index: int) -> ProofWorkItem:
    return ProofWorkItem(
        work_id=f"w:{unit.unit_id}",
        unit_id=unit.unit_id,
        work_class=_WORK_CLASS_BY_KIND[unit.kind.value],
        cpu=unit.cpu,
        memory_mb=unit.memory_mb,
        gpu=0,
        publication_order=index,
        simulated_gpu=False,
    )


def _aggregate(
    units: Sequence[VerificationUnit],
    records: Sequence[UnitVerificationRecord],
    *,
    expected_unit_ids: Sequence[str] | None,
    fan_in: int,
) -> ManifestAggregationResult:
    verified = []
    for unit, record in zip(units, records, strict=True):
        verified.append(
            VerifiedUnit(
                unit_id=unit.unit_id,
                proof_object_cid=record.digest,
                category=unit.kind.value,
                terminal_status=(
                    "integrity_verified" if record.accepted else record.reason.value
                ),
                failed=not record.accepted,
            )
        )
    aggregator = ProofAggregator(fan_in=fan_in)
    result = aggregator.aggregate_verified_units(
        verified,
        expected_unit_ids=expected_unit_ids,
        prefer_recursion=False,
    )
    if not isinstance(result, ManifestAggregationResult):
        raise ParallelVerificationError(
            "parallel verification must not select recursive aggregation"
        )
    return result


def _batch_reason_from_units(
    records: Sequence[UnitVerificationRecord],
    aggregation: ManifestAggregationResult,
) -> BatchVerificationReason:
    if any(item.reason is UnitVerificationReason.CANCELLED for item in records):
        return BatchVerificationReason.CANCELLED
    if any(item.reason is UnitVerificationReason.BOUND_EXCEEDED for item in records):
        return BatchVerificationReason.BOUND_EXCEEDED
    if any(item.reason is UnitVerificationReason.DUPLICATE_UNIT for item in records):
        return BatchVerificationReason.DUPLICATE_CHILD
    if not aggregation.accepted:
        mapping = {
            "missing_child": BatchVerificationReason.MISSING_CHILD,
            "duplicate_child": BatchVerificationReason.DUPLICATE_CHILD,
            "reordered_children": BatchVerificationReason.REORDERED_CHILDREN,
            "failed_child": BatchVerificationReason.FAILED_CHILD,
        }
        return mapping.get(
            aggregation.reason.value, BatchVerificationReason.UNIT_REJECTED
        )
    if any(not item.accepted for item in records):
        return BatchVerificationReason.UNIT_REJECTED
    if not records:
        return BatchVerificationReason.EMPTY
    return BatchVerificationReason.AGGREGATED


def verify_units_in_parallel(
    units: Sequence[VerificationUnit],
    *,
    bounds: VerificationBounds | None = None,
    allowlisted_signers: Sequence[str] = (),
    allowlisted_verification_keys: Sequence[str] = (),
    expected_unit_ids: Sequence[str] | None = None,
    shuffle_seed: int | None = None,
    cancellation: CancellationToken | None = None,
    hooks: ParallelVerificationHooks | None = None,
) -> ParallelProofVerificationResult:
    """Verify independent proof/certificate units under closed bounds.

    Execution may complete in any order.  Aggregation always folds declared
    unit order, so the aggregate root is completion-order independent.
    """

    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ParallelVerificationError("units must be a sequence of VerificationUnit")
    parsed = tuple(units)
    for item in parsed:
        if not isinstance(item, VerificationUnit):
            raise ParallelVerificationError("units entries must be VerificationUnit")
    envelope = bounds or VerificationBounds()
    signers = frozenset(str(item) for item in allowlisted_signers if str(item).strip())
    keys = frozenset(
        str(item) for item in allowlisted_verification_keys if str(item).strip()
    )
    declared_ids = tuple(item.unit_id for item in parsed)
    expected = (
        tuple(expected_unit_ids) if expected_unit_ids is not None else declared_ids
    )

    seen: set[str] = set()
    duplicates = False
    for item in parsed:
        if item.unit_id in seen:
            duplicates = True
            break
        seen.add(item.unit_id)

    records_by_id: dict[str, UnitVerificationRecord] = {}
    completion_order: list[str] = []

    if duplicates:
        records = tuple(
            _reject_unit(
                item,
                UnitVerificationReason.DUPLICATE_UNIT,
                f"duplicate unit_id {item.unit_id!r}",
            )
            for item in parsed
        )
        aggregation = _aggregate(
            parsed, records, expected_unit_ids=expected, fan_in=envelope.fan_in
        )
        reason = BatchVerificationReason.DUPLICATE_CHILD
        return ParallelProofVerificationResult(
            schema=RESULT_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            interface=PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE,
            accepted=False,
            reason=reason,
            declared_unit_ids=declared_ids,
            units=records,
            aggregate_root=aggregation.child_root,
            aggregation_label=AGGREGATION_LABEL_MANIFEST,
            recursively_verifies_children=False,
            claims_test_execution=False,
            bounds=envelope,
            schedule_work_ids=(),
            bounded=True,
            completion_order=declared_ids,
        )

    work_items = tuple(_work_item(item, index) for index, item in enumerate(parsed))
    schedule = (
        build_proof_schedule(work_items, envelope.resource_policy())
        if work_items
        else ()
    )
    schedule_ids = tuple(slot.item.work_id for slot in schedule)

    submit_order = list(range(len(parsed)))
    if shuffle_seed is not None:
        Random(shuffle_seed).shuffle(submit_order)

    def _run(index: int) -> tuple[int, UnitVerificationRecord]:
        unit = parsed[index]
        if unit.cpu > envelope.max_cpu or unit.memory_mb > envelope.max_memory_mb:
            return index, _reject_unit(
                unit,
                UnitVerificationReason.UNAVAILABLE,
                "resource policy rejected unit admission",
            )
        return index, _verify_one(
            unit,
            bounds=envelope,
            allowlisted_signers=signers,
            allowlisted_verification_keys=keys,
            cancellation=cancellation,
            hooks=hooks,
            deadline=deadline,
        )

    deadline = time.monotonic() + envelope.timeout_seconds
    workers = hash_worker_limit(
        min(envelope.max_parallel, max(1, len(parsed)))
    )
    if parsed:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run, index): index for index in submit_order
            }
            try:
                for future in as_completed(
                    futures, timeout=max(0.0, deadline - time.monotonic())
                ):
                    index = futures[future]
                    try:
                        index, record = future.result()
                    except Exception as exc:  # noqa: BLE001
                        record = _reject_unit(
                            parsed[index],
                            UnitVerificationReason.MALFORMED,
                            (
                                f"unit {parsed[index].unit_id!r} raised "
                                f"{type(exc).__name__}"
                            ),
                        )
                    records_by_id[parsed[index].unit_id] = record
                    completion_order.append(parsed[index].unit_id)
            except FuturesTimeout:
                for index, unit in enumerate(parsed):
                    if unit.unit_id not in records_by_id:
                        records_by_id[unit.unit_id] = _reject_unit(
                            unit,
                            UnitVerificationReason.TIMEOUT,
                            (
                                f"unit {unit.unit_id!r} exceeded batch "
                                f"timeout_seconds {envelope.timeout_seconds}"
                            ),
                        )
                        completion_order.append(unit.unit_id)

    records = tuple(
        records_by_id[item.unit_id]
        if item.unit_id in records_by_id
        else _reject_unit(
            item,
            UnitVerificationReason.MALFORMED,
            f"unit {item.unit_id!r} produced no verification record",
        )
        for item in parsed
    )
    aggregation = _aggregate(
        parsed, records, expected_unit_ids=expected, fan_in=envelope.fan_in
    )
    reason = _batch_reason_from_units(records, aggregation)
    if not parsed:
        reason = BatchVerificationReason.EMPTY
        accepted = expected_unit_ids is None or len(expected) == 0
        if accepted:
            reason = BatchVerificationReason.AGGREGATED
    else:
        accepted = (
            aggregation.accepted
            and all(item.accepted for item in records)
            and reason is BatchVerificationReason.AGGREGATED
        )
        if not accepted and reason is BatchVerificationReason.AGGREGATED:
            reason = BatchVerificationReason.UNIT_REJECTED

    return ParallelProofVerificationResult(
        schema=RESULT_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        interface=PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE,
        accepted=accepted,
        reason=reason,
        declared_unit_ids=declared_ids,
        units=records,
        aggregate_root=aggregation.child_root,
        aggregation_label=AGGREGATION_LABEL_MANIFEST,
        recursively_verifies_children=False,
        claims_test_execution=False,
        bounds=envelope,
        schedule_work_ids=schedule_ids,
        bounded=True,
        completion_order=tuple(completion_order),
    )


def verification_claim() -> dict[str, Any]:
    """Public claim projection.  Never production admission."""

    return {
        "claim_class": CLAIM_CLASS,
        "establishes": VERIFICATION_ESTABLISHES,
        "does_not": VERIFICATION_DOES_NOT,
        "production_admitted": False,
        "self_approved": False,
        "claim_unchanged": True,
        "publication_authority_invoked": False,
        "may_authorize_skip": False,
        "interface": PARALLEL_PROOF_VERIFICATION_INTERFACE,
    }


__all__ = (
    "ADMITTED_BACKEND_IDS",
    "BOUNDS_SCHEMA",
    "CLAIM_CLASS",
    "CLOSED_CHECK_KINDS",
    "DEFAULT_MAX_PARALLEL",
    "DEFAULT_MAX_UNIT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "EVIDENCE_SUBSET",
    "MAX_PARALLEL_CAP",
    "PARALLEL_PROOF_VERIFICATION_BOUNDS_INTERFACE",
    "PARALLEL_PROOF_VERIFICATION_INTERFACE",
    "PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE",
    "PREDECESSOR_INTERFACES",
    "RESULT_SCHEMA",
    "UNAVAILABLE_BACKEND_IDS",
    "UNIT_SCHEMA",
    "VERIFICATION_DOES_NOT",
    "VERIFICATION_ESTABLISHES",
    "VERIFICATION_POLICY",
    "WORKER_BOUNDS",
    "assert_no_sensitive_material",
    "BatchVerificationReason",
    "CheckKind",
    "ParallelProofVerificationResult",
    "ParallelVerificationError",
    "ParallelVerificationHooks",
    "UnitVerificationReason",
    "UnitVerificationRecord",
    "VerificationBounds",
    "VerificationUnit",
    "digest_bytes",
    "hermetic_proof_tag",
    "hermetic_signature_tag",
    "integrity_proof_tag",
    "record_typed_unavailable",
    "typed_unavailable_records",
    "verification_claim",
    "verify_units_in_parallel",
)
