"""Bounded manifest aggregation and capability-gated recursion (IPS-036).

Builds fail-closed leaf → batch → category → repository aggregates over
already-verified units.  Child sets bind exact identities, count, order,
no-duplicates, terminal status, repository, and environment.

Recursive verification claims appear only when a backend capability probe
has admitted recursion **and** the backend actually verifies every child.
Otherwise the result is explicitly labeled ``manifest_aggregation`` and
states that it does not recursively verify child proofs or test execution.

Receipt aggregation claims state signer trust and never claim underlying
test execution.

Interfaces: ``ProofAggregator``, ``ManifestAggregationResult``,
``RecursiveAggregationResult``, ``aggregate_verified_units``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends import (
    AggregationDisposition,
    ProofBackendCapability,
)

MANIFEST_AGGREGATION_EVIDENCE: Final[str] = "ips/manifest-aggregation@1"
RECURSIVE_AGGREGATION_EVIDENCE: Final[str] = "ips/recursive-aggregation@1"

MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "manifest-aggregation-result@1"
)
RECURSIVE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "recursive-aggregation-result@1"
)
VERIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "aggregation-verification@1"
)
CHILD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "verified-child@1"
)
NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "aggregate-node@1"
)

DEFAULT_FAN_IN: Final[int] = 8
MAX_CHILDREN: Final[int] = 1 << 20

# Domain separators keep leaf / batch / category / repository digests distinct.
_DOMAIN_LEAF: Final[str] = "ips.aggregation.leaf.v1"
_DOMAIN_BATCH: Final[str] = "ips.aggregation.batch.v1"
_DOMAIN_CATEGORY: Final[str] = "ips.aggregation.category.v1"
_DOMAIN_REPOSITORY: Final[str] = "ips.aggregation.repository.v1"
_DOMAIN_MANIFEST: Final[str] = "ips.aggregation.manifest.v1"
_DOMAIN_RECURSIVE: Final[str] = "ips.aggregation.recursive.v1"

# Closed successful terminal statuses eligible for aggregation.
_SUCCESS_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "proved",
        "integrity_verified",
        "signed_assertion_verified",
    }
)

_FAILED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "failed",
        "proof_failed",
        "disproved",
        "invalid",
        "stale",
        "unknown",
        "timeout",
        "unavailable",
        "cancelled",
        "not_modeled",
        "simulated",
    }
)

# Explicit claim language for each aggregation surface.
MANIFEST_ESTABLISHES: Final[str] = (
    "integrity and completeness of the ordered verified-unit manifest; "
    "exact unit identities, count, order, no duplicates, roots, terminal "
    "status, repository, and environment binding"
)
MANIFEST_DOES_NOT_ESTABLISH: Final[str] = (
    "recursive verification of child proofs; underlying test execution; "
    "semantic correctness of repository contents"
)

RECURSIVE_ESTABLISHES: Final[str] = (
    "backend-verified child validity under recursive aggregation; exact unit "
    "identities, count, order, no duplicates, child root, terminal status, "
    "repository, environment, and policy under circuit/key assumptions"
)
RECURSIVE_DOES_NOT_ESTABLISH: Final[str] = (
    "claims beyond the recursive circuit/key assumptions and verified "
    "children; underlying test execution; repository correctness outside "
    "the recursive statement"
)

RECEIPT_AGGREGATION_ESTABLISHES: Final[str] = (
    "admitted committed receipt fields and exact required receipt "
    "set/count/order; signer trust when signatures are verified under the "
    "declared allowlist"
)
RECEIPT_AGGREGATION_DOES_NOT_ESTABLISH: Final[str] = (
    "underlying test execution; recursive verification of child proofs; "
    "independent observation that tests ran without trusting the signer"
)

_RECEIPT_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "SignedExecutionReceipt",
        "ReceiptAggregationZkProof",
        "signed_receipt",
        "receipt_aggregation",
    }
)

_FORBIDDEN_EXECUTION_CLAIM_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "tests executed",
        "tests ran",
        "test execution",
        "pytest executed",
        "pytest ran",
        "underlying tests ran",
        "tests_executed",
    }
)


class AggregationError(ValueError):
    """Fail-closed aggregation contract violation."""


class AggregationMode(str, Enum):
    MANIFEST_AGGREGATION = "manifest_aggregation"
    RECURSIVE_VERIFICATION = "recursive_verification"


class AggregationLevel(str, Enum):
    LEAF = "leaf"
    BATCH = "batch"
    CATEGORY = "category"
    REPOSITORY = "repository"


class AggregationOutcome(str, Enum):
    AGGREGATED = "aggregated"
    REJECTED = "rejected"


class AggregationReasonCode(str, Enum):
    COVERED = "covered"
    MISSING_CHILD = "missing_child"
    DUPLICATE_CHILD = "duplicate_child"
    REORDERED_CHILD = "reordered_child"
    FAILED_CHILD = "failed_child"
    CHANGED_MANIFEST = "changed_manifest"
    OLD_AGGREGATE = "old_aggregate"
    EMPTY_CHILDREN = "empty_children"
    CONTEXT_MISMATCH = "context_mismatch"
    RECURSION_NOT_ADMITTED = "recursion_not_admitted"
    CHILD_NOT_VERIFIED = "child_not_verified"
    RECURSIVE_VERIFY_FAILED = "recursive_verify_failed"
    OVERCLAIM = "overclaim"
    FAN_IN_INVALID = "fan_in_invalid"
    TOO_MANY_CHILDREN = "too_many_children"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_payload(domain: str, payload: Mapping[str, Any]) -> str:
    body = {"domain": domain, **dict(payload)}
    return _sha256_hex(_canonical_json(body))


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AggregationError(f"{field} must be a non-empty string")
    text = value.strip()
    if text != value:
        raise AggregationError(f"{field} must not have surrounding whitespace")
    return text


def _require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise AggregationError(f"{field} must be a boolean")
    return value


def _require_positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or isinstance(value, bool) or value < minimum:
        raise AggregationError(f"{field} must be an int >= {minimum}")
    return value


def _claims_overclaim_execution(establishes: str) -> bool:
    text = establishes.casefold()
    return any(token in text for token in _FORBIDDEN_EXECUTION_CLAIM_TOKENS)


def closed_aggregation_modes() -> frozenset[str]:
    return frozenset(item.value for item in AggregationMode)


def closed_aggregation_reason_codes() -> frozenset[str]:
    return frozenset(item.value for item in AggregationReasonCode)


def closed_aggregation_levels() -> frozenset[str]:
    return frozenset(item.value for item in AggregationLevel)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifiedChild:
    """One already-verified unit presented for aggregation.

    Callers must supply children in the exact expected order.  Order is never
    silently rewritten.
    """

    unit_id: str
    proof_object_cid: str
    category: str
    terminal_status: str
    repository_id: str
    environment_cid: str
    verification_digest: str
    evidence_class: str = "IntegrityCommitment"
    verified: bool = True
    signer_id: str | None = None
    policy_cid: str = "n/a"
    schema: str = CHILD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _require_text(self.unit_id, "unit_id"))
        object.__setattr__(
            self,
            "proof_object_cid",
            _require_text(self.proof_object_cid, "proof_object_cid"),
        )
        object.__setattr__(self, "category", _require_text(self.category, "category"))
        object.__setattr__(
            self,
            "terminal_status",
            _require_text(self.terminal_status, "terminal_status"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _require_text(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "environment_cid",
            _require_text(self.environment_cid, "environment_cid"),
        )
        object.__setattr__(
            self,
            "verification_digest",
            _require_text(self.verification_digest, "verification_digest"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            _require_text(self.evidence_class, "evidence_class"),
        )
        object.__setattr__(self, "verified", _require_bool(self.verified, "verified"))
        object.__setattr__(
            self, "policy_cid", _require_text(self.policy_cid, "policy_cid")
        )
        if self.signer_id is not None:
            object.__setattr__(
                self, "signer_id", _require_text(self.signer_id, "signer_id")
            )

    @property
    def is_success(self) -> bool:
        return self.terminal_status in _SUCCESS_STATUSES and self.verified is True

    @property
    def is_receipt_class(self) -> bool:
        return self.evidence_class in _RECEIPT_EVIDENCE_CLASSES

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "unit_id": self.unit_id,
            "proof_object_cid": self.proof_object_cid,
            "category": self.category,
            "terminal_status": self.terminal_status,
            "repository_id": self.repository_id,
            "environment_cid": self.environment_cid,
            "verification_digest": self.verification_digest,
            "evidence_class": self.evidence_class,
            "verified": self.verified,
            "signer_id": self.signer_id,
            "policy_cid": self.policy_cid,
        }

    def leaf_digest(self) -> str:
        return _digest_payload(_DOMAIN_LEAF, self.to_canonical())


@dataclass(frozen=True, slots=True)
class AggregationContext:
    """Repository-level binding shared by every child in one aggregation."""

    repository_id: str
    environment_cid: str
    policy_cid: str = "n/a"
    parent_aggregate_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_id",
            _require_text(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "environment_cid",
            _require_text(self.environment_cid, "environment_cid"),
        )
        object.__setattr__(
            self, "policy_cid", _require_text(self.policy_cid, "policy_cid")
        )
        if self.parent_aggregate_root is not None:
            object.__setattr__(
                self,
                "parent_aggregate_root",
                _require_text(self.parent_aggregate_root, "parent_aggregate_root"),
            )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "parent_aggregate_root": self.parent_aggregate_root,
        }


# ---------------------------------------------------------------------------
# Aggregate tree nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AggregateNode:
    """One node in the bounded fan-in aggregation tree."""

    level: AggregationLevel
    node_id: str
    child_ids: tuple[str, ...]
    child_digests: tuple[str, ...]
    digest: str
    category: str | None = None
    unit_ids: tuple[str, ...] = ()
    schema: str = NODE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level",
            AggregationLevel(str(getattr(self.level, "value", self.level))),
        )
        object.__setattr__(self, "node_id", _require_text(self.node_id, "node_id"))
        object.__setattr__(self, "digest", _require_text(self.digest, "digest"))
        if not isinstance(self.child_ids, tuple):
            object.__setattr__(self, "child_ids", tuple(self.child_ids))
        if not isinstance(self.child_digests, tuple):
            object.__setattr__(self, "child_digests", tuple(self.child_digests))
        if not isinstance(self.unit_ids, tuple):
            object.__setattr__(self, "unit_ids", tuple(self.unit_ids))
        if len(self.child_ids) != len(self.child_digests):
            raise AggregationError("child_ids and child_digests length mismatch")
        if len(set(self.child_ids)) != len(self.child_ids):
            raise AggregationError("duplicate child_ids in aggregate node")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "level": self.level.value,
            "node_id": self.node_id,
            "child_ids": list(self.child_ids),
            "child_digests": list(self.child_digests),
            "digest": self.digest,
            "category": self.category,
            "unit_ids": list(self.unit_ids),
        }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifestAggregationResult:
    """Merkle integrity/completeness aggregation (non-recursive)."""

    schema: str
    evidence_subset: str
    outcome: AggregationOutcome
    mode: AggregationMode
    aggregate_root: str | None
    manifest_root: str | None
    child_count: int
    child_unit_ids: tuple[str, ...]
    child_digests: tuple[str, ...]
    category_roots: Mapping[str, str]
    batch_nodes: tuple[AggregateNode, ...]
    category_nodes: tuple[AggregateNode, ...]
    repository_node: AggregateNode | None
    rebuilt_categories: tuple[str, ...]
    repository_id: str
    environment_cid: str
    policy_cid: str
    establishes: str
    does_not_establish: str
    recursive_verification: bool
    children_individually_verified: bool
    test_execution_directly_proven: bool
    receipt_aggregation: bool
    signer_trust_stated: bool
    reason_codes: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            AggregationOutcome(str(getattr(self.outcome, "value", self.outcome))),
        )
        object.__setattr__(
            self,
            "mode",
            AggregationMode(str(getattr(self.mode, "value", self.mode))),
        )
        if self.mode is not AggregationMode.MANIFEST_AGGREGATION:
            raise AggregationError(
                "ManifestAggregationResult.mode must be manifest_aggregation"
            )
        if self.recursive_verification is True:
            raise AggregationError(
                "manifest aggregation must not claim recursive_verification"
            )
        if self.test_execution_directly_proven is True:
            raise AggregationError(
                "manifest aggregation must not claim test execution"
            )
        if _claims_overclaim_execution(self.establishes):
            raise AggregationError(
                "manifest aggregation establishes must not claim test execution"
            )
        if "recursive verification" not in self.does_not_establish.casefold():
            raise AggregationError(
                "manifest aggregation must nonclaim recursive verification"
            )
        if self.receipt_aggregation and not self.signer_trust_stated:
            raise AggregationError(
                "receipt aggregation must state signer trust"
            )
        if self.receipt_aggregation:
            does = self.does_not_establish.casefold()
            if "test execution" not in does and "tests ran" not in does:
                raise AggregationError(
                    "receipt aggregation must nonclaim underlying test execution"
                )
        if self.outcome is AggregationOutcome.AGGREGATED:
            if not self.aggregate_root or not self.manifest_root:
                raise AggregationError(
                    "successful manifest aggregation requires roots"
                )
            if self.child_count != len(self.child_unit_ids):
                raise AggregationError("child_count must equal len(child_unit_ids)")
        if not isinstance(self.child_unit_ids, tuple):
            object.__setattr__(self, "child_unit_ids", tuple(self.child_unit_ids))
        if not isinstance(self.child_digests, tuple):
            object.__setattr__(self, "child_digests", tuple(self.child_digests))
        if not isinstance(self.batch_nodes, tuple):
            object.__setattr__(self, "batch_nodes", tuple(self.batch_nodes))
        if not isinstance(self.category_nodes, tuple):
            object.__setattr__(self, "category_nodes", tuple(self.category_nodes))
        if not isinstance(self.rebuilt_categories, tuple):
            object.__setattr__(
                self, "rebuilt_categories", tuple(self.rebuilt_categories)
            )
        if not isinstance(self.reason_codes, tuple):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "category_roots", dict(self.category_roots))

    @property
    def succeeded(self) -> bool:
        return self.outcome is AggregationOutcome.AGGREGATED

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "outcome": self.outcome.value,
            "mode": self.mode.value,
            "aggregate_root": self.aggregate_root,
            "manifest_root": self.manifest_root,
            "child_count": self.child_count,
            "child_unit_ids": list(self.child_unit_ids),
            "child_digests": list(self.child_digests),
            "category_roots": dict(self.category_roots),
            "batch_nodes": [node.to_canonical() for node in self.batch_nodes],
            "category_nodes": [node.to_canonical() for node in self.category_nodes],
            "repository_node": (
                self.repository_node.to_canonical()
                if self.repository_node is not None
                else None
            ),
            "rebuilt_categories": list(self.rebuilt_categories),
            "repository_id": self.repository_id,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "establishes": self.establishes,
            "does_not_establish": self.does_not_establish,
            "recursive_verification": self.recursive_verification,
            "children_individually_verified": self.children_individually_verified,
            "test_execution_directly_proven": self.test_execution_directly_proven,
            "receipt_aggregation": self.receipt_aggregation,
            "signer_trust_stated": self.signer_trust_stated,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
        }

    def result_cid(self) -> str:
        return _sha256_hex(_canonical_json(self.to_canonical()))


@dataclass(frozen=True, slots=True)
class RecursiveAggregationResult:
    """Recursive aggregation admitted only after capability-gated child verify."""

    schema: str
    evidence_subset: str
    outcome: AggregationOutcome
    mode: AggregationMode
    aggregate_root: str | None
    child_root: str | None
    child_count: int
    child_unit_ids: tuple[str, ...]
    child_digests: tuple[str, ...]
    repository_id: str
    environment_cid: str
    policy_cid: str
    backend_id: str
    establishes: str
    does_not_establish: str
    recursive_verification: bool
    children_backend_verified: bool
    test_execution_directly_proven: bool
    reason_codes: tuple[str, ...]
    message: str
    recursive_proof_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            AggregationOutcome(str(getattr(self.outcome, "value", self.outcome))),
        )
        object.__setattr__(
            self,
            "mode",
            AggregationMode(str(getattr(self.mode, "value", self.mode))),
        )
        if self.mode is not AggregationMode.RECURSIVE_VERIFICATION:
            raise AggregationError(
                "RecursiveAggregationResult.mode must be recursive_verification"
            )
        if self.outcome is AggregationOutcome.AGGREGATED:
            if self.recursive_verification is not True:
                raise AggregationError(
                    "successful recursive aggregation requires recursive_verification=True"
                )
            if self.children_backend_verified is not True:
                raise AggregationError(
                    "successful recursive aggregation requires children_backend_verified=True"
                )
            if not self.aggregate_root or not self.child_root:
                raise AggregationError(
                    "successful recursive aggregation requires roots"
                )
            if not self.recursive_proof_digest:
                raise AggregationError(
                    "successful recursive aggregation requires recursive_proof_digest"
                )
        if self.test_execution_directly_proven is True:
            raise AggregationError(
                "recursive aggregation must not claim test execution by default"
            )
        if _claims_overclaim_execution(self.establishes):
            raise AggregationError(
                "recursive aggregation establishes must not claim test execution"
            )
        if not isinstance(self.child_unit_ids, tuple):
            object.__setattr__(self, "child_unit_ids", tuple(self.child_unit_ids))
        if not isinstance(self.child_digests, tuple):
            object.__setattr__(self, "child_digests", tuple(self.child_digests))
        if not isinstance(self.reason_codes, tuple):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    @property
    def succeeded(self) -> bool:
        return self.outcome is AggregationOutcome.AGGREGATED

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "outcome": self.outcome.value,
            "mode": self.mode.value,
            "aggregate_root": self.aggregate_root,
            "child_root": self.child_root,
            "child_count": self.child_count,
            "child_unit_ids": list(self.child_unit_ids),
            "child_digests": list(self.child_digests),
            "repository_id": self.repository_id,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "backend_id": self.backend_id,
            "establishes": self.establishes,
            "does_not_establish": self.does_not_establish,
            "recursive_verification": self.recursive_verification,
            "children_backend_verified": self.children_backend_verified,
            "test_execution_directly_proven": self.test_execution_directly_proven,
            "recursive_proof_digest": self.recursive_proof_digest,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
        }

    def result_cid(self) -> str:
        return _sha256_hex(_canonical_json(self.to_canonical()))


AggregationResult = ManifestAggregationResult | RecursiveAggregationResult


@dataclass(frozen=True, slots=True)
class AggregationVerificationResult:
    """Outcome of re-checking an aggregate against a presented child set."""

    schema: str
    accepted: bool
    reason_code: AggregationReasonCode
    message: str
    expected_aggregate_root: str | None = None
    observed_aggregate_root: str | None = None
    expected_manifest_root: str | None = None
    observed_manifest_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reason_code",
            AggregationReasonCode(
                str(getattr(self.reason_code, "value", self.reason_code))
            ),
        )
        if type(self.accepted) is not bool:
            raise AggregationError("accepted must be a boolean")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "reason_code": self.reason_code.value,
            "message": self.message,
            "expected_aggregate_root": self.expected_aggregate_root,
            "observed_aggregate_root": self.observed_aggregate_root,
            "expected_manifest_root": self.expected_manifest_root,
            "observed_manifest_root": self.observed_manifest_root,
        }


# ---------------------------------------------------------------------------
# Optional recursive aggregation backend
# ---------------------------------------------------------------------------


@runtime_checkable
class RecursiveAggregationBackend(Protocol):
    """Backend that verifies children and proves a recursive aggregate."""

    backend_id: str

    def verify_child(self, child: VerifiedChild) -> bool:
        """Return True only when the backend actually verifies the child."""

    def prove_recursive(
        self,
        children: Sequence[VerifiedChild],
        *,
        child_root: str,
        context: AggregationContext,
    ) -> bytes:
        """Prove recursive aggregate binding verified children."""

    def verify_recursive(
        self,
        proof: bytes,
        children: Sequence[VerifiedChild],
        *,
        child_root: str,
        context: AggregationContext,
    ) -> bool:
        """Verify the recursive aggregate proof."""


@dataclass
class HermeticRecursiveAggregationBackend:
    """In-process recursive aggregation backend for hermetic tests.

    Uses HMAC digests over child digests; never production keys.
    """

    backend_id: str = "hermetic-test-only-recursive-aggregation"
    fail_child_ids: frozenset[str] = field(default_factory=frozenset)
    verify_children: bool = True

    def verify_child(self, child: VerifiedChild) -> bool:
        if not self.verify_children:
            return False
        if child.unit_id in self.fail_child_ids:
            return False
        return child.is_success

    def prove_recursive(
        self,
        children: Sequence[VerifiedChild],
        *,
        child_root: str,
        context: AggregationContext,
    ) -> bytes:
        for child in children:
            if not self.verify_child(child):
                raise AggregationError(
                    f"cannot recurse over unverified child {child.unit_id!r}"
                )
        key = hashlib.sha256(
            b"ips-aggregation-recursive-test-only-key\n" + self.backend_id.encode()
        ).digest()
        material = (
            child_root.encode()
            + context.repository_id.encode()
            + context.environment_cid.encode()
            + context.policy_cid.encode()
            + b"|".join(child.leaf_digest().encode() for child in children)
        )
        return hmac.new(key, material, hashlib.sha256).digest()

    def verify_recursive(
        self,
        proof: bytes,
        children: Sequence[VerifiedChild],
        *,
        child_root: str,
        context: AggregationContext,
    ) -> bool:
        try:
            expected = self.prove_recursive(
                children, child_root=child_root, context=context
            )
        except AggregationError:
            return False
        return hmac.compare_digest(proof, expected)


# ---------------------------------------------------------------------------
# Child-set validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ChildValidation:
    children: tuple[VerifiedChild, ...]
    unit_ids: tuple[str, ...]
    digests: tuple[str, ...]
    reason: AggregationReasonCode | None = None
    message: str = ""


def _validate_child_set(
    children: Sequence[VerifiedChild],
    *,
    context: AggregationContext,
    expected_unit_ids: Sequence[str] | None = None,
) -> _ChildValidation:
    if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
        raise AggregationError("children must be a sequence of VerifiedChild")
    if len(children) > MAX_CHILDREN:
        return _ChildValidation(
            (),
            (),
            (),
            AggregationReasonCode.TOO_MANY_CHILDREN,
            f"child count {len(children)} exceeds bound {MAX_CHILDREN}",
        )
    if len(children) == 0:
        return _ChildValidation(
            (),
            (),
            (),
            AggregationReasonCode.EMPTY_CHILDREN,
            "aggregation requires at least one verified child",
        )

    coerced: list[VerifiedChild] = []
    for item in children:
        if not isinstance(item, VerifiedChild):
            raise AggregationError("each child must be VerifiedChild")
        coerced.append(item)

    unit_ids = [child.unit_id for child in coerced]
    if len(set(unit_ids)) != len(unit_ids):
        return _ChildValidation(
            (),
            (),
            (),
            AggregationReasonCode.DUPLICATE_CHILD,
            "duplicate child unit_id rejected",
        )

    if expected_unit_ids is not None:
        expected = tuple(_require_text(item, "expected_unit_id") for item in expected_unit_ids)
        if len(set(expected)) != len(expected):
            return _ChildValidation(
                (),
                (),
                (),
                AggregationReasonCode.DUPLICATE_CHILD,
                "expected unit set contains duplicates",
            )
        observed = tuple(unit_ids)
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            if missing:
                return _ChildValidation(
                    (),
                    (),
                    (),
                    AggregationReasonCode.MISSING_CHILD,
                    f"missing required children: {missing}",
                )
            return _ChildValidation(
                (),
                (),
                (),
                AggregationReasonCode.CHANGED_MANIFEST,
                f"unexpected children present: {extra}",
            )
        if observed != expected:
            return _ChildValidation(
                (),
                (),
                (),
                AggregationReasonCode.REORDERED_CHILD,
                "child order does not match expected manifest order",
            )
    # Without an expected list, still reject non-canonical order (never rewrite).
    elif unit_ids != sorted(unit_ids):
        return _ChildValidation(
            (),
            (),
            (),
            AggregationReasonCode.REORDERED_CHILD,
            "children must be provided in canonical unit_id order",
        )

    for child in coerced:
        if child.repository_id != context.repository_id:
            return _ChildValidation(
                (),
                (),
                (),
                AggregationReasonCode.CONTEXT_MISMATCH,
                (
                    f"child {child.unit_id!r} repository_id "
                    f"{child.repository_id!r} != context {context.repository_id!r}"
                ),
            )
        if child.environment_cid != context.environment_cid:
            return _ChildValidation(
                (),
                (),
                (),
                AggregationReasonCode.CONTEXT_MISMATCH,
                (
                    f"child {child.unit_id!r} environment_cid "
                    f"{child.environment_cid!r} != context {context.environment_cid!r}"
                ),
            )
        if (
            child.terminal_status in _FAILED_STATUSES
            or child.terminal_status not in _SUCCESS_STATUSES
            or child.verified is not True
        ):
            return _ChildValidation(
                (),
                (),
                (),
                AggregationReasonCode.FAILED_CHILD,
                (
                    f"child {child.unit_id!r} has non-success status "
                    f"{child.terminal_status!r} or verified={child.verified}"
                ),
            )

    digests = tuple(child.leaf_digest() for child in coerced)
    return _ChildValidation(tuple(coerced), tuple(unit_ids), digests)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def _chunk(items: Sequence[Any], size: int) -> list[tuple[Any, ...]]:
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def _build_batch_nodes(
    children: Sequence[VerifiedChild],
    *,
    fan_in: int,
    categories: Sequence[str] | None = None,
) -> tuple[list[AggregateNode], dict[str, list[AggregateNode]]]:
    by_category: dict[str, list[VerifiedChild]] = {}
    for child in children:
        by_category.setdefault(child.category, []).append(child)

    if categories is None:
        category_order = sorted(by_category)
    else:
        category_order = list(categories)

    all_batches: list[AggregateNode] = []
    batches_by_category: dict[str, list[AggregateNode]] = {}
    for category in category_order:
        group = by_category.get(category, ())
        if not group:
            continue
        category_batches: list[AggregateNode] = []
        for index, batch in enumerate(_chunk(group, fan_in)):
            child_ids = tuple(item.unit_id for item in batch)
            child_digests = tuple(item.leaf_digest() for item in batch)
            node_id = f"batch:{category}:{index}"
            digest = _digest_payload(
                _DOMAIN_BATCH,
                {
                    "node_id": node_id,
                    "category": category,
                    "index": index,
                    "child_ids": list(child_ids),
                    "child_digests": list(child_digests),
                    "fan_in": fan_in,
                },
            )
            node = AggregateNode(
                level=AggregationLevel.BATCH,
                node_id=node_id,
                child_ids=child_ids,
                child_digests=child_digests,
                digest=digest,
                category=category,
                unit_ids=child_ids,
            )
            category_batches.append(node)
            all_batches.append(node)
        batches_by_category[category] = category_batches
    return all_batches, batches_by_category


def _build_category_nodes(
    batches_by_category: Mapping[str, Sequence[AggregateNode]],
) -> list[AggregateNode]:
    nodes: list[AggregateNode] = []
    for category in sorted(batches_by_category):
        batches = list(batches_by_category[category])
        child_ids = tuple(node.node_id for node in batches)
        child_digests = tuple(node.digest for node in batches)
        unit_ids: list[str] = []
        for node in batches:
            unit_ids.extend(node.unit_ids)
        node_id = f"category:{category}"
        digest = _digest_payload(
            _DOMAIN_CATEGORY,
            {
                "node_id": node_id,
                "category": category,
                "child_ids": list(child_ids),
                "child_digests": list(child_digests),
                "unit_ids": unit_ids,
                "leaf_count": len(unit_ids),
            },
        )
        nodes.append(
            AggregateNode(
                level=AggregationLevel.CATEGORY,
                node_id=node_id,
                child_ids=child_ids,
                child_digests=child_digests,
                digest=digest,
                category=category,
                unit_ids=tuple(unit_ids),
            )
        )
    return nodes


def _build_repository_node(
    category_nodes: Sequence[AggregateNode],
    *,
    context: AggregationContext,
    child_unit_ids: Sequence[str],
    child_digests: Sequence[str],
) -> AggregateNode:
    child_ids = tuple(node.node_id for node in category_nodes)
    digests = tuple(node.digest for node in category_nodes)
    node_id = "repository"
    digest = _digest_payload(
        _DOMAIN_REPOSITORY,
        {
            "node_id": node_id,
            "repository_id": context.repository_id,
            "environment_cid": context.environment_cid,
            "policy_cid": context.policy_cid,
            "parent_aggregate_root": context.parent_aggregate_root,
            "child_ids": list(child_ids),
            "child_digests": list(digests),
            "unit_ids": list(child_unit_ids),
            "leaf_digests": list(child_digests),
            "child_count": len(child_unit_ids),
        },
    )
    return AggregateNode(
        level=AggregationLevel.REPOSITORY,
        node_id=node_id,
        child_ids=child_ids,
        child_digests=digests,
        digest=digest,
        category=None,
        unit_ids=tuple(child_unit_ids),
    )


def _manifest_root(
    *,
    child_unit_ids: Sequence[str],
    child_digests: Sequence[str],
    category_roots: Mapping[str, str],
    context: AggregationContext,
) -> str:
    return _digest_payload(
        _DOMAIN_MANIFEST,
        {
            "child_unit_ids": list(child_unit_ids),
            "child_digests": list(child_digests),
            "category_roots": dict(sorted(category_roots.items())),
            "repository_id": context.repository_id,
            "environment_cid": context.environment_cid,
            "policy_cid": context.policy_cid,
            "child_count": len(child_unit_ids),
        },
    )


def _child_root(child_digests: Sequence[str]) -> str:
    return _digest_payload(
        _DOMAIN_LEAF,
        {"ordered_child_digests": list(child_digests)},
    )


# ---------------------------------------------------------------------------
# Claims selection
# ---------------------------------------------------------------------------


def _select_claims(
    children: Sequence[VerifiedChild],
) -> tuple[str, str, bool, bool]:
    """Return establishes, does_not_establish, receipt_aggregation, signer_trust."""

    receipt = any(child.is_receipt_class for child in children)
    if receipt:
        return (
            RECEIPT_AGGREGATION_ESTABLISHES,
            RECEIPT_AGGREGATION_DOES_NOT_ESTABLISH,
            True,
            True,
        )
    return (
        MANIFEST_ESTABLISHES,
        MANIFEST_DOES_NOT_ESTABLISH,
        False,
        False,
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class ProofAggregator:
    """Bounded fan-in aggregator with capability-gated recursion selection."""

    def __init__(
        self,
        *,
        fan_in: int = DEFAULT_FAN_IN,
        capability: ProofBackendCapability | None = None,
        recursive_backend: RecursiveAggregationBackend | None = None,
        prefer_recursive: bool = True,
    ) -> None:
        self._fan_in = _require_positive_int(fan_in, "fan_in", minimum=2)
        self._capability = capability
        self._recursive_backend = recursive_backend
        self._prefer_recursive = prefer_recursive

    @property
    def fan_in(self) -> int:
        return self._fan_in

    @property
    def recursion_admitted(self) -> bool:
        if self._capability is None:
            return False
        return (
            self._capability.recursive_verification is True
            and self._capability.aggregation_disposition
            is AggregationDisposition.RECURSIVE_VERIFICATION
            and self._capability.recursion_probe.passed is True
        )

    def aggregate(
        self,
        children: Sequence[VerifiedChild],
        context: AggregationContext,
        *,
        expected_unit_ids: Sequence[str] | None = None,
        affected_unit_ids: Sequence[str] | None = None,
        force_mode: AggregationMode | None = None,
    ) -> AggregationResult:
        """Aggregate verified children under the admitted mode."""

        validation = _validate_child_set(
            children, context=context, expected_unit_ids=expected_unit_ids
        )
        if validation.reason is not None:
            return self._reject_manifest(
                context,
                reason=validation.reason,
                message=validation.message,
            )

        mode = self._select_mode(force_mode)
        if mode is AggregationMode.RECURSIVE_VERIFICATION:
            return self._aggregate_recursive(validation, context)
        return self._aggregate_manifest(
            validation,
            context,
            affected_unit_ids=affected_unit_ids,
        )

    def verify_aggregate(
        self,
        previous: AggregationResult,
        children: Sequence[VerifiedChild],
        context: AggregationContext,
        *,
        expected_unit_ids: Sequence[str] | None = None,
    ) -> AggregationVerificationResult:
        """Re-check an aggregate against the presented child set.

        Rejects missing, duplicate, reordered, or failed children; a changed
        manifest root; or an old aggregate root that no longer matches.
        """

        if previous.outcome is not AggregationOutcome.AGGREGATED:
            return AggregationVerificationResult(
                schema=VERIFICATION_SCHEMA,
                accepted=False,
                reason_code=AggregationReasonCode.OLD_AGGREGATE,
                message="previous aggregation did not succeed",
                expected_aggregate_root=previous.aggregate_root,
            )

        expected_ids = expected_unit_ids
        if expected_ids is None:
            expected_ids = previous.child_unit_ids

        validation = _validate_child_set(
            children, context=context, expected_unit_ids=expected_ids
        )
        if validation.reason is not None:
            return AggregationVerificationResult(
                schema=VERIFICATION_SCHEMA,
                accepted=False,
                reason_code=validation.reason,
                message=validation.message,
                expected_aggregate_root=previous.aggregate_root,
                expected_manifest_root=getattr(previous, "manifest_root", None),
            )

        # Context drift vs previous result.
        if (
            previous.repository_id != context.repository_id
            or previous.environment_cid != context.environment_cid
        ):
            return AggregationVerificationResult(
                schema=VERIFICATION_SCHEMA,
                accepted=False,
                reason_code=AggregationReasonCode.CONTEXT_MISMATCH,
                message="aggregation context does not match previous result",
                expected_aggregate_root=previous.aggregate_root,
            )

        recomputed = self.aggregate(
            validation.children,
            context,
            expected_unit_ids=expected_ids,
            force_mode=previous.mode,
        )
        if not recomputed.succeeded:
            reason = AggregationReasonCode.CHANGED_MANIFEST
            if recomputed.reason_codes:
                try:
                    reason = AggregationReasonCode(recomputed.reason_codes[0])
                except ValueError:
                    reason = AggregationReasonCode.CHANGED_MANIFEST
            return AggregationVerificationResult(
                schema=VERIFICATION_SCHEMA,
                accepted=False,
                reason_code=reason,
                message=recomputed.message,
                expected_aggregate_root=previous.aggregate_root,
                observed_aggregate_root=recomputed.aggregate_root,
                expected_manifest_root=getattr(previous, "manifest_root", None),
                observed_manifest_root=getattr(recomputed, "manifest_root", None),
            )

        # Manifest root check (manifest path).
        prev_manifest = getattr(previous, "manifest_root", None)
        new_manifest = getattr(recomputed, "manifest_root", None)
        if prev_manifest is not None and new_manifest is not None:
            if not hmac.compare_digest(str(prev_manifest), str(new_manifest)):
                return AggregationVerificationResult(
                    schema=VERIFICATION_SCHEMA,
                    accepted=False,
                    reason_code=AggregationReasonCode.CHANGED_MANIFEST,
                    message="manifest root changed relative to stored aggregate",
                    expected_aggregate_root=previous.aggregate_root,
                    observed_aggregate_root=recomputed.aggregate_root,
                    expected_manifest_root=prev_manifest,
                    observed_manifest_root=new_manifest,
                )

        if previous.aggregate_root is None or recomputed.aggregate_root is None:
            return AggregationVerificationResult(
                schema=VERIFICATION_SCHEMA,
                accepted=False,
                reason_code=AggregationReasonCode.OLD_AGGREGATE,
                message="aggregate root missing",
                expected_aggregate_root=previous.aggregate_root,
                observed_aggregate_root=recomputed.aggregate_root,
            )

        if not hmac.compare_digest(
            str(previous.aggregate_root), str(recomputed.aggregate_root)
        ):
            return AggregationVerificationResult(
                schema=VERIFICATION_SCHEMA,
                accepted=False,
                reason_code=AggregationReasonCode.OLD_AGGREGATE,
                message="stored aggregate root does not match recomputed root",
                expected_aggregate_root=previous.aggregate_root,
                observed_aggregate_root=recomputed.aggregate_root,
                expected_manifest_root=prev_manifest,
                observed_manifest_root=new_manifest,
            )

        # For recursive results, child_root must also match.
        prev_child_root = getattr(previous, "child_root", None)
        new_child_root = getattr(recomputed, "child_root", None)
        if prev_child_root is not None and new_child_root is not None:
            if not hmac.compare_digest(str(prev_child_root), str(new_child_root)):
                return AggregationVerificationResult(
                    schema=VERIFICATION_SCHEMA,
                    accepted=False,
                    reason_code=AggregationReasonCode.OLD_AGGREGATE,
                    message="stored recursive child_root does not match recomputed root",
                    expected_aggregate_root=previous.aggregate_root,
                    observed_aggregate_root=recomputed.aggregate_root,
                )

        return AggregationVerificationResult(
            schema=VERIFICATION_SCHEMA,
            accepted=True,
            reason_code=AggregationReasonCode.COVERED,
            message="aggregate matches presented verified children",
            expected_aggregate_root=previous.aggregate_root,
            observed_aggregate_root=recomputed.aggregate_root,
            expected_manifest_root=prev_manifest,
            observed_manifest_root=new_manifest,
        )

    def _select_mode(
        self, force_mode: AggregationMode | None
    ) -> AggregationMode:
        if force_mode is not None:
            mode = AggregationMode(str(getattr(force_mode, "value", force_mode)))
            if mode is AggregationMode.RECURSIVE_VERIFICATION:
                if not self.recursion_admitted or self._recursive_backend is None:
                    raise AggregationError(
                        "recursive mode forced but recursion is not admitted "
                        "or recursive backend is absent"
                    )
            return mode
        if (
            self._prefer_recursive
            and self.recursion_admitted
            and self._recursive_backend is not None
        ):
            return AggregationMode.RECURSIVE_VERIFICATION
        return AggregationMode.MANIFEST_AGGREGATION

    def _aggregate_manifest(
        self,
        validation: _ChildValidation,
        context: AggregationContext,
        *,
        affected_unit_ids: Sequence[str] | None,
    ) -> ManifestAggregationResult:
        children = validation.children
        establishes, does_not, receipt, signer_trust = _select_claims(children)

        batches, batches_by_category = _build_batch_nodes(
            children, fan_in=self._fan_in
        )
        if affected_unit_ids is not None:
            affected = set(affected_unit_ids)
            rebuilt_set = {
                child.category for child in children if child.unit_id in affected
            }
            # Empty affected set still records every category as rebuilt so the
            # repository binding remains complete and explicit.
            if not rebuilt_set:
                rebuilt_set = set(batches_by_category)
            rebuilt = tuple(sorted(rebuilt_set))
        else:
            rebuilt = tuple(sorted(batches_by_category))

        category_nodes = _build_category_nodes(batches_by_category)
        category_roots = {
            node.category: node.digest
            for node in category_nodes
            if node.category is not None
        }
        repository_node = _build_repository_node(
            category_nodes,
            context=context,
            child_unit_ids=validation.unit_ids,
            child_digests=validation.digests,
        )
        manifest = _manifest_root(
            child_unit_ids=validation.unit_ids,
            child_digests=validation.digests,
            category_roots=category_roots,
            context=context,
        )
        return ManifestAggregationResult(
            schema=MANIFEST_SCHEMA,
            evidence_subset=MANIFEST_AGGREGATION_EVIDENCE,
            outcome=AggregationOutcome.AGGREGATED,
            mode=AggregationMode.MANIFEST_AGGREGATION,
            aggregate_root=repository_node.digest,
            manifest_root=manifest,
            child_count=len(validation.unit_ids),
            child_unit_ids=validation.unit_ids,
            child_digests=validation.digests,
            category_roots=category_roots,
            batch_nodes=tuple(batches),
            category_nodes=tuple(category_nodes),
            repository_node=repository_node,
            rebuilt_categories=rebuilt,
            repository_id=context.repository_id,
            environment_cid=context.environment_cid,
            policy_cid=context.policy_cid,
            establishes=establishes,
            does_not_establish=does_not,
            recursive_verification=False,
            children_individually_verified=True,
            test_execution_directly_proven=False,
            receipt_aggregation=receipt,
            signer_trust_stated=signer_trust,
            reason_codes=(AggregationReasonCode.COVERED.value,),
            message=(
                "manifest_aggregation: integrity/completeness over individually "
                "verified leaves; not recursive verification of child proofs"
            ),
        )

    def _aggregate_recursive(
        self,
        validation: _ChildValidation,
        context: AggregationContext,
    ) -> RecursiveAggregationResult:
        backend = self._recursive_backend
        if backend is None or not self.recursion_admitted:
            return RecursiveAggregationResult(
                schema=RECURSIVE_SCHEMA,
                evidence_subset=RECURSIVE_AGGREGATION_EVIDENCE,
                outcome=AggregationOutcome.REJECTED,
                mode=AggregationMode.RECURSIVE_VERIFICATION,
                aggregate_root=None,
                child_root=None,
                child_count=len(validation.unit_ids),
                child_unit_ids=validation.unit_ids,
                child_digests=validation.digests,
                repository_id=context.repository_id,
                environment_cid=context.environment_cid,
                policy_cid=context.policy_cid,
                backend_id="absent",
                establishes=RECURSIVE_ESTABLISHES,
                does_not_establish=RECURSIVE_DOES_NOT_ESTABLISH,
                recursive_verification=False,
                children_backend_verified=False,
                test_execution_directly_proven=False,
                reason_codes=(AggregationReasonCode.RECURSION_NOT_ADMITTED.value,),
                message="recursive aggregation not admitted by capability probe",
            )

        # Backend must actually verify every child; claims require this.
        for child in validation.children:
            if backend.verify_child(child) is not True:
                return RecursiveAggregationResult(
                    schema=RECURSIVE_SCHEMA,
                    evidence_subset=RECURSIVE_AGGREGATION_EVIDENCE,
                    outcome=AggregationOutcome.REJECTED,
                    mode=AggregationMode.RECURSIVE_VERIFICATION,
                    aggregate_root=None,
                    child_root=None,
                    child_count=len(validation.unit_ids),
                    child_unit_ids=validation.unit_ids,
                    child_digests=validation.digests,
                    repository_id=context.repository_id,
                    environment_cid=context.environment_cid,
                    policy_cid=context.policy_cid,
                    backend_id=getattr(backend, "backend_id", "unknown"),
                    establishes=RECURSIVE_ESTABLISHES,
                    does_not_establish=RECURSIVE_DOES_NOT_ESTABLISH,
                    recursive_verification=False,
                    children_backend_verified=False,
                    test_execution_directly_proven=False,
                    reason_codes=(AggregationReasonCode.CHILD_NOT_VERIFIED.value,),
                    message=(
                        f"backend did not verify child {child.unit_id!r}; "
                        "recursive claims are forbidden without child verification"
                    ),
                )

        child_root = _child_root(validation.digests)
        try:
            proof = backend.prove_recursive(
                validation.children,
                child_root=child_root,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed on backend errors
            return RecursiveAggregationResult(
                schema=RECURSIVE_SCHEMA,
                evidence_subset=RECURSIVE_AGGREGATION_EVIDENCE,
                outcome=AggregationOutcome.REJECTED,
                mode=AggregationMode.RECURSIVE_VERIFICATION,
                aggregate_root=None,
                child_root=child_root,
                child_count=len(validation.unit_ids),
                child_unit_ids=validation.unit_ids,
                child_digests=validation.digests,
                repository_id=context.repository_id,
                environment_cid=context.environment_cid,
                policy_cid=context.policy_cid,
                backend_id=getattr(backend, "backend_id", "unknown"),
                establishes=RECURSIVE_ESTABLISHES,
                does_not_establish=RECURSIVE_DOES_NOT_ESTABLISH,
                recursive_verification=False,
                children_backend_verified=True,
                test_execution_directly_proven=False,
                reason_codes=(AggregationReasonCode.RECURSIVE_VERIFY_FAILED.value,),
                message=f"recursive prove failed: {exc}",
            )

        if not isinstance(proof, (bytes, bytearray)) or not proof:
            return RecursiveAggregationResult(
                schema=RECURSIVE_SCHEMA,
                evidence_subset=RECURSIVE_AGGREGATION_EVIDENCE,
                outcome=AggregationOutcome.REJECTED,
                mode=AggregationMode.RECURSIVE_VERIFICATION,
                aggregate_root=None,
                child_root=child_root,
                child_count=len(validation.unit_ids),
                child_unit_ids=validation.unit_ids,
                child_digests=validation.digests,
                repository_id=context.repository_id,
                environment_cid=context.environment_cid,
                policy_cid=context.policy_cid,
                backend_id=getattr(backend, "backend_id", "unknown"),
                establishes=RECURSIVE_ESTABLISHES,
                does_not_establish=RECURSIVE_DOES_NOT_ESTABLISH,
                recursive_verification=False,
                children_backend_verified=True,
                test_execution_directly_proven=False,
                reason_codes=(AggregationReasonCode.RECURSIVE_VERIFY_FAILED.value,),
                message="recursive proof bytes missing",
            )

        proof_bytes = bytes(proof)
        if backend.verify_recursive(
            proof_bytes,
            validation.children,
            child_root=child_root,
            context=context,
        ) is not True:
            return RecursiveAggregationResult(
                schema=RECURSIVE_SCHEMA,
                evidence_subset=RECURSIVE_AGGREGATION_EVIDENCE,
                outcome=AggregationOutcome.REJECTED,
                mode=AggregationMode.RECURSIVE_VERIFICATION,
                aggregate_root=None,
                child_root=child_root,
                child_count=len(validation.unit_ids),
                child_unit_ids=validation.unit_ids,
                child_digests=validation.digests,
                repository_id=context.repository_id,
                environment_cid=context.environment_cid,
                policy_cid=context.policy_cid,
                backend_id=getattr(backend, "backend_id", "unknown"),
                establishes=RECURSIVE_ESTABLISHES,
                does_not_establish=RECURSIVE_DOES_NOT_ESTABLISH,
                recursive_verification=False,
                children_backend_verified=True,
                test_execution_directly_proven=False,
                reason_codes=(AggregationReasonCode.RECURSIVE_VERIFY_FAILED.value,),
                message="recursive verify failed after prove",
            )

        proof_digest = "sha256:" + hashlib.sha256(proof_bytes).hexdigest()
        aggregate_root = _digest_payload(
            _DOMAIN_RECURSIVE,
            {
                "child_root": child_root,
                "proof_digest": proof_digest,
                "child_unit_ids": list(validation.unit_ids),
                "child_digests": list(validation.digests),
                "repository_id": context.repository_id,
                "environment_cid": context.environment_cid,
                "policy_cid": context.policy_cid,
                "backend_id": getattr(backend, "backend_id", "unknown"),
                "child_count": len(validation.unit_ids),
            },
        )
        return RecursiveAggregationResult(
            schema=RECURSIVE_SCHEMA,
            evidence_subset=RECURSIVE_AGGREGATION_EVIDENCE,
            outcome=AggregationOutcome.AGGREGATED,
            mode=AggregationMode.RECURSIVE_VERIFICATION,
            aggregate_root=aggregate_root,
            child_root=child_root,
            child_count=len(validation.unit_ids),
            child_unit_ids=validation.unit_ids,
            child_digests=validation.digests,
            repository_id=context.repository_id,
            environment_cid=context.environment_cid,
            policy_cid=context.policy_cid,
            backend_id=getattr(backend, "backend_id", "unknown"),
            establishes=RECURSIVE_ESTABLISHES,
            does_not_establish=RECURSIVE_DOES_NOT_ESTABLISH,
            recursive_verification=True,
            children_backend_verified=True,
            test_execution_directly_proven=False,
            recursive_proof_digest=proof_digest,
            reason_codes=(AggregationReasonCode.COVERED.value,),
            message=(
                "recursive_verification: backend verified every child and the "
                "recursive aggregate under admitted capability"
            ),
        )

    def _reject_manifest(
        self,
        context: AggregationContext,
        *,
        reason: AggregationReasonCode,
        message: str,
    ) -> ManifestAggregationResult:
        return ManifestAggregationResult(
            schema=MANIFEST_SCHEMA,
            evidence_subset=MANIFEST_AGGREGATION_EVIDENCE,
            outcome=AggregationOutcome.REJECTED,
            mode=AggregationMode.MANIFEST_AGGREGATION,
            aggregate_root=None,
            manifest_root=None,
            child_count=0,
            child_unit_ids=(),
            child_digests=(),
            category_roots={},
            batch_nodes=(),
            category_nodes=(),
            repository_node=None,
            rebuilt_categories=(),
            repository_id=context.repository_id,
            environment_cid=context.environment_cid,
            policy_cid=context.policy_cid,
            establishes=MANIFEST_ESTABLISHES,
            does_not_establish=MANIFEST_DOES_NOT_ESTABLISH,
            recursive_verification=False,
            children_individually_verified=False,
            test_execution_directly_proven=False,
            receipt_aggregation=False,
            signer_trust_stated=False,
            reason_codes=(reason.value,),
            message=message,
        )


def aggregate_verified_units(
    children: Sequence[VerifiedChild],
    context: AggregationContext | Mapping[str, Any],
    *,
    fan_in: int = DEFAULT_FAN_IN,
    capability: ProofBackendCapability | None = None,
    recursive_backend: RecursiveAggregationBackend | None = None,
    expected_unit_ids: Sequence[str] | None = None,
    affected_unit_ids: Sequence[str] | None = None,
    prefer_recursive: bool = True,
    force_mode: AggregationMode | str | None = None,
) -> AggregationResult:
    """Public facade for bounded manifest / capability-gated recursive aggregation."""

    ctx = (
        context
        if isinstance(context, AggregationContext)
        else AggregationContext(
            repository_id=str(context.get("repository_id") or ""),
            environment_cid=str(context.get("environment_cid") or ""),
            policy_cid=str(context.get("policy_cid") or "n/a"),
            parent_aggregate_root=context.get("parent_aggregate_root"),  # type: ignore[arg-type]
        )
    )
    mode: AggregationMode | None = None
    if force_mode is not None:
        mode = AggregationMode(str(getattr(force_mode, "value", force_mode)))
    return ProofAggregator(
        fan_in=fan_in,
        capability=capability,
        recursive_backend=recursive_backend,
        prefer_recursive=prefer_recursive,
    ).aggregate(
        children,
        ctx,
        expected_unit_ids=expected_unit_ids,
        affected_unit_ids=affected_unit_ids,
        force_mode=mode,
    )


__all__ = (
    "MANIFEST_AGGREGATION_EVIDENCE",
    "RECURSIVE_AGGREGATION_EVIDENCE",
    "MANIFEST_SCHEMA",
    "RECURSIVE_SCHEMA",
    "VERIFICATION_SCHEMA",
    "DEFAULT_FAN_IN",
    "MANIFEST_ESTABLISHES",
    "MANIFEST_DOES_NOT_ESTABLISH",
    "RECURSIVE_ESTABLISHES",
    "RECURSIVE_DOES_NOT_ESTABLISH",
    "RECEIPT_AGGREGATION_ESTABLISHES",
    "RECEIPT_AGGREGATION_DOES_NOT_ESTABLISH",
    "AggregationError",
    "AggregationMode",
    "AggregationLevel",
    "AggregationOutcome",
    "AggregationReasonCode",
    "VerifiedChild",
    "AggregationContext",
    "AggregateNode",
    "ManifestAggregationResult",
    "RecursiveAggregationResult",
    "AggregationVerificationResult",
    "RecursiveAggregationBackend",
    "HermeticRecursiveAggregationBackend",
    "ProofAggregator",
    "aggregate_verified_units",
    "closed_aggregation_modes",
    "closed_aggregation_reason_codes",
    "closed_aggregation_levels",
)
