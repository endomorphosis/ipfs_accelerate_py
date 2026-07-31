"""Federated exact proof-cache gate for the deterministic doctor (LPR-032).

This module is a thin federation adapter, not a second proof store.  Durable
authority remains :class:`FormalVerificationCache`.  Optional surfaces
(``ProverEvidenceStore``, ``RuntimeCAS``, MCP contract caches, attested proof
corpora, legacy IPFS transport hints) may nominate or accelerate lookups, but a
positive hit is reusable only after complete key validation and native receipt
reconstruction against the current semantic roots.

Negative hits, timeouts, resource exhaustion, raw countermodels, partial
receipts, and solver-only success are diagnostics — never proofs.  Cache
bindings are rechecked immediately before render and commit.  Semantic-root
changes invalidate descendants and write index tombstones; poisoned or
equivocal entries are quarantined.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..analysis import content_identity_bridge as _content_identity_bridge
from ..analysis.content_identity_bridge import (
    CID_VERSION,
    LOGIC_IR_PROFILE,
    MULTIBASE_BASE32,
    MULTICODEC_DAG_JSON,
    MULTICODEC_RAW,
    MULTIHASH_SHA2_256,
    STRICT_ARTIFACT_PROFILE,
    is_digest_shaped,
    sha256_digest_label,
)
from .formal_verification_cache import (
    CacheLookupStatus,
    CacheRejectionReason,
    FormalVerificationCache,
    ProofCacheKey as FormalProofCacheKey,
    build_proof_cache_key as build_formal_proof_cache_key,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    EvidenceFreshness,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    assurance_satisfies,
    canonical_json,
)


DOCTOR_PROOF_CACHE_INTERFACE: Final = "DoctorProofCacheGate@1"
DOCTOR_PROOF_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/doctor-proof-cache-key@1"
)
DOCTOR_PROOF_CACHE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/doctor-proof-cache-binding@1"
)
DOCTOR_CACHE_AUDIT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/doctor-cache-audit-receipt@1"
)
DOCTOR_CACHE_TOMBSTONE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/doctor-cache-tombstone@1"
)
DEFAULT_POSITIVE_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_NEGATIVE_TTL_SECONDS: Final = 60
MAX_NEGATIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_MAX_ENTRIES: Final = 1024
DEFAULT_MAX_BYTES: Final = 64 * 1024 * 1024

# Semantic root dimensions that, when changed, invalidate descendant entries.
_SEMANTIC_ROOT_FIELDS: Final = (
    "forest",
    "tree",
    "overlay",
    "ast",
    "graph",
    "corpus",
    "goal",
    "premises",
    "translation",
    "solver",
    "kernel",
    "toolchain",
    "registry",
    "policy",
    "budget",
    "sandbox",
    "environment",
    "candidate_tree",
)


class DoctorCacheDisposition(str, Enum):
    """Closed outcomes for federated cache consultation."""

    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"
    DIAGNOSTIC = "diagnostic"
    QUARANTINED = "quarantined"
    TOMBSTONED = "tombstoned"


class DoctorCacheStage(str, Enum):
    """Lifecycle points that revalidate positive hits."""

    LOOKUP = "lookup"
    RENDER = "render"
    COMMIT = "commit"
    INVALIDATE = "invalidate"
    QUARANTINE = "quarantine"


class DoctorCacheReason(str, Enum):
    """Stable audit reason codes for doctor cache federation."""

    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    IDENTITY_INVALID = "identity_invalid"
    DIGEST_PSEUDO_CID = "digest_like_pseudo_cid"
    DOUBLE_HASHING = "double_hashing"
    ALIAS_PROFILE_MISMATCH = "alias_profile_mismatch"
    CROSS_PROFILE = "cross_profile_cache_entry"
    PARTIAL_ENTRY = "partial_cache_entry"
    SOLVER_ONLY = "solver_only_cache_entry"
    RAW_COUNTERMODEL = "raw_countermodel_entry"
    EXPIRED = "expired_cache_entry"
    STALE = "stale_cache_entry"
    CORRUPT = "corrupt_cache_entry"
    BINDING_MISMATCH = "cache_binding_mismatch"
    WRONG_TREE = "wrong_repository_tree"
    REQUIRED_ASSURANCE = "required_assurance_not_satisfied"
    CANDIDATE_ONLY = "candidate_only_cache_entry"
    PRIVATE_MATERIAL = "private_material"
    EQUIVOCATION = "equivocation_quarantined"
    TOMBSTONED = "tombstoned_by_semantic_root"
    LEGACY_TRANSPORT_ONLY = "legacy_ipfs_transport_only"
    REVALIDATED = "revalidated_positive_hit"
    STORED = "cache_entry_stored"
    DIAGNOSTIC_NEGATIVE = "diagnostic_negative_hit"
    DIAGNOSTIC_TIMEOUT = "diagnostic_timeout"
    PROVIDER_RESULT_INVALID = "provider_result_invalid"
    POISONED = "poisoned_cache_entry"
    RECONSTRUCTION_FAILED = "reconstruction_failed"


class DoctorCacheValidationError(ValueError):
    """A key, identity, or provider result violated the doctor cache contract."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


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
                return True
            if _contains_private_material(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_private_material(item) for item in value)
    return False


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _logical_id(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DoctorCacheValidationError(
            f"{field_name}.logical_id must be a non-empty string",
            reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
        )
    return value.strip()


def _reject_digest_as_cid(value: Any, *, field_name: str = "cid") -> str:
    if not isinstance(value, str) or not value.strip():
        raise DoctorCacheValidationError(
            f"{field_name} must be a non-empty multiformat CID",
            reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
        )
    text = value.strip()
    if is_digest_shaped(text):
        raise DoctorCacheValidationError(
            f"{field_name} must be a CIDv1, not a digest-like pseudo-CID",
            reason_code=DoctorCacheReason.DIGEST_PSEUDO_CID.value,
        )
    # Bare 64-char hex without label is also digest-like.
    if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.casefold()):
        raise DoctorCacheValidationError(
            f"{field_name} must be a CIDv1, not a bare digest hex",
            reason_code=DoctorCacheReason.DIGEST_PSEUDO_CID.value,
        )
    return text


@dataclass(frozen=True, slots=True)
class DoctorIdentityBinding:
    """One declared profile/CID pair with retained canonical preimage."""

    logical_id: str
    profile: str
    cid: str
    canonical_bytes: bytes
    digest: str
    cid_version: int = CID_VERSION
    multibase: str = MULTIBASE_BASE32
    multicodec: str = MULTICODEC_DAG_JSON
    multihash: str = MULTIHASH_SHA2_256
    domain: str = ""
    artifact_schema: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_id", _logical_id(self.logical_id, field_name="identity")
        )
        if self.profile not in {STRICT_ARTIFACT_PROFILE, LOGIC_IR_PROFILE}:
            raise DoctorCacheValidationError(
                f"unknown identity profile: {self.profile!r}",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        if not isinstance(self.canonical_bytes, (bytes, bytearray, memoryview)):
            raise DoctorCacheValidationError(
                "canonical_bytes must be bytes-like",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        retained = bytes(self.canonical_bytes)
        object.__setattr__(self, "canonical_bytes", retained)
        expected_codec = (
            MULTICODEC_DAG_JSON
            if self.profile == STRICT_ARTIFACT_PROFILE
            else MULTICODEC_RAW
        )
        if self.multicodec != expected_codec:
            raise DoctorCacheValidationError(
                "identity profile and multicodec disagree",
                reason_code=DoctorCacheReason.ALIAS_PROFILE_MISMATCH.value,
            )
        if self.profile == LOGIC_IR_PROFILE and (
            not self.domain or not self.artifact_schema
        ):
            raise DoctorCacheValidationError(
                "logic IR identities require domain and artifact_schema",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        cid = _reject_digest_as_cid(self.cid, field_name="cid")
        object.__setattr__(self, "cid", cid)
        if self.cid_version != CID_VERSION:
            raise DoctorCacheValidationError(
                "only CIDv1 identities are admitted",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        expected_digest = sha256_digest_label(retained)
        if self.digest != expected_digest:
            # Reject double-hashing (digest of digest) and mismatched preimages.
            if is_digest_shaped(self.digest) and self.digest != expected_digest:
                # Check whether digest equals sha256 of the digest label itself.
                if self.digest == sha256_digest_label(expected_digest.encode("utf-8")):
                    raise DoctorCacheValidationError(
                        "double hashing of preimage digests is forbidden",
                        reason_code=DoctorCacheReason.DOUBLE_HASHING.value,
                    )
            raise DoctorCacheValidationError(
                "retained canonical bytes do not match the declared digest",
                reason_code=DoctorCacheReason.POISONED.value,
            )
        try:
            _content_identity_bridge.decode_and_verify_cid(
                cid,
                retained,
                expected_codec=expected_codec,
                expected_profile=self.profile,
                expected_base=self.multibase,
                expected_version=self.cid_version,
                expected_multihash=self.multihash,
            )
        except (
            _content_identity_bridge.CidValidationError,
            _content_identity_bridge.ContentIdentityError,
        ) as exc:
            raise DoctorCacheValidationError(
                f"CID/preimage validation failed: {exc}",
                reason_code=DoctorCacheReason.POISONED.value,
            ) from exc

    @classmethod
    def from_identity(
        cls,
        identity: _content_identity_bridge.ContentIdentity,
        *,
        logical_id: str,
    ) -> "DoctorIdentityBinding":
        if not isinstance(identity, _content_identity_bridge.ContentIdentity):
            raise TypeError("identity must be a ContentIdentity")
        return cls(
            logical_id=logical_id,
            profile=identity.profile,
            cid=identity.cid,
            canonical_bytes=identity.canonical_bytes,
            digest=identity.digest,
            cid_version=identity.cid_version,
            multibase=identity.multibase,
            multicodec=identity.multicodec,
            multihash=identity.multihash,
            domain=identity.domain,
            artifact_schema=identity.schema_version,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoctorIdentityBinding":
        if not isinstance(value, Mapping):
            raise DoctorCacheValidationError(
                "identity binding must be an object",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        schema = value.get("schema")
        if schema not in (None, DOCTOR_PROOF_CACHE_BINDING_SCHEMA):
            raise DoctorCacheValidationError(
                "unsupported identity-binding schema",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        encoded = value.get("canonical_bytes_hex")
        if not isinstance(encoded, str):
            raise DoctorCacheValidationError(
                "identity binding requires canonical_bytes_hex",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        try:
            retained = bytes.fromhex(encoded)
        except ValueError as exc:
            raise DoctorCacheValidationError(
                "canonical_bytes_hex is not valid hexadecimal",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            ) from exc
        return cls(
            logical_id=value.get("logical_id", ""),
            profile=value.get("profile", value.get("identity_profile", "")),
            cid=value.get("cid", ""),
            canonical_bytes=retained,
            digest=value.get("digest", ""),
            cid_version=value.get("cid_version", CID_VERSION),
            multibase=value.get("multibase", MULTIBASE_BASE32),
            multicodec=value.get("multicodec", MULTICODEC_DAG_JSON),
            multihash=value.get("multihash", MULTIHASH_SHA2_256),
            domain=value.get("domain", ""),
            artifact_schema=value.get(
                "artifact_schema", value.get("identity_schema_version", "")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_PROOF_CACHE_BINDING_SCHEMA,
            "logical_id": self.logical_id,
            "profile": self.profile,
            "cid": self.cid,
            "canonical_bytes_hex": self.canonical_bytes.hex(),
            "byte_length": len(self.canonical_bytes),
            "digest": self.digest,
            "cid_version": self.cid_version,
            "multibase": self.multibase,
            "multicodec": self.multicodec,
            "multihash": self.multihash,
            "domain": self.domain,
            "artifact_schema": self.artifact_schema,
        }

    @property
    def contains_private_material(self) -> bool:
        try:
            value = json.loads(self.canonical_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return _contains_private_material(value)

    def neutral_dict(self) -> dict[str, Any]:
        """Profile-neutral view used only to diagnose forbidden aliasing."""

        return {
            "logical_id": self.logical_id,
            "canonical_bytes_hex": self.canonical_bytes.hex(),
            "digest": self.digest,
        }


# Compatibility alias for IdentityBinding naming used elsewhere.
IdentityBinding = DoctorIdentityBinding
DoctorProofCacheBinding = DoctorIdentityBinding  # single-root binding unit


def _binding(
    value: DoctorIdentityBinding | Mapping[str, Any], name: str
) -> DoctorIdentityBinding:
    try:
        return (
            value
            if isinstance(value, DoctorIdentityBinding)
            else DoctorIdentityBinding.from_dict(value)
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, DoctorCacheValidationError):
            raise
        raise DoctorCacheValidationError(
            f"{name} is not a valid identity binding",
            reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
        ) from exc


def _bindings(
    values: Sequence[DoctorIdentityBinding | Mapping[str, Any]], name: str
) -> tuple[DoctorIdentityBinding, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DoctorCacheValidationError(
            f"{name} must be a sequence",
            reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
        )
    normalized = tuple(_binding(value, name) for value in values)
    return tuple(
        sorted(normalized, key=lambda item: (item.logical_id, item.profile, item.cid))
    )


@dataclass(frozen=True, slots=True)
class DoctorProofCacheKey:
    """Complete semantic key for doctor proof-cache federation.

    Binds forest/tree/overlay/AST/graph/corpus/goal/premises/translation/
    solver/kernel/toolchain/registry/policy/budget/sandbox/environment/
    candidate tree — every input capable of changing a reusable proof.
    """

    forest: DoctorIdentityBinding
    tree: DoctorIdentityBinding
    overlay: DoctorIdentityBinding
    ast: DoctorIdentityBinding
    graph: DoctorIdentityBinding
    corpus: DoctorIdentityBinding
    goal: DoctorIdentityBinding
    premises: tuple[DoctorIdentityBinding, ...]
    translation: DoctorIdentityBinding
    solver: DoctorIdentityBinding
    kernel: DoctorIdentityBinding
    toolchain: DoctorIdentityBinding
    registry: DoctorIdentityBinding
    policy: DoctorIdentityBinding
    budget: ResourceBudget
    sandbox: DoctorIdentityBinding
    environment: DoctorIdentityBinding
    candidate_tree: DoctorIdentityBinding
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED

    def __post_init__(self) -> None:
        for name in (
            "forest",
            "tree",
            "overlay",
            "ast",
            "graph",
            "corpus",
            "goal",
            "translation",
            "solver",
            "kernel",
            "toolchain",
            "registry",
            "policy",
            "sandbox",
            "environment",
            "candidate_tree",
        ):
            object.__setattr__(self, name, _binding(getattr(self, name), name))
        object.__setattr__(self, "premises", _bindings(self.premises, "premises"))
        budget = self.budget
        if isinstance(budget, Mapping):
            budget = ResourceBudget.from_dict(budget)
        if not isinstance(budget, ResourceBudget):
            raise DoctorCacheValidationError(
                "budget must be a ResourceBudget or object",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        object.__setattr__(self, "budget", budget)
        object.__setattr__(
            self, "required_assurance", AssuranceLevel(self.required_assurance)
        )
        # Tree and candidate_tree must agree on logical root for doctor runs.
        if self.tree.logical_id != self.candidate_tree.logical_id:
            raise DoctorCacheValidationError(
                "tree and candidate_tree logical_id must agree",
                reason_code=DoctorCacheReason.BINDING_MISMATCH.value,
            )

    @property
    def identities(self) -> tuple[DoctorIdentityBinding, ...]:
        return (
            self.forest,
            self.tree,
            self.overlay,
            self.ast,
            self.graph,
            self.corpus,
            self.goal,
            *self.premises,
            self.translation,
            self.solver,
            self.kernel,
            self.toolchain,
            self.registry,
            self.policy,
            self.sandbox,
            self.environment,
            self.candidate_tree,
        )

    @property
    def semantic_root_ids(self) -> dict[str, str]:
        return {
            "forest": self.forest.cid,
            "tree": self.tree.cid,
            "overlay": self.overlay.cid,
            "ast": self.ast.cid,
            "graph": self.graph.cid,
            "corpus": self.corpus.cid,
            "goal": self.goal.cid,
            "premises": ",".join(item.cid for item in self.premises),
            "translation": self.translation.cid,
            "solver": self.solver.cid,
            "kernel": self.kernel.cid,
            "toolchain": self.toolchain.cid,
            "registry": self.registry.cid,
            "policy": self.policy.cid,
            "budget": sha256_digest_label(
                canonical_json(self.budget.to_dict()).encode("utf-8")
            ),
            "sandbox": self.sandbox.cid,
            "environment": self.environment.cid,
            "candidate_tree": self.candidate_tree.cid,
        }

    @property
    def contains_private_material(self) -> bool:
        return any(item.contains_private_material for item in self.identities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_PROOF_CACHE_KEY_SCHEMA,
            "interface": DOCTOR_PROOF_CACHE_INTERFACE,
            "forest": self.forest.to_dict(),
            "tree": self.tree.to_dict(),
            "overlay": self.overlay.to_dict(),
            "ast": self.ast.to_dict(),
            "graph": self.graph.to_dict(),
            "corpus": self.corpus.to_dict(),
            "goal": self.goal.to_dict(),
            "premises": [item.to_dict() for item in self.premises],
            "translation": self.translation.to_dict(),
            "solver": self.solver.to_dict(),
            "kernel": self.kernel.to_dict(),
            "toolchain": self.toolchain.to_dict(),
            "registry": self.registry.to_dict(),
            "policy": self.policy.to_dict(),
            "budget": self.budget.to_dict(),
            "sandbox": self.sandbox.to_dict(),
            "environment": self.environment.to_dict(),
            "candidate_tree": self.candidate_tree.to_dict(),
            "required_assurance": self.required_assurance.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoctorProofCacheKey":
        if not isinstance(value, Mapping):
            raise DoctorCacheValidationError(
                "proof cache key must be an object",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        if value.get("schema") not in (None, DOCTOR_PROOF_CACHE_KEY_SCHEMA):
            raise DoctorCacheValidationError(
                "unsupported doctor proof-cache key schema",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        return cls(
            forest=value.get("forest"),
            tree=value.get("tree"),
            overlay=value.get("overlay"),
            ast=value.get("ast"),
            graph=value.get("graph"),
            corpus=value.get("corpus"),
            goal=value.get("goal"),
            premises=tuple(value.get("premises") or ()),
            translation=value.get("translation"),
            solver=value.get("solver"),
            kernel=value.get("kernel"),
            toolchain=value.get("toolchain"),
            registry=value.get("registry"),
            policy=value.get("policy"),
            budget=value.get("budget") or {},
            sandbox=value.get("sandbox"),
            environment=value.get("environment"),
            candidate_tree=value.get("candidate_tree"),
            required_assurance=value.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
        )

    @property
    def key_id(self) -> str:
        digest = hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()
        return f"doctor-proof-cache-key:sha256:{digest}"

    cache_key = key_id
    digest = key_id

    def to_formal_key(self) -> FormalProofCacheKey:
        semantic = self.to_dict()
        return build_formal_proof_cache_key(
            obligation={
                "obligation_id": self.goal.logical_id,
                "doctor_semantic_key": semantic,
            },
            premises=tuple(
                {"premise_id": item.logical_id, "identity": item.to_dict()}
                for item in self.premises
            ),
            translator={
                "translator_id": self.translation.logical_id,
                "identity": self.translation.to_dict(),
            },
            solver={
                "solver_id": self.solver.logical_id,
                "identity": self.solver.to_dict(),
            },
            kernel={
                "kernel_id": self.kernel.logical_id,
                "identity": self.kernel.to_dict(),
            },
            toolchain={
                "toolchain_id": self.toolchain.logical_id,
                "identity": self.toolchain.to_dict(),
            },
            theorem_registry={
                "theorem_registry_id": self.registry.logical_id,
                "identity": self.registry.to_dict(),
            },
            policy={
                "policy_id": self.policy.logical_id,
                "identity": self.policy.to_dict(),
            },
            resource_budget=self.budget.to_dict(),
            candidate_tree={
                "tree_id": self.candidate_tree.logical_id,
                "identity": self.candidate_tree.to_dict(),
                "forest": self.forest.to_dict(),
                "overlay": self.overlay.to_dict(),
                "ast": self.ast.to_dict(),
                "graph": self.graph.to_dict(),
                "corpus": self.corpus.to_dict(),
                "sandbox": self.sandbox.to_dict(),
                "environment": self.environment.to_dict(),
            },
        )

    def neutral_dict(self, *, omit_tree: bool = False) -> dict[str, Any]:
        def neutral(value: DoctorIdentityBinding) -> dict[str, Any]:
            return value.neutral_dict()

        payload: dict[str, Any] = {
            "forest": neutral(self.forest),
            "overlay": neutral(self.overlay),
            "ast": neutral(self.ast),
            "graph": neutral(self.graph),
            "corpus": neutral(self.corpus),
            "goal": neutral(self.goal),
            "premises": [neutral(item) for item in self.premises],
            "translation": neutral(self.translation),
            "solver": neutral(self.solver),
            "kernel": neutral(self.kernel),
            "toolchain": neutral(self.toolchain),
            "registry": neutral(self.registry),
            "policy": neutral(self.policy),
            "budget": self.budget.to_dict(),
            "sandbox": neutral(self.sandbox),
            "environment": neutral(self.environment),
            "required_assurance": self.required_assurance.value,
        }
        if not omit_tree:
            payload["tree"] = neutral(self.tree)
            payload["candidate_tree"] = neutral(self.candidate_tree)
        return payload


def build_doctor_proof_cache_key(**values: Any) -> DoctorProofCacheKey:
    """Construct a doctor key with a few non-ambiguous compatibility aliases."""

    aliases = {
        "resource_budget": "budget",
        "translator": "translation",
        "theorem_registry": "registry",
        "snapshot": "tree",
        "required_level": "required_assurance",
    }
    normalized = dict(values)
    for alias, canonical in aliases.items():
        if alias in normalized:
            if canonical in normalized and normalized[canonical] != normalized[alias]:
                raise ValueError(f"{alias} and {canonical} disagree")
            normalized[canonical] = normalized.pop(alias)
    return DoctorProofCacheKey(**normalized)


make_doctor_proof_cache_key = build_doctor_proof_cache_key
build_proof_cache_key = build_doctor_proof_cache_key


@dataclass(frozen=True, slots=True)
class DoctorCacheAuditReceipt:
    """Immutable audit of one federated cache consultation."""

    disposition: DoctorCacheDisposition
    stage: DoctorCacheStage
    key_id: str
    reason_codes: tuple[str, ...] = ()
    authoritative: bool = False
    reconstructed: bool = False
    receipt_id: str = ""
    assurance: str = AssuranceLevel.UNVERIFIED.value
    source: str = "formal_verification_cache"
    semantic_root_ids: Mapping[str, str] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", DoctorCacheDisposition(self.disposition)
        )
        object.__setattr__(self, "stage", DoctorCacheStage(self.stage))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "semantic_root_ids", dict(self.semantic_root_ids))
        object.__setattr__(self, "details", dict(self.details))
        if self.authoritative and self.disposition is not DoctorCacheDisposition.HIT:
            raise DoctorCacheValidationError(
                "only positive hits may claim authoritative reuse",
                reason_code=DoctorCacheReason.CANDIDATE_ONLY.value,
            )
        if self.authoritative and not self.reconstructed:
            raise DoctorCacheValidationError(
                "authoritative hits must be reconstructed against current roots",
                reason_code=DoctorCacheReason.RECONSTRUCTION_FAILED.value,
            )

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def hit(self) -> bool:
        return self.disposition is DoctorCacheDisposition.HIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_CACHE_AUDIT_RECEIPT_SCHEMA,
            "interface": DOCTOR_PROOF_CACHE_INTERFACE,
            "disposition": self.disposition.value,
            "stage": self.stage.value,
            "key_id": self.key_id,
            "reason_codes": list(self.reason_codes),
            "authoritative": self.authoritative,
            "reconstructed": self.reconstructed,
            "receipt_id": self.receipt_id,
            "assurance": self.assurance,
            "source": self.source,
            "semantic_root_ids": dict(self.semantic_root_ids),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoctorCacheAuditReceipt":
        if not isinstance(value, Mapping):
            raise DoctorCacheValidationError(
                "audit receipt must be an object",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        return cls(
            disposition=value.get("disposition", DoctorCacheDisposition.MISS),
            stage=value.get("stage", DoctorCacheStage.LOOKUP),
            key_id=str(value.get("key_id") or ""),
            reason_codes=tuple(value.get("reason_codes") or ()),
            authoritative=bool(value.get("authoritative", False)),
            reconstructed=bool(value.get("reconstructed", False)),
            receipt_id=str(value.get("receipt_id") or ""),
            assurance=str(value.get("assurance") or AssuranceLevel.UNVERIFIED.value),
            source=str(value.get("source") or "formal_verification_cache"),
            semantic_root_ids=dict(value.get("semantic_root_ids") or {}),
            details=dict(value.get("details") or {}),
        )


@dataclass(frozen=True, slots=True)
class DoctorCacheLookupResult:
    """Result of a federated doctor cache lookup."""

    disposition: DoctorCacheDisposition
    key: DoctorProofCacheKey
    receipt: ProofReceipt | None = None
    audit: DoctorCacheAuditReceipt | None = None
    reason_codes: tuple[str, ...] = ()
    diagnostic: bool = False

    @property
    def hit(self) -> bool:
        return self.disposition is DoctorCacheDisposition.HIT

    @property
    def status(self) -> CacheLookupStatus:
        if self.disposition is DoctorCacheDisposition.HIT:
            return CacheLookupStatus.HIT
        if self.disposition is DoctorCacheDisposition.MISS:
            return CacheLookupStatus.MISS
        return CacheLookupStatus.REJECTED

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        return (
            self.receipt.authoritative_assurance
            if self.receipt is not None and self.hit
            else AssuranceLevel.UNVERIFIED
        )


@dataclass(frozen=True, slots=True)
class DoctorCacheStoreResult:
    stored: bool
    key: DoctorProofCacheKey
    receipt: ProofReceipt | None = None
    reason_codes: tuple[str, ...] = ()
    audit: DoctorCacheAuditReceipt | None = None

    def __bool__(self) -> bool:
        return self.stored

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True, slots=True)
class DoctorCacheTombstone:
    """Index tombstone written when a semantic root invalidates descendants."""

    root_field: str
    root_cid: str
    key_id: str
    invalidated_at_ms: int
    reason: str = DoctorCacheReason.TOMBSTONED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_CACHE_TOMBSTONE_SCHEMA,
            "root_field": self.root_field,
            "root_cid": self.root_cid,
            "key_id": self.key_id,
            "invalidated_at_ms": self.invalidated_at_ms,
            "reason": self.reason,
        }


class DoctorProofCacheGate:
    """Thin federation gate over the authoritative formal-verification cache.

    Optional peers (``ProverEvidenceStore``, ``RuntimeCAS``, legacy IPFS, hammer
    provider-local caches, attested corpora) may be consulted for nomination,
    but only kernel-reconstructed positive hits on the formal cache may be
    reused as proof authority.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        cache: FormalVerificationCache | None = None,
        prover_evidence_store: Any | None = None,
        runtime_cas: Any | None = None,
        positive_ttl_seconds: int = DEFAULT_POSITIVE_TTL_SECONDS,
        negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if path is not None and cache is not None:
            raise ValueError("provide path or cache, not both")
        self.positive_ttl_seconds = _positive_int(
            positive_ttl_seconds, "positive_ttl_seconds"
        )
        self.negative_ttl_seconds = _positive_int(
            negative_ttl_seconds, "negative_ttl_seconds"
        )
        if self.negative_ttl_seconds > MAX_NEGATIVE_TTL_SECONDS:
            raise ValueError("negative_ttl_seconds exceeds the bounded negative TTL")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_bytes = _positive_int(max_bytes, "max_bytes")
        self._clock = clock or time.time
        if cache is None:
            kwargs: dict[str, Any] = {
                "default_ttl_seconds": self.positive_ttl_seconds
            }
            if clock is not None:
                kwargs["clock"] = clock
            cache = FormalVerificationCache(path, **kwargs)
        self._cache = cache
        self._prover_evidence_store = prover_evidence_store
        self._runtime_cas = runtime_cas
        self._lock = threading.RLock()
        # key_id -> quarantine reason
        self._quarantine: dict[str, str] = {}
        # key_id -> tombstone
        self._tombstones: dict[str, DoctorCacheTombstone] = {}
        # key_id -> set of receipt_ids observed (equivocation detection)
        self._observed_receipts: dict[str, set[str]] = {}
        # root_field:cid -> set of key_ids for descendant invalidation
        self._root_index: dict[str, set[str]] = {}
        # diagnostic-only negative/timeout outcomes (never authoritative)
        self._diagnostics: dict[str, dict[str, Any]] = {}

    @property
    def authoritative_cache(self) -> FormalVerificationCache:
        return self._cache

    @property
    def db_path(self) -> Path:
        return self._cache.db_path

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _coerce_key(
        self, key: DoctorProofCacheKey | Mapping[str, Any]
    ) -> DoctorProofCacheKey:
        return key if isinstance(key, DoctorProofCacheKey) else DoctorProofCacheKey.from_dict(key)

    def _index_roots(self, key: DoctorProofCacheKey) -> None:
        with self._lock:
            for field_name, cid in key.semantic_root_ids.items():
                index_key = f"{field_name}:{cid}"
                self._root_index.setdefault(index_key, set()).add(key.key_id)

    @staticmethod
    def _receipt_reasons(
        key: DoctorProofCacheKey, receipt: ProofReceipt, *, complete: bool = True
    ) -> set[str]:
        reasons: set[str] = set()
        if not complete:
            reasons.add(DoctorCacheReason.PARTIAL_ENTRY.value)
        if receipt.repository_tree_id != key.tree.logical_id:
            reasons.add(DoctorCacheReason.WRONG_TREE.value)
        if receipt.obligation_id != key.goal.logical_id:
            reasons.add(DoctorCacheReason.BINDING_MISMATCH.value)
        if receipt.premise_ids != tuple(item.logical_id for item in key.premises):
            reasons.add(DoctorCacheReason.BINDING_MISMATCH.value)
        bindings = (
            (receipt.translator_id, key.translation.logical_id),
            (receipt.solver_id, key.solver.logical_id),
            (receipt.kernel_id, key.kernel.logical_id),
            (receipt.toolchain_id, key.toolchain.logical_id),
            (receipt.theorem_registry_id, key.registry.logical_id),
            (receipt.policy_id, key.policy.logical_id),
        )
        if any(actual != expected for actual, expected in bindings):
            reasons.add(DoctorCacheReason.BINDING_MISMATCH.value)
        if receipt.resource_budget.to_dict() != key.budget.to_dict():
            reasons.add(DoctorCacheReason.BINDING_MISMATCH.value)
        if receipt.freshness is not EvidenceFreshness.CURRENT:
            reasons.add(DoctorCacheReason.STALE.value)
        metadata = dict(receipt.metadata or {})
        if metadata.get("raw_countermodel") is True:
            reasons.add(DoctorCacheReason.RAW_COUNTERMODEL.value)
        if metadata.get("partial") is True:
            reasons.add(DoctorCacheReason.PARTIAL_ENTRY.value)
        if receipt.verdict is not ProofVerdict.PROVED:
            if receipt.verdict is ProofVerdict.DISPROVED and metadata.get(
                "raw_countermodel", False
            ):
                reasons.add(DoctorCacheReason.RAW_COUNTERMODEL.value)
            reasons.add(DoctorCacheReason.CANDIDATE_ONLY.value)
        assurance = receipt.authoritative_assurance
        if assurance.rank <= AssuranceLevel.SOLVER_CHECKED.rank:
            reasons.add(DoctorCacheReason.SOLVER_ONLY.value)
            reasons.add(DoctorCacheReason.CANDIDATE_ONLY.value)
        if not assurance_satisfies(assurance, key.required_assurance):
            reasons.add(DoctorCacheReason.REQUIRED_ASSURANCE.value)
        return reasons

    def _audit(
        self,
        *,
        disposition: DoctorCacheDisposition,
        stage: DoctorCacheStage,
        key: DoctorProofCacheKey,
        reason_codes: Sequence[str] = (),
        receipt: ProofReceipt | None = None,
        reconstructed: bool = False,
        authoritative: bool = False,
        source: str = "formal_verification_cache",
        details: Mapping[str, Any] | None = None,
    ) -> DoctorCacheAuditReceipt:
        return DoctorCacheAuditReceipt(
            disposition=disposition,
            stage=stage,
            key_id=key.key_id,
            reason_codes=tuple(reason_codes),
            authoritative=authoritative,
            reconstructed=reconstructed,
            receipt_id=receipt.receipt_id if receipt is not None else "",
            assurance=(
                receipt.authoritative_assurance.value
                if receipt is not None and authoritative
                else AssuranceLevel.UNVERIFIED.value
            ),
            source=source,
            semantic_root_ids=key.semantic_root_ids,
            details=dict(details or {}),
        )

    def _check_quarantine_tombstone(
        self, key: DoctorProofCacheKey
    ) -> DoctorCacheLookupResult | None:
        with self._lock:
            if key.key_id in self._quarantine:
                reason = self._quarantine[key.key_id]
                audit = self._audit(
                    disposition=DoctorCacheDisposition.QUARANTINED,
                    stage=DoctorCacheStage.LOOKUP,
                    key=key,
                    reason_codes=(reason or DoctorCacheReason.EQUIVOCATION.value,),
                )
                return DoctorCacheLookupResult(
                    DoctorCacheDisposition.QUARANTINED,
                    key,
                    audit=audit,
                    reason_codes=audit.reason_codes,
                    diagnostic=True,
                )
            if key.key_id in self._tombstones:
                tombstone = self._tombstones[key.key_id]
                audit = self._audit(
                    disposition=DoctorCacheDisposition.TOMBSTONED,
                    stage=DoctorCacheStage.LOOKUP,
                    key=key,
                    reason_codes=(DoctorCacheReason.TOMBSTONED.value,),
                    details=tombstone.to_dict(),
                )
                return DoctorCacheLookupResult(
                    DoctorCacheDisposition.TOMBSTONED,
                    key,
                    audit=audit,
                    reason_codes=audit.reason_codes,
                    diagnostic=True,
                )
        return None

    def _record_observation(
        self, key: DoctorProofCacheKey, receipt: ProofReceipt
    ) -> str | None:
        """Track receipt identities; return quarantine reason on equivocation."""

        with self._lock:
            observed = self._observed_receipts.setdefault(key.key_id, set())
            rid = receipt.receipt_id
            if observed and rid not in observed:
                reason = DoctorCacheReason.EQUIVOCATION.value
                self._quarantine[key.key_id] = reason
                return reason
            observed.add(rid)
        return None

    def reconstruct_receipt(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any],
        receipt: ProofReceipt | Mapping[str, Any],
    ) -> tuple[ProofReceipt | None, tuple[str, ...]]:
        """Reconstruct and re-derive assurance against the current binding."""

        cache_key = self._coerce_key(key)
        try:
            typed = (
                receipt
                if isinstance(receipt, ProofReceipt)
                else ProofReceipt.from_dict(receipt)
            )
        except (TypeError, ValueError) as exc:
            return None, (DoctorCacheReason.CORRUPT.value,)
        # Round-trip through canonical form to prove independent reconstruction.
        try:
            reconstructed = ProofReceipt.from_dict(typed.to_dict())
        except (TypeError, ValueError):
            return None, (DoctorCacheReason.RECONSTRUCTION_FAILED.value,)
        if reconstructed.receipt_id != typed.receipt_id:
            return None, (DoctorCacheReason.RECONSTRUCTION_FAILED.value,)
        reasons = self._receipt_reasons(cache_key, reconstructed, complete=True)
        if reasons:
            return reconstructed, tuple(sorted(reasons))
        # Re-derive assurance; provider-claimed levels never upgrade.
        if (
            reconstructed.authoritative_assurance.rank
            < AssuranceLevel.KERNEL_VERIFIED.rank
        ):
            return reconstructed, (DoctorCacheReason.SOLVER_ONLY.value,)
        return reconstructed, ()

    def lookup(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any],
        *,
        stage: DoctorCacheStage = DoctorCacheStage.LOOKUP,
    ) -> DoctorCacheLookupResult:
        cache_key = self._coerce_key(key)
        if cache_key.contains_private_material:
            audit = self._audit(
                disposition=DoctorCacheDisposition.REJECTED,
                stage=stage,
                key=cache_key,
                reason_codes=(DoctorCacheReason.PRIVATE_MATERIAL.value,),
            )
            return DoctorCacheLookupResult(
                DoctorCacheDisposition.REJECTED,
                cache_key,
                audit=audit,
                reason_codes=audit.reason_codes,
            )
        blocked = self._check_quarantine_tombstone(cache_key)
        if blocked is not None:
            return blocked

        formal = self._cache.lookup(
            cache_key.to_formal_key(),
            required_assurance=cache_key.required_assurance,
            required_freshness=EvidenceFreshness.CURRENT,
        )
        if not formal.hit or formal.receipt is None:
            reasons = list(formal.reason_codes)
            mapped: list[str] = []
            for code in reasons:
                if code in {
                    CacheRejectionReason.STALE_ENTRY.value,
                    CacheRejectionReason.FRESHNESS_NOT_SATISFIED.value,
                }:
                    mapped.append(DoctorCacheReason.STALE.value)
                elif code in {
                    CacheRejectionReason.PARTIAL_ENTRY.value,
                    CacheRejectionReason.PARTIAL.value,
                }:
                    mapped.append(DoctorCacheReason.PARTIAL_ENTRY.value)
                elif code in {
                    CacheRejectionReason.SOLVER_ONLY_ENTRY.value,
                    CacheRejectionReason.SOLVER_ONLY.value,
                }:
                    mapped.append(DoctorCacheReason.SOLVER_ONLY.value)
                elif code in {
                    CacheRejectionReason.POISONED_ENTRY.value,
                    CacheRejectionReason.POISONED.value,
                }:
                    mapped.append(DoctorCacheReason.POISONED.value)
                elif code in {
                    CacheRejectionReason.MALFORMED_ENTRY.value,
                    CacheRejectionReason.MALFORMED.value,
                }:
                    mapped.append(DoctorCacheReason.CORRUPT.value)
                elif code == CacheRejectionReason.CACHE_MISS.value:
                    mapped.append(DoctorCacheReason.CACHE_MISS.value)
                else:
                    mapped.append(code)
            if not mapped:
                mapped = [DoctorCacheReason.CACHE_MISS.value]
            disposition = (
                DoctorCacheDisposition.MISS
                if formal.status is CacheLookupStatus.MISS
                else DoctorCacheDisposition.REJECTED
            )
            # Check diagnostic channel for negative/timeouts.
            with self._lock:
                diag = self._diagnostics.get(cache_key.key_id)
            if diag is not None:
                audit = self._audit(
                    disposition=DoctorCacheDisposition.DIAGNOSTIC,
                    stage=stage,
                    key=cache_key,
                    reason_codes=tuple(diag.get("reason_codes") or mapped),
                    details=diag,
                    source="diagnostic_channel",
                )
                return DoctorCacheLookupResult(
                    DoctorCacheDisposition.DIAGNOSTIC,
                    cache_key,
                    audit=audit,
                    reason_codes=audit.reason_codes,
                    diagnostic=True,
                )
            audit = self._audit(
                disposition=disposition,
                stage=stage,
                key=cache_key,
                reason_codes=tuple(mapped),
            )
            return DoctorCacheLookupResult(
                disposition,
                cache_key,
                audit=audit,
                reason_codes=audit.reason_codes,
            )

        reconstructed, reasons = self.reconstruct_receipt(cache_key, formal.receipt)
        if reasons or reconstructed is None:
            audit = self._audit(
                disposition=DoctorCacheDisposition.REJECTED,
                stage=stage,
                key=cache_key,
                reason_codes=reasons or (DoctorCacheReason.RECONSTRUCTION_FAILED.value,),
                receipt=reconstructed,
                reconstructed=False,
            )
            return DoctorCacheLookupResult(
                DoctorCacheDisposition.REJECTED,
                cache_key,
                receipt=reconstructed,
                audit=audit,
                reason_codes=audit.reason_codes,
            )
        equiv = self._record_observation(cache_key, reconstructed)
        if equiv is not None:
            audit = self._audit(
                disposition=DoctorCacheDisposition.QUARANTINED,
                stage=DoctorCacheStage.QUARANTINE,
                key=cache_key,
                reason_codes=(equiv,),
                receipt=reconstructed,
                reconstructed=True,
            )
            return DoctorCacheLookupResult(
                DoctorCacheDisposition.QUARANTINED,
                cache_key,
                receipt=reconstructed,
                audit=audit,
                reason_codes=audit.reason_codes,
                diagnostic=True,
            )
        revalidated = stage in {DoctorCacheStage.RENDER, DoctorCacheStage.COMMIT}
        reason_codes = (
            (DoctorCacheReason.REVALIDATED.value,)
            if revalidated
            else (DoctorCacheReason.CACHE_HIT.value,)
        )
        audit = self._audit(
            disposition=DoctorCacheDisposition.HIT,
            stage=stage,
            key=cache_key,
            reason_codes=reason_codes,
            receipt=reconstructed,
            reconstructed=True,
            authoritative=True,
        )
        return DoctorCacheLookupResult(
            DoctorCacheDisposition.HIT,
            cache_key,
            receipt=reconstructed,
            audit=audit,
            reason_codes=audit.reason_codes,
        )

    lookup_receipt = lookup

    def revalidate_for_render(
        self, key: DoctorProofCacheKey | Mapping[str, Any]
    ) -> DoctorCacheLookupResult:
        """Revalidate and reconstruct a positive hit immediately before render."""

        return self.lookup(key, stage=DoctorCacheStage.RENDER)

    def revalidate_for_commit(
        self, key: DoctorProofCacheKey | Mapping[str, Any]
    ) -> DoctorCacheLookupResult:
        """Revalidate and reconstruct a positive hit immediately before commit."""

        return self.lookup(key, stage=DoctorCacheStage.COMMIT)

    def get(
        self, key: DoctorProofCacheKey | Mapping[str, Any]
    ) -> ProofReceipt | None:
        result = self.lookup(key)
        return result.receipt if result.hit else None

    def put(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any],
        receipt: ProofReceipt | Mapping[str, Any],
        *,
        ttl_seconds: int | None = None,
        complete: bool = True,
    ) -> DoctorCacheStoreResult:
        cache_key = self._coerce_key(key)
        if cache_key.contains_private_material:
            return DoctorCacheStoreResult(
                False,
                cache_key,
                reason_codes=(DoctorCacheReason.PRIVATE_MATERIAL.value,),
            )
        blocked = self._check_quarantine_tombstone(cache_key)
        if blocked is not None:
            return DoctorCacheStoreResult(
                False,
                cache_key,
                reason_codes=blocked.reason_codes,
                audit=blocked.audit,
            )
        if not complete:
            return DoctorCacheStoreResult(
                False,
                cache_key,
                reason_codes=(DoctorCacheReason.PARTIAL_ENTRY.value,),
            )
        reconstructed, reasons = self.reconstruct_receipt(cache_key, receipt)
        if reconstructed is None or reasons:
            return DoctorCacheStoreResult(
                False,
                cache_key,
                receipt=reconstructed,
                reason_codes=reasons
                or (DoctorCacheReason.PROVIDER_RESULT_INVALID.value,),
            )
        equiv = self._record_observation(cache_key, reconstructed)
        if equiv is not None:
            return DoctorCacheStoreResult(
                False,
                cache_key,
                receipt=reconstructed,
                reason_codes=(equiv,),
            )
        ttl = self.positive_ttl_seconds if ttl_seconds is None else _positive_int(
            ttl_seconds, "ttl_seconds"
        )
        ttl = min(ttl, self.positive_ttl_seconds)
        stored = self._cache.put(
            cache_key.to_formal_key(), reconstructed, ttl_seconds=ttl, complete=True
        )
        if not stored.stored:
            mapped = []
            for code in stored.reason_codes:
                if code in {
                    CacheRejectionReason.SOLVER_ONLY_ENTRY.value,
                    CacheRejectionReason.SOLVER_ONLY.value,
                }:
                    mapped.append(DoctorCacheReason.SOLVER_ONLY.value)
                elif code in {
                    CacheRejectionReason.PARTIAL_ENTRY.value,
                    CacheRejectionReason.PARTIAL.value,
                }:
                    mapped.append(DoctorCacheReason.PARTIAL_ENTRY.value)
                else:
                    mapped.append(code)
            return DoctorCacheStoreResult(
                False,
                cache_key,
                receipt=reconstructed,
                reason_codes=tuple(mapped) or (DoctorCacheReason.PROVIDER_RESULT_INVALID.value,),
            )
        self._index_roots(cache_key)
        audit = self._audit(
            disposition=DoctorCacheDisposition.HIT,
            stage=DoctorCacheStage.LOOKUP,
            key=cache_key,
            reason_codes=(DoctorCacheReason.STORED.value,),
            receipt=reconstructed,
            reconstructed=True,
            authoritative=True,
        )
        return DoctorCacheStoreResult(
            True,
            cache_key,
            receipt=reconstructed,
            reason_codes=(DoctorCacheReason.STORED.value,),
            audit=audit,
        )

    store = put
    put_receipt = put

    def record_diagnostic(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any],
        *,
        kind: str,
        reason_codes: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> DoctorCacheAuditReceipt:
        """Record a negative hit or timeout as diagnostic-only (never a proof)."""

        cache_key = self._coerce_key(key)
        kind_norm = str(kind).strip().casefold()
        if kind_norm in {"timeout", "timed_out", "resource_exhaustion"}:
            codes = tuple(reason_codes) or (DoctorCacheReason.DIAGNOSTIC_TIMEOUT.value,)
        else:
            codes = tuple(reason_codes) or (
                DoctorCacheReason.DIAGNOSTIC_NEGATIVE.value,
            )
        payload = {
            "kind": kind_norm,
            "reason_codes": list(codes),
            "details": dict(details or {}),
            "recorded_at_ms": self._now_ms(),
            "ttl_seconds": self.negative_ttl_seconds,
        }
        with self._lock:
            self._diagnostics[cache_key.key_id] = payload
        return self._audit(
            disposition=DoctorCacheDisposition.DIAGNOSTIC,
            stage=DoctorCacheStage.LOOKUP,
            key=cache_key,
            reason_codes=codes,
            details=payload,
            source="diagnostic_channel",
        )

    def consult_legacy_ipfs(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any],
        legacy_entry: Mapping[str, Any] | None,
    ) -> DoctorCacheLookupResult:
        """Treat legacy IPFS proof-cache entries as transport hints only."""

        cache_key = self._coerce_key(key)
        audit = self._audit(
            disposition=DoctorCacheDisposition.DIAGNOSTIC,
            stage=DoctorCacheStage.LOOKUP,
            key=cache_key,
            reason_codes=(DoctorCacheReason.LEGACY_TRANSPORT_ONLY.value,),
            source="legacy_ipfs",
            details={"legacy_present": legacy_entry is not None},
        )
        return DoctorCacheLookupResult(
            DoctorCacheDisposition.DIAGNOSTIC,
            cache_key,
            audit=audit,
            reason_codes=audit.reason_codes,
            diagnostic=True,
        )

    def invalidate_semantic_root(
        self,
        *,
        root_field: str,
        root_cid: str,
        reason: str = DoctorCacheReason.TOMBSTONED.value,
    ) -> tuple[DoctorCacheTombstone, ...]:
        """Invalidate all keys indexed under a changed semantic root."""

        if root_field not in _SEMANTIC_ROOT_FIELDS:
            raise DoctorCacheValidationError(
                f"unknown semantic root field: {root_field!r}",
                reason_code=DoctorCacheReason.BINDING_MISMATCH.value,
            )
        if not isinstance(root_cid, str) or not root_cid.strip():
            raise DoctorCacheValidationError(
                "root_cid must be a non-empty string",
                reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
            )
        index_key = f"{root_field}:{root_cid.strip()}"
        now = self._now_ms()
        tombstones: list[DoctorCacheTombstone] = []
        with self._lock:
            key_ids = set(self._root_index.get(index_key, set()))
            for key_id in key_ids:
                tombstone = DoctorCacheTombstone(
                    root_field=root_field,
                    root_cid=root_cid.strip(),
                    key_id=key_id,
                    invalidated_at_ms=now,
                    reason=reason,
                )
                self._tombstones[key_id] = tombstone
                tombstones.append(tombstone)
                # Drop diagnostics and observed receipts for the tombstoned key.
                self._diagnostics.pop(key_id, None)
                self._observed_receipts.pop(key_id, None)
            self._root_index.pop(index_key, None)
        # Best-effort delete from formal cache by scanning known key ids is not
        # available without reverse maps; tombstones gate reuse locally.
        # Optionally notify RuntimeCAS if present.
        cas = self._runtime_cas
        if cas is not None:
            invalidate = getattr(cas, "invalidate_semantic_dependency", None)
            if callable(invalidate):
                try:
                    invalidate(root_cid.strip())
                except Exception:
                    pass
        return tuple(tombstones)

    def quarantine(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any] | str,
        *,
        reason: str = DoctorCacheReason.EQUIVOCATION.value,
    ) -> DoctorCacheAuditReceipt:
        """Quarantine a key so subsequent hits cannot promote it."""

        if isinstance(key, str):
            key_id = key.strip()
            if not key_id:
                raise DoctorCacheValidationError(
                    "key_id must be non-empty",
                    reason_code=DoctorCacheReason.IDENTITY_INVALID.value,
                )
            with self._lock:
                self._quarantine[key_id] = reason
            return DoctorCacheAuditReceipt(
                disposition=DoctorCacheDisposition.QUARANTINED,
                stage=DoctorCacheStage.QUARANTINE,
                key_id=key_id,
                reason_codes=(reason,),
                details={"quarantined": True},
            )
        cache_key = self._coerce_key(key)
        with self._lock:
            self._quarantine[cache_key.key_id] = reason
        return self._audit(
            disposition=DoctorCacheDisposition.QUARANTINED,
            stage=DoctorCacheStage.QUARANTINE,
            key=cache_key,
            reason_codes=(reason,),
            details={"quarantined": True},
        )

    def is_quarantined(
        self, key: DoctorProofCacheKey | Mapping[str, Any] | str
    ) -> bool:
        if isinstance(key, str):
            key_id = key.strip()
        else:
            key_id = self._coerce_key(key).key_id
        with self._lock:
            return key_id in self._quarantine

    def is_tombstoned(
        self, key: DoctorProofCacheKey | Mapping[str, Any] | str
    ) -> bool:
        if isinstance(key, str):
            key_id = key.strip()
        else:
            key_id = self._coerce_key(key).key_id
        with self._lock:
            return key_id in self._tombstones

    def purge_expired(self) -> int:
        return self._cache.purge_expired()

    def single_flight(
        self,
        key: DoctorProofCacheKey | Mapping[str, Any],
        execute: Callable[[], Any],
        **options: Any,
    ) -> Any:
        """Expose formal-cache single-flight without a second store."""

        cache_key = self._coerce_key(key)
        requested_ttl = options.pop(
            "outcome_ttl_seconds", self.negative_ttl_seconds
        )
        requested_ttl = _positive_int(requested_ttl, "outcome_ttl_seconds")
        return self._cache.single_flight(
            cache_key.to_formal_key(),
            execute,
            outcome_ttl_seconds=min(requested_ttl, self.negative_ttl_seconds),
            **options,
        )


# Public aliases matching the task AST symbols.
DoctorProofCache = DoctorProofCacheGate


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_NEGATIVE_TTL_SECONDS",
    "DEFAULT_POSITIVE_TTL_SECONDS",
    "DOCTOR_CACHE_AUDIT_RECEIPT_SCHEMA",
    "DOCTOR_CACHE_TOMBSTONE_SCHEMA",
    "DOCTOR_PROOF_CACHE_BINDING_SCHEMA",
    "DOCTOR_PROOF_CACHE_INTERFACE",
    "DOCTOR_PROOF_CACHE_KEY_SCHEMA",
    "DoctorCacheAuditReceipt",
    "DoctorCacheDisposition",
    "DoctorCacheLookupResult",
    "DoctorCacheReason",
    "DoctorCacheStage",
    "DoctorCacheStoreResult",
    "DoctorCacheTombstone",
    "DoctorCacheValidationError",
    "DoctorIdentityBinding",
    "DoctorProofCache",
    "DoctorProofCacheBinding",
    "DoctorProofCacheGate",
    "DoctorProofCacheKey",
    "IdentityBinding",
    "MAX_NEGATIVE_TTL_SECONDS",
    "build_doctor_proof_cache_key",
    "build_proof_cache_key",
    "make_doctor_proof_cache_key",
]
