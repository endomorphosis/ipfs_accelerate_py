"""Trust-aware caching for MCP contract proof results.

The durable trust root in this module is deliberately the existing
``FormalVerificationCache``.  This adapter adds the MCP-specific semantic key,
declared content-identity bindings, exact receipt checks, bounded retention,
and lookup-before-prove orchestration.  It does not maintain a second receipt
store and it never trusts a provider's claimed assurance.

Only current, independently kernel-verified positive receipts are retained as
authoritative cache entries.  Negative and inconclusive outcomes are shared
with concurrent followers through the formal cache's bounded single-flight
outcome channel, but are not promoted to proof evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..analysis.content_identity_bridge import (
    CID_VERSION,
    LOGIC_IR_PROFILE,
    MULTIBASE_BASE32,
    MULTICODEC_DAG_JSON,
    MULTICODEC_RAW,
    MULTIHASH_SHA2_256,
    STRICT_ARTIFACT_PROFILE,
    CidValidationError,
    ContentIdentity,
    ContentIdentityError,
    decode_and_verify_cid,
    sha256_digest_label,
)
from .formal_verification_cache import (
    CacheLookupStatus,
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
from .mcp_contract_prover import (
    ContractProofOutcome,
    ContractProofRoute,
    McpContractProofResult,
)


MCP_CONTRACT_PROOF_CACHE_INTERFACE: Final = "TrustAwareProofCache@1"
MCP_CONTRACT_PROOF_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-proof-cache-key@1"
)
MCP_CONTENT_IDENTITY_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-cache-identity-binding@1"
)
DEFAULT_POSITIVE_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_NEGATIVE_TTL_SECONDS: Final = 60
MAX_NEGATIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_MAX_ENTRIES: Final = 1024
DEFAULT_MAX_BYTES: Final = 64 * 1024 * 1024


class ProofCacheReason(str, Enum):
    """Stable audit reasons added by the MCP adapter."""

    CACHE_MISS = "cache_miss"
    CACHE_HIT = "cache_hit"
    IDENTITY_INVALID = "identity_invalid"
    CROSS_PROFILE = "cross_profile_cache_entry"
    WRONG_TREE = "wrong_repository_tree"
    PRIVATE_MATERIAL = "private_material"
    CANDIDATE_ONLY = "candidate_only_cache_entry"
    BINDING_MISMATCH = "cache_binding_mismatch"
    REQUIRED_ASSURANCE = "required_assurance_not_satisfied"
    STALE = "stale_cache_entry"
    POISONED = "poisoned_cache_entry"
    PROVIDER_RESULT_INVALID = "provider_result_invalid"
    STORED = "cache_entry_stored"
    SHARED_FLIGHT = "single_flight_shared"
    RETENTION_EVICTED = "retention_evicted"


class ProofCacheValidationError(ValueError):
    """A key, identity, or provider result violated the cache contract."""

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
        raise ProofCacheValidationError(
            f"{field_name}.logical_id must be a non-empty string",
            reason_code=ProofCacheReason.IDENTITY_INVALID.value,
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class IdentityBinding:
    """One declared profile/CID and its retained canonical preimage."""

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
            raise ProofCacheValidationError(
                f"unknown identity profile: {self.profile!r}",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        if not isinstance(self.canonical_bytes, (bytes, bytearray, memoryview)):
            raise ProofCacheValidationError(
                "canonical_bytes must be bytes-like",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        retained = bytes(self.canonical_bytes)
        object.__setattr__(self, "canonical_bytes", retained)
        expected_codec = (
            MULTICODEC_DAG_JSON
            if self.profile == STRICT_ARTIFACT_PROFILE
            else MULTICODEC_RAW
        )
        if self.multicodec != expected_codec:
            raise ProofCacheValidationError(
                "identity profile and multicodec disagree",
                reason_code=ProofCacheReason.CROSS_PROFILE.value,
            )
        if self.profile == LOGIC_IR_PROFILE and (
            not self.domain or not self.artifact_schema
        ):
            raise ProofCacheValidationError(
                "logic IR identities require domain and artifact_schema",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        expected_digest = sha256_digest_label(retained)
        if self.digest != expected_digest:
            raise ProofCacheValidationError(
                "retained canonical bytes do not match the declared digest",
                reason_code=ProofCacheReason.POISONED.value,
            )
        try:
            decode_and_verify_cid(
                self.cid,
                retained,
                expected_codec=expected_codec,
                expected_profile=self.profile,
                expected_base=self.multibase,
                expected_version=self.cid_version,
                expected_multihash=self.multihash,
            )
        except (CidValidationError, ContentIdentityError) as exc:
            raise ProofCacheValidationError(
                f"CID/preimage validation failed: {exc}",
                reason_code=ProofCacheReason.POISONED.value,
            ) from exc

    @classmethod
    def from_identity(
        cls, identity: ContentIdentity, *, logical_id: str
    ) -> "IdentityBinding":
        if not isinstance(identity, ContentIdentity):
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
    def from_dict(cls, value: Mapping[str, Any]) -> "IdentityBinding":
        if not isinstance(value, Mapping):
            raise ProofCacheValidationError(
                "identity binding must be an object",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        schema = value.get("schema")
        if schema not in (None, MCP_CONTENT_IDENTITY_BINDING_SCHEMA):
            raise ProofCacheValidationError(
                "unsupported identity-binding schema",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        encoded = value.get("canonical_bytes_hex")
        if not isinstance(encoded, str):
            raise ProofCacheValidationError(
                "identity binding requires canonical_bytes_hex",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        try:
            retained = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ProofCacheValidationError(
                "canonical_bytes_hex is not valid hexadecimal",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
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
            "schema": MCP_CONTENT_IDENTITY_BINDING_SCHEMA,
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


ContentIdentityBinding = IdentityBinding


def _binding(value: IdentityBinding | Mapping[str, Any], name: str) -> IdentityBinding:
    try:
        return value if isinstance(value, IdentityBinding) else IdentityBinding.from_dict(value)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ProofCacheValidationError):
            raise
        raise ProofCacheValidationError(
            f"{name} is not a valid identity binding",
            reason_code=ProofCacheReason.IDENTITY_INVALID.value,
        ) from exc


def _bindings(
    values: Sequence[IdentityBinding | Mapping[str, Any]], name: str
) -> tuple[IdentityBinding, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofCacheValidationError(
            f"{name} must be a sequence",
            reason_code=ProofCacheReason.IDENTITY_INVALID.value,
        )
    normalized = tuple(_binding(value, name) for value in values)
    return tuple(sorted(normalized, key=lambda item: (item.logical_id, item.profile, item.cid)))


@dataclass(frozen=True, slots=True)
class ProofCacheKey:
    """All semantic and execution inputs capable of changing an MCP proof."""

    snapshot: IdentityBinding
    scope: tuple[IdentityBinding, ...]
    property_catalog: IdentityBinding
    obligation: IdentityBinding
    premises: tuple[IdentityBinding, ...]
    assumptions: tuple[IdentityBinding, ...]
    provider: IdentityBinding
    translator: IdentityBinding
    solver: IdentityBinding
    kernel: IdentityBinding
    toolchain: IdentityBinding
    theorem_registry: IdentityBinding
    policy: IdentityBinding
    capability_report: IdentityBinding
    resource_budget: ResourceBudget
    required_assurance: AssuranceLevel
    route: ContractProofRoute

    def __post_init__(self) -> None:
        for name in (
            "snapshot",
            "property_catalog",
            "obligation",
            "provider",
            "translator",
            "solver",
            "kernel",
            "toolchain",
            "theorem_registry",
            "policy",
            "capability_report",
        ):
            object.__setattr__(self, name, _binding(getattr(self, name), name))
        for name in ("scope", "premises", "assumptions"):
            object.__setattr__(self, name, _bindings(getattr(self, name), name))
        budget = self.resource_budget
        if isinstance(budget, Mapping):
            budget = ResourceBudget.from_dict(budget)
        if not isinstance(budget, ResourceBudget):
            raise ProofCacheValidationError(
                "resource_budget must be a ResourceBudget or object",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        object.__setattr__(self, "resource_budget", budget)
        object.__setattr__(
            self, "required_assurance", AssuranceLevel(self.required_assurance)
        )
        object.__setattr__(self, "route", ContractProofRoute(self.route))
        if not self.scope:
            raise ProofCacheValidationError(
                "scope must contain at least one identity",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )

    @property
    def identities(self) -> tuple[IdentityBinding, ...]:
        return (
            self.snapshot,
            *self.scope,
            self.property_catalog,
            self.obligation,
            *self.premises,
            *self.assumptions,
            self.provider,
            self.translator,
            self.solver,
            self.kernel,
            self.toolchain,
            self.theorem_registry,
            self.policy,
            self.capability_report,
        )

    @property
    def contains_private_material(self) -> bool:
        return any(item.contains_private_material for item in self.identities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_PROOF_CACHE_KEY_SCHEMA,
            "interface": MCP_CONTRACT_PROOF_CACHE_INTERFACE,
            "snapshot": self.snapshot.to_dict(),
            "scope": [item.to_dict() for item in self.scope],
            "property_catalog": self.property_catalog.to_dict(),
            "obligation": self.obligation.to_dict(),
            "premises": [item.to_dict() for item in self.premises],
            "assumptions": [item.to_dict() for item in self.assumptions],
            "provider": self.provider.to_dict(),
            "translator": self.translator.to_dict(),
            "solver": self.solver.to_dict(),
            "kernel": self.kernel.to_dict(),
            "toolchain": self.toolchain.to_dict(),
            "theorem_registry": self.theorem_registry.to_dict(),
            "policy": self.policy.to_dict(),
            "capability_report": self.capability_report.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "required_assurance": self.required_assurance.value,
            "route": self.route.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofCacheKey":
        if not isinstance(value, Mapping):
            raise ProofCacheValidationError(
                "proof cache key must be an object",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        if value.get("schema") not in (None, MCP_CONTRACT_PROOF_CACHE_KEY_SCHEMA):
            raise ProofCacheValidationError(
                "unsupported MCP proof-cache key schema",
                reason_code=ProofCacheReason.IDENTITY_INVALID.value,
            )
        return cls(
            snapshot=value.get("snapshot"),
            scope=tuple(value.get("scope") or ()),
            property_catalog=value.get("property_catalog"),
            obligation=value.get("obligation"),
            premises=tuple(value.get("premises") or ()),
            assumptions=tuple(value.get("assumptions") or ()),
            provider=value.get("provider"),
            translator=value.get("translator"),
            solver=value.get("solver"),
            kernel=value.get("kernel"),
            toolchain=value.get("toolchain"),
            theorem_registry=value.get("theorem_registry"),
            policy=value.get("policy"),
            capability_report=value.get("capability_report"),
            resource_budget=value.get("resource_budget") or {},
            required_assurance=value.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            route=value.get("route", ContractProofRoute.NONE),
        )

    @property
    def key_id(self) -> str:
        digest = hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()
        return f"mcp-proof-cache-key:sha256:{digest}"

    cache_key = key_id
    digest = key_id

    def to_formal_key(self) -> FormalProofCacheKey:
        semantic = self.to_dict()
        return build_formal_proof_cache_key(
            obligation={
                "obligation_id": self.obligation.logical_id,
                "mcp_semantic_key": semantic,
            },
            premises=tuple(
                {"premise_id": item.logical_id, "identity": item.to_dict()}
                for item in self.premises
            ),
            translator={
                "translator_id": self.translator.logical_id,
                "identity": self.translator.to_dict(),
            },
            solver={
                "solver_id": self.solver.logical_id,
                "identity": self.solver.to_dict(),
                "provider": self.provider.to_dict(),
                "capability_report": self.capability_report.to_dict(),
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
                "theorem_registry_id": self.theorem_registry.logical_id,
                "identity": self.theorem_registry.to_dict(),
            },
            policy={
                "policy_id": self.policy.logical_id,
                "identity": self.policy.to_dict(),
            },
            resource_budget=self.resource_budget.to_dict(),
            candidate_tree={
                "tree_id": self.snapshot.logical_id,
                "identity": self.snapshot.to_dict(),
            },
        )

    def neutral_dict(self, *, omit_snapshot: bool = False) -> dict[str, Any]:
        def neutral(value: IdentityBinding) -> dict[str, Any]:
            return value.neutral_dict()

        payload: dict[str, Any] = {
            "scope": [neutral(item) for item in self.scope],
            "property_catalog": neutral(self.property_catalog),
            "obligation": neutral(self.obligation),
            "premises": [neutral(item) for item in self.premises],
            "assumptions": [neutral(item) for item in self.assumptions],
            "provider": neutral(self.provider),
            "translator": neutral(self.translator),
            "solver": neutral(self.solver),
            "kernel": neutral(self.kernel),
            "toolchain": neutral(self.toolchain),
            "theorem_registry": neutral(self.theorem_registry),
            "policy": neutral(self.policy),
            "capability_report": neutral(self.capability_report),
            "resource_budget": self.resource_budget.to_dict(),
            "required_assurance": self.required_assurance.value,
            "route": self.route.value,
        }
        if not omit_snapshot:
            payload["snapshot"] = neutral(self.snapshot)
        return payload


def build_proof_cache_key(**values: Any) -> ProofCacheKey:
    """Construct an MCP key with a few non-ambiguous compatibility aliases."""

    aliases = {
        "catalog": "property_catalog",
        "capability": "capability_report",
        "required_level": "required_assurance",
    }
    normalized = dict(values)
    for alias, canonical in aliases.items():
        if alias in normalized:
            if canonical in normalized and normalized[canonical] != normalized[alias]:
                raise ValueError(f"{alias} and {canonical} disagree")
            normalized[canonical] = normalized.pop(alias)
    return ProofCacheKey(**normalized)


make_proof_cache_key = build_proof_cache_key


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    status: CacheLookupStatus
    key: ProofCacheKey
    receipt: ProofReceipt | None = None
    result: McpContractProofResult | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.status is CacheLookupStatus.HIT

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
class CacheStoreResult:
    stored: bool
    key: ProofCacheKey
    receipt: ProofReceipt | None = None
    reason_codes: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.stored

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True, slots=True)
class CachedProofResult:
    result: McpContractProofResult
    cache_hit: bool
    shared_flight: bool
    reason_codes: tuple[str, ...]

    @property
    def receipt(self) -> ProofReceipt:
        return self.result.receipt

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        return self.receipt.authoritative_assurance


@dataclass(frozen=True, slots=True)
class RetentionStats:
    entries: int
    encoded_bytes: int
    max_entries: int
    max_bytes: int


class TrustAwareProofCache:
    """MCP adapter over the sole authoritative formal-verification cache."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        cache: FormalVerificationCache | None = None,
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
            raise ValueError(
                "negative_ttl_seconds exceeds the bounded negative TTL"
            )
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_bytes = _positive_int(max_bytes, "max_bytes")
        if cache is None:
            kwargs: dict[str, Any] = {
                "default_ttl_seconds": self.positive_ttl_seconds
            }
            if clock is not None:
                kwargs["clock"] = clock
            cache = FormalVerificationCache(path, **kwargs)
        self._cache = cache

    @property
    def authoritative_cache(self) -> FormalVerificationCache:
        return self._cache

    @property
    def db_path(self) -> Path:
        return self._cache.db_path

    @staticmethod
    def _receipt_reasons(key: ProofCacheKey, receipt: ProofReceipt) -> set[str]:
        reasons: set[str] = set()
        if receipt.repository_tree_id != key.snapshot.logical_id:
            reasons.add(ProofCacheReason.WRONG_TREE.value)
        if receipt.obligation_id != key.obligation.logical_id:
            reasons.add(ProofCacheReason.BINDING_MISMATCH.value)
        if receipt.ast_scope_ids != tuple(item.logical_id for item in key.scope):
            reasons.add(ProofCacheReason.BINDING_MISMATCH.value)
        if receipt.premise_ids != tuple(item.logical_id for item in key.premises):
            reasons.add(ProofCacheReason.BINDING_MISMATCH.value)
        bindings = (
            (receipt.translator_id, key.translator.logical_id),
            (receipt.solver_id, key.solver.logical_id),
            (receipt.kernel_id, key.kernel.logical_id),
            (receipt.toolchain_id, key.toolchain.logical_id),
            (receipt.theorem_registry_id, key.theorem_registry.logical_id),
            (receipt.policy_id, key.policy.logical_id),
        )
        if any(actual != expected for actual, expected in bindings):
            reasons.add(ProofCacheReason.BINDING_MISMATCH.value)
        if receipt.resource_budget.to_dict() != key.resource_budget.to_dict():
            reasons.add(ProofCacheReason.BINDING_MISMATCH.value)
        if receipt.freshness is not EvidenceFreshness.CURRENT:
            reasons.add(ProofCacheReason.STALE.value)
        if receipt.verdict is not ProofVerdict.PROVED:
            reasons.add(ProofCacheReason.CANDIDATE_ONLY.value)
        if receipt.authoritative_assurance.rank <= AssuranceLevel.SOLVER_CHECKED.rank:
            reasons.add(ProofCacheReason.CANDIDATE_ONLY.value)
        if not assurance_satisfies(
            receipt.authoritative_assurance, key.required_assurance
        ):
            reasons.add(ProofCacheReason.REQUIRED_ASSURANCE.value)
        return reasons

    @staticmethod
    def _result_from_receipt(
        key: ProofCacheKey, receipt: ProofReceipt
    ) -> McpContractProofResult:
        return McpContractProofResult(
            obligation_id=receipt.obligation_id,
            outcome=ContractProofOutcome.PROVED,
            route=key.route,
            reason_codes=(ProofCacheReason.CACHE_HIT.value,),
            receipt=receipt,
        )

    def _diagnose_miss(self, key: ProofCacheKey) -> tuple[str, ...]:
        """Find only close semantic siblings to produce an actionable miss."""

        connection = self._cache._connect()
        try:
            rows = connection.execute(
                "SELECT key_json FROM proof_cache_entries "
                "ORDER BY created_at_ms DESC LIMIT ?",
                (self.max_entries,),
            ).fetchall()
        finally:
            connection.close()
        requested_neutral = key.neutral_dict()
        requested_without_tree = key.neutral_dict(omit_snapshot=True)
        for row in rows:
            try:
                formal_payload = json.loads(str(row["key_json"]))
                semantic = formal_payload["obligation"]["mcp_semantic_key"]
                candidate = ProofCacheKey.from_dict(semantic)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                candidate.neutral_dict() == requested_neutral
                and candidate.to_dict() != key.to_dict()
            ):
                return (ProofCacheReason.CROSS_PROFILE.value,)
            if (
                candidate.neutral_dict(omit_snapshot=True) == requested_without_tree
                and candidate.snapshot.logical_id != key.snapshot.logical_id
            ):
                return (ProofCacheReason.WRONG_TREE.value,)
        return (ProofCacheReason.CACHE_MISS.value,)

    def lookup(self, key: ProofCacheKey | Mapping[str, Any]) -> CacheLookupResult:
        cache_key = key if isinstance(key, ProofCacheKey) else ProofCacheKey.from_dict(key)
        if cache_key.contains_private_material:
            return CacheLookupResult(
                CacheLookupStatus.REJECTED,
                cache_key,
                reason_codes=(ProofCacheReason.PRIVATE_MATERIAL.value,),
            )
        formal = self._cache.lookup(
            cache_key.to_formal_key(),
            required_assurance=cache_key.required_assurance,
            required_freshness=EvidenceFreshness.CURRENT,
        )
        if not formal.hit or formal.receipt is None:
            reasons = formal.reason_codes
            status = formal.status
            if formal.status is CacheLookupStatus.MISS:
                reasons = self._diagnose_miss(cache_key)
                if reasons != (ProofCacheReason.CACHE_MISS.value,):
                    status = CacheLookupStatus.REJECTED
            return CacheLookupResult(
                status,
                cache_key,
                reason_codes=tuple(reasons),
            )
        reasons = self._receipt_reasons(cache_key, formal.receipt)
        if reasons:
            return CacheLookupResult(
                CacheLookupStatus.REJECTED,
                cache_key,
                receipt=formal.receipt,
                reason_codes=tuple(sorted(reasons)),
            )
        return CacheLookupResult(
            CacheLookupStatus.HIT,
            cache_key,
            receipt=formal.receipt,
            result=self._result_from_receipt(cache_key, formal.receipt),
            reason_codes=(ProofCacheReason.CACHE_HIT.value,),
        )

    lookup_receipt = lookup

    def get(
        self, key: ProofCacheKey | Mapping[str, Any]
    ) -> ProofReceipt | None:
        """Return only an accepted receipt, matching the formal-cache helper."""

        lookup = self.lookup(key)
        return lookup.receipt if lookup.hit else None

    def put(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        value: McpContractProofResult | ProofReceipt | Mapping[str, Any],
        *,
        ttl_seconds: int | None = None,
    ) -> CacheStoreResult:
        cache_key = key if isinstance(key, ProofCacheKey) else ProofCacheKey.from_dict(key)
        if cache_key.contains_private_material:
            return CacheStoreResult(
                False,
                cache_key,
                reason_codes=(ProofCacheReason.PRIVATE_MATERIAL.value,),
            )
        try:
            if isinstance(value, McpContractProofResult):
                result = value
                receipt = result.receipt
                if result.outcome is not ContractProofOutcome.PROVED:
                    return CacheStoreResult(
                        False,
                        cache_key,
                        receipt=receipt,
                        reason_codes=(ProofCacheReason.CANDIDATE_ONLY.value,),
                    )
            elif isinstance(value, ProofReceipt):
                receipt = value
            elif "receipt" in value:
                result = McpContractProofResult.from_dict(value)
                receipt = result.receipt
                if result.outcome is not ContractProofOutcome.PROVED:
                    return CacheStoreResult(
                        False,
                        cache_key,
                        receipt=receipt,
                        reason_codes=(ProofCacheReason.CANDIDATE_ONLY.value,),
                    )
            else:
                receipt = ProofReceipt.from_dict(value)
        except (TypeError, ValueError) as exc:
            return CacheStoreResult(
                False,
                cache_key,
                reason_codes=(ProofCacheReason.PROVIDER_RESULT_INVALID.value,),
            )
        reasons = self._receipt_reasons(cache_key, receipt)
        if reasons:
            return CacheStoreResult(
                False,
                cache_key,
                receipt=receipt,
                reason_codes=tuple(sorted(reasons)),
            )
        ttl = self.positive_ttl_seconds if ttl_seconds is None else _positive_int(
            ttl_seconds, "ttl_seconds"
        )
        ttl = min(ttl, self.positive_ttl_seconds)
        stored = self._cache.put(
            cache_key.to_formal_key(), receipt, ttl_seconds=ttl
        )
        if not stored.stored:
            return CacheStoreResult(
                False,
                cache_key,
                receipt=receipt,
                reason_codes=stored.reason_codes,
            )
        self._enforce_retention()
        connection = self._cache._connect()
        try:
            retained = connection.execute(
                "SELECT 1 FROM proof_cache_entries WHERE key_id=?",
                (cache_key.to_formal_key().key_id,),
            ).fetchone()
        finally:
            connection.close()
        if retained is None:
            return CacheStoreResult(
                False,
                cache_key,
                receipt=receipt,
                reason_codes=(ProofCacheReason.RETENTION_EVICTED.value,),
            )
        return CacheStoreResult(
            True,
            cache_key,
            receipt=receipt,
            reason_codes=(ProofCacheReason.STORED.value,),
        )

    store = put
    put_receipt = put

    def get_or_prove(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        prove: Callable[[], McpContractProofResult | Mapping[str, Any]],
        *,
        lease_seconds: int = 300,
        wait_timeout_seconds: int = 600,
    ) -> CachedProofResult:
        """Return an exact hit or execute one provider call for all followers."""

        cache_key = key if isinstance(key, ProofCacheKey) else ProofCacheKey.from_dict(key)
        if not callable(prove):
            raise ValueError("prove must be callable")
        initial = self.lookup(cache_key)
        if initial.hit and initial.result is not None:
            return CachedProofResult(
                initial.result,
                cache_hit=True,
                shared_flight=False,
                reason_codes=initial.reason_codes,
            )
        if (
            initial.status is CacheLookupStatus.REJECTED
            and initial.reason_code == ProofCacheReason.PRIVATE_MATERIAL.value
        ):
            raise ProofCacheValidationError(
                "private material may not enter proof-cache coordination",
                reason_code=ProofCacheReason.PRIVATE_MATERIAL.value,
            )

        executed_here = False

        def execute() -> dict[str, Any]:
            nonlocal executed_here
            executed_here = True
            second = self.lookup(cache_key)
            if second.hit and second.receipt is not None:
                return {
                    "kind": "cache_hit",
                    "receipt": second.receipt.to_dict(),
                }
            raw = prove()
            try:
                result = (
                    raw
                    if isinstance(raw, McpContractProofResult)
                    else McpContractProofResult.from_dict(raw)
                )
            except (TypeError, ValueError) as exc:
                raise ProofCacheValidationError(
                    f"provider returned an invalid proof result: {exc}",
                    reason_code=ProofCacheReason.PROVIDER_RESULT_INVALID.value,
                ) from exc
            reasons = self._receipt_reasons(cache_key, result.receipt)
            structural_reasons = reasons & {
                ProofCacheReason.WRONG_TREE.value,
                ProofCacheReason.BINDING_MISMATCH.value,
                ProofCacheReason.STALE.value,
            }
            if structural_reasons:
                raise ProofCacheValidationError(
                    "provider result is detached from the requested semantic key",
                    reason_code=sorted(structural_reasons)[0],
                )
            if (
                result.outcome is ContractProofOutcome.PROVED
                and not reasons
            ):
                stored = self.put(cache_key, result)
                if not stored.stored:
                    raise ProofCacheValidationError(
                        "validated provider result could not be retained",
                        reason_code=stored.reason_code
                        or ProofCacheReason.PROVIDER_RESULT_INVALID.value,
                    )
            return {"kind": "provider_result", "result": result.to_dict()}

        payload = self._cache.single_flight(
            cache_key.to_formal_key(),
            execute,
            lease_seconds=lease_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            outcome_ttl_seconds=self.negative_ttl_seconds,
        )
        if not isinstance(payload, Mapping):
            raise ProofCacheValidationError(
                "single-flight returned a malformed result",
                reason_code=ProofCacheReason.PROVIDER_RESULT_INVALID.value,
            )
        if payload.get("kind") == "cache_hit":
            receipt_payload = payload.get("receipt")
            if not isinstance(receipt_payload, Mapping):
                raise ProofCacheValidationError(
                    "single-flight cache hit omitted its receipt",
                    reason_code=ProofCacheReason.PROVIDER_RESULT_INVALID.value,
                )
            result = self._result_from_receipt(
                cache_key, ProofReceipt.from_dict(receipt_payload)
            )
            return CachedProofResult(
                result,
                cache_hit=True,
                shared_flight=not executed_here,
                reason_codes=(ProofCacheReason.CACHE_HIT.value,),
            )
        result_payload = payload.get("result")
        if not isinstance(result_payload, Mapping):
            raise ProofCacheValidationError(
                "single-flight provider result is malformed",
                reason_code=ProofCacheReason.PROVIDER_RESULT_INVALID.value,
            )
        result = McpContractProofResult.from_dict(result_payload)
        shared = not executed_here
        return CachedProofResult(
            result,
            cache_hit=False,
            shared_flight=shared,
            reason_codes=(
                (ProofCacheReason.SHARED_FLIGHT.value,)
                if shared
                else tuple(result.reason_codes)
            ),
        )

    lookup_or_prove = get_or_prove
    execute = get_or_prove

    def single_flight(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        execute: Callable[[], Any],
        **options: Any,
    ) -> Any:
        """Expose coordination without exposing or creating another store."""

        cache_key = key if isinstance(key, ProofCacheKey) else ProofCacheKey.from_dict(key)
        requested_ttl = options.pop(
            "outcome_ttl_seconds", self.negative_ttl_seconds
        )
        requested_ttl = _positive_int(
            requested_ttl, "outcome_ttl_seconds"
        )
        return self._cache.single_flight(
            cache_key.to_formal_key(),
            execute,
            outcome_ttl_seconds=min(
                requested_ttl, self.negative_ttl_seconds
            ),
            **options,
        )

    def purge_expired(self) -> int:
        return self._cache.purge_expired()

    def _enforce_retention(self) -> int:
        """Evict oldest receipt rows until both configured bounds hold."""

        connection = self._cache._connect()
        evicted = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            while True:
                totals = connection.execute(
                    "SELECT COUNT(*) AS entries, "
                    "COALESCE(SUM(LENGTH(key_json) + LENGTH(entry_json)), 0) AS bytes "
                    "FROM proof_cache_entries"
                ).fetchone()
                entries = int(totals["entries"])
                encoded_bytes = int(totals["bytes"])
                if entries <= self.max_entries and encoded_bytes <= self.max_bytes:
                    break
                oldest = connection.execute(
                    "SELECT key_id FROM proof_cache_entries "
                    "ORDER BY created_at_ms ASC, key_id ASC LIMIT 1"
                ).fetchone()
                if oldest is None:
                    break
                key_id = str(oldest["key_id"])
                connection.execute(
                    "DELETE FROM proof_attestation_entries WHERE key_id=?", (key_id,)
                )
                connection.execute(
                    "DELETE FROM proof_cache_entries WHERE key_id=?", (key_id,)
                )
                evicted += 1
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return evicted

    def retention_stats(self) -> RetentionStats:
        connection = self._cache._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS entries, "
                "COALESCE(SUM(LENGTH(key_json) + LENGTH(entry_json)), 0) AS bytes "
                "FROM proof_cache_entries"
            ).fetchone()
        finally:
            connection.close()
        return RetentionStats(
            entries=int(row["entries"]),
            encoded_bytes=int(row["bytes"]),
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )


McpContractProofCache = TrustAwareProofCache


__all__ = [
    "CacheLookupStatus",
    "CacheLookupResult",
    "CacheStoreResult",
    "CachedProofResult",
    "ContentIdentityBinding",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_NEGATIVE_TTL_SECONDS",
    "DEFAULT_POSITIVE_TTL_SECONDS",
    "IdentityBinding",
    "MCP_CONTENT_IDENTITY_BINDING_SCHEMA",
    "MCP_CONTRACT_PROOF_CACHE_INTERFACE",
    "MCP_CONTRACT_PROOF_CACHE_KEY_SCHEMA",
    "MAX_NEGATIVE_TTL_SECONDS",
    "McpContractProofCache",
    "ProofCacheKey",
    "ProofCacheReason",
    "ProofCacheValidationError",
    "ProofReceipt",
    "RetentionStats",
    "TrustAwareProofCache",
    "build_proof_cache_key",
    "make_proof_cache_key",
]
