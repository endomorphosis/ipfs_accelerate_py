"""Exact, trust-preserving cache coordination for repository reasoning.

Interface: ``ReasoningCacheCoordinator@1``

This module is deliberately a coordinator, not another cache trust root:

* :mod:`analysis_cache` remains the analysis lookup/index store;
* :mod:`formal_verification_cache` remains the only proof-receipt cache;
* :class:`~agent_supervisor.runtime.artifact_store.BoundedArtifactStore`
  remains the content-addressed store for source receipts; and
* the native cache coordinators remain the single-flight implementations.

An analysis cache entry contains only a compact index and an immutable CAS
reference.  A hit is useful only after this module reloads that source receipt,
verifies both content addresses and the complete semantic preimage, checks the
declared dependency closure, and derives assurance again from evidence.  The
fresh, run-bound :class:`CacheUseReceipt` issued after those checks is an audit
receipt, never portable authority.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .analysis_cache import (
    ANALYSIS_CACHE_ENTRY_SCHEMA,
    AnalysisCache,
    AnalysisCacheKey,
    AnalysisCacheLookupStatus,
    AnalysisOutcome,
    build_analysis_cache_key,
)
from .cache_coordinator import (
    AnalysisCacheCoordinator,
    CacheCoordinationStatus,
)
from ..proof.formal_verification_cache import (
    CacheLookupStatus as ProofLookupStatus,
    CacheRequirements,
    FormalVerificationCache,
    ProofCacheKey,
    TrustAwareProofCache,
    build_proof_cache_key,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    ProofReceipt,
    ResourceBudget,
    canonical_json_bytes,
    content_identity,
)
from ..runtime.artifact_store import (
    ArtifactBlobIntegrityError,
    ArtifactOutcome,
    BoundedPersistenceError,
    BoundedArtifactStore,
    ProjectionReference,
    RetentionClass,
)


REASONING_CACHE_INTERFACE: Final[str] = "ReasoningCacheCoordinator@1"
REASONING_CACHE_COORDINATOR_INTERFACE: Final[str] = REASONING_CACHE_INTERFACE
REASONING_CACHE_VERSION: Final[int] = 1
REASONING_COMPUTATION_KEY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-computation-key@1"
)
REASONING_SOURCE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-source-receipt@1"
)
REASONING_CACHE_USE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-cache-use-receipt@1"
)
REASONING_INVALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-cache-invalidation@1"
)
REASONING_ANALYSIS_INDEX_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reasoning-analysis-index@1"
)

MAX_REASONING_KEY_BYTES: Final[int] = 256 * 1024
MAX_REASONING_RECEIPT_BYTES: Final[int] = 256 * 1024
MAX_REASONING_DEPENDENCIES: Final[int] = 8_192
MAX_REASON_CODES: Final[int] = 64
MAX_ISSUED_USE_RECEIPTS: Final[int] = 4_096

_PRIVATE_FIELD_MARKERS: Final[tuple[str, ...]] = (
    "api_key",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "private",
    "secret",
    "session",
    "token",
    "witness",
)
_TRUST_CLAIM_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "authoritative",
        "cache_authoritative",
        "completion_evidence",
        "is_completion_evidence",
        "trusted",
        "verified",
    }
)


class ReasoningCacheLane(str, Enum):
    """Native persistence lane used by a reasoning computation."""

    ANALYSIS = "analysis"
    PROOF = "proof"
    ARTIFACT = "artifact"


class ReasoningOutcome(str, Enum):
    """A source computation's result, independent of cache availability."""

    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def coerce(cls, value: "ReasoningOutcome | str") -> "ReasoningOutcome":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "complete": cls.SUCCESSFUL,
            "completed": cls.SUCCESSFUL,
            "success": cls.SUCCESSFUL,
            "ok": cls.SUCCESSFUL,
            "error": cls.FAILED,
            "unknown": cls.INCONCLUSIVE,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            raise ReasoningCacheError(
                "unsupported reasoning outcome",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            ) from exc

    @property
    def complete(self) -> bool:
        return self is ReasoningOutcome.SUCCESSFUL


class ReasoningCacheStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"
    STORED = "stored"
    PRODUCED = "produced"
    SHARED = "shared"


ReasoningCacheLookupStatus = ReasoningCacheStatus


class ReasoningCacheReason(str, Enum):
    """Stable fail-closed reason codes for audit and scheduling."""

    EXACT_KEY_HIT = "exact_key_hit"
    CACHE_MISS = "cache_miss"
    CACHE_MISS_NOT_REFUTATION = "cache_miss_not_refutation"
    WRONG_REPOSITORY_FOREST = "wrong_repository_forest"
    WRONG_SCOPE = "wrong_scope"
    WRONG_PARSER = "wrong_parser"
    WRONG_INDEX = "wrong_index"
    WRONG_TRANSLATOR = "wrong_translator"
    WRONG_TOOLCHAIN = "wrong_toolchain"
    WRONG_CAPABILITY = "wrong_capability"
    WRONG_POLICY = "wrong_policy"
    WRONG_IR = "wrong_ir"
    WRONG_CATALOG = "wrong_catalog"
    WRONG_SCHEMA = "wrong_schema"
    WRONG_ASSURANCE = "wrong_required_assurance"
    WRONG_BOUNDS = "wrong_bounds"
    SEMANTIC_KEY_MISMATCH = "semantic_key_mismatch"
    POISONED_ENTRY = "poisoned_cache_entry"
    PARTIAL_ENTRY = "partial_cache_entry"
    FORGED_RECEIPT = "forged_receipt"
    UNDECLARED_DEPENDENCY = "undeclared_dependency"
    PRIVATE_MATERIAL = "private_material"
    CROSS_RUN_REPLAY = "cross_run_replay"
    SOURCE_RECEIPT_MISSING = "source_receipt_missing"
    SOURCE_RECEIPT_CORRUPT = "source_receipt_corrupt"
    ARTIFACT_INTEGRITY_FAILED = "artifact_integrity_failed"
    ARTIFACT_PERSISTENCE_REJECTED = "artifact_persistence_rejected"
    INSUFFICIENT_ASSURANCE = "required_assurance_not_satisfied"
    ASSURANCE_REDERIVATION_FAILED = "assurance_rederivation_failed"
    ASSURANCE_CLAIM_MISMATCH = "assurance_claim_mismatch"
    DEPENDENCY_INVALIDATED = "dependency_invalidated"
    PROOF_CACHE_REJECTED = "proof_cache_rejected"
    NATIVE_CACHE_REJECTED = "native_cache_rejected"


class ReasoningCacheError(RuntimeError):
    """A bounded, reason-coded reasoning cache contract failure."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)


class ReasoningPrivateMaterialError(ReasoningCacheError, ValueError):
    def __init__(self, message: str = "private material is not cacheable") -> None:
        super().__init__(
            message, reason_code=ReasoningCacheReason.PRIVATE_MATERIAL.value
        )


def _normalized_name(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _contains_private_material(value: Any) -> bool:
    """Inspect field names only; never copy or report private values."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = _normalized_name(raw_name)
            segments = tuple(part for part in name.split("_") if part)
            sensitive_segment = any(
                marker in segments
                for marker in (
                    "cookie",
                    "credential",
                    "password",
                    "private",
                    "secret",
                    "witness",
                )
            )
            sensitive_token = (
                name == "token"
                or name.endswith("_token")
                or name
                in {
                    "access_token",
                    "auth_token",
                    "bearer_token",
                    "refresh_token",
                }
            )
            sensitive_session = name == "session" or name.endswith("_session")
            if (
                sensitive_segment
                or sensitive_token
                or sensitive_session
                or "api_key" in name
                or "authorization" in name
                or "bearer" in name
            ):
                return True
            if _contains_private_material(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return any(_contains_private_material(item) for item in value)
    return False


def _canonical_value(value: Any, field_name: str) -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if _contains_private_material(value):
        raise ReasoningPrivateMaterialError(
            f"{field_name} contains private material"
        )
    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError, ContractValidationError, json.JSONDecodeError) as exc:
        raise ReasoningCacheError(
            f"{field_name} must contain canonical JSON values",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        ) from exc


def _identity(value: Any, field_name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ReasoningCacheError(
            f"{field_name} is required",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )
    return _canonical_value(value, field_name)


def _canonical_set(values: Any, field_name: str) -> tuple[Any, ...]:
    if values is None:
        raise ReasoningCacheError(
            f"{field_name} is required; use an empty sequence",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise ReasoningCacheError(
            f"{field_name} must be a sequence",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )
    unique: dict[bytes, Any] = {}
    for item in values:
        normalized = _identity(item, field_name)
        unique[canonical_json_bytes(normalized)] = normalized
    return tuple(unique[key] for key in sorted(unique))


def _identifier(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Mapping):
        text = ""
        for name in (
            field_name,
            f"{field_name}_id",
            "content_id",
            "artifact_id",
            "id",
        ):
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break
    else:
        text = ""
    if not text:
        raise ReasoningCacheError(
            f"{field_name} requires a compact identifier",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )
    if any(char.isspace() for char in text) or "\0" in text:
        raise ReasoningCacheError(
            f"{field_name} identifier is invalid",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )
    return text


def _dependency_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return content_identity(_canonical_value(value, "dependency"))


def _dependency_ids(values: Sequence[Any]) -> tuple[str, ...]:
    if len(values) > MAX_REASONING_DEPENDENCIES:
        raise ReasoningCacheError(
            "dependency bound exceeded",
            reason_code=ReasoningCacheReason.WRONG_BOUNDS.value,
        )
    return tuple(sorted({_dependency_id(item) for item in values}))


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ReasoningComputationKey:
    """The complete semantic preimage for one reasoning computation."""

    operation: Any
    property: Any
    repository_forest: Any
    scope: Any
    premises: tuple[Any, ...]
    assumptions: tuple[Any, ...]
    parser: Any
    index: Any
    translator: Any
    toolchain: Any
    capability: Any
    policy: Any
    ir: Any
    catalog: Any
    required_assurance: AssuranceLevel
    bounds: Any
    dependencies: tuple[Any, ...] = ()
    schema: str = REASONING_COMPUTATION_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REASONING_COMPUTATION_KEY_SCHEMA:
            raise ReasoningCacheError(
                "unsupported reasoning computation-key schema",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        for name in (
            "operation",
            "property",
            "repository_forest",
            "scope",
            "parser",
            "index",
            "translator",
            "toolchain",
            "capability",
            "policy",
            "ir",
            "catalog",
            "bounds",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(
            self, "premises", _canonical_set(self.premises, "premises")
        )
        object.__setattr__(
            self, "assumptions", _canonical_set(self.assumptions, "assumptions")
        )
        object.__setattr__(
            self, "dependencies", _canonical_set(self.dependencies, "dependencies")
        )
        try:
            assurance = AssuranceLevel(self.required_assurance)
        except (TypeError, ValueError) as exc:
            raise ReasoningCacheError(
                "required_assurance is unsupported",
                reason_code=ReasoningCacheReason.WRONG_ASSURANCE.value,
            ) from exc
        object.__setattr__(self, "required_assurance", assurance)
        encoded = canonical_json_bytes(self._content())
        if len(encoded) > MAX_REASONING_KEY_BYTES:
            raise ReasoningCacheError(
                "reasoning computation key exceeds its byte bound",
                reason_code=ReasoningCacheReason.WRONG_BOUNDS.value,
            )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operation": self.operation,
            "property": self.property,
            "repository_forest": self.repository_forest,
            "scope": self.scope,
            "premises": list(self.premises),
            "assumptions": list(self.assumptions),
            "parser": self.parser,
            "index": self.index,
            "translator": self.translator,
            "toolchain": self.toolchain,
            "capability": self.capability,
            "policy": self.policy,
            "ir": self.ir,
            "catalog": self.catalog,
            "required_assurance": self.required_assurance.value,
            "bounds": self.bounds,
            "dependencies": list(self.dependencies),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "key_id": self.key_id}

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self._content())).hexdigest()

    @property
    def key_id(self) -> str:
        return f"reasoning-cache-key:sha256:{self.digest}"

    @property
    def cache_key(self) -> str:
        return self.key_id

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        return _dependency_ids(self.dependencies)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReasoningComputationKey":
        if not isinstance(value, Mapping):
            raise ReasoningCacheError(
                "reasoning computation key must be an object",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        allowed = {
            "schema",
            "key_id",
            "operation",
            "property",
            "repository_forest",
            "scope",
            "premises",
            "assumptions",
            "parser",
            "index",
            "translator",
            "toolchain",
            "capability",
            "policy",
            "ir",
            "catalog",
            "required_assurance",
            "bounds",
            "dependencies",
        }
        if set(value).difference(allowed):
            raise ReasoningCacheError(
                "reasoning computation key contains unknown fields",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        restored = cls(
            schema=str(value.get("schema") or ""),
            operation=value.get("operation"),
            property=value.get("property"),
            repository_forest=value.get("repository_forest"),
            scope=value.get("scope"),
            premises=tuple(value.get("premises") or ()),
            assumptions=tuple(value.get("assumptions") or ()),
            parser=value.get("parser"),
            index=value.get("index"),
            translator=value.get("translator"),
            toolchain=value.get("toolchain"),
            capability=value.get("capability"),
            policy=value.get("policy"),
            ir=value.get("ir"),
            catalog=value.get("catalog"),
            required_assurance=value.get("required_assurance"),
            bounds=value.get("bounds"),
            dependencies=tuple(value.get("dependencies") or ()),
        )
        supplied = value.get("key_id")
        if supplied not in (None, "", restored.key_id):
            raise ReasoningCacheError(
                "reasoning computation-key identity mismatch",
                reason_code=ReasoningCacheReason.POISONED_ENTRY.value,
            )
        return restored

    def to_analysis_cache_key(self) -> AnalysisCacheKey:
        """Project the full identity into the existing analysis-cache key."""

        return build_analysis_cache_key(
            repository_tree_identity=self.repository_forest,
            objective_revision={
                "property": self.property,
                "catalog": self.catalog,
            },
            analyzer_version={
                "parser": self.parser,
                "index": self.index,
                "translator": self.translator,
                "toolchain": self.toolchain,
                "capability": self.capability,
            },
            schema_version=REASONING_ANALYSIS_INDEX_SCHEMA,
            configuration_digest=_sha256(
                {
                    "operation": self.operation,
                    "scope": self.scope,
                    "premises": list(self.premises),
                    "assumptions": list(self.assumptions),
                    "bounds": self.bounds,
                    "dependencies": list(self.dependencies),
                }
            ),
            query_digest=self.key_id,
            policy_digest=_sha256(
                {
                    "policy": self.policy,
                    "ir": self.ir,
                    "catalog": self.catalog,
                    "required_assurance": self.required_assurance.value,
                }
            ),
        )

    def to_proof_cache_key(self) -> ProofCacheKey:
        """Build a native proof key whose wrappers bind every reasoning input.

        The wrapper ``id`` values continue to match the identifiers carried by
        :class:`ProofReceipt`; the additional fields extend the native key
        preimage without changing that receipt contract.
        """

        capability = (
            self.capability if isinstance(self.capability, Mapping) else {}
        )
        solver_value = capability.get("solver", capability)
        kernel_value = capability.get("kernel", capability)
        solver_id = _identifier(
            capability.get("solver_id", solver_value or self.capability),
            "solver",
        )
        kernel_id = _identifier(
            capability.get("kernel_id", kernel_value or self.capability),
            "kernel",
        )
        budget = ResourceBudget.from_dict(self.bounds).to_dict()
        return build_proof_cache_key(
            obligation={
                "id": _identifier(self.property, "property"),
                "operation": self.operation,
                "scope": self.scope,
                "parser": self.parser,
                "index": self.index,
                "assumptions": list(self.assumptions),
                "reasoning_key_id": self.key_id,
            },
            premises=self.premises,
            translator={
                "id": _identifier(self.translator, "translator"),
                "parser": self.parser,
                "index": self.index,
            },
            solver={
                "id": solver_id,
                "capability": self.capability,
            },
            kernel={
                "id": kernel_id,
                "capability": self.capability,
            },
            toolchain={
                "id": _identifier(self.toolchain, "toolchain"),
                "identity": self.toolchain,
            },
            theorem_registry={
                "id": _identifier(self.catalog, "catalog"),
                "catalog": self.catalog,
            },
            policy={
                "id": _identifier(self.policy, "policy"),
                "policy": self.policy,
                "ir": self.ir,
                "required_assurance": self.required_assurance.value,
            },
            resource_budget=budget,
            candidate_tree={
                "id": _identifier(self.repository_forest, "repository_forest"),
                "forest": self.repository_forest,
                "scope": self.scope,
                "dependencies": list(self.dependencies),
            },
        )


# Readable compatibility spellings.
SemanticComputationKey = ReasoningComputationKey
ReasoningCacheKey = ReasoningComputationKey


def build_reasoning_cache_key(
    *,
    operation: Any,
    property: Any = None,
    property_id: Any = None,
    repository_forest: Any = None,
    forest: Any = None,
    scope: Any,
    premises: Sequence[Any],
    assumptions: Sequence[Any],
    parser: Any,
    index: Any,
    translator: Any,
    toolchain: Any,
    capability: Any = None,
    provider_capability: Any = None,
    policy: Any,
    ir: Any = None,
    ir_roots: Any = None,
    catalog: Any,
    required_assurance: AssuranceLevel | str,
    bounds: Any = None,
    resource_bounds: Any = None,
    dependencies: Sequence[Any] = (),
) -> ReasoningComputationKey:
    """Build a key while accepting the plan's common dimension spellings."""

    def choose(name: str, first: Any, second: Any) -> Any:
        if first is not None and second is not None:
            if canonical_json_bytes(_canonical_value(first, name)) != canonical_json_bytes(
                _canonical_value(second, name)
            ):
                raise ReasoningCacheError(
                    f"{name} aliases disagree",
                    reason_code=ReasoningCacheReason.SEMANTIC_KEY_MISMATCH.value,
                )
        return first if first is not None else second

    return ReasoningComputationKey(
        operation=operation,
        property=choose("property", property, property_id),
        repository_forest=choose(
            "repository_forest", repository_forest, forest
        ),
        scope=scope,
        premises=tuple(premises),
        assumptions=tuple(assumptions),
        parser=parser,
        index=index,
        translator=translator,
        toolchain=toolchain,
        capability=choose("capability", capability, provider_capability),
        policy=policy,
        ir=choose("ir", ir, ir_roots),
        catalog=catalog,
        required_assurance=required_assurance,
        bounds=choose("bounds", bounds, resource_bounds),
        dependencies=tuple(dependencies),
    )


build_semantic_computation_key = build_reasoning_cache_key
build_reasoning_computation_key = build_reasoning_cache_key
make_reasoning_cache_key = build_reasoning_cache_key


@dataclass(frozen=True)
class ReasoningSourceReceipt:
    """Immutable source receipt reloaded from CAS for every analysis hit."""

    key: ReasoningComputationKey
    payload: Any
    declared_dependency_ids: tuple[str, ...]
    observed_dependency_ids: tuple[str, ...]
    evidence: tuple[Any, ...]
    claimed_assurance: AssuranceLevel
    outcome: ReasoningOutcome
    producer_run_id: str
    created_at_ms: int
    source_schema: str
    receipt_id: str = ""
    schema: str = REASONING_SOURCE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REASONING_SOURCE_RECEIPT_SCHEMA:
            raise ReasoningCacheError(
                "unsupported reasoning source-receipt schema",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        if not isinstance(self.key, ReasoningComputationKey):
            raise ReasoningCacheError(
                "source receipt requires a typed computation key",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        object.__setattr__(self, "payload", _canonical_value(self.payload, "payload"))
        object.__setattr__(
            self, "evidence", _canonical_set(self.evidence, "evidence")
        )
        declared = tuple(sorted(set(self.declared_dependency_ids)))
        observed = tuple(sorted(set(self.observed_dependency_ids)))
        if declared != self.key.dependency_ids:
            raise ReasoningCacheError(
                "source receipt dependency declaration does not match its key",
                reason_code=ReasoningCacheReason.UNDECLARED_DEPENDENCY.value,
            )
        if not set(observed).issubset(declared):
            raise ReasoningCacheError(
                "source receipt observed an undeclared dependency",
                reason_code=ReasoningCacheReason.UNDECLARED_DEPENDENCY.value,
            )
        if len(declared) > MAX_REASONING_DEPENDENCIES:
            raise ReasoningCacheError(
                "source receipt dependency bound exceeded",
                reason_code=ReasoningCacheReason.WRONG_BOUNDS.value,
            )
        object.__setattr__(self, "declared_dependency_ids", declared)
        object.__setattr__(self, "observed_dependency_ids", observed)
        try:
            object.__setattr__(
                self, "claimed_assurance", AssuranceLevel(self.claimed_assurance)
            )
        except (TypeError, ValueError) as exc:
            raise ReasoningCacheError(
                "claimed_assurance is unsupported",
                reason_code=ReasoningCacheReason.WRONG_ASSURANCE.value,
            ) from exc
        object.__setattr__(self, "outcome", ReasoningOutcome.coerce(self.outcome))
        if (
            not isinstance(self.producer_run_id, str)
            or not self.producer_run_id.strip()
        ):
            raise ReasoningCacheError(
                "producer_run_id is required",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms < 0
        ):
            raise ReasoningCacheError(
                "created_at_ms must be non-negative",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        if not isinstance(self.source_schema, str) or not self.source_schema.strip():
            raise ReasoningCacheError(
                "source_schema is required",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        computed = content_identity(self._unsigned())
        if self.receipt_id and self.receipt_id != computed:
            raise ReasoningCacheError(
                "source receipt content identity mismatch",
                reason_code=ReasoningCacheReason.FORGED_RECEIPT.value,
            )
        object.__setattr__(self, "receipt_id", computed)
        if len(canonical_json_bytes(self.to_dict())) > MAX_REASONING_RECEIPT_BYTES:
            raise ReasoningCacheError(
                "source receipt exceeds its byte bound",
                reason_code=ReasoningCacheReason.WRONG_BOUNDS.value,
            )

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key": self.key.to_dict(),
            "key_id": self.key.key_id,
            "payload": self.payload,
            "declared_dependency_ids": list(self.declared_dependency_ids),
            "observed_dependency_ids": list(self.observed_dependency_ids),
            "evidence": list(self.evidence),
            "claimed_assurance": self.claimed_assurance.value,
            "outcome": self.outcome.value,
            "producer_run_id": self.producer_run_id,
            "created_at_ms": self.created_at_ms,
            "source_schema": self.source_schema,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "receipt_id": self.receipt_id}

    @classmethod
    def create(
        cls,
        key: ReasoningComputationKey,
        payload: Any,
        *,
        producer_run_id: str,
        evidence: Sequence[Any] = (),
        claimed_assurance: AssuranceLevel | str = AssuranceLevel.UNVERIFIED,
        outcome: ReasoningOutcome | str = ReasoningOutcome.SUCCESSFUL,
        observed_dependencies: Sequence[Any] | None = None,
        source_schema: str = "reasoning-result@1",
        created_at_ms: int | None = None,
    ) -> "ReasoningSourceReceipt":
        observed = (
            key.dependencies
            if observed_dependencies is None
            else tuple(observed_dependencies)
        )
        return cls(
            key=key,
            payload=payload,
            declared_dependency_ids=key.dependency_ids,
            observed_dependency_ids=_dependency_ids(tuple(observed)),
            evidence=tuple(evidence),
            claimed_assurance=claimed_assurance,
            outcome=ReasoningOutcome.coerce(outcome),
            producer_run_id=producer_run_id,
            created_at_ms=(
                int(time.time() * 1000)
                if created_at_ms is None
                else created_at_ms
            ),
            source_schema=source_schema,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReasoningSourceReceipt":
        if not isinstance(value, Mapping):
            raise ReasoningCacheError(
                "source receipt must be an object",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        allowed = {
            "schema",
            "key",
            "key_id",
            "payload",
            "declared_dependency_ids",
            "observed_dependency_ids",
            "evidence",
            "claimed_assurance",
            "outcome",
            "producer_run_id",
            "created_at_ms",
            "source_schema",
            "receipt_id",
        }
        unknown = set(value).difference(allowed)
        if unknown or any(_normalized_name(item) in _TRUST_CLAIM_FIELDS for item in value):
            raise ReasoningCacheError(
                "source receipt contains unsupported trust or schema fields",
                reason_code=ReasoningCacheReason.FORGED_RECEIPT.value,
            )
        key = ReasoningComputationKey.from_dict(value.get("key") or {})
        if value.get("key_id") != key.key_id:
            raise ReasoningCacheError(
                "source receipt is bound to a different computation key",
                reason_code=ReasoningCacheReason.POISONED_ENTRY.value,
            )
        return cls(
            schema=str(value.get("schema") or ""),
            key=key,
            payload=value.get("payload"),
            declared_dependency_ids=tuple(
                value.get("declared_dependency_ids") or ()
            ),
            observed_dependency_ids=tuple(
                value.get("observed_dependency_ids") or ()
            ),
            evidence=tuple(value.get("evidence") or ()),
            claimed_assurance=value.get("claimed_assurance"),
            outcome=value.get("outcome"),
            producer_run_id=str(value.get("producer_run_id") or ""),
            created_at_ms=value.get("created_at_ms"),
            source_schema=str(value.get("source_schema") or ""),
            receipt_id=str(value.get("receipt_id") or ""),
        )


@dataclass(frozen=True)
class CacheUseReceipt:
    """Fresh audit evidence for one verified cache use.

    ``_seal`` is deliberately absent from :meth:`to_dict`; deserializing or
    copying a wire receipt cannot reconstruct live coordinator authority.
    """

    lane: ReasoningCacheLane
    key_id: str
    source_receipt_id: str
    source_reference_id: str
    source_digest: str
    current_run_id: str
    producer_run_id: str
    derived_assurance: AssuranceLevel
    issued_at_ms: int
    nonce: str
    receipt_id: str = ""
    schema: str = REASONING_CACHE_USE_RECEIPT_SCHEMA
    _seal: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != REASONING_CACHE_USE_RECEIPT_SCHEMA:
            raise ReasoningCacheError(
                "unsupported cache-use receipt schema",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        object.__setattr__(self, "lane", ReasoningCacheLane(self.lane))
        try:
            object.__setattr__(
                self, "derived_assurance", AssuranceLevel(self.derived_assurance)
            )
        except (TypeError, ValueError) as exc:
            raise ReasoningCacheError(
                "cache-use receipt assurance is unsupported",
                reason_code=ReasoningCacheReason.WRONG_ASSURANCE.value,
            ) from exc
        for name in (
            "key_id",
            "source_receipt_id",
            "source_reference_id",
            "source_digest",
            "current_run_id",
            "producer_run_id",
            "nonce",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ReasoningCacheError(
                    f"cache-use receipt {name} is required",
                    reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
                )
        if (
            isinstance(self.issued_at_ms, bool)
            or not isinstance(self.issued_at_ms, int)
            or self.issued_at_ms < 0
        ):
            raise ReasoningCacheError(
                "cache-use receipt issued_at_ms must be non-negative",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        unsigned = self._unsigned()
        computed = content_identity(unsigned)
        if self.receipt_id and self.receipt_id != computed:
            raise ReasoningCacheError(
                "cache-use receipt identity mismatch",
                reason_code=ReasoningCacheReason.FORGED_RECEIPT.value,
            )
        object.__setattr__(self, "receipt_id", computed)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lane": self.lane.value,
            "key_id": self.key_id,
            "source_receipt_id": self.source_receipt_id,
            "source_reference_id": self.source_reference_id,
            "source_digest": self.source_digest,
            "current_run_id": self.current_run_id,
            "producer_run_id": self.producer_run_id,
            "derived_assurance": self.derived_assurance.value,
            "issued_at_ms": self.issued_at_ms,
            "nonce": self.nonce,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheUseReceipt":
        """Decode for diagnostics; the result intentionally has no live seal."""

        return cls(
            schema=str(value.get("schema") or ""),
            lane=value.get("lane"),
            key_id=str(value.get("key_id") or ""),
            source_receipt_id=str(value.get("source_receipt_id") or ""),
            source_reference_id=str(value.get("source_reference_id") or ""),
            source_digest=str(value.get("source_digest") or ""),
            current_run_id=str(value.get("current_run_id") or ""),
            producer_run_id=str(value.get("producer_run_id") or ""),
            derived_assurance=value.get("derived_assurance"),
            issued_at_ms=value.get("issued_at_ms"),
            nonce=str(value.get("nonce") or ""),
            receipt_id=str(value.get("receipt_id") or ""),
        )


@dataclass(frozen=True)
class CacheUseVerification:
    valid: bool
    reason_codes: tuple[str, ...] = ()

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    def __bool__(self) -> bool:
        return self.valid


@dataclass(frozen=True)
class ReasoningCacheResult:
    status: ReasoningCacheStatus
    key: ReasoningComputationKey
    payload: Any = None
    source_receipt: ReasoningSourceReceipt | None = None
    assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    reason_codes: tuple[str, ...] = ()
    use_receipt: CacheUseReceipt | None = None
    native_result: Any = None

    @property
    def hit(self) -> bool:
        return self.status in {
            ReasoningCacheStatus.HIT,
            ReasoningCacheStatus.PRODUCED,
            ReasoningCacheStatus.SHARED,
        }

    @property
    def cache_hit(self) -> bool:
        return self.status is ReasoningCacheStatus.HIT

    @property
    def produced(self) -> bool:
        return self.status is ReasoningCacheStatus.PRODUCED

    @property
    def shared(self) -> bool:
        return self.status is ReasoningCacheStatus.SHARED

    @property
    def miss(self) -> bool:
        return self.status is ReasoningCacheStatus.MISS

    @property
    def rejected(self) -> bool:
        return self.status is ReasoningCacheStatus.REJECTED

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def is_completion_evidence(self) -> bool:
        return bool(
            self.hit
            and self.use_receipt is not None
            and self.assurance.satisfies(self.key.required_assurance)
        )

    @property
    def completion_evidence(self) -> bool:
        return self.is_completion_evidence

    @property
    def is_refutation(self) -> bool:
        # Absence or rejection of memoized evidence says nothing about truth.
        return False

    @property
    def refuted(self) -> bool:
        return False


@dataclass(frozen=True)
class ReasoningCacheStoreResult:
    stored: bool
    key: ReasoningComputationKey
    source_receipt: ReasoningSourceReceipt | None = None
    source_reference: ProjectionReference | None = None
    reason_codes: tuple[str, ...] = ()
    native_result: Any = None

    def __bool__(self) -> bool:
        return self.stored

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True)
class ReasoningInvalidationReceipt:
    changed_dependency_ids: tuple[str, ...]
    invalidated_analysis_key_ids: tuple[str, ...]
    invalidated_proof_key_ids: tuple[str, ...]
    retained_artifact_ids: tuple[str, ...]
    run_id: str
    created_at_ms: int
    schema: str = REASONING_INVALIDATION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "changed_dependency_ids": list(self.changed_dependency_ids),
            "invalidated_analysis_key_ids": list(
                self.invalidated_analysis_key_ids
            ),
            "invalidated_proof_key_ids": list(self.invalidated_proof_key_ids),
            "retained_artifact_ids": list(self.retained_artifact_ids),
            "run_id": self.run_id,
            "created_at_ms": self.created_at_ms,
            "receipt_id": self.receipt_id,
        }

    @property
    def receipt_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema,
                "changed_dependency_ids": list(self.changed_dependency_ids),
                "invalidated_analysis_key_ids": list(
                    self.invalidated_analysis_key_ids
                ),
                "invalidated_proof_key_ids": list(
                    self.invalidated_proof_key_ids
                ),
                "retained_artifact_ids": list(self.retained_artifact_ids),
                "run_id": self.run_id,
                "created_at_ms": self.created_at_ms,
            }
        )


AssuranceDeriver = Callable[[ReasoningSourceReceipt], AssuranceLevel | str]


class ReasoningCacheCoordinator:
    """Coordinate exact analysis/proof/CAS reuse without upgrading trust."""

    interface: Final[str] = REASONING_CACHE_INTERFACE

    def __init__(
        self,
        analysis_cache: AnalysisCache | str | Path,
        proof_cache: FormalVerificationCache | str | Path,
        artifact_store: BoundedArtifactStore | str | Path,
        *,
        run_id: str,
        assurance_deriver: AssuranceDeriver | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required")
        self.analysis_cache = (
            analysis_cache
            if isinstance(analysis_cache, AnalysisCache)
            else AnalysisCache(analysis_cache)
        )
        self.proof_cache: FormalVerificationCache = (
            proof_cache
            if isinstance(proof_cache, FormalVerificationCache)
            else FormalVerificationCache(proof_cache)
        )
        if not isinstance(self.proof_cache, TrustAwareProofCache):
            raise TypeError(
                "proof_cache must be the repository TrustAwareProofCache"
            )
        self.artifact_store = (
            artifact_store
            if isinstance(artifact_store, BoundedArtifactStore)
            else BoundedArtifactStore(artifact_store)
        )
        self.run_id = run_id.strip()
        self.assurance_deriver = assurance_deriver
        self._clock = clock
        self._analysis_flights = AnalysisCacheCoordinator(self.analysis_cache)
        self._use_seal = object()
        self._issued_use_receipts: dict[str, str] = {}
        self._issued_guard = threading.Lock()
        self._proof_bindings: dict[
            str, tuple[ProofCacheKey, tuple[str, ...]]
        ] = {}

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    @staticmethod
    def _coerce_key(
        key: ReasoningComputationKey | Mapping[str, Any],
    ) -> ReasoningComputationKey:
        return (
            key
            if isinstance(key, ReasoningComputationKey)
            else ReasoningComputationKey.from_dict(key)
        )

    def _derive_assurance(
        self, receipt: ReasoningSourceReceipt
    ) -> AssuranceLevel:
        if self.assurance_deriver is not None:
            try:
                return AssuranceLevel(self.assurance_deriver(receipt))
            except BaseException as exc:
                raise ReasoningCacheError(
                    "source assurance could not be re-derived",
                    reason_code=(
                        ReasoningCacheReason.ASSURANCE_REDERIVATION_FAILED.value
                    ),
                ) from exc

        # Proof receipts carry typed evidence with their own derivation logic.
        if isinstance(receipt.payload, Mapping):
            try:
                proof = ProofReceipt.from_dict(receipt.payload)
            except (TypeError, ValueError, ContractValidationError):
                proof = None
            if proof is not None:
                return proof.authoritative_assurance

        rank = AssuranceLevel.UNVERIFIED
        for raw in receipt.evidence:
            if not isinstance(raw, Mapping):
                continue
            verdict = _normalized_name(raw.get("verdict", raw.get("status", "")))
            if verdict not in {"accepted", "current", "passed", "successful"}:
                continue
            if raw.get("simulated") is True:
                continue
            kind = _normalized_name(raw.get("kind", ""))
            authority = _normalized_name(raw.get("authority", ""))
            independent = raw.get("independent") is True
            candidate = AssuranceLevel.CANDIDATE
            if "kernel" in kind and "kernel" in authority and independent:
                candidate = AssuranceLevel.KERNEL_VERIFIED
            elif "solver" in kind and authority in {
                "solver",
                "trusted_solver",
            }:
                candidate = AssuranceLevel.SOLVER_CHECKED
            if candidate.rank > rank.rank:
                rank = candidate
        return rank

    @staticmethod
    def _validate_source_for_key(
        receipt: ReasoningSourceReceipt,
        key: ReasoningComputationKey,
    ) -> None:
        if receipt.key.key_id != key.key_id or receipt.key.to_dict() != key.to_dict():
            raise ReasoningCacheError(
                "source receipt is bound to a different semantic preimage",
                reason_code=ReasoningCacheReason.POISONED_ENTRY.value,
            )
        if receipt.declared_dependency_ids != key.dependency_ids:
            raise ReasoningCacheError(
                "source receipt has an undeclared dependency",
                reason_code=ReasoningCacheReason.UNDECLARED_DEPENDENCY.value,
            )
        if not set(receipt.observed_dependency_ids).issubset(
            receipt.declared_dependency_ids
        ):
            raise ReasoningCacheError(
                "source receipt observed an undeclared dependency",
                reason_code=ReasoningCacheReason.UNDECLARED_DEPENDENCY.value,
            )

    def _prepare_source(
        self,
        key: ReasoningComputationKey,
        value: ReasoningSourceReceipt | Mapping[str, Any],
        *,
        evidence: Sequence[Any] = (),
        claimed_assurance: AssuranceLevel | str | None = None,
        outcome: ReasoningOutcome | str = ReasoningOutcome.SUCCESSFUL,
        observed_dependencies: Sequence[Any] | None = None,
        source_schema: str = "reasoning-result@1",
    ) -> tuple[ReasoningSourceReceipt, AssuranceLevel]:
        if isinstance(value, ReasoningSourceReceipt):
            receipt = value
        elif (
            isinstance(value, Mapping)
            and value.get("schema") == REASONING_SOURCE_RECEIPT_SCHEMA
        ):
            receipt = ReasoningSourceReceipt.from_dict(value)
        elif isinstance(value, Mapping):
            provisional_assurance = (
                AssuranceLevel.UNVERIFIED
                if claimed_assurance is None
                else claimed_assurance
            )
            receipt = ReasoningSourceReceipt.create(
                key,
                value,
                producer_run_id=self.run_id,
                evidence=evidence,
                claimed_assurance=provisional_assurance,
                outcome=outcome,
                observed_dependencies=observed_dependencies,
                source_schema=source_schema,
                created_at_ms=self._now_ms(),
            )
        else:
            raise ReasoningCacheError(
                "producer must return a source receipt or mapping",
                reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
            )
        self._validate_source_for_key(receipt, key)
        derived = self._derive_assurance(receipt)
        if receipt.claimed_assurance.rank > derived.rank:
            raise ReasoningCacheError(
                "source receipt claims more assurance than its evidence",
                reason_code=ReasoningCacheReason.ASSURANCE_CLAIM_MISMATCH.value,
            )
        return receipt, derived

    @staticmethod
    def _artifact_index_reference(
        reference: ProjectionReference,
    ) -> dict[str, Any]:
        return {
            "artifact_id": reference.artifact_id,
            "digest": reference.digest,
            "size_bytes": reference.size_bytes,
            "schema": reference.schema,
            "kind": "reasoning_source_receipt",
        }

    def _store_source_artifact(
        self, receipt: ReasoningSourceReceipt
    ) -> ProjectionReference:
        artifact_outcome = (
            ArtifactOutcome.SUCCESSFUL
            if receipt.outcome.complete
            else ArtifactOutcome.INCONCLUSIVE
        )
        return self.artifact_store.store_receipt(
            receipt.to_dict(),
            projection_kind="reasoning_source_receipt",
            retention_class=(
                RetentionClass.AUTHORITATIVE
                if receipt.outcome.complete
                else RetentionClass.NEGATIVE
            ),
            outcome=artifact_outcome,
        )

    @staticmethod
    def _analysis_index_receipt(
        key: ReasoningComputationKey,
        source: ReasoningSourceReceipt,
        reference: ProjectionReference,
    ) -> dict[str, Any]:
        status = (
            AnalysisOutcome.SUCCESSFUL.value
            if source.outcome.complete
            else AnalysisOutcome.PARTIAL.value
        )
        return {
            "status": status,
            "receipt_id": source.receipt_id,
            "summary": {
                "semantic_key_id": key.key_id,
                "source_receipt_id": source.receipt_id,
                "source_artifact_id": reference.artifact_id,
            },
            "artifact_refs": [
                ReasoningCacheCoordinator._artifact_index_reference(reference)
            ],
            "metadata": {
                "schema": REASONING_ANALYSIS_INDEX_SCHEMA,
                "native_entry_schema": ANALYSIS_CACHE_ENTRY_SCHEMA,
                "semantic_key": key.to_dict(),
                "source_receipt_id": source.receipt_id,
                "source_receipt_reference": reference.to_dict(),
                "declared_dependency_ids": list(
                    source.declared_dependency_ids
                ),
                "observed_dependency_ids": list(
                    source.observed_dependency_ids
                ),
                "source_schema": source.source_schema,
            },
        }

    def put_analysis(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
        value: ReasoningSourceReceipt | Mapping[str, Any],
        *,
        evidence: Sequence[Any] = (),
        claimed_assurance: AssuranceLevel | str | None = None,
        assurance: AssuranceLevel | str | None = None,
        outcome: ReasoningOutcome | str = ReasoningOutcome.SUCCESSFUL,
        observed_dependencies: Sequence[Any] | None = None,
        source_schema: str = "reasoning-result@1",
        ttl_seconds: int | None = None,
    ) -> ReasoningCacheStoreResult:
        semantic_key = self._coerce_key(key)
        if (
            claimed_assurance is not None
            and assurance is not None
            and AssuranceLevel(claimed_assurance) is not AssuranceLevel(assurance)
        ):
            raise ReasoningCacheError(
                "assurance aliases disagree",
                reason_code=ReasoningCacheReason.ASSURANCE_CLAIM_MISMATCH.value,
            )
        selected_assurance = (
            claimed_assurance if claimed_assurance is not None else assurance
        )
        try:
            source, _derived = self._prepare_source(
                semantic_key,
                value,
                evidence=evidence,
                claimed_assurance=selected_assurance,
                outcome=outcome,
                observed_dependencies=observed_dependencies,
                source_schema=source_schema,
            )
            reference = self._store_source_artifact(source)
            index_receipt = self._analysis_index_receipt(
                semantic_key, source, reference
            )
            native = self.analysis_cache.put(
                semantic_key.to_analysis_cache_key(),
                index_receipt,
                ttl_seconds=ttl_seconds,
            )
        except ReasoningCacheError as exc:
            return ReasoningCacheStoreResult(
                False, semantic_key, reason_codes=(exc.reason_code,)
            )
        except ArtifactBlobIntegrityError:
            return ReasoningCacheStoreResult(
                False,
                semantic_key,
                reason_codes=(
                    ReasoningCacheReason.ARTIFACT_INTEGRITY_FAILED.value,
                ),
            )
        except BoundedPersistenceError:
            return ReasoningCacheStoreResult(
                False,
                semantic_key,
                reason_codes=(
                    ReasoningCacheReason.ARTIFACT_PERSISTENCE_REJECTED.value,
                ),
            )
        if not native.stored:
            return ReasoningCacheStoreResult(
                False,
                semantic_key,
                source_receipt=source,
                source_reference=reference,
                reason_codes=(
                    ReasoningCacheReason.NATIVE_CACHE_REJECTED.value,
                    *native.reason_codes,
                ),
                native_result=native,
            )
        return ReasoningCacheStoreResult(
            True,
            semantic_key,
            source_receipt=source,
            source_reference=reference,
            native_result=native,
        )

    store_analysis = put_analysis

    @staticmethod
    def _semantic_difference_reasons(
        expected: ReasoningComputationKey,
        actual: ReasoningComputationKey,
    ) -> tuple[str, ...]:
        mapping = {
            "repository_forest": ReasoningCacheReason.WRONG_REPOSITORY_FOREST,
            "scope": ReasoningCacheReason.WRONG_SCOPE,
            "parser": ReasoningCacheReason.WRONG_PARSER,
            "index": ReasoningCacheReason.WRONG_INDEX,
            "translator": ReasoningCacheReason.WRONG_TRANSLATOR,
            "toolchain": ReasoningCacheReason.WRONG_TOOLCHAIN,
            "capability": ReasoningCacheReason.WRONG_CAPABILITY,
            "policy": ReasoningCacheReason.WRONG_POLICY,
            "ir": ReasoningCacheReason.WRONG_IR,
            "catalog": ReasoningCacheReason.WRONG_CATALOG,
            "required_assurance": ReasoningCacheReason.WRONG_ASSURANCE,
            "bounds": ReasoningCacheReason.WRONG_BOUNDS,
        }
        reasons = [
            reason.value
            for name, reason in mapping.items()
            if getattr(expected, name) != getattr(actual, name)
        ]
        for name in (
            "operation",
            "property",
            "premises",
            "assumptions",
            "dependencies",
        ):
            if getattr(expected, name) != getattr(actual, name):
                reasons.append(ReasoningCacheReason.SEMANTIC_KEY_MISMATCH.value)
                break
        return tuple(dict.fromkeys(reasons))[:MAX_REASON_CODES]

    def _nearby_semantic_reasons(
        self, key: ReasoningComputationKey
    ) -> tuple[str, ...]:
        """Return bounded diagnostics from native entries, never evidence."""

        candidates: list[tuple[int, str, tuple[str, ...]]] = []
        for path in self.analysis_cache._entry_paths():
            try:
                entry = self.analysis_cache._read_path(path)
                metadata = entry.receipt.get("metadata")
                if not isinstance(metadata, Mapping):
                    continue
                actual = ReasoningComputationKey.from_dict(
                    metadata.get("semantic_key") or {}
                )
            except (OSError, TypeError, ValueError, ReasoningCacheError):
                continue
            if (
                actual.operation != key.operation
                or actual.property != key.property
            ):
                continue
            reasons = self._semantic_difference_reasons(key, actual)
            candidates.append((len(reasons), actual.key_id, reasons))
        return min(candidates)[2] if candidates else ()

    def _remove_analysis_index(self, key: ReasoningComputationKey) -> bool:
        try:
            key.to_analysis_cache_key()
            self.analysis_cache.entry_path(
                key.to_analysis_cache_key()
            ).unlink()
            return True
        except FileNotFoundError:
            return False

    def _issue_use_receipt(
        self,
        *,
        lane: ReasoningCacheLane,
        key: ReasoningComputationKey,
        source_receipt_id: str,
        source_reference_id: str,
        source_digest: str,
        producer_run_id: str,
        assurance: AssuranceLevel,
    ) -> CacheUseReceipt:
        receipt = CacheUseReceipt(
            lane=lane,
            key_id=key.key_id,
            source_receipt_id=source_receipt_id,
            source_reference_id=source_reference_id,
            source_digest=source_digest,
            current_run_id=self.run_id,
            producer_run_id=producer_run_id,
            derived_assurance=assurance,
            issued_at_ms=self._now_ms(),
            nonce=secrets.token_hex(16),
            _seal=self._use_seal,
        )
        with self._issued_guard:
            self._issued_use_receipts[receipt.receipt_id] = key.key_id
            while len(self._issued_use_receipts) > MAX_ISSUED_USE_RECEIPTS:
                self._issued_use_receipts.pop(next(iter(self._issued_use_receipts)))
        return receipt

    def lookup_analysis(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
    ) -> ReasoningCacheResult:
        semantic_key = self._coerce_key(key)
        native_key = semantic_key.to_analysis_cache_key()
        native = self.analysis_cache.lookup(
            native_key, require_completion_evidence=True
        )
        if native.status is AnalysisCacheLookupStatus.MISS:
            nearby = self._nearby_semantic_reasons(semantic_key)
            if nearby:
                return ReasoningCacheResult(
                    ReasoningCacheStatus.REJECTED,
                    semantic_key,
                    reason_codes=nearby,
                    native_result=native,
                )
            return ReasoningCacheResult(
                ReasoningCacheStatus.MISS,
                semantic_key,
                reason_codes=(
                    ReasoningCacheReason.CACHE_MISS.value,
                    ReasoningCacheReason.CACHE_MISS_NOT_REFUTATION.value,
                ),
                native_result=native,
            )
        if not native.hit or native.entry is None:
            nearby = self._nearby_semantic_reasons(semantic_key)
            exact_path_exists = self.analysis_cache.entry_path(native_key).exists()
            if native.entry is None and not nearby and not exact_path_exists:
                return ReasoningCacheResult(
                    ReasoningCacheStatus.MISS,
                    semantic_key,
                    reason_codes=(
                        ReasoningCacheReason.CACHE_MISS.value,
                        ReasoningCacheReason.CACHE_MISS_NOT_REFUTATION.value,
                    ),
                    native_result=native,
                )
            native_reasons = tuple(native.reason_codes)
            if (
                native.entry is not None
                and not native.entry.status.is_completion_evidence
            ):
                native_reasons = (
                    ReasoningCacheReason.PARTIAL_ENTRY.value,
                    *native_reasons,
                )
            elif "corrupt_entry" in native_reasons:
                native_reasons = (
                    ReasoningCacheReason.POISONED_ENTRY.value,
                    *native_reasons,
                )
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=nearby
                or native_reasons
                or (ReasoningCacheReason.NATIVE_CACHE_REJECTED.value,),
                native_result=native,
            )
        try:
            receipt = native.entry.receipt
            metadata = receipt.get("metadata")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("schema") != REASONING_ANALYSIS_INDEX_SCHEMA
                or metadata.get("native_entry_schema")
                != ANALYSIS_CACHE_ENTRY_SCHEMA
            ):
                raise ReasoningCacheError(
                    "analysis index schema is invalid",
                    reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
                )
            stored_key = ReasoningComputationKey.from_dict(
                metadata.get("semantic_key") or {}
            )
            if stored_key.to_dict() != semantic_key.to_dict():
                differences = self._semantic_difference_reasons(
                    semantic_key, stored_key
                )
                raise ReasoningCacheError(
                    "analysis index semantic preimage differs",
                    reason_code=(
                        differences[0]
                        if differences
                        else ReasoningCacheReason.POISONED_ENTRY.value
                    ),
                )
            reference = ProjectionReference.from_dict(
                metadata.get("source_receipt_reference") or {}
            )
            source_payload = self.artifact_store.read_projection(reference)
            source = ReasoningSourceReceipt.from_dict(source_payload)
            self._validate_source_for_key(source, semantic_key)
            if (
                metadata.get("source_receipt_id") != source.receipt_id
                or receipt.get("receipt_id") != source.receipt_id
                or reference.artifact_id
                != (receipt.get("summary") or {}).get("source_artifact_id")
            ):
                raise ReasoningCacheError(
                    "analysis index/source receipt binding is poisoned",
                    reason_code=ReasoningCacheReason.POISONED_ENTRY.value,
                )
            if not source.outcome.complete:
                raise ReasoningCacheError(
                    "partial source receipt cannot satisfy a cache hit",
                    reason_code=ReasoningCacheReason.PARTIAL_ENTRY.value,
                )
            derived = self._derive_assurance(source)
            if source.claimed_assurance.rank > derived.rank:
                raise ReasoningCacheError(
                    "source receipt assurance claim was not reproduced",
                    reason_code=(
                        ReasoningCacheReason.ASSURANCE_CLAIM_MISMATCH.value
                    ),
                )
            if not derived.satisfies(semantic_key.required_assurance):
                raise ReasoningCacheError(
                    "source receipt does not meet required assurance",
                    reason_code=(
                        ReasoningCacheReason.INSUFFICIENT_ASSURANCE.value
                    ),
                )
        except ReasoningCacheError as exc:
            self._remove_analysis_index(semantic_key)
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=(exc.reason_code,),
                native_result=native,
            )
        except ArtifactBlobIntegrityError:
            self._remove_analysis_index(semantic_key)
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=(
                    ReasoningCacheReason.ARTIFACT_INTEGRITY_FAILED.value,
                ),
                native_result=native,
            )
        except (TypeError, ValueError, KeyError, ContractValidationError):
            self._remove_analysis_index(semantic_key)
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=(ReasoningCacheReason.POISONED_ENTRY.value,),
                native_result=native,
            )
        use = self._issue_use_receipt(
            lane=ReasoningCacheLane.ANALYSIS,
            key=semantic_key,
            source_receipt_id=source.receipt_id,
            source_reference_id=reference.artifact_id,
            source_digest=reference.digest,
            producer_run_id=source.producer_run_id,
            assurance=derived,
        )
        return ReasoningCacheResult(
            ReasoningCacheStatus.HIT,
            semantic_key,
            payload=source.payload,
            source_receipt=source,
            assurance=derived,
            reason_codes=(ReasoningCacheReason.EXACT_KEY_HIT.value,),
            use_receipt=use,
            native_result=native,
        )

    get_analysis = lookup_analysis

    def get_or_compute_analysis(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
        producer: Callable[[], ReasoningSourceReceipt | Mapping[str, Any]],
        *,
        evidence: Sequence[Any] = (),
        claimed_assurance: AssuranceLevel | str | None = None,
        assurance: AssuranceLevel | str | None = None,
        observed_dependencies: Sequence[Any] | None = None,
        source_schema: str = "reasoning-result@1",
        ttl_seconds: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ReasoningCacheResult:
        if not callable(producer):
            raise ValueError("producer must be callable")
        semantic_key = self._coerce_key(key)
        current = self.lookup_analysis(semantic_key)
        if current.hit:
            return current
        if current.rejected:
            self._remove_analysis_index(semantic_key)
        selected_assurance = (
            claimed_assurance if claimed_assurance is not None else assurance
        )

        def produce_index() -> Mapping[str, Any]:
            source, _derived = self._prepare_source(
                semantic_key,
                producer(),
                evidence=evidence,
                claimed_assurance=selected_assurance,
                observed_dependencies=observed_dependencies,
                source_schema=source_schema,
            )
            reference = self._store_source_artifact(source)
            return self._analysis_index_receipt(
                semantic_key, source, reference
            )

        native = self._analysis_flights.get_or_compute(
            semantic_key.to_analysis_cache_key(),
            produce_index,
            ttl_seconds=ttl_seconds,
            wait_timeout_seconds=timeout_seconds,
        )
        result = self.lookup_analysis(semantic_key)
        if not result.hit:
            return replace(result, native_result=native)
        status = {
            CacheCoordinationStatus.PRODUCED: ReasoningCacheStatus.PRODUCED,
            CacheCoordinationStatus.SHARED: ReasoningCacheStatus.SHARED,
            CacheCoordinationStatus.CACHE_HIT: ReasoningCacheStatus.HIT,
        }[native.status]
        return replace(result, status=status, native_result=native)

    coordinate_analysis = get_or_compute_analysis
    compute_analysis = get_or_compute_analysis

    def _coerce_proof_key(
        self,
        semantic_key: ReasoningComputationKey,
        proof_key: ProofCacheKey | Mapping[str, Any] | None,
    ) -> ProofCacheKey:
        expected = semantic_key.to_proof_cache_key()
        supplied = (
            expected
            if proof_key is None
            else (
                proof_key
                if isinstance(proof_key, ProofCacheKey)
                else ProofCacheKey.from_dict(proof_key)
            )
        )
        if supplied.key_id != expected.key_id:
            raise ReasoningCacheError(
                "native proof key does not bind the complete reasoning key",
                reason_code=ReasoningCacheReason.SEMANTIC_KEY_MISMATCH.value,
            )
        self._proof_bindings[semantic_key.key_id] = (
            supplied,
            semantic_key.dependency_ids,
        )
        return supplied

    def lookup_proof(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
        *,
        proof_key: ProofCacheKey | Mapping[str, Any] | None = None,
        requirements: CacheRequirements | None = None,
    ) -> ReasoningCacheResult:
        semantic_key = self._coerce_key(key)
        try:
            native_key = self._coerce_proof_key(semantic_key, proof_key)
        except ReasoningCacheError as exc:
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=(exc.reason_code,),
            )
        requested = requirements or CacheRequirements(
            required_assurance=semantic_key.required_assurance
        )
        native = self.proof_cache.lookup(native_key, requirements=requested)
        if native.status is ProofLookupStatus.MISS:
            return ReasoningCacheResult(
                ReasoningCacheStatus.MISS,
                semantic_key,
                reason_codes=(
                    ReasoningCacheReason.CACHE_MISS.value,
                    ReasoningCacheReason.CACHE_MISS_NOT_REFUTATION.value,
                ),
                native_result=native,
            )
        if not native.hit or native.receipt is None or native.entry is None:
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=tuple(native.reason_codes)
                or (ReasoningCacheReason.PROOF_CACHE_REJECTED.value,),
                native_result=native,
            )
        assurance = native.authoritative_assurance
        if not assurance.satisfies(semantic_key.required_assurance):
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=(
                    ReasoningCacheReason.INSUFFICIENT_ASSURANCE.value,
                ),
                native_result=native,
            )
        use = self._issue_use_receipt(
            lane=ReasoningCacheLane.PROOF,
            key=semantic_key,
            source_receipt_id=native.receipt.receipt_id,
            source_reference_id=native_key.key_id,
            source_digest=native.entry.entry_digest,
            producer_run_id=self.run_id,
            assurance=assurance,
        )
        return ReasoningCacheResult(
            ReasoningCacheStatus.HIT,
            semantic_key,
            payload=native.receipt.to_dict(),
            assurance=assurance,
            reason_codes=(ReasoningCacheReason.EXACT_KEY_HIT.value,),
            use_receipt=use,
            native_result=native,
        )

    get_proof = lookup_proof

    def get_or_compute_proof(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
        producer: Callable[[], ProofReceipt | Mapping[str, Any]],
        *,
        proof_key: ProofCacheKey | Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
        requirements: CacheRequirements | None = None,
        **single_flight_options: Any,
    ) -> ReasoningCacheResult:
        semantic_key = self._coerce_key(key)
        try:
            native_key = self._coerce_proof_key(semantic_key, proof_key)
        except ReasoningCacheError as exc:
            return ReasoningCacheResult(
                ReasoningCacheStatus.REJECTED,
                semantic_key,
                reason_codes=(exc.reason_code,),
            )
        current = self.lookup_proof(
            semantic_key, proof_key=native_key, requirements=requirements
        )
        if current.hit:
            return current
        produced_here = False

        def execute() -> Mapping[str, Any]:
            nonlocal produced_here
            produced_here = True
            raw = producer()
            receipt = (
                raw
                if isinstance(raw, ProofReceipt)
                else ProofReceipt.from_dict(raw)
            )
            stored = self.proof_cache.put(
                native_key, receipt, ttl_seconds=ttl_seconds
            )
            if not stored.stored:
                raise ReasoningCacheError(
                    "proof producer result was rejected by TrustAwareProofCache",
                    reason_code=(
                        stored.reason_code
                        or ReasoningCacheReason.PROOF_CACHE_REJECTED.value
                    ),
                )
            return {
                "proof_key_id": native_key.key_id,
                "receipt_id": receipt.receipt_id,
            }

        self.proof_cache.single_flight(
            native_key, execute, **single_flight_options
        )
        result = self.lookup_proof(
            semantic_key, proof_key=native_key, requirements=requirements
        )
        if not result.hit:
            return result
        return replace(
            result,
            status=(
                ReasoningCacheStatus.PRODUCED
                if produced_here
                else ReasoningCacheStatus.SHARED
            ),
        )

    coordinate_proof = get_or_compute_proof

    def lookup(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
        *,
        lane: ReasoningCacheLane | str = ReasoningCacheLane.ANALYSIS,
        **options: Any,
    ) -> ReasoningCacheResult:
        """Dispatch a lookup while retaining each native cache's authority."""

        selected = ReasoningCacheLane(lane)
        if selected is ReasoningCacheLane.ANALYSIS:
            if options:
                raise TypeError(
                    "analysis lookup does not accept proof-cache options"
                )
            return self.lookup_analysis(key)
        if selected is ReasoningCacheLane.PROOF:
            return self.lookup_proof(key, **options)
        raise ReasoningCacheError(
            "artifact-only references require a semantic analysis index",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )

    get = lookup

    def get_or_compute(
        self,
        key: ReasoningComputationKey | Mapping[str, Any],
        producer: Callable[[], Any],
        *,
        lane: ReasoningCacheLane | str = ReasoningCacheLane.ANALYSIS,
        **options: Any,
    ) -> ReasoningCacheResult:
        """Dispatch coordinated work to the existing native single flight."""

        selected = ReasoningCacheLane(lane)
        if selected is ReasoningCacheLane.ANALYSIS:
            return self.get_or_compute_analysis(key, producer, **options)
        if selected is ReasoningCacheLane.PROOF:
            return self.get_or_compute_proof(key, producer, **options)
        raise ReasoningCacheError(
            "artifact-only computations must use an analysis or proof index",
            reason_code=ReasoningCacheReason.WRONG_SCHEMA.value,
        )

    coordinate = get_or_compute

    def verify_use_receipt(
        self,
        receipt: CacheUseReceipt | Mapping[str, Any],
        key: ReasoningComputationKey | Mapping[str, Any],
        *,
        current_run_id: str | None = None,
    ) -> CacheUseVerification:
        semantic_key = self._coerce_key(key)
        try:
            typed = (
                receipt
                if isinstance(receipt, CacheUseReceipt)
                else CacheUseReceipt.from_dict(receipt)
            )
        except (TypeError, ValueError, ReasoningCacheError):
            return CacheUseVerification(
                False, (ReasoningCacheReason.FORGED_RECEIPT.value,)
            )
        expected_run = self.run_id if current_run_id is None else current_run_id
        if typed.current_run_id != expected_run or expected_run != self.run_id:
            return CacheUseVerification(
                False, (ReasoningCacheReason.CROSS_RUN_REPLAY.value,)
            )
        if typed.key_id != semantic_key.key_id:
            return CacheUseVerification(
                False, (ReasoningCacheReason.SEMANTIC_KEY_MISMATCH.value,)
            )
        with self._issued_guard:
            known_key = self._issued_use_receipts.get(typed.receipt_id)
        if typed._seal is not self._use_seal or known_key != semantic_key.key_id:
            return CacheUseVerification(
                False, (ReasoningCacheReason.FORGED_RECEIPT.value,)
            )
        if typed.lane is ReasoningCacheLane.ANALYSIS:
            current = self.lookup_analysis(semantic_key)
        elif typed.lane is ReasoningCacheLane.PROOF:
            current = self.lookup_proof(semantic_key)
        else:
            return CacheUseVerification(
                False, (ReasoningCacheReason.WRONG_SCHEMA.value,)
            )
        if (
            not current.hit
            or current.use_receipt is None
            or current.use_receipt.source_receipt_id
            != typed.source_receipt_id
            or current.use_receipt.source_reference_id
            != typed.source_reference_id
            or current.use_receipt.source_digest != typed.source_digest
            or current.assurance is not typed.derived_assurance
        ):
            return CacheUseVerification(
                False, (ReasoningCacheReason.SOURCE_RECEIPT_CORRUPT.value,)
            )
        return CacheUseVerification(True)

    verify_cache_use_receipt = verify_use_receipt

    def invalidate_dependencies(
        self, changed_dependencies: Sequence[Any]
    ) -> ReasoningInvalidationReceipt:
        changed = set(_dependency_ids(tuple(changed_dependencies)))
        invalidated_analysis: list[str] = []
        invalidated_proof: list[str] = []
        retained_artifacts: list[str] = []

        for path in self.analysis_cache._entry_paths():
            try:
                entry = self.analysis_cache._read_path(path)
                metadata = entry.receipt.get("metadata")
                if not isinstance(metadata, Mapping):
                    continue
                key = ReasoningComputationKey.from_dict(
                    metadata.get("semantic_key") or {}
                )
                dependencies = set(
                    metadata.get("declared_dependency_ids") or ()
                )
                reference = ProjectionReference.from_dict(
                    metadata.get("source_receipt_reference") or {}
                )
            except (
                OSError,
                TypeError,
                ValueError,
                ReasoningCacheError,
                ArtifactBlobIntegrityError,
            ):
                continue
            if changed.intersection(dependencies):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                invalidated_analysis.append(key.key_id)
                retained_artifacts.append(reference.artifact_id)

        proof_bindings = dict(self._proof_bindings)
        connection = self.proof_cache._connect()
        try:
            rows = connection.execute(
                "SELECT key_json FROM proof_cache_entries"
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            try:
                proof_key = ProofCacheKey.from_dict(
                    json.loads(str(row["key_json"]))
                )
                obligation = proof_key.obligation
                candidate_tree = proof_key.candidate_tree
                if not isinstance(obligation, Mapping) or not isinstance(
                    candidate_tree, Mapping
                ):
                    continue
                reasoning_key_id = obligation.get("reasoning_key_id")
                if (
                    not isinstance(reasoning_key_id, str)
                    or not reasoning_key_id.startswith(
                        "reasoning-cache-key:sha256:"
                    )
                ):
                    continue
                dependencies = _dependency_ids(
                    tuple(candidate_tree.get("dependencies") or ())
                )
                proof_bindings[reasoning_key_id] = (
                    proof_key,
                    dependencies,
                )
            except (TypeError, ValueError, ReasoningCacheError):
                # Native proof-cache lookup remains responsible for reporting
                # malformed/poisoned rows; invalidation never repairs them.
                continue

        for key_id, (proof_key, dependencies) in tuple(proof_bindings.items()):
            if changed.intersection(dependencies):
                self.proof_cache.delete(proof_key)
                invalidated_proof.append(proof_key.key_id)
                self._proof_bindings.pop(key_id, None)

        return ReasoningInvalidationReceipt(
            changed_dependency_ids=tuple(sorted(changed)),
            invalidated_analysis_key_ids=tuple(sorted(set(invalidated_analysis))),
            invalidated_proof_key_ids=tuple(sorted(set(invalidated_proof))),
            retained_artifact_ids=tuple(sorted(set(retained_artifacts))),
            run_id=self.run_id,
            created_at_ms=self._now_ms(),
        )

    invalidate = invalidate_dependencies
    invalidate_changed_dependencies = invalidate_dependencies


# Explicitly expose the existing proof-cache authority; this is an import alias,
# not a competing implementation.
ProofReceiptCache = TrustAwareProofCache


__all__ = [
    "REASONING_CACHE_INTERFACE",
    "REASONING_CACHE_COORDINATOR_INTERFACE",
    "REASONING_CACHE_VERSION",
    "REASONING_COMPUTATION_KEY_SCHEMA",
    "REASONING_SOURCE_RECEIPT_SCHEMA",
    "REASONING_CACHE_USE_RECEIPT_SCHEMA",
    "REASONING_INVALIDATION_RECEIPT_SCHEMA",
    "REASONING_ANALYSIS_INDEX_SCHEMA",
    "ReasoningCacheLane",
    "ReasoningOutcome",
    "ReasoningCacheStatus",
    "ReasoningCacheLookupStatus",
    "ReasoningCacheReason",
    "ReasoningCacheError",
    "ReasoningPrivateMaterialError",
    "ReasoningComputationKey",
    "SemanticComputationKey",
    "ReasoningCacheKey",
    "ReasoningSourceReceipt",
    "CacheUseReceipt",
    "CacheUseVerification",
    "ReasoningCacheResult",
    "ReasoningCacheStoreResult",
    "ReasoningInvalidationReceipt",
    "ReasoningCacheCoordinator",
    "TrustAwareProofCache",
    "ProofReceiptCache",
    "build_reasoning_cache_key",
    "build_reasoning_computation_key",
    "build_semantic_computation_key",
    "make_reasoning_cache_key",
]
