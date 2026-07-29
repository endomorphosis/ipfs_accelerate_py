"""Dependency-aware program-analysis cache for VFS symbolic assurance.

This module is a coordination facade, not a new object store.  Compact stage
receipts live in :class:`~analysis.analysis_cache.AnalysisCache`; process- and
cross-process single-flight collapse is provided by
:class:`~analysis.cache_coordinator.AnalysisCacheCoordinator`; immutable
artifact envelopes and reverse-dependency invalidation are owned by
:class:`~runtime.runtime_cas.RuntimeCAS`; optional large bodies are held in
:class:`~runtime.artifact_store.BoundedArtifactStore`.

Every key binds the complete forest / objective / policy / analyzer / schema /
config / query / capability / assumption / toolchain population and one stage
component (inventory, AST, graph, contract, proof, runtime, or ZK).  Compact
receipts carry only shallow immutable artifact references.  Negative and
non-success outcomes have bounded TTLs and never satisfy completion authority.
Authority namespaces (authoritative, diagnostic, proposal, draft) are closed
and cannot be upgraded by lookup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .analysis.analysis_cache import (
    AnalysisCache,
    AnalysisCacheEntry,
    AnalysisCacheKey,
    AnalysisCacheLookupResult,
    AnalysisCacheLookupStatus,
    AnalysisCacheReason,
    AnalysisOutcome,
    AnalysisReceipt,
    ReceiptValidationError,
    compact_analysis_receipt,
)
from .analysis.cache_coordinator import (
    AnalysisCacheCoordinator,
    CacheCoordinationResult,
    CachePublication,
)
from .runtime.artifact_store import (
    ArtifactQuotaPolicy,
    ArtifactOutcome,
    BoundedArtifactStore,
    BlobReference,
    RetentionClass,
)
from .runtime.runtime_cas import (
    DEPENDENCY_CAS_REQUIREMENT_ID,
    AuthorityIsolationError,
    RuntimeArtifactRecord,
    RuntimeAuthority,
    RuntimeCAS,
    RuntimeCASLookup,
    RuntimeTier,
)
from .self_improvement.supervisor_v2_contracts import (
    EvidenceFreshness,
    ResultBinding,
    SemanticDependencyIdentity,
)


PROGRAM_ANALYSIS_CACHE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-cache@1"
)
PROGRAM_ANALYSIS_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-cache-key@1"
)
PROGRAM_ANALYSIS_CACHE_ENTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-cache-entry@1"
)
PROGRAM_ANALYSIS_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/program-analysis-receipt@1"
)
DEPENDENCY_CACHE_EVIDENCE: Final = "vfs/dependency-cache@1"
CACHE_INVALIDATION_PROOF_EVIDENCE: Final = "vfs/cache-invalidation-proof@1"
DEPENDENCY_CACHE_REQUIREMENT_ID: Final = DEPENDENCY_CAS_REQUIREMENT_ID

DEFAULT_MAX_ENTRIES: Final = 512
DEFAULT_MAX_BYTES: Final = 32 * 1024 * 1024
DEFAULT_MAX_ENTRY_BYTES: Final = 128 * 1024
DEFAULT_MAX_RECEIPT_BYTES: Final = 96 * 1024
DEFAULT_NEGATIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_MAX_NEGATIVE_TTL_SECONDS: Final = 60 * 60
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_WAIT_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_ARTIFACT_BLOB_BYTES: Final = 4 * 1024 * 1024

_KEY_DIGEST_PREFIX = "program-analysis-cache-key:sha256:"
_PROGRAM_KEY_FIELD = "program_key"
_RUNTIME_ARTIFACT_FIELD = "runtime_artifact_id"
_BLOB_REFS_FIELD = "blob_refs"

# Stage pipeline: later components may depend on earlier ones; siblings do not.
_COMPONENT_ORDER: Final[tuple[str, ...]] = (
    "inventory",
    "ast",
    "graph",
    "contract",
    "proof",
    "runtime",
    "zk",
)
_UPSTREAM_OF: Final[dict[str, tuple[str, ...]]] = {
    "inventory": (),
    "ast": ("inventory",),
    "graph": ("inventory", "ast"),
    "contract": ("inventory", "ast", "graph"),
    "proof": ("inventory", "ast", "graph", "contract"),
    "runtime": ("inventory", "ast", "graph", "contract"),
    "zk": ("inventory", "ast", "graph", "contract", "proof"),
}


class ProgramAnalysisCacheError(RuntimeError):
    """Base class for program-analysis cache failures."""


class ProgramAnalysisCacheValidationError(
    ProgramAnalysisCacheError, ValueError
):
    """A key, receipt, or authority claim failed validation."""


class ProgramAnalysisComponentKind(str, Enum):
    """Closed vocabulary of program-analysis stage receipts."""

    INVENTORY = "inventory"
    AST = "ast"
    GRAPH = "graph"
    CONTRACT = "contract"
    PROOF = "proof"
    RUNTIME = "runtime"
    ZK = "zk"

    @classmethod
    def coerce(cls, value: Any) -> "ProgramAnalysisComponentKind":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "inventory_receipt": cls.INVENTORY,
            "ast_receipt": cls.AST,
            "graph_receipt": cls.GRAPH,
            "contract_receipt": cls.CONTRACT,
            "proof_receipt": cls.PROOF,
            "runtime_receipt": cls.RUNTIME,
            "zk_receipt": cls.ZK,
            "zero_knowledge": cls.ZK,
            "symbol": cls.AST,
            "symbols": cls.AST,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ProgramAnalysisCacheValidationError(
                f"component_kind must be one of: {choices}"
            ) from exc


ComponentKind = ProgramAnalysisComponentKind
AnalysisComponentKind = ProgramAnalysisComponentKind


class ProgramAnalysisAuthority(str, Enum):
    """Closed authority namespaces; never ranked or upgraded by lookup."""

    AUTHORITATIVE = "authoritative"
    DIAGNOSTIC = "diagnostic"
    PROPOSAL = "proposal"
    DRAFT = "draft"

    @classmethod
    def coerce(cls, value: Any) -> "ProgramAnalysisAuthority":
        if isinstance(value, cls):
            return value
        if isinstance(value, RuntimeAuthority):
            return cls(value.value)
        normalized = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "auth": cls.AUTHORITATIVE,
            "authority": cls.AUTHORITATIVE,
            "complete": cls.AUTHORITATIVE,
            "completion": cls.AUTHORITATIVE,
            "diag": cls.DIAGNOSTIC,
            "suggest": cls.PROPOSAL,
            "suggestion": cls.PROPOSAL,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ProgramAnalysisCacheValidationError(
                f"authority must be one of: {choices}"
            ) from exc

    def to_runtime(self) -> RuntimeAuthority:
        return RuntimeAuthority(self.value)

    @property
    def is_completion_capable(self) -> bool:
        return self is ProgramAnalysisAuthority.AUTHORITATIVE


class ProgramAnalysisLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"
    REJECTED = "rejected"


class ProgramAnalysisCacheReason(str, Enum):
    """Stable reason codes for metrics, audit records, and scheduling."""

    EXACT_KEY_HIT = "exact_key_hit"
    CACHE_MISS = "cache_miss"
    FOREST_IDENTITY_CHANGED = "forest_identity_changed"
    OBJECTIVE_REVISION_CHANGED = "objective_revision_changed"
    POLICY_REVISION_CHANGED = "policy_revision_changed"
    ANALYZER_VERSION_CHANGED = "analyzer_version_changed"
    SCHEMA_VERSION_CHANGED = "schema_version_changed"
    CONFIGURATION_DIGEST_CHANGED = "configuration_digest_changed"
    QUERY_DIGEST_CHANGED = "query_digest_changed"
    CAPABILITY_REVISION_CHANGED = "capability_revision_changed"
    ASSUMPTION_DIGEST_CHANGED = "assumption_digest_changed"
    TOOLCHAIN_VERSION_CHANGED = "toolchain_version_changed"
    COMPONENT_KIND_CHANGED = "component_kind_changed"
    AUTHORITY_CHANGED = "authority_changed"
    STALE_ENTRY = "stale_entry"
    STALE_NEGATIVE_ENTRY = "stale_negative_entry"
    CORRUPT_ENTRY = "corrupt_entry"
    NOT_COMPLETION_EVIDENCE = "not_completion_evidence"
    AUTHORITY_ISOLATION = "authority_isolation"
    DEPENDENCY_INVALIDATED = "dependency_invalidated"
    MALFORMED_RECEIPT = "malformed_receipt"
    ENTRY_TOO_LARGE = "entry_too_large"
    RUNTIME_ARTIFACT_MISS = "runtime_artifact_miss"
    RUNTIME_ARTIFACT_STALE = "runtime_artifact_stale"
    FORBIDDEN_PAYLOAD = "forbidden_payload"

    # Compatibility aliases used by AnalysisCache dimension names.
    REPOSITORY_TREE_IDENTITY_CHANGED = "forest_identity_changed"
    POLICY_DIGEST_CHANGED = "policy_revision_changed"


LookupStatus = ProgramAnalysisLookupStatus
CacheReason = ProgramAnalysisCacheReason


_KEY_DIMENSIONS: tuple[tuple[str, ProgramAnalysisCacheReason], ...] = (
    ("forest_identity", ProgramAnalysisCacheReason.FOREST_IDENTITY_CHANGED),
    (
        "objective_revision",
        ProgramAnalysisCacheReason.OBJECTIVE_REVISION_CHANGED,
    ),
    ("policy_revision", ProgramAnalysisCacheReason.POLICY_REVISION_CHANGED),
    ("analyzer_version", ProgramAnalysisCacheReason.ANALYZER_VERSION_CHANGED),
    ("schema_version", ProgramAnalysisCacheReason.SCHEMA_VERSION_CHANGED),
    (
        "configuration_digest",
        ProgramAnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED,
    ),
    ("query_digest", ProgramAnalysisCacheReason.QUERY_DIGEST_CHANGED),
    (
        "capability_revision",
        ProgramAnalysisCacheReason.CAPABILITY_REVISION_CHANGED,
    ),
    (
        "assumption_digest",
        ProgramAnalysisCacheReason.ASSUMPTION_DIGEST_CHANGED,
    ),
    ("toolchain_version", ProgramAnalysisCacheReason.TOOLCHAIN_VERSION_CHANGED),
    ("component_kind", ProgramAnalysisCacheReason.COMPONENT_KIND_CHANGED),
    ("authority", ProgramAnalysisCacheReason.AUTHORITY_CHANGED),
)


def _canonical_json_bytes(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("canonical JSON cannot contain NaN or infinity")
            return item
        if isinstance(item, Enum):
            return normalize(item.value)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("canonical JSON object keys must be strings")
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            return normalize(converter())
        raise ValueError(
            f"unsupported canonical JSON value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_program_analysis_json(value: Any) -> str:
    """Return the canonical JSON representation used by this cache."""

    return _canonical_json_bytes(value).decode("utf-8")


def digest_program_analysis_input(value: Any) -> str:
    """Return a lowercase SHA-256 hex digest of canonical JSON input."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identity_component(value: Any, name: str) -> Any:
    if value is None:
        raise ProgramAnalysisCacheValidationError(f"{name} is required")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ProgramAnalysisCacheValidationError(f"{name} must not be empty")
        if "\x00" in value:
            raise ProgramAnalysisCacheValidationError(
                f"{name} must not contain NUL"
            )
    try:
        return json.loads(canonical_program_analysis_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProgramAnalysisCacheValidationError(
            f"{name} must be canonical JSON"
        ) from exc


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class ProgramAnalysisCacheKey:
    """Content address binding every input that can change a stage result."""

    forest_identity: Any
    objective_revision: Any
    policy_revision: Any
    analyzer_version: Any
    schema_version: Any
    configuration_digest: Any
    query_digest: Any
    capability_revision: Any
    assumption_digest: Any
    toolchain_version: Any
    component_kind: ProgramAnalysisComponentKind
    authority: ProgramAnalysisAuthority

    def __init__(
        self,
        forest_identity: Any = None,
        objective_revision: Any = None,
        policy_revision: Any = None,
        analyzer_version: Any = None,
        schema_version: Any = None,
        configuration_digest: Any = None,
        query_digest: Any = None,
        capability_revision: Any = None,
        assumption_digest: Any = None,
        toolchain_version: Any = None,
        component_kind: ProgramAnalysisComponentKind | str = (
            ProgramAnalysisComponentKind.INVENTORY
        ),
        authority: ProgramAnalysisAuthority | str = (
            ProgramAnalysisAuthority.AUTHORITATIVE
        ),
        *,
        repository_tree_identity: Any = None,
        repository_forest_identity: Any = None,
        policy_digest: Any = None,
        capability_digest: Any = None,
        assumptions_digest: Any = None,
        toolchain: Any = None,
    ) -> None:
        forest_candidates = [
            item
            for item in (
                forest_identity,
                repository_forest_identity,
                repository_tree_identity,
            )
            if item is not None
        ]
        if not forest_candidates:
            forest = None
        else:
            forest = forest_candidates[0]
            canonical_forest = canonical_program_analysis_json(forest)
            if any(
                canonical_program_analysis_json(item) != canonical_forest
                for item in forest_candidates[1:]
            ):
                raise ProgramAnalysisCacheValidationError(
                    "forest identity aliases disagree"
                )
        policy = (
            policy_revision if policy_revision is not None else policy_digest
        )
        capability = (
            capability_revision
            if capability_revision is not None
            else capability_digest
        )
        assumption = (
            assumption_digest
            if assumption_digest is not None
            else assumptions_digest
        )
        toolchain_value = (
            toolchain_version if toolchain_version is not None else toolchain
        )
        values = {
            "forest_identity": forest,
            "objective_revision": objective_revision,
            "policy_revision": policy,
            "analyzer_version": analyzer_version,
            "schema_version": schema_version,
            "configuration_digest": configuration_digest,
            "query_digest": query_digest,
            "capability_revision": capability,
            "assumption_digest": assumption,
            "toolchain_version": toolchain_value,
        }
        for name, value in values.items():
            object.__setattr__(self, name, _identity_component(value, name))
        object.__setattr__(
            self,
            "component_kind",
            ProgramAnalysisComponentKind.coerce(component_kind),
        )
        object.__setattr__(
            self, "authority", ProgramAnalysisAuthority.coerce(authority)
        )

    @property
    def repository_tree_identity(self) -> Any:
        return self.forest_identity

    @property
    def policy_digest(self) -> Any:
        return self.policy_revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_ANALYSIS_CACHE_KEY_SCHEMA,
            "forest_identity": self.forest_identity,
            "objective_revision": self.objective_revision,
            "policy_revision": self.policy_revision,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "capability_revision": self.capability_revision,
            "assumption_digest": self.assumption_digest,
            "toolchain_version": self.toolchain_version,
            "component_kind": self.component_kind.value,
            "authority": self.authority.value,
        }

    @property
    def digest(self) -> str:
        return digest_program_analysis_input(self.to_dict())

    @property
    def key_id(self) -> str:
        return f"{_KEY_DIGEST_PREFIX}{self.digest}"

    @property
    def cache_key(self) -> str:
        return self.key_id

    def dimension_values(self) -> dict[str, Any]:
        return {
            "forest_identity": self.forest_identity,
            "objective_revision": self.objective_revision,
            "policy_revision": self.policy_revision,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "capability_revision": self.capability_revision,
            "assumption_digest": self.assumption_digest,
            "toolchain_version": self.toolchain_version,
            "component_kind": self.component_kind.value,
            "authority": self.authority.value,
        }

    def to_analysis_cache_key(self) -> AnalysisCacheKey:
        """Map the full population onto the durable AnalysisCache key surface.

        Extra dimensions that AnalysisCache does not name individually are
        folded into ``configuration_digest`` so any change still yields a
        distinct content address.  Program-level reason codes are recovered by
        comparing the full key stored inside the compact receipt.
        """

        folded_configuration = {
            "configuration_digest": self.configuration_digest,
            "capability_revision": self.capability_revision,
            "assumption_digest": self.assumption_digest,
            "toolchain_version": self.toolchain_version,
            "component_kind": self.component_kind.value,
            "authority": self.authority.value,
            "program_key_schema": PROGRAM_ANALYSIS_CACHE_KEY_SCHEMA,
        }
        return AnalysisCacheKey(
            repository_tree_identity=self.forest_identity,
            objective_revision=self.objective_revision,
            analyzer_version=self.analyzer_version,
            schema_version=self.schema_version,
            configuration_digest=digest_program_analysis_input(
                folded_configuration
            ),
            query_digest=self.query_digest,
            policy_digest=self.policy_revision,
        )

    def population_dependencies(self) -> tuple[SemanticDependencyIdentity, ...]:
        """Semantic dependency population used by RuntimeCAS bindings."""

        dimensions = (
            ("forest", "forest_identity", self.forest_identity),
            ("objective", "objective_revision", self.objective_revision),
            ("policy", "policy_revision", self.policy_revision),
            ("analyzer", "analyzer_version", self.analyzer_version),
            ("schema", "schema_version", self.schema_version),
            ("configuration", "configuration_digest", self.configuration_digest),
            ("query", "query_digest", self.query_digest),
            ("capability", "capability_revision", self.capability_revision),
            ("assumption", "assumption_digest", self.assumption_digest),
            ("toolchain", "toolchain_version", self.toolchain_version),
            ("component", "component_kind", self.component_kind.value),
            ("authority", "authority", self.authority.value),
        )
        dependencies: list[SemanticDependencyIdentity] = []
        for namespace, key, value in dimensions:
            revision = (
                value
                if isinstance(value, str)
                else canonical_program_analysis_json(value)
            )
            dependencies.append(
                SemanticDependencyIdentity(
                    namespace=f"program_analysis/{namespace}",
                    key=key,
                    revision=revision,
                    digest=_sha256_text(revision),
                )
            )
        return tuple(dependencies)

    def build_result_binding(
        self,
        *,
        repository_id: str = "repository:program-analysis",
        task_id: str = "VFS-011",
        environment_id: str = "environment:program-analysis-cache",
        environment_revision: str = "environment:program-analysis-cache@1",
    ) -> ResultBinding:
        forest = (
            self.forest_identity
            if isinstance(self.forest_identity, str)
            else canonical_program_analysis_json(self.forest_identity)
        )
        objective = (
            self.objective_revision
            if isinstance(self.objective_revision, str)
            else canonical_program_analysis_json(self.objective_revision)
        )
        policy = (
            self.policy_revision
            if isinstance(self.policy_revision, str)
            else canonical_program_analysis_json(self.policy_revision)
        )
        capability = (
            self.capability_revision
            if isinstance(self.capability_revision, str)
            else canonical_program_analysis_json(self.capability_revision)
        )
        analyzer = (
            self.analyzer_version
            if isinstance(self.analyzer_version, str)
            else canonical_program_analysis_json(self.analyzer_version)
        )
        return ResultBinding(
            repository_id=repository_id,
            tree_id=forest,
            objective_id="VFS-G031",
            objective_revision=objective,
            task_id=task_id,
            task_revision=f"task:{task_id}@{self.component_kind.value}",
            policy_id="policy:program-analysis-cache",
            policy_revision=policy,
            producer_id="producer:program-analysis-cache",
            producer_revision=analyzer,
            capability_id="capability:program-analysis-cache",
            capability_revision=capability,
            environment_id=environment_id,
            environment_revision=environment_revision,
            semantic_dependencies=self.population_dependencies(),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProgramAnalysisCacheKey":
        if not isinstance(value, Mapping):
            raise ProgramAnalysisCacheValidationError(
                "program analysis cache key must be an object"
            )
        schema = value.get("schema", value.get("schema_version_id"))
        if schema not in (None, PROGRAM_ANALYSIS_CACHE_KEY_SCHEMA):
            raise ProgramAnalysisCacheValidationError(
                "unsupported program analysis cache-key schema"
            )
        return cls(
            forest_identity=value.get(
                "forest_identity",
                value.get(
                    "repository_forest_identity",
                    value.get("repository_tree_identity"),
                ),
            ),
            objective_revision=value.get("objective_revision"),
            policy_revision=value.get(
                "policy_revision", value.get("policy_digest")
            ),
            analyzer_version=value.get("analyzer_version"),
            schema_version=value.get("schema_version"),
            configuration_digest=value.get("configuration_digest"),
            query_digest=value.get("query_digest"),
            capability_revision=value.get(
                "capability_revision", value.get("capability_digest")
            ),
            assumption_digest=value.get(
                "assumption_digest", value.get("assumptions_digest")
            ),
            toolchain_version=value.get(
                "toolchain_version", value.get("toolchain")
            ),
            component_kind=value.get("component_kind", "inventory"),
            authority=value.get(
                "authority", ProgramAnalysisAuthority.AUTHORITATIVE.value
            ),
        )


def build_program_analysis_cache_key(
    *,
    forest_identity: Any = None,
    repository_tree_identity: Any = None,
    repository_forest_identity: Any = None,
    objective_revision: Any,
    policy_revision: Any = None,
    policy_digest: Any = None,
    analyzer_version: Any,
    schema_version: Any,
    configuration_digest: Any,
    query_digest: Any,
    capability_revision: Any = None,
    capability_digest: Any = None,
    assumption_digest: Any = None,
    assumptions_digest: Any = None,
    toolchain_version: Any = None,
    toolchain: Any = None,
    component_kind: ProgramAnalysisComponentKind | str = (
        ProgramAnalysisComponentKind.INVENTORY
    ),
    authority: ProgramAnalysisAuthority | str = (
        ProgramAnalysisAuthority.AUTHORITATIVE
    ),
) -> ProgramAnalysisCacheKey:
    """Build a program-analysis key while accepting common aliases."""

    return ProgramAnalysisCacheKey(
        forest_identity=forest_identity,
        repository_tree_identity=repository_tree_identity,
        repository_forest_identity=repository_forest_identity,
        objective_revision=objective_revision,
        policy_revision=policy_revision,
        policy_digest=policy_digest,
        analyzer_version=analyzer_version,
        schema_version=schema_version,
        configuration_digest=configuration_digest,
        query_digest=query_digest,
        capability_revision=capability_revision,
        capability_digest=capability_digest,
        assumption_digest=assumption_digest,
        assumptions_digest=assumptions_digest,
        toolchain_version=toolchain_version,
        toolchain=toolchain,
        component_kind=component_kind,
        authority=authority,
    )


make_program_analysis_cache_key = build_program_analysis_cache_key


def _difference_reasons(
    stored: ProgramAnalysisCacheKey, requested: ProgramAnalysisCacheKey
) -> tuple[str, ...]:
    reasons: list[str] = []
    for name, reason in _KEY_DIMENSIONS:
        left = getattr(stored, name)
        right = getattr(requested, name)
        if name in {"component_kind", "authority"}:
            left = left.value if isinstance(left, Enum) else left
            right = right.value if isinstance(right, Enum) else right
        if left != right:
            reasons.append(reason.value)
    return tuple(reasons)


@dataclass(frozen=True)
class ProgramAnalysisLookupResult:
    status: ProgramAnalysisLookupStatus
    key: ProgramAnalysisCacheKey
    entry: AnalysisCacheEntry | None = None
    runtime_artifact: RuntimeArtifactRecord | None = None
    reason_codes: tuple[str, ...] = ()
    coordination: CacheCoordinationResult | None = None

    @property
    def hit(self) -> bool:
        return self.status is ProgramAnalysisLookupStatus.HIT

    @property
    def miss(self) -> bool:
        return self.status is ProgramAnalysisLookupStatus.MISS

    @property
    def invalidated(self) -> bool:
        return self.status is ProgramAnalysisLookupStatus.INVALIDATED

    @property
    def rejected(self) -> bool:
        return self.status is ProgramAnalysisLookupStatus.REJECTED

    @property
    def receipt(self) -> Mapping[str, Any] | None:
        return self.entry.receipt if self.hit and self.entry is not None else None

    @property
    def outcome(self) -> AnalysisOutcome | None:
        return self.entry.status if self.entry is not None else None

    @property
    def is_completion_evidence(self) -> bool:
        return bool(
            self.hit
            and self.entry is not None
            and self.entry.is_completion_evidence
            and self.key.authority.is_completion_capable
            and (
                self.runtime_artifact is None
                or self.runtime_artifact.identity.authority
                is RuntimeAuthority.AUTHORITATIVE
            )
        )

    @property
    def completion_evidence(self) -> bool:
        return self.is_completion_evidence

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def reason(self) -> str:
        return self.reason_code


@dataclass(frozen=True)
class ProgramAnalysisStoreResult:
    stored: bool
    key: ProgramAnalysisCacheKey
    entry: AnalysisCacheEntry | None = None
    runtime_artifact: RuntimeArtifactRecord | None = None
    blob_refs: tuple[BlobReference, ...] = ()
    reason_codes: tuple[str, ...] = ()
    evicted_count: int = 0

    def __bool__(self) -> bool:
        return self.stored

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True)
class ProgramAnalysisCacheStats:
    entry_count: int
    total_bytes: int
    successful_count: int
    partial_count: int
    failed_count: int
    timed_out_count: int
    inconclusive_count: int
    corrupt_count: int = 0
    runtime_artifact_count: int = 0
    blob_count: int = 0
    artifact_bytes: int = 0
    max_artifacts: int = 0
    max_artifact_bytes: int = 0
    component_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "total_bytes": self.total_bytes,
            "successful_count": self.successful_count,
            "partial_count": self.partial_count,
            "failed_count": self.failed_count,
            "timed_out_count": self.timed_out_count,
            "inconclusive_count": self.inconclusive_count,
            "corrupt_count": self.corrupt_count,
            "runtime_artifact_count": self.runtime_artifact_count,
            "blob_count": self.blob_count,
            "artifact_bytes": self.artifact_bytes,
            "max_artifacts": self.max_artifacts,
            "max_artifact_bytes": self.max_artifact_bytes,
            "component_counts": dict(self.component_counts),
        }


def _extract_program_key(
    receipt: Mapping[str, Any] | None,
) -> ProgramAnalysisCacheKey | None:
    if not isinstance(receipt, Mapping):
        return None
    raw = receipt.get(_PROGRAM_KEY_FIELD)
    if not isinstance(raw, Mapping):
        return None
    try:
        return ProgramAnalysisCacheKey.from_dict(raw)
    except (TypeError, ValueError, ProgramAnalysisCacheValidationError):
        return None


def compact_program_analysis_receipt(
    receipt: Mapping[str, Any] | Any,
    *,
    key: ProgramAnalysisCacheKey | None = None,
    status: AnalysisOutcome | str | None = None,
    runtime_artifact_id: str | None = None,
    blob_refs: Sequence[Mapping[str, Any] | BlobReference] = (),
    max_receipt_bytes: int = DEFAULT_MAX_RECEIPT_BYTES,
) -> dict[str, Any]:
    """Normalize a compact program-analysis stage receipt.

    Heavy bodies must already be artifact/blob references.  The full program
    key is embedded so dimension-level invalidation reasons remain recoverable
    after the AnalysisCache folding step.
    """

    converter = getattr(receipt, "to_dict", None)
    if callable(converter):
        receipt = converter()
    if not isinstance(receipt, Mapping):
        raise ReceiptValidationError("program analysis receipt must be an object")
    payload = dict(receipt)
    if key is not None:
        payload[_PROGRAM_KEY_FIELD] = key.to_dict()
    if runtime_artifact_id is not None:
        payload[_RUNTIME_ARTIFACT_FIELD] = runtime_artifact_id
    if blob_refs:
        payload[_BLOB_REFS_FIELD] = [
            item.to_dict() if isinstance(item, BlobReference) else dict(item)
            for item in blob_refs
        ]
    payload.setdefault("schema", PROGRAM_ANALYSIS_RECEIPT_SCHEMA)
    if key is not None:
        payload.setdefault("component_kind", key.component_kind.value)
        payload.setdefault("authority", key.authority.value)
    normalized = compact_analysis_receipt(
        payload,
        status=status,
        max_receipt_bytes=max_receipt_bytes,
    )
    return normalized


class ProgramAnalysisCache:
    """Dependency-aware cache for program-analysis stage receipts.

    Storage layers:

    * ``receipts/`` — :class:`AnalysisCache` compact receipt entries
    * ``runtime/`` — :class:`RuntimeCAS` immutable envelopes + invalidation
    * ``artifacts/`` — :class:`BoundedArtifactStore` optional large bodies
    * coordinator — process and cross-process single-flight collapse
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
        max_receipt_bytes: int = DEFAULT_MAX_RECEIPT_BYTES,
        default_negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS,
        max_negative_ttl_seconds: int = DEFAULT_MAX_NEGATIVE_TTL_SECONDS,
        default_success_ttl_seconds: int | None = None,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        wait_timeout_seconds: float | None = DEFAULT_WAIT_TIMEOUT_SECONDS,
        max_artifact_blob_bytes: int = DEFAULT_MAX_ARTIFACT_BLOB_BYTES,
        max_artifact_bytes: int | None = None,
        max_artifacts: int | None = None,
        clock: Callable[[], float] = time.time,
        shared_store: Any | None = None,
    ) -> None:
        if path is None:
            path = tempfile.mkdtemp(prefix="program-analysis-cache-")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path, 0o700)
        except OSError:
            pass
        self.max_receipt_bytes = max_receipt_bytes
        self.default_negative_ttl_seconds = default_negative_ttl_seconds
        self.max_negative_ttl_seconds = max_negative_ttl_seconds
        self.default_success_ttl_seconds = default_success_ttl_seconds
        self._clock = clock
        self._index_lock = threading.RLock()
        self._component_index: dict[str, str] = {}

        self.analysis_cache = AnalysisCache(
            self.path / "receipts",
            max_entries=max_entries,
            max_bytes=max_bytes,
            max_entry_bytes=max_entry_bytes,
            max_receipt_bytes=max_receipt_bytes,
            default_negative_ttl_seconds=default_negative_ttl_seconds,
            max_negative_ttl_seconds=max_negative_ttl_seconds,
            default_success_ttl_seconds=default_success_ttl_seconds,
            lock_timeout_seconds=lock_timeout_seconds,
            clock=clock,
        )
        self.coordinator = AnalysisCacheCoordinator(
            self.analysis_cache,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        self.runtime_cas = RuntimeCAS(
            self.path / "runtime",
            shared_store=shared_store,
            clock=clock,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        artifact_byte_quota = (
            max(max_bytes, max_artifact_blob_bytes * 4)
            if max_artifact_bytes is None
            else max_artifact_bytes
        )
        artifact_count_quota = (
            max(max_entries * 4, 64)
            if max_artifacts is None
            else max_artifacts
        )
        self.artifact_store = BoundedArtifactStore(
            self.path / "artifacts",
            quotas=ArtifactQuotaPolicy(
                max_bytes=artifact_byte_quota,
                max_blobs=artifact_count_quota,
                max_projections=max(max_entries * 2, 64),
                max_blob_bytes=max_artifact_blob_bytes,
                max_receipt_bytes=min(max_receipt_bytes, 262_144),
                max_projection_bytes=max(max_receipt_bytes * 4, 256 * 1024),
                negative_ttl_seconds=default_negative_ttl_seconds,
                inconclusive_ttl_seconds=default_negative_ttl_seconds,
                max_ttl_seconds=max(
                    max_negative_ttl_seconds,
                    default_success_ttl_seconds or max_negative_ttl_seconds,
                ),
            ),
            clock=clock,
        )
        self._rebuild_component_index()

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _coerce_key(
        self, key: ProgramAnalysisCacheKey | Mapping[str, Any]
    ) -> ProgramAnalysisCacheKey:
        return (
            key
            if isinstance(key, ProgramAnalysisCacheKey)
            else ProgramAnalysisCacheKey.from_dict(key)
        )

    def _index_path(self) -> Path:
        return self.path / "component-index.json"

    def _rebuild_component_index(self) -> None:
        index: dict[str, str] = {}
        for path in self.analysis_cache._entry_paths():  # noqa: SLF001
            try:
                entry = self.analysis_cache._read_path(path)  # noqa: SLF001
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                continue
            program_key = _extract_program_key(entry.receipt)
            if program_key is None:
                continue
            runtime_id = str(
                entry.receipt.get(_RUNTIME_ARTIFACT_FIELD) or ""
            )
            if runtime_id:
                index[program_key.key_id] = runtime_id
        path = self._index_path()
        try:
            if path.exists():
                payload = json.loads(path.read_bytes())
                if isinstance(payload, Mapping):
                    stored = payload.get("entries")
                    if isinstance(stored, Mapping):
                        for key_id, artifact_id in stored.items():
                            if (
                                isinstance(key_id, str)
                                and isinstance(artifact_id, str)
                                and key_id not in index
                            ):
                                index[key_id] = artifact_id
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            try:
                path.unlink()
            except OSError:
                pass
        with self._index_lock:
            self._component_index = index

    def _persist_component_index(self) -> None:
        with self._index_lock:
            payload = {
                "schema": PROGRAM_ANALYSIS_CACHE_SCHEMA,
                "entries": dict(sorted(self._component_index.items())),
            }
            encoded = _canonical_json_bytes(payload) + b"\n"
        path = self._index_path()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".component-index.",
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

    def _register_index(
        self, key: ProgramAnalysisCacheKey, runtime_artifact_id: str
    ) -> None:
        with self._index_lock:
            self._component_index[key.key_id] = runtime_artifact_id
        self._persist_component_index()

    def _unregister_index(self, key_ids: Sequence[str]) -> None:
        with self._index_lock:
            for key_id in key_ids:
                self._component_index.pop(key_id, None)
        self._persist_component_index()

    def entry_path(self, key: ProgramAnalysisCacheKey | Mapping[str, Any]) -> Path:
        cache_key = self._coerce_key(key)
        return self.analysis_cache.entry_path(cache_key.to_analysis_cache_key())

    def put_blob(
        self,
        value: Any,
        *,
        kind: str = "program_analysis_body",
        retention_class: RetentionClass | str = RetentionClass.ROUTINE,
        outcome: ArtifactOutcome | str = ArtifactOutcome.SUCCESSFUL,
        ttl_seconds: int | None = None,
    ) -> BlobReference:
        """Store an immutable large body and return a shallow reference."""

        return self.artifact_store.put_blob(
            value,
            kind=kind,
            retention_class=retention_class,
            outcome=outcome,
            ttl_seconds=ttl_seconds,
        )

    def _runtime_namespace(self, key: ProgramAnalysisCacheKey) -> str:
        return f"program_analysis/{key.authority.value}/{key.component_kind.value}"

    def _put_runtime_artifact(
        self,
        key: ProgramAnalysisCacheKey,
        receipt: Mapping[str, Any],
        *,
        dependencies: Sequence[
            RuntimeArtifactRecord | Mapping[str, Any] | str
        ] = (),
        ttl_seconds: int | None,
        outcome: AnalysisOutcome,
    ) -> RuntimeArtifactRecord:
        binding = key.build_result_binding()
        authority = key.authority.to_runtime()
        if authority is RuntimeAuthority.AUTHORITATIVE and any(
            (
                item.identity.authority is RuntimeAuthority.DRAFT
                if isinstance(item, RuntimeArtifactRecord)
                else False
            )
            for item in dependencies
        ):
            raise AuthorityIsolationError(
                "authoritative program receipts cannot depend on drafts"
            )
        payload = {
            "schema": PROGRAM_ANALYSIS_CACHE_ENTRY_SCHEMA,
            "key_id": key.key_id,
            "key": key.to_dict(),
            "receipt": dict(receipt),
            "status": outcome.value,
            "evidence": [
                DEPENDENCY_CACHE_EVIDENCE,
                CACHE_INVALIDATION_PROOF_EVIDENCE,
            ],
        }
        # Freshness describes generation quality, not outcome polarity.
        # Non-success outcomes still use FRESH + bounded TTL so negative
        # caches remain reusable until they expire.
        effective_ttl = ttl_seconds
        if not outcome.is_completion_evidence and effective_ttl is None:
            effective_ttl = self.default_negative_ttl_seconds
        if not outcome.is_completion_evidence and effective_ttl is not None:
            effective_ttl = min(effective_ttl, self.max_negative_ttl_seconds)
        return self.runtime_cas.put(
            payload,
            binding=binding,
            namespace=self._runtime_namespace(key),
            artifact_kind=f"program_analysis_{key.component_kind.value}",
            authority=authority,
            dependencies=dependencies,
            freshness=EvidenceFreshness.FRESH,
            ttl_seconds=effective_ttl,
            tiers=(
                RuntimeTier.PROCESS_LOCAL,
                RuntimeTier.HOST_DURABLE,
            ),
            payload_schema=PROGRAM_ANALYSIS_CACHE_ENTRY_SCHEMA,
            projection_key=key.key_id
            if authority is RuntimeAuthority.AUTHORITATIVE
            and outcome.is_completion_evidence
            else None,
            tree_id=(
                key.forest_identity
                if isinstance(key.forest_identity, str)
                else None
            ),
        )

    def _lookup_runtime(
        self,
        artifact_id: str,
        *,
        key: ProgramAnalysisCacheKey,
        require_fresh: bool,
    ) -> RuntimeCASLookup:
        return self.runtime_cas.lookup(
            artifact_id,
            expected_namespace=self._runtime_namespace(key),
            expected_authority=key.authority.to_runtime(),
            require_fresh=require_fresh,
        )

    def _closest_program_candidate(
        self, key: ProgramAnalysisCacheKey
    ) -> ProgramAnalysisCacheKey | None:
        candidates: list[tuple[int, int, str, ProgramAnalysisCacheKey]] = []
        now_ms = self._now_ms()
        for path in self.analysis_cache._entry_paths():  # noqa: SLF001
            try:
                entry = self.analysis_cache._read_path(path)  # noqa: SLF001
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                continue
            if self.analysis_cache._is_stale(entry, now_ms):  # noqa: SLF001
                continue
            program_key = _extract_program_key(entry.receipt)
            if program_key is None:
                continue
            distance = 0
            for name, _reason in _KEY_DIMENSIONS:
                left = getattr(program_key, name)
                right = getattr(key, name)
                if name in {"component_kind", "authority"}:
                    left = left.value if isinstance(left, Enum) else left
                    right = right.value if isinstance(right, Enum) else right
                if left != right:
                    distance += 1
            if distance == 0 or distance == len(_KEY_DIMENSIONS):
                continue
            candidates.append(
                (
                    distance,
                    -entry.created_at_ms,
                    program_key.key_id,
                    program_key,
                )
            )
        return min(candidates)[-1] if candidates else None

    def lookup(
        self,
        key: ProgramAnalysisCacheKey | Mapping[str, Any],
        *,
        require_completion_evidence: bool = False,
        require_runtime_artifact: bool = True,
    ) -> ProgramAnalysisLookupResult:
        """Return an exact, fresh, authority-isolated hit or a typed miss."""

        cache_key = self._coerce_key(key)
        analysis_key = cache_key.to_analysis_cache_key()
        analysis_lookup = self.analysis_cache.lookup(
            analysis_key,
            require_completion_evidence=require_completion_evidence,
        )

        if analysis_lookup.status is AnalysisCacheLookupStatus.HIT:
            entry = analysis_lookup.entry
            assert entry is not None
            program_key = _extract_program_key(entry.receipt) or cache_key
            if program_key.key_id != cache_key.key_id:
                # Folded key collision should not happen; treat as miss.
                return ProgramAnalysisLookupResult(
                    ProgramAnalysisLookupStatus.MISS,
                    cache_key,
                    reason_codes=(ProgramAnalysisCacheReason.CACHE_MISS.value,),
                )
            if program_key.authority != cache_key.authority:
                return ProgramAnalysisLookupResult(
                    ProgramAnalysisLookupStatus.REJECTED,
                    cache_key,
                    entry=entry,
                    reason_codes=(
                        ProgramAnalysisCacheReason.AUTHORITY_ISOLATION.value,
                    ),
                )
            runtime_artifact: RuntimeArtifactRecord | None = None
            runtime_id = str(entry.receipt.get(_RUNTIME_ARTIFACT_FIELD) or "")
            if require_runtime_artifact and runtime_id:
                runtime_lookup = self._lookup_runtime(
                    runtime_id,
                    key=cache_key,
                    require_fresh=(
                        require_completion_evidence
                        or cache_key.authority.is_completion_capable
                    ),
                )
                if not runtime_lookup.hit:
                    reason = ProgramAnalysisCacheReason.RUNTIME_ARTIFACT_MISS
                    if "stale_artifact" in runtime_lookup.reason_codes:
                        reason = ProgramAnalysisCacheReason.RUNTIME_ARTIFACT_STALE
                    if "invalidated" in runtime_lookup.reason_codes:
                        reason = ProgramAnalysisCacheReason.DEPENDENCY_INVALIDATED
                    return ProgramAnalysisLookupResult(
                        ProgramAnalysisLookupStatus.INVALIDATED,
                        cache_key,
                        entry=entry,
                        reason_codes=(reason.value, *runtime_lookup.reason_codes),
                    )
                runtime_artifact = runtime_lookup.artifact
            elif require_runtime_artifact and not runtime_id:
                return ProgramAnalysisLookupResult(
                    ProgramAnalysisLookupStatus.INVALIDATED,
                    cache_key,
                    entry=entry,
                    reason_codes=(
                        ProgramAnalysisCacheReason.RUNTIME_ARTIFACT_MISS.value,
                    ),
                )
            if require_completion_evidence and not (
                entry.is_completion_evidence
                and cache_key.authority.is_completion_capable
            ):
                return ProgramAnalysisLookupResult(
                    ProgramAnalysisLookupStatus.INVALIDATED,
                    cache_key,
                    entry=entry,
                    runtime_artifact=runtime_artifact,
                    reason_codes=(
                        ProgramAnalysisCacheReason.NOT_COMPLETION_EVIDENCE.value,
                    ),
                )
            return ProgramAnalysisLookupResult(
                ProgramAnalysisLookupStatus.HIT,
                cache_key,
                entry=entry,
                runtime_artifact=runtime_artifact,
                reason_codes=(ProgramAnalysisCacheReason.EXACT_KEY_HIT.value,),
            )

        if analysis_lookup.status is AnalysisCacheLookupStatus.INVALIDATED:
            reasons = tuple(analysis_lookup.reason_codes)
            mapped: list[str] = []
            for reason in reasons:
                if reason == AnalysisCacheReason.STALE_ENTRY.value:
                    mapped.append(ProgramAnalysisCacheReason.STALE_ENTRY.value)
                elif reason == AnalysisCacheReason.STALE_NEGATIVE_ENTRY.value:
                    mapped.append(
                        ProgramAnalysisCacheReason.STALE_NEGATIVE_ENTRY.value
                    )
                elif reason == AnalysisCacheReason.CORRUPT_ENTRY.value:
                    mapped.append(
                        ProgramAnalysisCacheReason.CORRUPT_ENTRY.value
                    )
                elif reason == AnalysisCacheReason.NOT_COMPLETION_EVIDENCE.value:
                    mapped.append(
                        ProgramAnalysisCacheReason.NOT_COMPLETION_EVIDENCE.value
                    )
                elif reason == (
                    AnalysisCacheReason.REPOSITORY_TREE_IDENTITY_CHANGED.value
                ):
                    mapped.append(
                        ProgramAnalysisCacheReason.FOREST_IDENTITY_CHANGED.value
                    )
                elif reason == AnalysisCacheReason.POLICY_DIGEST_CHANGED.value:
                    mapped.append(
                        ProgramAnalysisCacheReason.POLICY_REVISION_CHANGED.value
                    )
                else:
                    mapped.append(reason)
            # Prefer program-key dimension reasons when a near neighbour exists.
            candidate = self._closest_program_candidate(cache_key)
            if candidate is not None:
                program_reasons = _difference_reasons(candidate, cache_key)
                if program_reasons:
                    mapped = list(program_reasons)
            return ProgramAnalysisLookupResult(
                ProgramAnalysisLookupStatus.INVALIDATED,
                cache_key,
                entry=analysis_lookup.entry,
                reason_codes=tuple(mapped)
                or (ProgramAnalysisCacheReason.CACHE_MISS.value,),
            )

        candidate = self._closest_program_candidate(cache_key)
        if candidate is not None:
            return ProgramAnalysisLookupResult(
                ProgramAnalysisLookupStatus.INVALIDATED,
                cache_key,
                reason_codes=_difference_reasons(candidate, cache_key)
                or (ProgramAnalysisCacheReason.CACHE_MISS.value,),
            )
        return ProgramAnalysisLookupResult(
            ProgramAnalysisLookupStatus.MISS,
            cache_key,
            reason_codes=(ProgramAnalysisCacheReason.CACHE_MISS.value,),
        )

    get = lookup

    def put(
        self,
        key: ProgramAnalysisCacheKey | Mapping[str, Any],
        receipt: Mapping[str, Any] | AnalysisReceipt,
        *,
        status: AnalysisOutcome | str | None = None,
        ttl_seconds: int | None = None,
        blob_bodies: Sequence[Any] = (),
        dependencies: Sequence[
            RuntimeArtifactRecord | Mapping[str, Any] | str
        ] = (),
        store_runtime_artifact: bool = True,
    ) -> ProgramAnalysisStoreResult:
        """Persist a compact receipt plus optional immutable artifact bodies."""

        cache_key = self._coerce_key(key)
        blob_refs: list[BlobReference] = []
        try:
            for index, body in enumerate(blob_bodies):
                blob_refs.append(
                    self.put_blob(
                        body,
                        kind=f"{cache_key.component_kind.value}_body",
                        retention_class=(
                            RetentionClass.AUTHORITATIVE
                            if cache_key.authority
                            is ProgramAnalysisAuthority.AUTHORITATIVE
                            else RetentionClass.ROUTINE
                        ),
                        outcome=(
                            ArtifactOutcome.SUCCESSFUL
                            if status is None
                            or AnalysisOutcome.coerce(status).is_completion_evidence
                            else ArtifactOutcome.FAILED
                        ),
                        ttl_seconds=ttl_seconds,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - map to store rejection
            return ProgramAnalysisStoreResult(
                False,
                cache_key,
                reason_codes=(
                    ProgramAnalysisCacheReason.MALFORMED_RECEIPT.value,
                    type(exc).__name__,
                ),
            )

        try:
            draft = compact_program_analysis_receipt(
                receipt,
                key=cache_key,
                status=status,
                blob_refs=blob_refs,
                max_receipt_bytes=self.max_receipt_bytes,
            )
        except (TypeError, ValueError, ReceiptValidationError):
            return ProgramAnalysisStoreResult(
                False,
                cache_key,
                reason_codes=(ProgramAnalysisCacheReason.MALFORMED_RECEIPT.value,),
            )

        outcome = AnalysisOutcome.coerce(draft["status"])
        runtime_artifact: RuntimeArtifactRecord | None = None
        if store_runtime_artifact:
            try:
                runtime_artifact = self._put_runtime_artifact(
                    cache_key,
                    draft,
                    dependencies=dependencies,
                    ttl_seconds=ttl_seconds,
                    outcome=outcome,
                )
                draft = compact_program_analysis_receipt(
                    draft,
                    key=cache_key,
                    status=outcome,
                    runtime_artifact_id=runtime_artifact.artifact_id,
                    blob_refs=blob_refs,
                    max_receipt_bytes=self.max_receipt_bytes,
                )
            except AuthorityIsolationError:
                return ProgramAnalysisStoreResult(
                    False,
                    cache_key,
                    reason_codes=(
                        ProgramAnalysisCacheReason.AUTHORITY_ISOLATION.value,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                return ProgramAnalysisStoreResult(
                    False,
                    cache_key,
                    reason_codes=(
                        ProgramAnalysisCacheReason.MALFORMED_RECEIPT.value,
                        type(exc).__name__,
                    ),
                )

        stored = self.analysis_cache.put(
            cache_key.to_analysis_cache_key(),
            draft,
            status=outcome,
            ttl_seconds=ttl_seconds,
        )
        if not stored.stored:
            reasons = []
            for reason in stored.reason_codes:
                if reason == AnalysisCacheReason.MALFORMED_RECEIPT.value:
                    reasons.append(
                        ProgramAnalysisCacheReason.MALFORMED_RECEIPT.value
                    )
                elif reason == AnalysisCacheReason.ENTRY_TOO_LARGE.value:
                    reasons.append(
                        ProgramAnalysisCacheReason.ENTRY_TOO_LARGE.value
                    )
                else:
                    reasons.append(reason)
            return ProgramAnalysisStoreResult(
                False,
                cache_key,
                reason_codes=tuple(reasons)
                or (ProgramAnalysisCacheReason.MALFORMED_RECEIPT.value,),
            )

        if runtime_artifact is not None:
            self._register_index(cache_key, runtime_artifact.artifact_id)

        return ProgramAnalysisStoreResult(
            True,
            cache_key,
            entry=stored.entry,
            runtime_artifact=runtime_artifact,
            blob_refs=tuple(blob_refs),
            evicted_count=stored.evicted_count,
        )

    store = put

    def put_component(
        self,
        key: ProgramAnalysisCacheKey | Mapping[str, Any],
        receipt: Mapping[str, Any] | AnalysisReceipt,
        *,
        upstream: Sequence[ProgramAnalysisCacheKey | Mapping[str, Any]] = (),
        status: AnalysisOutcome | str | None = None,
        ttl_seconds: int | None = None,
        blob_bodies: Sequence[Any] = (),
    ) -> ProgramAnalysisStoreResult:
        """Store a stage receipt, wiring RuntimeCAS edges to upstream stages."""

        cache_key = self._coerce_key(key)
        dependencies: list[RuntimeArtifactRecord | str] = []
        for item in upstream:
            upstream_key = self._coerce_key(item)
            allowed = _UPSTREAM_OF.get(cache_key.component_kind.value, ())
            if upstream_key.component_kind.value not in allowed and allowed:
                raise ProgramAnalysisCacheValidationError(
                    f"{cache_key.component_kind.value} cannot depend on "
                    f"{upstream_key.component_kind.value}"
                )
            lookup = self.lookup(
                upstream_key,
                require_completion_evidence=False,
                require_runtime_artifact=True,
            )
            if not lookup.hit or lookup.runtime_artifact is None:
                raise ProgramAnalysisCacheValidationError(
                    f"upstream component missing: {upstream_key.component_kind.value}"
                )
            dependencies.append(lookup.runtime_artifact)
        return self.put(
            cache_key,
            receipt,
            status=status,
            ttl_seconds=ttl_seconds,
            blob_bodies=blob_bodies,
            dependencies=dependencies,
        )

    def get_or_compute(
        self,
        key: ProgramAnalysisCacheKey | Mapping[str, Any],
        producer: Callable[[], Any],
        *,
        ttl_seconds: int | None = None,
        wait_timeout_seconds: float | None = None,
        require_completion_evidence: bool = True,
        store_runtime_artifact: bool = True,
        dependencies: Sequence[
            RuntimeArtifactRecord | Mapping[str, Any] | str
        ] = (),
    ) -> ProgramAnalysisLookupResult:
        """Return a completion hit or run ``producer`` once under single-flight.

        Concurrent callers sharing an exact key collapse to one producer both
        inside the process and across processes that share this cache path.
        """

        cache_key = self._coerce_key(key)
        analysis_key = cache_key.to_analysis_cache_key()

        def _completion_validator(lookup: AnalysisCacheLookupResult) -> bool:
            if not lookup.is_completion_evidence:
                return False
            program_result = self.lookup(
                cache_key,
                require_completion_evidence=require_completion_evidence,
                require_runtime_artifact=store_runtime_artifact,
            )
            return program_result.is_completion_evidence

        def _wrapped_producer() -> Any:
            value = producer()
            if isinstance(value, CachePublication):
                inner = value.value
                store = value.store
                pub_ttl = value.ttl_seconds
            else:
                inner = value
                store = True
                pub_ttl = ttl_seconds
            if isinstance(inner, ProgramAnalysisStoreResult):
                if not inner.stored:
                    raise ProgramAnalysisCacheError(
                        "producer returned a failed store result"
                    )
                return CachePublication(
                    inner.entry.receipt if inner.entry is not None else {},
                    store=False,
                    ttl_seconds=pub_ttl,
                )
            if isinstance(inner, ProgramAnalysisLookupResult):
                if inner.receipt is None:
                    raise ProgramAnalysisCacheError(
                        "producer returned a non-hit lookup result"
                    )
                return CachePublication(
                    dict(inner.receipt),
                    store=False,
                    ttl_seconds=pub_ttl,
                )
            if not store:
                return CachePublication(inner, store=False, ttl_seconds=pub_ttl)
            stored = self.put(
                cache_key,
                inner,
                ttl_seconds=pub_ttl,
                dependencies=dependencies,
                store_runtime_artifact=store_runtime_artifact,
            )
            if not stored.stored:
                raise ProgramAnalysisCacheError(
                    "failed to persist program analysis receipt: "
                    + ",".join(stored.reason_codes)
                )
            # Already persisted; coordinator must not double-store.
            return CachePublication(
                stored.entry.receipt if stored.entry is not None else {},
                store=False,
                ttl_seconds=pub_ttl,
            )

        coordination = self.coordinator.get_or_compute(
            analysis_key,
            _wrapped_producer,
            ttl_seconds=ttl_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            completion_validator=(
                _completion_validator if require_completion_evidence else None
            ),
        )
        # Re-read through the program facade so runtime invalidation and
        # authority isolation always gate completion.
        result = self.lookup(
            cache_key,
            require_completion_evidence=require_completion_evidence,
            require_runtime_artifact=store_runtime_artifact,
        )
        if result.hit:
            return replace(result, coordination=coordination)
        # Shared non-completion outcomes (negative TTL entries) may still be
        # visible without counting as completion evidence.
        if coordination.lookup is not None and coordination.lookup.entry is not None:
            return ProgramAnalysisLookupResult(
                ProgramAnalysisLookupStatus.HIT
                if coordination.lookup.hit
                else ProgramAnalysisLookupStatus.INVALIDATED,
                cache_key,
                entry=coordination.lookup.entry,
                reason_codes=result.reason_codes
                or tuple(coordination.lookup.reason_codes),
                coordination=coordination,
            )
        return replace(result, coordination=coordination)

    def invalidate_component(
        self,
        key: ProgramAnalysisCacheKey | Mapping[str, Any],
        *,
        include_root: bool = True,
        reason: str = "program_analysis_component_changed",
    ) -> Mapping[str, Any]:
        """Tombstone one component and only its RuntimeCAS dependents."""

        cache_key = self._coerce_key(key)
        with self._index_lock:
            artifact_id = self._component_index.get(cache_key.key_id, "")
        if not artifact_id:
            lookup = self.lookup(
                cache_key,
                require_completion_evidence=False,
                require_runtime_artifact=True,
            )
            if lookup.runtime_artifact is not None:
                artifact_id = lookup.runtime_artifact.artifact_id
        if not artifact_id:
            return {
                "invalidated_artifact_ids": (),
                "preserved_artifact_ids": (),
                "reason": reason,
                "root_key_id": cache_key.key_id,
            }
        invalidation = self.runtime_cas.invalidate(
            artifact_id,
            include_root=include_root,
            reason=reason,
        )
        # Drop compact receipts whose runtime envelope was tombstoned.
        removed_keys: list[str] = []
        for path in list(self.analysis_cache._entry_paths()):  # noqa: SLF001
            try:
                entry = self.analysis_cache._read_path(path)  # noqa: SLF001
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            runtime_id = str(entry.receipt.get(_RUNTIME_ARTIFACT_FIELD) or "")
            if runtime_id in set(invalidation.invalidated_artifact_ids):
                try:
                    path.unlink()
                except OSError:
                    pass
                program_key = _extract_program_key(entry.receipt)
                if program_key is not None:
                    removed_keys.append(program_key.key_id)
        if removed_keys:
            self._unregister_index(removed_keys)
        return {
            "invalidated_artifact_ids": invalidation.invalidated_artifact_ids,
            "preserved_artifact_ids": invalidation.preserved_artifact_ids,
            "reason": reason,
            "root_key_id": cache_key.key_id,
            "removed_key_ids": tuple(removed_keys),
        }

    def invalidate_dimension(
        self,
        *,
        forest_identity: Any | None = None,
        objective_revision: Any | None = None,
        policy_revision: Any | None = None,
        analyzer_version: Any | None = None,
        schema_version: Any | None = None,
        configuration_digest: Any | None = None,
        query_digest: Any | None = None,
        capability_revision: Any | None = None,
        assumption_digest: Any | None = None,
        toolchain_version: Any | None = None,
        component_kind: ProgramAnalysisComponentKind | str | None = None,
        authority: ProgramAnalysisAuthority | str | None = None,
        reason: str = "program_analysis_dimension_changed",
    ) -> Mapping[str, Any]:
        """Invalidate every entry matching the supplied dimension filters."""

        filters: dict[str, Any] = {}
        if forest_identity is not None:
            filters["forest_identity"] = _identity_component(
                forest_identity, "forest_identity"
            )
        if objective_revision is not None:
            filters["objective_revision"] = _identity_component(
                objective_revision, "objective_revision"
            )
        if policy_revision is not None:
            filters["policy_revision"] = _identity_component(
                policy_revision, "policy_revision"
            )
        if analyzer_version is not None:
            filters["analyzer_version"] = _identity_component(
                analyzer_version, "analyzer_version"
            )
        if schema_version is not None:
            filters["schema_version"] = _identity_component(
                schema_version, "schema_version"
            )
        if configuration_digest is not None:
            filters["configuration_digest"] = _identity_component(
                configuration_digest, "configuration_digest"
            )
        if query_digest is not None:
            filters["query_digest"] = _identity_component(
                query_digest, "query_digest"
            )
        if capability_revision is not None:
            filters["capability_revision"] = _identity_component(
                capability_revision, "capability_revision"
            )
        if assumption_digest is not None:
            filters["assumption_digest"] = _identity_component(
                assumption_digest, "assumption_digest"
            )
        if toolchain_version is not None:
            filters["toolchain_version"] = _identity_component(
                toolchain_version, "toolchain_version"
            )
        if component_kind is not None:
            filters["component_kind"] = ProgramAnalysisComponentKind.coerce(
                component_kind
            ).value
        if authority is not None:
            filters["authority"] = ProgramAnalysisAuthority.coerce(authority).value
        if not filters:
            raise ProgramAnalysisCacheValidationError(
                "invalidate_dimension requires at least one filter"
            )

        removed = 0
        preserved = 0
        removed_key_ids: list[str] = []
        runtime_ids: list[str] = []
        for path in list(self.analysis_cache._entry_paths()):  # noqa: SLF001
            try:
                entry = self.analysis_cache._read_path(path)  # noqa: SLF001
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
                continue
            program_key = _extract_program_key(entry.receipt)
            if program_key is None:
                preserved += 1
                continue
            values = program_key.dimension_values()
            if all(values.get(name) == expected for name, expected in filters.items()):
                runtime_id = str(entry.receipt.get(_RUNTIME_ARTIFACT_FIELD) or "")
                if runtime_id:
                    runtime_ids.append(runtime_id)
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
                removed_key_ids.append(program_key.key_id)
            else:
                preserved += 1

        invalidated_runtime: list[str] = []
        for artifact_id in sorted(set(runtime_ids)):
            try:
                result = self.runtime_cas.invalidate(
                    artifact_id, include_root=True, reason=reason
                )
                invalidated_runtime.extend(result.invalidated_artifact_ids)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                continue
        if removed_key_ids:
            self._unregister_index(removed_key_ids)
        return {
            "removed_entries": removed,
            "preserved_entries": preserved,
            "removed_key_ids": tuple(removed_key_ids),
            "invalidated_artifact_ids": tuple(sorted(set(invalidated_runtime))),
            "reason": reason,
            "filters": filters,
        }

    def invalidate_semantic_dependency(
        self,
        dependency: SemanticDependencyIdentity | Mapping[str, Any] | str,
        *,
        reason: str = "semantic_dependency_changed",
    ) -> Mapping[str, Any]:
        """Invalidate RuntimeCAS dependents of one semantic dependency."""

        result = self.runtime_cas.invalidate_semantic_dependency(
            dependency, reason=reason
        )
        removed_key_ids: list[str] = []
        for path in list(self.analysis_cache._entry_paths()):  # noqa: SLF001
            try:
                entry = self.analysis_cache._read_path(path)  # noqa: SLF001
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                continue
            runtime_id = str(entry.receipt.get(_RUNTIME_ARTIFACT_FIELD) or "")
            if runtime_id in set(result.invalidated_artifact_ids):
                try:
                    path.unlink()
                except OSError:
                    pass
                program_key = _extract_program_key(entry.receipt)
                if program_key is not None:
                    removed_key_ids.append(program_key.key_id)
        if removed_key_ids:
            self._unregister_index(removed_key_ids)
        return {
            "invalidated_artifact_ids": result.invalidated_artifact_ids,
            "preserved_artifact_ids": result.preserved_artifact_ids,
            "removed_key_ids": tuple(removed_key_ids),
            "reason": reason,
        }

    def prune(self) -> int:
        """Garbage-collect expired/corrupt receipts and enforce quotas."""

        removed = self.analysis_cache.prune()
        try:
            self.artifact_store.compact()
        except Exception:  # noqa: BLE001 - compaction is best-effort
            pass
        self._rebuild_component_index()
        return removed

    def stats(self) -> ProgramAnalysisCacheStats:
        """Return receipt and large-body usage with their declared bounds."""

        base = self.analysis_cache.stats()
        component_counts: dict[str, int] = {
            kind.value: 0 for kind in ProgramAnalysisComponentKind
        }
        runtime_count = 0
        for path in self.analysis_cache._entry_paths():  # noqa: SLF001
            try:
                entry = self.analysis_cache._read_path(path)  # noqa: SLF001
            except (OSError, TypeError, ValueError, ReceiptValidationError):
                continue
            program_key = _extract_program_key(entry.receipt)
            if program_key is not None:
                component_counts[program_key.component_kind.value] = (
                    component_counts.get(program_key.component_kind.value, 0) + 1
                )
            if entry.receipt.get(_RUNTIME_ARTIFACT_FIELD):
                runtime_count += 1
        artifact_usage: Mapping[str, Any] = {}
        artifact_quotas: Mapping[str, Any] = (
            self.artifact_store.quotas.to_dict()
        )
        try:
            artifact_usage = self.artifact_store.usage()
        except Exception:  # noqa: BLE001
            artifact_usage = {}
        usage_quotas = artifact_usage.get("quotas")
        if isinstance(usage_quotas, Mapping):
            artifact_quotas = usage_quotas
        return ProgramAnalysisCacheStats(
            entry_count=base.entry_count,
            total_bytes=base.total_bytes,
            successful_count=base.successful_count,
            partial_count=base.partial_count,
            failed_count=base.failed_count,
            timed_out_count=base.timed_out_count,
            inconclusive_count=base.inconclusive_count,
            corrupt_count=base.corrupt_count,
            runtime_artifact_count=runtime_count,
            blob_count=int(artifact_usage.get("blob_count", 0)),
            artifact_bytes=int(artifact_usage.get("total_bytes", 0)),
            max_artifacts=int(artifact_quotas.get("max_blobs", 0)),
            max_artifact_bytes=int(artifact_quotas.get("max_bytes", 0)),
            component_counts=component_counts,
        )

    def clear(self) -> int:
        removed = self.analysis_cache.clear()
        with self._index_lock:
            self._component_index.clear()
        try:
            self._index_path().unlink()
        except OSError:
            pass
        return removed

    def reopen(self) -> "ProgramAnalysisCache":
        """Return a new facade bound to the same durable path (restart simulation)."""

        return ProgramAnalysisCache(
            self.path,
            max_entries=self.analysis_cache.max_entries,
            max_bytes=self.analysis_cache.max_bytes,
            max_entry_bytes=self.analysis_cache.max_entry_bytes,
            max_receipt_bytes=self.max_receipt_bytes,
            default_negative_ttl_seconds=self.default_negative_ttl_seconds,
            max_negative_ttl_seconds=self.max_negative_ttl_seconds,
            default_success_ttl_seconds=self.default_success_ttl_seconds,
            lock_timeout_seconds=self.analysis_cache.lock_timeout_seconds,
            wait_timeout_seconds=self.coordinator.wait_timeout_seconds,
            max_artifact_blob_bytes=self.artifact_store.quotas.max_blob_bytes,
            max_artifact_bytes=self.artifact_store.quotas.max_bytes,
            max_artifacts=self.artifact_store.quotas.max_blobs,
            clock=self._clock,
        )


# Compatibility aliases matching sibling cache modules.
DependencyAwareProgramAnalysisCache = ProgramAnalysisCache
ProgramAnalysisCacheEntry = AnalysisCacheEntry
ProgramAnalysisReceipt = AnalysisReceipt


__all__ = [
    "CACHE_INVALIDATION_PROOF_EVIDENCE",
    "DEPENDENCY_CACHE_EVIDENCE",
    "DEPENDENCY_CACHE_REQUIREMENT_ID",
    "DEPENDENCY_CAS_REQUIREMENT_ID",
    "AnalysisComponentKind",
    "ComponentKind",
    "DependencyAwareProgramAnalysisCache",
    "LookupStatus",
    "PROGRAM_ANALYSIS_CACHE_ENTRY_SCHEMA",
    "PROGRAM_ANALYSIS_CACHE_KEY_SCHEMA",
    "PROGRAM_ANALYSIS_CACHE_SCHEMA",
    "PROGRAM_ANALYSIS_RECEIPT_SCHEMA",
    "ProgramAnalysisAuthority",
    "ProgramAnalysisCache",
    "ProgramAnalysisCacheEntry",
    "ProgramAnalysisCacheError",
    "ProgramAnalysisCacheKey",
    "ProgramAnalysisCacheReason",
    "ProgramAnalysisCacheStats",
    "ProgramAnalysisCacheValidationError",
    "ProgramAnalysisComponentKind",
    "ProgramAnalysisLookupResult",
    "ProgramAnalysisLookupStatus",
    "ProgramAnalysisReceipt",
    "ProgramAnalysisStoreResult",
    "build_program_analysis_cache_key",
    "canonical_program_analysis_json",
    "compact_program_analysis_receipt",
    "digest_program_analysis_input",
    "make_program_analysis_cache_key",
]
