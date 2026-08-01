"""Typed registry for bounded, read-only supervisor analysis operations.

The registry is the policy boundary above :mod:`analysis_transport`.  It
describes *what* an operation is allowed to do independently from *where* it
runs, registers local and optional datasets producers without activating
either, and normalizes both paths to the transport's compact reference shape.

Nothing registered here can edit a checkout, select validation omissions, or
accept its own proposal.  Results are diagnostic references and candidates;
existing validators, proof kernels, merge gates, and completion policy retain
their authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .analysis_transport import (
    ANALYSIS_TRANSPORT_REQUEST_SCHEMA,
    ANALYSIS_TRANSPORT_RESULT_SCHEMA,
    AnalysisCapability,
    AnalysisProviderHealth,
    AnalysisProviderKind,
    AnalysisRequest,
    AnalysisResult,
    AnalysisTransport,
    AnalysisTransportBounds,
    AnalysisTransportError,
    AnalysisTransportPolicy,
)


ANALYSIS_OPERATION_REGISTRY_VERSION: Final[int] = 1
ANALYSIS_OPERATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-operation@1"
)
ANALYSIS_PRODUCER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-producer@1"
)
ANALYSIS_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-reference@1"
)
LOCAL_ANALYSIS_PRODUCER_ID: Final[str] = "supervisor-local-analysis"
IPFS_DATASETS_ANALYSIS_PRODUCER_ID: Final[str] = "ipfs-datasets-analysis"

_MAX_TEXT_BYTES: Final[int] = 8 * 1024


class AnalysisOperationRegistryError(AnalysisTransportError):
    """An operation, producer, or normalized reference is invalid."""


class AnalysisOperation(str, Enum):
    """Canonical operation names routed through the shared transport."""

    SYMBOL_IMPACT = "symbol_impact"
    AST_SYMBOL_IMPACT = "symbol_impact"
    GRAPH_RAG_RETRIEVAL = "graphrag_retrieval"
    GRAPH_RETRIEVAL = "graphrag_retrieval"
    PREMISE_SELECTION = "premise_selection"
    CONTRADICTION_SEARCH = "contradiction_search"
    LOGIC_TRANSLATION = "logic_translation"
    LEGAL_LOGIC_TRANSLATION = "logic_translation"
    PROOF_CANDIDATE_ANALYSIS = "proof_candidate_analysis"
    PROOF_CANDIDATE_SELECTION = "proof_candidate_analysis"
    COUNTEREXAMPLE_CANDIDATE_ANALYSIS = "counterexample_candidate_analysis"
    COUNTEREXAMPLE_ANALYSIS = "counterexample_candidate_analysis"


class LogicFamily(str, Enum):
    """Families remain explicit; no generic "logic" label erases semantics."""

    TDFOL = "tdfol"
    DCEC = "dcec"
    FLOGIC = "flogic"
    MODAL = "modal"
    DEONTIC = "deontic"
    FRAME = "frame"
    KNOWLEDGE_GRAPH = "kg"
    KG = "kg"
    EVENT_CALCULUS = "event_calculus"


class CacheScope(str, Enum):
    TREE = "exact_tree"
    OBJECTIVE = "objective_revision"
    REQUEST = "request"
    NONE = "none"


class ProvenanceRequirement(str, Enum):
    REPOSITORY = "repository_id"
    TREE = "tree_id"
    OBJECTIVE = "objective_revision"
    ARTIFACT = "artifact_reference"
    PRODUCER = "producer_id"
    CAPABILITY = "capability_revision"
    POLICY = "policy_id"
    LOGIC_FAMILY = "logic_family"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise AnalysisOperationRegistryError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise AnalysisOperationRegistryError(f"{name} must not be empty")
    if "\x00" in result:
        raise AnalysisOperationRegistryError(f"{name} must not contain NUL bytes")
    if len(result.encode("utf-8")) > maximum:
        raise AnalysisOperationRegistryError(
            f"{name} exceeds {maximum} UTF-8 bytes"
        )
    return result


def _positive_int(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise AnalysisOperationRegistryError(
            f"{name} must be an integer from 1 through {maximum}"
        )
    return value


def _canonical(value: Any, *, name: str = "value", depth: int = 0) -> Any:
    if depth > 12:
        raise AnalysisOperationRegistryError(f"{name} exceeds maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisOperationRegistryError(f"{name} must be finite")
        return format(value, ".17g")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AnalysisOperationRegistryError(f"{name} keys must be strings")
        return {
            key: _canonical(item, name=name, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_canonical(item, name=name, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict(), name=name, depth=depth + 1)
    raise AnalysisOperationRegistryError(
        f"{name} contains unsupported {type(value).__name__}"
    )


def _content_id(namespace: str, value: Any) -> str:
    encoded = json.dumps(
        _canonical(value, name=namespace),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


_OPERATION_ALIASES: Final = {
    "ast": AnalysisOperation.SYMBOL_IMPACT,
    "ast_impact": AnalysisOperation.SYMBOL_IMPACT,
    "ast_symbol_impact": AnalysisOperation.SYMBOL_IMPACT,
    "symbol": AnalysisOperation.SYMBOL_IMPACT,
    "graph": AnalysisOperation.GRAPH_RAG_RETRIEVAL,
    "graph_retrieval": AnalysisOperation.GRAPH_RAG_RETRIEVAL,
    "graphrag": AnalysisOperation.GRAPH_RAG_RETRIEVAL,
    "premises": AnalysisOperation.PREMISE_SELECTION,
    "contradictions": AnalysisOperation.CONTRADICTION_SEARCH,
    "legal_logic_analysis": AnalysisOperation.LOGIC_TRANSLATION,
    "legal_logic_translation": AnalysisOperation.LOGIC_TRANSLATION,
    "proof_candidate": AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
    "proof_candidates": AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
    "proof_candidate_selection": AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
    "counterexample": AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
    "counterexample_candidate": AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
}


def normalize_analysis_operation(value: Any) -> AnalysisOperation:
    if isinstance(value, AnalysisOperation):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    if raw in _OPERATION_ALIASES:
        return _OPERATION_ALIASES[raw]
    try:
        return AnalysisOperation(raw)
    except ValueError as exc:
        raise AnalysisOperationRegistryError(
            "unknown analysis operation: " + str(value)
        ) from exc


_FAMILY_ALIASES: Final = {
    "first_order_temporal_deontic": LogicFamily.TDFOL,
    "temporal_deontic_first_order": LogicFamily.TDFOL,
    "cognitive_event_calculus": LogicFamily.DCEC,
    "cec": LogicFamily.DCEC,
    "f_logic": LogicFamily.FLOGIC,
    "frame_logic": LogicFamily.FRAME,
    "knowledge_graph": LogicFamily.KNOWLEDGE_GRAPH,
    "knowledge_graphs": LogicFamily.KNOWLEDGE_GRAPH,
    "event-calculus": LogicFamily.EVENT_CALCULUS,
}


def normalize_logic_family(value: Any) -> LogicFamily:
    if isinstance(value, LogicFamily):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    if raw in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[raw]
    try:
        return LogicFamily(raw)
    except ValueError as exc:
        raise AnalysisOperationRegistryError(
            "unknown logic family: " + str(value)
        ) from exc


def to_canonical_logic_family_id(value: Any) -> str:
    """Project a supervisor analysis family onto the datasets family_id space.

    Delegates to :mod:`canonical_logic_adapter` so routing/cache code can use
    one shared vocabulary without importing the datasets package.
    """

    from ..canonical_logic_adapter import map_analysis_family_to_canonical

    try:
        return map_analysis_family_to_canonical(value)
    except Exception as exc:
        raise AnalysisOperationRegistryError(
            "cannot project logic family to canonical id: " + str(value)
        ) from exc


@dataclass(frozen=True)
class AnalysisCacheSemantics:
    """Cache identity and reuse constraints for one operation."""

    cacheable: bool = True
    content_addressed: bool = True
    scope: CacheScope = CacheScope.TREE
    key_dimensions: tuple[str, ...] = (
        "operation",
        "repository_id",
        "tree_id",
        "objective_revision",
        "question",
        "artifact_references",
        "policy_id",
        "capability_revision",
        "logic_family",
    )
    allow_stale: bool = False
    reuse_requires_equivalent_provenance: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", CacheScope(self.scope))
        dimensions = tuple(
            sorted({_text(item, "cache key dimension", maximum=128) for item in self.key_dimensions})
        )
        if self.cacheable and not dimensions:
            raise AnalysisOperationRegistryError(
                "cacheable operations require key dimensions"
            )
        if self.scope is CacheScope.TREE and "tree_id" not in dimensions:
            raise AnalysisOperationRegistryError(
                "exact-tree cache semantics require tree_id"
            )
        if self.allow_stale:
            raise AnalysisOperationRegistryError(
                "analysis operations cannot reuse stale evidence"
            )
        object.__setattr__(self, "key_dimensions", dimensions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cacheable": self.cacheable,
            "content_addressed": self.content_addressed,
            "scope": self.scope.value,
            "key_dimensions": list(self.key_dimensions),
            "allow_stale": False,
            "reuse_requires_equivalent_provenance": (
                self.reuse_requires_equivalent_provenance
            ),
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisCacheSemantics | Mapping[str, Any] | None"
    ) -> "AnalysisCacheSemantics":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError("cache semantics must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown cache semantics: " + ", ".join(sorted(unknown))
            )
        fields = dict(value)
        if "key_dimensions" in fields:
            fields["key_dimensions"] = tuple(fields["key_dimensions"])
        return cls(**fields)


@dataclass(frozen=True)
class AnalysisOperationBounds:
    """Operation-specific bounds, always within transport hard bounds."""

    max_question_bytes: int = 8 * 1024
    max_artifact_references: int = 64
    max_evidence_references: int = 64
    max_provenance_references: int = 64
    max_reference_bytes: int = 8 * 1024
    max_batch_size: int = 16
    timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        maxima = {
            "max_question_bytes": 1024 * 1024,
            "max_artifact_references": 4096,
            "max_evidence_references": 4096,
            "max_provenance_references": 4096,
            "max_reference_bytes": 256 * 1024,
            "max_batch_size": 256,
            "timeout_ms": 10 * 60 * 1000,
        }
        for name, maximum in maxima.items():
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name, maximum)
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisOperationBounds | Mapping[str, Any] | None"
    ) -> "AnalysisOperationBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError("operation bounds must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown operation bounds: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class AnalysisProvenanceSemantics:
    required: tuple[ProvenanceRequirement, ...] = (
        ProvenanceRequirement.REPOSITORY,
        ProvenanceRequirement.TREE,
        ProvenanceRequirement.OBJECTIVE,
        ProvenanceRequirement.ARTIFACT,
        ProvenanceRequirement.PRODUCER,
        ProvenanceRequirement.CAPABILITY,
        ProvenanceRequirement.POLICY,
    )
    preserve_source_references: bool = True
    content_ids_required: bool = True

    def __post_init__(self) -> None:
        requirements = tuple(
            sorted(
                {ProvenanceRequirement(item) for item in self.required},
                key=lambda item: item.value,
            )
        )
        if not requirements:
            raise AnalysisOperationRegistryError(
                "provenance requirements must not be empty"
            )
        object.__setattr__(self, "required", requirements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": [item.value for item in self.required],
            "preserve_source_references": self.preserve_source_references,
            "content_ids_required": self.content_ids_required,
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisProvenanceSemantics | Mapping[str, Any] | None"
    ) -> "AnalysisProvenanceSemantics":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError(
                "provenance semantics must be an object"
            )
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown provenance semantics: " + ", ".join(sorted(unknown))
            )
        fields = dict(value)
        if "required" in fields:
            fields["required"] = tuple(fields["required"])
        return cls(**fields)


@dataclass(frozen=True)
class AnalysisAuthoritySemantics:
    """Fixed trust boundary.  There are deliberately no enabling switches."""

    verdict_tier: str = "diagnostic_candidate"

    def __post_init__(self) -> None:
        if self.verdict_tier != "diagnostic_candidate":
            raise AnalysisOperationRegistryError(
                "registered analysis must remain diagnostic_candidate"
            )

    @property
    def repository_mutation(self) -> bool:
        return False

    @property
    def validation_omission_selection(self) -> bool:
        return False

    @property
    def candidate_promotion(self) -> bool:
        return False

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def completion_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_tier": self.verdict_tier,
            "repository_mutation": False,
            "validation_omission_selection": False,
            "candidate_promotion": False,
            "proof_authority": False,
            "completion_authority": False,
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisAuthoritySemantics | Mapping[str, Any] | None"
    ) -> "AnalysisAuthoritySemantics":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError("authority must be an object")
        allowed = {
            "verdict_tier",
            "repository_mutation",
            "validation_omission_selection",
            "candidate_promotion",
            "proof_authority",
            "completion_authority",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown authority semantics: " + ", ".join(sorted(unknown))
            )
        fixed_false = allowed - {"verdict_tier"}
        forged = [name for name in fixed_false if value.get(name, False) is not False]
        if forged:
            raise AnalysisOperationRegistryError(
                "analysis declaration claims forbidden authority: "
                + ", ".join(sorted(forged))
            )
        return cls(verdict_tier=value.get("verdict_tier", "diagnostic_candidate"))


@dataclass(frozen=True)
class AnalysisFallbackSemantics:
    strategy: str = "deterministic_local"
    provider_id: str = LOCAL_ANALYSIS_PRODUCER_ID
    explicit_receipt: bool = True
    fail_closed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy", _text(self.strategy, "fallback strategy", maximum=128)
        )
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, "fallback provider_id", maximum=256)
        )
        if self.strategy != "deterministic_local":
            raise AnalysisOperationRegistryError(
                "analysis fallback must be deterministic_local"
            )
        if not self.explicit_receipt or not self.fail_closed:
            raise AnalysisOperationRegistryError(
                "analysis fallback must be explicit and fail closed"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider_id": self.provider_id,
            "explicit_receipt": True,
            "fail_closed": True,
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisFallbackSemantics | Mapping[str, Any] | None"
    ) -> "AnalysisFallbackSemantics":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError("fallback must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown fallback semantics: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class AnalysisBatchingSemantics:
    supported: bool = True
    max_batch_size: int = 16
    same_operation_required: bool = True
    same_tree_required: bool = True
    preserve_member_identity: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_batch_size",
            _positive_int(self.max_batch_size, "max_batch_size", 256),
        )
        if not self.supported and self.max_batch_size != 1:
            raise AnalysisOperationRegistryError(
                "non-batch operation max_batch_size must be 1"
            )
        if not self.same_operation_required or not self.same_tree_required:
            raise AnalysisOperationRegistryError(
                "analysis batches must share operation and tree"
            )
        if not self.preserve_member_identity:
            raise AnalysisOperationRegistryError(
                "analysis batches must preserve member identity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "max_batch_size": self.max_batch_size,
            "same_operation_required": True,
            "same_tree_required": True,
            "preserve_member_identity": True,
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisBatchingSemantics | Mapping[str, Any] | None"
    ) -> "AnalysisBatchingSemantics":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError("batching must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown batching semantics: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class AnalysisOperationSpec:
    """Complete declaration for one routed operation."""

    operation: AnalysisOperation
    cache: AnalysisCacheSemantics = field(default_factory=AnalysisCacheSemantics)
    bounds: AnalysisOperationBounds = field(default_factory=AnalysisOperationBounds)
    provenance: AnalysisProvenanceSemantics = field(
        default_factory=AnalysisProvenanceSemantics
    )
    authority: AnalysisAuthoritySemantics = field(
        default_factory=AnalysisAuthoritySemantics
    )
    fallback: AnalysisFallbackSemantics = field(
        default_factory=AnalysisFallbackSemantics
    )
    batching: AnalysisBatchingSemantics = field(
        default_factory=AnalysisBatchingSemantics
    )
    capability_requirements: tuple[str, ...] = ()
    logic_families: tuple[LogicFamily, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", normalize_analysis_operation(self.operation)
        )
        requirements = tuple(
            sorted(
                {
                    _text(item, "capability requirement", maximum=256)
                    for item in self.capability_requirements
                }
            )
        )
        if not requirements:
            raise AnalysisOperationRegistryError(
                f"{self.operation.value} must declare capability requirements"
            )
        object.__setattr__(self, "capability_requirements", requirements)
        families = tuple(
            sorted(
                {normalize_logic_family(item) for item in self.logic_families},
                key=lambda item: item.value,
            )
        )
        logical = self.operation in {
            AnalysisOperation.PREMISE_SELECTION,
            AnalysisOperation.CONTRADICTION_SEARCH,
            AnalysisOperation.LOGIC_TRANSLATION,
            AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
            AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
        }
        if logical and not families:
            raise AnalysisOperationRegistryError(
                f"{self.operation.value} must retain supported logic families"
            )
        if (
            ProvenanceRequirement.LOGIC_FAMILY not in self.provenance.required
            and logical
        ):
            raise AnalysisOperationRegistryError(
                f"{self.operation.value} provenance must retain logic_family"
            )
        object.__setattr__(self, "logic_families", families)
        if self.batching.max_batch_size > self.bounds.max_batch_size:
            raise AnalysisOperationRegistryError(
                "batching max_batch_size exceeds operation bounds"
            )

    @property
    def operation_id(self) -> str:
        return self.operation.value

    @property
    def spec_id(self) -> str:
        return _content_id("analysis-operation", self._payload())

    def supports_family(self, family: Any) -> bool:
        return normalize_logic_family(family) in self.logic_families

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_OPERATION_SCHEMA,
            "registry_version": ANALYSIS_OPERATION_REGISTRY_VERSION,
            "operation": self.operation.value,
            "cache": self.cache.to_dict(),
            "bounds": self.bounds.to_dict(),
            "provenance": self.provenance.to_dict(),
            "authority": self.authority.to_dict(),
            "fallback": self.fallback.to_dict(),
            "batching": self.batching.to_dict(),
            "capability_requirements": list(self.capability_requirements),
            "logic_families": [item.value for item in self.logic_families],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"spec_id": self.spec_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisOperationSpec":
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError(
                "operation declaration must be an object"
            )
        allowed = {
            "schema",
            "registry_version",
            "spec_id",
            "operation",
            "cache",
            "bounds",
            "provenance",
            "authority",
            "fallback",
            "batching",
            "capability_requirements",
            "logic_families",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown operation declaration fields: "
                + ", ".join(sorted(unknown))
            )
        if value.get("schema", ANALYSIS_OPERATION_SCHEMA) != ANALYSIS_OPERATION_SCHEMA:
            raise AnalysisOperationRegistryError("unsupported operation schema")
        if value.get(
            "registry_version", ANALYSIS_OPERATION_REGISTRY_VERSION
        ) != ANALYSIS_OPERATION_REGISTRY_VERSION:
            raise AnalysisOperationRegistryError(
                "unsupported operation registry version"
            )
        result = cls(
            operation=value.get("operation", ""),
            cache=AnalysisCacheSemantics.from_value(value.get("cache")),
            bounds=AnalysisOperationBounds.from_value(value.get("bounds")),
            provenance=AnalysisProvenanceSemantics.from_value(
                value.get("provenance")
            ),
            authority=AnalysisAuthoritySemantics.from_value(value.get("authority")),
            fallback=AnalysisFallbackSemantics.from_value(value.get("fallback")),
            batching=AnalysisBatchingSemantics.from_value(value.get("batching")),
            capability_requirements=tuple(
                value.get("capability_requirements") or ()
            ),
            logic_families=tuple(value.get("logic_families") or ()),
        )
        claimed = value.get("spec_id")
        if claimed is not None and claimed != result.spec_id:
            raise AnalysisOperationRegistryError(
                "operation declaration identity does not match"
            )
        return result


@dataclass(frozen=True)
class AnalysisProducer:
    """Side-effect-free declaration for a local or optional producer."""

    producer_id: str
    provider_kind: AnalysisProviderKind
    operations: tuple[AnalysisOperation, ...]
    capability_revision: str
    provider_version: str = "1"
    capabilities: tuple[str, ...] = ()
    logic_families: tuple[LogicFamily, ...] = ()
    max_batch_size: int = 16
    max_concurrency: int = 1
    supports_cancellation: bool = True
    supports_progress: bool = False
    supports_batching: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "producer_id", _text(self.producer_id, "producer_id", maximum=256)
        )
        object.__setattr__(
            self,
            "provider_kind",
            AnalysisProviderKind(self.provider_kind),
        )
        operations = tuple(
            sorted(
                {normalize_analysis_operation(item) for item in self.operations},
                key=lambda item: item.value,
            )
        )
        if not operations:
            raise AnalysisOperationRegistryError(
                "producer operations must not be empty"
            )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "capability_revision",
            _text(self.capability_revision, "capability_revision", maximum=256),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(self.provider_version, "provider_version", maximum=256),
        )
        capabilities = tuple(
            sorted(
                {_text(item, "producer capability", maximum=256) for item in self.capabilities}
            )
        )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "logic_families",
            tuple(
                sorted(
                    {normalize_logic_family(item) for item in self.logic_families},
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "max_batch_size",
            _positive_int(self.max_batch_size, "max_batch_size", 256),
        )
        object.__setattr__(
            self,
            "max_concurrency",
            _positive_int(self.max_concurrency, "max_concurrency", 1024),
        )
        if not self.supports_batching and self.max_batch_size != 1:
            raise AnalysisOperationRegistryError(
                "non-batching producer max_batch_size must be 1"
            )

    @property
    def producer_declaration_id(self) -> str:
        return _content_id("analysis-producer", self._payload())

    @property
    def capability(self) -> AnalysisCapability:
        return AnalysisCapability(
            provider_id=self.producer_id,
            provider_kind=self.provider_kind,
            provider_version=self.provider_version,
            capability_revision=self.capability_revision,
            operations=tuple(item.value for item in self.operations),
            health=(
                AnalysisProviderHealth.HEALTHY
                if self.provider_kind is AnalysisProviderKind.LOCAL
                else AnalysisProviderHealth.LAZY
            ),
            max_batch_size=self.max_batch_size,
            max_concurrency=self.max_concurrency,
            supports_cancellation=self.supports_cancellation,
            supports_progress=self.supports_progress,
            supports_batching=self.supports_batching,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_PRODUCER_SCHEMA,
            "registry_version": ANALYSIS_OPERATION_REGISTRY_VERSION,
            "producer_id": self.producer_id,
            "provider_kind": self.provider_kind.value,
            "provider_version": self.provider_version,
            "capability_revision": self.capability_revision,
            "operations": [item.value for item in self.operations],
            "capabilities": list(self.capabilities),
            "logic_families": [item.value for item in self.logic_families],
            "max_batch_size": self.max_batch_size,
            "max_concurrency": self.max_concurrency,
            "supports_cancellation": self.supports_cancellation,
            "supports_progress": self.supports_progress,
            "supports_batching": self.supports_batching,
            "authority": AnalysisAuthoritySemantics().to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_declaration_id": self.producer_declaration_id,
            "transport_capability_id": self.capability.capability_id,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisProducer":
        if not isinstance(value, Mapping):
            raise AnalysisOperationRegistryError(
                "producer declaration must be an object"
            )
        allowed = {
            "schema",
            "registry_version",
            "producer_declaration_id",
            "transport_capability_id",
            "producer_id",
            "provider_kind",
            "provider_version",
            "capability_revision",
            "operations",
            "capabilities",
            "logic_families",
            "max_batch_size",
            "max_concurrency",
            "supports_cancellation",
            "supports_progress",
            "supports_batching",
            "authority",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisOperationRegistryError(
                "unknown producer declaration fields: "
                + ", ".join(sorted(unknown))
            )
        if value.get("schema", ANALYSIS_PRODUCER_SCHEMA) != ANALYSIS_PRODUCER_SCHEMA:
            raise AnalysisOperationRegistryError("unsupported producer schema")
        if value.get(
            "registry_version", ANALYSIS_OPERATION_REGISTRY_VERSION
        ) != ANALYSIS_OPERATION_REGISTRY_VERSION:
            raise AnalysisOperationRegistryError(
                "unsupported producer registry version"
            )
        AnalysisAuthoritySemantics.from_value(value.get("authority"))
        result = cls(
            producer_id=value.get("producer_id", ""),
            provider_kind=value.get("provider_kind", ""),
            provider_version=value.get("provider_version", "1"),
            capability_revision=value.get("capability_revision", ""),
            operations=tuple(value.get("operations") or ()),
            capabilities=tuple(value.get("capabilities") or ()),
            logic_families=tuple(value.get("logic_families") or ()),
            max_batch_size=value.get("max_batch_size", 16),
            max_concurrency=value.get("max_concurrency", 1),
            supports_cancellation=value.get("supports_cancellation", True),
            supports_progress=value.get("supports_progress", False),
            supports_batching=value.get("supports_batching", True),
        )
        claimed = value.get("producer_declaration_id")
        if claimed is not None and claimed != result.producer_declaration_id:
            raise AnalysisOperationRegistryError(
                "producer declaration identity does not match"
            )
        capability_id = value.get("transport_capability_id")
        if capability_id is not None and capability_id != result.capability.capability_id:
            raise AnalysisOperationRegistryError(
                "producer transport capability identity does not match"
            )
        return result


_REFERENCE_FIELDS: Final = (
    "artifact_content_id",
    "artifact_id",
    "byte_count",
    "chunk_id",
    "cid",
    "dataset_id",
    "digest",
    "evidence_id",
    "kind",
    "media_type",
    "model_id",
    "path",
    "producer_id",
    "provider_id",
    "record_id",
    "reference_id",
    "revision",
    "score_millionths",
    "sha256",
    "summary",
    "symbol",
    "tree_id",
    "uri",
)
_REFERENCE_ALIASES: Final = {
    "id": "reference_id",
    "content_id": "artifact_content_id",
    "artifact_cid": "artifact_content_id",
    "source_id": "artifact_id",
    "dataset": "dataset_id",
    "chunk": "chunk_id",
    "node_id": "record_id",
    "graph_node_id": "record_id",
    "score": "score_millionths",
}
_FORBIDDEN_REFERENCE_FIELDS: Final = frozenset(
    {
        "body",
        "content",
        "embedding",
        "file_contents",
        "graph",
        "model_output",
        "patch",
        "prompt",
        "raw",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "transcript",
    }
)


def normalize_analysis_reference(
    value: Mapping[str, Any] | Any,
    *,
    default_kind: str = "analysis",
    producer_id: str = "",
) -> Mapping[str, Any]:
    """Normalize local and remote references to the exact same compact shape."""

    converter = getattr(value, "to_dict", None)
    if not isinstance(value, Mapping) and callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise AnalysisOperationRegistryError("analysis reference must be an object")
    lowered = {str(key).casefold() for key in value}
    forbidden = lowered.intersection(_FORBIDDEN_REFERENCE_FIELDS)
    if forbidden:
        raise AnalysisOperationRegistryError(
            "analysis reference embeds forbidden payloads: "
            + ", ".join(sorted(forbidden))
        )
    normalized: dict[str, Any] = {}
    for original_key, raw in value.items():
        key = _REFERENCE_ALIASES.get(str(original_key), str(original_key))
        if key not in _REFERENCE_FIELDS:
            # Provider-specific decorations cannot leak into the stable shape.
            continue
        if raw in (None, ""):
            continue
        if key == "score_millionths":
            if isinstance(raw, bool):
                raise AnalysisOperationRegistryError("reference score is invalid")
            score = float(raw)
            if not math.isfinite(score):
                raise AnalysisOperationRegistryError("reference score must be finite")
            if str(original_key) == "score":
                score *= 1_000_000
            score_int = int(round(score))
            if not 0 <= score_int <= 1_000_000:
                raise AnalysisOperationRegistryError(
                    "reference score must be between zero and one"
                )
            normalized[key] = score_int
        elif key == "byte_count":
            normalized[key] = _positive_int(raw, "reference byte_count", 2**63 - 1)
        else:
            normalized[key] = _text(
                raw, f"reference {key}", required=False, maximum=2048
            )
    normalized.setdefault("kind", _text(default_kind, "default_kind", maximum=128))
    if producer_id:
        claimed = normalized.get("producer_id")
        if claimed and claimed != producer_id:
            raise AnalysisOperationRegistryError(
                "reference producer_id does not match active producer"
            )
        normalized["producer_id"] = producer_id
    if not normalized.get("reference_id"):
        identity = {
            key: item
            for key, item in normalized.items()
            if key not in {"reference_id", "producer_id", "provider_id"}
        }
        normalized["reference_id"] = _content_id("analysis-reference", identity)
    # Consistent field ordering also makes local/remote equality straightforward.
    ordered = {
        key: normalized[key] for key in _REFERENCE_FIELDS if key in normalized
    }
    return MappingProxyType(ordered)


def normalized_reference_payload(
    value: Mapping[str, Any] | Any,
    *,
    default_kind: str = "analysis",
    producer_id: str = "",
) -> dict[str, Any]:
    """Mutable JSON projection suitable for provider/transport responses."""

    return dict(
        normalize_analysis_reference(
            value, default_kind=default_kind, producer_id=producer_id
        )
    )


def _normalize_reference_collection(
    values: Sequence[Mapping[str, Any]],
    *,
    default_kind: str,
    producer_id: str = "",
) -> tuple[Mapping[str, Any], ...]:
    """Canonical sort/dedupe independent of provider scheduling order."""

    unique: dict[bytes, Mapping[str, Any]] = {}
    for value in values:
        normalized = normalize_analysis_reference(
            value,
            default_kind=default_kind,
            producer_id=producer_id,
        )
        encoded = json.dumps(
            dict(normalized),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        unique[encoded] = normalized
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class _ProducerRegistration:
    declaration: AnalysisProducer
    factory: Callable[[], Any] | None = None
    instance: Any = None


class AnalysisOperationRegistry:
    """Immutable-on-first-use registry backed by :class:`AnalysisTransport`."""

    def __init__(
        self,
        *,
        transport_bounds: AnalysisTransportBounds | Mapping[str, Any] | None = None,
    ) -> None:
        self.transport_bounds = AnalysisTransportBounds.from_value(transport_bounds)
        self._operations: dict[AnalysisOperation, AnalysisOperationSpec] = {}
        self._producers: dict[str, _ProducerRegistration] = {}
        self._producer_order: list[str] = []
        self._transport: AnalysisTransport | None = None

    @property
    def frozen(self) -> bool:
        return self._transport is not None

    def _require_mutable(self) -> None:
        if self.frozen:
            raise AnalysisOperationRegistryError(
                "analysis operation registry is frozen after first dispatch"
            )

    def register_operation(
        self,
        spec: AnalysisOperationSpec,
        *,
        replace_existing: bool = False,
    ) -> None:
        self._require_mutable()
        if not isinstance(spec, AnalysisOperationSpec):
            raise AnalysisOperationRegistryError(
                "operation must be an AnalysisOperationSpec"
            )
        if spec.operation in self._operations and not replace_existing:
            raise AnalysisOperationRegistryError(
                f"operation already registered: {spec.operation.value}"
            )
        self._operations[spec.operation] = spec

    def register_producer(
        self,
        declaration: AnalysisProducer,
        *,
        provider: Any = None,
        factory: Callable[[], Any] | None = None,
        replace_existing: bool = False,
    ) -> None:
        self._require_mutable()
        if not isinstance(declaration, AnalysisProducer):
            raise AnalysisOperationRegistryError(
                "producer must be an AnalysisProducer"
            )
        if (provider is None) == (factory is None):
            raise AnalysisOperationRegistryError(
                "supply exactly one of provider or factory"
            )
        if factory is not None and not callable(factory):
            raise AnalysisOperationRegistryError("producer factory must be callable")
        missing = [item.value for item in declaration.operations if item not in self._operations]
        if missing:
            raise AnalysisOperationRegistryError(
                "producer references unregistered operations: "
                + ", ".join(sorted(missing))
            )
        for operation in declaration.operations:
            spec = self._operations[operation]
            absent = set(spec.capability_requirements) - set(declaration.capabilities)
            if absent:
                raise AnalysisOperationRegistryError(
                    f"{declaration.producer_id} lacks {operation.value} capabilities: "
                    + ", ".join(sorted(absent))
                )
            if spec.logic_families and not set(spec.logic_families).issubset(
                declaration.logic_families
            ):
                raise AnalysisOperationRegistryError(
                    f"{declaration.producer_id} erases logic families for "
                    f"{operation.value}"
                )
        if declaration.producer_id in self._producers and not replace_existing:
            raise AnalysisOperationRegistryError(
                f"producer already registered: {declaration.producer_id}"
            )
        if declaration.producer_id not in self._producers:
            self._producer_order.append(declaration.producer_id)
        self._producers[declaration.producer_id] = _ProducerRegistration(
            declaration=declaration,
            factory=factory,
            instance=provider,
        )

    def operation(self, value: Any) -> AnalysisOperationSpec:
        operation = normalize_analysis_operation(value)
        try:
            return self._operations[operation]
        except KeyError as exc:
            raise AnalysisOperationRegistryError(
                f"analysis operation is not registered: {operation.value}"
            ) from exc

    get = operation

    def operations(self) -> tuple[AnalysisOperationSpec, ...]:
        return tuple(
            self._operations[item]
            for item in sorted(self._operations, key=lambda value: value.value)
        )

    list_operations = operations

    def producers(
        self, operation: Any | None = None
    ) -> tuple[AnalysisProducer, ...]:
        requested = (
            normalize_analysis_operation(operation) if operation is not None else None
        )
        result = []
        for producer_id in self._producer_order:
            declaration = self._producers[producer_id].declaration
            if requested is None or requested in declaration.operations:
                result.append(declaration)
        return tuple(result)

    list_producers = producers

    def discover_capabilities(
        self, operation: Any | None = None
    ) -> tuple[AnalysisCapability, ...]:
        return tuple(item.capability for item in self.producers(operation))

    @property
    def registry_id(self) -> str:
        return _content_id(
            "analysis-operation-registry",
            {
                "version": ANALYSIS_OPERATION_REGISTRY_VERSION,
                "operations": [item.to_dict() for item in self.operations()],
                "producers": [item.to_dict() for item in self.producers()],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/analysis-operation-registry@1",
            "registry_version": ANALYSIS_OPERATION_REGISTRY_VERSION,
            "registry_id": self.registry_id,
            "frozen": self.frozen,
            "operations": [item.to_dict() for item in self.operations()],
            "producers": [item.to_dict() for item in self.producers()],
            "authority": AnalysisAuthoritySemantics().to_dict(),
        }

    def build_request(
        self,
        operation: Any,
        question: str,
        *,
        artifact_references: Sequence[Mapping[str, Any]] = (),
        logic_family: LogicFamily | str | None = None,
        repository_id: str,
        tree_id: str,
        objective_revision: str,
        policy_id: str = "analysis-operation-policy:default",
        preferred_provider_id: str = "",
        timeout_ms: int | None = None,
        request_id: str = "",
    ) -> AnalysisRequest:
        spec = self.operation(operation)
        family = normalize_logic_family(logic_family) if logic_family is not None else None
        if spec.logic_families:
            if family is None:
                raise AnalysisOperationRegistryError(
                    f"{spec.operation.value} requires an explicit logic_family"
                )
            if family not in spec.logic_families:
                raise AnalysisOperationRegistryError(
                    f"{family.value} is not supported by {spec.operation.value}"
                )
        elif family is not None:
            raise AnalysisOperationRegistryError(
                f"{spec.operation.value} does not accept a logic_family"
            )
        normalized_refs = _normalize_reference_collection(
            artifact_references,
            default_kind="artifact",
        )
        if len(normalized_refs) > spec.bounds.max_artifact_references:
            raise AnalysisOperationRegistryError(
                "artifact references exceed operation bound"
            )
        normalized_question = _text(
            question,
            "question",
            maximum=spec.bounds.max_question_bytes,
        )
        metadata = {
            "registry_id": self.registry_id,
            "operation_spec_id": spec.spec_id,
            "repository_id": _text(repository_id, "repository_id", maximum=512),
            "tree_id": _text(tree_id, "tree_id", maximum=512),
            "objective_revision": _text(
                objective_revision, "objective_revision", maximum=512
            ),
            "policy_id": _text(policy_id, "policy_id", maximum=512),
            "logic_family": family.value if family is not None else "",
        }
        request = AnalysisRequest(
            operation=spec.operation.value,
            question=normalized_question,
            artifact_references=normalized_refs,
            request_id=request_id,
            preferred_provider_id=preferred_provider_id,
            timeout_ms=min(timeout_ms or spec.bounds.timeout_ms, spec.bounds.timeout_ms),
            metadata=metadata,
        )
        request.validate_bounds(self.transport_bounds)
        return request

    def _build_transport(self) -> AnalysisTransport:
        if self._transport is not None:
            return self._transport
        local_ids = [
            item.producer_id
            for item in self.producers()
            if item.provider_kind is AnalysisProviderKind.LOCAL
        ]
        if len(local_ids) != 1:
            raise AnalysisOperationRegistryError(
                "registry requires exactly one deterministic local fallback producer"
            )
        policy = AnalysisTransportPolicy(
            bounds=self.transport_bounds,
            fallback_provider_id=local_ids[0],
            request_schemas=(ANALYSIS_TRANSPORT_REQUEST_SCHEMA,),
            result_schemas=(ANALYSIS_TRANSPORT_RESULT_SCHEMA,),
        )
        transport = AnalysisTransport(policy=policy)
        # Optional producers are tried before the local fallback by default.
        ordered = sorted(
            self._producer_order,
            key=lambda producer_id: (
                self._producers[producer_id].declaration.provider_kind
                is AnalysisProviderKind.LOCAL,
                self._producer_order.index(producer_id),
            ),
        )
        for producer_id in ordered:
            registration = self._producers[producer_id]
            capability = registration.declaration.capability
            if registration.instance is not None:
                transport.register_provider(
                    capability, provider=registration.instance
                )
            else:
                assert registration.factory is not None
                transport.register_lazy_provider(
                    capability, factory=registration.factory
                )
        self._transport = transport
        return transport

    def _validate_dispatch_request(
        self,
        request: AnalysisRequest,
        *,
        provider_id: str = "",
    ) -> tuple[AnalysisOperationSpec, str]:
        spec = self.operation(request.operation)
        request.validate_bounds(self.transport_bounds)
        metadata = request.metadata
        if metadata.get("registry_id") != self.registry_id:
            raise AnalysisOperationRegistryError(
                "request is not bound to this registry revision"
            )
        if metadata.get("operation_spec_id") != spec.spec_id:
            raise AnalysisOperationRegistryError(
                "request is not bound to this operation declaration"
            )
        required_metadata = (
            "repository_id",
            "tree_id",
            "objective_revision",
            "policy_id",
        )
        for name in required_metadata:
            _text(metadata.get(name), name, maximum=512)
        if len(request.question.encode("utf-8")) > spec.bounds.max_question_bytes:
            raise AnalysisOperationRegistryError(
                "question exceeds operation bound"
            )
        if len(request.artifact_references) > spec.bounds.max_artifact_references:
            raise AnalysisOperationRegistryError(
                "artifact references exceed operation bound"
            )
        tree_id = metadata["tree_id"]
        for reference in request.artifact_references:
            encoded = json.dumps(
                _canonical(reference, name="artifact reference"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) > spec.bounds.max_reference_bytes:
                raise AnalysisOperationRegistryError(
                    "artifact reference exceeds operation byte bound"
                )
            reference_tree = reference.get("tree_id")
            if reference_tree and reference_tree != tree_id:
                raise AnalysisOperationRegistryError(
                    "artifact reference tree_id does not match request tree_id"
                )
        family_raw = metadata.get("logic_family")
        if spec.logic_families:
            if not family_raw or not spec.supports_family(family_raw):
                raise AnalysisOperationRegistryError(
                    "request logic_family does not match operation declaration"
                )
        elif family_raw:
            raise AnalysisOperationRegistryError(
                "request logic_family is not valid for this operation"
            )
        selected = _text(
            provider_id or request.preferred_provider_id,
            "provider_id",
            required=False,
            maximum=256,
        )
        if selected:
            declarations = {item.producer_id: item for item in self.producers()}
            declaration = declarations.get(selected)
            if declaration is None or spec.operation not in declaration.operations:
                raise AnalysisOperationRegistryError(
                    f"producer {selected!r} does not support {spec.operation.value}"
                )
        return spec, selected

    @staticmethod
    def _validate_result_references(
        references: Sequence[Mapping[str, Any]],
        *,
        maximum_count: int,
        maximum_bytes: int,
        name: str,
        expected_tree_id: str,
    ) -> None:
        if len(references) > maximum_count:
            raise AnalysisOperationRegistryError(
                f"{name} references exceed operation bound"
            )
        for reference in references:
            encoded = json.dumps(
                dict(reference),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) > maximum_bytes:
                raise AnalysisOperationRegistryError(
                    f"{name} reference exceeds operation byte bound"
                )
            reference_tree = reference.get("tree_id")
            if reference_tree and reference_tree != expected_tree_id:
                raise AnalysisOperationRegistryError(
                    f"{name} reference tree_id does not match request tree_id"
                )

    @staticmethod
    def _attach_logic_family_provenance(
        provenance: Sequence[Mapping[str, Any]],
        *,
        logic_family: str,
        producer_id: str,
        maximum_count: int,
    ) -> tuple[tuple[Mapping[str, Any], ...], bool]:
        family_reference = normalize_analysis_reference(
            {
                "kind": "logic_family",
                "record_id": logic_family,
                "summary": f"logic family: {logic_family}",
            },
            producer_id=producer_id,
        )
        family_id = family_reference["reference_id"]
        without_duplicate = tuple(
            item
            for item in provenance
            if item.get("reference_id") != family_id
            and not (
                item.get("kind") == "logic_family"
                and item.get("record_id") == logic_family
            )
        )
        truncated = len(without_duplicate) >= maximum_count
        retained = without_duplicate[: max(0, maximum_count - 1)]
        return retained + (family_reference,), truncated

    async def dispatch(
        self,
        request: AnalysisRequest | Mapping[str, Any],
        *,
        provider_id: str = "",
        cancellation_token: Any = None,
        progress_callback: Callable[[Any], Any] | None = None,
    ) -> AnalysisResult:
        normalized = AnalysisRequest.from_value(request)
        spec, selected = self._validate_dispatch_request(
            normalized,
            provider_id=provider_id,
        )
        family_raw = normalized.metadata.get("logic_family")
        result = await self._build_transport().dispatch(
            normalized,
            provider_id=selected,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        evidence = _normalize_reference_collection(
            result.evidence_references,
            default_kind="candidate",
            producer_id=result.provider_id,
        )
        provenance = _normalize_reference_collection(
            result.provenance_references,
            default_kind="provenance",
            producer_id=result.provider_id,
        )
        provenance_truncated = False
        if family_raw and result.successful:
            provenance, provenance_truncated = (
                self._attach_logic_family_provenance(
                    provenance,
                    logic_family=family_raw,
                    producer_id=result.provider_id,
                    maximum_count=spec.bounds.max_provenance_references,
                )
            )
        self._validate_result_references(
            evidence,
            maximum_count=spec.bounds.max_evidence_references,
            maximum_bytes=spec.bounds.max_reference_bytes,
            name="evidence",
            expected_tree_id=normalized.metadata["tree_id"],
        )
        self._validate_result_references(
            provenance,
            maximum_count=spec.bounds.max_provenance_references,
            maximum_bytes=spec.bounds.max_reference_bytes,
            name="provenance",
            expected_tree_id=normalized.metadata["tree_id"],
        )
        return replace(
            result,
            evidence_references=evidence,
            provenance_references=provenance,
            truncated=result.truncated or provenance_truncated,
        )

    async def dispatch_batch(
        self,
        requests: Sequence[AnalysisRequest | Mapping[str, Any]],
        *,
        provider_id: str = "",
        cancellation_token: Any = None,
        progress_callback: Callable[[Any], Any] | None = None,
    ) -> tuple[AnalysisResult, ...]:
        if isinstance(requests, (str, bytes, bytearray)) or not isinstance(
            requests, Sequence
        ):
            raise AnalysisOperationRegistryError("requests must be a sequence")
        normalized = tuple(AnalysisRequest.from_value(item) for item in requests)
        if not normalized:
            raise AnalysisOperationRegistryError("requests must not be empty")
        validated = tuple(
            self._validate_dispatch_request(item, provider_id=provider_id)
            for item in normalized
        )
        specs = tuple(item[0] for item in validated)
        first = specs[0]
        if any(item.operation is not first.operation for item in specs):
            raise AnalysisOperationRegistryError(
                "batch members must use the same registered operation"
            )
        if not first.batching.supported:
            raise AnalysisOperationRegistryError(
                f"{first.operation.value} does not support batching"
            )
        if len(normalized) > first.batching.max_batch_size:
            raise AnalysisOperationRegistryError(
                "batch exceeds operation max_batch_size"
            )
        tree_ids = {item.metadata.get("tree_id") for item in normalized}
        if len(tree_ids) != 1:
            raise AnalysisOperationRegistryError(
                "batch members must bind the same tree_id"
            )
        raw = await self._build_transport().dispatch_batch(
            normalized,
            provider_id=provider_id,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        # Reuse the same post-normalization path without redispatch.
        results: list[AnalysisResult] = []
        for request, result in zip(normalized, raw):
            family_raw = request.metadata.get("logic_family")
            evidence = _normalize_reference_collection(
                result.evidence_references,
                default_kind="candidate",
                producer_id=result.provider_id,
            )
            provenance = _normalize_reference_collection(
                result.provenance_references,
                default_kind="provenance",
                producer_id=result.provider_id,
            )
            provenance_truncated = False
            if family_raw and result.successful:
                provenance, provenance_truncated = (
                    self._attach_logic_family_provenance(
                        provenance,
                        logic_family=family_raw,
                        producer_id=result.provider_id,
                        maximum_count=first.bounds.max_provenance_references,
                    )
                )
            self._validate_result_references(
                evidence,
                maximum_count=first.bounds.max_evidence_references,
                maximum_bytes=first.bounds.max_reference_bytes,
                name="evidence",
                expected_tree_id=request.metadata["tree_id"],
            )
            self._validate_result_references(
                provenance,
                maximum_count=first.bounds.max_provenance_references,
                maximum_bytes=first.bounds.max_reference_bytes,
                name="provenance",
                expected_tree_id=request.metadata["tree_id"],
            )
            results.append(
                replace(
                    result,
                    evidence_references=evidence,
                    provenance_references=provenance,
                    truncated=result.truncated or provenance_truncated,
                )
            )
        return tuple(results)


_ALL_LOGIC_FAMILIES: Final = tuple(LogicFamily)


def _provenance(*, logical: bool) -> AnalysisProvenanceSemantics:
    required = list(AnalysisProvenanceSemantics().required)
    if logical:
        required.append(ProvenanceRequirement.LOGIC_FAMILY)
    return AnalysisProvenanceSemantics(required=tuple(required))


def default_operation_specs() -> tuple[AnalysisOperationSpec, ...]:
    """Return the canonical complete ASI-098 operation portfolio."""

    common = {
        "authority": AnalysisAuthoritySemantics(),
        "fallback": AnalysisFallbackSemantics(),
    }
    return (
        AnalysisOperationSpec(
            operation=AnalysisOperation.SYMBOL_IMPACT,
            capability_requirements=("ast_index_read", "symbol_impact"),
            provenance=_provenance(logical=False),
            **common,
        ),
        AnalysisOperationSpec(
            operation=AnalysisOperation.GRAPH_RAG_RETRIEVAL,
            capability_requirements=("graph_read", "graphrag_retrieval"),
            provenance=_provenance(logical=False),
            **common,
        ),
        AnalysisOperationSpec(
            operation=AnalysisOperation.PREMISE_SELECTION,
            capability_requirements=("logic_family_routing", "premise_selection"),
            logic_families=_ALL_LOGIC_FAMILIES,
            provenance=_provenance(logical=True),
            **common,
        ),
        AnalysisOperationSpec(
            operation=AnalysisOperation.CONTRADICTION_SEARCH,
            capability_requirements=("contradiction_search", "logic_family_routing"),
            logic_families=_ALL_LOGIC_FAMILIES,
            provenance=_provenance(logical=True),
            **common,
        ),
        AnalysisOperationSpec(
            operation=AnalysisOperation.LOGIC_TRANSLATION,
            capability_requirements=("logic_family_routing", "logic_translation"),
            logic_families=_ALL_LOGIC_FAMILIES,
            provenance=_provenance(logical=True),
            **common,
        ),
        AnalysisOperationSpec(
            operation=AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
            capability_requirements=("logic_family_routing", "proof_candidate_analysis"),
            logic_families=_ALL_LOGIC_FAMILIES,
            provenance=_provenance(logical=True),
            **common,
        ),
        AnalysisOperationSpec(
            operation=AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
            capability_requirements=(
                "counterexample_candidate_analysis",
                "logic_family_routing",
            ),
            logic_families=_ALL_LOGIC_FAMILIES,
            provenance=_provenance(logical=True),
            **common,
        ),
    )


def create_default_analysis_operation_registry(
    *,
    transport_bounds: AnalysisTransportBounds | Mapping[str, Any] | None = None,
    local_provider: Any = None,
    optional_provider_factory: Callable[[], Any] | None = None,
) -> AnalysisOperationRegistry:
    """Build the production registry without importing optional datasets code."""

    registry = AnalysisOperationRegistry(transport_bounds=transport_bounds)
    for spec in default_operation_specs():
        registry.register_operation(spec)

    # Imports are local to keep declaration/inspection side-effect-free and to
    # avoid coupling provider module import order to the registry.
    from ..integrations.ipfs_datasets_analysis_provider import (
        create_local_registry_analysis_producer,
        create_optional_registry_analysis_producer,
        registry_analysis_producer_declarations,
    )
    from ..integrations.ipfs_datasets_logic_provider import (
        create_local_registry_logic_producer,
        create_optional_registry_logic_producer,
        registry_logic_producer_declarations,
    )

    analysis_local, analysis_optional = registry_analysis_producer_declarations()
    logic_local, logic_optional = registry_logic_producer_declarations()
    local_declaration = _merge_producer_declarations(
        LOCAL_ANALYSIS_PRODUCER_ID, AnalysisProviderKind.LOCAL, analysis_local, logic_local
    )
    optional_declaration = _merge_producer_declarations(
        IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
        AnalysisProviderKind.IPFS_DATASETS,
        analysis_optional,
        logic_optional,
    )

    def local_factory() -> Any:
        if local_provider is not None:
            return local_provider
        return _OperationRouter(
            create_local_registry_analysis_producer(),
            create_local_registry_logic_producer(),
            local_declaration.capability,
        )

    def optional_factory() -> Any:
        if optional_provider_factory is not None:
            return optional_provider_factory()
        return _OperationRouter(
            create_optional_registry_analysis_producer(),
            create_optional_registry_logic_producer(),
            optional_declaration.capability,
        )

    registry.register_producer(local_declaration, factory=local_factory)
    registry.register_producer(optional_declaration, factory=optional_factory)
    return registry


def _merge_producer_declarations(
    producer_id: str,
    provider_kind: AnalysisProviderKind,
    *declarations: AnalysisProducer,
) -> AnalysisProducer:
    operations = tuple(
        sorted(
            {operation for item in declarations for operation in item.operations},
            key=lambda item: item.value,
        )
    )
    return AnalysisProducer(
        producer_id=producer_id,
        provider_kind=provider_kind,
        operations=operations,
        capability_revision=_content_id(
            "analysis-producer-capability",
            [item.to_dict() for item in declarations],
        ),
        provider_version="1.0.0",
        capabilities=tuple(
            sorted({capability for item in declarations for capability in item.capabilities})
        ),
        logic_families=tuple(
            sorted(
                {family for item in declarations for family in item.logic_families},
                key=lambda item: item.value,
            )
        ),
        max_batch_size=min(item.max_batch_size for item in declarations),
        max_concurrency=min(item.max_concurrency for item in declarations),
        supports_cancellation=all(item.supports_cancellation for item in declarations),
        supports_progress=all(item.supports_progress for item in declarations),
        supports_batching=all(item.supports_batching for item in declarations),
    )


class _OperationRouter:
    """Combine provider-family adapters behind one negotiated capability."""

    def __init__(self, *providers_and_capability: Any) -> None:
        *providers, capability = providers_and_capability
        self._providers = tuple(providers)
        self._capability = capability

    def capabilities(self) -> AnalysisCapability:
        return self._capability

    capability = capabilities

    def _provider(self, operation: str) -> Any:
        for provider in self._providers:
            supports = getattr(provider, "supports", None)
            if callable(supports) and supports(operation):
                return provider
        raise AnalysisOperationRegistryError(
            f"no routed producer supports {operation}"
        )

    def analyze(self, request: AnalysisRequest, **kwargs: Any) -> Any:
        return self._provider(request.operation).analyze(request, **kwargs)

    def analyze_batch(self, requests: Sequence[AnalysisRequest], **kwargs: Any) -> Any:
        if not requests:
            raise AnalysisOperationRegistryError("batch must not be empty")
        provider = self._provider(requests[0].operation)
        method = getattr(provider, "analyze_batch", None)
        if callable(method):
            return method(requests, **kwargs)
        return tuple(provider.analyze(item, **kwargs) for item in requests)


build_default_analysis_operation_registry = create_default_analysis_operation_registry
OperationRegistry = AnalysisOperationRegistry
OperationSpec = AnalysisOperationSpec
ProducerDeclaration = AnalysisProducer


# ---------------------------------------------------------------------------
# Property-to-strategy routing bridge (PDR-012)
# ---------------------------------------------------------------------------
#
# Strategy selection lives in :mod:`analysis_strategy_registry`.  These helpers
# keep the operation registry as the transport policy boundary while exposing
# closed property-class routing without importing optional providers or
# inferring support from importability.


def operation_property_classes(operation: Any) -> tuple[str, ...]:
    """Return property-class ids linked to a transport operation."""

    from .analysis_strategy_registry import property_class_for_operation

    return tuple(item.value for item in property_class_for_operation(operation))


def route_operation_strategies(
    operation: Any,
    *,
    required_assurance: Any | None = None,
    available_capabilities: Mapping[str, Any] | None = None,
    strategy_registry: Any | None = None,
) -> tuple[Any, ...]:
    """Select least-cost strategies for each property class of ``operation``.

    Returns a tuple of :class:`~analysis_strategy_registry.StrategySelection`
    objects.  Discovery remains cold/lazy: optional providers are not imported.
    """

    from .analysis_strategy_registry import (
        create_default_analysis_strategy_registry,
        property_class_for_operation,
    )

    registry = strategy_registry or create_default_analysis_strategy_registry()
    selections = []
    for property_class in property_class_for_operation(operation):
        selections.append(
            registry.select(
                property_class,
                required_assurance=required_assurance,
                available_capabilities=available_capabilities,
            )
        )
    return tuple(selections)


def attach_strategy_routing(
    operation_registry: AnalysisOperationRegistry,
    *,
    strategy_registry: Any | None = None,
) -> Any:
    """Bind a strategy registry onto an operation registry for joint queries.

    The returned object is the strategy registry.  The operation registry is
    left unchanged for transport dispatch; callers use the strategy registry
    for property-class routing and the operation registry for request dispatch.
    """

    from .analysis_strategy_registry import (
        AnalysisStrategyRegistry,
        create_default_analysis_strategy_registry,
    )

    if strategy_registry is None:
        strategy_registry = create_default_analysis_strategy_registry()
    if not isinstance(strategy_registry, AnalysisStrategyRegistry):
        raise AnalysisOperationRegistryError(
            "strategy_registry must be an AnalysisStrategyRegistry"
        )
    # Validate that every strategy-linked operation is registered when the
    # operation registry already has declarations.
    known = {item.operation for item in operation_registry.operations()}
    if known:
        for spec in strategy_registry.strategies():
            for operation_id in spec.analysis_operations:
                try:
                    op = normalize_analysis_operation(operation_id)
                except AnalysisOperationRegistryError:
                    continue
                if op not in known:
                    raise AnalysisOperationRegistryError(
                        f"strategy {spec.property_class.value} references "
                        f"unregistered operation {operation_id}"
                    )
    return strategy_registry


# Preserve attribute access for callers that monkey-patch routing on the class.
AnalysisOperationRegistry.operation_property_classes = staticmethod(  # type: ignore[attr-defined]
    operation_property_classes
)
AnalysisOperationRegistry.route_operation_strategies = staticmethod(  # type: ignore[attr-defined]
    route_operation_strategies
)
AnalysisOperationRegistry.attach_strategy_routing = staticmethod(  # type: ignore[attr-defined]
    attach_strategy_routing
)


__all__ = [
    "ANALYSIS_OPERATION_REGISTRY_VERSION",
    "ANALYSIS_OPERATION_SCHEMA",
    "ANALYSIS_PRODUCER_SCHEMA",
    "ANALYSIS_REFERENCE_SCHEMA",
    "IPFS_DATASETS_ANALYSIS_PRODUCER_ID",
    "LOCAL_ANALYSIS_PRODUCER_ID",
    "AnalysisAuthoritySemantics",
    "AnalysisBatchingSemantics",
    "AnalysisCacheSemantics",
    "AnalysisFallbackSemantics",
    "AnalysisOperation",
    "AnalysisOperationBounds",
    "AnalysisOperationRegistry",
    "AnalysisOperationRegistryError",
    "AnalysisOperationSpec",
    "AnalysisProducer",
    "AnalysisProvenanceSemantics",
    "CacheScope",
    "LogicFamily",
    "OperationRegistry",
    "OperationSpec",
    "ProducerDeclaration",
    "ProvenanceRequirement",
    "attach_strategy_routing",
    "build_default_analysis_operation_registry",
    "create_default_analysis_operation_registry",
    "default_operation_specs",
    "normalize_analysis_operation",
    "normalize_analysis_reference",
    "normalize_logic_family",
    "normalized_reference_payload",
    "operation_property_classes",
    "route_operation_strategies",
    "to_canonical_logic_family_id",
]
