"""Versioned, independently auditable semantic-entropy metrics (PCAR-004).

`SemanticEntropyReport` retains every required dimension as its own evidence-bound
record. Change amplification is reported alongside those dimensions. A ranking
may be derived for prioritization, but no aesthetic aggregate, score, or lower
entropy vector can establish safety, equivalence, ownership, deletion, or
promotion.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .architecture_ir import ArchitectureEdge, ArchitectureIR, ArchitectureNode
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
)

SEMANTIC_ENTROPY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-entropy-report@1"
)
SEMANTIC_ENTROPY_VERSION = 1
SEMANTIC_ENTROPY_EVIDENCE = "pcar/semantic-entropy-report@1"
ENTROPY_REPORT_EVIDENCE = "pcar/entropy-report@1"
ENTROPY_DIMENSION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-entropy-dimension@1"
)
ENTROPY_DIMENSION_VERSION = 1
CHANGE_AMPLIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-amplification@1"
)
CHANGE_AMPLIFICATION_VERSION = 1
FROZEN_TASK_CORPUS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/frozen-task-corpus@1"
)
FROZEN_TASK_CORPUS_VERSION = 1
ENTROPY_EXTRACTOR_IDENTITY = "pcar-004-semantic-entropy"
UNBOUND_FROZEN_TASK_CORPUS_ID = "pcar-004-unbound-corpus"
RANKING_IS_NON_PROBATIVE = True
ENTROPY_IS_PRIORITIZATION_ONLY = True

_UNKNOWN_FIELD_MESSAGE = "unknown semantic-entropy field"
_MISSING_FIELD_MESSAGE = "missing semantic-entropy field"
_AGGREGATE_FIELD_MESSAGE = "undocumented aggregation is not an entropy proof"
_BASIS_POINTS = 10_000
_DOC_SUFFIXES = (".md", ".markdown", ".rst", ".txt", ".adoc")
_CONFIDENCE_RANK = {
    Confidence.EXACT: 0,
    Confidence.CONSERVATIVE: 1,
    Confidence.HEURISTIC: 2,
    Confidence.OPAQUE: 3,
}
_BANNED_AGGREGATE_FIELDS = frozenset(
    {
        "aesthetic",
        "aggregate",
        "aggregate_score",
        "composite",
        "composite_score",
        "grade",
        "ranking",
        "safety",
        "safety_score",
        "score",
    }
)
_IMPLEMENTATION_NODE_KINDS = frozenset(
    {NodeKind.SYMBOL, NodeKind.MODULE, NodeKind.FILE, NodeKind.OPERATION}
)
_SCHEMA_RECEIPT_KINDS = frozenset({NodeKind.SCHEMA, NodeKind.RECEIPT})
_COMPATIBILITY_NODE_KINDS = frozenset(
    {
        NodeKind.INTERFACE,
        NodeKind.ENTRYPOINT,
        NodeKind.COMPATIBILITY,
        NodeKind.PROVIDER,
    }
)
_DISPATCH_EDGE_KINDS = frozenset(
    {EdgeKind.CALLS, EdgeKind.EXECUTES, EdgeKind.FALLBACKS_TO}
)
_EFFECT_EDGE_KINDS = frozenset(
    {EdgeKind.READS, EdgeKind.WRITES, EdgeKind.MUTATES, EdgeKind.OBSERVES}
)
_OWNER_EDGE_KINDS = frozenset({EdgeKind.AUTHORIZES})
_CONFLICT_EDGE_KINDS = frozenset(
    {EdgeKind.WRITES, EdgeKind.MUTATES, EdgeKind.PERSISTS}
)
_DRIFT_EDGE_KINDS = frozenset(
    {EdgeKind.SUPERSEDES, EdgeKind.DEPRECATES, EdgeKind.INVALIDATES}
)
_CONE_EDGE_KINDS = frozenset(
    {
        EdgeKind.CONTAINS,
        EdgeKind.IMPORTS,
        EdgeKind.CALLS,
        EdgeKind.CONSTRUCTS,
        EdgeKind.IMPLEMENTS,
        EdgeKind.EXECUTES,
        EdgeKind.READS,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.AUTHORIZES,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.PERSISTS,
        EdgeKind.GENERATES,
        EdgeKind.TESTS,
        EdgeKind.PROVES,
        EdgeKind.ADAPTS,
        EdgeKind.REEXPORTS,
        EdgeKind.FALLBACKS_TO,
    }
)
_REVERSE_CONE_EDGE_KINDS = frozenset(
    {
        EdgeKind.CONTAINS,
        EdgeKind.TESTS,
        EdgeKind.PROVES,
        EdgeKind.AUTHORIZES,
        EdgeKind.CONFIRMS,
        EdgeKind.GENERATES,
        EdgeKind.IMPLEMENTS,
        EdgeKind.ADAPTS,
    }
)
_CONTEXT_NODE_KINDS = frozenset(
    {
        NodeKind.FILE,
        NodeKind.SYMBOL,
        NodeKind.INTERFACE,
        NodeKind.SCHEMA,
        NodeKind.EFFECT,
        NodeKind.TEST,
        NodeKind.PROOF,
        NodeKind.PROVIDER,
        NodeKind.AUTHORITY,
        NodeKind.POLICY,
        NodeKind.STATE,
        NodeKind.ARTIFACT,
        NodeKind.ENTRYPOINT,
        NodeKind.OPERATION,
    }
)
_AMPLIFICATION_COUNT_FIELDS = (
    "files",
    "symbols",
    "interfaces",
    "schemas",
    "effects",
    "tests",
    "proofs",
    "providers",
    "runtime_paths",
    "owners",
)
_DOCUMENTATION_NODE_KINDS = frozenset(
    {NodeKind.FILE, NodeKind.ARTIFACT, NodeKind.GENERATED, NodeKind.MODULE}
)

NON_COMPENSABLE_INVARIANTS = (
    "NoAuthorityWeakening",
    "NoEffectExpansion",
    "NoHiddenBehaviorChange",
    "NoSimulatedAsLive",
    "NoValidationReduction",
    "NoProofObligationLoss",
    "NoPublicContractBreakWithoutVersionedMigration",
    "NoStaleEvidencePromotion",
    "NoUnboundedRefactor",
    "NoProcedureSelfAuthorization",
    "NoArchitectureCandidateSelfPromotion",
    "NoCrossRepositoryWrite",
    "NoSecretOrPrivateDataLeak",
    "NoFalseCompletion",
)
ENTROPY_NON_AUTHORITY_CLAIMS = (
    "safety",
    "equivalence",
    "ownership",
    "dead_code",
    "promotion",
    "deletion",
    "correctness",
    "rollback",
)


class EntropyContractError(ArchitectureContractError):
    """Fail-closed contract violation for semantic-entropy metrics."""


class EntropyAuthorityError(EntropyContractError):
    """Raised when entropy is treated as safety or promotion authority."""


class EntropyDimensionKind(str, Enum):
    """Closed semantic-entropy dimension vocabulary (PCAR-PLAN-R1)."""

    AUTHORITY_MULTIPLICITY = "AuthorityMultiplicity"
    IMPLEMENTATION_DUPLICATION = "ImplementationDuplication"
    PUBLIC_SURFACE_AREA = "PublicSurfaceArea"
    DEPENDENCY_CONE_SIZE = "DependencyConeSize"
    DYNAMIC_DISPATCH_UNCERTAINTY = "DynamicDispatchUncertainty"
    STATE_OWNERSHIP_AMBIGUITY = "StateOwnershipAmbiguity"
    EFFECT_OPACITY = "EffectOpacity"
    COMPATIBILITY_BURDEN = "CompatibilityBurden"
    VALIDATION_AMPLIFICATION = "ValidationAmplification"
    CACHE_FRAGMENTATION = "CacheFragmentation"
    SCHEMA_DRIFT = "SchemaDrift"
    RECEIPT_DRIFT = "ReceiptDrift"
    DOCUMENTATION_DRIFT = "DocumentationDrift"
    MERGE_CONFLICT_DENSITY = "MergeConflictDensity"
    CONTEXT_BURDEN = "ContextBurden"


REQUIRED_ENTROPY_DIMENSIONS: tuple[EntropyDimensionKind, ...] = tuple(
    EntropyDimensionKind
)
CLOSED_ENTROPY_DIMENSIONS: frozenset[str] = frozenset(
    kind.value for kind in EntropyDimensionKind
)
CLOSED_SAFETY_PREDICATES: frozenset[str] = frozenset(NON_COMPENSABLE_INVARIANTS)
CLOSED_NON_AUTHORITY_CLAIMS: frozenset[str] = frozenset(ENTROPY_NON_AUTHORITY_CLAIMS)

_DIMENSION_UNITS: dict[EntropyDimensionKind, str] = {
    EntropyDimensionKind.AUTHORITY_MULTIPLICITY: "authorities",
    EntropyDimensionKind.IMPLEMENTATION_DUPLICATION: "implementations",
    EntropyDimensionKind.PUBLIC_SURFACE_AREA: "surfaces",
    EntropyDimensionKind.DEPENDENCY_CONE_SIZE: "nodes",
    EntropyDimensionKind.DYNAMIC_DISPATCH_UNCERTAINTY: "dispatches",
    EntropyDimensionKind.STATE_OWNERSHIP_AMBIGUITY: "states",
    EntropyDimensionKind.EFFECT_OPACITY: "effects",
    EntropyDimensionKind.COMPATIBILITY_BURDEN: "adapters",
    EntropyDimensionKind.VALIDATION_AMPLIFICATION: "validations",
    EntropyDimensionKind.CACHE_FRAGMENTATION: "caches",
    EntropyDimensionKind.SCHEMA_DRIFT: "schemas",
    EntropyDimensionKind.RECEIPT_DRIFT: "receipts",
    EntropyDimensionKind.DOCUMENTATION_DRIFT: "documents",
    EntropyDimensionKind.MERGE_CONFLICT_DENSITY: "conflicts",
    EntropyDimensionKind.CONTEXT_BURDEN: "nodes",
}
CLOSED_ENTROPY_UNITS: frozenset[str] = frozenset(_DIMENSION_UNITS.values()) | {
    "amplified_units"
}

_UNCERTAINTY_FIELDS = frozenset(
    {"confidence", "unknown_numerator", "unknown_denominator"}
)
_EVIDENCE_FIELDS = frozenset({"edge_ids", "extractor_identity", "node_ids"})
_DIMENSION_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "content_identity",
        "denominator",
        "evidence",
        "evidence_identity",
        "freshness",
        "kind",
        "numerator",
        "repository_tree",
        "schema",
        "uncertainty",
        "unit",
        "version",
    }
)
_CORPUS_FIELDS = frozenset(
    {"content_identity", "corpus_id", "schema", "task_ids", "version"}
)
_AMPLIFICATION_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "content_identity",
        "denominator",
        "effects",
        "evidence_identity",
        "files",
        "freshness",
        "frozen_task_corpus_identity",
        "hops",
        "interfaces",
        "numerator",
        "owners",
        "proofs",
        "providers",
        "raw_expansions",
        "repository_tree",
        "runtime_paths",
        "schema",
        "schemas",
        "symbols",
        "tests",
        "tokens",
        "uncertainty",
        "unit",
        "version",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "change_amplification",
        "content_identity",
        "dimensions",
        "freshness",
        "frozen_task_corpus",
        "repository_tree",
        "schema",
        "version",
    }
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise EntropyContractError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if not extra:
        return
    banned = sorted(set(extra) & _BANNED_AGGREGATE_FIELDS)
    if banned:
        raise EntropyContractError(f"{_AGGREGATE_FIELD_MESSAGE}: {banned}")
    raise EntropyContractError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_entropy_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise EntropyContractError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_non_negative_int(value: Any, name: str) -> int:
    number = _require_int(value, name, error_type=EntropyContractError)
    if number < 0:
        raise EntropyContractError(f"{name} must be a non-negative integer")
    return number


def _require_optional_denominator(value: Any, name: str) -> int | None:
    if value is None:
        return None
    number = _require_non_negative_int(value, name)
    if number == 0:
        raise EntropyContractError(f"{name} must be a positive integer or null")
    return number


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EntropyContractError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=EntropyContractError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _require_architecture_ir(graph: ArchitectureIR | Mapping[str, Any]) -> ArchitectureIR:
    if isinstance(graph, ArchitectureIR):
        return graph
    try:
        return ArchitectureIR.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise EntropyContractError(str(exc)) from exc


def _worst_confidence(
    confidences: Iterable[Confidence],
    *,
    empty: Confidence = Confidence.EXACT,
) -> Confidence:
    worst = empty
    for item in confidences:
        if _CONFIDENCE_RANK[item] > _CONFIDENCE_RANK[worst]:
            worst = item
    return worst


def _is_documentation_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith(_DOC_SUFFIXES) or "/docs/" in f"/{normalized}/"


@dataclass(frozen=True)
class DimensionUncertainty:
    """Explicit uncertainty bound for one independent entropy dimension."""

    confidence: Confidence
    unknown_numerator: int
    unknown_denominator: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence",
            _closed_enum(
                self.confidence,
                Confidence,
                "uncertainty confidence",
                error_type=EntropyContractError,
            ),
        )
        object.__setattr__(
            self,
            "unknown_numerator",
            _require_non_negative_int(self.unknown_numerator, "unknown_numerator"),
        )
        object.__setattr__(
            self,
            "unknown_denominator",
            _require_non_negative_int(self.unknown_denominator, "unknown_denominator"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence.value,
            "unknown_denominator": self.unknown_denominator,
            "unknown_numerator": self.unknown_numerator,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DimensionUncertainty":
        mapping = _require_mapping(payload, error_type=EntropyContractError)
        _require_entropy_fields(mapping, _UNCERTAINTY_FIELDS)
        return cls(
            confidence=mapping["confidence"],
            unknown_numerator=mapping["unknown_numerator"],
            unknown_denominator=mapping["unknown_denominator"],
        )

    from_dict = from_mapping


@dataclass(frozen=True)
class DimensionEvidence:
    """Source identities that independently justify one dimension."""

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    extractor_identity: str = ENTROPY_EXTRACTOR_IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", _require_text_tuple(self.node_ids, "node_ids"))
        object.__setattr__(self, "edge_ids", _require_text_tuple(self.edge_ids, "edge_ids"))
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(
                self.extractor_identity,
                "extractor_identity",
                error_type=EntropyContractError,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_ids": list(self.edge_ids),
            "extractor_identity": self.extractor_identity,
            "node_ids": list(self.node_ids),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DimensionEvidence":
        mapping = _require_mapping(payload, error_type=EntropyContractError)
        _require_entropy_fields(mapping, _EVIDENCE_FIELDS)
        return cls(
            node_ids=mapping["node_ids"],
            edge_ids=mapping["edge_ids"],
            extractor_identity=mapping["extractor_identity"],
        )

    from_dict = from_mapping


def _evidence_identity(evidence: DimensionEvidence) -> str:
    return _content_identity(evidence.to_dict())


@dataclass(frozen=True)
class FrozenTaskCorpus:
    """Frozen task corpus identity retained by every entropy report."""

    corpus_id: str
    task_ids: tuple[str, ...] = ()
    schema: str = FROZEN_TASK_CORPUS_SCHEMA
    version: int = FROZEN_TASK_CORPUS_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=EntropyContractError)
        if schema != FROZEN_TASK_CORPUS_SCHEMA:
            raise EntropyContractError("unexpected frozen-task-corpus schema")
        version = _require_int(self.version, "version", error_type=EntropyContractError)
        if version != FROZEN_TASK_CORPUS_VERSION:
            raise EntropyContractError("unexpected frozen-task-corpus version")
        corpus_id = _require_text(
            self.corpus_id, "corpus_id", error_type=EntropyContractError
        )
        task_ids = _require_text_tuple(self.task_ids, "task_ids")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "corpus_id", corpus_id)
        object.__setattr__(self, "task_ids", task_ids)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=EntropyContractError,
                )
            )
            if claimed != identity:
                raise EntropyContractError("frozen-task-corpus content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "schema": self.schema,
            "task_ids": list(self.task_ids),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise EntropyContractError("frozen-task-corpus content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def unbound(cls) -> "FrozenTaskCorpus":
        return cls(corpus_id=UNBOUND_FROZEN_TASK_CORPUS_ID, task_ids=())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenTaskCorpus":
        mapping = _require_mapping(payload, error_type=EntropyContractError)
        _require_entropy_fields(mapping, _CORPUS_FIELDS)
        corpus = cls(
            corpus_id=mapping["corpus_id"],
            task_ids=mapping["task_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != corpus.content_identity:
            raise EntropyContractError("frozen-task-corpus content identity mismatch")
        return corpus

    from_dict = from_mapping


@dataclass(frozen=True)
class EntropyDimensionRecord:
    """One independently auditable semantic-entropy dimension."""

    kind: EntropyDimensionKind
    numerator: int
    denominator: int | None
    unit: str
    uncertainty: DimensionUncertainty
    evidence: DimensionEvidence
    architecture_ir_identity: str
    repository_tree: str
    freshness: str
    schema: str = ENTROPY_DIMENSION_SCHEMA
    version: int = ENTROPY_DIMENSION_VERSION
    evidence_identity: str = ""
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=EntropyContractError)
        if schema != ENTROPY_DIMENSION_SCHEMA:
            raise EntropyContractError("unexpected semantic-entropy dimension schema")
        version = _require_int(self.version, "version", error_type=EntropyContractError)
        if version != ENTROPY_DIMENSION_VERSION:
            raise EntropyContractError("unexpected semantic-entropy dimension version")
        kind = _closed_enum(
            self.kind,
            EntropyDimensionKind,
            "entropy dimension kind",
            error_type=EntropyContractError,
        )
        expected_unit = _DIMENSION_UNITS[kind]
        unit = _require_text(self.unit, "unit", error_type=EntropyContractError)
        if unit != expected_unit:
            raise EntropyContractError(
                f"unexpected unit for {kind.value}: {unit!r}"
            )
        numerator = _require_non_negative_int(self.numerator, "numerator")
        denominator = _require_optional_denominator(self.denominator, "denominator")
        uncertainty = (
            self.uncertainty
            if isinstance(self.uncertainty, DimensionUncertainty)
            else DimensionUncertainty.from_mapping(self.uncertainty)
        )
        evidence = (
            self.evidence
            if isinstance(self.evidence, DimensionEvidence)
            else DimensionEvidence.from_mapping(self.evidence)
        )
        architecture_ir_identity = _validate_dag_json_cid(
            _require_text(
                self.architecture_ir_identity,
                "architecture_ir_identity",
                error_type=EntropyContractError,
            )
        )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=EntropyContractError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=EntropyContractError
        )
        computed_evidence = _evidence_identity(evidence)
        if self.evidence_identity:
            claimed_evidence = _validate_dag_json_cid(
                _require_text(
                    self.evidence_identity,
                    "evidence_identity",
                    error_type=EntropyContractError,
                )
            )
            if claimed_evidence != computed_evidence:
                raise EntropyContractError("dimension evidence identity mismatch")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "evidence_identity", computed_evidence)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=EntropyContractError,
                )
            )
            if claimed != identity:
                raise EntropyContractError("dimension content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "denominator": self.denominator,
            "evidence": self.evidence.to_dict(),
            "evidence_identity": self.evidence_identity,
            "freshness": self.freshness,
            "kind": self.kind.value,
            "numerator": self.numerator,
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "uncertainty": self.uncertainty.to_dict(),
            "unit": self.unit,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise EntropyContractError("dimension content identity mismatch")
        return {**payload, "content_identity": identity}

    def ratio_basis_points(self) -> int | None:
        if self.denominator is None:
            return None
        return (self.numerator * _BASIS_POINTS) // self.denominator

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EntropyDimensionRecord":
        mapping = _require_mapping(payload, error_type=EntropyContractError)
        _require_entropy_fields(mapping, _DIMENSION_FIELDS)
        record = cls(
            kind=mapping["kind"],
            numerator=mapping["numerator"],
            denominator=mapping["denominator"],
            unit=mapping["unit"],
            uncertainty=mapping["uncertainty"],
            evidence=mapping["evidence"],
            architecture_ir_identity=mapping["architecture_ir_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            schema=mapping["schema"],
            version=mapping["version"],
            evidence_identity=mapping["evidence_identity"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise EntropyContractError("dimension content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ChangeAmplificationMeasure:
    """Independent change-amplification measure retained beside entropy dimensions."""

    numerator: int
    denominator: int | None
    unit: str
    uncertainty: DimensionUncertainty
    files: int
    symbols: int
    interfaces: int
    schemas: int
    effects: int
    tests: int
    proofs: int
    providers: int
    runtime_paths: int
    tokens: int
    raw_expansions: int
    hops: int
    owners: int
    architecture_ir_identity: str
    repository_tree: str
    freshness: str
    frozen_task_corpus_identity: str
    evidence_identity: str
    schema: str = CHANGE_AMPLIFICATION_SCHEMA
    version: int = CHANGE_AMPLIFICATION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=EntropyContractError)
        if schema != CHANGE_AMPLIFICATION_SCHEMA:
            raise EntropyContractError("unexpected change-amplification schema")
        version = _require_int(self.version, "version", error_type=EntropyContractError)
        if version != CHANGE_AMPLIFICATION_VERSION:
            raise EntropyContractError("unexpected change-amplification version")
        unit = _require_text(self.unit, "unit", error_type=EntropyContractError)
        if unit != "amplified_units":
            raise EntropyContractError("unexpected change-amplification unit")
        counts = {
            "numerator": self.numerator,
            "files": self.files,
            "symbols": self.symbols,
            "interfaces": self.interfaces,
            "schemas": self.schemas,
            "effects": self.effects,
            "tests": self.tests,
            "proofs": self.proofs,
            "providers": self.providers,
            "runtime_paths": self.runtime_paths,
            "tokens": self.tokens,
            "raw_expansions": self.raw_expansions,
            "hops": self.hops,
            "owners": self.owners,
        }
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "unit", unit)
        for name, value in counts.items():
            object.__setattr__(self, name, _require_non_negative_int(value, name))
        object.__setattr__(
            self,
            "denominator",
            _require_optional_denominator(self.denominator, "denominator"),
        )
        uncertainty = (
            self.uncertainty
            if isinstance(self.uncertainty, DimensionUncertainty)
            else DimensionUncertainty.from_mapping(self.uncertainty)
        )
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(
            self,
            "architecture_ir_identity",
            _validate_dag_json_cid(
                _require_text(
                    self.architecture_ir_identity,
                    "architecture_ir_identity",
                    error_type=EntropyContractError,
                )
            ),
        )
        object.__setattr__(
            self,
            "repository_tree",
            _require_text(
                self.repository_tree, "repository_tree", error_type=EntropyContractError
            ),
        )
        object.__setattr__(
            self,
            "freshness",
            _require_text(self.freshness, "freshness", error_type=EntropyContractError),
        )
        object.__setattr__(
            self,
            "frozen_task_corpus_identity",
            _validate_dag_json_cid(
                _require_text(
                    self.frozen_task_corpus_identity,
                    "frozen_task_corpus_identity",
                    error_type=EntropyContractError,
                )
            ),
        )
        object.__setattr__(
            self,
            "evidence_identity",
            _validate_dag_json_cid(
                _require_text(
                    self.evidence_identity,
                    "evidence_identity",
                    error_type=EntropyContractError,
                )
            ),
        )
        known_sum = sum(getattr(self, field) for field in _AMPLIFICATION_COUNT_FIELDS)
        known_sum += self.hops
        if self.numerator != known_sum:
            raise EntropyContractError(
                "change-amplification numerator must equal the documented component sum"
            )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=EntropyContractError,
                )
            )
            if claimed != identity:
                raise EntropyContractError(
                    "change-amplification content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "denominator": self.denominator,
            "effects": self.effects,
            "evidence_identity": self.evidence_identity,
            "files": self.files,
            "freshness": self.freshness,
            "frozen_task_corpus_identity": self.frozen_task_corpus_identity,
            "hops": self.hops,
            "interfaces": self.interfaces,
            "numerator": self.numerator,
            "owners": self.owners,
            "proofs": self.proofs,
            "providers": self.providers,
            "raw_expansions": self.raw_expansions,
            "repository_tree": self.repository_tree,
            "runtime_paths": self.runtime_paths,
            "schema": self.schema,
            "schemas": self.schemas,
            "symbols": self.symbols,
            "tests": self.tests,
            "tokens": self.tokens,
            "uncertainty": self.uncertainty.to_dict(),
            "unit": self.unit,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise EntropyContractError("change-amplification content identity mismatch")
        return {**payload, "content_identity": identity}

    def ratio_basis_points(self) -> int | None:
        if self.denominator is None:
            return None
        return (self.numerator * _BASIS_POINTS) // self.denominator

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChangeAmplificationMeasure":
        mapping = _require_mapping(payload, error_type=EntropyContractError)
        _require_entropy_fields(mapping, _AMPLIFICATION_FIELDS)
        measure = cls(
            numerator=mapping["numerator"],
            denominator=mapping["denominator"],
            unit=mapping["unit"],
            uncertainty=mapping["uncertainty"],
            files=mapping["files"],
            symbols=mapping["symbols"],
            interfaces=mapping["interfaces"],
            schemas=mapping["schemas"],
            effects=mapping["effects"],
            tests=mapping["tests"],
            proofs=mapping["proofs"],
            providers=mapping["providers"],
            runtime_paths=mapping["runtime_paths"],
            tokens=mapping["tokens"],
            raw_expansions=mapping["raw_expansions"],
            hops=mapping["hops"],
            owners=mapping["owners"],
            architecture_ir_identity=mapping["architecture_ir_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            frozen_task_corpus_identity=mapping["frozen_task_corpus_identity"],
            evidence_identity=mapping["evidence_identity"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != measure.content_identity:
            raise EntropyContractError("change-amplification content identity mismatch")
        return measure

    from_dict = from_mapping


@dataclass(frozen=True)
class SemanticEntropyReport:
    """Closed semantic-entropy report with independently auditable dimensions."""

    architecture_ir_identity: str
    repository_tree: str
    freshness: str
    frozen_task_corpus: FrozenTaskCorpus
    dimensions: tuple[EntropyDimensionRecord, ...]
    change_amplification: ChangeAmplificationMeasure
    schema: str = SEMANTIC_ENTROPY_SCHEMA
    version: int = SEMANTIC_ENTROPY_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=EntropyContractError)
        if schema != SEMANTIC_ENTROPY_SCHEMA:
            raise EntropyContractError("unexpected semantic-entropy schema")
        version = _require_int(self.version, "version", error_type=EntropyContractError)
        if version != SEMANTIC_ENTROPY_VERSION:
            raise EntropyContractError("unexpected semantic-entropy version")
        architecture_ir_identity = _validate_dag_json_cid(
            _require_text(
                self.architecture_ir_identity,
                "architecture_ir_identity",
                error_type=EntropyContractError,
            )
        )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=EntropyContractError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=EntropyContractError
        )
        corpus = (
            self.frozen_task_corpus
            if isinstance(self.frozen_task_corpus, FrozenTaskCorpus)
            else FrozenTaskCorpus.from_mapping(self.frozen_task_corpus)
        )
        if isinstance(self.dimensions, (str, bytes, bytearray)) or not isinstance(
            self.dimensions, Sequence
        ):
            raise EntropyContractError("dimensions must be a sequence")
        records = tuple(
            item
            if isinstance(item, EntropyDimensionRecord)
            else EntropyDimensionRecord.from_mapping(item)
            for item in self.dimensions
        )
        kinds = tuple(item.kind for item in records)
        if len(kinds) != len(set(kinds)):
            raise EntropyContractError("entropy dimensions must be unique")
        missing = [
            kind.value for kind in REQUIRED_ENTROPY_DIMENSIONS if kind not in set(kinds)
        ]
        if missing:
            raise EntropyContractError(f"missing entropy dimensions: {missing}")
        unexpected = [kind.value for kind in kinds if kind not in REQUIRED_ENTROPY_DIMENSIONS]
        if unexpected:
            raise EntropyContractError(f"unsupported entropy dimensions: {unexpected}")
        ordered = tuple(
            next(item for item in records if item.kind is kind)
            for kind in REQUIRED_ENTROPY_DIMENSIONS
        )
        amplification = (
            self.change_amplification
            if isinstance(self.change_amplification, ChangeAmplificationMeasure)
            else ChangeAmplificationMeasure.from_mapping(self.change_amplification)
        )
        for item in (*ordered, amplification):
            if item.architecture_ir_identity != architecture_ir_identity:
                raise EntropyContractError(
                    "entropy measure architecture_ir_identity must match the report"
                )
            if item.repository_tree != repository_tree:
                raise EntropyContractError(
                    "entropy measure repository_tree must match the report"
                )
            if item.freshness != freshness:
                raise EntropyContractError(
                    "entropy measure freshness must match the report"
                )
        if amplification.frozen_task_corpus_identity != corpus.content_identity:
            raise EntropyContractError(
                "change amplification must bind the report frozen-task-corpus identity"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "frozen_task_corpus", corpus)
        object.__setattr__(self, "dimensions", ordered)
        object.__setattr__(self, "change_amplification", amplification)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=EntropyContractError,
                )
            )
            if claimed != identity:
                raise EntropyContractError("semantic-entropy content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "change_amplification": self.change_amplification.to_dict(),
            "dimensions": [item.to_dict() for item in self.dimensions],
            "freshness": self.freshness,
            "frozen_task_corpus": self.frozen_task_corpus.to_dict(),
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise EntropyContractError("semantic-entropy content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    def dimension(self, kind: EntropyDimensionKind | str) -> EntropyDimensionRecord:
        resolved = _closed_enum(
            kind,
            EntropyDimensionKind,
            "entropy dimension kind",
            error_type=EntropyContractError,
        )
        return next(item for item in self.dimensions if item.kind is resolved)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SemanticEntropyReport":
        mapping = _require_mapping(payload, error_type=EntropyContractError)
        _require_entropy_fields(mapping, _REPORT_FIELDS)
        dimensions_payload = mapping["dimensions"]
        if not isinstance(dimensions_payload, list):
            raise EntropyContractError("dimensions must be a list")
        report = cls(
            architecture_ir_identity=mapping["architecture_ir_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            frozen_task_corpus=mapping["frozen_task_corpus"],
            dimensions=tuple(dimensions_payload),
            change_amplification=mapping["change_amplification"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != report.content_identity:
            raise EntropyContractError("semantic-entropy content identity mismatch")
        return report

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "SemanticEntropyReport":
        if type(payload) is not str or not payload:
            raise EntropyContractError("semantic-entropy JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EntropyContractError("semantic-entropy JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise EntropyContractError("semantic-entropy JSON must contain an object")
        return cls.from_mapping(decoded)


@dataclass(frozen=True)
class _RawMeasure:
    numerator: int
    denominator: int | None
    confidence: Confidence
    unknown_numerator: int
    unknown_denominator: int
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class _GraphView:
    graph: ArchitectureIR
    nodes_by_id: dict[str, ArchitectureNode]
    nodes_by_kind: dict[NodeKind, tuple[ArchitectureNode, ...]]
    edges_by_kind: dict[EdgeKind, tuple[ArchitectureEdge, ...]]
    roots: tuple[ArchitectureNode, ...]
    cones: dict[str, dict[str, int]]
    cone_edge_ids: tuple[str, ...]
    cone_confidence: Confidence
    max_hops: int
    corpus_bound: bool


def _nodes_of(view: _GraphView, *kinds: NodeKind) -> tuple[ArchitectureNode, ...]:
    items: list[ArchitectureNode] = []
    for kind in kinds:
        items.extend(view.nodes_by_kind.get(kind, ()))
    return tuple(items)


def _edges_of(view: _GraphView, *kinds: EdgeKind) -> tuple[ArchitectureEdge, ...]:
    items: list[ArchitectureEdge] = []
    for kind in kinds:
        items.extend(view.edges_by_kind.get(kind, ()))
    return tuple(items)


def _empty_measure(*, unknown_denominator: int = 1) -> _RawMeasure:
    return _RawMeasure(
        numerator=0,
        denominator=None,
        confidence=Confidence.CONSERVATIVE,
        unknown_numerator=0,
        unknown_denominator=unknown_denominator,
        node_ids=(),
        edge_ids=(),
    )


def _measure(
    *,
    numerator: int,
    denominator: int | None,
    facts: Sequence[ArchitectureNode | ArchitectureEdge],
    extra_confidence: Confidence | None = None,
    unknown_numerator: int = 0,
    unknown_denominator: int = 0,
) -> _RawMeasure:
    confidences = [fact.provenance.confidence for fact in facts]
    if extra_confidence is not None:
        confidences.append(extra_confidence)
    empty = Confidence.CONSERVATIVE if denominator is None else Confidence.EXACT
    confidence = _worst_confidence(confidences, empty=empty)
    if unknown_numerator or unknown_denominator:
        confidence = _worst_confidence(
            [confidence, Confidence.CONSERVATIVE], empty=confidence
        )
    node_ids = tuple(
        fact.node_id for fact in facts if isinstance(fact, ArchitectureNode)
    )
    edge_ids = tuple(
        fact.edge_id for fact in facts if isinstance(fact, ArchitectureEdge)
    )
    return _RawMeasure(
        numerator=numerator,
        denominator=denominator,
        confidence=confidence,
        unknown_numerator=unknown_numerator,
        unknown_denominator=unknown_denominator,
        node_ids=node_ids,
        edge_ids=edge_ids,
    )


def _population(count: int) -> int | None:
    return count if count > 0 else None


def _build_view(
    graph: ArchitectureIR, *, corpus_bound: bool
) -> _GraphView:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    nodes_by_kind: dict[NodeKind, list[ArchitectureNode]] = defaultdict(list)
    for node in graph.nodes:
        nodes_by_kind[node.kind].append(node)
    edges_by_kind: dict[EdgeKind, list[ArchitectureEdge]] = defaultdict(list)
    outgoing: dict[str, list[ArchitectureEdge]] = defaultdict(list)
    incoming: dict[str, list[ArchitectureEdge]] = defaultdict(list)
    for edge in graph.edges:
        edges_by_kind[edge.kind].append(edge)
        if edge.kind in _CONE_EDGE_KINDS:
            outgoing[edge.source].append(edge)
        if edge.kind in _REVERSE_CONE_EDGE_KINDS:
            incoming[edge.target].append(edge)
    root_nodes = tuple(
        node
        for node in graph.nodes
        if node.kind in {NodeKind.OPERATION, NodeKind.ENTRYPOINT}
    )
    cones: dict[str, dict[str, int]] = {}
    cone_edge_ids: list[str] = []
    cone_confidences: list[Confidence] = []
    max_hops = 0
    for root in root_nodes:
        seen = {root.node_id: 0}
        queue: deque[str] = deque([root.node_id])
        while queue:
            current = queue.popleft()
            for edge in outgoing.get(current, ()):
                cone_edge_ids.append(edge.edge_id)
                cone_confidences.append(edge.provenance.confidence)
                if edge.target in seen:
                    continue
                hops = seen[current] + 1
                seen[edge.target] = hops
                if hops > max_hops:
                    max_hops = hops
                queue.append(edge.target)
            for edge in incoming.get(current, ()):
                cone_edge_ids.append(edge.edge_id)
                cone_confidences.append(edge.provenance.confidence)
                if edge.source in seen:
                    continue
                hops = seen[current] + 1
                seen[edge.source] = hops
                if hops > max_hops:
                    max_hops = hops
                queue.append(edge.source)
        cones[root.node_id] = seen
    return _GraphView(
        graph=graph,
        nodes_by_id=nodes_by_id,
        nodes_by_kind={kind: tuple(items) for kind, items in nodes_by_kind.items()},
        edges_by_kind={kind: tuple(items) for kind, items in edges_by_kind.items()},
        roots=root_nodes,
        cones=cones,
        cone_edge_ids=tuple(sorted(set(cone_edge_ids))),
        cone_confidence=_worst_confidence(cone_confidences, empty=Confidence.EXACT),
        max_hops=max_hops,
        corpus_bound=corpus_bound,
    )


def _authority_multiplicity(view: _GraphView) -> _RawMeasure:
    authorities = _nodes_of(view, NodeKind.AUTHORITY)
    authorizes = [
        edge
        for edge in _edges_of(view, EdgeKind.AUTHORIZES)
        if view.nodes_by_id[edge.source].kind is NodeKind.AUTHORITY
    ]
    reexports = [
        edge
        for edge in _edges_of(view, EdgeKind.REEXPORTS)
        if view.nodes_by_id[edge.target].kind is NodeKind.AUTHORITY
    ]
    if not authorities and not authorizes and not reexports:
        return _empty_measure()
    owners_by_target: dict[str, set[str]] = defaultdict(set)
    attached: set[str] = set()
    for edge in authorizes:
        owners_by_target[edge.target].add(edge.source)
        attached.add(edge.source)
    extras = sum(max(0, len(owners) - 1) for owners in owners_by_target.values())
    extras += sum(1 for node in authorities if node.node_id not in attached)
    extras += len(reexports)
    denominator = _population(len(owners_by_target))
    unknown_denominator = 0 if denominator is not None else 1
    return _measure(
        numerator=extras,
        denominator=denominator,
        facts=(*authorities, *authorizes, *reexports),
        unknown_denominator=unknown_denominator,
    )


def _implementation_duplication(view: _GraphView) -> _RawMeasure:
    implementations = _nodes_of(view, *_IMPLEMENTATION_NODE_KINDS)
    candidates = _edges_of(
        view, EdgeKind.DUPLICATES, EdgeKind.SHADOWS, EdgeKind.SUPERSEDES
    )
    selected: list[ArchitectureEdge] = []
    for edge in candidates:
        source_kind = view.nodes_by_id[edge.source].kind
        target_kind = view.nodes_by_id[edge.target].kind
        if source_kind in _SCHEMA_RECEIPT_KINDS or target_kind in _SCHEMA_RECEIPT_KINDS:
            continue
        if source_kind in _IMPLEMENTATION_NODE_KINDS or target_kind in _IMPLEMENTATION_NODE_KINDS:
            selected.append(edge)
    if not implementations and not selected:
        return _empty_measure()
    return _measure(
        numerator=len(selected),
        denominator=_population(len(implementations)),
        facts=(*implementations, *selected),
        unknown_denominator=0 if implementations else 1,
    )


def _public_surface_area(view: _GraphView) -> _RawMeasure:
    surfaces = _nodes_of(view, NodeKind.INTERFACE, NodeKind.ENTRYPOINT)
    enclosing = _nodes_of(
        view,
        NodeKind.PACKAGE,
        NodeKind.MODULE,
        NodeKind.FILE,
        NodeKind.INTERFACE,
        NodeKind.ENTRYPOINT,
    )
    if not surfaces and not enclosing:
        return _empty_measure()
    return _measure(
        numerator=len(surfaces),
        denominator=_population(len(enclosing)),
        facts=(*surfaces, *enclosing),
        unknown_denominator=0 if enclosing else 1,
    )


def _dependency_cone_size(view: _GraphView) -> _RawMeasure:
    if not view.roots:
        return _empty_measure()
    numerator = sum(len(cone) for cone in view.cones.values())
    node_ids = {node_id for cone in view.cones.values() for node_id in cone}
    nodes = tuple(view.nodes_by_id[node_id] for node_id in node_ids)
    edges = tuple(
        edge for edge in view.graph.edges if edge.edge_id in set(view.cone_edge_ids)
    )
    extra = view.cone_confidence
    if not view.corpus_bound:
        extra = _worst_confidence([extra, Confidence.CONSERVATIVE])
    return _measure(
        numerator=numerator,
        denominator=len(view.roots),
        facts=(*view.roots, *nodes, *edges),
        extra_confidence=extra,
    )


def _dynamic_dispatch_uncertainty(view: _GraphView) -> _RawMeasure:
    dispatch = _edges_of(view, *_DISPATCH_EDGE_KINDS)
    if not dispatch:
        return _empty_measure()
    uncertain = tuple(
        edge
        for edge in dispatch
        if edge.provenance.confidence in {Confidence.HEURISTIC, Confidence.OPAQUE}
    )
    return _measure(
        numerator=len(uncertain),
        denominator=len(dispatch),
        facts=dispatch,
    )


def _state_ownership_ambiguity(view: _GraphView) -> _RawMeasure:
    states = _nodes_of(view, NodeKind.STATE)
    owner_edges = [
        edge
        for edge in _edges_of(view, *_OWNER_EDGE_KINDS)
        if view.nodes_by_id[edge.target].kind is NodeKind.STATE
    ]
    if not states:
        return _empty_measure()
    owners_by_state: dict[str, set[str]] = {node.node_id: set() for node in states}
    for edge in owner_edges:
        owners_by_state[edge.target].add(edge.source)
    ambiguous = tuple(
        node for node in states if len(owners_by_state[node.node_id]) != 1
    )
    return _measure(
        numerator=len(ambiguous),
        denominator=len(states),
        facts=(*states, *owner_edges),
    )


def _effect_opacity(view: _GraphView) -> _RawMeasure:
    effects = _nodes_of(view, NodeKind.EFFECT)
    effect_edges = _edges_of(view, *_EFFECT_EDGE_KINDS)
    facts: tuple[ArchitectureNode | ArchitectureEdge, ...] = (*effects, *effect_edges)
    if not facts:
        return _empty_measure()
    opaque = tuple(
        fact
        for fact in facts
        if fact.provenance.confidence in {Confidence.HEURISTIC, Confidence.OPAQUE}
    )
    return _measure(
        numerator=len(opaque),
        denominator=len(facts),
        facts=facts,
    )


def _compatibility_burden(view: _GraphView) -> _RawMeasure:
    compatibility_nodes = _nodes_of(view, NodeKind.COMPATIBILITY)
    adapts = _edges_of(view, EdgeKind.ADAPTS)
    deprecates = [
        edge
        for edge in _edges_of(view, EdgeKind.DEPRECATES)
        if view.nodes_by_id[edge.source].kind in _COMPATIBILITY_NODE_KINDS
        or view.nodes_by_id[edge.target].kind in _COMPATIBILITY_NODE_KINDS
    ]
    population = _nodes_of(
        view, NodeKind.INTERFACE, NodeKind.ENTRYPOINT, NodeKind.COMPATIBILITY
    )
    selected = (*compatibility_nodes, *adapts, *deprecates)
    if not selected and not population:
        return _empty_measure()
    return _measure(
        numerator=len(selected),
        denominator=_population(len(population)),
        facts=(*population, *selected),
        unknown_denominator=0 if population else 1,
    )


def _validation_amplification(view: _GraphView) -> _RawMeasure:
    validations = _nodes_of(view, NodeKind.TEST, NodeKind.PROOF)
    units = _nodes_of(view, NodeKind.OPERATION, NodeKind.ENTRYPOINT)
    evidence_edges = _edges_of(view, EdgeKind.TESTS, EdgeKind.PROVES, EdgeKind.INVALIDATES)
    if not validations and not units:
        return _empty_measure()
    return _measure(
        numerator=len(validations),
        denominator=_population(len(units)),
        facts=(*validations, *units, *evidence_edges),
        unknown_denominator=0 if units else 1,
    )


def _cache_fragmentation(view: _GraphView) -> _RawMeasure:
    caches = _nodes_of(view, NodeKind.ARTIFACT, NodeKind.GENERATED)
    generates = _edges_of(view, EdgeKind.GENERATES)
    if not caches and not generates:
        return _empty_measure()
    sources_by_target: dict[str, set[str]] = defaultdict(set)
    referenced: set[str] = set()
    for edge in generates:
        sources_by_target[edge.target].add(edge.source)
        referenced.add(edge.source)
        referenced.add(edge.target)
    extras = sum(max(0, len(sources) - 1) for sources in sources_by_target.values())
    unreferenced = tuple(node for node in caches if node.node_id not in referenced)
    extras += max(0, len(unreferenced) - 1) if unreferenced else 0
    logical = len(sources_by_target) + (1 if unreferenced else 0)
    return _measure(
        numerator=extras,
        denominator=_population(logical),
        facts=(*caches, *generates),
        unknown_denominator=0 if logical else 1,
    )


def _kind_drift(view: _GraphView, kind: NodeKind) -> _RawMeasure:
    nodes = _nodes_of(view, kind)
    drift_edges = [
        edge
        for edge in _edges_of(view, *_DRIFT_EDGE_KINDS)
        if view.nodes_by_id[edge.source].kind is kind
        or view.nodes_by_id[edge.target].kind is kind
    ]
    if not nodes:
        return _empty_measure()
    drifted_ids = {edge.source for edge in drift_edges} | {edge.target for edge in drift_edges}
    drifted = tuple(
        node
        for node in nodes
        if node.node_id in drifted_ids
    )
    return _measure(
        numerator=len(drifted),
        denominator=len(nodes),
        facts=(*nodes, *drift_edges),
    )


def _documentation_nodes(view: _GraphView) -> tuple[ArchitectureNode, ...]:
    return tuple(
        node
        for node in _nodes_of(view, *_DOCUMENTATION_NODE_KINDS)
        if _is_documentation_path(node.provenance.span.path)
    )


def _documentation_drift(view: _GraphView) -> _RawMeasure:
    documents = _documentation_nodes(view)
    if not documents:
        return _empty_measure()
    confirms = {edge.source for edge in _edges_of(view, EdgeKind.CONFIRMS)}
    drift_edges = [
        edge
        for edge in _edges_of(view, *_DRIFT_EDGE_KINDS)
        if edge.source in {node.node_id for node in documents}
        or edge.target in {node.node_id for node in documents}
    ]
    drifted_ids = {edge.source for edge in drift_edges} | {edge.target for edge in drift_edges}
    unconfirmed = tuple(
        node
        for node in documents
        if node.node_id not in confirms and node.node_id not in drifted_ids
    )
    drifted = tuple(node for node in documents if node.node_id in drifted_ids)
    extra = Confidence.CONSERVATIVE if unconfirmed else None
    return _measure(
        numerator=len(drifted) + len(unconfirmed),
        denominator=len(documents),
        facts=(*documents, *drift_edges, *_edges_of(view, EdgeKind.CONFIRMS)),
        extra_confidence=extra,
    )


def _merge_conflict_density(view: _GraphView) -> _RawMeasure:
    mutable = _nodes_of(
        view, NodeKind.FILE, NodeKind.STATE, NodeKind.SCHEMA, NodeKind.ARTIFACT
    )
    conflict_edges = _edges_of(view, *_CONFLICT_EDGE_KINDS)
    if not mutable and not conflict_edges:
        return _empty_measure()
    if not mutable:
        return _empty_measure()
    writers: dict[str, set[str]] = defaultdict(set)
    for edge in conflict_edges:
        writers[edge.target].add(edge.source)
    contended = tuple(
        node for node in mutable if len(writers.get(node.node_id, ())) > 1
    )
    return _measure(
        numerator=len(contended),
        denominator=len(mutable),
        facts=(*mutable, *conflict_edges),
    )


def _context_burden(view: _GraphView) -> _RawMeasure:
    if not view.roots:
        return _empty_measure()
    context_ids: list[str] = []
    for cone in view.cones.values():
        for node_id in cone:
            if view.nodes_by_id[node_id].kind in _CONTEXT_NODE_KINDS:
                context_ids.append(node_id)
    nodes = tuple(view.nodes_by_id[node_id] for node_id in set(context_ids))
    extra = view.cone_confidence
    if not view.corpus_bound:
        extra = _worst_confidence([extra, Confidence.CONSERVATIVE])
    unknown_numerator = 2 if not view.corpus_bound else 0
    return _measure(
        numerator=len(context_ids),
        denominator=len(view.roots),
        facts=(*view.roots, *nodes),
        extra_confidence=extra,
        unknown_numerator=unknown_numerator,
    )


def _kind_drift_schema(view: _GraphView) -> _RawMeasure:
    return _kind_drift(view, NodeKind.SCHEMA)


def _kind_drift_receipt(view: _GraphView) -> _RawMeasure:
    return _kind_drift(view, NodeKind.RECEIPT)


_CALCULATORS = {
    EntropyDimensionKind.AUTHORITY_MULTIPLICITY: _authority_multiplicity,
    EntropyDimensionKind.IMPLEMENTATION_DUPLICATION: _implementation_duplication,
    EntropyDimensionKind.PUBLIC_SURFACE_AREA: _public_surface_area,
    EntropyDimensionKind.DEPENDENCY_CONE_SIZE: _dependency_cone_size,
    EntropyDimensionKind.DYNAMIC_DISPATCH_UNCERTAINTY: _dynamic_dispatch_uncertainty,
    EntropyDimensionKind.STATE_OWNERSHIP_AMBIGUITY: _state_ownership_ambiguity,
    EntropyDimensionKind.EFFECT_OPACITY: _effect_opacity,
    EntropyDimensionKind.COMPATIBILITY_BURDEN: _compatibility_burden,
    EntropyDimensionKind.VALIDATION_AMPLIFICATION: _validation_amplification,
    EntropyDimensionKind.CACHE_FRAGMENTATION: _cache_fragmentation,
    EntropyDimensionKind.SCHEMA_DRIFT: _kind_drift_schema,
    EntropyDimensionKind.RECEIPT_DRIFT: _kind_drift_receipt,
    EntropyDimensionKind.DOCUMENTATION_DRIFT: _documentation_drift,
    EntropyDimensionKind.MERGE_CONFLICT_DENSITY: _merge_conflict_density,
    EntropyDimensionKind.CONTEXT_BURDEN: _context_burden,
}


def _record_from_raw(
    kind: EntropyDimensionKind,
    raw: _RawMeasure,
    graph: ArchitectureIR,
) -> EntropyDimensionRecord:
    return EntropyDimensionRecord(
        kind=kind,
        numerator=raw.numerator,
        denominator=raw.denominator,
        unit=_DIMENSION_UNITS[kind],
        uncertainty=DimensionUncertainty(
            confidence=raw.confidence,
            unknown_numerator=raw.unknown_numerator,
            unknown_denominator=raw.unknown_denominator,
        ),
        evidence=DimensionEvidence(
            node_ids=raw.node_ids,
            edge_ids=raw.edge_ids,
            extractor_identity=ENTROPY_EXTRACTOR_IDENTITY,
        ),
        architecture_ir_identity=graph.content_identity,
        repository_tree=graph.repository_tree,
        freshness=graph.freshness,
    )


def _count_kinds(
    view: _GraphView, node_ids: Iterable[str], kind: NodeKind
) -> int:
    return sum(1 for node_id in node_ids if view.nodes_by_id[node_id].kind is kind)


def _change_amplification(
    view: _GraphView, corpus: FrozenTaskCorpus
) -> ChangeAmplificationMeasure:
    union_ids = {node_id for cone in view.cones.values() for node_id in cone}
    files = _count_kinds(view, union_ids, NodeKind.FILE)
    symbols = _count_kinds(view, union_ids, NodeKind.SYMBOL)
    interfaces = _count_kinds(view, union_ids, NodeKind.INTERFACE)
    schemas = _count_kinds(view, union_ids, NodeKind.SCHEMA)
    effects = _count_kinds(view, union_ids, NodeKind.EFFECT)
    tests = _count_kinds(view, union_ids, NodeKind.TEST)
    proofs = _count_kinds(view, union_ids, NodeKind.PROOF)
    providers = _count_kinds(view, union_ids, NodeKind.PROVIDER)
    runtime_paths = _count_kinds(view, union_ids, NodeKind.ENTRYPOINT) + _count_kinds(
        view, union_ids, NodeKind.PROVIDER
    )
    owners = _count_kinds(view, union_ids, NodeKind.AUTHORITY)
    hops = view.max_hops
    tokens = 0
    raw_expansions = 0
    numerator = (
        files
        + symbols
        + interfaces
        + schemas
        + effects
        + tests
        + proofs
        + providers
        + runtime_paths
        + owners
        + hops
    )
    denominator = _population(len(view.roots))
    extra = view.cone_confidence
    unknown_numerator = 2
    unknown_denominator = 0 if denominator is not None else 1
    if not view.corpus_bound:
        extra = _worst_confidence([extra, Confidence.CONSERVATIVE])
        unknown_numerator += 1
    evidence = DimensionEvidence(
        node_ids=tuple(union_ids),
        edge_ids=view.cone_edge_ids,
        extractor_identity=ENTROPY_EXTRACTOR_IDENTITY,
    )
    return ChangeAmplificationMeasure(
        numerator=numerator,
        denominator=denominator,
        unit="amplified_units",
        uncertainty=DimensionUncertainty(
            confidence=_worst_confidence(
                [extra, Confidence.CONSERVATIVE],
                empty=Confidence.CONSERVATIVE,
            ),
            unknown_numerator=unknown_numerator,
            unknown_denominator=unknown_denominator,
        ),
        files=files,
        symbols=symbols,
        interfaces=interfaces,
        schemas=schemas,
        effects=effects,
        tests=tests,
        proofs=proofs,
        providers=providers,
        runtime_paths=runtime_paths,
        tokens=tokens,
        raw_expansions=raw_expansions,
        hops=hops,
        owners=owners,
        architecture_ir_identity=view.graph.content_identity,
        repository_tree=view.graph.repository_tree,
        freshness=view.graph.freshness,
        frozen_task_corpus_identity=corpus.content_identity,
        evidence_identity=_evidence_identity(evidence),
    )


def measure_entropy_dimensions(
    graph: ArchitectureIR | Mapping[str, Any],
    *,
    frozen_task_corpus: FrozenTaskCorpus | Mapping[str, Any] | None = None,
) -> tuple[EntropyDimensionRecord, ...]:
    """Calculate every closed entropy dimension independently from ArchitectureIR."""

    architecture = _require_architecture_ir(graph)
    corpus = (
        FrozenTaskCorpus.unbound()
        if frozen_task_corpus is None
        else (
            frozen_task_corpus
            if isinstance(frozen_task_corpus, FrozenTaskCorpus)
            else FrozenTaskCorpus.from_mapping(frozen_task_corpus)
        )
    )
    view = _build_view(
        architecture,
        corpus_bound=corpus.corpus_id != UNBOUND_FROZEN_TASK_CORPUS_ID and bool(corpus.task_ids),
    )
    return tuple(
        _record_from_raw(kind, _CALCULATORS[kind](view), architecture)
        for kind in REQUIRED_ENTROPY_DIMENSIONS
    )


def measure_change_amplification(
    graph: ArchitectureIR | Mapping[str, Any],
    *,
    frozen_task_corpus: FrozenTaskCorpus | Mapping[str, Any] | None = None,
) -> ChangeAmplificationMeasure:
    """Calculate the independent change-amplification measure for one graph."""

    architecture = _require_architecture_ir(graph)
    corpus = (
        FrozenTaskCorpus.unbound()
        if frozen_task_corpus is None
        else (
            frozen_task_corpus
            if isinstance(frozen_task_corpus, FrozenTaskCorpus)
            else FrozenTaskCorpus.from_mapping(frozen_task_corpus)
        )
    )
    view = _build_view(
        architecture,
        corpus_bound=corpus.corpus_id != UNBOUND_FROZEN_TASK_CORPUS_ID and bool(corpus.task_ids),
    )
    return _change_amplification(view, corpus)


def measure_semantic_entropy(
    graph: ArchitectureIR | Mapping[str, Any],
    *,
    frozen_task_corpus: FrozenTaskCorpus | Mapping[str, Any] | None = None,
) -> SemanticEntropyReport:
    """Build a versioned SemanticEntropyReport from a closed ArchitectureIR graph."""

    architecture = _require_architecture_ir(graph)
    corpus = (
        FrozenTaskCorpus.unbound()
        if frozen_task_corpus is None
        else (
            frozen_task_corpus
            if isinstance(frozen_task_corpus, FrozenTaskCorpus)
            else FrozenTaskCorpus.from_mapping(frozen_task_corpus)
        )
    )
    dimensions = measure_entropy_dimensions(
        architecture, frozen_task_corpus=corpus
    )
    amplification = measure_change_amplification(
        architecture, frozen_task_corpus=corpus
    )
    return SemanticEntropyReport(
        architecture_ir_identity=architecture.content_identity,
        repository_tree=architecture.repository_tree,
        freshness=architecture.freshness,
        frozen_task_corpus=corpus,
        dimensions=dimensions,
        change_amplification=amplification,
    )


def canonical_entropy_vector(
    report: SemanticEntropyReport,
) -> tuple[tuple[str, int, int | None, str], ...]:
    """Return the independently auditable (kind, numerator, denominator, unit) vector."""

    return tuple(
        (item.kind.value, item.numerator, item.denominator, item.unit)
        for item in report.dimensions
    )


def derive_non_probative_ranking(
    report: SemanticEntropyReport,
) -> tuple[str, ...]:
    """Derive a prioritization ranking. This is never proof or safety authority."""

    if not RANKING_IS_NON_PROBATIVE or not ENTROPY_IS_PRIORITIZATION_ONLY:
        raise EntropyAuthorityError("entropy ranking cannot be treated as authoritative")

    def _key(item: EntropyDimensionRecord) -> tuple[int, int]:
        ratio = item.ratio_basis_points()
        if ratio is None:
            return (1, 0)
        return (0, -ratio)

    return tuple(item.kind.value for item in sorted(report.dimensions, key=_key))


def entropy_satisfies_safety_predicate(
    report: SemanticEntropyReport,
    predicate: str,
) -> bool:
    """Entropy scores never satisfy a non-compensable safety predicate."""

    if not isinstance(report, SemanticEntropyReport):
        raise EntropyContractError("safety checks require a SemanticEntropyReport")
    name = _require_text(predicate, "predicate", error_type=EntropyContractError)
    if name not in CLOSED_SAFETY_PREDICATES:
        raise EntropyContractError(f"unsupported safety predicate: {name!r}")
    return False


def entropy_establishes(report: SemanticEntropyReport, claim: str) -> bool:
    """Entropy never establishes safety, equivalence, ownership, or promotion."""

    if not isinstance(report, SemanticEntropyReport):
        raise EntropyContractError("authority checks require a SemanticEntropyReport")
    name = _require_text(claim, "claim", error_type=EntropyContractError)
    if name not in CLOSED_NON_AUTHORITY_CLAIMS:
        raise EntropyContractError(f"unsupported entropy authority claim: {name!r}")
    return False


def refuse_entropy_authority(action: str) -> None:
    """Reject metric-driven deletion, promotion, or safety inference."""

    name = _require_text(action, "action", error_type=EntropyContractError)
    raise EntropyAuthorityError(
        f"entropy metrics cannot establish {name}; they are prioritization signals only"
    )
