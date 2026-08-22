"""Candidate-tier contract extraction with retained conflicts (PCAR-010).

`ContractCandidateExtractor` mines input/output schemas, pre/postconditions,
effects, frame conditions, errors, idempotency, reversibility, authority,
policy, confirmation, resource bounds, freshness, and observation
requirements from declared types, schemas, tests, runtime checks, proofs,
accepted receipts, negative tests, mutants, public contracts, and explicitly
marked authoritative documents. ArchitectureIR edges may add implementation
observations only.

Every subject emits all contract dimensions. Absent dimensions stay explicit.
Source comparison retains precedence ranks without resolving a winner.
Conflicting evidence yields typed `ContractAmbiguity`. Negative and mutant
evidence is retained. Implementation cannot win. Unmarked documentation is
not authority. Tests are not assumed complete. Repetition in code or tests is
not promoted to a requirement. Candidates remain candidate-tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .architecture_ir import ArchitectureIR, ArchitectureNode
from .contracts import (
    ArchitectureContractError,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
)

CONTRACT_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-candidate@1"
)
CONTRACT_CANDIDATE_VERSION = 1
CONTRACT_CANDIDATE_EVIDENCE = "pcar/contract-candidate@1"
CONTRACT_AMBIGUITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-ambiguity@1"
)
CONTRACT_AMBIGUITY_VERSION = 1
CONTRACT_AMBIGUITY_EVIDENCE = "pcar/contract-ambiguity@1"
CONTRACT_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-evidence-unit@1"
)
CONTRACT_EVIDENCE_VERSION = 1
CONTRACT_DIMENSION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-dimension@1"
)
CONTRACT_DIMENSION_VERSION = 1
SOURCE_COMPARISON_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-source-comparison@1"
)
SOURCE_COMPARISON_VERSION = 1
CONTRACT_EXTRACTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-extraction-result@1"
)
CONTRACT_EXTRACTION_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-010-contract-candidate-extractor"
TASK_ID = "PCAR-010"
DEFAULT_FRESHNESS = "pcar-010-contract-candidate"
EFFECT_CLASS = "read_only_analysis"
EXTRACTOR_CAN_PROMOTE_REQUIREMENT = False
EXTRACTOR_CAN_RESOLVE_IMPLEMENTATION_WINS = False
EXTRACTOR_CAN_HIDE_CONFLICTS = False
EXTRACTOR_CAN_TREAT_TESTS_AS_COMPLETE = False
EXTRACTOR_CAN_TREAT_UNMARKED_DOCS_AS_AUTHORITY = False
CANDIDATE_TIER_ONLY = True
IMPLEMENTATION_WINS_PROHIBITED = True
HIDDEN_CONFLICT_PROHIBITED = True
TEST_COMPLETENESS_ASSUMPTION_PROHIBITED = True
UNMARKED_DOCUMENTATION_IS_NOT_AUTHORITY = True
REPETITION_IS_NOT_A_REQUIREMENT = True

_UNKNOWN_FIELD_MESSAGE = "unknown contract-candidate field"
_MISSING_FIELD_MESSAGE = "missing contract-candidate field"
_HIDDEN_CONFLICT_MESSAGE = "hidden contract conflict is prohibited"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")


class ContractExtractorError(ArchitectureContractError):
    """Fail-closed contract-candidate extraction error."""


class ContractExtractorAuthorityError(ContractExtractorError):
    """Raised when extraction is asked to promote or silently resolve."""


class ContractDimension(str, Enum):
    """Closed candidate-contract dimension vocabulary (PCAR-PLAN-R1)."""

    INPUTS = "inputs"
    OUTPUTS = "outputs"
    PRECONDITIONS = "preconditions"
    POSTCONDITIONS = "postconditions"
    EFFECTS = "effects"
    FRAMES = "frames"
    ERRORS = "errors"
    IDEMPOTENCY = "idempotency"
    REVERSIBILITY = "reversibility"
    AUTHORITY = "authority"
    POLICY = "policy"
    CONFIRMATION = "confirmation"
    BOUNDS = "bounds"
    FRESHNESS = "freshness"
    OBSERVATIONS = "observations"


REQUIRED_CONTRACT_DIMENSIONS: tuple[ContractDimension, ...] = tuple(ContractDimension)
CLOSED_CONTRACT_DIMENSIONS: frozenset[str] = frozenset(
    item.value for item in ContractDimension
)


class ContractEvidenceSource(str, Enum):
    """Closed evidence-source vocabulary for candidate contracts."""

    PUBLIC_CONTRACT = "public_contract"
    TYPE = "type"
    SCHEMA = "schema"
    PROOF = "proof"
    ACCEPTED_RECEIPT = "accepted_receipt"
    TEST = "test"
    NEGATIVE_TEST = "negative_test"
    RUNTIME_CHECK = "runtime_check"
    MUTANT = "mutant"
    AUTHORITATIVE_DOCUMENT = "authoritative_document"
    IMPLEMENTATION = "implementation"


CLOSED_EVIDENCE_SOURCES: frozenset[str] = frozenset(
    item.value for item in ContractEvidenceSource
)
SOURCE_PRECEDENCE: tuple[ContractEvidenceSource, ...] = (
    ContractEvidenceSource.PUBLIC_CONTRACT,
    ContractEvidenceSource.TYPE,
    ContractEvidenceSource.SCHEMA,
    ContractEvidenceSource.PROOF,
    ContractEvidenceSource.ACCEPTED_RECEIPT,
    ContractEvidenceSource.TEST,
    ContractEvidenceSource.NEGATIVE_TEST,
    ContractEvidenceSource.RUNTIME_CHECK,
    ContractEvidenceSource.MUTANT,
    ContractEvidenceSource.AUTHORITATIVE_DOCUMENT,
    ContractEvidenceSource.IMPLEMENTATION,
)
SOURCE_PRECEDENCE_RANK: dict[ContractEvidenceSource, int] = {
    ContractEvidenceSource.PUBLIC_CONTRACT: 0,
    ContractEvidenceSource.TYPE: 1,
    ContractEvidenceSource.SCHEMA: 2,
    ContractEvidenceSource.PROOF: 3,
    ContractEvidenceSource.ACCEPTED_RECEIPT: 4,
    ContractEvidenceSource.TEST: 5,
    ContractEvidenceSource.NEGATIVE_TEST: 5,
    ContractEvidenceSource.RUNTIME_CHECK: 6,
    ContractEvidenceSource.MUTANT: 7,
    ContractEvidenceSource.AUTHORITATIVE_DOCUMENT: 8,
    ContractEvidenceSource.IMPLEMENTATION: 100,
}
REVIEWED_SOURCES: frozenset[ContractEvidenceSource] = frozenset(
    {
        ContractEvidenceSource.PUBLIC_CONTRACT,
        ContractEvidenceSource.TYPE,
        ContractEvidenceSource.SCHEMA,
        ContractEvidenceSource.PROOF,
        ContractEvidenceSource.ACCEPTED_RECEIPT,
    }
)
REPETITION_SOURCES: frozenset[ContractEvidenceSource] = frozenset(
    {
        ContractEvidenceSource.TEST,
        ContractEvidenceSource.NEGATIVE_TEST,
        ContractEvidenceSource.RUNTIME_CHECK,
        ContractEvidenceSource.IMPLEMENTATION,
        ContractEvidenceSource.MUTANT,
    }
)
NON_DEFINING_SOURCES: frozenset[ContractEvidenceSource] = frozenset(
    {ContractEvidenceSource.IMPLEMENTATION}
)


class EvidencePolarity(str, Enum):
    """Whether a unit asserts, forbids, or mutates a dimension value."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    MUTANT = "mutant"


CLOSED_EVIDENCE_POLARITIES: frozenset[str] = frozenset(
    item.value for item in EvidencePolarity
)


class DimensionStatus(str, Enum):
    """Closed per-dimension presence vocabulary. Absent stays explicit."""

    PRESENT = "present"
    ABSENT = "absent"
    CONFLICTED = "conflicted"
    NEGATIVE = "negative"


CLOSED_DIMENSION_STATUSES: frozenset[str] = frozenset(
    item.value for item in DimensionStatus
)


class AmbiguityKind(str, Enum):
    """Closed conflict vocabulary retained instead of silent resolution."""

    CONFLICTING_VALUES = "conflicting_values"
    IMPLEMENTATION_WINS_REJECTED = "implementation_wins_rejected"
    UNMARKED_DOCUMENTATION = "unmarked_documentation"
    NEGATIVE_CONTRADICTS_POSITIVE = "negative_contradicts_positive"
    MUTANT_SURVIVES = "mutant_survives"
    TEST_COMPLETENESS_ASSUMED = "test_completeness_assumed"
    SOURCE_PRECEDENCE_UNRESOLVED = "source_precedence_unresolved"


CLOSED_AMBIGUITY_KINDS: frozenset[str] = frozenset(
    item.value for item in AmbiguityKind
)
REQUIRED_AMBIGUITY_KINDS: tuple[AmbiguityKind, ...] = tuple(AmbiguityKind)


class ComparisonDisposition(str, Enum):
    """Closed pairwise source-comparison vocabulary."""

    AGREE = "agree"
    CONFLICT = "conflict"
    REPETITION = "repetition"


CLOSED_COMPARISON_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in ComparisonDisposition
)


class ContractTier(str, Enum):
    """Closed promotion vocabulary. Only candidate-tier is admitted."""

    CANDIDATE = "candidate"


CLOSED_CONTRACT_TIERS: frozenset[str] = frozenset(item.value for item in ContractTier)


_SUBJECT_NODE_KINDS = frozenset(
    {
        NodeKind.OPERATION,
        NodeKind.SYMBOL,
        NodeKind.INTERFACE,
        NodeKind.ENTRYPOINT,
    }
)
_EDGE_DIMENSIONS: dict[EdgeKind, tuple[ContractDimension, str]] = {
    EdgeKind.READS: (ContractDimension.EFFECTS, "reads"),
    EdgeKind.WRITES: (ContractDimension.EFFECTS, "writes"),
    EdgeKind.MUTATES: (ContractDimension.EFFECTS, "mutates"),
    EdgeKind.PERSISTS: (ContractDimension.EFFECTS, "persists"),
    EdgeKind.EXECUTES: (ContractDimension.EFFECTS, "executes"),
    EdgeKind.GENERATES: (ContractDimension.EFFECTS, "generates"),
    EdgeKind.SERIALIZES: (ContractDimension.OUTPUTS, "serializes"),
    EdgeKind.DESERIALIZES: (ContractDimension.INPUTS, "deserializes"),
    EdgeKind.OBSERVES: (ContractDimension.OBSERVATIONS, "observes"),
    EdgeKind.AUTHORIZES: (ContractDimension.AUTHORITY, "authorizes"),
    EdgeKind.EVALUATES_POLICY: (ContractDimension.POLICY, "evaluates_policy"),
    EdgeKind.CONFIRMS: (ContractDimension.CONFIRMATION, "confirms"),
    EdgeKind.INVALIDATES: (ContractDimension.ERRORS, "invalidates"),
}
_NODE_SOURCE_KINDS: dict[NodeKind, ContractEvidenceSource] = {
    NodeKind.TEST: ContractEvidenceSource.TEST,
    NodeKind.PROOF: ContractEvidenceSource.PROOF,
    NodeKind.RECEIPT: ContractEvidenceSource.ACCEPTED_RECEIPT,
    NodeKind.SCHEMA: ContractEvidenceSource.SCHEMA,
    NodeKind.INTERFACE: ContractEvidenceSource.TYPE,
}

_EVIDENCE_FIELDS = frozenset(
    {
        "assumes_test_completeness",
        "content_identity",
        "dimension",
        "marked_authoritative",
        "polarity",
        "provenance",
        "public_contract",
        "schema",
        "source_kind",
        "subject",
        "value",
        "version",
    }
)
_DIMENSION_RECORD_FIELDS = frozenset(
    {
        "content_identity",
        "dimension",
        "observation_only",
        "schema",
        "source_kinds",
        "status",
        "values",
        "version",
    }
)
_AMBIGUITY_FIELDS = frozenset(
    {
        "content_identity",
        "dimension",
        "kind",
        "message",
        "provenance",
        "retained",
        "schema",
        "source_kinds",
        "subject",
        "values",
        "version",
    }
)
_COMPARISON_FIELDS = frozenset(
    {
        "content_identity",
        "dimension",
        "disposition",
        "left_precedence",
        "left_source",
        "left_value",
        "retained_conflict",
        "right_precedence",
        "right_source",
        "right_value",
        "schema",
        "subject",
        "version",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "ambiguities",
        "can_promote_requirement",
        "comparisons",
        "content_identity",
        "dimensions",
        "evidence",
        "freshness",
        "repository_tree",
        "schema",
        "subject",
        "tier",
        "version",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "ambiguities",
        "architecture_ir_identity",
        "can_hide_conflicts",
        "can_promote_requirement",
        "can_resolve_implementation_wins",
        "can_treat_tests_as_complete",
        "can_treat_unmarked_docs_as_authority",
        "candidate_tier",
        "candidates",
        "comparisons",
        "content_identity",
        "effect_class",
        "freshness",
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
        raise ContractExtractorError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise ContractExtractorError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise ContractExtractorError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractExtractorError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=ContractExtractorError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ContractExtractorError(f"{name} must be a boolean")
    return value


def _wrap_contract(exc: ArchitectureContractError) -> ContractExtractorError:
    if isinstance(exc, ContractExtractorError):
        return exc
    return ContractExtractorError(str(exc))


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _require_architecture_ir(
    graph: ArchitectureIR | Mapping[str, Any] | None,
) -> ArchitectureIR | None:
    if graph is None:
        return None
    if isinstance(graph, ArchitectureIR):
        return graph
    try:
        return ArchitectureIR.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise ContractExtractorError(str(exc)) from exc


def source_precedence_rank(source: ContractEvidenceSource | str) -> int:
    """Return the retained precedence rank for one evidence source kind."""

    parsed = _closed_enum(
        source,
        ContractEvidenceSource,
        "evidence source",
        error_type=ContractExtractorError,
    )
    return SOURCE_PRECEDENCE_RANK[parsed]


def _source_kind_tuple(value: Any, name: str) -> tuple[ContractEvidenceSource, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractExtractorError(f"{name} must be a list of evidence sources")
    items = tuple(
        _closed_enum(
            item,
            ContractEvidenceSource,
            "evidence source",
            error_type=ContractExtractorError,
        )
        for item in value
    )
    return tuple(sorted(set(items), key=lambda item: (SOURCE_PRECEDENCE_RANK[item], item.value)))


def _record_tuple(value: Any, name: str, record_type: type[Any]) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractExtractorError(f"{name} must be a list of objects")
    records = tuple(
        item if isinstance(item, record_type) else record_type.from_mapping(item)
        for item in value
    )
    return tuple(sorted(records, key=lambda item: item.content_identity))


def _graph_subject(source: ArchitectureNode, target: ArchitectureNode) -> str:
    if target.kind in _SUBJECT_NODE_KINDS:
        return target.node_id
    if source.kind in _SUBJECT_NODE_KINDS:
        return source.node_id
    return source.node_id


def _graph_source_kind(source: ArchitectureNode, target: ArchitectureNode) -> ContractEvidenceSource:
    mapped = _NODE_SOURCE_KINDS.get(source.kind)
    if mapped is not None:
        return mapped
    return ContractEvidenceSource.IMPLEMENTATION


def mine_graph_evidence(architecture: ArchitectureIR) -> tuple["ContractEvidenceUnit", ...]:
    """Project ArchitectureIR edges into implementation-tier observations."""

    nodes = {node.node_id: node for node in architecture.nodes}
    units: list[ContractEvidenceUnit] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for edge in architecture.edges:
        mapped = _EDGE_DIMENSIONS.get(edge.kind)
        if mapped is None:
            continue
        source = nodes[edge.source]
        target = nodes[edge.target]
        dimension, verb = mapped
        subject = _graph_subject(source, target)
        other = target.node_id if subject == source.node_id else source.node_id
        value = f"{verb}:{other}"
        source_kind = _graph_source_kind(source, target)
        key = (subject, source_kind.value, dimension.value, value, EvidencePolarity.POSITIVE.value)
        if key in seen:
            continue
        seen.add(key)
        units.append(
            ContractEvidenceUnit(
                subject=subject,
                source_kind=source_kind,
                dimension=dimension,
                value=value,
                polarity=EvidencePolarity.POSITIVE,
                marked_authoritative=False,
                public_contract=False,
                assumes_test_completeness=False,
                provenance=edge.provenance,
            )
        )
    return tuple(sorted(units, key=lambda item: item.content_identity))


def _unit_contributes(unit: "ContractEvidenceUnit") -> bool:
    if unit.source_kind is ContractEvidenceSource.AUTHORITATIVE_DOCUMENT:
        return unit.marked_authoritative is True
    return True


def _dimension_state(
    units: Sequence["ContractEvidenceUnit"],
) -> tuple[DimensionStatus, tuple[str, ...], tuple[ContractEvidenceSource, ...], bool]:
    contributing = tuple(unit for unit in units if _unit_contributes(unit))
    source_kinds = _source_kind_tuple(
        tuple(unit.source_kind for unit in contributing),
        "source_kinds",
    )
    positive = {unit.value for unit in contributing if unit.polarity is EvidencePolarity.POSITIVE}
    negative = {unit.value for unit in contributing if unit.polarity is EvidencePolarity.NEGATIVE}
    mutant = {unit.value for unit in contributing if unit.polarity is EvidencePolarity.MUTANT}
    if not contributing or not (positive | negative | mutant):
        return DimensionStatus.ABSENT, (), source_kinds, False
    conflicted = len(positive) > 1
    if positive & negative:
        conflicted = True
    if positive and mutant:
        conflicted = True
    if negative and mutant and not positive and negative != mutant:
        conflicted = True
    observation_only = bool(contributing) and all(
        unit.source_kind in NON_DEFINING_SOURCES for unit in contributing
    )
    if conflicted:
        values = tuple(sorted(positive | negative | mutant))
        return DimensionStatus.CONFLICTED, values, source_kinds, observation_only
    if not positive:
        return DimensionStatus.NEGATIVE, tuple(sorted(negative | mutant)), source_kinds, observation_only
    return DimensionStatus.PRESENT, tuple(sorted(positive)), source_kinds, observation_only


def _comparison_disposition(
    left: "ContractEvidenceUnit",
    right: "ContractEvidenceUnit",
) -> ComparisonDisposition:
    if left.value != right.value:
        return ComparisonDisposition.CONFLICT
    if left.source_kind in REVIEWED_SOURCES or right.source_kind in REVIEWED_SOURCES:
        return ComparisonDisposition.AGREE
    if {left.source_kind, right.source_kind} <= REPETITION_SOURCES:
        return ComparisonDisposition.REPETITION
    return ComparisonDisposition.AGREE


@dataclass(frozen=True)
class ContractEvidenceUnit:
    """One compact, source-bound claim or counterexample for a dimension."""

    subject: str
    source_kind: ContractEvidenceSource
    dimension: ContractDimension
    value: str
    provenance: SourceFactIdentity
    polarity: EvidencePolarity = EvidencePolarity.POSITIVE
    marked_authoritative: bool = False
    public_contract: bool = False
    assumes_test_completeness: bool = False
    schema: str = CONTRACT_EVIDENCE_SCHEMA
    version: int = CONTRACT_EVIDENCE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ContractExtractorError)
        if schema != CONTRACT_EVIDENCE_SCHEMA:
            raise ContractExtractorError("unexpected contract-evidence schema")
        version = _require_int(self.version, "version", error_type=ContractExtractorError)
        if version != CONTRACT_EVIDENCE_VERSION:
            raise ContractExtractorError("unexpected contract-evidence version")
        subject = _require_text(self.subject, "subject", error_type=ContractExtractorError)
        if _looks_like_content_identity(subject):
            raise ContractExtractorError("content identity is not a contract subject")
        source_kind = _closed_enum(
            self.source_kind,
            ContractEvidenceSource,
            "evidence source",
            error_type=ContractExtractorError,
        )
        dimension = _closed_enum(
            self.dimension,
            ContractDimension,
            "contract dimension",
            error_type=ContractExtractorError,
        )
        value = _require_text(self.value, "value", error_type=ContractExtractorError)
        polarity = _closed_enum(
            self.polarity,
            EvidencePolarity,
            "evidence polarity",
            error_type=ContractExtractorError,
        )
        marked = _require_bool(self.marked_authoritative, "marked_authoritative")
        public_contract = _require_bool(self.public_contract, "public_contract")
        assumes = _require_bool(self.assumes_test_completeness, "assumes_test_completeness")
        if source_kind is ContractEvidenceSource.IMPLEMENTATION and marked:
            raise ContractExtractorError("implementation cannot be marked authoritative")
        if source_kind is ContractEvidenceSource.PUBLIC_CONTRACT:
            public_contract = True
        elif public_contract and source_kind is not ContractEvidenceSource.PUBLIC_CONTRACT:
            raise ContractExtractorError("public_contract requires source kind public_contract")
        try:
            provenance = (
                self.provenance
                if isinstance(self.provenance, SourceFactIdentity)
                else SourceFactIdentity.from_mapping(self.provenance)
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "marked_authoritative", marked)
        object.__setattr__(self, "public_contract", public_contract)
        object.__setattr__(self, "assumes_test_completeness", assumes)
        object.__setattr__(self, "provenance", provenance)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ContractExtractorError,
                )
            )
            if claimed != identity:
                raise ContractExtractorError("contract-evidence content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "assumes_test_completeness": self.assumes_test_completeness,
            "dimension": self.dimension.value,
            "marked_authoritative": self.marked_authoritative,
            "polarity": self.polarity.value,
            "provenance": self.provenance.to_dict(),
            "public_contract": self.public_contract,
            "schema": self.schema,
            "source_kind": self.source_kind.value,
            "subject": self.subject,
            "value": self.value,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ContractExtractorError("contract-evidence content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContractEvidenceUnit":
        mapping = _require_mapping(payload, error_type=ContractExtractorError)
        _require_fields(mapping, _EVIDENCE_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise ContractExtractorError("evidence provenance must be an object")
        record = cls(
            subject=mapping["subject"],
            source_kind=mapping["source_kind"],
            dimension=mapping["dimension"],
            value=mapping["value"],
            polarity=mapping["polarity"],
            marked_authoritative=mapping["marked_authoritative"],
            public_contract=mapping["public_contract"],
            assumes_test_completeness=mapping["assumes_test_completeness"],
            provenance=SourceFactIdentity.from_mapping(provenance_payload),
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise ContractExtractorError("contract-evidence content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ContractDimensionRecord:
    """One subject's observed state for a single contract dimension."""

    dimension: ContractDimension
    status: DimensionStatus
    values: tuple[str, ...] = ()
    source_kinds: tuple[ContractEvidenceSource, ...] = ()
    observation_only: bool = False
    schema: str = CONTRACT_DIMENSION_SCHEMA
    version: int = CONTRACT_DIMENSION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ContractExtractorError)
        if schema != CONTRACT_DIMENSION_SCHEMA:
            raise ContractExtractorError("unexpected contract-dimension schema")
        version = _require_int(self.version, "version", error_type=ContractExtractorError)
        if version != CONTRACT_DIMENSION_VERSION:
            raise ContractExtractorError("unexpected contract-dimension version")
        dimension = _closed_enum(
            self.dimension,
            ContractDimension,
            "contract dimension",
            error_type=ContractExtractorError,
        )
        status = _closed_enum(
            self.status,
            DimensionStatus,
            "dimension status",
            error_type=ContractExtractorError,
        )
        values = _require_text_tuple(self.values, "values")
        source_kinds = _source_kind_tuple(self.source_kinds, "source_kinds")
        observation_only = _require_bool(self.observation_only, "observation_only")
        if status is DimensionStatus.ABSENT:
            if values:
                raise ContractExtractorError("absent dimensions cannot retain values")
        elif not values:
            raise ContractExtractorError("non-absent dimensions must retain values")
        if status is DimensionStatus.PRESENT and len(values) != 1:
            raise ContractExtractorError("present dimensions must have exactly one value")
        if observation_only and ContractEvidenceSource.IMPLEMENTATION not in source_kinds:
            raise ContractExtractorError("observation-only dimensions require implementation sources")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_kinds", source_kinds)
        object.__setattr__(self, "observation_only", observation_only)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ContractExtractorError,
                )
            )
            if claimed != identity:
                raise ContractExtractorError("contract-dimension content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    @property
    def absent(self) -> bool:
        return self.status is DimensionStatus.ABSENT

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "observation_only": self.observation_only,
            "schema": self.schema,
            "source_kinds": [item.value for item in self.source_kinds],
            "status": self.status.value,
            "values": list(self.values),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ContractExtractorError("contract-dimension content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContractDimensionRecord":
        mapping = _require_mapping(payload, error_type=ContractExtractorError)
        _require_fields(mapping, _DIMENSION_RECORD_FIELDS)
        record = cls(
            dimension=mapping["dimension"],
            status=mapping["status"],
            values=mapping["values"],
            source_kinds=mapping["source_kinds"],
            observation_only=mapping["observation_only"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise ContractExtractorError("contract-dimension content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ContractAmbiguity:
    """Typed conflict that extraction retains instead of resolving."""

    subject: str
    dimension: ContractDimension
    kind: AmbiguityKind
    values: tuple[str, ...]
    source_kinds: tuple[ContractEvidenceSource, ...]
    message: str
    provenance: SourceFactIdentity
    retained: bool = True
    schema: str = CONTRACT_AMBIGUITY_SCHEMA
    version: int = CONTRACT_AMBIGUITY_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ContractExtractorError)
        if schema != CONTRACT_AMBIGUITY_SCHEMA:
            raise ContractExtractorError("unexpected contract-ambiguity schema")
        version = _require_int(self.version, "version", error_type=ContractExtractorError)
        if version != CONTRACT_AMBIGUITY_VERSION:
            raise ContractExtractorError("unexpected contract-ambiguity version")
        subject = _require_text(self.subject, "subject", error_type=ContractExtractorError)
        dimension = _closed_enum(
            self.dimension,
            ContractDimension,
            "contract dimension",
            error_type=ContractExtractorError,
        )
        kind = _closed_enum(
            self.kind,
            AmbiguityKind,
            "ambiguity kind",
            error_type=ContractExtractorError,
        )
        values = _require_text_tuple(self.values, "values")
        source_kinds = _source_kind_tuple(self.source_kinds, "source_kinds")
        message = _require_text(self.message, "message", error_type=ContractExtractorError)
        retained = _require_bool(self.retained, "retained")
        if retained is not True:
            raise ContractExtractorError(_HIDDEN_CONFLICT_MESSAGE)
        try:
            provenance = (
                self.provenance
                if isinstance(self.provenance, SourceFactIdentity)
                else SourceFactIdentity.from_mapping(self.provenance)
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_kinds", source_kinds)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "retained", True)
        object.__setattr__(self, "provenance", provenance)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ContractExtractorError,
                )
            )
            if claimed != identity:
                raise ContractExtractorError("contract-ambiguity content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "kind": self.kind.value,
            "message": self.message,
            "provenance": self.provenance.to_dict(),
            "retained": True,
            "schema": self.schema,
            "source_kinds": [item.value for item in self.source_kinds],
            "subject": self.subject,
            "values": list(self.values),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ContractExtractorError("contract-ambiguity content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContractAmbiguity":
        mapping = _require_mapping(payload, error_type=ContractExtractorError)
        _require_fields(mapping, _AMBIGUITY_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise ContractExtractorError("ambiguity provenance must be an object")
        record = cls(
            subject=mapping["subject"],
            dimension=mapping["dimension"],
            kind=mapping["kind"],
            values=mapping["values"],
            source_kinds=mapping["source_kinds"],
            message=mapping["message"],
            provenance=SourceFactIdentity.from_mapping(provenance_payload),
            retained=mapping["retained"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise ContractExtractorError("contract-ambiguity content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class SourceComparison:
    """Pairwise comparison that retains both values and precedence ranks."""

    subject: str
    dimension: ContractDimension
    left_source: ContractEvidenceSource
    right_source: ContractEvidenceSource
    left_value: str
    right_value: str
    left_precedence: int
    right_precedence: int
    disposition: ComparisonDisposition
    retained_conflict: bool = False
    schema: str = SOURCE_COMPARISON_SCHEMA
    version: int = SOURCE_COMPARISON_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ContractExtractorError)
        if schema != SOURCE_COMPARISON_SCHEMA:
            raise ContractExtractorError("unexpected contract-source-comparison schema")
        version = _require_int(self.version, "version", error_type=ContractExtractorError)
        if version != SOURCE_COMPARISON_VERSION:
            raise ContractExtractorError("unexpected contract-source-comparison version")
        subject = _require_text(self.subject, "subject", error_type=ContractExtractorError)
        dimension = _closed_enum(
            self.dimension,
            ContractDimension,
            "contract dimension",
            error_type=ContractExtractorError,
        )
        left_source = _closed_enum(
            self.left_source,
            ContractEvidenceSource,
            "evidence source",
            error_type=ContractExtractorError,
        )
        right_source = _closed_enum(
            self.right_source,
            ContractEvidenceSource,
            "evidence source",
            error_type=ContractExtractorError,
        )
        left_value = _require_text(self.left_value, "left_value", error_type=ContractExtractorError)
        right_value = _require_text(self.right_value, "right_value", error_type=ContractExtractorError)
        left_precedence = _require_int(
            self.left_precedence, "left_precedence", error_type=ContractExtractorError
        )
        right_precedence = _require_int(
            self.right_precedence, "right_precedence", error_type=ContractExtractorError
        )
        if left_precedence != SOURCE_PRECEDENCE_RANK[left_source]:
            raise ContractExtractorError("left_precedence must match retained source rank")
        if right_precedence != SOURCE_PRECEDENCE_RANK[right_source]:
            raise ContractExtractorError("right_precedence must match retained source rank")
        disposition = _closed_enum(
            self.disposition,
            ComparisonDisposition,
            "comparison disposition",
            error_type=ContractExtractorError,
        )
        retained_conflict = _require_bool(self.retained_conflict, "retained_conflict")
        expected_conflict = disposition is ComparisonDisposition.CONFLICT
        if retained_conflict is not expected_conflict:
            raise ContractExtractorError("retained_conflict must match comparison disposition")
        ordered = sorted(
            (
                (left_source.value, left_value, left_precedence),
                (right_source.value, right_value, right_precedence),
            )
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(
            self,
            "left_source",
            _closed_enum(
                ordered[0][0],
                ContractEvidenceSource,
                "evidence source",
                error_type=ContractExtractorError,
            ),
        )
        object.__setattr__(
            self,
            "right_source",
            _closed_enum(
                ordered[1][0],
                ContractEvidenceSource,
                "evidence source",
                error_type=ContractExtractorError,
            ),
        )
        object.__setattr__(self, "left_value", ordered[0][1])
        object.__setattr__(self, "right_value", ordered[1][1])
        object.__setattr__(self, "left_precedence", ordered[0][2])
        object.__setattr__(self, "right_precedence", ordered[1][2])
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "retained_conflict", expected_conflict)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ContractExtractorError,
                )
            )
            if claimed != identity:
                raise ContractExtractorError("source-comparison content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "disposition": self.disposition.value,
            "left_precedence": self.left_precedence,
            "left_source": self.left_source.value,
            "left_value": self.left_value,
            "retained_conflict": self.retained_conflict,
            "right_precedence": self.right_precedence,
            "right_source": self.right_source.value,
            "right_value": self.right_value,
            "schema": self.schema,
            "subject": self.subject,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ContractExtractorError("source-comparison content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceComparison":
        mapping = _require_mapping(payload, error_type=ContractExtractorError)
        _require_fields(mapping, _COMPARISON_FIELDS)
        record = cls(
            subject=mapping["subject"],
            dimension=mapping["dimension"],
            left_source=mapping["left_source"],
            right_source=mapping["right_source"],
            left_value=mapping["left_value"],
            right_value=mapping["right_value"],
            left_precedence=mapping["left_precedence"],
            right_precedence=mapping["right_precedence"],
            disposition=mapping["disposition"],
            retained_conflict=mapping["retained_conflict"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise ContractExtractorError("source-comparison content identity mismatch")
        return record

    from_dict = from_mapping


def _require_dimension_closure(
    records: Sequence[ContractDimensionRecord],
) -> tuple[ContractDimensionRecord, ...]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise ContractExtractorError("dimensions must be a list of objects")
    parsed = tuple(
        item if isinstance(item, ContractDimensionRecord) else ContractDimensionRecord.from_mapping(item)
        for item in records
    )
    by_kind: dict[ContractDimension, ContractDimensionRecord] = {}
    for item in parsed:
        if item.dimension in by_kind:
            raise ContractExtractorError("candidate must emit every contract dimension")
        by_kind[item.dimension] = item
    if set(by_kind) != set(REQUIRED_CONTRACT_DIMENSIONS):
        raise ContractExtractorError("candidate must emit every contract dimension")
    return tuple(by_kind[dimension] for dimension in REQUIRED_CONTRACT_DIMENSIONS)


@dataclass(frozen=True)
class ContractCandidate:
    """Candidate-tier contract for one subject, with conflicts retained."""

    subject: str
    repository_tree: str
    freshness: str
    dimensions: tuple[ContractDimensionRecord, ...]
    evidence: tuple[ContractEvidenceUnit, ...] = ()
    ambiguities: tuple[ContractAmbiguity, ...] = ()
    comparisons: tuple[SourceComparison, ...] = ()
    tier: ContractTier = ContractTier.CANDIDATE
    can_promote_requirement: bool = EXTRACTOR_CAN_PROMOTE_REQUIREMENT
    schema: str = CONTRACT_CANDIDATE_SCHEMA
    version: int = CONTRACT_CANDIDATE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ContractExtractorError)
        if schema != CONTRACT_CANDIDATE_SCHEMA:
            raise ContractExtractorError("unexpected contract-candidate schema")
        version = _require_int(self.version, "version", error_type=ContractExtractorError)
        if version != CONTRACT_CANDIDATE_VERSION:
            raise ContractExtractorError("unexpected contract-candidate version")
        subject = _require_text(self.subject, "subject", error_type=ContractExtractorError)
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=ContractExtractorError
        )
        freshness = _require_text(self.freshness, "freshness", error_type=ContractExtractorError)
        tier = _closed_enum(self.tier, ContractTier, "contract tier", error_type=ContractExtractorError)
        if tier is not ContractTier.CANDIDATE:
            raise ContractExtractorAuthorityError("mined contracts remain candidate-tier")
        if self.can_promote_requirement is not False:
            raise ContractExtractorAuthorityError(
                "contract extractor cannot promote a requirement"
            )
        dimensions = _require_dimension_closure(self.dimensions)
        evidence = _record_tuple(self.evidence, "evidence", ContractEvidenceUnit)
        ambiguities = _record_tuple(self.ambiguities, "ambiguities", ContractAmbiguity)
        comparisons = _record_tuple(self.comparisons, "comparisons", SourceComparison)
        for unit in evidence:
            if unit.subject != subject:
                raise ContractExtractorError("evidence subject must match candidate")
            if unit.provenance.repository_tree != repository_tree:
                raise ContractExtractorError("evidence repository_tree must match candidate")
        for item in ambiguities:
            if item.subject != subject:
                raise ContractExtractorError("ambiguity subject must match candidate")
            if item.retained is not True:
                raise ContractExtractorError(_HIDDEN_CONFLICT_MESSAGE)
        for item in comparisons:
            if item.subject != subject:
                raise ContractExtractorError("comparison subject must match candidate")
        by_dimension: dict[ContractDimension, list[ContractEvidenceUnit]] = {
            dimension: [] for dimension in REQUIRED_CONTRACT_DIMENSIONS
        }
        for unit in evidence:
            by_dimension[unit.dimension].append(unit)
        records_by_kind = {item.dimension: item for item in dimensions}
        for dimension in REQUIRED_CONTRACT_DIMENSIONS:
            status, values, source_kinds, observation_only = _dimension_state(
                by_dimension[dimension]
            )
            record = records_by_kind[dimension]
            if (
                record.status is not status
                or record.values != values
                or record.source_kinds != source_kinds
                or record.observation_only is not observation_only
            ):
                raise ContractExtractorError(
                    "dimension record must match retained evidence without resolution"
                )
            if record.status is DimensionStatus.CONFLICTED and not any(
                item.dimension is dimension for item in ambiguities
            ):
                raise ContractExtractorError(_HIDDEN_CONFLICT_MESSAGE)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "ambiguities", ambiguities)
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(self, "tier", ContractTier.CANDIDATE)
        object.__setattr__(self, "can_promote_requirement", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ContractExtractorError,
                )
            )
            if claimed != identity:
                raise ContractExtractorError("contract-candidate content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def dimension(self, kind: ContractDimension | str) -> ContractDimensionRecord:
        parsed = _closed_enum(
            kind,
            ContractDimension,
            "contract dimension",
            error_type=ContractExtractorError,
        )
        for record in self.dimensions:
            if record.dimension is parsed:
                return record
        raise ContractExtractorError("candidate is missing a required dimension")

    def ambiguities_of(self, kind: AmbiguityKind | str) -> tuple[ContractAmbiguity, ...]:
        parsed = _closed_enum(
            kind, AmbiguityKind, "ambiguity kind", error_type=ContractExtractorError
        )
        return tuple(item for item in self.ambiguities if item.kind is parsed)

    def promote_to_requirement(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_requirement_promotion("requirement")

    def resolve_by_implementation(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_implementation_resolution("implementation-wins resolution")

    def hide_conflicts(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_hidden_conflict("hidden conflict")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "ambiguities": [item.to_dict() for item in self.ambiguities],
            "can_promote_requirement": False,
            "comparisons": [item.to_dict() for item in self.comparisons],
            "dimensions": [item.to_dict() for item in self.dimensions],
            "evidence": [item.to_dict() for item in self.evidence],
            "freshness": self.freshness,
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "subject": self.subject,
            "tier": ContractTier.CANDIDATE.value,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ContractExtractorError("contract-candidate content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContractCandidate":
        mapping = _require_mapping(payload, error_type=ContractExtractorError)
        _require_fields(mapping, _CANDIDATE_FIELDS)
        record = cls(
            subject=mapping["subject"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            dimensions=mapping["dimensions"],
            evidence=mapping["evidence"],
            ambiguities=mapping["ambiguities"],
            comparisons=mapping["comparisons"],
            tier=mapping["tier"],
            can_promote_requirement=mapping["can_promote_requirement"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise ContractExtractorError("contract-candidate content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "ContractCandidate":
        if type(payload) is not str or not payload:
            raise ContractExtractorError("contract-candidate JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractExtractorError("contract-candidate JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise ContractExtractorError("contract-candidate JSON must contain an object")
        return cls.from_mapping(decoded)


@dataclass(frozen=True)
class ContractExtractionResult:
    """Closed extraction report of candidates, comparisons, and ambiguities."""

    repository_tree: str
    freshness: str
    candidates: tuple[ContractCandidate, ...] = ()
    ambiguities: tuple[ContractAmbiguity, ...] = ()
    comparisons: tuple[SourceComparison, ...] = ()
    architecture_ir_identity: str = ""
    schema: str = CONTRACT_EXTRACTION_SCHEMA
    version: int = CONTRACT_EXTRACTION_VERSION
    effect_class: str = EFFECT_CLASS
    candidate_tier: bool = True
    can_promote_requirement: bool = EXTRACTOR_CAN_PROMOTE_REQUIREMENT
    can_resolve_implementation_wins: bool = EXTRACTOR_CAN_RESOLVE_IMPLEMENTATION_WINS
    can_hide_conflicts: bool = EXTRACTOR_CAN_HIDE_CONFLICTS
    can_treat_tests_as_complete: bool = EXTRACTOR_CAN_TREAT_TESTS_AS_COMPLETE
    can_treat_unmarked_docs_as_authority: bool = (
        EXTRACTOR_CAN_TREAT_UNMARKED_DOCS_AS_AUTHORITY
    )
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ContractExtractorError)
        if schema != CONTRACT_EXTRACTION_SCHEMA:
            raise ContractExtractorError("unexpected contract-extraction schema")
        version = _require_int(self.version, "version", error_type=ContractExtractorError)
        if version != CONTRACT_EXTRACTION_VERSION:
            raise ContractExtractorError("unexpected contract-extraction version")
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=ContractExtractorError
        )
        if effect_class != EFFECT_CLASS:
            raise ContractExtractorError("unexpected contract-extraction effect class")
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=ContractExtractorError
        )
        freshness = _require_text(self.freshness, "freshness", error_type=ContractExtractorError)
        if self.candidate_tier is not True:
            raise ContractExtractorAuthorityError("mined contracts remain candidate-tier")
        if self.can_promote_requirement is not False:
            raise ContractExtractorAuthorityError(
                "contract extractor cannot promote a requirement"
            )
        if self.can_resolve_implementation_wins is not False:
            raise ContractExtractorAuthorityError(
                "contract extractor cannot apply implementation-wins resolution"
            )
        if self.can_hide_conflicts is not False:
            raise ContractExtractorError(_HIDDEN_CONFLICT_MESSAGE)
        if self.can_treat_tests_as_complete is not False:
            raise ContractExtractorError("test completeness assumption is prohibited")
        if self.can_treat_unmarked_docs_as_authority is not False:
            raise ContractExtractorError("unmarked documentation is not authority")
        candidates = _record_tuple(self.candidates, "candidates", ContractCandidate)
        ambiguities = _record_tuple(self.ambiguities, "ambiguities", ContractAmbiguity)
        comparisons = _record_tuple(self.comparisons, "comparisons", SourceComparison)
        architecture_ir_identity = self.architecture_ir_identity
        if architecture_ir_identity:
            architecture_ir_identity = _validate_dag_json_cid(
                _require_text(
                    architecture_ir_identity,
                    "architecture_ir_identity",
                    error_type=ContractExtractorError,
                )
            )
        else:
            architecture_ir_identity = ""
        expected_ambiguities = _record_tuple(
            tuple(item for candidate in candidates for item in candidate.ambiguities),
            "ambiguities",
            ContractAmbiguity,
        )
        expected_comparisons = _record_tuple(
            tuple(item for candidate in candidates for item in candidate.comparisons),
            "comparisons",
            SourceComparison,
        )
        if ambiguities != expected_ambiguities:
            raise ContractExtractorError("result ambiguities must retain candidate conflicts")
        if comparisons != expected_comparisons:
            raise ContractExtractorError("result comparisons must retain candidate source pairs")
        for candidate in candidates:
            if candidate.repository_tree != repository_tree:
                raise ContractExtractorError("candidate repository_tree must match result")
            if candidate.freshness != freshness:
                raise ContractExtractorError("candidate freshness must match result")
            if candidate.tier is not ContractTier.CANDIDATE:
                raise ContractExtractorAuthorityError("mined contracts remain candidate-tier")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "effect_class", EFFECT_CLASS)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "ambiguities", ambiguities)
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "candidate_tier", True)
        object.__setattr__(self, "can_promote_requirement", False)
        object.__setattr__(self, "can_resolve_implementation_wins", False)
        object.__setattr__(self, "can_hide_conflicts", False)
        object.__setattr__(self, "can_treat_tests_as_complete", False)
        object.__setattr__(self, "can_treat_unmarked_docs_as_authority", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ContractExtractorError,
                )
            )
            if claimed != identity:
                raise ContractExtractorError(
                    "contract-extraction content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def candidate(self, subject: str) -> ContractCandidate:
        name = _require_text(subject, "subject", error_type=ContractExtractorError)
        for item in self.candidates:
            if item.subject == name:
                return item
        raise ContractExtractorError(f"no contract candidate for subject {name!r}")

    def ambiguities_of(self, kind: AmbiguityKind | str) -> tuple[ContractAmbiguity, ...]:
        parsed = _closed_enum(
            kind, AmbiguityKind, "ambiguity kind", error_type=ContractExtractorError
        )
        return tuple(item for item in self.ambiguities if item.kind is parsed)

    def promote_to_requirement(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_requirement_promotion("requirement")

    def resolve_by_implementation(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_implementation_resolution("implementation-wins resolution")

    def hide_conflicts(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_hidden_conflict("hidden conflict")

    def treat_tests_as_complete(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_test_completeness("test completeness")

    def treat_unmarked_docs_as_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_unmarked_documentation_authority("unmarked documentation")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "ambiguities": [item.to_dict() for item in self.ambiguities],
            "architecture_ir_identity": self.architecture_ir_identity,
            "can_hide_conflicts": False,
            "can_promote_requirement": False,
            "can_resolve_implementation_wins": False,
            "can_treat_tests_as_complete": False,
            "can_treat_unmarked_docs_as_authority": False,
            "candidate_tier": True,
            "candidates": [item.to_dict() for item in self.candidates],
            "comparisons": [item.to_dict() for item in self.comparisons],
            "effect_class": EFFECT_CLASS,
            "freshness": self.freshness,
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ContractExtractorError("contract-extraction content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContractExtractionResult":
        mapping = _require_mapping(payload, error_type=ContractExtractorError)
        _require_fields(mapping, _RESULT_FIELDS)
        record = cls(
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            candidates=mapping["candidates"],
            ambiguities=mapping["ambiguities"],
            comparisons=mapping["comparisons"],
            architecture_ir_identity=mapping["architecture_ir_identity"],
            schema=mapping["schema"],
            version=mapping["version"],
            effect_class=mapping["effect_class"],
            candidate_tier=mapping["candidate_tier"],
            can_promote_requirement=mapping["can_promote_requirement"],
            can_resolve_implementation_wins=mapping["can_resolve_implementation_wins"],
            can_hide_conflicts=mapping["can_hide_conflicts"],
            can_treat_tests_as_complete=mapping["can_treat_tests_as_complete"],
            can_treat_unmarked_docs_as_authority=mapping[
                "can_treat_unmarked_docs_as_authority"
            ],
        )
        if mapping["content_identity"] != record.content_identity:
            raise ContractExtractorError("contract-extraction content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "ContractExtractionResult":
        if type(payload) is not str or not payload:
            raise ContractExtractorError(
                "contract-extraction JSON must be a nonempty string"
            )
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractExtractorError("contract-extraction JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise ContractExtractorError("contract-extraction JSON must contain an object")
        return cls.from_mapping(decoded)


def _normalize_units(
    units: Sequence[ContractEvidenceUnit | Mapping[str, Any]],
) -> tuple[ContractEvidenceUnit, ...]:
    if isinstance(units, (str, bytes, bytearray)) or not isinstance(units, Sequence):
        raise ContractExtractorError("evidence units must be a list of objects")
    records = tuple(
        item if isinstance(item, ContractEvidenceUnit) else ContractEvidenceUnit.from_mapping(item)
        for item in units
    )
    unique: dict[str, ContractEvidenceUnit] = {}
    for item in records:
        unique[item.content_identity] = item
    return tuple(sorted(unique.values(), key=lambda item: item.content_identity))


def compare_contract_evidence(
    units: Sequence[ContractEvidenceUnit | Mapping[str, Any]],
) -> tuple[SourceComparison, ...]:
    """Compare every same-subject, same-dimension source pair without resolving."""

    records = _normalize_units(units)
    comparisons: list[SourceComparison] = []
    seen: set[str] = set()
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left.subject != right.subject or left.dimension is not right.dimension:
                continue
            comparison = SourceComparison(
                subject=left.subject,
                dimension=left.dimension,
                left_source=left.source_kind,
                right_source=right.source_kind,
                left_value=left.value,
                right_value=right.value,
                left_precedence=SOURCE_PRECEDENCE_RANK[left.source_kind],
                right_precedence=SOURCE_PRECEDENCE_RANK[right.source_kind],
                disposition=_comparison_disposition(left, right),
                retained_conflict=left.value != right.value,
            )
            if comparison.content_identity in seen:
                continue
            seen.add(comparison.content_identity)
            comparisons.append(comparison)
    return tuple(sorted(comparisons, key=lambda item: item.content_identity))


evidence_comparison = compare_contract_evidence


def _ambiguity(
    *,
    subject: str,
    dimension: ContractDimension,
    kind: AmbiguityKind,
    values: Iterable[str],
    source_kinds: Iterable[ContractEvidenceSource],
    message: str,
    provenance: SourceFactIdentity,
) -> ContractAmbiguity:
    return ContractAmbiguity(
        subject=subject,
        dimension=dimension,
        kind=kind,
        values=tuple(values),
        source_kinds=tuple(source_kinds),
        message=message,
        provenance=provenance,
        retained=True,
    )


def _ambiguities_for(
    subject: str,
    dimension: ContractDimension,
    units: Sequence[ContractEvidenceUnit],
) -> tuple[ContractAmbiguity, ...]:
    if not units:
        return ()
    provenance = min(units, key=lambda item: item.content_identity).provenance
    kinds = tuple(unit.source_kind for unit in units)
    contributing = tuple(unit for unit in units if _unit_contributes(unit))
    positive = {unit.value for unit in contributing if unit.polarity is EvidencePolarity.POSITIVE}
    negative = {unit.value for unit in contributing if unit.polarity is EvidencePolarity.NEGATIVE}
    mutant = {unit.value for unit in contributing if unit.polarity is EvidencePolarity.MUTANT}
    impl_positive = {
        unit.value
        for unit in contributing
        if unit.source_kind is ContractEvidenceSource.IMPLEMENTATION
        and unit.polarity is EvidencePolarity.POSITIVE
    }
    other_positive = {
        unit.value
        for unit in contributing
        if unit.source_kind is not ContractEvidenceSource.IMPLEMENTATION
        and unit.polarity is EvidencePolarity.POSITIVE
    }
    found: dict[str, ContractAmbiguity] = {}

    def add(item: ContractAmbiguity) -> None:
        found[item.content_identity] = item

    for unit in units:
        if (
            unit.source_kind is ContractEvidenceSource.AUTHORITATIVE_DOCUMENT
            and not unit.marked_authoritative
        ):
            add(
                _ambiguity(
                    subject=subject,
                    dimension=dimension,
                    kind=AmbiguityKind.UNMARKED_DOCUMENTATION,
                    values=(unit.value,),
                    source_kinds=(unit.source_kind,),
                    message="unmarked documentation is not authority",
                    provenance=unit.provenance,
                )
            )
        if unit.assumes_test_completeness:
            add(
                _ambiguity(
                    subject=subject,
                    dimension=dimension,
                    kind=AmbiguityKind.TEST_COMPLETENESS_ASSUMED,
                    values=(unit.value,),
                    source_kinds=(unit.source_kind,),
                    message="test completeness assumption is prohibited",
                    provenance=unit.provenance,
                )
            )
    if len(positive) > 1:
        add(
            _ambiguity(
                subject=subject,
                dimension=dimension,
                kind=AmbiguityKind.CONFLICTING_VALUES,
                values=sorted(positive),
                source_kinds=kinds,
                message="conflicting positive evidence is retained without resolution",
                provenance=provenance,
            )
        )
        ranks = {SOURCE_PRECEDENCE_RANK[unit.source_kind] for unit in contributing if unit.polarity is EvidencePolarity.POSITIVE}
        if len(ranks) > 1:
            add(
                _ambiguity(
                    subject=subject,
                    dimension=dimension,
                    kind=AmbiguityKind.SOURCE_PRECEDENCE_UNRESOLVED,
                    values=sorted(positive),
                    source_kinds=kinds,
                    message="source precedence is retained and does not select a winner",
                    provenance=provenance,
                )
            )
    if impl_positive and other_positive and impl_positive != other_positive:
        add(
            _ambiguity(
                subject=subject,
                dimension=dimension,
                kind=AmbiguityKind.IMPLEMENTATION_WINS_REJECTED,
                values=sorted(impl_positive | other_positive),
                source_kinds=kinds,
                message="implementation-wins resolution is prohibited",
                provenance=provenance,
            )
        )
    if positive & negative:
        add(
            _ambiguity(
                subject=subject,
                dimension=dimension,
                kind=AmbiguityKind.NEGATIVE_CONTRADICTS_POSITIVE,
                values=sorted(positive | negative),
                source_kinds=kinds,
                message="negative evidence contradicts a positive claim",
                provenance=provenance,
            )
        )
    if positive & mutant:
        add(
            _ambiguity(
                subject=subject,
                dimension=dimension,
                kind=AmbiguityKind.MUTANT_SURVIVES,
                values=sorted(positive | mutant),
                source_kinds=kinds,
                message="mutant evidence still matches the claimed contract",
                provenance=provenance,
            )
        )
    if mutant - positive and positive:
        add(
            _ambiguity(
                subject=subject,
                dimension=dimension,
                kind=AmbiguityKind.NEGATIVE_CONTRADICTS_POSITIVE,
                values=sorted(positive | mutant),
                source_kinds=kinds,
                message="mutant evidence observes a value the positive claim excludes",
                provenance=provenance,
            )
        )
    return tuple(sorted(found.values(), key=lambda item: item.content_identity))


def _candidate_for(
    subject: str,
    units: Sequence[ContractEvidenceUnit],
    *,
    repository_tree: str,
    freshness: str,
) -> ContractCandidate:
    by_dimension: dict[ContractDimension, list[ContractEvidenceUnit]] = {
        dimension: [] for dimension in REQUIRED_CONTRACT_DIMENSIONS
    }
    for unit in units:
        by_dimension[unit.dimension].append(unit)
    dimensions: list[ContractDimensionRecord] = []
    ambiguities: list[ContractAmbiguity] = []
    for dimension in REQUIRED_CONTRACT_DIMENSIONS:
        dim_units = tuple(by_dimension[dimension])
        status, values, source_kinds, observation_only = _dimension_state(dim_units)
        dimensions.append(
            ContractDimensionRecord(
                dimension=dimension,
                status=status,
                values=values,
                source_kinds=source_kinds,
                observation_only=observation_only,
            )
        )
        ambiguities.extend(_ambiguities_for(subject, dimension, dim_units))
    return ContractCandidate(
        subject=subject,
        repository_tree=repository_tree,
        freshness=freshness,
        dimensions=tuple(dimensions),
        evidence=tuple(units),
        ambiguities=tuple(ambiguities),
        comparisons=compare_contract_evidence(units),
        tier=ContractTier.CANDIDATE,
        can_promote_requirement=False,
    )


def extract_contract_candidates(
    units: Sequence[ContractEvidenceUnit | Mapping[str, Any]] = (),
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    architecture: ArchitectureIR | Mapping[str, Any] | None = None,
    public_contracts: Sequence[ContractEvidenceUnit | Mapping[str, Any]] = (),
) -> ContractExtractionResult:
    """Mine candidate contracts from declared evidence without promoting them."""

    tree = _require_text(repository_tree, "repository_tree", error_type=ContractExtractorError)
    token = _require_text(freshness, "freshness", error_type=ContractExtractorError)
    graph = _require_architecture_ir(architecture)
    if graph is not None and graph.repository_tree != tree:
        raise ContractExtractorError("architecture repository_tree must match extraction")
    merged = _normalize_units((*units, *public_contracts))
    if graph is not None:
        merged = _normalize_units((*merged, *mine_graph_evidence(graph)))
    for unit in merged:
        if unit.provenance.repository_tree != tree:
            raise ContractExtractorError("evidence repository_tree must match extraction")
    grouped: dict[str, list[ContractEvidenceUnit]] = {}
    for unit in merged:
        grouped.setdefault(unit.subject, []).append(unit)
    candidates = tuple(
        _candidate_for(subject, grouped[subject], repository_tree=tree, freshness=token)
        for subject in sorted(grouped)
    )
    return ContractExtractionResult(
        repository_tree=tree,
        freshness=token,
        candidates=candidates,
        ambiguities=tuple(item for candidate in candidates for item in candidate.ambiguities),
        comparisons=tuple(item for candidate in candidates for item in candidate.comparisons),
        architecture_ir_identity="" if graph is None else graph.content_identity,
        candidate_tier=True,
        can_promote_requirement=False,
        can_resolve_implementation_wins=False,
        can_hide_conflicts=False,
        can_treat_tests_as_complete=False,
        can_treat_unmarked_docs_as_authority=False,
    )


def build_contract_extraction_result(
    units: Sequence[ContractEvidenceUnit | Mapping[str, Any]] = (),
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    architecture: ArchitectureIR | Mapping[str, Any] | None = None,
    public_contracts: Sequence[ContractEvidenceUnit | Mapping[str, Any]] = (),
) -> ContractExtractionResult:
    """Alias for :func:`extract_contract_candidates`."""

    return extract_contract_candidates(
        units,
        repository_tree=repository_tree,
        freshness=freshness,
        architecture=architecture,
        public_contracts=public_contracts,
    )


@dataclass(frozen=True)
class ContractCandidateExtractor:
    """Read-only miner for candidate-tier contracts and retained ambiguities."""

    extractor_identity: str = EXTRACTOR_IDENTITY
    can_promote_requirement: bool = EXTRACTOR_CAN_PROMOTE_REQUIREMENT
    can_resolve_implementation_wins: bool = EXTRACTOR_CAN_RESOLVE_IMPLEMENTATION_WINS
    can_hide_conflicts: bool = EXTRACTOR_CAN_HIDE_CONFLICTS
    can_treat_tests_as_complete: bool = EXTRACTOR_CAN_TREAT_TESTS_AS_COMPLETE
    can_treat_unmarked_docs_as_authority: bool = (
        EXTRACTOR_CAN_TREAT_UNMARKED_DOCS_AS_AUTHORITY
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(
                self.extractor_identity,
                "extractor_identity",
                error_type=ContractExtractorError,
            ),
        )
        if self.can_promote_requirement is not False:
            raise ContractExtractorAuthorityError(
                "contract extractor cannot promote a requirement"
            )
        if self.can_resolve_implementation_wins is not False:
            raise ContractExtractorAuthorityError(
                "contract extractor cannot apply implementation-wins resolution"
            )
        if self.can_hide_conflicts is not False:
            raise ContractExtractorError(_HIDDEN_CONFLICT_MESSAGE)
        if self.can_treat_tests_as_complete is not False:
            raise ContractExtractorError("test completeness assumption is prohibited")
        if self.can_treat_unmarked_docs_as_authority is not False:
            raise ContractExtractorError("unmarked documentation is not authority")
        object.__setattr__(self, "can_promote_requirement", False)
        object.__setattr__(self, "can_resolve_implementation_wins", False)
        object.__setattr__(self, "can_hide_conflicts", False)
        object.__setattr__(self, "can_treat_tests_as_complete", False)
        object.__setattr__(self, "can_treat_unmarked_docs_as_authority", False)

    def extract(
        self,
        units: Sequence[ContractEvidenceUnit | Mapping[str, Any]] = (),
        *,
        repository_tree: str,
        freshness: str = DEFAULT_FRESHNESS,
        architecture: ArchitectureIR | Mapping[str, Any] | None = None,
        public_contracts: Sequence[ContractEvidenceUnit | Mapping[str, Any]] = (),
    ) -> ContractExtractionResult:
        return extract_contract_candidates(
            units,
            repository_tree=repository_tree,
            freshness=freshness,
            architecture=architecture,
            public_contracts=public_contracts,
        )

    def compare(
        self,
        units: Sequence[ContractEvidenceUnit | Mapping[str, Any]],
    ) -> tuple[SourceComparison, ...]:
        return compare_contract_evidence(units)

    def promote_to_requirement(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_requirement_promotion("requirement")

    def resolve_by_implementation(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_implementation_resolution("implementation-wins resolution")

    def hide_conflicts(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_hidden_conflict("hidden conflict")

    def treat_tests_as_complete(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_test_completeness("test completeness")

    def treat_unmarked_docs_as_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_unmarked_documentation_authority("unmarked documentation")


def refuse_requirement_promotion(action: str) -> None:
    """Reject attempts to promote a candidate contract to a requirement."""

    name = _require_text(action, "action", error_type=ContractExtractorError)
    raise ContractExtractorAuthorityError(
        f"contract extractor cannot promote a {name}"
    )


def refuse_implementation_resolution(action: str) -> None:
    """Reject attempts to let implementation evidence win a conflict."""

    name = _require_text(action, "action", error_type=ContractExtractorError)
    raise ContractExtractorAuthorityError(
        f"contract extractor cannot apply {name}"
    )


def refuse_hidden_conflict(action: str) -> None:
    """Reject attempts to drop or silence a typed conflict."""

    name = _require_text(action, "action", error_type=ContractExtractorError)
    raise ContractExtractorError(f"{name} is prohibited")


def refuse_test_completeness(action: str) -> None:
    """Reject attempts to treat tests as a complete contract."""

    name = _require_text(action, "action", error_type=ContractExtractorError)
    raise ContractExtractorError(f"{name} assumption is prohibited")


def refuse_unmarked_documentation_authority(action: str) -> None:
    """Reject attempts to treat unmarked documents as contract authority."""

    name = _require_text(action, "action", error_type=ContractExtractorError)
    raise ContractExtractorError(f"{name} is not authority")
