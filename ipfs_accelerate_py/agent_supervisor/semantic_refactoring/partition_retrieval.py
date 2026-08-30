"""SPAR-015 analogous-refactor retrieval and advisory partition ranking.

This module extends current supervisor partition orchestration with
``AnalogousRefactorRetriever@1``.  It retrieves exact, lexical, graph, and
vector prior refactors, then optionally reorders already-admitted SPAR-013
candidates.  Vector and model channels remain advisory.  Retrieval cannot
admit a candidate, hide a hard-constraint violation, or mint authority.

It reuses SPAR-006 projection search only as an injected adapter.  The
adapter never replaces exact identity, never suppresses raw-source fallback,
and cannot authorize a transition or completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Final, Mapping, Sequence
import math
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import (
    cid_for_dag_json,
    validate_cid,
)

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as GENERATOR_IDENTITY_EXCLUDED_FIELDS,
    PartitionGenerationReceipt,
    ProgramPartitionCandidate,
)


TASK_ID: Final[str] = "SPAR-015"
GOAL_ID: Final[str] = "SPAR-G032"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_retrieval@1"
)

ANALOGOUS_REFACTOR_RECORD_INTERFACE: Final[str] = "AnalogousRefactorRecord@1"
ANALOGOUS_REFACTOR_HIT_INTERFACE: Final[str] = "AnalogousRefactorHit@1"
ANALOGOUS_REFACTOR_QUERY_INTERFACE: Final[str] = "AnalogousRefactorQuery@1"
ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE: Final[str] = (
    "AnalogousRefactorRetrievalReceipt@1"
)
PARTITION_RANKING_RECEIPT_INTERFACE: Final[str] = "PartitionRankingReceipt@1"
ANALOGOUS_REFACTOR_RETRIEVER_INTERFACE: Final[str] = "AnalogousRefactorRetriever@1"
ADVISORY_PARTITION_RANKER_INTERFACE: Final[str] = "AdvisoryPartitionRanker@1"

ANALOGOUS_REFACTOR_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analogous-refactor-record@1"
)
ANALOGOUS_REFACTOR_HIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analogous-refactor-hit@1"
)
ANALOGOUS_REFACTOR_QUERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analogous-refactor-query@1"
)
ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analogous-refactor-retrieval-receipt@1"
)
PARTITION_RANKING_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-ranking-receipt@1"
)

PARTITION_RETRIEVAL_CONTRACT_VERSION: Final[str] = "1"

RETRIEVAL_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
RETRIEVAL_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
RETRIEVAL_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
RETRIEVAL_IS_NOMINATION_ONLY: Final[bool] = True
RANKER_IS_ADVISORY: Final[bool] = True
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_RECORDS: Final[int] = 16_384
MAX_HITS: Final[int] = 16_384
MAX_EDGES: Final[int] = 65_536
MAX_TOKENS: Final[int] = 16_384
MAX_VECTOR_DIM: Final[int] = 4_096
MAX_MILLIRANK: Final[int] = 1_000

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    GENERATOR_IDENTITY_EXCLUDED_FIELDS
)

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FunctionSemanticCapsule",
        "MethodSemanticCapsule",
        "ClassSemanticCapsule",
        "TopLevelBlockCapsule",
        "ModuleSemanticCapsule",
        "PackageSemanticCapsule",
        "CallsiteSemanticCapsule",
        "StateOwnerCapsule",
        "RegistrationCapsule",
        "ResourceLifecycleCapsule",
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "projection_is_authority",
    "vector_authoritative",
)

CHANNEL_ORDER: Final[tuple[str, ...]] = ("exact", "lexical", "graph", "vector")
ADVISORY_CHANNELS: Final[frozenset[str]] = frozenset({"vector"})
ACCEPTED_OUTCOME: Final[str] = "accepted"
REJECTED_OUTCOME: Final[str] = "rejected"
ADVISORY_OUTCOME: Final[str] = "advisory"
RECORD_OUTCOMES: Final[frozenset[str]] = frozenset(
    {ACCEPTED_OUTCOME, REJECTED_OUTCOME, ADVISORY_OUTCOME}
)


class PartitionRetrievalError(ValueError):
    """Fail-closed violation of a SPAR-015 partition-retrieval contract."""


class RetrievalChannel(str, Enum):
    EXACT = "exact"
    LEXICAL = "lexical"
    GRAPH = "graph"
    VECTOR = "vector"


class RecordOutcome(str, Enum):
    ACCEPTED = ACCEPTED_OUTCOME
    REJECTED = REJECTED_OUTCOME
    ADVISORY = ADVISORY_OUTCOME


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise PartitionRetrievalError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise PartitionRetrievalError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise PartitionRetrievalError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise PartitionRetrievalError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise PartitionRetrievalError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise PartitionRetrievalError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PartitionRetrievalError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise PartitionRetrievalError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise PartitionRetrievalError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise PartitionRetrievalError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise PartitionRetrievalError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise PartitionRetrievalError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise PartitionRetrievalError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise PartitionRetrievalError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise PartitionRetrievalError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise PartitionRetrievalError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise PartitionRetrievalError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise PartitionRetrievalError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise PartitionRetrievalError(f"unknown {name}: {text}") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise PartitionRetrievalError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PartitionRetrievalError("non-finite number is not DAG-JSON")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _project(to_dict())
    raise PartitionRetrievalError(
        f"unsupported projected type {type(value).__name__}"
    )


def _millirank(value: Any, name: str) -> int:
    if type(value) is bool or type(value) is not int:
        raise PartitionRetrievalError(f"{name} must be an integer millirank")
    if value < 0 or value > MAX_MILLIRANK:
        raise PartitionRetrievalError(f"{name} must be between 0 and {MAX_MILLIRANK}")
    return value


def _jaccard_millirank(left: Sequence[str], right: Sequence[str]) -> int:
    first = set(left)
    second = set(right)
    if not first or not second:
        return 0
    intersection = len(first & second)
    if intersection == 0:
        return 0
    union = len(first | second)
    return (intersection * MAX_MILLIRANK) // union


def _finite_vector(value: Any, name: str) -> tuple[float, ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise PartitionRetrievalError(f"{name} must be a list")
    if len(value) > MAX_VECTOR_DIM:
        raise PartitionRetrievalError(f"{name} exceeds vector dimension bound")
    vector: list[float] = []
    for item in value:
        if type(item) is bool or type(item) not in {int, float}:
            raise PartitionRetrievalError(f"{name} items must be finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise PartitionRetrievalError(f"{name} contains a non-finite component")
        vector.append(number)
    return tuple(vector)


def _cosine_millirank(left: Sequence[float], right: Sequence[float]) -> int:
    if len(left) != len(right):
        raise PartitionRetrievalError("vector dimension mismatch")
    if not left:
        return 0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for first, second in zip(left, right):
        dot += first * second
        left_norm += first * first
        right_norm += second * second
    if left_norm == 0.0 or right_norm == 0.0:
        raise PartitionRetrievalError("zero vector is not a valid projection")
    cosine = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
    if cosine <= 0.0:
        return 0
    scaled = int(math.floor(cosine * MAX_MILLIRANK + 1e-12))
    if scaled > MAX_MILLIRANK:
        return MAX_MILLIRANK
    return scaled


def _graph_fingerprints(
    member_ids: Sequence[str],
    edges: Sequence["AnalogousGraphEdge"],
) -> tuple[str, ...]:
    fingerprints = list(member_ids)
    fingerprints.extend(
        f"{item.source_id}\t{item.kind}\t{item.target_id}" for item in edges
    )
    return tuple(sorted(set(fingerprints)))


def partition_retrieval_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


@dataclass(frozen=True, slots=True)
class AnalogousGraphEdge:
    """One typed graph edge used for analogous-refactor retrieval."""

    source_id: str
    target_id: str
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AnalogousGraphEdge":
        payload = _mapping(data, "graph_edges")
        unknown = set(payload) - {"source_id", "target_id", "kind"}
        if unknown:
            raise PartitionRetrievalError(
                f"unknown graph edge field: {sorted(unknown)}"
            )
        return cls(
            source_id=payload["source_id"],
            target_id=payload["target_id"],
            kind=payload["kind"],
        )


def _coerce_edge(value: AnalogousGraphEdge | Mapping[str, Any]) -> AnalogousGraphEdge:
    if isinstance(value, AnalogousGraphEdge):
        return value
    if isinstance(value, Mapping):
        return AnalogousGraphEdge.from_mapping(value)
    raise PartitionRetrievalError("graph edge must be an object")


def _edges(value: Any, name: str) -> tuple[AnalogousGraphEdge, ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise PartitionRetrievalError(f"{name} must be a list")
    edges = tuple(sorted(
        (_coerce_edge(item) for item in value),
        key=lambda item: (item.source_id, item.kind, item.target_id),
    ))
    if len(edges) > MAX_EDGES:
        raise PartitionRetrievalError(f"{name} exceeds maximum length")
    seen = [item.to_dict() for item in edges]
    encoded = [tuple(sorted(item.items())) for item in seen]
    if len(encoded) != len(set(encoded)):
        raise PartitionRetrievalError(f"{name} must not contain duplicates")
    return edges


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise PartitionRetrievalError(f"{name} cannot claim {flag}")


@dataclass(frozen=True, slots=True)
class AnalogousRefactorQuery:
    """Sealed query slice for analogous-refactor retrieval."""

    tree_id: str
    member_ids: Sequence[str] = ()
    tokens: Sequence[str] = ()
    graph_edges: Sequence[AnalogousGraphEdge | Mapping[str, Any]] = ()
    vector: Sequence[float] = ()
    vector_available: bool = False

    interface: ClassVar[str] = ANALOGOUS_REFACTOR_QUERY_INTERFACE
    schema: ClassVar[str] = ANALOGOUS_REFACTOR_QUERY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "member_ids",
            "tokens",
            "graph_edges",
            "vector",
            "vector_available",
            "query_cid",
        }
    )

    def __post_init__(self) -> None:
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        tokens = _unique_sorted_text(list(self.tokens), "tokens", limit=MAX_TOKENS)
        edges = _edges(self.graph_edges, "graph_edges")
        available = _bool(self.vector_available, "vector_available")
        vector = _finite_vector(self.vector, "vector")
        if available and not vector:
            raise PartitionRetrievalError(
                "vector_available requires a nonempty finite vector"
            )
        if not available and vector:
            raise PartitionRetrievalError(
                "unavailable vector channel cannot carry vector components"
            )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "graph_edges", edges)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "vector_available", available)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ANALOGOUS_REFACTOR_QUERY_SCHEMA,
            "interface": ANALOGOUS_REFACTOR_QUERY_INTERFACE,
            "tree_id": self.tree_id,
            "member_ids": list(self.member_ids),
            "tokens": list(self.tokens),
            "graph_edges": [item.to_dict() for item in self.graph_edges],
            "vector": list(self.vector),
            "vector_available": self.vector_available,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def query_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["query_cid"] = self.query_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalogousRefactorQuery":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("query_cid")
        if payload.pop("schema") != ANALOGOUS_REFACTOR_QUERY_SCHEMA:
            raise PartitionRetrievalError("unsupported AnalogousRefactorQuery schema")
        if payload.pop("interface") != ANALOGOUS_REFACTOR_QUERY_INTERFACE:
            raise PartitionRetrievalError(
                "unsupported AnalogousRefactorQuery interface"
            )
        result = cls(**payload)
        _verify_cid(claimed, result.query_cid, "AnalogousRefactorQuery query_cid")
        return result

    def overlay_candidate(
        self, candidate: ProgramPartitionCandidate
    ) -> "AnalogousRefactorQuery":
        if candidate.tree_id != self.tree_id:
            raise PartitionRetrievalError("candidate tree_id does not match query")
        incident = tuple(
            item
            for item in self.graph_edges
            if item.source_id in candidate.member_ids
            or item.target_id in candidate.member_ids
        )
        tokens = tuple(
            sorted(set(self.tokens) | set(candidate.member_ids))
        )
        return AnalogousRefactorQuery(
            tree_id=self.tree_id,
            member_ids=candidate.member_ids,
            tokens=tokens,
            graph_edges=incident,
            vector=self.vector,
            vector_available=self.vector_available,
        )


def _coerce_query(
    value: AnalogousRefactorQuery | Mapping[str, Any],
) -> AnalogousRefactorQuery:
    if isinstance(value, AnalogousRefactorQuery):
        return value
    if isinstance(value, Mapping):
        if "query_cid" in value:
            return AnalogousRefactorQuery.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "query_cid"}
        }
        return AnalogousRefactorQuery(**payload)
    raise PartitionRetrievalError("query must be an AnalogousRefactorQuery")


@dataclass(frozen=True, slots=True)
class AnalogousRefactorRecord:
    """One sealed prior-refactor analog. Not admission or completion authority."""

    tree_id: str
    member_ids: Sequence[str]
    tokens: Sequence[str] = ()
    graph_edges: Sequence[AnalogousGraphEdge | Mapping[str, Any]] = ()
    vector: Sequence[float] = ()
    vector_available: bool = False
    evidence_class: str = "exact_static_fact"
    outcome: RecordOutcome | str = ACCEPTED_OUTCOME

    interface: ClassVar[str] = ANALOGOUS_REFACTOR_RECORD_INTERFACE
    schema: ClassVar[str] = ANALOGOUS_REFACTOR_RECORD_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "member_ids",
            "tokens",
            "graph_edges",
            "vector",
            "vector_available",
            "evidence_class",
            "outcome",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "vector_authoritative",
            "record_cid",
        }
    )

    def __post_init__(self) -> None:
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        if not members:
            raise PartitionRetrievalError("analogous refactor record requires members")
        tokens = _unique_sorted_text(list(self.tokens), "tokens", limit=MAX_TOKENS)
        edges = _edges(self.graph_edges, "graph_edges")
        available = _bool(self.vector_available, "vector_available")
        vector = _finite_vector(self.vector, "vector")
        if available and not vector:
            raise PartitionRetrievalError(
                "vector_available requires a nonempty finite vector"
            )
        if not available and vector:
            raise PartitionRetrievalError(
                "unavailable vector channel cannot carry vector components"
            )
        outcome = _enum(self.outcome, RecordOutcome, "outcome")
        evidence = _text(self.evidence_class, "evidence_class")
        if outcome == ACCEPTED_OUTCOME and evidence in _NON_ADMITTING_EVIDENCE:
            raise PartitionRetrievalError(
                "vector or model evidence cannot mark a prior refactor accepted"
            )
        if outcome == ADVISORY_OUTCOME and evidence not in _NON_ADMITTING_EVIDENCE:
            evidence = "vector_candidate"
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "graph_edges", edges)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "vector_available", available)
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "outcome", outcome)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def vector_authoritative(self) -> bool:
        return False

    @property
    def advisory(self) -> bool:
        return self.outcome == ADVISORY_OUTCOME or (
            self.evidence_class in _NON_ADMITTING_EVIDENCE
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ANALOGOUS_REFACTOR_RECORD_SCHEMA,
            "interface": ANALOGOUS_REFACTOR_RECORD_INTERFACE,
            "tree_id": self.tree_id,
            "member_ids": list(self.member_ids),
            "tokens": list(self.tokens),
            "graph_edges": [item.to_dict() for item in self.graph_edges],
            "vector": list(self.vector),
            "vector_available": self.vector_available,
            "evidence_class": self.evidence_class,
            "outcome": self.outcome,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "vector_authoritative": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def record_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["record_cid"] = self.record_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalogousRefactorRecord":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("record_cid")
        if payload.pop("schema") != ANALOGOUS_REFACTOR_RECORD_SCHEMA:
            raise PartitionRetrievalError("unsupported AnalogousRefactorRecord schema")
        if payload.pop("interface") != ANALOGOUS_REFACTOR_RECORD_INTERFACE:
            raise PartitionRetrievalError(
                "unsupported AnalogousRefactorRecord interface"
            )
        _pop_authority_flags(payload, "analogous refactor record")
        result = cls(**payload)
        _verify_cid(
            claimed, result.record_cid, "AnalogousRefactorRecord record_cid"
        )
        return result


def _coerce_record(
    value: AnalogousRefactorRecord | Mapping[str, Any],
) -> AnalogousRefactorRecord:
    if isinstance(value, AnalogousRefactorRecord):
        return value
    if isinstance(value, Mapping):
        if "record_cid" in value:
            return AnalogousRefactorRecord.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "record_cid",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return AnalogousRefactorRecord(**payload)
    raise PartitionRetrievalError("corpus item must be an AnalogousRefactorRecord")


@dataclass(frozen=True, slots=True)
class AnalogousRefactorHit:
    """One retrieval nomination. Never admits or hides a violation."""

    record_cid: str
    channel: RetrievalChannel | str
    millirank: int
    overlapping_member_ids: Sequence[str]
    evidence_class: str
    outcome: RecordOutcome | str
    advisory: bool

    interface: ClassVar[str] = ANALOGOUS_REFACTOR_HIT_INTERFACE
    schema: ClassVar[str] = ANALOGOUS_REFACTOR_HIT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "record_cid",
            "channel",
            "millirank",
            "overlapping_member_ids",
            "evidence_class",
            "outcome",
            "advisory",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "vector_authoritative",
            "hit_cid",
        }
    )

    def __post_init__(self) -> None:
        channel = _enum(self.channel, RetrievalChannel, "channel")
        millirank = _millirank(self.millirank, "millirank")
        if millirank <= 0:
            raise PartitionRetrievalError("retrieval hit requires a positive millirank")
        advisory = _bool(self.advisory, "advisory")
        outcome = _enum(self.outcome, RecordOutcome, "outcome")
        evidence = _text(self.evidence_class, "evidence_class")
        if channel in ADVISORY_CHANNELS:
            advisory = True
        if evidence in _NON_ADMITTING_EVIDENCE:
            advisory = True
        if outcome == ADVISORY_OUTCOME:
            advisory = True
        if not advisory and evidence in _NON_ADMITTING_EVIDENCE:
            raise PartitionRetrievalError(
                "vector or model evidence cannot emit a non-advisory hit"
            )
        object.__setattr__(self, "record_cid", _cid(self.record_cid, "record_cid"))
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "millirank", millirank)
        object.__setattr__(
            self,
            "overlapping_member_ids",
            _unique_sorted_text(
                list(self.overlapping_member_ids),
                "overlapping_member_ids",
                limit=MAX_MEMBERS,
            ),
        )
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "advisory", advisory)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def vector_authoritative(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ANALOGOUS_REFACTOR_HIT_SCHEMA,
            "interface": ANALOGOUS_REFACTOR_HIT_INTERFACE,
            "record_cid": self.record_cid,
            "channel": self.channel,
            "millirank": self.millirank,
            "overlapping_member_ids": list(self.overlapping_member_ids),
            "evidence_class": self.evidence_class,
            "outcome": self.outcome,
            "advisory": self.advisory,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "vector_authoritative": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def hit_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["hit_cid"] = self.hit_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalogousRefactorHit":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("hit_cid")
        if payload.pop("schema") != ANALOGOUS_REFACTOR_HIT_SCHEMA:
            raise PartitionRetrievalError("unsupported AnalogousRefactorHit schema")
        if payload.pop("interface") != ANALOGOUS_REFACTOR_HIT_INTERFACE:
            raise PartitionRetrievalError(
                "unsupported AnalogousRefactorHit interface"
            )
        _pop_authority_flags(payload, "analogous refactor hit")
        result = cls(**payload)
        _verify_cid(claimed, result.hit_cid, "AnalogousRefactorHit hit_cid")
        return result


def _coerce_hit(
    value: AnalogousRefactorHit | Mapping[str, Any],
) -> AnalogousRefactorHit:
    if isinstance(value, AnalogousRefactorHit):
        return value
    if isinstance(value, Mapping):
        if "hit_cid" in value:
            return AnalogousRefactorHit.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "hit_cid",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return AnalogousRefactorHit(**payload)
    raise PartitionRetrievalError("hit must be an AnalogousRefactorHit")


def _coerce_candidate(
    value: ProgramPartitionCandidate | Mapping[str, Any],
) -> ProgramPartitionCandidate:
    if isinstance(value, ProgramPartitionCandidate):
        return value
    if isinstance(value, Mapping):
        if "candidate_cid" in value:
            return ProgramPartitionCandidate.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "candidate_cid",
                "can_authorize_transition",
                "can_authorize_completion",
                "can_create_authority",
                "projection_is_authority",
            }
        }
        return ProgramPartitionCandidate(**payload)
    raise PartitionRetrievalError("candidate must be a ProgramPartitionCandidate")


def _candidates_from(
    value: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | ProgramPartitionCandidate
    | Mapping[str, Any],
) -> tuple[ProgramPartitionCandidate, ...]:
    if isinstance(value, PartitionGenerationReceipt):
        return tuple(value.candidates)
    if isinstance(value, ProgramPartitionCandidate):
        return (value,)
    if isinstance(value, Mapping) and (
        "candidate_cid" in value or "generator_kind" in value
    ):
        return (_coerce_candidate(value),)
    if isinstance(value, Mapping) and "candidates" in value:
        value = value["candidates"]
    if isinstance(value, (list, tuple)):
        candidates = tuple(_coerce_candidate(item) for item in value)
        if len(candidates) > MAX_RECORDS:
            raise PartitionRetrievalError("candidates exceed maximum length")
        return candidates
    raise PartitionRetrievalError("candidates must be a list or generation receipt")


def _corpus_from(
    value: Sequence[AnalogousRefactorRecord | Mapping[str, Any]],
) -> tuple[AnalogousRefactorRecord, ...]:
    if not isinstance(value, (list, tuple)):
        raise PartitionRetrievalError("corpus must be a list")
    records = tuple(_coerce_record(item) for item in value)
    if len(records) > MAX_RECORDS:
        raise PartitionRetrievalError("corpus exceeds maximum length")
    seen = [item.record_cid for item in records]
    if len(seen) != len(set(seen)):
        raise PartitionRetrievalError("duplicate analogous refactor identity")
    return records


def _channel_evidence(record: AnalogousRefactorRecord, channel: str) -> str:
    if channel == RetrievalChannel.VECTOR.value:
        return "vector_candidate"
    return record.evidence_class


def _channel_advisory(record: AnalogousRefactorRecord, channel: str) -> bool:
    if channel in ADVISORY_CHANNELS:
        return True
    return record.advisory


def _hits_for_record(
    query: AnalogousRefactorQuery,
    record: AnalogousRefactorRecord,
) -> tuple[AnalogousRefactorHit, ...]:
    if record.tree_id != query.tree_id:
        raise PartitionRetrievalError("stale tree_id in analogous refactor corpus")
    overlap = tuple(sorted(set(query.member_ids) & set(record.member_ids)))
    hits: list[AnalogousRefactorHit] = []
    if query.member_ids and query.member_ids == record.member_ids:
        hits.append(
            AnalogousRefactorHit(
                record_cid=record.record_cid,
                channel=RetrievalChannel.EXACT,
                millirank=MAX_MILLIRANK,
                overlapping_member_ids=overlap,
                evidence_class=_channel_evidence(
                    record, RetrievalChannel.EXACT.value
                ),
                outcome=record.outcome,
                advisory=_channel_advisory(record, RetrievalChannel.EXACT.value),
            )
        )
    lexical = _jaccard_millirank(query.tokens, record.tokens)
    if lexical > 0:
        hits.append(
            AnalogousRefactorHit(
                record_cid=record.record_cid,
                channel=RetrievalChannel.LEXICAL,
                millirank=lexical,
                overlapping_member_ids=overlap,
                evidence_class=_channel_evidence(
                    record, RetrievalChannel.LEXICAL.value
                ),
                outcome=record.outcome,
                advisory=_channel_advisory(record, RetrievalChannel.LEXICAL.value),
            )
        )
    graph = _jaccard_millirank(
        _graph_fingerprints(query.member_ids, query.graph_edges),
        _graph_fingerprints(record.member_ids, record.graph_edges),
    )
    if graph > 0:
        hits.append(
            AnalogousRefactorHit(
                record_cid=record.record_cid,
                channel=RetrievalChannel.GRAPH,
                millirank=graph,
                overlapping_member_ids=overlap,
                evidence_class=_channel_evidence(
                    record, RetrievalChannel.GRAPH.value
                ),
                outcome=record.outcome,
                advisory=_channel_advisory(record, RetrievalChannel.GRAPH.value),
            )
        )
    if query.vector_available and record.vector_available:
        vector = _cosine_millirank(query.vector, record.vector)
        if vector > 0:
            hits.append(
                AnalogousRefactorHit(
                    record_cid=record.record_cid,
                    channel=RetrievalChannel.VECTOR,
                    millirank=vector,
                    overlapping_member_ids=overlap,
                    evidence_class="vector_candidate",
                    outcome=record.outcome
                    if record.outcome == REJECTED_OUTCOME
                    else ADVISORY_OUTCOME,
                    advisory=True,
                )
            )
    return tuple(hits)


def _store_vector_hits(
    query: AnalogousRefactorQuery,
    records: Sequence[AnalogousRefactorRecord],
    projection_search: Callable[..., Any] | None,
) -> tuple[tuple[AnalogousRefactorHit, ...], bool]:
    if projection_search is None or not query.vector_available:
        return (), False
    try:
        results = projection_search(query.vector, k=len(records) or 1)
    except Exception as exc:
        reason = str(getattr(exc, "reason_code", "") or "")
        name = type(exc).__name__
        if reason in {"neural_unavailable", "projection_capability"} or name in {
            "ProjectionCapabilityError",
            "VectorChannelUnavailable",
        }:
            return (), True
        raise
    if results in (None, ()):
        return (), False
    by_cid = {item.record_cid: item for item in records}
    hits: list[AnalogousRefactorHit] = []
    if not isinstance(results, (list, tuple)):
        raise PartitionRetrievalError("projection search must return a list")
    for item in results:
        projected = _project(item)
        if not isinstance(projected, dict):
            metadata = getattr(item, "metadata", None)
            score = getattr(item, "score", None)
            if isinstance(metadata, Mapping):
                projected = {**_project(metadata), "score": score}
            else:
                raise PartitionRetrievalError(
                    "projection search hits must be objects"
                )
        record_cid = projected.get("record_cid") or projected.get("projection_cid")
        if record_cid not in by_cid:
            raise PartitionRetrievalError(
                "projection search hit does not resolve to the sealed corpus"
            )
        record = by_cid[record_cid]
        millirank = _cosine_millirank(query.vector, record.vector)
        if millirank <= 0:
            continue
        hits.append(
            AnalogousRefactorHit(
                record_cid=record.record_cid,
                channel=RetrievalChannel.VECTOR,
                millirank=millirank,
                overlapping_member_ids=tuple(
                    sorted(set(query.member_ids) & set(record.member_ids))
                ),
                evidence_class="vector_candidate",
                outcome=ADVISORY_OUTCOME,
                advisory=True,
            )
        )
    return tuple(hits), False


@dataclass(frozen=True, slots=True)
class AnalogousRefactorRetrievalReceipt:
    """Deterministic retrieval nomination receipt. Not ranking authority."""

    tree_id: str
    query: AnalogousRefactorQuery | Mapping[str, Any]
    corpus_record_cids: Sequence[str]
    hits: Sequence[AnalogousRefactorHit | Mapping[str, Any]]
    analyzer_id: str = ANALYZER_ID
    vector_channel_unavailable: bool = False

    interface: ClassVar[str] = ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE
    schema: ClassVar[str] = ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "query",
            "corpus_record_cids",
            "hits",
            "analyzer_id",
            "vector_channel_unavailable",
            "matched_record_cids",
            "unmatched_record_cids",
            "rejected_record_cids",
            "advisory_hit_cids",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "vector_authoritative",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise PartitionRetrievalError(
                "analyzer_id must remain the SPAR-015 analyzer"
            )
        query = _coerce_query(self.query)
        tree_id = _tree_id(self.tree_id)
        if query.tree_id != tree_id:
            raise PartitionRetrievalError("retrieval receipt tree_id does not match query")
        corpus = _unique_sorted_text(
            list(self.corpus_record_cids),
            "corpus_record_cids",
            limit=MAX_RECORDS,
        )
        hits = tuple(
            sorted(
                (_coerce_hit(item) for item in self.hits),
                key=lambda item: (
                    CHANNEL_ORDER.index(item.channel)
                    if item.channel in CHANNEL_ORDER
                    else len(CHANNEL_ORDER),
                    -item.millirank,
                    item.record_cid,
                    item.hit_cid,
                ),
            )
        )
        if len(hits) > MAX_HITS:
            raise PartitionRetrievalError("hits exceed maximum length")
        unknown = [item.record_cid for item in hits if item.record_cid not in corpus]
        if unknown:
            raise PartitionRetrievalError(
                f"hit record_cid is not in corpus: {sorted(set(unknown))}"
            )
        seen = [item.hit_cid for item in hits]
        if len(seen) != len(set(seen)):
            raise PartitionRetrievalError("duplicate retrieval hit identity")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "corpus_record_cids", corpus)
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(
            self,
            "vector_channel_unavailable",
            _bool(self.vector_channel_unavailable, "vector_channel_unavailable"),
        )

    @property
    def matched_record_cids(self) -> tuple[str, ...]:
        return tuple(sorted({item.record_cid for item in self.hits}))

    @property
    def unmatched_record_cids(self) -> tuple[str, ...]:
        matched = set(self.matched_record_cids)
        return tuple(
            item for item in self.corpus_record_cids if item not in matched
        )

    @property
    def rejected_record_cids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.record_cid
                    for item in self.hits
                    if item.outcome == REJECTED_OUTCOME
                }
            )
        )

    @property
    def advisory_hit_cids(self) -> tuple[str, ...]:
        return tuple(item.hit_cid for item in self.hits if item.advisory)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def vector_authoritative(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_SCHEMA,
            "interface": ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "query": self.query.to_dict(),
            "corpus_record_cids": list(self.corpus_record_cids),
            "hits": [item.to_dict() for item in self.hits],
            "analyzer_id": self.analyzer_id,
            "vector_channel_unavailable": self.vector_channel_unavailable,
            "matched_record_cids": list(self.matched_record_cids),
            "unmatched_record_cids": list(self.unmatched_record_cids),
            "rejected_record_cids": list(self.rejected_record_cids),
            "advisory_hit_cids": list(self.advisory_hit_cids),
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "vector_authoritative": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "AnalogousRefactorRetrievalReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_SCHEMA:
            raise PartitionRetrievalError(
                "unsupported AnalogousRefactorRetrievalReceipt schema"
            )
        if payload.pop("interface") != ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE:
            raise PartitionRetrievalError(
                "unsupported AnalogousRefactorRetrievalReceipt interface"
            )
        _pop_authority_flags(payload, "retrieval receipt")
        payload.pop("matched_record_cids")
        payload.pop("unmatched_record_cids")
        payload.pop("rejected_record_cids")
        payload.pop("advisory_hit_cids")
        result = cls(**payload)
        _verify_cid(
            claimed,
            result.receipt_cid,
            "AnalogousRefactorRetrievalReceipt receipt_cid",
        )
        return result


@dataclass(frozen=True, slots=True)
class CandidateRankScore:
    """Advisory per-candidate score. Cannot change admission or violations."""

    candidate_cid: str
    exact_millirank: int = 0
    lexical_millirank: int = 0
    graph_millirank: int = 0
    vector_millirank: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "exact_millirank",
            _millirank(self.exact_millirank, "exact_millirank"),
        )
        object.__setattr__(
            self,
            "lexical_millirank",
            _millirank(self.lexical_millirank, "lexical_millirank"),
        )
        object.__setattr__(
            self,
            "graph_millirank",
            _millirank(self.graph_millirank, "graph_millirank"),
        )
        object.__setattr__(
            self,
            "vector_millirank",
            _millirank(self.vector_millirank, "vector_millirank"),
        )

    @property
    def rank_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.exact_millirank,
            self.lexical_millirank,
            self.graph_millirank,
            self.vector_millirank,
            self.candidate_cid,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_cid": self.candidate_cid,
            "exact_millirank": self.exact_millirank,
            "lexical_millirank": self.lexical_millirank,
            "graph_millirank": self.graph_millirank,
            "vector_millirank": self.vector_millirank,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CandidateRankScore":
        payload = _mapping(data, "scores")
        unknown = set(payload) - {
            "candidate_cid",
            "exact_millirank",
            "lexical_millirank",
            "graph_millirank",
            "vector_millirank",
        }
        if unknown:
            raise PartitionRetrievalError(
                f"unknown rank score field: {sorted(unknown)}"
            )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PartitionRankingReceipt:
    """Advisory reorder of already-admitted candidates. Not completion."""

    tree_id: str
    input_candidate_cids: Sequence[str]
    ranked_candidate_cids: Sequence[str]
    scores: Sequence[CandidateRankScore | Mapping[str, Any]]
    admitted_candidate_cids: Sequence[str]
    preserved_non_admitted_cids: Sequence[str]
    hard_constraint_violations: Mapping[str, Sequence[str]]
    retrieval_receipt_cid: str = ""
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = PARTITION_RANKING_RECEIPT_INTERFACE
    schema: ClassVar[str] = PARTITION_RANKING_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "input_candidate_cids",
            "ranked_candidate_cids",
            "scores",
            "admitted_candidate_cids",
            "preserved_non_admitted_cids",
            "hard_constraint_violations",
            "retrieval_receipt_cid",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "vector_authoritative",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise PartitionRetrievalError(
                "analyzer_id must remain the SPAR-015 analyzer"
            )
        input_cids = tuple(
            _cid(item, "input_candidate_cids") for item in self.input_candidate_cids
        )
        if len(input_cids) != len(set(input_cids)):
            raise PartitionRetrievalError("duplicate input candidate identity")
        ranked = tuple(
            _cid(item, "ranked_candidate_cids") for item in self.ranked_candidate_cids
        )
        if tuple(sorted(ranked)) != tuple(sorted(input_cids)):
            raise PartitionRetrievalError(
                "ranking cannot drop, add, or hide partition candidates"
            )
        admitted = tuple(
            _cid(item, "admitted_candidate_cids")
            for item in self.admitted_candidate_cids
        )
        preserved = tuple(
            _cid(item, "preserved_non_admitted_cids")
            for item in self.preserved_non_admitted_cids
        )
        if tuple(sorted((*admitted, *preserved))) != tuple(sorted(input_cids)):
            raise PartitionRetrievalError(
                "ranking must preserve admitted and non-admitted partitions"
            )
        if set(admitted) & set(preserved):
            raise PartitionRetrievalError(
                "admitted candidates cannot also be preserved non-admitted"
            )
        violations = _mapping(self.hard_constraint_violations, "hard_constraint_violations")
        normalized_violations: dict[str, tuple[str, ...]] = {}
        for key, items in violations.items():
            cid = _cid(key, "hard_constraint_violations")
            if cid not in input_cids:
                raise PartitionRetrievalError(
                    "violation snapshot refers to an unknown candidate"
                )
            normalized_violations[cid] = _unique_sorted_text(
                list(items),
                "hard_constraint_violations",
                limit=MAX_MEMBERS,
            )
        if set(normalized_violations) != set(input_cids):
            raise PartitionRetrievalError(
                "ranking cannot hide hard-constraint violations"
            )
        for cid in admitted:
            if normalized_violations[cid]:
                raise PartitionRetrievalError(
                    "admitted ranked candidate cannot carry hidden violations"
                )
        scores = tuple(
            item
            if isinstance(item, CandidateRankScore)
            else CandidateRankScore.from_mapping(item)
            for item in self.scores
        )
        score_cids = [item.candidate_cid for item in scores]
        if tuple(sorted(score_cids)) != tuple(sorted(admitted)):
            raise PartitionRetrievalError(
                "rank scores apply only to already-admitted candidates"
            )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "input_candidate_cids", input_cids)
        object.__setattr__(self, "ranked_candidate_cids", ranked)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "admitted_candidate_cids", admitted)
        object.__setattr__(self, "preserved_non_admitted_cids", preserved)
        object.__setattr__(
            self,
            "hard_constraint_violations",
            {
                key: list(value)
                for key, value in sorted(normalized_violations.items())
            },
        )
        object.__setattr__(
            self,
            "retrieval_receipt_cid",
            ""
            if self.retrieval_receipt_cid in (None, "")
            else _cid(self.retrieval_receipt_cid, "retrieval_receipt_cid"),
        )
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def vector_authoritative(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PARTITION_RANKING_RECEIPT_SCHEMA,
            "interface": PARTITION_RANKING_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "input_candidate_cids": list(self.input_candidate_cids),
            "ranked_candidate_cids": list(self.ranked_candidate_cids),
            "scores": [item.to_dict() for item in self.scores],
            "admitted_candidate_cids": list(self.admitted_candidate_cids),
            "preserved_non_admitted_cids": list(self.preserved_non_admitted_cids),
            "hard_constraint_violations": {
                key: list(value)
                for key, value in self.hard_constraint_violations.items()
            },
            "retrieval_receipt_cid": self.retrieval_receipt_cid,
            "analyzer_id": self.analyzer_id,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "vector_authoritative": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionRankingReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != PARTITION_RANKING_RECEIPT_SCHEMA:
            raise PartitionRetrievalError(
                "unsupported PartitionRankingReceipt schema"
            )
        if payload.pop("interface") != PARTITION_RANKING_RECEIPT_INTERFACE:
            raise PartitionRetrievalError(
                "unsupported PartitionRankingReceipt interface"
            )
        _pop_authority_flags(payload, "ranking receipt")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "PartitionRankingReceipt receipt_cid"
        )
        return result


def _score_candidate(
    candidate: ProgramPartitionCandidate,
    hits: Sequence[AnalogousRefactorHit],
) -> CandidateRankScore:
    best = {
        RetrievalChannel.EXACT.value: 0,
        RetrievalChannel.LEXICAL.value: 0,
        RetrievalChannel.GRAPH.value: 0,
        RetrievalChannel.VECTOR.value: 0,
    }
    members = set(candidate.member_ids)
    for hit in hits:
        if hit.outcome == REJECTED_OUTCOME:
            continue
        overlap = set(hit.overlapping_member_ids)
        if hit.channel == RetrievalChannel.EXACT.value:
            if tuple(candidate.member_ids) != tuple(hit.overlapping_member_ids):
                continue
        elif overlap and not (members & overlap):
            continue
        elif not overlap and hit.channel != RetrievalChannel.VECTOR.value:
            continue
        if hit.millirank > best[hit.channel]:
            best[hit.channel] = hit.millirank
    return CandidateRankScore(
        candidate_cid=candidate.candidate_cid,
        exact_millirank=best[RetrievalChannel.EXACT.value],
        lexical_millirank=best[RetrievalChannel.LEXICAL.value],
        graph_millirank=best[RetrievalChannel.GRAPH.value],
        vector_millirank=best[RetrievalChannel.VECTOR.value],
    )


def _reorder_admitted(
    candidates: Sequence[ProgramPartitionCandidate],
    scores: Mapping[str, CandidateRankScore],
) -> tuple[ProgramPartitionCandidate, ...]:
    admitted_indexes = [
        index for index, item in enumerate(candidates) if item.admitted
    ]
    ordered = sorted(
        (candidates[index] for index in admitted_indexes),
        key=lambda item: (
            -scores[item.candidate_cid].exact_millirank,
            -scores[item.candidate_cid].lexical_millirank,
            -scores[item.candidate_cid].graph_millirank,
            -scores[item.candidate_cid].vector_millirank,
            item.candidate_cid,
        ),
    )
    ranked = list(candidates)
    for index, item in zip(admitted_indexes, ordered):
        ranked[index] = item
    return tuple(ranked)


class AnalogousRefactorRetriever:
    """Exact/lexical/graph/vector analog retrieval. Nomination only."""

    interface: ClassVar[str] = ANALOGOUS_REFACTOR_RETRIEVER_INTERFACE

    def __init__(
        self,
        projection_search: Callable[..., Any] | None = None,
    ) -> None:
        self.projection_search = projection_search

    def retrieve(
        self,
        query: AnalogousRefactorQuery | Mapping[str, Any],
        corpus: Sequence[AnalogousRefactorRecord | Mapping[str, Any]],
    ) -> AnalogousRefactorRetrievalReceipt:
        sealed_query = _coerce_query(query)
        records = _corpus_from(corpus)
        hits: list[AnalogousRefactorHit] = []
        for record in records:
            hits.extend(_hits_for_record(sealed_query, record))
        store_hits, unavailable = _store_vector_hits(
            sealed_query, records, self.projection_search
        )
        known = {item.hit_cid for item in hits}
        for item in store_hits:
            if item.hit_cid not in known:
                hits.append(item)
                known.add(item.hit_cid)
        return AnalogousRefactorRetrievalReceipt(
            tree_id=sealed_query.tree_id,
            query=sealed_query,
            corpus_record_cids=tuple(item.record_cid for item in records),
            hits=tuple(hits),
            vector_channel_unavailable=unavailable,
        )

    def rank(
        self,
        candidates: PartitionGenerationReceipt
        | Sequence[ProgramPartitionCandidate | Mapping[str, Any]],
        corpus: Sequence[AnalogousRefactorRecord | Mapping[str, Any]] = (),
        *,
        query: AnalogousRefactorQuery | Mapping[str, Any] | None = None,
        retrieval: AnalogousRefactorRetrievalReceipt | Mapping[str, Any] | None = None,
    ) -> PartitionRankingReceipt:
        return AdvisoryPartitionRanker(retriever=self).rank(
            candidates,
            corpus,
            query=query,
            retrieval=retrieval,
        )


class AdvisoryPartitionRanker:
    """Reorder already-admitted candidates without hiding violations."""

    interface: ClassVar[str] = ADVISORY_PARTITION_RANKER_INTERFACE

    def __init__(
        self,
        retriever: AnalogousRefactorRetriever | None = None,
    ) -> None:
        self.retriever = retriever or AnalogousRefactorRetriever()

    def rank(
        self,
        candidates: PartitionGenerationReceipt
        | Sequence[ProgramPartitionCandidate | Mapping[str, Any]],
        corpus: Sequence[AnalogousRefactorRecord | Mapping[str, Any]] = (),
        *,
        query: AnalogousRefactorQuery | Mapping[str, Any] | None = None,
        retrieval: AnalogousRefactorRetrievalReceipt | Mapping[str, Any] | None = None,
    ) -> PartitionRankingReceipt:
        sealed = _candidates_from(candidates)
        if not sealed:
            raise PartitionRetrievalError("ranking requires partition candidates")
        tree_id = sealed[0].tree_id
        if any(item.tree_id != tree_id for item in sealed):
            raise PartitionRetrievalError("ranked candidates must share tree_id")
        input_cids = tuple(item.candidate_cid for item in sealed)
        if len(input_cids) != len(set(input_cids)):
            raise PartitionRetrievalError("duplicate partition candidate identity")
        retrieval_cid = ""
        per_candidate_hits: dict[str, tuple[AnalogousRefactorHit, ...]] = {
            item.candidate_cid: () for item in sealed if item.admitted
        }
        if retrieval is not None:
            if isinstance(retrieval, AnalogousRefactorRetrievalReceipt):
                receipt = retrieval
            elif isinstance(retrieval, Mapping):
                receipt = AnalogousRefactorRetrievalReceipt.from_dict(retrieval)
            else:
                raise PartitionRetrievalError("retrieval must be a retrieval receipt")
            if receipt.tree_id != tree_id:
                raise PartitionRetrievalError("retrieval tree_id does not match candidates")
            retrieval_cid = receipt.receipt_cid
            for item in sealed:
                if item.admitted:
                    per_candidate_hits[item.candidate_cid] = receipt.hits
        elif corpus:
            base_query = (
                _coerce_query(query)
                if query is not None
                else AnalogousRefactorQuery(tree_id=tree_id)
            )
            if base_query.tree_id != tree_id:
                raise PartitionRetrievalError("query tree_id does not match candidates")
            receipts: list[AnalogousRefactorRetrievalReceipt] = []
            for item in sealed:
                if not item.admitted:
                    continue
                overlaid = base_query.overlay_candidate(item)
                receipt = self.retriever.retrieve(overlaid, corpus)
                receipts.append(receipt)
                per_candidate_hits[item.candidate_cid] = receipt.hits
            if len({item.receipt_cid for item in receipts}) == 1:
                retrieval_cid = receipts[0].receipt_cid
        scores = {
            item.candidate_cid: _score_candidate(
                item, per_candidate_hits.get(item.candidate_cid, ())
            )
            for item in sealed
            if item.admitted
        }
        ranked = _reorder_admitted(sealed, scores)
        if tuple(item.candidate_cid for item in ranked) and set(
            item.candidate_cid for item in ranked
        ) != set(input_cids):
            raise PartitionRetrievalError(
                "ranking cannot drop, add, or hide partition candidates"
            )
        for original, result in zip(sealed, ranked):
            if original.admitted != result.admitted:
                raise PartitionRetrievalError(
                    "advisory ranker cannot change admission"
                )
            if original.hard_constraint_violations != result.hard_constraint_violations:
                raise PartitionRetrievalError(
                    "advisory ranker cannot hide hard-constraint violations"
                )
        return PartitionRankingReceipt(
            tree_id=tree_id,
            input_candidate_cids=input_cids,
            ranked_candidate_cids=tuple(item.candidate_cid for item in ranked),
            scores=tuple(scores[item.candidate_cid] for item in ranked if item.admitted),
            admitted_candidate_cids=tuple(
                item.candidate_cid for item in ranked if item.admitted
            ),
            preserved_non_admitted_cids=tuple(
                item.candidate_cid for item in ranked if not item.admitted
            ),
            hard_constraint_violations={
                item.candidate_cid: list(item.hard_constraint_violations)
                for item in sealed
            },
            retrieval_receipt_cid=retrieval_cid,
        )


def retrieve_analogous_refactors(
    query: AnalogousRefactorQuery | Mapping[str, Any],
    corpus: Sequence[AnalogousRefactorRecord | Mapping[str, Any]],
    *,
    projection_search: Callable[..., Any] | None = None,
) -> AnalogousRefactorRetrievalReceipt:
    """Retrieve exact/lexical/graph/vector prior refactors as nominations."""

    return AnalogousRefactorRetriever(
        projection_search=projection_search
    ).retrieve(query, corpus)


def rank_admitted_partitions(
    candidates: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]],
    corpus: Sequence[AnalogousRefactorRecord | Mapping[str, Any]] = (),
    *,
    query: AnalogousRefactorQuery | Mapping[str, Any] | None = None,
    retrieval: AnalogousRefactorRetrievalReceipt | Mapping[str, Any] | None = None,
    projection_search: Callable[..., Any] | None = None,
) -> PartitionRankingReceipt:
    """Optionally reorder already-admitted candidates without granting authority."""

    retriever = AnalogousRefactorRetriever(projection_search=projection_search)
    return retriever.rank(
        candidates, corpus, query=query, retrieval=retrieval
    )


def encode_canonical_retrieval_receipt(
    receipt: AnalogousRefactorRetrievalReceipt,
) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_retrieval_receipt(
    payload: Mapping[str, Any],
) -> AnalogousRefactorRetrievalReceipt:
    return AnalogousRefactorRetrievalReceipt.from_dict(payload)


def encode_canonical_ranking_receipt(
    receipt: PartitionRankingReceipt,
) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_ranking_receipt(
    payload: Mapping[str, Any],
) -> PartitionRankingReceipt:
    return PartitionRankingReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise PartitionRetrievalError(
            f"partition retrieval must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADVISORY_CHANNELS",
    "ADVISORY_PARTITION_RANKER_INTERFACE",
    "ANALYZER_ID",
    "ANALOGOUS_REFACTOR_HIT_INTERFACE",
    "ANALOGOUS_REFACTOR_QUERY_INTERFACE",
    "ANALOGOUS_REFACTOR_RECORD_INTERFACE",
    "ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE",
    "ANALOGOUS_REFACTOR_RETRIEVER_INTERFACE",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "AdvisoryPartitionRanker",
    "AnalogousGraphEdge",
    "AnalogousRefactorHit",
    "AnalogousRefactorQuery",
    "AnalogousRefactorRecord",
    "AnalogousRefactorRetrievalReceipt",
    "AnalogousRefactorRetriever",
    "CHANNEL_ORDER",
    "CandidateRankScore",
    "DUCKLAKE_IS_AUTHORITY",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PARTITION_RANKING_RECEIPT_INTERFACE",
    "PARTITION_RETRIEVAL_CONTRACT_VERSION",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "PartitionRankingReceipt",
    "PartitionRetrievalError",
    "RANKER_IS_ADVISORY",
    "RAW_SOURCE_REQUIRED",
    "RETRIEVAL_CAN_AUTHORIZE_COMPLETION",
    "RETRIEVAL_CAN_AUTHORIZE_TRANSITION",
    "RETRIEVAL_CAN_CREATE_AUTHORITY",
    "RETRIEVAL_IS_NOMINATION_ONLY",
    "RecordOutcome",
    "RetrievalChannel",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "assert_not_competing_capsule_family",
    "decode_canonical_ranking_receipt",
    "decode_canonical_retrieval_receipt",
    "encode_canonical_ranking_receipt",
    "encode_canonical_retrieval_receipt",
    "partition_retrieval_cid_profile",
    "provider_free_exports",
    "rank_admitted_partitions",
    "retrieve_analogous_refactors",
]
