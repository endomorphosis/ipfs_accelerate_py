"""SPAR-013 deterministic candidate partition generators.

This module extends current supervisor partition orchestration with
``ProgramPartitionCandidate@1``.  It consumes sealed SPAR-009/010/011/012
evidence (SCC, state, contract, tests/proofs, graph, co-change) and emits
multiple deterministic partition nominations.  Projection clustering remains
advisory and cannot admit a candidate.

It does not replace datasets semantic authority, does not mint a competing
task/graph/identity/VFS/proof/context/scheduler/vector/merge/state authority,
and cannot authorize a transition or completion.  Model, vector, and
heuristic output remain nomination-only.  Observational metadata is excluded
from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import (
    cid_for_dag_json,
    validate_cid,
)


TASK_ID: Final[str] = "SPAR-013"
GOAL_ID: Final[str] = "SPAR-G032"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators@1"
)

PROGRAM_PARTITION_CANDIDATE_INTERFACE: Final[str] = "ProgramPartitionCandidate@1"
PARTITION_GENERATION_RECEIPT_INTERFACE: Final[str] = "PartitionGenerationReceipt@1"
PARTITION_CUT_EDGE_INTERFACE: Final[str] = "PartitionCutEdge@1"
PARTITION_EVIDENCE_BUNDLE_INTERFACE: Final[str] = "PartitionEvidenceBundle@1"

PROGRAM_PARTITION_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-partition-candidate@1"
)
PARTITION_GENERATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-generation-receipt@1"
)
PARTITION_CUT_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-cut-edge@1"
)
PARTITION_EVIDENCE_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-evidence-bundle@1"
)

PARTITION_CONTRACT_VERSION: Final[str] = "1"

PARTITION_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
PARTITION_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
PARTITION_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_CANDIDATES: Final[int] = 16_384
MAX_EDGES: Final[int] = 65_536
MAX_VIOLATIONS: Final[int] = 4_096
MAX_EVIDENCE_CIDS: Final[int] = 1_024

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "timestamp",
        "timestamps",
        "process_id",
        "pid",
        "local_path",
        "local_paths",
        "checkout_path",
        "model_output",
        "model",
        "provider",
        "prompt",
        "lease",
        "fence",
        "generation",
        "receipt",
        "acceptance",
        "wall_clock",
        "clock",
    }
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

GRAPH_CLUSTER_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "contains",
        "defines",
        "member_of",
        "exports",
        "references",
        "documents",
        "derived_from",
        "implements",
    }
)
TESTS_PROOFS_EDGE_KINDS: Final[frozenset[str]] = frozenset({"tests", "proves"})
HARD_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "aliases",
        "calls",
        "depends_on",
        "derived_from",
        "implements",
        "imports",
        "registers",
        "uses_resource",
    }
)
_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)
_UNIQUE_OWNER: Final[str] = "unique"
GENERATOR_ORDER: Final[tuple[str, ...]] = (
    "scc",
    "state",
    "contract",
    "tests_proofs",
    "graph",
    "cochange",
    "projection",
)


class PartitionGeneratorError(ValueError):
    """Fail-closed violation of a SPAR-013 partition-generator contract."""


class GeneratorKind(str, Enum):
    SCC = "scc"
    STATE = "state"
    CONTRACT = "contract"
    TESTS_PROOFS = "tests_proofs"
    GRAPH = "graph"
    COCHANGE = "cochange"
    PROJECTION = "projection"


class ConstraintClass(str, Enum):
    SCC_SPLIT = "scc_split"
    OVERSIZED_CYCLE = "oversized_cycle"
    UNIQUE_OWNER_SPLIT = "unique_owner_split"
    UNIQUE_OWNER_OVERLAP = "unique_owner_overlap"
    UNKNOWN_MEMBER = "unknown_member"
    PROJECTION_ADVISORY = "projection_clustering_is_advisory"
    VECTOR_OR_MODEL = "vector_or_model_evidence"
    UNRESOLVED_STATE = "unresolved_state"
    UNRESOLVED_ALIAS = "unresolved_alias"


ADVISORY_GENERATOR_KINDS: Final[frozenset[str]] = frozenset(
    {GeneratorKind.PROJECTION.value}
)
DETERMINISTIC_GENERATOR_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in GeneratorKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise PartitionGeneratorError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise PartitionGeneratorError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise PartitionGeneratorError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise PartitionGeneratorError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise PartitionGeneratorError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise PartitionGeneratorError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PartitionGeneratorError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise PartitionGeneratorError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise PartitionGeneratorError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise PartitionGeneratorError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise PartitionGeneratorError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise PartitionGeneratorError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise PartitionGeneratorError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise PartitionGeneratorError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise PartitionGeneratorError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise PartitionGeneratorError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise PartitionGeneratorError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise PartitionGeneratorError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise PartitionGeneratorError(f"unknown {name}: {text}") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise PartitionGeneratorError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
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
    raise PartitionGeneratorError(
        f"unsupported projected type {type(value).__name__}"
    )


def _sequence_maps(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        projected = _project(value)
        if isinstance(projected, list):
            value = projected
        else:
            raise PartitionGeneratorError(f"{name} must be a list")
    items = []
    for item in value:
        projected = _project(item)
        if not isinstance(projected, dict):
            raise PartitionGeneratorError(f"{name} items must be objects")
        items.append(projected)
    if len(items) > MAX_EDGES:
        raise PartitionGeneratorError(f"{name} exceeds maximum length")
    return tuple(items)


def partition_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _union_find_components(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    present = set(parent)
    for source, target in edges:
        if source in present and target in present:
            union(source, target)
    groups: dict[str, list[str]] = {}
    for node in sorted(present):
        groups.setdefault(find(node), []).append(node)
    components = [tuple(members) for members in groups.values()]
    components.sort(key=lambda members: (members[0], len(members), members))
    return tuple(components)


@dataclass(frozen=True, slots=True)
class PartitionCutEdge:
    """One cut edge recorded against a partition candidate."""

    source_id: str
    target_id: str
    kind: str
    constraint_class: str = "soft"

    interface: ClassVar[str] = PARTITION_CUT_EDGE_INTERFACE
    schema: ClassVar[str] = PARTITION_CUT_EDGE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "source_id",
            "target_id",
            "kind",
            "constraint_class",
            "edge_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(
            self,
            "constraint_class",
            _text(self.constraint_class, "constraint_class"),
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PARTITION_CUT_EDGE_SCHEMA,
            "interface": PARTITION_CUT_EDGE_INTERFACE,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind,
            "constraint_class": self.constraint_class,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def edge_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["edge_cid"] = self.edge_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionCutEdge":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("edge_cid")
        if payload.pop("schema") != PARTITION_CUT_EDGE_SCHEMA:
            raise PartitionGeneratorError("unsupported PartitionCutEdge schema")
        if payload.pop("interface") != PARTITION_CUT_EDGE_INTERFACE:
            raise PartitionGeneratorError("unsupported PartitionCutEdge interface")
        result = cls(**payload)
        _verify_cid(claimed, result.edge_cid, "PartitionCutEdge edge_cid")
        return result


def _coerce_cut_edge(value: PartitionCutEdge | Mapping[str, Any]) -> PartitionCutEdge:
    if isinstance(value, PartitionCutEdge):
        return value
    if isinstance(value, Mapping):
        if "edge_cid" in value:
            return PartitionCutEdge.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "edge_cid"}
        }
        return PartitionCutEdge(**payload)
    raise PartitionGeneratorError("cut edge must be a PartitionCutEdge")


@dataclass(frozen=True, slots=True)
class ProgramPartitionCandidate:
    """One deterministic partition nomination. Not transition authority."""

    tree_id: str
    generator_kind: GeneratorKind | str
    member_ids: Sequence[str]
    scc_ids: Sequence[str] = ()
    cut_edges: Sequence[PartitionCutEdge | Mapping[str, Any]] = ()
    evidence_cids: Sequence[str] = ()
    consumer_ids: Sequence[str] = ()
    obligation_ids: Sequence[str] = ()
    state_owner_ids: Sequence[str] = ()
    hard_constraint_violations: Sequence[str] = ()
    evidence_class: str = "exact_static_fact"
    advisory: bool = False
    admitted: bool = False

    interface: ClassVar[str] = PROGRAM_PARTITION_CANDIDATE_INTERFACE
    schema: ClassVar[str] = PROGRAM_PARTITION_CANDIDATE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "generator_kind",
            "member_ids",
            "scc_ids",
            "cut_edges",
            "evidence_cids",
            "consumer_ids",
            "obligation_ids",
            "state_owner_ids",
            "hard_constraint_violations",
            "evidence_class",
            "advisory",
            "admitted",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "candidate_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.generator_kind, GeneratorKind, "generator_kind")
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        if not members:
            raise PartitionGeneratorError("partition candidate requires members")
        cuts = tuple(
            sorted(
                (_coerce_cut_edge(item) for item in self.cut_edges),
                key=lambda item: item.edge_cid,
            )
        )
        if len(cuts) > MAX_EDGES:
            raise PartitionGeneratorError("cut_edges exceed maximum length")
        violations = _unique_sorted_text(
            list(self.hard_constraint_violations),
            "hard_constraint_violations",
            limit=MAX_VIOLATIONS,
        )
        advisory = _bool(self.advisory, "advisory")
        admitted = _bool(self.admitted, "admitted")
        if kind in ADVISORY_GENERATOR_KINDS:
            advisory = True
            admitted = False
            if ConstraintClass.PROJECTION_ADVISORY.value not in violations:
                violations = tuple(
                    sorted((*violations, ConstraintClass.PROJECTION_ADVISORY.value))
                )
        if admitted and violations:
            raise PartitionGeneratorError(
                "admitted partition candidate cannot carry hard-constraint violations"
            )
        if admitted and advisory:
            raise PartitionGeneratorError(
                "advisory partition candidate cannot be admitted"
            )
        evidence = _text(self.evidence_class, "evidence_class")
        if admitted and evidence in _NON_ADMITTING_EVIDENCE:
            raise PartitionGeneratorError(
                "vector or model evidence cannot admit a partition candidate"
            )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "generator_kind", kind)
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(
            self,
            "scc_ids",
            _unique_sorted_text(list(self.scc_ids), "scc_ids", limit=MAX_MEMBERS),
        )
        object.__setattr__(self, "cut_edges", cuts)
        evidence_cids = tuple(
            sorted(_cid(item, "evidence_cids") for item in self.evidence_cids)
        )
        if len(evidence_cids) != len(set(evidence_cids)):
            raise PartitionGeneratorError("evidence_cids must not contain duplicates")
        if len(evidence_cids) > MAX_EVIDENCE_CIDS:
            raise PartitionGeneratorError("evidence_cids exceed maximum length")
        object.__setattr__(self, "evidence_cids", evidence_cids)
        object.__setattr__(
            self,
            "consumer_ids",
            _unique_sorted_text(
                list(self.consumer_ids), "consumer_ids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _unique_sorted_text(
                list(self.obligation_ids), "obligation_ids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "state_owner_ids",
            _unique_sorted_text(
                list(self.state_owner_ids), "state_owner_ids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(self, "hard_constraint_violations", violations)
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "advisory", advisory)
        object.__setattr__(self, "admitted", admitted)

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PROGRAM_PARTITION_CANDIDATE_SCHEMA,
            "interface": PROGRAM_PARTITION_CANDIDATE_INTERFACE,
            "tree_id": self.tree_id,
            "generator_kind": self.generator_kind,
            "member_ids": list(self.member_ids),
            "scc_ids": list(self.scc_ids),
            "cut_edges": [item.to_dict() for item in self.cut_edges],
            "evidence_cids": list(self.evidence_cids),
            "consumer_ids": list(self.consumer_ids),
            "obligation_ids": list(self.obligation_ids),
            "state_owner_ids": list(self.state_owner_ids),
            "hard_constraint_violations": list(self.hard_constraint_violations),
            "evidence_class": self.evidence_class,
            "advisory": self.advisory,
            "admitted": self.admitted,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def candidate_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["candidate_cid"] = self.candidate_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramPartitionCandidate":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("candidate_cid")
        if payload.pop("schema") != PROGRAM_PARTITION_CANDIDATE_SCHEMA:
            raise PartitionGeneratorError(
                "unsupported ProgramPartitionCandidate schema"
            )
        if payload.pop("interface") != PROGRAM_PARTITION_CANDIDATE_INTERFACE:
            raise PartitionGeneratorError(
                "unsupported ProgramPartitionCandidate interface"
            )
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
        ):
            if payload.pop(flag) is not False:
                raise PartitionGeneratorError(
                    f"partition candidate cannot claim {flag}"
                )
        result = cls(**payload)
        _verify_cid(
            claimed, result.candidate_cid, "ProgramPartitionCandidate candidate_cid"
        )
        return result


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
    raise PartitionGeneratorError("candidate must be a ProgramPartitionCandidate")


@dataclass(frozen=True, slots=True)
class _SccIndex:
    snapshot_cid: str
    node_to_scc: Mapping[str, str]
    members: Mapping[str, tuple[str, ...]]
    cyclic: Mapping[str, bool]
    oversized: Mapping[str, bool]
    state_owner_ids: Mapping[str, tuple[str, ...]]
    condensation: tuple[tuple[str, str, str], ...]
    node_ids: tuple[str, ...]

    def sccs_for(self, member_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted({self.node_to_scc[item] for item in member_ids}))

    def close(self, member_ids: Sequence[str]) -> tuple[str, ...]:
        unknown = [item for item in member_ids if item not in self.node_to_scc]
        if unknown:
            raise PartitionGeneratorError(
                f"unknown SCC member: {sorted(set(unknown))}"
            )
        closed: list[str] = []
        for scc_id in self.sccs_for(member_ids):
            closed.extend(self.members[scc_id])
        return tuple(sorted(set(closed)))

    def oversized_in(self, member_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            scc_id
            for scc_id in self.sccs_for(member_ids)
            if self.oversized.get(scc_id, False)
        )


def _build_scc_index(snapshot: Mapping[str, Any]) -> _SccIndex:
    payload = _mapping(snapshot, "scc_snapshot")
    snapshot_cid = _cid(
        payload.get("snapshot_cid") or payload.get("graph_view_cid"),
        "scc_snapshot.snapshot_cid",
    )
    components = _sequence_maps(payload.get("components"), "scc_snapshot.components")
    if not components:
        raise PartitionGeneratorError("scc_snapshot.components must be nonempty")
    node_to_scc: dict[str, str] = {}
    members: dict[str, tuple[str, ...]] = {}
    cyclic: dict[str, bool] = {}
    oversized: dict[str, bool] = {}
    state_owners: dict[str, tuple[str, ...]] = {}
    for item in components:
        scc_id = _cid(item.get("scc_id"), "scc_id")
        member_ids = _unique_sorted_text(
            list(item.get("member_ids") or ()), "member_ids", limit=MAX_MEMBERS
        )
        if not member_ids:
            raise PartitionGeneratorError("SCC component requires members")
        if scc_id in members:
            raise PartitionGeneratorError("duplicate SCC identity")
        for member in member_ids:
            if member in node_to_scc:
                raise PartitionGeneratorError("SCC members must be partitioned")
            node_to_scc[member] = scc_id
        members[scc_id] = member_ids
        cyclic[scc_id] = _bool(item.get("cyclic", False), "cyclic")
        oversized_flag = _bool(item.get("oversized", False), "oversized")
        if oversized_flag and not cyclic[scc_id]:
            raise PartitionGeneratorError("oversized flag applies only to cyclic SCCs")
        oversized[scc_id] = oversized_flag
        state_owners[scc_id] = _unique_sorted_text(
            list(item.get("state_owner_ids") or ()),
            "state_owner_ids",
            limit=MAX_MEMBERS,
        )
    condensation_raw = _sequence_maps(
        payload.get("condensation_edges"), "scc_snapshot.condensation_edges"
    )
    condensation = tuple(
        sorted(
            (
                _text(item["source_scc_id"], "source_scc_id"),
                _text(item["target_scc_id"], "target_scc_id"),
                _text(item.get("witness_kind", "depends_on"), "witness_kind"),
            )
            for item in condensation_raw
        )
    )
    present = set(members)
    for source, target, _kind in condensation:
        if source not in present or target not in present:
            raise PartitionGeneratorError("condensation edge names an unknown SCC")
    return _SccIndex(
        snapshot_cid=snapshot_cid,
        node_to_scc=node_to_scc,
        members=members,
        cyclic=cyclic,
        oversized=oversized,
        state_owner_ids=state_owners,
        condensation=condensation,
        node_ids=tuple(sorted(node_to_scc)),
    )


@dataclass(frozen=True, slots=True)
class _StateIndex:
    graph_cid: str
    unique_owners: Mapping[str, tuple[str, ...]]
    owner_of: Mapping[str, str]
    unresolved_subjects: frozenset[str]
    unresolved_aliases: frozenset[str]


def _build_state_index(state: Mapping[str, Any] | None) -> _StateIndex | None:
    if state is None:
        return None
    payload = _mapping(state, "state_ownership")
    graph_cid = _cid(
        payload.get("graph_cid") or payload.get("subject_cid"),
        "state_ownership.graph_cid",
    )
    alias_by_id: dict[str, tuple[str, ...]] = {}
    unresolved_aliases: set[str] = set()
    for item in _sequence_maps(payload.get("alias_sets"), "alias_sets"):
        alias_id = _text(
            item.get("alias_set_id")
            or f"alias:{item.get('representative_id', '')}",
            "alias_set_id",
        )
        members = _unique_sorted_text(
            list(item.get("member_ids") or ()), "alias member_ids", limit=MAX_MEMBERS
        )
        alias_by_id[alias_id] = members
        if _bool(item.get("unresolved", False), "unresolved"):
            unresolved_aliases.add(alias_id)
    unique_owners: dict[str, tuple[str, ...]] = {}
    owner_of: dict[str, str] = {}
    for item in _sequence_maps(payload.get("owners"), "owners"):
        uniqueness = _text(item.get("uniqueness", ""), "uniqueness", empty=True)
        if uniqueness != _UNIQUE_OWNER:
            continue
        owner_id = _text(item.get("owner_id"), "owner_id")
        alias_id = _text(item.get("alias_set_id"), "alias_set_id")
        members = alias_by_id.get(alias_id)
        if members is None:
            symbol = _text(
                item.get("owning_symbol_id") or owner_id, "owning_symbol_id"
            )
            members = (symbol,)
        unique_owners[owner_id] = members
        for member in members:
            previous = owner_of.get(member)
            if previous is not None and previous != owner_id:
                raise PartitionGeneratorError(
                    "overlapping unique state owners are forbidden"
                )
            owner_of[member] = owner_id
    unresolved_subjects = {
        _text(item.get("subject_id"), "subject_id")
        for item in _sequence_maps(payload.get("unresolved"), "unresolved")
    }
    return _StateIndex(
        graph_cid=graph_cid,
        unique_owners=unique_owners,
        owner_of=owner_of,
        unresolved_subjects=frozenset(unresolved_subjects),
        unresolved_aliases=frozenset(unresolved_aliases),
    )


@dataclass(frozen=True, slots=True)
class _EvidenceView:
    tree_id: str
    scc: _SccIndex
    state: _StateIndex | None
    graph_cid: str
    graph_edges: tuple[tuple[str, str, str, str, str], ...]
    init_edges: tuple[tuple[str, str, str], ...]
    obligations: tuple[dict[str, Any], ...]
    consumers: tuple[dict[str, Any], ...]
    cochange_edges: tuple[tuple[str, str, str], ...]
    projection_clusters: tuple[dict[str, Any], ...]
    compatibility_cid: str
    initialization_cid: str
    analyzer_id: str


def _edge_tuple(
    item: Mapping[str, Any],
    *,
    default_kind: str,
    default_evidence: str,
) -> tuple[str, str, str, str, str]:
    return (
        _text(item.get("source_id"), "source_id"),
        _text(item.get("target_id"), "target_id"),
        _text(item.get("kind", default_kind), "kind"),
        _text(item.get("evidence_class", default_evidence), "evidence_class"),
        _text(item.get("confidence", "exact"), "confidence"),
    )


def compile_partition_evidence(
    evidence: Mapping[str, Any] | Any,
    *,
    analyzer_id: str = ANALYZER_ID,
) -> _EvidenceView:
    """Normalize sealed SPAR-009/010/011/012 evidence for generation."""

    payload = _mapping(evidence, "partition evidence")
    tree_id = _tree_id(payload.get("tree_id"))
    snapshot = payload.get("scc_snapshot") or payload.get("scc")
    if snapshot is None:
        raise PartitionGeneratorError("scc_snapshot is required")
    scc = _build_scc_index(snapshot)
    state = _build_state_index(payload.get("state_ownership") or payload.get("state"))
    graph_payload = payload.get("graph_view") or payload.get("graph") or {}
    graph_map = _mapping(graph_payload, "graph_view") if graph_payload else {}
    graph_cid = _optional_cid(
        graph_map.get("graph_view_cid")
        or graph_map.get("view_cid")
        or graph_map.get("snapshot_cid"),
        "graph_view_cid",
    )
    graph_edges = tuple(
        _edge_tuple(item, default_kind="calls", default_evidence="exact_static_fact")
        for item in _sequence_maps(graph_map.get("edges"), "graph_view.edges")
    )
    init_payload = payload.get("initialization") or payload.get(
        "initialization_graph"
    )
    init_map = _mapping(init_payload, "initialization") if init_payload else {}
    initialization_cid = _optional_cid(
        init_map.get("graph_cid"), "initialization.graph_cid"
    )
    init_edges = tuple(
        (
            _text(item.get("source_id"), "source_id"),
            _text(item.get("target_id"), "target_id"),
            _text(item.get("kind", "happens_before"), "kind"),
        )
        for item in _sequence_maps(init_map.get("edges"), "initialization.edges")
    )
    compatibility = payload.get("compatibility") or payload.get(
        "compatibility_inventory"
    )
    compatibility_map = (
        _mapping(compatibility, "compatibility") if compatibility else {}
    )
    compatibility_cid = _optional_cid(
        compatibility_map.get("inventory_cid"), "compatibility.inventory_cid"
    )
    obligations = _sequence_maps(
        compatibility_map.get("obligations"), "compatibility.obligations"
    )
    consumers = _sequence_maps(
        compatibility_map.get("consumers"), "compatibility.consumers"
    )
    cochange = payload.get("cochange_edges") or payload.get("co_change_edges") or ()
    cochange_edges = tuple(
        (
            _text(item.get("source_id"), "source_id"),
            _text(item.get("target_id"), "target_id"),
            _text(
                item.get("evidence_class", "runtime_observation"),
                "evidence_class",
            ),
        )
        for item in _sequence_maps(cochange, "cochange_edges")
    )
    projection_clusters = _sequence_maps(
        payload.get("projection_clusters"), "projection_clusters"
    )
    declared_analyzer = _text(
        payload.get("analyzer_id", analyzer_id), "analyzer_id"
    )
    if declared_analyzer != ANALYZER_ID:
        raise PartitionGeneratorError("analyzer_id must remain the SPAR-013 analyzer")
    return _EvidenceView(
        tree_id=tree_id,
        scc=scc,
        state=state,
        graph_cid=graph_cid,
        graph_edges=graph_edges,
        init_edges=init_edges,
        obligations=obligations,
        consumers=consumers,
        cochange_edges=cochange_edges,
        projection_clusters=projection_clusters,
        compatibility_cid=compatibility_cid,
        initialization_cid=initialization_cid,
        analyzer_id=declared_analyzer,
    )


def _owner_violations(
    members: Sequence[str],
    state: _StateIndex | None,
) -> tuple[str, ...]:
    if state is None:
        return ()
    violations: list[str] = []
    owners = {state.owner_of[item] for item in members if item in state.owner_of}
    for owner_id in owners:
        required = set(state.unique_owners[owner_id])
        present = required & set(members)
        if present != required:
            violations.append(ConstraintClass.UNIQUE_OWNER_SPLIT.value)
        if owner_id in state.unresolved_aliases:
            violations.append(ConstraintClass.UNRESOLVED_ALIAS.value)
    if len(owners) > 1:
        # Multiple unique owners in one candidate is allowed only when the
        # generator closed a hard SCC that already contains them.
        pass
    unresolved = [item for item in members if item in state.unresolved_subjects]
    if unresolved:
        violations.append(ConstraintClass.UNRESOLVED_STATE.value)
    return tuple(sorted(set(violations)))


def _close_unique_owners(
    members: Sequence[str],
    state: _StateIndex | None,
) -> tuple[str, ...]:
    if state is None:
        return tuple(members)
    closed = set(members)
    for member in list(closed):
        owner_id = state.owner_of.get(member)
        if owner_id is None:
            continue
        closed.update(state.unique_owners[owner_id])
    return tuple(sorted(closed))


def _constraint_class_for_kind(kind: str) -> str:
    if kind in HARD_EDGE_KINDS or kind in {"happens_before", "initialization_order"}:
        return "hard"
    if kind in TESTS_PROOFS_EDGE_KINDS:
        return "tests_proofs"
    if kind == "cochange":
        return "soft"
    return "soft"


def _cut_edges_for(
    members: Sequence[str],
    view: _EvidenceView,
) -> tuple[PartitionCutEdge, ...]:
    inside = set(members)
    cuts: list[PartitionCutEdge] = []
    for source, target, kind, _evidence, _confidence in view.graph_edges:
        if (source in inside) ^ (target in inside):
            cuts.append(
                PartitionCutEdge(
                    source_id=source,
                    target_id=target,
                    kind=kind,
                    constraint_class=_constraint_class_for_kind(kind),
                )
            )
    for source, target, kind in view.init_edges:
        if (source in inside) ^ (target in inside):
            cuts.append(
                PartitionCutEdge(
                    source_id=source,
                    target_id=target,
                    kind=kind,
                    constraint_class="hard",
                )
            )
    member_sccs = set(view.scc.sccs_for(members))
    for source, target, kind in view.scc.condensation:
        if (source in member_sccs) ^ (target in member_sccs):
            cuts.append(
                PartitionCutEdge(
                    source_id=source,
                    target_id=target,
                    kind=kind,
                    constraint_class="hard",
                )
            )
    unique: dict[str, PartitionCutEdge] = {}
    for item in cuts:
        unique[item.edge_cid] = item
    return tuple(sorted(unique.values(), key=lambda item: item.edge_cid))


def _evidence_cids(view: _EvidenceView, extra: Iterable[str] = ()) -> tuple[str, ...]:
    cids = [view.scc.snapshot_cid]
    if view.state is not None:
        cids.append(view.state.graph_cid)
    if view.graph_cid:
        cids.append(view.graph_cid)
    if view.compatibility_cid:
        cids.append(view.compatibility_cid)
    if view.initialization_cid:
        cids.append(view.initialization_cid)
    cids.extend(item for item in extra if item)
    return tuple(sorted(set(cids)))


def _build_candidate(
    *,
    view: _EvidenceView,
    kind: GeneratorKind,
    member_ids: Sequence[str],
    evidence_class: str,
    extra_evidence: Iterable[str] = (),
    consumer_ids: Sequence[str] = (),
    obligation_ids: Sequence[str] = (),
    extra_violations: Sequence[str] = (),
    close_owners: bool = True,
) -> ProgramPartitionCandidate:
    members = tuple(sorted(set(member_ids)))
    violations = list(extra_violations)
    try:
        members = view.scc.close(members)
    except PartitionGeneratorError:
        violations.append(ConstraintClass.UNKNOWN_MEMBER.value)
        unknown = [item for item in members if item not in view.scc.node_to_scc]
        members = tuple(sorted(set(members) - set(unknown)))
        if not members:
            raise
        members = view.scc.close(members)
    if close_owners:
        members = view.scc.close(_close_unique_owners(members, view.state))
    original = set(member_ids)
    if original and original < set(members) and set(members) - original:
        # Closing under SCC/owner is required; it is not a split violation.
        pass
    if any(item in original and item not in view.scc.node_to_scc for item in member_ids):
        violations.append(ConstraintClass.UNKNOWN_MEMBER.value)
    oversized = view.scc.oversized_in(members)
    if oversized:
        violations.append(ConstraintClass.OVERSIZED_CYCLE.value)
    owner_violations = _owner_violations(members, view.state)
    violations.extend(owner_violations)
    if evidence_class in _NON_ADMITTING_EVIDENCE:
        violations.append(ConstraintClass.VECTOR_OR_MODEL.value)
    advisory = kind.value in ADVISORY_GENERATOR_KINDS
    if advisory:
        violations.append(ConstraintClass.PROJECTION_ADVISORY.value)
    unique_violations = tuple(sorted(set(violations)))
    admitted = not unique_violations and not advisory
    state_owner_ids = ()
    if view.state is not None:
        state_owner_ids = tuple(
            sorted(
                {
                    view.state.owner_of[item]
                    for item in members
                    if item in view.state.owner_of
                }
            )
        )
    return ProgramPartitionCandidate(
        tree_id=view.tree_id,
        generator_kind=kind,
        member_ids=members,
        scc_ids=view.scc.sccs_for(members),
        cut_edges=_cut_edges_for(members, view),
        evidence_cids=_evidence_cids(view, extra_evidence),
        consumer_ids=consumer_ids,
        obligation_ids=obligation_ids,
        state_owner_ids=state_owner_ids,
        hard_constraint_violations=unique_violations,
        evidence_class=evidence_class,
        advisory=advisory,
        admitted=admitted,
    )


def generate_scc_candidates(view: _EvidenceView) -> tuple[ProgramPartitionCandidate, ...]:
    candidates: list[ProgramPartitionCandidate] = []
    for scc_id, members in sorted(view.scc.members.items()):
        extra = []
        if view.scc.oversized[scc_id]:
            extra.append(ConstraintClass.OVERSIZED_CYCLE.value)
        candidates.append(
            _build_candidate(
                view=view,
                kind=GeneratorKind.SCC,
                member_ids=members,
                evidence_class="exact_static_fact",
                extra_violations=extra,
                close_owners=True,
            )
        )
    return tuple(candidates)


def generate_state_candidates(
    view: _EvidenceView,
) -> tuple[ProgramPartitionCandidate, ...]:
    if view.state is None:
        return ()
    candidates: list[ProgramPartitionCandidate] = []
    for owner_id, members in sorted(view.state.unique_owners.items()):
        extra: list[str] = []
        if owner_id in view.state.unresolved_aliases:
            extra.append(ConstraintClass.UNRESOLVED_ALIAS.value)
        candidates.append(
            _build_candidate(
                view=view,
                kind=GeneratorKind.STATE,
                member_ids=members,
                evidence_class="exact_static_fact",
                extra_evidence=(view.state.graph_cid,),
                extra_violations=extra,
                close_owners=True,
            )
        )
    return tuple(candidates)


def generate_contract_candidates(
    view: _EvidenceView,
) -> tuple[ProgramPartitionCandidate, ...]:
    if not view.obligations:
        return ()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in view.obligations:
        subject = _text(item.get("subject_id"), "subject_id")
        grouped.setdefault(subject, []).append(item)
    candidates: list[ProgramPartitionCandidate] = []
    for subject, items in sorted(grouped.items()):
        obligation_ids = _unique_sorted_text(
            [_text(item.get("obligation_id"), "obligation_id") for item in items],
            "obligation_ids",
            limit=MAX_MEMBERS,
        )
        consumer_ids = _unique_sorted_text(
            [
                _text(item.get("consumer_id"), "consumer_id")
                for item in items
                if item.get("consumer_id")
            ],
            "consumer_ids",
            limit=MAX_MEMBERS,
        )
        extra_cids = [view.compatibility_cid] if view.compatibility_cid else []
        candidates.append(
            _build_candidate(
                view=view,
                kind=GeneratorKind.CONTRACT,
                member_ids=(subject,),
                evidence_class="exact_static_fact",
                extra_evidence=extra_cids,
                consumer_ids=consumer_ids,
                obligation_ids=obligation_ids,
            )
        )
    return tuple(candidates)


def _clusters_from_edges(
    view: _EvidenceView,
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    involved: set[str] = set()
    for source, target in pairs:
        involved.add(source)
        involved.add(target)
    unknown = sorted(item for item in involved if item not in view.scc.node_to_scc)
    if unknown:
        raise PartitionGeneratorError(f"unknown SCC member: {unknown}")
    if not involved:
        return ()
    return _union_find_components(sorted(involved), pairs)


def generate_tests_proofs_candidates(
    view: _EvidenceView,
) -> tuple[ProgramPartitionCandidate, ...]:
    pairs = [
        (source, target)
        for source, target, kind, evidence, confidence in view.graph_edges
        if kind in TESTS_PROOFS_EDGE_KINDS
        and evidence not in _NON_ADMITTING_EVIDENCE
        and confidence in {"exact", "conservative"}
    ]
    clusters = _clusters_from_edges(view, pairs)
    return tuple(
        _build_candidate(
            view=view,
            kind=GeneratorKind.TESTS_PROOFS,
            member_ids=cluster,
            evidence_class="test",
            extra_evidence=(view.graph_cid,) if view.graph_cid else (),
        )
        for cluster in clusters
    )


def generate_graph_candidates(
    view: _EvidenceView,
) -> tuple[ProgramPartitionCandidate, ...]:
    pairs = [
        (source, target)
        for source, target, kind, evidence, confidence in view.graph_edges
        if kind in GRAPH_CLUSTER_EDGE_KINDS
        and evidence not in _NON_ADMITTING_EVIDENCE
        and confidence == "exact"
    ]
    clusters = _clusters_from_edges(view, pairs)
    return tuple(
        _build_candidate(
            view=view,
            kind=GeneratorKind.GRAPH,
            member_ids=cluster,
            evidence_class="exact_static_fact",
            extra_evidence=(view.graph_cid,) if view.graph_cid else (),
        )
        for cluster in clusters
    )


def generate_cochange_candidates(
    view: _EvidenceView,
) -> tuple[ProgramPartitionCandidate, ...]:
    if not view.cochange_edges:
        return ()
    admitting_pairs = [
        (source, target)
        for source, target, evidence in view.cochange_edges
        if evidence not in _NON_ADMITTING_EVIDENCE
    ]
    advisory_pairs = [
        (source, target)
        for source, target, evidence in view.cochange_edges
        if evidence in _NON_ADMITTING_EVIDENCE
    ]
    candidates: list[ProgramPartitionCandidate] = []
    for cluster in _clusters_from_edges(view, admitting_pairs):
        evidence_classes = {
            evidence
            for source, target, evidence in view.cochange_edges
            if source in cluster or target in cluster
        }
        evidence_class = (
            "runtime_observation"
            if "runtime_observation" in evidence_classes
            else sorted(evidence_classes)[0]
        )
        extra = []
        if evidence_class in _NON_ADMITTING_EVIDENCE:
            extra.append(ConstraintClass.VECTOR_OR_MODEL.value)
        candidates.append(
            _build_candidate(
                view=view,
                kind=GeneratorKind.COCHANGE,
                member_ids=cluster,
                evidence_class=evidence_class,
                extra_violations=extra,
            )
        )
    for cluster in _clusters_from_edges(view, advisory_pairs):
        candidates.append(
            _build_candidate(
                view=view,
                kind=GeneratorKind.COCHANGE,
                member_ids=cluster,
                evidence_class="vector_candidate",
                extra_violations=(ConstraintClass.VECTOR_OR_MODEL.value,),
            )
        )
    unique: dict[str, ProgramPartitionCandidate] = {}
    for item in candidates:
        unique[item.candidate_cid] = item
    return tuple(sorted(unique.values(), key=lambda item: item.candidate_cid))


def generate_projection_candidates(
    view: _EvidenceView,
) -> tuple[ProgramPartitionCandidate, ...]:
    candidates: list[ProgramPartitionCandidate] = []
    for item in view.projection_clusters:
        members = _unique_sorted_text(
            list(item.get("member_ids") or ()), "member_ids", limit=MAX_MEMBERS
        )
        extra_cids = []
        pin = item.get("model_pin_cid") or item.get("projection_cid")
        if pin:
            extra_cids.append(_cid(pin, "projection_cid"))
        extra = [ConstraintClass.PROJECTION_ADVISORY.value]
        if set(members) - set(view.scc.node_to_scc):
            extra.append(ConstraintClass.UNKNOWN_MEMBER.value)
        original = set(members)
        closed = view.scc.close(
            [member for member in members if member in view.scc.node_to_scc]
        )
        if original and original != set(closed):
            extra.append(ConstraintClass.SCC_SPLIT.value)
        candidates.append(
            _build_candidate(
                view=view,
                kind=GeneratorKind.PROJECTION,
                member_ids=closed or members,
                evidence_class="vector_candidate",
                extra_evidence=extra_cids,
                extra_violations=extra,
            )
        )
    return tuple(candidates)


GENERATOR_DISPATCH: Final[Mapping[str, Any]] = {
    GeneratorKind.SCC.value: generate_scc_candidates,
    GeneratorKind.STATE.value: generate_state_candidates,
    GeneratorKind.CONTRACT.value: generate_contract_candidates,
    GeneratorKind.TESTS_PROOFS.value: generate_tests_proofs_candidates,
    GeneratorKind.GRAPH.value: generate_graph_candidates,
    GeneratorKind.COCHANGE.value: generate_cochange_candidates,
    GeneratorKind.PROJECTION.value: generate_projection_candidates,
}


@dataclass(frozen=True, slots=True)
class PartitionGenerationReceipt:
    """Deterministic multi-generator partition nomination receipt."""

    tree_id: str
    evidence_cids: Sequence[str]
    candidates: Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = PARTITION_GENERATION_RECEIPT_INTERFACE
    schema: ClassVar[str] = PARTITION_GENERATION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "evidence_cids",
            "candidates",
            "analyzer_id",
            "admitted_candidate_cids",
            "advisory_candidate_cids",
            "rejected_candidate_cids",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise PartitionGeneratorError(
                "analyzer_id must remain the SPAR-013 analyzer"
            )
        candidates = tuple(_coerce_candidate(item) for item in self.candidates)
        if len(candidates) > MAX_CANDIDATES:
            raise PartitionGeneratorError("candidates exceed maximum length")
        candidates = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    GENERATOR_ORDER.index(item.generator_kind)
                    if item.generator_kind in GENERATOR_ORDER
                    else len(GENERATOR_ORDER),
                    item.member_ids,
                    item.candidate_cid,
                ),
            )
        )
        seen = [item.candidate_cid for item in candidates]
        if len(seen) != len(set(seen)):
            raise PartitionGeneratorError("duplicate partition candidate identity")
        evidence = tuple(_cid(item, "evidence_cids") for item in self.evidence_cids)
        if len(evidence) != len(set(evidence)):
            raise PartitionGeneratorError("evidence_cids must not contain duplicates")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "evidence_cids", tuple(sorted(evidence)))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def admitted_candidates(self) -> tuple[ProgramPartitionCandidate, ...]:
        return tuple(item for item in self.candidates if item.admitted)

    @property
    def advisory_candidates(self) -> tuple[ProgramPartitionCandidate, ...]:
        return tuple(item for item in self.candidates if item.advisory)

    @property
    def rejected_candidates(self) -> tuple[ProgramPartitionCandidate, ...]:
        return tuple(
            item
            for item in self.candidates
            if not item.admitted and not item.advisory
        )

    @property
    def admitted_candidate_cids(self) -> tuple[str, ...]:
        return tuple(item.candidate_cid for item in self.admitted_candidates)

    @property
    def advisory_candidate_cids(self) -> tuple[str, ...]:
        return tuple(item.candidate_cid for item in self.advisory_candidates)

    @property
    def rejected_candidate_cids(self) -> tuple[str, ...]:
        return tuple(item.candidate_cid for item in self.rejected_candidates)

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PARTITION_GENERATION_RECEIPT_SCHEMA,
            "interface": PARTITION_GENERATION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "evidence_cids": list(self.evidence_cids),
            "candidates": [item.to_dict() for item in self.candidates],
            "analyzer_id": self.analyzer_id,
            "admitted_candidate_cids": list(self.admitted_candidate_cids),
            "advisory_candidate_cids": list(self.advisory_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionGenerationReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != PARTITION_GENERATION_RECEIPT_SCHEMA:
            raise PartitionGeneratorError(
                "unsupported PartitionGenerationReceipt schema"
            )
        if payload.pop("interface") != PARTITION_GENERATION_RECEIPT_INTERFACE:
            raise PartitionGeneratorError(
                "unsupported PartitionGenerationReceipt interface"
            )
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
        ):
            if payload.pop(flag) is not False:
                raise PartitionGeneratorError(f"receipt cannot claim {flag}")
        payload.pop("admitted_candidate_cids")
        payload.pop("advisory_candidate_cids")
        payload.pop("rejected_candidate_cids")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "PartitionGenerationReceipt receipt_cid"
        )
        return result


def generate_partition_candidates(
    evidence: Mapping[str, Any] | Any,
    *,
    generators: Sequence[str] | None = None,
) -> PartitionGenerationReceipt:
    """Emit deterministic partition nominations from sealed residual evidence.

    Projection clustering is generated when present and remains advisory.
    Soft co-change/test/graph signals cannot split an SCC or unique owner.
    """

    view = compile_partition_evidence(evidence)
    selected = tuple(generators) if generators is not None else GENERATOR_ORDER
    unknown = [item for item in selected if item not in GENERATOR_DISPATCH]
    if unknown:
        raise PartitionGeneratorError(f"unknown generator kind: {unknown}")
    produced: list[ProgramPartitionCandidate] = []
    for kind in selected:
        produced.extend(GENERATOR_DISPATCH[kind](view))
    unique: dict[str, ProgramPartitionCandidate] = {}
    for item in produced:
        unique[item.candidate_cid] = item
    return PartitionGenerationReceipt(
        tree_id=view.tree_id,
        evidence_cids=_evidence_cids(view),
        candidates=tuple(unique.values()),
        analyzer_id=view.analyzer_id,
    )


def encode_canonical_receipt(receipt: PartitionGenerationReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
) -> PartitionGenerationReceipt:
    return PartitionGenerationReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise PartitionGeneratorError(
            f"partition generators must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADVISORY_GENERATOR_KINDS",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "ConstraintClass",
    "DETERMINISTIC_GENERATOR_KINDS",
    "DUCKLAKE_IS_AUTHORITY",
    "GENERATOR_ORDER",
    "GOAL_ID",
    "GRAPH_CLUSTER_EDGE_KINDS",
    "HARD_EDGE_KINDS",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PARTITION_CAN_AUTHORIZE_COMPLETION",
    "PARTITION_CAN_AUTHORIZE_TRANSITION",
    "PARTITION_CAN_CREATE_AUTHORITY",
    "PARTITION_CONTRACT_VERSION",
    "PARTITION_CUT_EDGE_INTERFACE",
    "PARTITION_EVIDENCE_BUNDLE_INTERFACE",
    "PARTITION_GENERATION_RECEIPT_INTERFACE",
    "PARTITION_GENERATION_RECEIPT_SCHEMA",
    "PROGRAM",
    "PROGRAM_PARTITION_CANDIDATE_INTERFACE",
    "PROGRAM_PARTITION_CANDIDATE_SCHEMA",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "PartitionCutEdge",
    "PartitionGenerationReceipt",
    "PartitionGeneratorError",
    "ProgramPartitionCandidate",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TESTS_PROOFS_EDGE_KINDS",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "GeneratorKind",
    "assert_not_competing_capsule_family",
    "compile_partition_evidence",
    "decode_canonical_receipt",
    "encode_canonical_receipt",
    "generate_cochange_candidates",
    "generate_contract_candidates",
    "generate_graph_candidates",
    "generate_partition_candidates",
    "generate_projection_candidates",
    "generate_scc_candidates",
    "generate_state_candidates",
    "generate_tests_proofs_candidates",
    "partition_cid_profile",
    "provider_free_exports",
]
