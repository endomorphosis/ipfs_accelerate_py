"""Canonical authority ownership graph for the initial PCAR concern vocabulary.

`AuthorityOwnershipGraph` records reviewed ownership. It cannot transfer an
existing authority or authorize code changes. Each initial concern resolves to
exactly one evidence-backed canonical owner or a typed hard blocker. Adapters,
projections, and quarantined legacy/simulation paths remain explicit.
Unknown production ownership, re-export-only claims, heuristic or opaque
ownership, and multiple production authorities without formal arbitration fail
closed. Content identity, import, and re-export never establish authority.
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

from .architecture_ir import ArchitectureEdge, ArchitectureIR, ArchitectureNode
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
    NON_PROBATIVE_CONFIDENCE,
)

AUTHORITY_OWNERSHIP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/authority-ownership-graph@1"
)
AUTHORITY_OWNERSHIP_VERSION = 1
AUTHORITY_OWNERSHIP_EVIDENCE = "pcar/authority-ownership-graph@1"
CONCERN_OWNERSHIP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/concern-ownership@1"
)
CONCERN_OWNERSHIP_VERSION = 1
FORMAL_ARBITRATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/formal-arbitration@1"
)
FORMAL_ARBITRATION_VERSION = 1
OWNERSHIP_BLOCKER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/ownership-blocker@1"
)
OWNERSHIP_BLOCKER_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-006-authority-ownership-graph"
TASK_ID = "PCAR-006"
DEFAULT_FRESHNESS = "pcar-006-authority-ownership"
OWNERSHIP_GRAPH_CAN_AUTHORIZE_CHANGES = False
OWNERSHIP_GRAPH_CAN_TRANSFER_AUTHORITY = False
CONTENT_IDENTITY_IS_NOT_AUTHORITY = True
REEXPORT_IS_NOT_AUTHORITY = True
SILENT_ARBITRATION_PROHIBITED = True

_UNKNOWN_FIELD_MESSAGE = "unknown authority-ownership field"
_MISSING_FIELD_MESSAGE = "missing authority-ownership field"
_PROBATIVE_OWNER_EDGE_KINDS = frozenset(
    {
        EdgeKind.AUTHORIZES,
        EdgeKind.CONFIRMS,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.PERSISTS,
        EdgeKind.EXECUTES,
        EdgeKind.IMPLEMENTS,
        EdgeKind.PROVES,
        EdgeKind.TESTS,
    }
)
_NON_PROBATIVE_OWNER_EDGE_KINDS = frozenset(
    {
        EdgeKind.REEXPORTS,
        EdgeKind.IMPORTS,
        EdgeKind.DUPLICATES,
        EdgeKind.SHADOWS,
    }
)
_SILENT_ARBITRATOR_IDENTITIES = frozenset(
    {
        "content_identity",
        "first",
        "first_listed",
        "heuristic",
        "import",
        "majority",
        "reexport",
        "silent",
    }
)
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")


class AuthorityGraphError(ArchitectureContractError):
    """Fail-closed authority-ownership contract violation."""


class AuthorityGraphAuthorityError(AuthorityGraphError):
    """Raised when the ownership graph is asked to authorize or transfer."""


class ConcernKind(str, Enum):
    """Closed initial concern vocabulary (PCAR-PLAN-R1)."""

    CONTENT_IDENTITY = "content identity"
    OPERATION_IDENTITY = "operation identity"
    PROVIDER_CAPABILITY = "provider capability"
    PROVIDER_SELECTION = "provider selection"
    EXECUTION_RESULT = "execution result"
    TASK_IDENTITY = "task identity"
    OBJECTIVE_IDENTITY = "objective identity"
    POLICY_DECISION = "policy decision"
    AUTHORIZATION = "authorization"
    CONFIRMATION = "confirmation"
    LEASE_AND_FENCING = "lease and fencing"
    STATE_PERSISTENCE = "state persistence"
    PROOF_VERIFICATION = "proof verification"
    TEST_EVIDENCE = "test evidence"
    COMPLETION_EVIDENCE = "completion evidence"
    RELEASE_QUALIFICATION = "release qualification"


INITIAL_CONCERNS: tuple[ConcernKind, ...] = tuple(ConcernKind)
CLOSED_CONCERNS: frozenset[str] = frozenset(item.value for item in ConcernKind)


class OwnerDisposition(str, Enum):
    """Closed owner-path vocabulary for one concern."""

    CANONICAL = "canonical"
    ADAPTER = "adapter"
    PROJECTION = "projection"
    LEGACY = "legacy"
    SIMULATION = "simulation"
    UNKNOWN = "unknown"


CLOSED_OWNER_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in OwnerDisposition
)
NON_CANONICAL_DISPOSITIONS: frozenset[OwnerDisposition] = frozenset(
    {
        OwnerDisposition.ADAPTER,
        OwnerDisposition.PROJECTION,
        OwnerDisposition.LEGACY,
        OwnerDisposition.SIMULATION,
        OwnerDisposition.UNKNOWN,
    }
)


class OwnershipBlockerKind(str, Enum):
    """Closed hard-blocker vocabulary for unresolved ownership."""

    UNKNOWN_OWNER = "unknown_owner"
    MULTIPLE_PRODUCTION_AUTHORITIES = "multiple_production_authorities"
    MISSING_ARBITRATION = "missing_arbitration"
    NON_PROBATIVE_OWNERSHIP = "non_probative_ownership"
    REEXPORT_CLAIMED_AUTHORITY = "reexport_claimed_authority"
    CONTENT_IDENTITY_INFERRED_AUTHORITY = "content_identity_inferred_authority"
    SIMULATED_AS_LIVE = "simulated_as_live"
    NON_AUTHORITY_CANONICAL_CLAIM = "non_authority_canonical_claim"
    UNKNOWN_PRODUCTION_OWNER = "unknown_production_owner"
    CONFLICTING_DISPOSITION = "conflicting_disposition"
    UNCLASSIFIED_COMPETITOR = "unclassified_competitor"
    SILENT_ARBITRATION = "silent_arbitration"


CLOSED_OWNERSHIP_BLOCKERS: frozenset[str] = frozenset(
    item.value for item in OwnershipBlockerKind
)


class ArbitrationRationale(str, Enum):
    """Closed formal-arbitration rationale vocabulary."""

    EXPLICIT_REVIEWED_CONTRACT = "explicit_reviewed_contract"
    SUPERSEDES_EDGE = "supersedes_edge"
    DEPRECATES_EDGE = "deprecates_edge"
    ADAPTS_EDGE = "adapts_edge"


CLOSED_ARBITRATION_RATIONALES: frozenset[str] = frozenset(
    item.value for item in ArbitrationRationale
)


@dataclass(frozen=True)
class ConcernSourceBinding:
    """Current-tree source binding for one initial concern nomination."""

    concern: ConcernKind
    path: str
    nominated_symbol: str
    inventory_confidence: Confidence
    recommended_disposition: OwnerDisposition
    start_line: int
    end_line: int


INITIAL_CONCERN_SOURCE_BINDINGS: tuple[ConcernSourceBinding, ...] = (
    ConcernSourceBinding(
        ConcernKind.CONTENT_IDENTITY,
        "ipfs_accelerate_py/assurance/content_identity.py",
        "ContentIdentity",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        115,
        115,
    ),
    ConcernSourceBinding(
        ConcernKind.CONTENT_IDENTITY,
        "ipfs_accelerate_py/agent_supervisor/core/multiformats_identity.py",
        "CIDProfile",
        Confidence.CONSERVATIVE,
        OwnerDisposition.ADAPTER,
        216,
        216,
    ),
    ConcernSourceBinding(
        ConcernKind.OPERATION_IDENTITY,
        "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py",
        "OPERATION_CATALOG_V2",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        8346,
        8346,
    ),
    ConcernSourceBinding(
        ConcernKind.PROVIDER_CAPABILITY,
        "ipfs_accelerate_py/agent_supervisor/control/capability_resolver.py",
        "ProviderCapabilityEvidence",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        233,
        233,
    ),
    ConcernSourceBinding(
        ConcernKind.PROVIDER_SELECTION,
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_route.py",
        "ProviderRoutePolicy",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        42,
        42,
    ),
    ConcernSourceBinding(
        ConcernKind.EXECUTION_RESULT,
        "ipfs_accelerate_py/agent_supervisor/contracts/execution.py",
        "InvocationMode",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        167,
        167,
    ),
    ConcernSourceBinding(
        ConcernKind.TASK_IDENTITY,
        "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
        "DatabaseTaskSource",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        325,
        325,
    ),
    ConcernSourceBinding(
        ConcernKind.OBJECTIVE_IDENTITY,
        "ipfs_accelerate_py/agent_supervisor/objectives/objective_graph.py",
        "ObjectiveGoal",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        1531,
        1531,
    ),
    ConcernSourceBinding(
        ConcernKind.POLICY_DECISION,
        "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py",
        "ALLOWED_LOCAL_CAPABILITIES",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        39,
        39,
    ),
    ConcernSourceBinding(
        ConcernKind.AUTHORIZATION,
        "ipfs_accelerate_py/agent_supervisor/control/authorization_logic.py",
        "AuthorizationPolicy",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        533,
        533,
    ),
    ConcernSourceBinding(
        ConcernKind.CONFIRMATION,
        "ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
        "SupervisorControlService",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        4383,
        4383,
    ),
    ConcernSourceBinding(
        ConcernKind.LEASE_AND_FENCING,
        "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
        "LeaseCoordinator",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        1561,
        1561,
    ),
    ConcernSourceBinding(
        ConcernKind.STATE_PERSISTENCE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "DuckDBConnection",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        354,
        354,
    ),
    ConcernSourceBinding(
        ConcernKind.PROOF_VERIFICATION,
        "ipfs_accelerate_py/agent_supervisor/verification/planner.py",
        "IncrementalVerificationPlanner",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        2310,
        2310,
    ),
    ConcernSourceBinding(
        ConcernKind.TEST_EVIDENCE,
        "ipfs_accelerate_py/agent_supervisor/proof/test_execution_contracts.py",
        "TestExecutionKey",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        658,
        658,
    ),
    ConcernSourceBinding(
        ConcernKind.COMPLETION_EVIDENCE,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py",
        "AuthoritativeCompletionGate",
        Confidence.EXACT,
        OwnerDisposition.CANONICAL,
        178,
        178,
    ),
    ConcernSourceBinding(
        ConcernKind.RELEASE_QUALIFICATION,
        "ipfs_accelerate_py/agent_supervisor/evaluation/dcr_release.py",
        "DeterministicRepairRelease",
        Confidence.CONSERVATIVE,
        OwnerDisposition.CANONICAL,
        144,
        144,
    ),
)

_OWNER_FIELDS = frozenset(
    {
        "content_identity",
        "disposition",
        "evidence_edge_ids",
        "node_id",
        "node_kind",
        "provenance",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "edge_ids",
        "extractor_identity",
        "node_ids",
    }
)
_BLOCKER_FIELDS = frozenset(
    {
        "concern",
        "content_identity",
        "edge_ids",
        "kind",
        "message",
        "node_ids",
        "schema",
        "version",
    }
)
_LOSER_FIELDS = frozenset({"disposition", "node_id"})
_ARBITRATION_FIELDS = frozenset(
    {
        "arbitrator_identity",
        "canonical_owner_node_id",
        "concern",
        "content_identity",
        "evidence_edge_ids",
        "evidence_node_ids",
        "loser_classifications",
        "provenance",
        "rationale",
        "schema",
        "version",
    }
)
_CLAIM_FIELDS = frozenset(
    {"concern", "disposition", "evidence_edge_ids", "owner_node_id"}
)
_OWNERSHIP_FIELDS = frozenset(
    {
        "adapters",
        "arbitration",
        "blocker",
        "canonical_owner",
        "concern",
        "content_identity",
        "evidence",
        "legacy_owners",
        "projections",
        "schema",
        "simulation_owners",
        "unknown_owners",
        "version",
    }
)
_GRAPH_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "arbitrations",
        "blockers",
        "can_authorize_changes",
        "can_transfer_authority",
        "concerns",
        "content_identity",
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
        raise AuthorityGraphError("content identity must be a dag-json CIDv1") from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise AuthorityGraphError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise AuthorityGraphError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AuthorityGraphError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=AuthorityGraphError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _require_architecture_ir(graph: ArchitectureIR | Mapping[str, Any]) -> ArchitectureIR:
    if isinstance(graph, ArchitectureIR):
        return graph
    try:
        return ArchitectureIR.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise AuthorityGraphError(str(exc)) from exc


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _wrap_contract(exc: ArchitectureContractError) -> AuthorityGraphError:
    if isinstance(exc, AuthorityGraphError):
        return exc
    return AuthorityGraphError(str(exc))


@dataclass(frozen=True)
class ConcernOwner:
    """One explicit owner, adapter, projection, or quarantined path."""

    node_id: str
    disposition: OwnerDisposition
    node_kind: NodeKind
    provenance: SourceFactIdentity
    evidence_edge_ids: tuple[str, ...] = ()
    content_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            _require_text(self.node_id, "node_id", error_type=AuthorityGraphError),
        )
        if _looks_like_content_identity(self.node_id):
            raise AuthorityGraphError(
                "content identity is not inferred to be authority"
            )
        object.__setattr__(
            self,
            "disposition",
            _closed_enum(
                self.disposition,
                OwnerDisposition,
                "owner disposition",
                error_type=AuthorityGraphError,
            ),
        )
        object.__setattr__(
            self,
            "node_kind",
            _closed_enum(
                self.node_kind,
                NodeKind,
                "node kind",
                error_type=AuthorityGraphError,
            ),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "evidence_edge_ids",
            _require_text_tuple(self.evidence_edge_ids, "evidence_edge_ids"),
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=AuthorityGraphError,
                )
            )
            if claimed != identity:
                raise AuthorityGraphError("owner content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "evidence_edge_ids": list(self.evidence_edge_ids),
            "node_id": self.node_id,
            "node_kind": self.node_kind.value,
            "provenance": self.provenance.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise AuthorityGraphError("owner content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConcernOwner":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _OWNER_FIELDS)
        try:
            owner = cls(
                node_id=mapping["node_id"],
                disposition=mapping["disposition"],
                node_kind=mapping["node_kind"],
                provenance=mapping["provenance"],
                evidence_edge_ids=mapping["evidence_edge_ids"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != owner.content_identity:
            raise AuthorityGraphError("owner content identity mismatch")
        return owner

    from_dict = from_mapping


@dataclass(frozen=True)
class OwnershipEvidence:
    """Source-bound ArchitectureIR facts supporting one concern resolution."""

    architecture_ir_identity: str
    extractor_identity: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "architecture_ir_identity",
            _validate_dag_json_cid(
                _require_text(
                    self.architecture_ir_identity,
                    "architecture_ir_identity",
                    error_type=AuthorityGraphError,
                )
            ),
        )
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(
                self.extractor_identity,
                "extractor_identity",
                error_type=AuthorityGraphError,
            ),
        )
        object.__setattr__(
            self, "node_ids", _require_text_tuple(self.node_ids, "node_ids")
        )
        object.__setattr__(
            self, "edge_ids", _require_text_tuple(self.edge_ids, "edge_ids")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "edge_ids": list(self.edge_ids),
            "extractor_identity": self.extractor_identity,
            "node_ids": list(self.node_ids),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OwnershipEvidence":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _EVIDENCE_FIELDS)
        return cls(
            architecture_ir_identity=mapping["architecture_ir_identity"],
            extractor_identity=mapping["extractor_identity"],
            node_ids=mapping["node_ids"],
            edge_ids=mapping["edge_ids"],
        )

    from_dict = from_mapping


@dataclass(frozen=True)
class OwnershipBlocker:
    """Typed hard blocker that prevents canonical ownership."""

    concern: ConcernKind
    kind: OwnershipBlockerKind
    message: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    schema: str = OWNERSHIP_BLOCKER_SCHEMA
    version: int = OWNERSHIP_BLOCKER_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=AuthorityGraphError)
        if schema != OWNERSHIP_BLOCKER_SCHEMA:
            raise AuthorityGraphError("unexpected ownership-blocker schema")
        version = _require_int(self.version, "version", error_type=AuthorityGraphError)
        if version != OWNERSHIP_BLOCKER_VERSION:
            raise AuthorityGraphError("unexpected ownership-blocker version")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self,
            "concern",
            _closed_enum(
                self.concern, ConcernKind, "concern", error_type=AuthorityGraphError
            ),
        )
        object.__setattr__(
            self,
            "kind",
            _closed_enum(
                self.kind,
                OwnershipBlockerKind,
                "ownership blocker kind",
                error_type=AuthorityGraphError,
            ),
        )
        object.__setattr__(
            self,
            "message",
            _require_text(self.message, "message", error_type=AuthorityGraphError),
        )
        object.__setattr__(
            self, "node_ids", _require_text_tuple(self.node_ids, "node_ids")
        )
        object.__setattr__(
            self, "edge_ids", _require_text_tuple(self.edge_ids, "edge_ids")
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=AuthorityGraphError,
                )
            )
            if claimed != identity:
                raise AuthorityGraphError("blocker content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "concern": self.concern.value,
            "edge_ids": list(self.edge_ids),
            "kind": self.kind.value,
            "message": self.message,
            "node_ids": list(self.node_ids),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise AuthorityGraphError("blocker content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OwnershipBlocker":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _BLOCKER_FIELDS)
        blocker = cls(
            concern=mapping["concern"],
            kind=mapping["kind"],
            message=mapping["message"],
            node_ids=mapping["node_ids"],
            edge_ids=mapping["edge_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != blocker.content_identity:
            raise AuthorityGraphError("blocker content identity mismatch")
        return blocker

    from_dict = from_mapping


@dataclass(frozen=True)
class LoserClassification:
    """Explicit non-canonical disposition for an arbitrated competitor."""

    node_id: str
    disposition: OwnerDisposition

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            _require_text(self.node_id, "node_id", error_type=AuthorityGraphError),
        )
        disposition = _closed_enum(
            self.disposition,
            OwnerDisposition,
            "owner disposition",
            error_type=AuthorityGraphError,
        )
        if disposition is OwnerDisposition.CANONICAL:
            raise AuthorityGraphError("arbitrated loser cannot remain canonical")
        object.__setattr__(self, "disposition", disposition)

    def to_dict(self) -> dict[str, Any]:
        return {"disposition": self.disposition.value, "node_id": self.node_id}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LoserClassification":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _LOSER_FIELDS)
        return cls(node_id=mapping["node_id"], disposition=mapping["disposition"])

    from_dict = from_mapping


@dataclass(frozen=True)
class FormalArbitration:
    """Explicit contract that selects one canonical owner among competitors."""

    concern: ConcernKind
    canonical_owner_node_id: str
    loser_classifications: tuple[LoserClassification, ...]
    arbitrator_identity: str
    rationale: ArbitrationRationale
    provenance: SourceFactIdentity
    evidence_node_ids: tuple[str, ...] = ()
    evidence_edge_ids: tuple[str, ...] = ()
    schema: str = FORMAL_ARBITRATION_SCHEMA
    version: int = FORMAL_ARBITRATION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=AuthorityGraphError)
        if schema != FORMAL_ARBITRATION_SCHEMA:
            raise AuthorityGraphError("unexpected formal-arbitration schema")
        version = _require_int(self.version, "version", error_type=AuthorityGraphError)
        if version != FORMAL_ARBITRATION_VERSION:
            raise AuthorityGraphError("unexpected formal-arbitration version")
        concern = _closed_enum(
            self.concern, ConcernKind, "concern", error_type=AuthorityGraphError
        )
        canonical = _require_text(
            self.canonical_owner_node_id,
            "canonical_owner_node_id",
            error_type=AuthorityGraphError,
        )
        if _looks_like_content_identity(canonical):
            raise AuthorityGraphError(
                "content identity is not inferred to be authority"
            )
        arbitrator = _require_text(
            self.arbitrator_identity,
            "arbitrator_identity",
            error_type=AuthorityGraphError,
        )
        if arbitrator.lower() in _SILENT_ARBITRATOR_IDENTITIES:
            raise AuthorityGraphError("silent arbitration is prohibited")
        rationale = _closed_enum(
            self.rationale,
            ArbitrationRationale,
            "arbitration rationale",
            error_type=AuthorityGraphError,
        )
        if isinstance(self.loser_classifications, (str, bytes, bytearray)) or not isinstance(
            self.loser_classifications, Sequence
        ):
            raise AuthorityGraphError("loser_classifications must be a sequence")
        losers = tuple(
            item
            if isinstance(item, LoserClassification)
            else LoserClassification.from_mapping(item)
            for item in self.loser_classifications
        )
        ordered = tuple(sorted(losers, key=lambda item: item.node_id))
        loser_ids = tuple(item.node_id for item in ordered)
        if len(loser_ids) != len(set(loser_ids)):
            raise AuthorityGraphError("arbitrated losers must be unique")
        if canonical in set(loser_ids):
            raise AuthorityGraphError("arbitration winner cannot also be a loser")
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        if provenance.confidence in NON_PROBATIVE_CONFIDENCE:
            raise AuthorityGraphError(
                "heuristic or opaque facts cannot prove ownership"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "canonical_owner_node_id", canonical)
        object.__setattr__(self, "loser_classifications", ordered)
        object.__setattr__(self, "arbitrator_identity", arbitrator)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "evidence_node_ids",
            _require_text_tuple(self.evidence_node_ids, "evidence_node_ids"),
        )
        object.__setattr__(
            self,
            "evidence_edge_ids",
            _require_text_tuple(self.evidence_edge_ids, "evidence_edge_ids"),
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=AuthorityGraphError,
                )
            )
            if claimed != identity:
                raise AuthorityGraphError("arbitration content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "arbitrator_identity": self.arbitrator_identity,
            "canonical_owner_node_id": self.canonical_owner_node_id,
            "concern": self.concern.value,
            "evidence_edge_ids": list(self.evidence_edge_ids),
            "evidence_node_ids": list(self.evidence_node_ids),
            "loser_classifications": [
                item.to_dict() for item in self.loser_classifications
            ],
            "provenance": self.provenance.to_dict(),
            "rationale": self.rationale.value,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise AuthorityGraphError("arbitration content identity mismatch")
        return {**payload, "content_identity": identity}

    def loser_ids(self) -> frozenset[str]:
        return frozenset(item.node_id for item in self.loser_classifications)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FormalArbitration":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _ARBITRATION_FIELDS)
        try:
            record = cls(
                concern=mapping["concern"],
                canonical_owner_node_id=mapping["canonical_owner_node_id"],
                loser_classifications=mapping["loser_classifications"],
                arbitrator_identity=mapping["arbitrator_identity"],
                rationale=mapping["rationale"],
                provenance=mapping["provenance"],
                evidence_node_ids=mapping["evidence_node_ids"],
                evidence_edge_ids=mapping["evidence_edge_ids"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise AuthorityGraphError("arbitration content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ConcernClaim:
    """Explicit source-bound claim that a graph node plays a role for a concern."""

    concern: ConcernKind
    owner_node_id: str
    disposition: OwnerDisposition
    evidence_edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "concern",
            _closed_enum(
                self.concern, ConcernKind, "concern", error_type=AuthorityGraphError
            ),
        )
        owner_node_id = _require_text(
            self.owner_node_id, "owner_node_id", error_type=AuthorityGraphError
        )
        if _looks_like_content_identity(owner_node_id):
            raise AuthorityGraphError(
                "content identity is not inferred to be authority"
            )
        object.__setattr__(self, "owner_node_id", owner_node_id)
        object.__setattr__(
            self,
            "disposition",
            _closed_enum(
                self.disposition,
                OwnerDisposition,
                "owner disposition",
                error_type=AuthorityGraphError,
            ),
        )
        object.__setattr__(
            self,
            "evidence_edge_ids",
            _require_text_tuple(self.evidence_edge_ids, "evidence_edge_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "concern": self.concern.value,
            "disposition": self.disposition.value,
            "evidence_edge_ids": list(self.evidence_edge_ids),
            "owner_node_id": self.owner_node_id,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConcernClaim":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _CLAIM_FIELDS)
        return cls(
            concern=mapping["concern"],
            owner_node_id=mapping["owner_node_id"],
            disposition=mapping["disposition"],
            evidence_edge_ids=mapping["evidence_edge_ids"],
        )

    from_dict = from_mapping


def _optional_owner(value: Any) -> ConcernOwner | None:
    if value is None:
        return None
    if isinstance(value, ConcernOwner):
        return value
    if isinstance(value, Mapping):
        return ConcernOwner.from_mapping(value)
    raise AuthorityGraphError("canonical_owner must be an object or null")


def _optional_blocker(value: Any) -> OwnershipBlocker | None:
    if value is None:
        return None
    if isinstance(value, OwnershipBlocker):
        return value
    if isinstance(value, Mapping):
        return OwnershipBlocker.from_mapping(value)
    raise AuthorityGraphError("blocker must be an object or null")


def _optional_arbitration(value: Any) -> FormalArbitration | None:
    if value is None:
        return None
    if isinstance(value, FormalArbitration):
        return value
    if isinstance(value, Mapping):
        return FormalArbitration.from_mapping(value)
    raise AuthorityGraphError("arbitration must be an object or null")


def _owner_tuple(value: Any, name: str) -> tuple[ConcernOwner, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AuthorityGraphError(f"{name} must be a list of owner objects")
    owners = tuple(
        item if isinstance(item, ConcernOwner) else ConcernOwner.from_mapping(item)
        for item in value
    )
    ordered = tuple(sorted(owners, key=lambda item: item.node_id))
    ids = tuple(item.node_id for item in ordered)
    if len(ids) != len(set(ids)):
        raise AuthorityGraphError(f"{name} node ids must be unique")
    return ordered


@dataclass(frozen=True)
class ConcernOwnership:
    """Exactly one canonical owner or one typed hard blocker for a concern."""

    concern: ConcernKind
    canonical_owner: ConcernOwner | None
    adapters: tuple[ConcernOwner, ...]
    projections: tuple[ConcernOwner, ...]
    legacy_owners: tuple[ConcernOwner, ...]
    simulation_owners: tuple[ConcernOwner, ...]
    unknown_owners: tuple[ConcernOwner, ...]
    blocker: OwnershipBlocker | None
    arbitration: FormalArbitration | None
    evidence: OwnershipEvidence
    schema: str = CONCERN_OWNERSHIP_SCHEMA
    version: int = CONCERN_OWNERSHIP_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=AuthorityGraphError)
        if schema != CONCERN_OWNERSHIP_SCHEMA:
            raise AuthorityGraphError("unexpected concern-ownership schema")
        version = _require_int(self.version, "version", error_type=AuthorityGraphError)
        if version != CONCERN_OWNERSHIP_VERSION:
            raise AuthorityGraphError("unexpected concern-ownership version")
        concern = _closed_enum(
            self.concern, ConcernKind, "concern", error_type=AuthorityGraphError
        )
        canonical = _optional_owner(self.canonical_owner)
        blocker = _optional_blocker(self.blocker)
        arbitration = _optional_arbitration(self.arbitration)
        if canonical is None and blocker is None:
            raise AuthorityGraphError(
                "each concern must have exactly one canonical owner or a typed blocker"
            )
        if canonical is not None and blocker is not None:
            raise AuthorityGraphError(
                "a concern cannot have both a canonical owner and a blocker"
            )
        if canonical is not None and canonical.disposition is not OwnerDisposition.CANONICAL:
            raise AuthorityGraphError("canonical owner disposition must be canonical")
        if blocker is not None and blocker.concern is not concern:
            raise AuthorityGraphError("blocker concern must match ownership concern")
        if arbitration is not None and arbitration.concern is not concern:
            raise AuthorityGraphError("arbitration concern must match ownership concern")
        if canonical is not None and arbitration is not None:
            if arbitration.canonical_owner_node_id != canonical.node_id:
                raise AuthorityGraphError(
                    "arbitration winner must match the canonical owner"
                )
        adapters = _owner_tuple(self.adapters, "adapters")
        projections = _owner_tuple(self.projections, "projections")
        legacy_owners = _owner_tuple(self.legacy_owners, "legacy_owners")
        simulation_owners = _owner_tuple(self.simulation_owners, "simulation_owners")
        unknown_owners = _owner_tuple(self.unknown_owners, "unknown_owners")
        for owner, expected in (
            (adapters, OwnerDisposition.ADAPTER),
            (projections, OwnerDisposition.PROJECTION),
            (legacy_owners, OwnerDisposition.LEGACY),
            (simulation_owners, OwnerDisposition.SIMULATION),
            (unknown_owners, OwnerDisposition.UNKNOWN),
        ):
            if any(item.disposition is not expected for item in owner):
                raise AuthorityGraphError(
                    f"{expected.value} owners must use disposition {expected.value}"
                )
        evidence = (
            self.evidence
            if isinstance(self.evidence, OwnershipEvidence)
            else OwnershipEvidence.from_mapping(self.evidence)
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "canonical_owner", canonical)
        object.__setattr__(self, "adapters", adapters)
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "legacy_owners", legacy_owners)
        object.__setattr__(self, "simulation_owners", simulation_owners)
        object.__setattr__(self, "unknown_owners", unknown_owners)
        object.__setattr__(self, "blocker", blocker)
        object.__setattr__(self, "arbitration", arbitration)
        object.__setattr__(self, "evidence", evidence)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=AuthorityGraphError,
                )
            )
            if claimed != identity:
                raise AuthorityGraphError("concern-ownership content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "adapters": [item.to_dict() for item in self.adapters],
            "arbitration": None if self.arbitration is None else self.arbitration.to_dict(),
            "blocker": None if self.blocker is None else self.blocker.to_dict(),
            "canonical_owner": (
                None if self.canonical_owner is None else self.canonical_owner.to_dict()
            ),
            "concern": self.concern.value,
            "evidence": self.evidence.to_dict(),
            "legacy_owners": [item.to_dict() for item in self.legacy_owners],
            "projections": [item.to_dict() for item in self.projections],
            "schema": self.schema,
            "simulation_owners": [item.to_dict() for item in self.simulation_owners],
            "unknown_owners": [item.to_dict() for item in self.unknown_owners],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise AuthorityGraphError("concern-ownership content identity mismatch")
        return {**payload, "content_identity": identity}

    @property
    def has_canonical_owner(self) -> bool:
        return self.canonical_owner is not None and self.blocker is None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConcernOwnership":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _OWNERSHIP_FIELDS)
        record = cls(
            concern=mapping["concern"],
            canonical_owner=mapping["canonical_owner"],
            adapters=mapping["adapters"],
            projections=mapping["projections"],
            legacy_owners=mapping["legacy_owners"],
            simulation_owners=mapping["simulation_owners"],
            unknown_owners=mapping["unknown_owners"],
            blocker=mapping["blocker"],
            arbitration=mapping["arbitration"],
            evidence=mapping["evidence"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise AuthorityGraphError("concern-ownership content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class AuthorityOwnershipGraph:
    """Reviewed ownership records for the closed initial concern set."""

    architecture_ir_identity: str
    repository_tree: str
    freshness: str
    concerns: tuple[ConcernOwnership, ...]
    schema: str = AUTHORITY_OWNERSHIP_SCHEMA
    version: int = AUTHORITY_OWNERSHIP_VERSION
    can_authorize_changes: bool = OWNERSHIP_GRAPH_CAN_AUTHORIZE_CHANGES
    can_transfer_authority: bool = OWNERSHIP_GRAPH_CAN_TRANSFER_AUTHORITY
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=AuthorityGraphError)
        if schema != AUTHORITY_OWNERSHIP_SCHEMA:
            raise AuthorityGraphError("unexpected authority-ownership schema")
        version = _require_int(self.version, "version", error_type=AuthorityGraphError)
        if version != AUTHORITY_OWNERSHIP_VERSION:
            raise AuthorityGraphError("unexpected authority-ownership version")
        if self.can_authorize_changes is not False:
            raise AuthorityGraphError(
                "authority ownership graph cannot authorize changes"
            )
        if self.can_transfer_authority is not True and self.can_transfer_authority is not False:
            raise AuthorityGraphError("can_transfer_authority must be a boolean")
        if self.can_transfer_authority is not False:
            raise AuthorityGraphError(
                "authority ownership graph cannot transfer authority"
            )
        architecture_ir_identity = _validate_dag_json_cid(
            _require_text(
                self.architecture_ir_identity,
                "architecture_ir_identity",
                error_type=AuthorityGraphError,
            )
        )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=AuthorityGraphError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=AuthorityGraphError
        )
        if isinstance(self.concerns, (str, bytes, bytearray)) or not isinstance(
            self.concerns, Sequence
        ):
            raise AuthorityGraphError("concerns must be a sequence")
        records = tuple(
            item
            if isinstance(item, ConcernOwnership)
            else ConcernOwnership.from_mapping(item)
            for item in self.concerns
        )
        by_kind = {item.concern: item for item in records}
        if len(by_kind) != len(records):
            raise AuthorityGraphError("concern ownership records must be unique")
        missing = [item.value for item in INITIAL_CONCERNS if item not in by_kind]
        extra = sorted(
            item.concern.value for item in records if item.concern not in set(INITIAL_CONCERNS)
        )
        if missing:
            raise AuthorityGraphError(f"missing initial concerns: {missing}")
        if extra:
            raise AuthorityGraphError(f"unsupported concerns: {extra}")
        ordered = tuple(by_kind[kind] for kind in INITIAL_CONCERNS)
        for record in ordered:
            if record.evidence.architecture_ir_identity != architecture_ir_identity:
                raise AuthorityGraphError(
                    "concern evidence architecture_ir_identity must match the graph"
                )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "concerns", ordered)
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "can_transfer_authority", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=AuthorityGraphError,
                )
            )
            if claimed != identity:
                raise AuthorityGraphError(
                    "authority-ownership content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "arbitrations": [item.to_dict() for item in self.arbitrations],
            "blockers": [item.to_dict() for item in self.blockers],
            "can_authorize_changes": False,
            "can_transfer_authority": False,
            "concerns": [item.to_dict() for item in self.concerns],
            "freshness": self.freshness,
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise AuthorityGraphError("authority-ownership content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def arbitrations(self) -> tuple[FormalArbitration, ...]:
        return tuple(
            item.arbitration for item in self.concerns if item.arbitration is not None
        )

    @property
    def blockers(self) -> tuple[OwnershipBlocker, ...]:
        return tuple(item.blocker for item in self.concerns if item.blocker is not None)

    @property
    def covers_initial_concerns(self) -> bool:
        return tuple(item.concern for item in self.concerns) == INITIAL_CONCERNS

    @property
    def fails_closed(self) -> bool:
        return bool(self.blockers)

    def ownership_for(self, concern: ConcernKind | str) -> ConcernOwnership:
        kind = _closed_enum(
            concern, ConcernKind, "concern", error_type=AuthorityGraphError
        )
        for record in self.concerns:
            if record.concern is kind:
                return record
        raise AuthorityGraphError(f"missing initial concern: {kind.value}")

    def canonical_owner(self, concern: ConcernKind | str) -> ConcernOwner:
        record = self.ownership_for(concern)
        if record.canonical_owner is None or record.blocker is not None:
            raise AuthorityGraphError(
                f"{record.concern.value} has no canonical owner"
            )
        return record.canonical_owner

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_ownership_authorization("change")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AuthorityOwnershipGraph":
        mapping = _require_mapping(payload, error_type=AuthorityGraphError)
        _require_fields(mapping, _GRAPH_FIELDS)
        graph = cls(
            architecture_ir_identity=mapping["architecture_ir_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            concerns=mapping["concerns"],
            schema=mapping["schema"],
            version=mapping["version"],
            can_authorize_changes=mapping["can_authorize_changes"],
            can_transfer_authority=mapping["can_transfer_authority"],
        )
        if mapping["content_identity"] != graph.content_identity:
            raise AuthorityGraphError("authority-ownership content identity mismatch")
        expected_arbitrations = [item.to_dict() for item in graph.arbitrations]
        expected_blockers = [item.to_dict() for item in graph.blockers]
        if mapping["arbitrations"] != expected_arbitrations:
            raise AuthorityGraphError("arbitrations projection mismatch")
        if mapping["blockers"] != expected_blockers:
            raise AuthorityGraphError("blockers projection mismatch")
        return graph

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "AuthorityOwnershipGraph":
        if type(payload) is not str or not payload:
            raise AuthorityGraphError(
                "authority-ownership JSON must be a nonempty string"
            )
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AuthorityGraphError("authority-ownership JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise AuthorityGraphError("authority-ownership JSON must contain an object")
        return cls.from_mapping(decoded)


def refuse_ownership_authorization(action: str) -> None:
    """Reject attempts to treat the ownership graph as change authority."""

    name = _require_text(action, "action", error_type=AuthorityGraphError)
    raise AuthorityGraphAuthorityError(
        f"authority ownership graph cannot authorize {name}"
    )


def refuse_authority_transfer(action: str) -> None:
    """Reject attempts to transfer an existing authority through this graph."""

    name = _require_text(action, "action", error_type=AuthorityGraphError)
    raise AuthorityGraphAuthorityError(
        f"authority ownership graph cannot {name} an existing authority"
    )


def lookup_owner_by_content_identity(*_args: Any, **_kwargs: Any) -> None:
    """Content identity never selects or proves a canonical owner."""

    raise AuthorityGraphError("content identity is not inferred to be authority")


def silently_select_canonical(*_args: Any, **_kwargs: Any) -> None:
    """Refuse first-listed, majority, or identity-sorted owner selection."""

    raise AuthorityGraphError("silent arbitration is prohibited")


@dataclass(frozen=True)
class _GraphView:
    architecture: ArchitectureIR
    nodes_by_id: dict[str, ArchitectureNode]
    edges_by_id: dict[str, ArchitectureEdge]
    outgoing: dict[str, tuple[ArchitectureEdge, ...]]
    incoming: dict[str, tuple[ArchitectureEdge, ...]]


def _build_view(architecture: ArchitectureIR) -> _GraphView:
    outgoing: dict[str, list[ArchitectureEdge]] = {
        node.node_id: [] for node in architecture.nodes
    }
    incoming: dict[str, list[ArchitectureEdge]] = {
        node.node_id: [] for node in architecture.nodes
    }
    for edge in architecture.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
    return _GraphView(
        architecture=architecture,
        nodes_by_id={node.node_id: node for node in architecture.nodes},
        edges_by_id={edge.edge_id: edge for edge in architecture.edges},
        outgoing={key: tuple(value) for key, value in outgoing.items()},
        incoming={key: tuple(value) for key, value in incoming.items()},
    )


def _related_edges(view: _GraphView, node_id: str) -> tuple[ArchitectureEdge, ...]:
    return view.outgoing.get(node_id, ()) + view.incoming.get(node_id, ())


def _claim_edges(
    view: _GraphView, claim: ConcernClaim
) -> tuple[ArchitectureEdge, ...]:
    if not claim.evidence_edge_ids:
        return _related_edges(view, claim.owner_node_id)
    missing = [
        edge_id
        for edge_id in claim.evidence_edge_ids
        if edge_id not in view.edges_by_id
    ]
    if missing:
        raise AuthorityGraphError(f"claim evidence edges are unknown: {missing}")
    selected = tuple(view.edges_by_id[edge_id] for edge_id in claim.evidence_edge_ids)
    for edge in selected:
        if claim.owner_node_id not in {edge.source, edge.target}:
            raise AuthorityGraphError(
                "claim evidence edges must incident the claimed owner"
            )
    return selected


def _owner_from_node(
    node: ArchitectureNode,
    disposition: OwnerDisposition,
    evidence_edge_ids: Iterable[str] = (),
) -> ConcernOwner:
    return ConcernOwner(
        node_id=node.node_id,
        disposition=disposition,
        node_kind=node.kind,
        provenance=node.provenance,
        evidence_edge_ids=tuple(evidence_edge_ids),
    )


def _qualify_canonical(
    view: _GraphView, claim: ConcernClaim
) -> OwnershipBlockerKind | None:
    node = view.nodes_by_id[claim.owner_node_id]
    if node.kind is NodeKind.SIMULATION:
        return OwnershipBlockerKind.SIMULATED_AS_LIVE
    if node.kind is not NodeKind.AUTHORITY:
        return OwnershipBlockerKind.NON_AUTHORITY_CANONICAL_CLAIM
    if node.provenance.confidence in NON_PROBATIVE_CONFIDENCE:
        return OwnershipBlockerKind.NON_PROBATIVE_OWNERSHIP
    edges = _claim_edges(view, claim)
    kinds = frozenset(edge.kind for edge in edges)
    if kinds and kinds <= _NON_PROBATIVE_OWNER_EDGE_KINDS:
        if EdgeKind.REEXPORTS in kinds:
            return OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY
        return OwnershipBlockerKind.NON_PROBATIVE_OWNERSHIP
    reexport_only = bool(edges) and all(
        edge.kind is EdgeKind.REEXPORTS for edge in edges
    )
    if reexport_only:
        return OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY
    if kinds and not (kinds & _PROBATIVE_OWNER_EDGE_KINDS) and EdgeKind.REEXPORTS in kinds:
        return OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY
    return None


def _blocker(
    concern: ConcernKind,
    kind: OwnershipBlockerKind,
    message: str,
    node_ids: Iterable[str] = (),
    edge_ids: Iterable[str] = (),
) -> OwnershipBlocker:
    return OwnershipBlocker(
        concern=concern,
        kind=kind,
        message=message,
        node_ids=tuple(node_ids),
        edge_ids=tuple(edge_ids),
    )


def _normalize_claims(
    claims: Sequence[ConcernClaim | Mapping[str, Any]] | None,
) -> tuple[ConcernClaim, ...]:
    if claims is None:
        return ()
    if isinstance(claims, (str, bytes, bytearray)) or not isinstance(claims, Sequence):
        raise AuthorityGraphError("claims must be a sequence")
    return tuple(
        item if isinstance(item, ConcernClaim) else ConcernClaim.from_mapping(item)
        for item in claims
    )


def _normalize_arbitrations(
    records: Sequence[FormalArbitration | Mapping[str, Any]] | None,
) -> dict[ConcernKind, FormalArbitration]:
    if records is None:
        return {}
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise AuthorityGraphError("arbitrations must be a sequence")
    parsed = tuple(
        item
        if isinstance(item, FormalArbitration)
        else FormalArbitration.from_mapping(item)
        for item in records
    )
    by_concern: dict[ConcernKind, FormalArbitration] = {}
    for record in parsed:
        if record.concern in by_concern:
            raise AuthorityGraphError(
                f"duplicate arbitration for {record.concern.value}"
            )
        by_concern[record.concern] = record
    return by_concern


def _arbitration_edge_kind(rationale: ArbitrationRationale) -> EdgeKind | None:
    if rationale is ArbitrationRationale.SUPERSEDES_EDGE:
        return EdgeKind.SUPERSEDES
    if rationale is ArbitrationRationale.DEPRECATES_EDGE:
        return EdgeKind.DEPRECATES
    if rationale is ArbitrationRationale.ADAPTS_EDGE:
        return EdgeKind.ADAPTS
    return None


def _arbitration_covers(
    view: _GraphView,
    arbitration: FormalArbitration,
    production_ids: frozenset[str],
) -> OwnershipBlocker | None:
    winner = arbitration.canonical_owner_node_id
    if winner not in production_ids:
        return _blocker(
            arbitration.concern,
            OwnershipBlockerKind.MISSING_ARBITRATION,
            "formal arbitration winner is not a production authority candidate",
            node_ids=(winner, *sorted(production_ids)),
        )
    uncovered = production_ids - {winner} - arbitration.loser_ids()
    if uncovered:
        return _blocker(
            arbitration.concern,
            OwnershipBlockerKind.UNCLASSIFIED_COMPETITOR,
            "formal arbitration left competing production authorities unclassified",
            node_ids=tuple(sorted(uncovered)),
        )
    extra = arbitration.loser_ids() - production_ids
    if extra:
        return _blocker(
            arbitration.concern,
            OwnershipBlockerKind.MISSING_ARBITRATION,
            "formal arbitration classifies nodes that are not production candidates",
            node_ids=tuple(sorted(extra)),
        )
    required_kind = _arbitration_edge_kind(arbitration.rationale)
    if required_kind is None:
        return None
    for loser in arbitration.loser_classifications:
        if required_kind is EdgeKind.ADAPTS:
            found = any(
                edge.kind is EdgeKind.ADAPTS
                and edge.source == loser.node_id
                and edge.target == winner
                for edge in view.outgoing.get(loser.node_id, ())
            )
        else:
            found = any(
                edge.kind is required_kind
                and edge.source == winner
                and edge.target == loser.node_id
                for edge in view.outgoing.get(winner, ())
            )
        if not found:
            return _blocker(
                arbitration.concern,
                OwnershipBlockerKind.MISSING_ARBITRATION,
                f"arbitration rationale {arbitration.rationale.value} lacks a matching edge",
                node_ids=(winner, loser.node_id),
            )
    return None


def _auto_non_owners(
    view: _GraphView,
    owner_ids: Iterable[str],
    occupied: set[str],
) -> dict[OwnerDisposition, list[tuple[ArchitectureNode, tuple[str, ...]]]]:
    owner_id_set = set(owner_ids)
    buckets: dict[OwnerDisposition, dict[str, tuple[ArchitectureNode, list[str]]]] = {
        OwnerDisposition.ADAPTER: {},
        OwnerDisposition.PROJECTION: {},
        OwnerDisposition.LEGACY: {},
        OwnerDisposition.SIMULATION: {},
    }
    assigned: dict[str, OwnerDisposition] = {}
    _priority = {
        OwnerDisposition.SIMULATION: 0,
        OwnerDisposition.LEGACY: 1,
        OwnerDisposition.PROJECTION: 2,
        OwnerDisposition.ADAPTER: 3,
    }

    def _add(
        node: ArchitectureNode,
        disposition: OwnerDisposition,
        edge_id: str,
    ) -> None:
        if node.node_id in occupied or node.node_id in owner_id_set:
            return
        current_disposition = assigned.get(node.node_id)
        if current_disposition is not None and current_disposition is not disposition:
            if _priority[disposition] >= _priority[current_disposition]:
                if current_disposition in buckets and node.node_id in buckets[current_disposition]:
                    _node, edge_ids = buckets[current_disposition][node.node_id]
                    if edge_id not in edge_ids:
                        edge_ids.append(edge_id)
                return
            del buckets[current_disposition][node.node_id]
        bucket = buckets[disposition]
        current = bucket.get(node.node_id)
        if current is None:
            bucket[node.node_id] = (node, [edge_id])
        elif edge_id not in current[1]:
            current[1].append(edge_id)
        assigned[node.node_id] = disposition

    for owner_id in owner_id_set:
        for edge in view.outgoing.get(owner_id, ()):
            target = view.nodes_by_id[edge.target]
            if edge.kind is EdgeKind.GENERATES:
                _add(target, OwnerDisposition.PROJECTION, edge.edge_id)
            elif edge.kind in {EdgeKind.SUPERSEDES, EdgeKind.DEPRECATES}:
                _add(target, OwnerDisposition.LEGACY, edge.edge_id)
            elif edge.kind is EdgeKind.FALLBACKS_TO:
                _add(target, OwnerDisposition.SIMULATION, edge.edge_id)
            elif edge.kind is EdgeKind.ADAPTS:
                _add(target, OwnerDisposition.ADAPTER, edge.edge_id)
        for edge in view.incoming.get(owner_id, ()):
            source = view.nodes_by_id[edge.source]
            if edge.kind in {EdgeKind.ADAPTS, EdgeKind.REEXPORTS}:
                _add(source, OwnerDisposition.ADAPTER, edge.edge_id)
            elif edge.kind is EdgeKind.GENERATES:
                _add(source, OwnerDisposition.PROJECTION, edge.edge_id)
            elif edge.kind is EdgeKind.FALLBACKS_TO and source.kind is NodeKind.SIMULATION:
                _add(source, OwnerDisposition.SIMULATION, edge.edge_id)
        node = view.nodes_by_id.get(owner_id)
        if node is None:
            continue
        for edge in _related_edges(view, owner_id):
            other_id = edge.target if edge.source == owner_id else edge.source
            other = view.nodes_by_id[other_id]
            if other.kind is NodeKind.COMPATIBILITY:
                _add(other, OwnerDisposition.ADAPTER, edge.edge_id)
            elif other.kind is NodeKind.GENERATED:
                _add(other, OwnerDisposition.PROJECTION, edge.edge_id)
            elif other.kind is NodeKind.SIMULATION:
                _add(other, OwnerDisposition.SIMULATION, edge.edge_id)
    return {
        disposition: [
            (node, tuple(sorted(set(edge_ids))))
            for node, edge_ids in sorted(bucket.values(), key=lambda item: item[0].node_id)
        ]
        for disposition, bucket in buckets.items()
    }


def _merge_owners(
    claimed: Sequence[ConcernOwner],
    discovered: Sequence[tuple[ArchitectureNode, tuple[str, ...]]],
    disposition: OwnerDisposition,
) -> tuple[ConcernOwner, ...]:
    by_id = {owner.node_id: owner for owner in claimed}
    for node, edge_ids in discovered:
        if node.node_id in by_id:
            continue
        by_id[node.node_id] = _owner_from_node(node, disposition, edge_ids)
    return tuple(sorted(by_id.values(), key=lambda item: item.node_id))


def _evidence_for(
    architecture: ArchitectureIR,
    owners: Sequence[ConcernOwner],
    extra_node_ids: Iterable[str] = (),
    extra_edge_ids: Iterable[str] = (),
    arbitration: FormalArbitration | None = None,
    blocker: OwnershipBlocker | None = None,
) -> OwnershipEvidence:
    node_ids = [owner.node_id for owner in owners]
    node_ids.extend(extra_node_ids)
    edge_ids = [edge_id for owner in owners for edge_id in owner.evidence_edge_ids]
    edge_ids.extend(extra_edge_ids)
    if arbitration is not None:
        node_ids.extend(arbitration.evidence_node_ids)
        node_ids.append(arbitration.canonical_owner_node_id)
        node_ids.extend(sorted(arbitration.loser_ids()))
        edge_ids.extend(arbitration.evidence_edge_ids)
    if blocker is not None:
        node_ids.extend(blocker.node_ids)
        edge_ids.extend(blocker.edge_ids)
    return OwnershipEvidence(
        architecture_ir_identity=architecture.content_identity,
        extractor_identity=EXTRACTOR_IDENTITY,
        node_ids=tuple(node_ids),
        edge_ids=tuple(edge_ids),
    )


def _resolve_concern(
    view: _GraphView,
    concern: ConcernKind,
    claims: Sequence[ConcernClaim],
    arbitration: FormalArbitration | None,
) -> ConcernOwnership:
    by_node: dict[str, list[ConcernClaim]] = {}
    for claim in claims:
        if claim.owner_node_id not in view.nodes_by_id:
            raise AuthorityGraphError(
                f"claim owner {claim.owner_node_id!r} is not in ArchitectureIR"
            )
        by_node.setdefault(claim.owner_node_id, []).append(claim)

    conflict_nodes = [
        node_id
        for node_id, node_claims in by_node.items()
        if len({item.disposition for item in node_claims}) > 1
    ]
    claimed_owners: dict[OwnerDisposition, list[ConcernOwner]] = {
        disposition: [] for disposition in OwnerDisposition
    }
    qualification_failures: list[tuple[ConcernClaim, OwnershipBlockerKind]] = []
    for node_id, node_claims in sorted(by_node.items()):
        node = view.nodes_by_id[node_id]
        dispositions = {item.disposition for item in node_claims}
        if len(dispositions) > 1:
            continue
        claim = node_claims[0]
        merged_edges = sorted(
            {
                edge_id
                for item in node_claims
                for edge_id in (
                    item.evidence_edge_ids or tuple(edge.edge_id for edge in _claim_edges(view, item))
                )
            }
        )
        if claim.disposition is OwnerDisposition.CANONICAL:
            failure = _qualify_canonical(view, claim)
            if failure is not None:
                qualification_failures.append((claim, failure))
                fallback = {
                    OwnershipBlockerKind.SIMULATED_AS_LIVE: OwnerDisposition.SIMULATION,
                    OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY: OwnerDisposition.ADAPTER,
                }.get(failure)
                if failure is OwnershipBlockerKind.NON_AUTHORITY_CANONICAL_CLAIM:
                    if node.kind is NodeKind.SIMULATION:
                        fallback = OwnerDisposition.SIMULATION
                    elif node.kind is NodeKind.GENERATED:
                        fallback = OwnerDisposition.PROJECTION
                    elif node.kind is NodeKind.COMPATIBILITY:
                        fallback = OwnerDisposition.ADAPTER
                if fallback is not None:
                    claimed_owners[fallback].append(
                        _owner_from_node(node, fallback, merged_edges)
                    )
                continue
        claimed_owners[claim.disposition].append(
            _owner_from_node(node, claim.disposition, merged_edges)
        )

    production = tuple(claimed_owners[OwnerDisposition.CANONICAL])
    unknown = tuple(claimed_owners[OwnerDisposition.UNKNOWN])
    occupied = {owner.node_id for owners in claimed_owners.values() for owner in owners}
    occupied.update(conflict_nodes)
    seed_ids = [owner.node_id for owner in production] or list(occupied)
    discovered = _auto_non_owners(view, seed_ids, occupied)
    adapters = _merge_owners(
        claimed_owners[OwnerDisposition.ADAPTER],
        discovered[OwnerDisposition.ADAPTER],
        OwnerDisposition.ADAPTER,
    )
    projections = _merge_owners(
        claimed_owners[OwnerDisposition.PROJECTION],
        discovered[OwnerDisposition.PROJECTION],
        OwnerDisposition.PROJECTION,
    )
    legacy_owners = _merge_owners(
        claimed_owners[OwnerDisposition.LEGACY],
        discovered[OwnerDisposition.LEGACY],
        OwnerDisposition.LEGACY,
    )
    simulation_owners = _merge_owners(
        claimed_owners[OwnerDisposition.SIMULATION],
        discovered[OwnerDisposition.SIMULATION],
        OwnerDisposition.SIMULATION,
    )
    unknown_owners = tuple(
        sorted(unknown, key=lambda item: item.node_id)
    )

    blocker: OwnershipBlocker | None = None
    canonical: ConcernOwner | None = None
    applied_arbitration: FormalArbitration | None = None

    if conflict_nodes:
        blocker = _blocker(
            concern,
            OwnershipBlockerKind.CONFLICTING_DISPOSITION,
            "the same node was claimed with conflicting owner dispositions",
            node_ids=conflict_nodes,
        )
    elif unknown_owners:
        blocker = _blocker(
            concern,
            OwnershipBlockerKind.UNKNOWN_PRODUCTION_OWNER
            if production
            else OwnershipBlockerKind.UNKNOWN_OWNER,
            "unknown production ownership fails closed",
            node_ids=[owner.node_id for owner in unknown_owners],
        )
    elif qualification_failures:
        _claim, kind = qualification_failures[0]
        messages = {
            OwnershipBlockerKind.SIMULATED_AS_LIVE: (
                "simulation paths cannot be canonical production owners"
            ),
            OwnershipBlockerKind.NON_AUTHORITY_CANONICAL_CLAIM: (
                "canonical owners must be ArchitectureIR authority nodes"
            ),
            OwnershipBlockerKind.NON_PROBATIVE_OWNERSHIP: (
                "heuristic or opaque facts cannot prove ownership"
            ),
            OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY: (
                "re-export is not authority"
            ),
        }
        blocker = _blocker(
            concern,
            kind,
            messages[kind],
            node_ids=[item.owner_node_id for item, _ in qualification_failures],
            edge_ids=[
                edge_id
                for item, _ in qualification_failures
                for edge_id in item.evidence_edge_ids
            ],
        )
    elif not production:
        blocker = _blocker(
            concern,
            OwnershipBlockerKind.UNKNOWN_OWNER,
            "no evidence-backed canonical owner was claimed",
        )
    elif len(production) == 1:
        canonical = production[0]
    else:
        production_ids = frozenset(owner.node_id for owner in production)
        if arbitration is None:
            blocker = _blocker(
                concern,
                OwnershipBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES,
                "multiple production authorities require formal arbitration",
                node_ids=sorted(production_ids),
            )
        else:
            coverage = _arbitration_covers(view, arbitration, production_ids)
            if coverage is not None:
                blocker = coverage
            else:
                winner = next(
                    owner
                    for owner in production
                    if owner.node_id == arbitration.canonical_owner_node_id
                )
                canonical = winner
                applied_arbitration = arbitration
                loser_by_id = {
                    item.node_id: item.disposition
                    for item in arbitration.loser_classifications
                }
                remaining = [
                    owner
                    for owner in production
                    if owner.node_id != winner.node_id
                ]
                for owner in remaining:
                    disposition = loser_by_id[owner.node_id]
                    reclassed = ConcernOwner(
                        node_id=owner.node_id,
                        disposition=disposition,
                        node_kind=owner.node_kind,
                        provenance=owner.provenance,
                        evidence_edge_ids=owner.evidence_edge_ids,
                    )
                    if disposition is OwnerDisposition.ADAPTER:
                        adapters = _owner_tuple((*adapters, reclassed), "adapters")
                    elif disposition is OwnerDisposition.PROJECTION:
                        projections = _owner_tuple(
                            (*projections, reclassed), "projections"
                        )
                    elif disposition is OwnerDisposition.LEGACY:
                        legacy_owners = _owner_tuple(
                            (*legacy_owners, reclassed), "legacy_owners"
                        )
                    elif disposition is OwnerDisposition.SIMULATION:
                        simulation_owners = _owner_tuple(
                            (*simulation_owners, reclassed), "simulation_owners"
                        )
                    else:
                        unknown_owners = _owner_tuple(
                            (*unknown_owners, reclassed), "unknown_owners"
                        )
                        blocker = _blocker(
                            concern,
                            OwnershipBlockerKind.UNKNOWN_PRODUCTION_OWNER,
                            "arbitrated competitor classified as unknown fails closed",
                            node_ids=(owner.node_id,),
                        )
                        canonical = None
                        applied_arbitration = None
                        break

    owners = []
    if canonical is not None:
        owners.append(canonical)
    owners.extend(adapters)
    owners.extend(projections)
    owners.extend(legacy_owners)
    owners.extend(simulation_owners)
    owners.extend(unknown_owners)
    evidence = _evidence_for(
        view.architecture,
        owners,
        extra_node_ids=conflict_nodes,
        arbitration=applied_arbitration,
        blocker=blocker,
    )
    return ConcernOwnership(
        concern=concern,
        canonical_owner=canonical,
        adapters=adapters,
        projections=projections,
        legacy_owners=legacy_owners,
        simulation_owners=simulation_owners,
        unknown_owners=unknown_owners,
        blocker=blocker,
        arbitration=applied_arbitration,
        evidence=evidence,
    )


def resolve_authority_ownership(
    graph: ArchitectureIR | Mapping[str, Any],
    claims: Sequence[ConcernClaim | Mapping[str, Any]] | None = None,
    arbitrations: Sequence[FormalArbitration | Mapping[str, Any]] | None = None,
) -> AuthorityOwnershipGraph:
    """Resolve each initial concern from source-bound ArchitectureIR facts."""

    architecture = _require_architecture_ir(graph)
    view = _build_view(architecture)
    parsed_claims = _normalize_claims(claims)
    by_concern: dict[ConcernKind, list[ConcernClaim]] = {
        kind: [] for kind in INITIAL_CONCERNS
    }
    for claim in parsed_claims:
        by_concern[claim.concern].append(claim)
    parsed_arbitrations = _normalize_arbitrations(arbitrations)
    records = tuple(
        _resolve_concern(
            view,
            concern,
            by_concern[concern],
            parsed_arbitrations.get(concern),
        )
        for concern in INITIAL_CONCERNS
    )
    return AuthorityOwnershipGraph(
        architecture_ir_identity=architecture.content_identity,
        repository_tree=architecture.repository_tree,
        freshness=architecture.freshness,
        concerns=records,
    )


build_authority_ownership_graph = resolve_authority_ownership


def canonical_owners(
    graph: AuthorityOwnershipGraph,
) -> tuple[tuple[str, str], ...]:
    """Return ``(concern, node_id)`` pairs for resolved canonical owners."""

    return tuple(
        (item.concern.value, item.canonical_owner.node_id)
        for item in graph.concerns
        if item.canonical_owner is not None
    )


__all__ = [
    "AUTHORITY_OWNERSHIP_EVIDENCE",
    "AUTHORITY_OWNERSHIP_SCHEMA",
    "AUTHORITY_OWNERSHIP_VERSION",
    "ArbitrationRationale",
    "AuthorityGraphAuthorityError",
    "AuthorityGraphError",
    "AuthorityOwnershipGraph",
    "CLOSED_ARBITRATION_RATIONALES",
    "CLOSED_CONCERNS",
    "CLOSED_OWNERSHIP_BLOCKERS",
    "CLOSED_OWNER_DISPOSITIONS",
    "CONTENT_IDENTITY_IS_NOT_AUTHORITY",
    "CONCERN_OWNERSHIP_SCHEMA",
    "CONCERN_OWNERSHIP_VERSION",
    "ConcernClaim",
    "ConcernKind",
    "ConcernOwner",
    "ConcernOwnership",
    "ConcernSourceBinding",
    "DEFAULT_FRESHNESS",
    "EXTRACTOR_IDENTITY",
    "FORMAL_ARBITRATION_SCHEMA",
    "FORMAL_ARBITRATION_VERSION",
    "FormalArbitration",
    "INITIAL_CONCERNS",
    "INITIAL_CONCERN_SOURCE_BINDINGS",
    "LoserClassification",
    "NON_CANONICAL_DISPOSITIONS",
    "OWNERSHIP_BLOCKER_SCHEMA",
    "OWNERSHIP_BLOCKER_VERSION",
    "OWNERSHIP_GRAPH_CAN_AUTHORIZE_CHANGES",
    "OWNERSHIP_GRAPH_CAN_TRANSFER_AUTHORITY",
    "OwnerDisposition",
    "OwnershipBlocker",
    "OwnershipBlockerKind",
    "OwnershipEvidence",
    "REEXPORT_IS_NOT_AUTHORITY",
    "SILENT_ARBITRATION_PROHIBITED",
    "TASK_ID",
    "build_authority_ownership_graph",
    "canonical_owners",
    "lookup_owner_by_content_identity",
    "refuse_authority_transfer",
    "refuse_ownership_authorization",
    "resolve_authority_ownership",
    "silently_select_canonical",
]
