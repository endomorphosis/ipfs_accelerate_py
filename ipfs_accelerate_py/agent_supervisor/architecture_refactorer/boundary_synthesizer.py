"""Interface-boundary synthesis around coherent authorities (PCAR-011).

`InterfaceBoundarySynthesizer` proposes smaller stable interfaces for the
closed initial boundary-concern set. Each proposal names its interface,
canonical owner, callers, effects, state owner, migration adapters,
deprecations, tests, proofs, rollback, and predicted context/cone
reductions. Hard constraints are checked independently and cannot be
compensated by a lower cost vector. Ranking inputs are deterministic and
non-probative. Proposals remain candidate-tier and cannot apply, transfer,
create, or promote authority.
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
from .authority_graph import AuthorityOwnershipGraph, ConcernKind
from .contract_extractor import ContractExtractionResult
from .contracts import (
    ArchitectureContractError,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
    NON_PROBATIVE_CONFIDENCE,
)
from .entropy import NON_COMPENSABLE_INVARIANTS, SemanticEntropyReport

BOUNDARY_PROPOSAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-proposal@1"
)
BOUNDARY_PROPOSAL_VERSION = 1
BOUNDARY_PROPOSAL_EVIDENCE = "pcar/boundary-proposal@1"
BOUNDARY_COST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-cost@1"
)
BOUNDARY_COST_VERSION = 1
BOUNDARY_COST_VECTOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-cost-vector@1"
)
BOUNDARY_COST_VECTOR_VERSION = 1
HARD_CONSTRAINT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-hard-constraint@1"
)
HARD_CONSTRAINT_VERSION = 1
PROPOSED_INTERFACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-interface@1"
)
PROPOSED_INTERFACE_VERSION = 1
RANKING_INPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-ranking-inputs@1"
)
RANKING_INPUT_VERSION = 1
BOUNDARY_SYNTHESIS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-synthesis-result@1"
)
BOUNDARY_SYNTHESIS_VERSION = 1
BOUNDARY_SYNTHESIS_EVIDENCE = "pcar/boundary-proposal@1"
EXTRACTOR_IDENTITY = "pcar-011-interface-boundary-synthesizer"
TASK_ID = "PCAR-011"
DEFAULT_FRESHNESS = "pcar-011-boundary-synthesis"
EFFECT_CLASS = "read_only_planning"
SYNTHESIZER_CAN_APPLY_CANDIDATE = False
SYNTHESIZER_CAN_TRANSFER_AUTHORITY = False
SYNTHESIZER_CAN_CREATE_AUTHORITY = False
SYNTHESIZER_CAN_PROMOTE_CANDIDATE = False
SYNTHESIZER_CAN_AUTHORIZE_CHANGES = False
CANDIDATE_TIER_ONLY = True
RANKING_IS_NON_PROBATIVE = True
COMPLETION_AUTHORITATIVE = False
ROLLBACK_DECLARATION = (
    "revert proposed interface; candidates have no applied effects"
)
MAX_BOUNDARY_MEMBERS = 32

_UNKNOWN_FIELD_MESSAGE = "unknown boundary-proposal field"
_MISSING_FIELD_MESSAGE = "missing boundary-proposal field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_CROSS_REPO_PREFIXES = (
    "ipfs_datasets_py/",
    "ipfs_kit_py/",
    "ipfs_accelerate_py/mcplusplus/",
)
_SECRET_MARKERS = (
    ".env",
    "secret",
    "credential",
    "private_key",
    "id_rsa",
    "password",
)
_PRODUCTION_NODE_KINDS = frozenset(
    {
        NodeKind.OPERATION,
        NodeKind.ENTRYPOINT,
        NodeKind.PROVIDER,
        NodeKind.AUTHORITY,
        NodeKind.POLICY,
        NodeKind.STATE,
        NodeKind.RECEIPT,
        NodeKind.PROOF,
        NodeKind.INTERFACE,
        NodeKind.SYMBOL,
        NodeKind.MODULE,
        NodeKind.SCHEMA,
    }
)
_IMPLEMENTATION_NODE_KINDS = frozenset(
    {NodeKind.SYMBOL, NodeKind.MODULE, NodeKind.FILE, NodeKind.OPERATION}
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
_EFFECT_EDGE_KINDS = frozenset(
    {
        EdgeKind.READS,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.OBSERVES,
        EdgeKind.EXECUTES,
        EdgeKind.PERSISTS,
        EdgeKind.GENERATES,
    }
)
_CALL_EDGE_KINDS = frozenset(
    {
        EdgeKind.CALLS,
        EdgeKind.IMPORTS,
        EdgeKind.CONSTRUCTS,
        EdgeKind.IMPLEMENTS,
        EdgeKind.EXECUTES,
        EdgeKind.FALLBACKS_TO,
    }
)
_MUTABLE_EDGE_KINDS = frozenset(
    {EdgeKind.WRITES, EdgeKind.MUTATES, EdgeKind.PERSISTS}
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
_EXPAND_EDGE_KINDS = frozenset(
    {
        EdgeKind.CONTAINS,
        EdgeKind.CALLS,
        EdgeKind.IMPLEMENTS,
        EdgeKind.CONSTRUCTS,
        EdgeKind.ADAPTS,
        EdgeKind.REEXPORTS,
        EdgeKind.TESTS,
        EdgeKind.PROVES,
    }
)
_EXPAND_NODE_KINDS = frozenset(
    {
        NodeKind.SYMBOL,
        NodeKind.MODULE,
        NodeKind.FILE,
        NodeKind.INTERFACE,
        NodeKind.EFFECT,
        NodeKind.SCHEMA,
        NodeKind.OPERATION,
        NodeKind.ENTRYPOINT,
        NodeKind.RECEIPT,
        NodeKind.TEST,
        NodeKind.PROOF,
        NodeKind.POLICY,
        NodeKind.STATE,
        NodeKind.PROVIDER,
        NodeKind.COMPATIBILITY,
        NodeKind.SIMULATION,
        NodeKind.ARTIFACT,
        NodeKind.GENERATED,
    }
)
_PUBLIC_NODE_KINDS = frozenset(
    {NodeKind.INTERFACE, NodeKind.ENTRYPOINT, NodeKind.PROVIDER}
)
_QUARANTINE_NODE_KINDS = frozenset({NodeKind.SIMULATION, NodeKind.COMPATIBILITY})


class BoundarySynthesizerError(ArchitectureContractError):
    """Fail-closed interface-boundary synthesis error."""


class BoundarySynthesizerAuthorityError(BoundarySynthesizerError):
    """Raised when synthesis is asked to apply, transfer, or promote."""


class BoundaryConcern(str, Enum):
    """Closed initial boundary-concern vocabulary (PCAR-PLAN-R1)."""

    PROVIDER_CAPABILITY_SELECTION = "provider capability/selection"
    EXECUTION_REQUESTS_OUTCOMES = "execution requests/outcomes"
    ANALYSIS_CONTEXT = "analysis/context"
    PROOF_VERIFICATION_SCHEDULING = "proof and verification scheduling"
    TASK_OBJECTIVE_STATE = "task/objective state"
    CONTROL_OPERATIONS = "control operations"
    RECEIPT_EVIDENCE_QUERIES = "receipt/evidence queries"
    LEGACY_COMPATIBILITY = "legacy compatibility"
    SIMULATIONS = "simulations"


INITIAL_BOUNDARY_CONCERNS: tuple[BoundaryConcern, ...] = tuple(BoundaryConcern)
CLOSED_BOUNDARY_CONCERNS: frozenset[str] = frozenset(
    item.value for item in BoundaryConcern
)
_QUARANTINE_CONCERN_BY_KIND: dict[NodeKind, BoundaryConcern] = {
    NodeKind.SIMULATION: BoundaryConcern.SIMULATIONS,
    NodeKind.COMPATIBILITY: BoundaryConcern.LEGACY_COMPATIBILITY,
}


class ProposalDisposition(str, Enum):
    """Closed admission vocabulary for one boundary proposal."""

    ADMITTED = "admitted"
    REJECTED = "rejected"


CLOSED_PROPOSAL_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in ProposalDisposition
)


class ProposalTier(str, Enum):
    """Closed promotion vocabulary. Only candidate-tier is admitted."""

    CANDIDATE = "candidate"


CLOSED_PROPOSAL_TIERS: frozenset[str] = frozenset(item.value for item in ProposalTier)


class HardConstraintKind(str, Enum):
    """Closed hard-constraint vocabulary preserved by every proposal."""

    NO_AUTHORITY_WEAKENING = "NoAuthorityWeakening"
    NO_EFFECT_EXPANSION = "NoEffectExpansion"
    NO_HIDDEN_BEHAVIOR_CHANGE = "NoHiddenBehaviorChange"
    NO_SIMULATED_AS_LIVE = "NoSimulatedAsLive"
    NO_VALIDATION_REDUCTION = "NoValidationReduction"
    NO_PROOF_OBLIGATION_LOSS = "NoProofObligationLoss"
    NO_PUBLIC_CONTRACT_BREAK_WITHOUT_VERSIONED_MIGRATION = (
        "NoPublicContractBreakWithoutVersionedMigration"
    )
    NO_STALE_EVIDENCE_PROMOTION = "NoStaleEvidencePromotion"
    NO_UNBOUNDED_REFACTOR = "NoUnboundedRefactor"
    NO_PROCEDURE_SELF_AUTHORIZATION = "NoProcedureSelfAuthorization"
    NO_ARCHITECTURE_CANDIDATE_SELF_PROMOTION = "NoArchitectureCandidateSelfPromotion"
    NO_CROSS_REPOSITORY_WRITE = "NoCrossRepositoryWrite"
    NO_SECRET_OR_PRIVATE_DATA_LEAK = "NoSecretOrPrivateDataLeak"
    NO_FALSE_COMPLETION = "NoFalseCompletion"
    NO_UNRESOLVED_AMBIGUITY = "NoUnresolvedAmbiguity"
    NO_UNRESOLVED_AUTHORITY = "NoUnresolvedAuthority"
    NO_INCOHERENT_AUTHORITY = "NoIncoherentAuthority"
    NO_UNRESOLVED_STATE_MOVEMENT = "NoUnresolvedStateMovement"
    NO_BOUNDARY_CYCLE = "NoBoundaryCycle"
    NO_CROSS_BOUNDARY_MUTABLE_SHARING = "NoCrossBoundaryMutableSharing"
    NO_ROLLBACK_LOSS = "NoRollbackLoss"
    NO_AUTONOMOUS_APPLICATION = "NoAutonomousApplication"


REQUIRED_HARD_CONSTRAINTS: tuple[HardConstraintKind, ...] = tuple(HardConstraintKind)
CLOSED_HARD_CONSTRAINTS: frozenset[str] = frozenset(
    item.value for item in HardConstraintKind
)


class BoundaryCostDimension(str, Enum):
    """Closed cost dimensions minimized by boundary synthesis."""

    CROSS_BOUNDARY_EFFECTS = "cross_boundary_effects"
    MUTABLE_SHARING = "mutable_sharing"
    CYCLES = "cycles"
    PUBLIC_SYMBOLS = "public_symbols"
    CHANGE_AMPLIFICATION = "change_amplification"
    CONTEXT_AMPLIFICATION = "context_amplification"
    VALIDATION_AMPLIFICATION = "validation_amplification"
    DEPENDENCY_CONE = "dependency_cone"


REQUIRED_COST_DIMENSIONS: tuple[BoundaryCostDimension, ...] = tuple(
    BoundaryCostDimension
)
CLOSED_COST_DIMENSIONS: frozenset[str] = frozenset(
    item.value for item in BoundaryCostDimension
)

REQUIRED_PROPOSAL_DECLARATIONS: tuple[str, ...] = (
    "allowed_callers",
    "allowed_effects",
    "canonical_owner",
    "cost",
    "deprecated_paths",
    "deprecations",
    "hard_constraints",
    "migration_adapters",
    "predicted_cone_reduction",
    "predicted_context_reduction",
    "proofs",
    "required_interface",
    "rollback",
    "state_owner",
    "tests",
)

_INTERFACE_NAMES: dict[BoundaryConcern, str] = {
    BoundaryConcern.PROVIDER_CAPABILITY_SELECTION: "provider.capability_selection",
    BoundaryConcern.EXECUTION_REQUESTS_OUTCOMES: "execution.request_outcome",
    BoundaryConcern.ANALYSIS_CONTEXT: "analysis.context",
    BoundaryConcern.PROOF_VERIFICATION_SCHEDULING: "proof.verification_schedule",
    BoundaryConcern.TASK_OBJECTIVE_STATE: "task.objective_state",
    BoundaryConcern.CONTROL_OPERATIONS: "control.operations",
    BoundaryConcern.RECEIPT_EVIDENCE_QUERIES: "receipt.evidence_query",
    BoundaryConcern.LEGACY_COMPATIBILITY: "legacy.compatibility",
    BoundaryConcern.SIMULATIONS: "simulation.quarantine",
}
_PRIMARY_KIND_CONCERNS: dict[NodeKind, tuple[BoundaryConcern, ...]] = {
    NodeKind.PROVIDER: (BoundaryConcern.PROVIDER_CAPABILITY_SELECTION,),
    NodeKind.PROOF: (BoundaryConcern.PROOF_VERIFICATION_SCHEDULING,),
    NodeKind.STATE: (BoundaryConcern.TASK_OBJECTIVE_STATE,),
    NodeKind.COMPATIBILITY: (BoundaryConcern.LEGACY_COMPATIBILITY,),
    NodeKind.SIMULATION: (BoundaryConcern.SIMULATIONS,),
    NodeKind.POLICY: (BoundaryConcern.CONTROL_OPERATIONS,),
    NodeKind.OPERATION: (
        BoundaryConcern.CONTROL_OPERATIONS,
        BoundaryConcern.EXECUTION_REQUESTS_OUTCOMES,
    ),
    NodeKind.ENTRYPOINT: (BoundaryConcern.CONTROL_OPERATIONS,),
    NodeKind.RECEIPT: (
        BoundaryConcern.RECEIPT_EVIDENCE_QUERIES,
        BoundaryConcern.EXECUTION_REQUESTS_OUTCOMES,
    ),
    NodeKind.TEST: (
        BoundaryConcern.PROOF_VERIFICATION_SCHEDULING,
        BoundaryConcern.RECEIPT_EVIDENCE_QUERIES,
    ),
    NodeKind.SCHEMA: (BoundaryConcern.ANALYSIS_CONTEXT,),
    NodeKind.MODULE: (BoundaryConcern.ANALYSIS_CONTEXT,),
    NodeKind.ARTIFACT: (BoundaryConcern.ANALYSIS_CONTEXT,),
    NodeKind.GENERATED: (BoundaryConcern.ANALYSIS_CONTEXT,),
}
_PATH_MARKERS: dict[BoundaryConcern, tuple[str, ...]] = {
    BoundaryConcern.PROVIDER_CAPABILITY_SELECTION: (
        "provider",
        "capability",
    ),
    BoundaryConcern.EXECUTION_REQUESTS_OUTCOMES: (
        "execution",
        "invocation",
    ),
    BoundaryConcern.ANALYSIS_CONTEXT: (
        "context",
        "analysis",
        "compiler",
        "entropy",
        "graph_builder",
        "semantic",
    ),
    BoundaryConcern.PROOF_VERIFICATION_SCHEDULING: (
        "/proof/",
        "verification",
    ),
    BoundaryConcern.TASK_OBJECTIVE_STATE: (
        "task_sources",
        "objective",
        "duckdb_state",
        "lease_coordination",
    ),
    BoundaryConcern.CONTROL_OPERATIONS: (
        "/control/",
        "authorization",
        "entrypoint",
    ),
    BoundaryConcern.RECEIPT_EVIDENCE_QUERIES: (
        "receipt",
        "authoritative_completion",
        "test_execution",
    ),
    BoundaryConcern.LEGACY_COMPATIBILITY: ("legacy", "compat"),
    BoundaryConcern.SIMULATIONS: ("simulation", "simulat"),
}
_OWNERSHIP_CONCERNS: dict[BoundaryConcern, tuple[ConcernKind, ...]] = {
    BoundaryConcern.PROVIDER_CAPABILITY_SELECTION: (
        ConcernKind.PROVIDER_CAPABILITY,
        ConcernKind.PROVIDER_SELECTION,
    ),
    BoundaryConcern.EXECUTION_REQUESTS_OUTCOMES: (ConcernKind.EXECUTION_RESULT,),
    BoundaryConcern.ANALYSIS_CONTEXT: (),
    BoundaryConcern.PROOF_VERIFICATION_SCHEDULING: (
        ConcernKind.PROOF_VERIFICATION,
        ConcernKind.TEST_EVIDENCE,
    ),
    BoundaryConcern.TASK_OBJECTIVE_STATE: (
        ConcernKind.TASK_IDENTITY,
        ConcernKind.OBJECTIVE_IDENTITY,
        ConcernKind.STATE_PERSISTENCE,
    ),
    BoundaryConcern.CONTROL_OPERATIONS: (
        ConcernKind.OPERATION_IDENTITY,
        ConcernKind.AUTHORIZATION,
        ConcernKind.CONFIRMATION,
        ConcernKind.POLICY_DECISION,
    ),
    BoundaryConcern.RECEIPT_EVIDENCE_QUERIES: (
        ConcernKind.COMPLETION_EVIDENCE,
        ConcernKind.TEST_EVIDENCE,
    ),
    BoundaryConcern.LEGACY_COMPATIBILITY: (),
    BoundaryConcern.SIMULATIONS: (),
}
_COST_VECTOR_FIELDS = frozenset(
    {
        "change_amplification",
        "content_identity",
        "context_amplification",
        "cross_boundary_effects",
        "cycles",
        "dependency_cone",
        "mutable_sharing",
        "public_symbols",
        "schema",
        "validation_amplification",
        "version",
    }
)
_COST_FIELDS = frozenset(
    {
        "after",
        "before",
        "content_identity",
        "reductions",
        "schema",
        "version",
    }
)
_CONSTRAINT_FIELDS = frozenset(
    {
        "constraint",
        "content_identity",
        "evidence_edge_ids",
        "evidence_node_ids",
        "explanation",
        "satisfied",
        "schema",
        "version",
    }
)
_INTERFACE_FIELDS = frozenset(
    {
        "concern",
        "content_identity",
        "members",
        "name",
        "public_symbols",
        "schema",
        "version",
    }
)
_RANKING_FIELDS = frozenset(
    {
        "change_amplification_after",
        "concern",
        "content_identity",
        "context_amplification_after",
        "cross_boundary_effects_after",
        "cycles_after",
        "mutable_sharing_after",
        "predicted_cone_reduction",
        "predicted_context_reduction",
        "public_symbols_after",
        "schema",
        "validation_amplification_after",
        "version",
    }
)
_PROPOSAL_FIELDS = frozenset(
    {
        "allowed_callers",
        "allowed_effects",
        "can_apply",
        "can_authorize_changes",
        "can_create_authority",
        "can_promote",
        "can_transfer_authority",
        "canonical_owner",
        "completion_authoritative",
        "concern",
        "content_identity",
        "cost",
        "deprecated_paths",
        "deprecations",
        "disposition",
        "freshness",
        "hard_constraints",
        "members",
        "migration_adapters",
        "predicted_cone_reduction",
        "predicted_context_reduction",
        "proofs",
        "ranking_inputs",
        "ranking_is_non_probative",
        "rejection_reasons",
        "repository_tree",
        "required_interface",
        "rollback",
        "schema",
        "state_owner",
        "tests",
        "tier",
        "version",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "can_apply",
        "can_authorize_changes",
        "can_create_authority",
        "can_promote",
        "can_transfer_authority",
        "candidate_tier",
        "completion_authoritative",
        "concerns",
        "content_identity",
        "contract_identity",
        "effect_class",
        "entropy_identity",
        "freshness",
        "ownership_identity",
        "proposals",
        "ranked_proposal_identities",
        "ranking_is_non_probative",
        "rejected_concerns",
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
        raise BoundarySynthesizerError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise BoundarySynthesizerError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise BoundarySynthesizerError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise BoundarySynthesizerError(f"{name} must be a boolean")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    number = _require_int(value, name, error_type=BoundarySynthesizerError)
    if number < 0:
        raise BoundarySynthesizerError(f"{name} must be a non-negative integer")
    return number


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoundarySynthesizerError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=BoundarySynthesizerError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _wrap_contract(exc: ArchitectureContractError) -> BoundarySynthesizerError:
    if isinstance(exc, BoundarySynthesizerError):
        return exc
    return BoundarySynthesizerError(str(exc))


def _require_architecture_ir(
    graph: ArchitectureIR | Mapping[str, Any],
) -> ArchitectureIR:
    if isinstance(graph, ArchitectureIR):
        return graph
    try:
        return ArchitectureIR.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise _wrap_contract(exc) from exc


def _optional_identity(value: Any, name: str) -> str:
    if value in ("", None):
        return ""
    text = _require_text(value, name, error_type=BoundarySynthesizerError)
    return _validate_dag_json_cid(text)


def _record_tuple(value: Any, name: str, record_type: type[Any]) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoundarySynthesizerError(f"{name} must be a list of objects")
    records = tuple(
        item if isinstance(item, record_type) else record_type.from_mapping(item)
        for item in value
    )
    return records


@dataclass(frozen=True)
class BoundarySourceBinding:
    """Current-tree source binding for one initial boundary concern."""

    concern: BoundaryConcern
    path: str
    nominated_symbol: str
    start_line: int
    related_authority_concerns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "concern",
            _closed_enum(
                self.concern,
                BoundaryConcern,
                "boundary concern",
                error_type=BoundarySynthesizerError,
            ),
        )
        object.__setattr__(
            self,
            "path",
            _require_text(self.path, "path", error_type=BoundarySynthesizerError),
        )
        object.__setattr__(
            self,
            "nominated_symbol",
            _require_text(
                self.nominated_symbol,
                "nominated_symbol",
                error_type=BoundarySynthesizerError,
            ),
        )
        object.__setattr__(
            self,
            "start_line",
            _require_non_negative_int(self.start_line, "start_line"),
        )
        object.__setattr__(
            self,
            "related_authority_concerns",
            _require_text_tuple(
                self.related_authority_concerns, "related_authority_concerns"
            ),
        )


INITIAL_BOUNDARY_SOURCE_BINDINGS: tuple[BoundarySourceBinding, ...] = (
    BoundarySourceBinding(
        BoundaryConcern.PROVIDER_CAPABILITY_SELECTION,
        "ipfs_accelerate_py/agent_supervisor/control/capability_resolver.py",
        "ProviderCapabilityEvidence",
        233,
        ("provider capability", "provider selection"),
    ),
    BoundarySourceBinding(
        BoundaryConcern.EXECUTION_REQUESTS_OUTCOMES,
        "ipfs_accelerate_py/agent_supervisor/contracts/execution.py",
        "InvocationMode",
        167,
        ("execution result",),
    ),
    BoundarySourceBinding(
        BoundaryConcern.ANALYSIS_CONTEXT,
        "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
        "ContextCompiler",
        4028,
        (),
    ),
    BoundarySourceBinding(
        BoundaryConcern.PROOF_VERIFICATION_SCHEDULING,
        "ipfs_accelerate_py/agent_supervisor/verification/planner.py",
        "IncrementalVerificationPlanner",
        2310,
        ("proof verification", "test evidence"),
    ),
    BoundarySourceBinding(
        BoundaryConcern.TASK_OBJECTIVE_STATE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "DuckDBConnection",
        354,
        ("task identity", "objective identity", "state persistence"),
    ),
    BoundarySourceBinding(
        BoundaryConcern.CONTROL_OPERATIONS,
        "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py",
        "OPERATION_CATALOG_V2",
        8346,
        ("operation identity", "authorization", "confirmation"),
    ),
    BoundarySourceBinding(
        BoundaryConcern.RECEIPT_EVIDENCE_QUERIES,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py",
        "AuthoritativeCompletionGate",
        178,
        ("completion evidence", "test evidence"),
    ),
    BoundarySourceBinding(
        BoundaryConcern.LEGACY_COMPATIBILITY,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_review.py",
        "legacy_landed_review",
        1,
        (),
    ),
    BoundarySourceBinding(
        BoundaryConcern.SIMULATIONS,
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_usage.py",
        "provider_usage",
        1,
        (),
    ),
)


@dataclass(frozen=True)
class BoundaryCostVector:
    """Independent integer counts for one before/after/reduction snapshot."""

    cross_boundary_effects: int
    mutable_sharing: int
    cycles: int
    public_symbols: int
    change_amplification: int
    context_amplification: int
    validation_amplification: int
    dependency_cone: int
    schema: str = BOUNDARY_COST_VECTOR_SCHEMA
    version: int = BOUNDARY_COST_VECTOR_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_COST_VECTOR_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-cost-vector schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_COST_VECTOR_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-cost-vector version")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        for name in REQUIRED_COST_DIMENSIONS:
            object.__setattr__(
                self,
                name.value,
                _require_non_negative_int(getattr(self, name.value), name.value),
            )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError("boundary-cost-vector content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "change_amplification": self.change_amplification,
            "context_amplification": self.context_amplification,
            "cross_boundary_effects": self.cross_boundary_effects,
            "cycles": self.cycles,
            "dependency_cone": self.dependency_cone,
            "mutable_sharing": self.mutable_sharing,
            "public_symbols": self.public_symbols,
            "schema": self.schema,
            "validation_amplification": self.validation_amplification,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-cost-vector content identity mismatch")
        return {**payload, "content_identity": identity}

    def dimension(self, kind: BoundaryCostDimension | str) -> int:
        parsed = _closed_enum(
            kind,
            BoundaryCostDimension,
            "cost dimension",
            error_type=BoundarySynthesizerError,
        )
        return int(getattr(self, parsed.value))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryCostVector":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _COST_VECTOR_FIELDS)
        record = cls(
            cross_boundary_effects=mapping["cross_boundary_effects"],
            mutable_sharing=mapping["mutable_sharing"],
            cycles=mapping["cycles"],
            public_symbols=mapping["public_symbols"],
            change_amplification=mapping["change_amplification"],
            context_amplification=mapping["context_amplification"],
            validation_amplification=mapping["validation_amplification"],
            dependency_cone=mapping["dependency_cone"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-cost-vector content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def zeros(cls) -> "BoundaryCostVector":
        return cls(
            cross_boundary_effects=0,
            mutable_sharing=0,
            cycles=0,
            public_symbols=0,
            change_amplification=0,
            context_amplification=0,
            validation_amplification=0,
            dependency_cone=0,
        )


def _cost_reductions(
    before: BoundaryCostVector, after: BoundaryCostVector
) -> BoundaryCostVector:
    return BoundaryCostVector(
        cross_boundary_effects=max(0, before.cross_boundary_effects - after.cross_boundary_effects),
        mutable_sharing=max(0, before.mutable_sharing - after.mutable_sharing),
        cycles=max(0, before.cycles - after.cycles),
        public_symbols=max(0, before.public_symbols - after.public_symbols),
        change_amplification=max(
            0, before.change_amplification - after.change_amplification
        ),
        context_amplification=max(
            0, before.context_amplification - after.context_amplification
        ),
        validation_amplification=max(
            0, before.validation_amplification - after.validation_amplification
        ),
        dependency_cone=max(0, before.dependency_cone - after.dependency_cone),
    )


@dataclass(frozen=True)
class BoundaryCost:
    """Before/after/reduction cost model for one proposed boundary."""

    before: BoundaryCostVector
    after: BoundaryCostVector
    reductions: BoundaryCostVector = BoundaryCostVector.zeros()
    schema: str = BOUNDARY_COST_SCHEMA
    version: int = BOUNDARY_COST_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_COST_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-cost schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_COST_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-cost version")
        before = (
            self.before
            if isinstance(self.before, BoundaryCostVector)
            else BoundaryCostVector.from_mapping(self.before)
        )
        after = (
            self.after
            if isinstance(self.after, BoundaryCostVector)
            else BoundaryCostVector.from_mapping(self.after)
        )
        expected = _cost_reductions(before, after)
        reductions = (
            self.reductions
            if isinstance(self.reductions, BoundaryCostVector)
            else BoundaryCostVector.from_mapping(self.reductions)
        )
        if reductions != expected:
            raise BoundarySynthesizerError("boundary-cost reductions must match before-after")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "reductions", reductions)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError("boundary-cost content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "after": self.after.to_dict(),
            "before": self.before.to_dict(),
            "reductions": self.reductions.to_dict(),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-cost content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryCost":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _COST_FIELDS)
        record = cls(
            before=mapping["before"],
            after=mapping["after"],
            reductions=mapping["reductions"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-cost content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_vectors(
        cls, before: BoundaryCostVector, after: BoundaryCostVector
    ) -> "BoundaryCost":
        return cls(before=before, after=after, reductions=_cost_reductions(before, after))


@dataclass(frozen=True)
class HardConstraintCheck:
    """One independently recorded hard-constraint outcome."""

    constraint: HardConstraintKind
    satisfied: bool
    explanation: str
    evidence_node_ids: tuple[str, ...] = ()
    evidence_edge_ids: tuple[str, ...] = ()
    schema: str = HARD_CONSTRAINT_SCHEMA
    version: int = HARD_CONSTRAINT_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != HARD_CONSTRAINT_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-hard-constraint schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != HARD_CONSTRAINT_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-hard-constraint version")
        constraint = _closed_enum(
            self.constraint,
            HardConstraintKind,
            "hard constraint",
            error_type=BoundarySynthesizerError,
        )
        satisfied = _require_bool(self.satisfied, "satisfied")
        explanation = _require_text(
            self.explanation, "explanation", error_type=BoundarySynthesizerError
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "satisfied", satisfied)
        object.__setattr__(self, "explanation", explanation)
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
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError(
                    "boundary-hard-constraint content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint.value,
            "evidence_edge_ids": list(self.evidence_edge_ids),
            "evidence_node_ids": list(self.evidence_node_ids),
            "explanation": self.explanation,
            "satisfied": self.satisfied,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError(
                "boundary-hard-constraint content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HardConstraintCheck":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _CONSTRAINT_FIELDS)
        record = cls(
            constraint=mapping["constraint"],
            satisfied=mapping["satisfied"],
            explanation=mapping["explanation"],
            evidence_node_ids=mapping["evidence_node_ids"],
            evidence_edge_ids=mapping["evidence_edge_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError(
                "boundary-hard-constraint content identity mismatch"
            )
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ProposedInterface:
    """Candidate-tier stable interface named by one boundary proposal."""

    name: str
    concern: BoundaryConcern
    members: tuple[str, ...] = ()
    public_symbols: tuple[str, ...] = ()
    schema: str = PROPOSED_INTERFACE_SCHEMA
    version: int = PROPOSED_INTERFACE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != PROPOSED_INTERFACE_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-interface schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != PROPOSED_INTERFACE_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-interface version")
        name = _require_text(self.name, "name", error_type=BoundarySynthesizerError)
        concern = _closed_enum(
            self.concern,
            BoundaryConcern,
            "boundary concern",
            error_type=BoundarySynthesizerError,
        )
        if name != _INTERFACE_NAMES[concern]:
            raise BoundarySynthesizerError("proposed interface name must match concern")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "members", _require_text_tuple(self.members, "members"))
        symbols = _require_text_tuple(self.public_symbols, "public_symbols")
        if name not in symbols:
            symbols = tuple(sorted(set(symbols) | {name}))
        object.__setattr__(self, "public_symbols", symbols)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError("boundary-interface content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "concern": self.concern.value,
            "members": list(self.members),
            "name": self.name,
            "public_symbols": list(self.public_symbols),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-interface content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProposedInterface":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _INTERFACE_FIELDS)
        record = cls(
            name=mapping["name"],
            concern=mapping["concern"],
            members=mapping["members"],
            public_symbols=mapping["public_symbols"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-interface content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryRankingInputs:
    """Deterministic, non-probative ranking inputs for one proposal."""

    concern: str
    predicted_cone_reduction: int
    predicted_context_reduction: int
    cross_boundary_effects_after: int
    mutable_sharing_after: int
    cycles_after: int
    public_symbols_after: int
    change_amplification_after: int
    context_amplification_after: int
    validation_amplification_after: int
    schema: str = RANKING_INPUT_SCHEMA
    version: int = RANKING_INPUT_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != RANKING_INPUT_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-ranking-inputs schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != RANKING_INPUT_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-ranking-inputs version")
        concern = _require_text(self.concern, "concern", error_type=BoundarySynthesizerError)
        if concern not in CLOSED_BOUNDARY_CONCERNS:
            raise BoundarySynthesizerError(f"unsupported boundary concern: {concern!r}")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "concern", concern)
        for name in (
            "predicted_cone_reduction",
            "predicted_context_reduction",
            "cross_boundary_effects_after",
            "mutable_sharing_after",
            "cycles_after",
            "public_symbols_after",
            "change_amplification_after",
            "context_amplification_after",
            "validation_amplification_after",
        ):
            object.__setattr__(self, name, _require_non_negative_int(getattr(self, name), name))
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError(
                    "boundary-ranking-inputs content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def sort_key(self) -> tuple[int, int, int, int, int, int, int, int, int, str]:
        return (
            -self.predicted_cone_reduction,
            -self.predicted_context_reduction,
            self.cross_boundary_effects_after,
            self.mutable_sharing_after,
            self.cycles_after,
            self.public_symbols_after,
            self.change_amplification_after,
            self.context_amplification_after,
            self.validation_amplification_after,
            self.concern,
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "change_amplification_after": self.change_amplification_after,
            "concern": self.concern,
            "context_amplification_after": self.context_amplification_after,
            "cross_boundary_effects_after": self.cross_boundary_effects_after,
            "cycles_after": self.cycles_after,
            "mutable_sharing_after": self.mutable_sharing_after,
            "predicted_cone_reduction": self.predicted_cone_reduction,
            "predicted_context_reduction": self.predicted_context_reduction,
            "public_symbols_after": self.public_symbols_after,
            "schema": self.schema,
            "validation_amplification_after": self.validation_amplification_after,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError(
                "boundary-ranking-inputs content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryRankingInputs":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _RANKING_FIELDS)
        record = cls(
            concern=mapping["concern"],
            predicted_cone_reduction=mapping["predicted_cone_reduction"],
            predicted_context_reduction=mapping["predicted_context_reduction"],
            cross_boundary_effects_after=mapping["cross_boundary_effects_after"],
            mutable_sharing_after=mapping["mutable_sharing_after"],
            cycles_after=mapping["cycles_after"],
            public_symbols_after=mapping["public_symbols_after"],
            change_amplification_after=mapping["change_amplification_after"],
            context_amplification_after=mapping["context_amplification_after"],
            validation_amplification_after=mapping["validation_amplification_after"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError(
                "boundary-ranking-inputs content identity mismatch"
            )
        return record

    from_dict = from_mapping


def _constraint_tuple(
    value: Any,
) -> tuple[HardConstraintCheck, ...]:
    records = _record_tuple(value, "hard_constraints", HardConstraintCheck)
    by_kind = {item.constraint: item for item in records}
    if len(by_kind) != len(records):
        raise BoundarySynthesizerError("hard constraints must be unique")
    missing = [item.value for item in REQUIRED_HARD_CONSTRAINTS if item not in by_kind]
    extra = sorted(
        item.constraint.value
        for item in records
        if item.constraint not in set(REQUIRED_HARD_CONSTRAINTS)
    )
    if missing:
        raise BoundarySynthesizerError(f"missing hard constraints: {missing}")
    if extra:
        raise BoundarySynthesizerError(f"unsupported hard constraints: {extra}")
    return tuple(by_kind[kind] for kind in REQUIRED_HARD_CONSTRAINTS)


@dataclass(frozen=True)
class BoundaryProposal:
    """Complete candidate-tier interface-boundary proposal."""

    concern: BoundaryConcern
    required_interface: ProposedInterface
    canonical_owner: str
    allowed_callers: tuple[str, ...]
    allowed_effects: tuple[str, ...]
    state_owner: str
    migration_adapters: tuple[str, ...]
    deprecations: tuple[str, ...]
    deprecated_paths: tuple[str, ...]
    tests: tuple[str, ...]
    proofs: tuple[str, ...]
    rollback: str
    cost: BoundaryCost
    hard_constraints: tuple[HardConstraintCheck, ...]
    repository_tree: str
    freshness: str
    members: tuple[str, ...] = ()
    predicted_context_reduction: int = 0
    predicted_cone_reduction: int = 0
    disposition: ProposalDisposition = ProposalDisposition.ADMITTED
    rejection_reasons: tuple[str, ...] = ()
    ranking_inputs: BoundaryRankingInputs | None = None
    ranking_is_non_probative: bool = True
    completion_authoritative: bool = False
    can_apply: bool = False
    can_transfer_authority: bool = False
    can_create_authority: bool = False
    can_promote: bool = False
    can_authorize_changes: bool = False
    tier: ProposalTier = ProposalTier.CANDIDATE
    schema: str = BOUNDARY_PROPOSAL_SCHEMA
    version: int = BOUNDARY_PROPOSAL_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_PROPOSAL_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-proposal schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_PROPOSAL_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-proposal version")
        concern = _closed_enum(
            self.concern,
            BoundaryConcern,
            "boundary concern",
            error_type=BoundarySynthesizerError,
        )
        required = (
            self.required_interface
            if isinstance(self.required_interface, ProposedInterface)
            else ProposedInterface.from_mapping(self.required_interface)
        )
        if required.concern is not concern:
            raise BoundarySynthesizerError("required interface concern must match proposal")
        disposition = _closed_enum(
            self.disposition,
            ProposalDisposition,
            "proposal disposition",
            error_type=BoundarySynthesizerError,
        )
        tier = _closed_enum(
            self.tier, ProposalTier, "proposal tier", error_type=BoundarySynthesizerError
        )
        if tier is not ProposalTier.CANDIDATE:
            raise BoundarySynthesizerAuthorityError(
                "boundary proposals remain candidate-tier"
            )
        rollback = _require_text(self.rollback, "rollback", error_type=BoundarySynthesizerError)
        if rollback != ROLLBACK_DECLARATION:
            raise BoundarySynthesizerError("rollback declaration must remain exact")
        if self.can_apply is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot apply a candidate"
            )
        if self.can_transfer_authority is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot transfer an existing authority"
            )
        if self.can_create_authority is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot create authority"
            )
        if self.can_promote is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot promote a candidate"
            )
        if self.can_authorize_changes is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot authorize changes"
            )
        if self.completion_authoritative is not False:
            raise BoundarySynthesizerError("boundary proposals are not completion evidence")
        if self.ranking_is_non_probative is not True:
            raise BoundarySynthesizerError("boundary ranking is non-probative")
        owner = self.canonical_owner
        if type(owner) is not str:
            raise BoundarySynthesizerError("canonical_owner must be a string")
        if owner and _looks_like_content_identity(owner):
            raise BoundarySynthesizerError("content identity is not inferred to be authority")
        state_owner = self.state_owner
        if type(state_owner) is not str or not state_owner:
            raise BoundarySynthesizerError("state_owner must be a nonempty string")
        constraints = _constraint_tuple(self.hard_constraints)
        failed = tuple(item for item in constraints if item.satisfied is False)
        if failed and disposition is not ProposalDisposition.REJECTED:
            raise BoundarySynthesizerError("failed hard constraints must reject the proposal")
        if not failed and disposition is not ProposalDisposition.ADMITTED:
            raise BoundarySynthesizerError("satisfied hard constraints must admit the proposal")
        if disposition is ProposalDisposition.ADMITTED and not owner:
            raise BoundarySynthesizerError("admitted proposals must name a canonical owner")
        reasons = _require_text_tuple(self.rejection_reasons, "rejection_reasons")
        expected_reasons = tuple(sorted({item.constraint.value for item in failed}))
        if disposition is ProposalDisposition.REJECTED:
            if reasons != expected_reasons:
                raise BoundarySynthesizerError(
                    "rejection_reasons must list unsatisfied hard constraints"
                )
        elif reasons:
            raise BoundarySynthesizerError("admitted proposals cannot retain rejection reasons")
        cost = (
            self.cost if isinstance(self.cost, BoundaryCost) else BoundaryCost.from_mapping(self.cost)
        )
        predicted_context = _require_non_negative_int(
            self.predicted_context_reduction, "predicted_context_reduction"
        )
        predicted_cone = _require_non_negative_int(
            self.predicted_cone_reduction, "predicted_cone_reduction"
        )
        if predicted_context != cost.reductions.context_amplification:
            raise BoundarySynthesizerError(
                "predicted_context_reduction must match context-amplification reduction"
            )
        if predicted_cone != cost.reductions.dependency_cone:
            raise BoundarySynthesizerError(
                "predicted_cone_reduction must match dependency-cone reduction"
            )
        ranking = self.ranking_inputs
        if ranking is None:
            ranking = BoundaryRankingInputs(
                concern=concern.value,
                predicted_cone_reduction=predicted_cone,
                predicted_context_reduction=predicted_context,
                cross_boundary_effects_after=cost.after.cross_boundary_effects,
                mutable_sharing_after=cost.after.mutable_sharing,
                cycles_after=cost.after.cycles,
                public_symbols_after=cost.after.public_symbols,
                change_amplification_after=cost.after.change_amplification,
                context_amplification_after=cost.after.context_amplification,
                validation_amplification_after=cost.after.validation_amplification,
            )
        elif not isinstance(ranking, BoundaryRankingInputs):
            ranking = BoundaryRankingInputs.from_mapping(ranking)
        if ranking.concern != concern.value:
            raise BoundarySynthesizerError("ranking concern must match proposal")
        if ranking.predicted_cone_reduction != predicted_cone:
            raise BoundarySynthesizerError("ranking cone reduction must match proposal")
        if ranking.predicted_context_reduction != predicted_context:
            raise BoundarySynthesizerError("ranking context reduction must match proposal")
        members = _require_text_tuple(self.members, "members")
        if members != required.members:
            raise BoundarySynthesizerError("proposal members must match required interface")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "required_interface", required)
        object.__setattr__(self, "canonical_owner", owner)
        object.__setattr__(
            self, "allowed_callers", _require_text_tuple(self.allowed_callers, "allowed_callers")
        )
        object.__setattr__(
            self, "allowed_effects", _require_text_tuple(self.allowed_effects, "allowed_effects")
        )
        object.__setattr__(self, "state_owner", state_owner)
        object.__setattr__(
            self,
            "migration_adapters",
            _require_text_tuple(self.migration_adapters, "migration_adapters"),
        )
        object.__setattr__(
            self, "deprecations", _require_text_tuple(self.deprecations, "deprecations")
        )
        object.__setattr__(
            self,
            "deprecated_paths",
            _require_text_tuple(self.deprecated_paths, "deprecated_paths"),
        )
        object.__setattr__(self, "tests", _require_text_tuple(self.tests, "tests"))
        object.__setattr__(self, "proofs", _require_text_tuple(self.proofs, "proofs"))
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "hard_constraints", constraints)
        object.__setattr__(
            self,
            "repository_tree",
            _require_text(
                self.repository_tree, "repository_tree", error_type=BoundarySynthesizerError
            ),
        )
        object.__setattr__(
            self,
            "freshness",
            _require_text(self.freshness, "freshness", error_type=BoundarySynthesizerError),
        )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "predicted_context_reduction", predicted_context)
        object.__setattr__(self, "predicted_cone_reduction", predicted_cone)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "ranking_inputs", ranking)
        object.__setattr__(self, "ranking_is_non_probative", True)
        object.__setattr__(self, "completion_authoritative", False)
        object.__setattr__(self, "can_apply", False)
        object.__setattr__(self, "can_transfer_authority", False)
        object.__setattr__(self, "can_create_authority", False)
        object.__setattr__(self, "can_promote", False)
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "tier", ProposalTier.CANDIDATE)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError("boundary-proposal content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    @property
    def admitted(self) -> bool:
        return self.disposition is ProposalDisposition.ADMITTED

    def constraint(self, kind: HardConstraintKind | str) -> HardConstraintCheck:
        parsed = _closed_enum(
            kind,
            HardConstraintKind,
            "hard constraint",
            error_type=BoundarySynthesizerError,
        )
        for item in self.hard_constraints:
            if item.constraint is parsed:
                return item
        raise BoundarySynthesizerError(f"missing hard constraint: {parsed.value}")

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_candidate_application("apply")

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_candidate_promotion("promote")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    def create_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_creation("create")

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_change_authorization("change")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "allowed_callers": list(self.allowed_callers),
            "allowed_effects": list(self.allowed_effects),
            "can_apply": False,
            "can_authorize_changes": False,
            "can_create_authority": False,
            "can_promote": False,
            "can_transfer_authority": False,
            "canonical_owner": self.canonical_owner,
            "completion_authoritative": False,
            "concern": self.concern.value,
            "cost": self.cost.to_dict(),
            "deprecated_paths": list(self.deprecated_paths),
            "deprecations": list(self.deprecations),
            "disposition": self.disposition.value,
            "freshness": self.freshness,
            "hard_constraints": [item.to_dict() for item in self.hard_constraints],
            "members": list(self.members),
            "migration_adapters": list(self.migration_adapters),
            "predicted_cone_reduction": self.predicted_cone_reduction,
            "predicted_context_reduction": self.predicted_context_reduction,
            "proofs": list(self.proofs),
            "ranking_inputs": self.ranking_inputs.to_dict() if self.ranking_inputs else {},
            "ranking_is_non_probative": True,
            "rejection_reasons": list(self.rejection_reasons),
            "repository_tree": self.repository_tree,
            "required_interface": self.required_interface.to_dict(),
            "rollback": self.rollback,
            "schema": self.schema,
            "state_owner": self.state_owner,
            "tests": list(self.tests),
            "tier": ProposalTier.CANDIDATE.value,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-proposal content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryProposal":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _PROPOSAL_FIELDS)
        record = cls(
            concern=mapping["concern"],
            required_interface=mapping["required_interface"],
            canonical_owner=mapping["canonical_owner"],
            allowed_callers=mapping["allowed_callers"],
            allowed_effects=mapping["allowed_effects"],
            state_owner=mapping["state_owner"],
            migration_adapters=mapping["migration_adapters"],
            deprecations=mapping["deprecations"],
            deprecated_paths=mapping["deprecated_paths"],
            tests=mapping["tests"],
            proofs=mapping["proofs"],
            rollback=mapping["rollback"],
            cost=mapping["cost"],
            hard_constraints=mapping["hard_constraints"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            members=mapping["members"],
            predicted_context_reduction=mapping["predicted_context_reduction"],
            predicted_cone_reduction=mapping["predicted_cone_reduction"],
            disposition=mapping["disposition"],
            rejection_reasons=mapping["rejection_reasons"],
            ranking_inputs=mapping["ranking_inputs"],
            ranking_is_non_probative=mapping["ranking_is_non_probative"],
            completion_authoritative=mapping["completion_authoritative"],
            can_apply=mapping["can_apply"],
            can_transfer_authority=mapping["can_transfer_authority"],
            can_create_authority=mapping["can_create_authority"],
            can_promote=mapping["can_promote"],
            can_authorize_changes=mapping["can_authorize_changes"],
            tier=mapping["tier"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-proposal content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "BoundaryProposal":
        if type(payload) is not str or not payload:
            raise BoundarySynthesizerError("boundary-proposal JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BoundarySynthesizerError("boundary-proposal JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise BoundarySynthesizerError("boundary-proposal JSON must contain an object")
        return cls.from_mapping(decoded)


@dataclass(frozen=True)
class BoundarySynthesisResult:
    """Complete synthesis of the closed initial boundary-concern set."""

    repository_tree: str
    freshness: str
    architecture_ir_identity: str
    proposals: tuple[BoundaryProposal, ...]
    ownership_identity: str = ""
    contract_identity: str = ""
    entropy_identity: str = ""
    schema: str = BOUNDARY_SYNTHESIS_SCHEMA
    version: int = BOUNDARY_SYNTHESIS_VERSION
    effect_class: str = EFFECT_CLASS
    candidate_tier: bool = True
    ranking_is_non_probative: bool = True
    completion_authoritative: bool = False
    can_apply: bool = False
    can_transfer_authority: bool = False
    can_create_authority: bool = False
    can_promote: bool = False
    can_authorize_changes: bool = False
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_SYNTHESIS_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-synthesis schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_SYNTHESIS_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-synthesis version")
        if self.effect_class != EFFECT_CLASS:
            raise BoundarySynthesizerError("unexpected boundary-synthesis effect class")
        if self.candidate_tier is not True:
            raise BoundarySynthesizerAuthorityError("boundary proposals remain candidate-tier")
        if self.ranking_is_non_probative is not True:
            raise BoundarySynthesizerError("boundary ranking is non-probative")
        if self.completion_authoritative is not False:
            raise BoundarySynthesizerError("boundary synthesis is not completion evidence")
        if self.can_apply is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot apply a candidate"
            )
        if self.can_transfer_authority is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot transfer an existing authority"
            )
        if self.can_create_authority is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot create authority"
            )
        if self.can_promote is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot promote a candidate"
            )
        if self.can_authorize_changes is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot authorize changes"
            )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=BoundarySynthesizerError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=BoundarySynthesizerError
        )
        architecture_ir_identity = _validate_dag_json_cid(
            _require_text(
                self.architecture_ir_identity,
                "architecture_ir_identity",
                error_type=BoundarySynthesizerError,
            )
        )
        proposals = _record_tuple(self.proposals, "proposals", BoundaryProposal)
        by_concern = {item.concern: item for item in proposals}
        if len(by_concern) != len(proposals):
            raise BoundarySynthesizerError("boundary proposals must be unique by concern")
        missing = [
            item.value for item in INITIAL_BOUNDARY_CONCERNS if item not in by_concern
        ]
        extra = sorted(
            item.concern.value
            for item in proposals
            if item.concern not in set(INITIAL_BOUNDARY_CONCERNS)
        )
        if missing:
            raise BoundarySynthesizerError(f"missing initial boundary concerns: {missing}")
        if extra:
            raise BoundarySynthesizerError(f"unsupported boundary concerns: {extra}")
        ordered = tuple(by_concern[kind] for kind in INITIAL_BOUNDARY_CONCERNS)
        for item in ordered:
            if item.repository_tree != repository_tree:
                raise BoundarySynthesizerError("proposal repository_tree must match result")
            if item.freshness != freshness:
                raise BoundarySynthesizerError("proposal freshness must match result")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "effect_class", EFFECT_CLASS)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "proposals", ordered)
        object.__setattr__(
            self, "ownership_identity", _optional_identity(self.ownership_identity, "ownership_identity")
        )
        object.__setattr__(
            self, "contract_identity", _optional_identity(self.contract_identity, "contract_identity")
        )
        object.__setattr__(
            self, "entropy_identity", _optional_identity(self.entropy_identity, "entropy_identity")
        )
        object.__setattr__(self, "candidate_tier", True)
        object.__setattr__(self, "ranking_is_non_probative", True)
        object.__setattr__(self, "completion_authoritative", False)
        object.__setattr__(self, "can_apply", False)
        object.__setattr__(self, "can_transfer_authority", False)
        object.__setattr__(self, "can_create_authority", False)
        object.__setattr__(self, "can_promote", False)
        object.__setattr__(self, "can_authorize_changes", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=BoundarySynthesizerError,
                )
            )
            if claimed != identity:
                raise BoundarySynthesizerError(
                    "boundary-synthesis content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    @property
    def concerns(self) -> tuple[str, ...]:
        return tuple(item.concern.value for item in self.proposals)

    @property
    def rejected_concerns(self) -> tuple[str, ...]:
        return tuple(
            item.concern.value
            for item in self.proposals
            if item.disposition is ProposalDisposition.REJECTED
        )

    @property
    def admitted_proposals(self) -> tuple[BoundaryProposal, ...]:
        return tuple(item for item in self.proposals if item.admitted)

    @property
    def ranked_proposal_identities(self) -> tuple[str, ...]:
        return tuple(
            item.content_identity
            for item in sorted(
                self.admitted_proposals, key=lambda item: item.ranking_inputs.sort_key()
            )
        )

    def proposal(self, concern: BoundaryConcern | str) -> BoundaryProposal:
        parsed = _closed_enum(
            concern,
            BoundaryConcern,
            "boundary concern",
            error_type=BoundarySynthesizerError,
        )
        for item in self.proposals:
            if item.concern is parsed:
                return item
        raise BoundarySynthesizerError(f"missing initial boundary concern: {parsed.value}")

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_candidate_application("apply")

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_candidate_promotion("promote")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    def create_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_creation("create")

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_change_authorization("change")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "can_apply": False,
            "can_authorize_changes": False,
            "can_create_authority": False,
            "can_promote": False,
            "can_transfer_authority": False,
            "candidate_tier": True,
            "completion_authoritative": False,
            "concerns": list(self.concerns),
            "contract_identity": self.contract_identity,
            "effect_class": EFFECT_CLASS,
            "entropy_identity": self.entropy_identity,
            "freshness": self.freshness,
            "ownership_identity": self.ownership_identity,
            "proposals": [item.to_dict() for item in self.proposals],
            "ranked_proposal_identities": list(self.ranked_proposal_identities),
            "ranking_is_non_probative": True,
            "rejected_concerns": list(self.rejected_concerns),
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-synthesis content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundarySynthesisResult":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _RESULT_FIELDS)
        record = cls(
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            architecture_ir_identity=mapping["architecture_ir_identity"],
            proposals=mapping["proposals"],
            ownership_identity=mapping["ownership_identity"],
            contract_identity=mapping["contract_identity"],
            entropy_identity=mapping["entropy_identity"],
            schema=mapping["schema"],
            version=mapping["version"],
            effect_class=mapping["effect_class"],
            candidate_tier=mapping["candidate_tier"],
            ranking_is_non_probative=mapping["ranking_is_non_probative"],
            completion_authoritative=mapping["completion_authoritative"],
            can_apply=mapping["can_apply"],
            can_transfer_authority=mapping["can_transfer_authority"],
            can_create_authority=mapping["can_create_authority"],
            can_promote=mapping["can_promote"],
            can_authorize_changes=mapping["can_authorize_changes"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-synthesis content identity mismatch")
        if mapping["concerns"] != list(record.concerns):
            raise BoundarySynthesizerError("concerns projection mismatch")
        if mapping["rejected_concerns"] != list(record.rejected_concerns):
            raise BoundarySynthesizerError("rejected_concerns projection mismatch")
        if mapping["ranked_proposal_identities"] != list(record.ranked_proposal_identities):
            raise BoundarySynthesizerError("ranked_proposal_identities projection mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "BoundarySynthesisResult":
        if type(payload) is not str or not payload:
            raise BoundarySynthesizerError(
                "boundary-synthesis JSON must be a nonempty string"
            )
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BoundarySynthesizerError("boundary-synthesis JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise BoundarySynthesizerError("boundary-synthesis JSON must contain an object")
        return cls.from_mapping(decoded)


@dataclass(frozen=True)
class _GraphView:
    architecture: ArchitectureIR
    nodes_by_id: dict[str, ArchitectureNode]
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
        outgoing={key: tuple(value) for key, value in outgoing.items()},
        incoming={key: tuple(value) for key, value in incoming.items()},
    )


def _quarantine_concern(kind: NodeKind) -> BoundaryConcern | None:
    return _QUARANTINE_CONCERN_BY_KIND.get(kind)


def _node_belongs_to_concern(node: ArchitectureNode, concern: BoundaryConcern) -> bool:
    quarantine = _quarantine_concern(node.kind)
    if quarantine is not None:
        return concern is quarantine
    return concern in _node_concerns(node)


def _node_concerns(node: ArchitectureNode) -> frozenset[BoundaryConcern]:
    quarantine = _quarantine_concern(node.kind)
    if quarantine is not None:
        return frozenset({quarantine})
    found: set[BoundaryConcern] = set(_PRIMARY_KIND_CONCERNS.get(node.kind, ()))
    path = node.provenance.span.path.lower()
    for concern, markers in _PATH_MARKERS.items():
        if any(marker in path for marker in markers):
            found.add(concern)
    return frozenset(found)


def _seed_members(
    view: _GraphView, concern: BoundaryConcern
) -> tuple[ArchitectureNode, ...]:
    seeds = []
    for node in view.architecture.nodes:
        if node.kind is NodeKind.AUTHORITY:
            continue
        if node.kind is NodeKind.PACKAGE or node.kind is NodeKind.REPOSITORY:
            continue
        if _node_belongs_to_concern(node, concern):
            seeds.append(node)
    return tuple(sorted(seeds, key=lambda item: item.node_id))


def _expand_members(
    view: _GraphView,
    seeds: Sequence[ArchitectureNode],
    owner_ids: set[str],
    concern: BoundaryConcern,
) -> tuple[ArchitectureNode, ...]:
    members = {node.node_id: node for node in seeds}
    pending = deque(node.node_id for node in seeds)
    while pending:
        current = pending.popleft()
        related = (*view.outgoing.get(current, ()), *view.incoming.get(current, ()))
        for edge in related:
            if edge.kind not in _EXPAND_EDGE_KINDS:
                continue
            other_id = edge.target if edge.source == current else edge.source
            if other_id in members:
                continue
            other = view.nodes_by_id[other_id]
            if other.kind not in _EXPAND_NODE_KINDS:
                continue
            quarantine = _quarantine_concern(other.kind)
            if quarantine is not None and concern is not quarantine:
                continue
            authorized = False
            if owner_ids:
                authorized = any(
                    edge.kind is EdgeKind.AUTHORIZES and edge.source in owner_ids
                    for edge in view.incoming.get(other_id, ())
                )
            if not authorized and other.kind in {
                NodeKind.SYMBOL,
                NodeKind.FILE,
                NodeKind.EFFECT,
            }:
                if other.kind is NodeKind.EFFECT or edge.kind in {
                    EdgeKind.CONTAINS,
                    EdgeKind.IMPLEMENTS,
                    EdgeKind.TESTS,
                    EdgeKind.PROVES,
                }:
                    authorized = True
            if not authorized and other.kind in {
                NodeKind.TEST,
                NodeKind.PROOF,
                NodeKind.INTERFACE,
            }:
                authorized = True
            if not authorized and other.kind in _QUARANTINE_NODE_KINDS:
                authorized = concern is _quarantine_concern(other.kind)
            if not authorized:
                continue
            members[other_id] = other
            pending.append(other_id)
            if len(members) > MAX_BOUNDARY_MEMBERS:
                break
        if len(members) > MAX_BOUNDARY_MEMBERS:
            break
    return tuple(sorted(members.values(), key=lambda item: item.node_id))


def _authorizers(view: _GraphView, node_ids: Iterable[str]) -> tuple[str, ...]:
    owners: set[str] = set()
    pending = deque(node_ids)
    seen: set[str] = set()
    while pending:
        current = pending.popleft()
        if current in seen or current not in view.nodes_by_id:
            continue
        seen.add(current)
        current_node = view.nodes_by_id[current]
        if current_node.kind is NodeKind.AUTHORITY:
            owners.add(current)
        for edge in view.incoming.get(current, ()):
            if edge.kind is EdgeKind.AUTHORIZES:
                source = view.nodes_by_id[edge.source]
                if source.kind is NodeKind.AUTHORITY:
                    owners.add(source.node_id)
            elif edge.kind in {EdgeKind.TESTS, EdgeKind.PROVES}:
                pending.append(edge.source)
        for edge in view.outgoing.get(current, ()):
            if edge.kind in {
                EdgeKind.TESTS,
                EdgeKind.PROVES,
                EdgeKind.IMPLEMENTS,
                EdgeKind.ADAPTS,
            }:
                pending.append(edge.target)
    return tuple(sorted(owners))


def _reachable(
    view: _GraphView,
    starts: Iterable[str],
    *,
    edge_kinds: frozenset[EdgeKind],
    reverse: bool = False,
) -> set[str]:
    seen: set[str] = set()
    pending = deque(node_id for node_id in starts if node_id in view.nodes_by_id)
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        edges = view.incoming.get(current, ()) if reverse else view.outgoing.get(current, ())
        for edge in edges:
            if edge.kind not in edge_kinds:
                continue
            nxt = edge.source if reverse else edge.target
            if nxt not in seen and nxt in view.nodes_by_id:
                pending.append(nxt)
    return seen


def _cross_boundary_effect_edges(
    view: _GraphView, members: set[str]
) -> tuple[ArchitectureEdge, ...]:
    found = []
    for edge in view.architecture.edges:
        if edge.kind not in _EFFECT_EDGE_KINDS:
            continue
        source_in = edge.source in members
        target_in = edge.target in members
        if source_in != target_in:
            found.append(edge)
    return tuple(sorted(found, key=lambda item: item.edge_id))


def _mutable_shared_states(
    view: _GraphView, members: set[str]
) -> tuple[str, ...]:
    writers: dict[str, set[str]] = defaultdict(set)
    for edge in view.architecture.edges:
        if edge.kind not in _MUTABLE_EDGE_KINDS:
            continue
        target = view.nodes_by_id[edge.target]
        if target.kind is not NodeKind.STATE:
            continue
        writers[target.node_id].add(edge.source)
    shared = []
    for state_id, sources in writers.items():
        member_writers = sources & members
        outsider_writers = sources - members
        if member_writers and (outsider_writers or len(member_writers) > 1):
            shared.append(state_id)
    return tuple(sorted(shared))


def _cross_boundary_cycle_nodes(
    view: _GraphView, members: set[str]
) -> tuple[str, ...]:
    external_called = set()
    callers = set()
    for edge in view.architecture.edges:
        if edge.kind not in _CALL_EDGE_KINDS:
            continue
        if edge.source in members and edge.target not in members:
            external_called.add(edge.target)
        if edge.source not in members and edge.target in members:
            callers.add(edge.source)
    if not external_called or not callers:
        return ()
    reachable = _reachable(view, external_called, edge_kinds=_CALL_EDGE_KINDS)
    cyclic = sorted((reachable & callers) | (external_called & callers))
    return tuple(cyclic)


def _cone_ids(view: _GraphView, starts: Iterable[str]) -> set[str]:
    return _reachable(view, starts, edge_kinds=_CONE_EDGE_KINDS)


def _count_kinds(view: _GraphView, node_ids: Iterable[str], kinds: frozenset[NodeKind]) -> int:
    return sum(
        1
        for node_id in node_ids
        if node_id in view.nodes_by_id and view.nodes_by_id[node_id].kind in kinds
    )


def _amplification(view: _GraphView, node_ids: set[str]) -> tuple[int, int, int]:
    change = _count_kinds(
        view,
        node_ids,
        frozenset(
            {
                NodeKind.FILE,
                NodeKind.SYMBOL,
                NodeKind.INTERFACE,
                NodeKind.SCHEMA,
                NodeKind.EFFECT,
                NodeKind.TEST,
                NodeKind.PROOF,
                NodeKind.PROVIDER,
                NodeKind.ENTRYPOINT,
                NodeKind.AUTHORITY,
            }
        ),
    )
    context = _count_kinds(view, node_ids, _CONTEXT_NODE_KINDS)
    validation = _count_kinds(view, node_ids, frozenset({NodeKind.TEST, NodeKind.PROOF}))
    return change, context, validation


def _cost_vector(
    *,
    cross_boundary_effects: int,
    mutable_sharing: int,
    cycles: int,
    public_symbols: int,
    change_amplification: int,
    context_amplification: int,
    validation_amplification: int,
    dependency_cone: int,
) -> BoundaryCostVector:
    return BoundaryCostVector(
        cross_boundary_effects=cross_boundary_effects,
        mutable_sharing=mutable_sharing,
        cycles=cycles,
        public_symbols=public_symbols,
        change_amplification=change_amplification,
        context_amplification=context_amplification,
        validation_amplification=validation_amplification,
        dependency_cone=dependency_cone,
    )


def measure_boundary_cost(
    view: _GraphView,
    members: Sequence[ArchitectureNode],
    *,
    facade_ids: Sequence[str] = (),
) -> tuple[BoundaryCostVector, BoundaryCostVector]:
    """Measure current and predicted cost vectors for one member cluster."""

    member_ids = {node.node_id for node in members}
    effect_edges = _cross_boundary_effect_edges(view, member_ids)
    shared = _mutable_shared_states(view, member_ids)
    cyclic = _cross_boundary_cycle_nodes(view, member_ids)
    public_nodes = tuple(
        node.node_id for node in members if node.kind in _PUBLIC_NODE_KINDS
    )
    cone = _cone_ids(view, member_ids)
    change, context, validation = _amplification(view, cone)
    before = _cost_vector(
        cross_boundary_effects=len(effect_edges),
        mutable_sharing=len(shared),
        cycles=len(cyclic),
        public_symbols=len(public_nodes),
        change_amplification=change,
        context_amplification=context,
        validation_amplification=validation,
        dependency_cone=len(cone),
    )
    if not facade_ids:
        return before, before
    facade = set(facade_ids)
    after_cone = set(facade)
    after_change, after_context, after_validation = _amplification(view, after_cone)
    after_public = 1 if facade else 0
    after = _cost_vector(
        cross_boundary_effects=min(before.cross_boundary_effects, len(effect_edges) and 1 or 0)
        if member_ids
        else 0,
        mutable_sharing=0 if len(shared) <= 1 else before.mutable_sharing,
        cycles=0 if not cyclic else before.cycles,
        public_symbols=after_public,
        change_amplification=min(before.change_amplification, max(after_change, after_public)),
        context_amplification=min(before.context_amplification, max(after_context, after_public)),
        validation_amplification=before.validation_amplification,
        dependency_cone=min(before.dependency_cone, max(len(after_cone), after_public)),
    )
    if after.validation_amplification != before.validation_amplification:
        after = BoundaryCostVector(
            cross_boundary_effects=after.cross_boundary_effects,
            mutable_sharing=after.mutable_sharing,
            cycles=after.cycles,
            public_symbols=after.public_symbols,
            change_amplification=after.change_amplification,
            context_amplification=after.context_amplification,
            validation_amplification=before.validation_amplification,
            dependency_cone=after.dependency_cone,
        )
    return before, after


def _normalize_ownership(
    ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None,
) -> tuple[AuthorityOwnershipGraph | None, dict[str, str], tuple[str, ...]]:
    if ownership is None:
        return None, {}, ()
    if isinstance(ownership, AuthorityOwnershipGraph):
        owners: dict[str, str] = {}
        blockers: list[str] = []
        for record in ownership.concerns:
            if record.blocker is not None:
                blockers.append(record.concern.value)
            elif record.canonical_owner is not None:
                owners[record.concern.value] = record.canonical_owner.node_id
        return ownership, owners, tuple(sorted(set(blockers)))
    mapping = _require_mapping(ownership, error_type=BoundarySynthesizerError)
    if "concerns" in mapping and "schema" in mapping:
        return _normalize_ownership(AuthorityOwnershipGraph.from_mapping(mapping))
    owners = {}
    for key, value in mapping.items():
        owner_id = _require_text(value, "canonical owner", error_type=BoundarySynthesizerError)
        if _looks_like_content_identity(owner_id):
            raise BoundarySynthesizerError("content identity is not inferred to be authority")
        if key in CLOSED_BOUNDARY_CONCERNS:
            owners[key] = owner_id
            continue
        try:
            concern = ConcernKind(key)
        except ValueError as exc:
            raise BoundarySynthesizerError(f"unsupported ownership key: {key!r}") from exc
        owners[concern.value] = owner_id
    return None, owners, ()


def _normalize_contracts(
    contracts: ContractExtractionResult | Mapping[str, Any] | None,
) -> ContractExtractionResult | None:
    if contracts is None:
        return None
    if isinstance(contracts, ContractExtractionResult):
        return contracts
    return ContractExtractionResult.from_mapping(contracts)


def _normalize_entropy(
    entropy: SemanticEntropyReport | Mapping[str, Any] | None,
) -> SemanticEntropyReport | None:
    if entropy is None:
        return None
    if isinstance(entropy, SemanticEntropyReport):
        return entropy
    return SemanticEntropyReport.from_mapping(entropy)


def _owner_for_concern(
    concern: BoundaryConcern,
    *,
    view: _GraphView,
    members: Sequence[ArchitectureNode],
    owner_map: Mapping[str, str],
    blocked: Sequence[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    mapped = _OWNERSHIP_CONCERNS[concern]
    blocked_here = tuple(item.value for item in mapped if item.value in set(blocked))
    mapped_owners = []
    for item in mapped:
        owner_id = owner_map.get(item.value)
        if owner_id:
            mapped_owners.append(owner_id)
    concern_owner = owner_map.get(concern.value)
    if concern_owner:
        mapped_owners.append(concern_owner)
    unique_mapped = tuple(sorted(set(mapped_owners)))
    graph_owners = _authorizers(view, (node.node_id for node in members))
    production_graph = tuple(
        owner
        for owner in graph_owners
        if view.nodes_by_id[owner].kind is NodeKind.AUTHORITY
    )
    if unique_mapped and production_graph and set(unique_mapped) != set(production_graph):
        if not set(unique_mapped) & set(production_graph):
            return "", unique_mapped + production_graph, blocked_here
    if len(unique_mapped) > 1:
        return "", unique_mapped, blocked_here
    if unique_mapped:
        return unique_mapped[0], unique_mapped, blocked_here
    if len(production_graph) == 1:
        return production_graph[0], production_graph, blocked_here
    return "", production_graph, blocked_here


def _related_nodes(
    view: _GraphView,
    members: Sequence[ArchitectureNode],
    kind: NodeKind,
    edge_kind: EdgeKind,
) -> tuple[str, ...]:
    member_ids = {node.node_id for node in members}
    found: set[str] = set()
    for node in members:
        if node.kind is kind:
            found.add(node.node_id)
    for edge in view.architecture.edges:
        if edge.kind is not edge_kind:
            continue
        if edge.target in member_ids and view.nodes_by_id[edge.source].kind is kind:
            found.add(edge.source)
        if edge.source in member_ids and view.nodes_by_id[edge.target].kind is kind:
            found.add(edge.target)
    return tuple(sorted(found))


def _state_nodes(view: _GraphView, members: Sequence[ArchitectureNode]) -> tuple[str, ...]:
    member_ids = {node.node_id for node in members}
    found: set[str] = set()
    for node in members:
        if node.kind is NodeKind.STATE:
            found.add(node.node_id)
    for edge in view.architecture.edges:
        if edge.kind not in _MUTABLE_EDGE_KINDS:
            continue
        if edge.source in member_ids and view.nodes_by_id[edge.target].kind is NodeKind.STATE:
            found.add(edge.target)
    return tuple(sorted(found))


def _external_callers(view: _GraphView, members: Sequence[ArchitectureNode]) -> tuple[str, ...]:
    member_ids = {node.node_id for node in members}
    found: set[str] = set()
    for edge in view.architecture.edges:
        if edge.kind not in _CALL_EDGE_KINDS and edge.kind not in {
            EdgeKind.READS,
            EdgeKind.OBSERVES,
        }:
            continue
        if edge.target in member_ids and edge.source not in member_ids:
            found.add(edge.source)
    return tuple(sorted(found))


def _observed_effects(view: _GraphView, members: Sequence[ArchitectureNode]) -> tuple[str, ...]:
    member_ids = {node.node_id for node in members}
    found: set[str] = set()
    for edge in view.architecture.edges:
        if edge.kind not in _EFFECT_EDGE_KINDS:
            continue
        if edge.source in member_ids or edge.target in member_ids:
            found.add(f"{edge.kind.value}:{edge.target}")
    return tuple(sorted(found))


def _deprecated_paths(
    view: _GraphView, node_ids: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                view.nodes_by_id[node_id].provenance.span.path
                for node_id in node_ids
                if node_id in view.nodes_by_id
            }
        )
    )


def _check(
    kind: HardConstraintKind,
    satisfied: bool,
    explanation: str,
    node_ids: Iterable[str] = (),
    edge_ids: Iterable[str] = (),
) -> HardConstraintCheck:
    return HardConstraintCheck(
        constraint=kind,
        satisfied=satisfied,
        explanation=explanation,
        evidence_node_ids=tuple(node_ids),
        evidence_edge_ids=tuple(edge_ids),
    )


def _ambiguities_for(
    contracts: ContractExtractionResult | None, members: Sequence[ArchitectureNode]
) -> tuple[str, ...]:
    if contracts is None:
        return ()
    member_ids = {node.node_id for node in members}
    found: list[str] = []
    for candidate in contracts.candidates:
        if candidate.subject in member_ids or any(
            candidate.subject == node.provenance.span.path for node in members
        ):
            if candidate.ambiguities:
                found.extend(item.kind.value for item in candidate.ambiguities)
    return tuple(sorted(set(found)))


def check_hard_constraints(
    *,
    concern: BoundaryConcern,
    view: _GraphView,
    members: Sequence[ArchitectureNode],
    canonical_owner: str,
    competing_owners: Sequence[str],
    blocked_concerns: Sequence[str],
    state_ids: Sequence[str],
    callers: Sequence[str],
    effects: Sequence[str],
    tests: Sequence[str],
    proofs: Sequence[str],
    adapters: Sequence[str],
    deprecations: Sequence[str],
    proposed_effects: Sequence[str],
    predicted_tests: Sequence[str],
    predicted_proofs: Sequence[str],
    ambiguities: Sequence[str],
    cyclic: Sequence[str],
    shared_states: Sequence[str],
) -> tuple[HardConstraintCheck, ...]:
    """Evaluate every closed hard constraint for one proposed boundary."""

    member_ids = tuple(node.node_id for node in members)
    member_set = set(member_ids)
    owner_node = view.nodes_by_id.get(canonical_owner)
    owner_is_authority = owner_node is not None and owner_node.kind is NodeKind.AUTHORITY
    simulation_members = tuple(
        node.node_id for node in members if node.kind is NodeKind.SIMULATION
    )
    production_using_sim = []
    for edge in view.architecture.edges:
        if edge.target not in set(simulation_members) and edge.source not in set(
            simulation_members
        ):
            continue
        if edge.kind not in _CALL_EDGE_KINDS | _EFFECT_EDGE_KINDS:
            continue
        other = edge.source if edge.target in set(simulation_members) else edge.target
        other_node = view.nodes_by_id.get(other)
        if other_node is None:
            continue
        if other_node.kind in _PRODUCTION_NODE_KINDS and other_node.kind is not NodeKind.SIMULATION:
            if concern is not BoundaryConcern.SIMULATIONS:
                production_using_sim.append(other)
            elif other_node.kind not in {NodeKind.TEST, NodeKind.SIMULATION, NodeKind.COMPATIBILITY}:
                if edge.kind in {EdgeKind.CALLS, EdgeKind.EXECUTES, EdgeKind.IMPLEMENTS}:
                    production_using_sim.append(other)
    heuristic_members = tuple(
        node.node_id
        for node in members
        if node.provenance.confidence in NON_PROBATIVE_CONFIDENCE
    )
    implementation_ids = {
        node.node_id
        for node in view.architecture.nodes
        if node.kind in _IMPLEMENTATION_NODE_KINDS
    }
    unbounded = len(members) > MAX_BOUNDARY_MEMBERS
    if (
        implementation_ids
        and len(implementation_ids) >= 8
        and implementation_ids <= member_set
    ):
        unbounded = True
    sibling_paths = tuple(
        node.provenance.span.path
        for node in members
        if any(node.provenance.span.path.startswith(prefix) for prefix in _CROSS_REPO_PREFIXES)
    )
    secret_paths = tuple(
        node.provenance.span.path
        for node in members
        if any(marker in node.provenance.span.path.lower() for marker in _SECRET_MARKERS)
    )
    public_interfaces = tuple(
        node.node_id for node in members if node.kind is NodeKind.INTERFACE
    )
    missing_migration = tuple(
        node_id
        for node_id in public_interfaces
        if node_id not in set(adapters) and node_id not in set(deprecations)
    )
    expanded_effects = tuple(sorted(set(proposed_effects) - set(effects)))
    lost_callers = ()
    hidden = False
    if concern not in {
        BoundaryConcern.LEGACY_COMPATIBILITY,
        BoundaryConcern.SIMULATIONS,
    }:
        hidden = False
    lost_tests = tuple(sorted(set(tests) - set(predicted_tests)))
    lost_proofs = tuple(sorted(set(proofs) - set(predicted_proofs)))
    multiple_states = len(state_ids) > 1 and bool(shared_states)
    incoherent = len(set(competing_owners)) > 1
    unresolved = not canonical_owner or bool(blocked_concerns) or not owner_is_authority
    if concern in {BoundaryConcern.LEGACY_COMPATIBILITY, BoundaryConcern.SIMULATIONS}:
        if canonical_owner and owner_is_authority:
            unresolved = bool(blocked_concerns)
        elif not members:
            unresolved = True
    simulated_as_live = bool(production_using_sim) or (
        owner_node is not None and owner_node.kind is NodeKind.SIMULATION
    )
    if concern is not BoundaryConcern.SIMULATIONS and simulation_members:
        simulated_as_live = True
    checks = {
        HardConstraintKind.NO_AUTHORITY_WEAKENING: _check(
            HardConstraintKind.NO_AUTHORITY_WEAKENING,
            satisfied=not unresolved and not incoherent and owner_is_authority,
            explanation=(
                "canonical owner is the existing ArchitectureIR authority"
                if not unresolved and not incoherent and owner_is_authority
                else "proposal would create, transfer, or weaken canonical authority"
            ),
            node_ids=(canonical_owner,) if canonical_owner else competing_owners,
        ),
        HardConstraintKind.NO_EFFECT_EXPANSION: _check(
            HardConstraintKind.NO_EFFECT_EXPANSION,
            satisfied=not expanded_effects,
            explanation=(
                "allowed effects are a subset of observed cluster effects"
                if not expanded_effects
                else "proposal would expand effects beyond the observed cluster"
            ),
            node_ids=member_ids,
        ),
        HardConstraintKind.NO_HIDDEN_BEHAVIOR_CHANGE: _check(
            HardConstraintKind.NO_HIDDEN_BEHAVIOR_CHANGE,
            satisfied=not lost_callers and hidden is False,
            explanation=(
                "observed callers and effects remain declared"
                if not lost_callers
                else "proposal would hide observed callers or effects"
            ),
            node_ids=callers,
        ),
        HardConstraintKind.NO_SIMULATED_AS_LIVE: _check(
            HardConstraintKind.NO_SIMULATED_AS_LIVE,
            satisfied=not simulated_as_live,
            explanation=(
                "simulation nodes remain quarantined from production predicates"
                if not simulated_as_live
                else "simulation flow would satisfy a production predicate"
            ),
            node_ids=tuple(sorted(set(simulation_members) | set(production_using_sim))),
        ),
        HardConstraintKind.NO_VALIDATION_REDUCTION: _check(
            HardConstraintKind.NO_VALIDATION_REDUCTION,
            satisfied=not lost_tests,
            explanation=(
                "selected tests are preserved"
                if not lost_tests
                else "proposal would drop tests covering the cluster"
            ),
            node_ids=lost_tests or tests,
        ),
        HardConstraintKind.NO_PROOF_OBLIGATION_LOSS: _check(
            HardConstraintKind.NO_PROOF_OBLIGATION_LOSS,
            satisfied=not lost_proofs,
            explanation=(
                "proof obligations are preserved"
                if not lost_proofs
                else "proposal would drop proof obligations"
            ),
            node_ids=lost_proofs or proofs,
        ),
        HardConstraintKind.NO_PUBLIC_CONTRACT_BREAK_WITHOUT_VERSIONED_MIGRATION: _check(
            HardConstraintKind.NO_PUBLIC_CONTRACT_BREAK_WITHOUT_VERSIONED_MIGRATION,
            satisfied=not missing_migration,
            explanation=(
                "existing public interfaces are versioned adapters or deprecations"
                if not missing_migration
                else "public interface would break without versioned migration"
            ),
            node_ids=missing_migration or public_interfaces,
        ),
        HardConstraintKind.NO_STALE_EVIDENCE_PROMOTION: _check(
            HardConstraintKind.NO_STALE_EVIDENCE_PROMOTION,
            satisfied=not (
                heuristic_members and owner_node is not None
                and owner_node.provenance.confidence in NON_PROBATIVE_CONFIDENCE
            ),
            explanation=(
                "heuristic or opaque facts are not treated as ownership proof"
                if not (
                    heuristic_members
                    and owner_node is not None
                    and owner_node.provenance.confidence in NON_PROBATIVE_CONFIDENCE
                )
                else "heuristic or opaque evidence cannot prove a coherent owner"
            ),
            node_ids=heuristic_members,
        ),
        HardConstraintKind.NO_UNBOUNDED_REFACTOR: _check(
            HardConstraintKind.NO_UNBOUNDED_REFACTOR,
            satisfied=not unbounded,
            explanation=(
                "proposal stays inside the maximum member and cone bound"
                if not unbounded
                else "proposal would refactor an unbounded portion of the graph"
            ),
            node_ids=member_ids,
        ),
        HardConstraintKind.NO_PROCEDURE_SELF_AUTHORIZATION: _check(
            HardConstraintKind.NO_PROCEDURE_SELF_AUTHORIZATION,
            True,
            "proposals cannot authorize themselves",
        ),
        HardConstraintKind.NO_ARCHITECTURE_CANDIDATE_SELF_PROMOTION: _check(
            HardConstraintKind.NO_ARCHITECTURE_CANDIDATE_SELF_PROMOTION,
            True,
            "proposals cannot promote themselves",
        ),
        HardConstraintKind.NO_CROSS_REPOSITORY_WRITE: _check(
            HardConstraintKind.NO_CROSS_REPOSITORY_WRITE,
            satisfied=not sibling_paths,
            explanation=(
                "proposal stays inside the current repository tree"
                if not sibling_paths
                else "proposal would write or own a sibling repository path"
            ),
            node_ids=member_ids,
        ),
        HardConstraintKind.NO_SECRET_OR_PRIVATE_DATA_LEAK: _check(
            HardConstraintKind.NO_SECRET_OR_PRIVATE_DATA_LEAK,
            satisfied=not secret_paths,
            explanation=(
                "proposal does not expose secret or private paths"
                if not secret_paths
                else "proposal would surface secret or private data paths"
            ),
        ),
        HardConstraintKind.NO_FALSE_COMPLETION: _check(
            HardConstraintKind.NO_FALSE_COMPLETION,
            True,
            "proposals are not completion evidence",
        ),
        HardConstraintKind.NO_UNRESOLVED_AMBIGUITY: _check(
            HardConstraintKind.NO_UNRESOLVED_AMBIGUITY,
            satisfied=not ambiguities,
            explanation=(
                "no unresolved contract ambiguity covers this cluster"
                if not ambiguities
                else "unresolved contract ambiguity hard-rejects the proposal"
            ),
            node_ids=member_ids,
        ),
        HardConstraintKind.NO_UNRESOLVED_AUTHORITY: _check(
            HardConstraintKind.NO_UNRESOLVED_AUTHORITY,
            satisfied=not unresolved,
            explanation=(
                "exactly one evidence-backed canonical owner is named"
                if not unresolved
                else "canonical owner is missing, blocked, or not an authority node"
            ),
            node_ids=competing_owners or ((canonical_owner,) if canonical_owner else ()),
        ),
        HardConstraintKind.NO_INCOHERENT_AUTHORITY: _check(
            HardConstraintKind.NO_INCOHERENT_AUTHORITY,
            satisfied=not incoherent,
            explanation=(
                "cluster members share one canonical authority"
                if not incoherent
                else "cluster members are authorized by multiple production authorities"
            ),
            node_ids=competing_owners,
        ),
        HardConstraintKind.NO_UNRESOLVED_STATE_MOVEMENT: _check(
            HardConstraintKind.NO_UNRESOLVED_STATE_MOVEMENT,
            satisfied=not multiple_states,
            explanation=(
                "mutable state has one named owner or is absent"
                if not multiple_states
                else "mutable sharing would move state without a single owner"
            ),
            node_ids=shared_states or state_ids,
        ),
        HardConstraintKind.NO_BOUNDARY_CYCLE: _check(
            HardConstraintKind.NO_BOUNDARY_CYCLE,
            satisfied=not cyclic,
            explanation=(
                "no call-graph cycle crosses the proposed boundary"
                if not cyclic
                else "a call-graph cycle would remain across the proposed boundary"
            ),
            node_ids=cyclic,
        ),
        HardConstraintKind.NO_CROSS_BOUNDARY_MUTABLE_SHARING: _check(
            HardConstraintKind.NO_CROSS_BOUNDARY_MUTABLE_SHARING,
            satisfied=not shared_states,
            explanation=(
                "mutable stores are not shared across the proposed boundary"
                if not shared_states
                else "mutable state is written from both sides of the proposed boundary"
            ),
            node_ids=shared_states,
        ),
        HardConstraintKind.NO_ROLLBACK_LOSS: _check(
            HardConstraintKind.NO_ROLLBACK_LOSS,
            True,
            ROLLBACK_DECLARATION,
        ),
        HardConstraintKind.NO_AUTONOMOUS_APPLICATION: _check(
            HardConstraintKind.NO_AUTONOMOUS_APPLICATION,
            True,
            "proposals cannot be applied autonomously",
        ),
    }
    if concern is BoundaryConcern.SIMULATIONS and simulation_members and not production_using_sim:
        checks[HardConstraintKind.NO_SIMULATED_AS_LIVE] = _check(
            HardConstraintKind.NO_SIMULATED_AS_LIVE,
            True,
            "simulation interface remains a quarantined facade",
            node_ids=simulation_members,
        )
    return tuple(checks[kind] for kind in REQUIRED_HARD_CONSTRAINTS)


def rank_boundary_proposals(
    proposals: Sequence[BoundaryProposal],
) -> tuple[BoundaryProposal, ...]:
    """Order admitted proposals by deterministic non-probative ranking inputs."""

    admitted = [item for item in proposals if item.admitted]
    return tuple(sorted(admitted, key=lambda item: item.ranking_inputs.sort_key()))


def _facade_ids(
    *,
    interface_name: str,
    owner: str,
    state_owner: str,
    tests: Sequence[str],
    proofs: Sequence[str],
    adapters: Sequence[str],
    deprecations: Sequence[str],
) -> tuple[str, ...]:
    found = {interface_name, *tests, *proofs, *adapters, *deprecations}
    if owner:
        found.add(owner)
    if state_owner not in {"none", "unresolved"}:
        found.add(state_owner)
    return tuple(sorted(found))


def _synthesize_concern(
    concern: BoundaryConcern,
    view: _GraphView,
    *,
    owner_map: Mapping[str, str],
    blocked: Sequence[str],
    contracts: ContractExtractionResult | None,
) -> BoundaryProposal:
    seeds = _seed_members(view, concern)
    preview_owners = _authorizers(view, (node.node_id for node in seeds))
    members = _expand_members(view, seeds, set(preview_owners), concern)
    owner, competing, blocked_here = _owner_for_concern(
        concern,
        view=view,
        members=members,
        owner_map=owner_map,
        blocked=blocked,
    )
    if owner:
        members = _expand_members(view, members or seeds, {owner}, concern)
        owner, competing, blocked_here = _owner_for_concern(
            concern,
            view=view,
            members=members,
            owner_map=owner_map,
            blocked=blocked,
        )
    member_ids = {node.node_id for node in members}
    callers = _external_callers(view, members)
    effects = _observed_effects(view, members)
    tests = _related_nodes(view, members, NodeKind.TEST, EdgeKind.TESTS)
    proofs = _related_nodes(view, members, NodeKind.PROOF, EdgeKind.PROVES)
    adapters = _related_nodes(view, members, NodeKind.COMPATIBILITY, EdgeKind.ADAPTS)
    adapters = tuple(
        sorted(
            set(adapters)
            | set(_related_nodes(view, members, NodeKind.COMPATIBILITY, EdgeKind.REEXPORTS))
        )
    )
    interface_members = tuple(
        node.node_id for node in members if node.kind is NodeKind.INTERFACE
    )
    deprecations = tuple(
        sorted(
            set(_related_nodes(view, members, NodeKind.INTERFACE, EdgeKind.DEPRECATES))
            | {
                node.node_id
                for node in members
                if node.kind in _PUBLIC_NODE_KINDS and node.node_id in callers
            }
        )
    )
    deprecations = tuple(
        item for item in sorted(set(deprecations) | set(interface_members)) if item in member_ids
    )
    state_ids = _state_nodes(view, members)
    shared = _mutable_shared_states(view, member_ids)
    cyclic = _cross_boundary_cycle_nodes(view, member_ids)
    if not state_ids:
        state_owner = "none"
    elif len(state_ids) == 1:
        state_owner = state_ids[0]
    elif owner and not shared:
        state_owner = owner
    else:
        state_owner = "unresolved"
    ambiguities = _ambiguities_for(contracts, members)
    interface_name = _INTERFACE_NAMES[concern]
    required = ProposedInterface(
        name=interface_name,
        concern=concern,
        members=tuple(node.node_id for node in members),
        public_symbols=(interface_name,),
    )
    facade = _facade_ids(
        interface_name=interface_name,
        owner=owner,
        state_owner=state_owner,
        tests=tests,
        proofs=proofs,
        adapters=adapters,
        deprecations=deprecations,
    )
    before, predicted_after = measure_boundary_cost(view, members, facade_ids=facade)
    constraints = check_hard_constraints(
        concern=concern,
        view=view,
        members=members,
        canonical_owner=owner,
        competing_owners=competing,
        blocked_concerns=blocked_here,
        state_ids=state_ids,
        callers=callers,
        effects=effects,
        tests=tests,
        proofs=proofs,
        adapters=adapters,
        deprecations=deprecations,
        proposed_effects=effects,
        predicted_tests=tests,
        predicted_proofs=proofs,
        ambiguities=ambiguities,
        cyclic=cyclic,
        shared_states=shared,
    )
    failed = tuple(item for item in constraints if item.satisfied is False)
    if failed:
        after = before
        disposition = ProposalDisposition.REJECTED
        reasons = tuple(sorted({item.constraint.value for item in failed}))
        if not owner:
            owner = ""
        if state_owner == "unresolved" and not state_ids:
            state_owner = "none"
        elif failed and state_owner == "unresolved":
            state_owner = "unresolved"
    else:
        after = predicted_after
        disposition = ProposalDisposition.ADMITTED
        reasons = ()
    cost = BoundaryCost.from_vectors(before, after)
    return BoundaryProposal(
        concern=concern,
        required_interface=required,
        canonical_owner=owner,
        allowed_callers=callers,
        allowed_effects=effects,
        state_owner=state_owner,
        migration_adapters=adapters,
        deprecations=deprecations,
        deprecated_paths=_deprecated_paths(view, deprecations),
        tests=tests,
        proofs=proofs,
        rollback=ROLLBACK_DECLARATION,
        cost=cost,
        hard_constraints=constraints,
        repository_tree=view.architecture.repository_tree,
        freshness=view.architecture.freshness,
        members=required.members,
        predicted_context_reduction=cost.reductions.context_amplification,
        predicted_cone_reduction=cost.reductions.dependency_cone,
        disposition=disposition,
        rejection_reasons=reasons,
    )


def synthesize_boundaries(
    architecture: ArchitectureIR | Mapping[str, Any],
    *,
    ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
    contracts: ContractExtractionResult | Mapping[str, Any] | None = None,
    entropy: SemanticEntropyReport | Mapping[str, Any] | None = None,
    freshness: str | None = None,
) -> BoundarySynthesisResult:
    """Propose stable interfaces for every initial boundary concern."""

    graph = _require_architecture_ir(architecture)
    token = _require_text(
        freshness or graph.freshness,
        "freshness",
        error_type=BoundarySynthesizerError,
    )
    if token != graph.freshness:
        raise BoundarySynthesizerError("freshness must match ArchitectureIR")
    ownership_graph, owner_map, blocked = _normalize_ownership(ownership)
    contract_result = _normalize_contracts(contracts)
    entropy_report = _normalize_entropy(entropy)
    if ownership_graph is not None:
        if ownership_graph.architecture_ir_identity != graph.content_identity:
            raise BoundarySynthesizerError("ownership architecture_ir_identity must match graph")
        if ownership_graph.repository_tree != graph.repository_tree:
            raise BoundarySynthesizerError("ownership repository_tree must match graph")
    if contract_result is not None:
        if (
            contract_result.architecture_ir_identity
            and contract_result.architecture_ir_identity != graph.content_identity
        ):
            raise BoundarySynthesizerError("contract architecture_ir_identity must match graph")
        if contract_result.repository_tree != graph.repository_tree:
            raise BoundarySynthesizerError("contract repository_tree must match graph")
    if entropy_report is not None:
        if entropy_report.architecture_ir_identity != graph.content_identity:
            raise BoundarySynthesizerError("entropy architecture_ir_identity must match graph")
        if entropy_report.repository_tree != graph.repository_tree:
            raise BoundarySynthesizerError("entropy repository_tree must match graph")
    view = _build_view(graph)
    proposals = tuple(
        _synthesize_concern(
            concern,
            view,
            owner_map=owner_map,
            blocked=blocked,
            contracts=contract_result,
        )
        for concern in INITIAL_BOUNDARY_CONCERNS
    )
    return BoundarySynthesisResult(
        repository_tree=graph.repository_tree,
        freshness=token,
        architecture_ir_identity=graph.content_identity,
        proposals=proposals,
        ownership_identity="" if ownership_graph is None else ownership_graph.content_identity,
        contract_identity="" if contract_result is None else contract_result.content_identity,
        entropy_identity="" if entropy_report is None else entropy_report.content_identity,
    )


def build_boundary_synthesis_result(
    architecture: ArchitectureIR | Mapping[str, Any],
    *,
    ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
    contracts: ContractExtractionResult | Mapping[str, Any] | None = None,
    entropy: SemanticEntropyReport | Mapping[str, Any] | None = None,
    freshness: str | None = None,
) -> BoundarySynthesisResult:
    """Alias for :func:`synthesize_boundaries`."""

    return synthesize_boundaries(
        architecture,
        ownership=ownership,
        contracts=contracts,
        entropy=entropy,
        freshness=freshness,
    )


@dataclass(frozen=True)
class InterfaceBoundarySynthesizer:
    """Read-only planner for candidate-tier interface-boundary proposals."""

    extractor_identity: str = EXTRACTOR_IDENTITY
    can_apply: bool = SYNTHESIZER_CAN_APPLY_CANDIDATE
    can_transfer_authority: bool = SYNTHESIZER_CAN_TRANSFER_AUTHORITY
    can_create_authority: bool = SYNTHESIZER_CAN_CREATE_AUTHORITY
    can_promote: bool = SYNTHESIZER_CAN_PROMOTE_CANDIDATE
    can_authorize_changes: bool = SYNTHESIZER_CAN_AUTHORIZE_CHANGES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(
                self.extractor_identity,
                "extractor_identity",
                error_type=BoundarySynthesizerError,
            ),
        )
        if self.can_apply is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot apply a candidate"
            )
        if self.can_transfer_authority is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot transfer an existing authority"
            )
        if self.can_create_authority is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot create authority"
            )
        if self.can_promote is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot promote a candidate"
            )
        if self.can_authorize_changes is not False:
            raise BoundarySynthesizerAuthorityError(
                "boundary synthesizer cannot authorize changes"
            )
        object.__setattr__(self, "can_apply", False)
        object.__setattr__(self, "can_transfer_authority", False)
        object.__setattr__(self, "can_create_authority", False)
        object.__setattr__(self, "can_promote", False)
        object.__setattr__(self, "can_authorize_changes", False)

    def synthesize(
        self,
        architecture: ArchitectureIR | Mapping[str, Any],
        *,
        ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
        contracts: ContractExtractionResult | Mapping[str, Any] | None = None,
        entropy: SemanticEntropyReport | Mapping[str, Any] | None = None,
        freshness: str | None = None,
    ) -> BoundarySynthesisResult:
        return synthesize_boundaries(
            architecture,
            ownership=ownership,
            contracts=contracts,
            entropy=entropy,
            freshness=freshness,
        )

    def rank(
        self, proposals: Sequence[BoundaryProposal]
    ) -> tuple[BoundaryProposal, ...]:
        return rank_boundary_proposals(proposals)

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_candidate_application("apply")

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_candidate_promotion("promote")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    def create_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_creation("create")

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_change_authorization("change")


def refuse_candidate_application(action: str) -> None:
    """Reject attempts to apply a boundary proposal."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"boundary synthesizer cannot {name} a candidate"
    )


def refuse_candidate_promotion(action: str) -> None:
    """Reject attempts to promote a boundary proposal."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"boundary synthesizer cannot {name} a candidate"
    )


def refuse_authority_transfer(action: str) -> None:
    """Reject attempts to transfer an existing authority through a proposal."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"boundary synthesizer cannot {name} an existing authority"
    )


def refuse_authority_creation(action: str) -> None:
    """Reject attempts to create authority through a proposal."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"boundary synthesizer cannot {name} authority"
    )


def refuse_change_authorization(action: str) -> None:
    """Reject attempts to treat a proposal as change authority."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"boundary synthesizer cannot authorize {name}"
    )


def initial_boundary_concerns() -> tuple[BoundaryConcern, ...]:
    """Return the closed initial boundary-concern set."""

    return INITIAL_BOUNDARY_CONCERNS


def hard_constraints_include_non_compensable_invariants() -> bool:
    """Plan non-compensable invariants remain hard constraints."""

    return set(NON_COMPENSABLE_INVARIANTS) <= CLOSED_HARD_CONSTRAINTS
