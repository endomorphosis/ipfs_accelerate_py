"""Interface-boundary synthesis around coherent authorities (PCAR-011).

`InterfaceBoundarySynthesizer` proposes smaller stable candidate interfaces
around provider capability/selection, execution requests/outcomes,
analysis/context, proof and verification scheduling, task/objective state,
control operations, receipt/evidence queries, legacy compatibility, and
simulations. Every proposal names the required interface, canonical owner,
allowed callers and effects, state owner, migration adapters, deprecations,
tests, proofs, rollback, and predicted context and cone reductions.

Proposals preserve existing canonical owners and cannot create, transfer, or
promote authority. They name owner and migration but perform no state change
and cannot apply themselves. Unresolved ownership, contract ambiguity, dual
state authority, cross-boundary cycles, effect expansion, missing rollback,
and incomplete declarations hard-reject the affected proposal before ranking.
Cost measures remain independently auditable and never compensate for a hard
constraint.
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
from .authority_graph import (
    AuthorityOwnershipGraph,
    ConcernKind,
    OwnershipBlocker,
)
from .contract_extractor import ContractAmbiguity, ContractExtractionResult
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    NON_PROBATIVE_CONFIDENCE,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
)
from .entropy import NON_COMPENSABLE_INVARIANTS

BOUNDARY_PROPOSAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-proposal@1"
)
BOUNDARY_PROPOSAL_VERSION = 1
BOUNDARY_PROPOSAL_EVIDENCE = "pcar/boundary-proposal@1"
BOUNDARY_REJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-rejection@1"
)
BOUNDARY_REJECTION_VERSION = 1
BOUNDARY_SYNTHESIS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-synthesis-result@1"
)
BOUNDARY_SYNTHESIS_VERSION = 1
BOUNDARY_COST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-cost-measure@1"
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
BOUNDARY_INTERFACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-interface@1"
)
BOUNDARY_INTERFACE_VERSION = 1
BOUNDARY_MIGRATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-migration@1"
)
BOUNDARY_MIGRATION_VERSION = 1
BOUNDARY_ROLLBACK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-rollback@1"
)
BOUNDARY_ROLLBACK_VERSION = 1
BOUNDARY_PREDICTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-prediction@1"
)
BOUNDARY_PREDICTION_VERSION = 1
RANKING_INPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/boundary-ranking-input@1"
)
RANKING_INPUT_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-011-interface-boundary-synthesizer"
TASK_ID = "PCAR-011"
DEFAULT_FRESHNESS = "pcar-011-boundary-synthesis"
EFFECT_CLASS = "read_only_planning"
SYNTHESIZER_CAN_AUTHORIZE_CHANGES = False
SYNTHESIZER_CAN_TRANSFER_AUTHORITY = False
SYNTHESIZER_CAN_PROMOTE_AUTHORITY = False
SYNTHESIZER_CAN_MUTATE_STATE = False
SYNTHESIZER_CAN_APPLY_PROPOSALS = False
SYNTHESIZER_CAN_OVERRIDE_HARD_CONSTRAINTS = False
RANKING_IS_NON_PROBATIVE = True
CANDIDATE_INTERFACES_ONLY = True
HARD_CONSTRAINTS_PRECEDE_RANKING = True
UNRESOLVED_AMBIGUITY_REJECTS = True
UNRESOLVED_AUTHORITY_REJECTS = True

_UNKNOWN_FIELD_MESSAGE = "unknown boundary-synthesis field"
_MISSING_FIELD_MESSAGE = "missing boundary-synthesis field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_SIBLING_PREFIXES = ("ipfs_datasets_py/", "ipfs_kit_py/", "ipfs_accelerate_py/mcplusplus/")
_ROLLBACK_ACTION = "revert_candidate_interface"
_MIGRATION_PHASES = (
    "declare_interface",
    "adapt_callers",
    "deprecate_paths",
    "validate_and_seal",
)
_EFFECT_EDGE_KINDS = frozenset(
    {
        EdgeKind.READS,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.PERSISTS,
        EdgeKind.OBSERVES,
        EdgeKind.EXECUTES,
        EdgeKind.GENERATES,
        EdgeKind.SERIALIZES,
        EdgeKind.DESERIALIZES,
    }
)
_CALLER_EDGE_KINDS = frozenset(
    {
        EdgeKind.CALLS,
        EdgeKind.EXECUTES,
        EdgeKind.IMPORTS,
        EdgeKind.CONSTRUCTS,
        EdgeKind.IMPLEMENTS,
        EdgeKind.ADAPTS,
        EdgeKind.FALLBACKS_TO,
        EdgeKind.REEXPORTS,
    }
)
_CYCLE_EDGE_KINDS = frozenset(
    {
        EdgeKind.CALLS,
        EdgeKind.IMPORTS,
        EdgeKind.EXECUTES,
        EdgeKind.CONSTRUCTS,
        EdgeKind.FALLBACKS_TO,
        EdgeKind.ADAPTS,
        EdgeKind.REEXPORTS,
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
_PUBLIC_NODE_KINDS = frozenset(
    {NodeKind.INTERFACE, NodeKind.ENTRYPOINT, NodeKind.SYMBOL}
)
_CLUSTER_EXPAND_EDGE_KINDS = frozenset(
    {
        EdgeKind.AUTHORIZES,
        EdgeKind.IMPLEMENTS,
        EdgeKind.CONTAINS,
        EdgeKind.PERSISTS,
        EdgeKind.ADAPTS,
        EdgeKind.GENERATES,
        EdgeKind.SUPERSEDES,
        EdgeKind.DEPRECATES,
    }
)


class BoundarySynthesizerError(ArchitectureContractError):
    """Fail-closed interface-boundary synthesis error."""


class BoundarySynthesizerAuthorityError(BoundarySynthesizerError):
    """Raised when synthesis is asked to authorize, transfer, mutate, or apply."""


class BoundaryKind(str, Enum):
    """Closed initial interface-boundary vocabulary (PCAR-PLAN-R1)."""

    PROVIDER_CAPABILITY_SELECTION = "provider_capability_selection"
    EXECUTION_REQUEST_OUTCOME = "execution_request_outcome"
    ANALYSIS_CONTEXT = "analysis_context"
    PROOF_VERIFICATION_SCHEDULING = "proof_verification_scheduling"
    TASK_OBJECTIVE_STATE = "task_objective_state"
    CONTROL_OPERATIONS = "control_operations"
    RECEIPT_EVIDENCE_QUERY = "receipt_evidence_query"
    LEGACY_COMPATIBILITY = "legacy_compatibility"
    SIMULATION = "simulation"


INITIAL_BOUNDARIES: tuple[BoundaryKind, ...] = tuple(BoundaryKind)
CLOSED_BOUNDARIES: frozenset[str] = frozenset(item.value for item in BoundaryKind)
REQUIRED_BOUNDARIES: tuple[BoundaryKind, ...] = INITIAL_BOUNDARIES


class CostDimensionKind(str, Enum):
    """Closed independently auditable boundary-cost vocabulary."""

    CROSS_BOUNDARY_EFFECTS = "CrossBoundaryEffects"
    MUTABLE_SHARING = "MutableSharing"
    CYCLES = "Cycles"
    PUBLIC_SYMBOLS = "PublicSymbols"
    CHANGE_AMPLIFICATION = "ChangeAmplification"
    CONTEXT_BURDEN = "ContextBurden"
    VALIDATION_AMPLIFICATION = "ValidationAmplification"
    DEPENDENCY_CONE = "DependencyCone"


REQUIRED_COST_DIMENSIONS: tuple[CostDimensionKind, ...] = tuple(CostDimensionKind)
CLOSED_COST_DIMENSIONS: frozenset[str] = frozenset(
    item.value for item in CostDimensionKind
)
_COST_UNITS: dict[CostDimensionKind, str] = {
    CostDimensionKind.CROSS_BOUNDARY_EFFECTS: "effects",
    CostDimensionKind.MUTABLE_SHARING: "states",
    CostDimensionKind.CYCLES: "cycles",
    CostDimensionKind.PUBLIC_SYMBOLS: "symbols",
    CostDimensionKind.CHANGE_AMPLIFICATION: "amplified_units",
    CostDimensionKind.CONTEXT_BURDEN: "nodes",
    CostDimensionKind.VALIDATION_AMPLIFICATION: "validations",
    CostDimensionKind.DEPENDENCY_CONE: "nodes",
}
CLOSED_COST_UNITS: frozenset[str] = frozenset(_COST_UNITS.values())


class HardConstraintKind(str, Enum):
    """Closed hard-gate vocabulary. Ranking cannot override these."""

    NO_AUTHORITY_WEAKENING = "NoAuthorityWeakening"
    NO_EFFECT_EXPANSION = "NoEffectExpansion"
    NO_HIDDEN_BEHAVIOR_CHANGE = "NoHiddenBehaviorChange"
    NO_SIMULATED_AS_LIVE = "NoSimulatedAsLive"
    NO_VALIDATION_REDUCTION = "NoValidationReduction"
    NO_PROOF_OBLIGATION_LOSS = "NoProofObligationLoss"
    NO_PUBLIC_CONTRACT_BREAK = "NoPublicContractBreakWithoutVersionedMigration"
    NO_STALE_EVIDENCE_PROMOTION = "NoStaleEvidencePromotion"
    NO_UNBOUNDED_REFACTOR = "NoUnboundedRefactor"
    NO_PROCEDURE_SELF_AUTHORIZATION = "NoProcedureSelfAuthorization"
    NO_ARCHITECTURE_CANDIDATE_SELF_PROMOTION = "NoArchitectureCandidateSelfPromotion"
    NO_CROSS_REPOSITORY_WRITE = "NoCrossRepositoryWrite"
    NO_SECRET_OR_PRIVATE_DATA_LEAK = "NoSecretOrPrivateDataLeak"
    NO_FALSE_COMPLETION = "NoFalseCompletion"
    UNRESOLVED_AUTHORITY = "UnresolvedAuthority"
    UNRESOLVED_AMBIGUITY = "UnresolvedAmbiguity"
    DUAL_STATE_AUTHORITY = "DualStateAuthority"
    MISSING_ROLLBACK = "MissingRollback"
    INCOMPLETE_PROPOSAL = "IncompleteProposal"
    AUTHORITY_TRANSFER = "AuthorityTransfer"
    STATE_MOVEMENT = "StateMovement"
    CROSS_BOUNDARY_CYCLE = "CrossBoundaryCycle"
    MUTABLE_SHARING = "MutableSharing"
    SCOPE_ESCAPE = "ScopeEscape"


REQUIRED_HARD_CONSTRAINTS: tuple[HardConstraintKind, ...] = tuple(HardConstraintKind)
CLOSED_HARD_CONSTRAINTS: frozenset[str] = frozenset(
    item.value for item in HardConstraintKind
)
MISSING_PLAN_HARD_GATES: tuple[str, ...] = tuple(
    name for name in NON_COMPENSABLE_INVARIANTS if name not in CLOSED_HARD_CONSTRAINTS
)


class ProposalDisposition(str, Enum):
    """Closed proposal admission vocabulary."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


CLOSED_PROPOSAL_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in ProposalDisposition
)


class RejectionKind(str, Enum):
    """Closed hard-reject vocabulary retained with explanations."""

    HARD_CONSTRAINT = "hard_constraint"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    UNRESOLVED_AUTHORITY = "unresolved_authority"
    INCOMPLETE_PROPOSAL = "incomplete_proposal"
    AUTHORITY_TRANSFER = "authority_transfer"
    STATE_MOVEMENT = "state_movement"
    AUTONOMOUS_APPLICATION = "autonomous_application"
    CYCLE = "cycle"
    EFFECT_EXPANSION = "effect_expansion"
    MUTABLE_SHARING = "mutable_sharing"


CLOSED_REJECTION_KINDS: frozenset[str] = frozenset(item.value for item in RejectionKind)


class InterfaceStability(str, Enum):
    """Closed stability vocabulary. Synthesis emits candidate interfaces only."""

    CANDIDATE = "candidate"


CLOSED_INTERFACE_STABILITIES: frozenset[str] = frozenset(
    item.value for item in InterfaceStability
)

BOUNDARY_CONCERNS: dict[BoundaryKind, tuple[ConcernKind, ...]] = {
    BoundaryKind.PROVIDER_CAPABILITY_SELECTION: (
        ConcernKind.PROVIDER_CAPABILITY,
        ConcernKind.PROVIDER_SELECTION,
    ),
    BoundaryKind.EXECUTION_REQUEST_OUTCOME: (ConcernKind.EXECUTION_RESULT,),
    BoundaryKind.ANALYSIS_CONTEXT: (ConcernKind.CONTENT_IDENTITY,),
    BoundaryKind.PROOF_VERIFICATION_SCHEDULING: (
        ConcernKind.PROOF_VERIFICATION,
        ConcernKind.TEST_EVIDENCE,
    ),
    BoundaryKind.TASK_OBJECTIVE_STATE: (
        ConcernKind.TASK_IDENTITY,
        ConcernKind.OBJECTIVE_IDENTITY,
        ConcernKind.LEASE_AND_FENCING,
        ConcernKind.STATE_PERSISTENCE,
    ),
    BoundaryKind.CONTROL_OPERATIONS: (
        ConcernKind.POLICY_DECISION,
        ConcernKind.AUTHORIZATION,
        ConcernKind.CONFIRMATION,
        ConcernKind.OPERATION_IDENTITY,
    ),
    BoundaryKind.RECEIPT_EVIDENCE_QUERY: (
        ConcernKind.COMPLETION_EVIDENCE,
        ConcernKind.RELEASE_QUALIFICATION,
    ),
    BoundaryKind.LEGACY_COMPATIBILITY: (ConcernKind.AUTHORIZATION,),
    BoundaryKind.SIMULATION: (ConcernKind.PROVIDER_CAPABILITY,),
}

_INTERFACE_FIELDS = frozenset(
    {
        "allowed_callers",
        "allowed_effects",
        "canonical_owner_node_id",
        "concerns",
        "content_identity",
        "kind",
        "name",
        "public_symbols",
        "schema",
        "stability",
        "version",
    }
)
_MIGRATION_FIELDS = frozenset(
    {
        "adapters",
        "content_identity",
        "deprecated_paths",
        "mutates_state",
        "phases",
        "schema",
        "state_owner_node_id",
        "transfers_authority",
        "version",
    }
)
_ROLLBACK_FIELDS = frozenset(
    {
        "action",
        "applied_effects",
        "content_identity",
        "message",
        "restores_tree",
        "schema",
        "version",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "cone_reduction",
        "content_identity",
        "context_reduction",
        "current_cone_size",
        "current_context_nodes",
        "current_public_symbols",
        "current_validation_units",
        "proposed_cone_size",
        "proposed_context_nodes",
        "proposed_public_symbols",
        "proposed_validation_units",
        "public_symbol_reduction",
        "schema",
        "validation_amplification_reduction",
        "validation_coverage_loss",
        "version",
    }
)
_COST_FIELDS = frozenset(
    {
        "content_identity",
        "denominator",
        "evidence_edge_ids",
        "evidence_node_ids",
        "kind",
        "numerator",
        "schema",
        "unit",
        "version",
    }
)
_COST_VECTOR_FIELDS = frozenset(
    {
        "content_identity",
        "measures",
        "schema",
        "total_numerator",
        "version",
    }
)
_CONSTRAINT_FIELDS = frozenset(
    {
        "content_identity",
        "evidence_edge_ids",
        "evidence_node_ids",
        "kind",
        "message",
        "passed",
        "schema",
        "version",
    }
)
_RANKING_FIELDS = frozenset(
    {
        "content_identity",
        "costs",
        "kind",
        "schema",
        "total_numerator",
        "version",
    }
)
_PROPOSAL_FIELDS = frozenset(
    {
        "adapters",
        "callers",
        "canonical_owner_node_id",
        "content_identity",
        "costs",
        "deprecated_paths",
        "disposition",
        "effects",
        "freshness",
        "hard_constraints",
        "interface",
        "kind",
        "migration",
        "prediction",
        "proofs",
        "repository_tree",
        "rollback",
        "schema",
        "state_owner_node_id",
        "tests",
        "version",
    }
)
_REJECTION_FIELDS = frozenset(
    {
        "canonical_owner_node_id",
        "content_identity",
        "costs",
        "disposition",
        "failed_constraints",
        "freshness",
        "hard_constraints",
        "kind",
        "message",
        "node_ids",
        "rejection_kind",
        "repository_tree",
        "schema",
        "version",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "can_apply_proposals",
        "can_authorize_changes",
        "can_mutate_state",
        "can_override_hard_constraints",
        "can_promote_authority",
        "can_transfer_authority",
        "candidate_interfaces_only",
        "content_identity",
        "covers_initial_boundaries",
        "effect_class",
        "freshness",
        "ownership_identity",
        "proposals",
        "ranking",
        "ranking_inputs",
        "ranking_is_non_probative",
        "rejections",
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


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoundarySynthesizerError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=BoundarySynthesizerError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _require_ordered_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoundarySynthesizerError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=BoundarySynthesizerError)
        for item in value
    )
    if len(items) != len(set(items)):
        raise BoundarySynthesizerError(f"{name} must be unique")
    return items


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise BoundarySynthesizerError(f"{name} must be a boolean")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    number = _require_int(value, name, error_type=BoundarySynthesizerError)
    if number < 0:
        raise BoundarySynthesizerError(f"{name} must be a non-negative integer")
    return number


def _require_optional_denominator(value: Any, name: str) -> int | None:
    if value is None:
        return None
    number = _require_non_negative_int(value, name)
    if number == 0:
        raise BoundarySynthesizerError(f"{name} must be a positive integer or null")
    return number


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
        raise BoundarySynthesizerError(str(exc)) from exc


def _require_ownership(
    graph: AuthorityOwnershipGraph | Mapping[str, Any],
) -> AuthorityOwnershipGraph:
    if isinstance(graph, AuthorityOwnershipGraph):
        return graph
    try:
        return AuthorityOwnershipGraph.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise BoundarySynthesizerError(str(exc)) from exc


def _record_tuple(value: Any, name: str, record_type: type[Any]) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoundarySynthesizerError(f"{name} must be a list of objects")
    records = tuple(
        item if isinstance(item, record_type) else record_type.from_mapping(item)
        for item in value
    )
    return tuple(sorted(records, key=lambda item: item.content_identity))


def _interface_name(kind: BoundaryKind) -> str:
    return f"pcar.boundary.{kind.value}@1"


def refuse_authority_transfer(action: str = "transfer") -> None:
    """Reject attempts to create, move, or replace a canonical owner."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"interface-boundary synthesizer cannot {name} an existing authority"
    )


def refuse_authority_promotion(action: str = "promote") -> None:
    """Reject attempts to promote a candidate interface into authority."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"interface-boundary synthesizer cannot {name} authority"
    )


def refuse_state_mutation(action: str = "mutate") -> None:
    """Reject attempts to perform a state change through a proposal."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"interface-boundary synthesizer cannot {name} state; it names owner and migration only"
    )


def refuse_autonomous_application(action: str = "apply") -> None:
    """Reject attempts to execute a candidate interface autonomously."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"interface-boundary synthesizer cannot {name} a proposal"
    )


def refuse_hard_constraint_override(action: str = "override") -> None:
    """Reject attempts to admit an unsafe proposal because its cost is lower."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"boundary ranking cannot {name} a hard constraint"
    )


def refuse_ownership_authorization(action: str = "authorize") -> None:
    """Reject attempts to treat synthesis as change authority."""

    name = _require_text(action, "action", error_type=BoundarySynthesizerError)
    raise BoundarySynthesizerAuthorityError(
        f"interface-boundary synthesizer cannot {name} changes"
    )


@dataclass(frozen=True)
class BoundaryInterface:
    """Candidate interface declared by one boundary proposal."""

    name: str
    kind: BoundaryKind
    canonical_owner_node_id: str
    concerns: tuple[str, ...] = ()
    allowed_callers: tuple[str, ...] = ()
    allowed_effects: tuple[str, ...] = ()
    public_symbols: tuple[str, ...] = ()
    stability: InterfaceStability = InterfaceStability.CANDIDATE
    schema: str = BOUNDARY_INTERFACE_SCHEMA
    version: int = BOUNDARY_INTERFACE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_INTERFACE_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-interface schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_INTERFACE_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-interface version")
        name = _require_text(self.name, "name", error_type=BoundarySynthesizerError)
        kind = _closed_enum(
            self.kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        owner = _require_text(
            self.canonical_owner_node_id,
            "canonical_owner_node_id",
            error_type=BoundarySynthesizerError,
        )
        if _looks_like_content_identity(owner):
            raise BoundarySynthesizerError(
                "content identity is not inferred to be authority"
            )
        stability = _closed_enum(
            self.stability,
            InterfaceStability,
            "interface stability",
            error_type=BoundarySynthesizerError,
        )
        if stability is not InterfaceStability.CANDIDATE:
            raise BoundarySynthesizerAuthorityError(
                "synthesized interfaces remain candidate-tier"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "canonical_owner_node_id", owner)
        object.__setattr__(self, "concerns", _require_text_tuple(self.concerns, "concerns"))
        object.__setattr__(
            self, "allowed_callers", _require_text_tuple(self.allowed_callers, "allowed_callers")
        )
        object.__setattr__(
            self, "allowed_effects", _require_text_tuple(self.allowed_effects, "allowed_effects")
        )
        object.__setattr__(
            self, "public_symbols", _require_text_tuple(self.public_symbols, "public_symbols")
        )
        object.__setattr__(self, "stability", InterfaceStability.CANDIDATE)
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
            "allowed_callers": list(self.allowed_callers),
            "allowed_effects": list(self.allowed_effects),
            "canonical_owner_node_id": self.canonical_owner_node_id,
            "concerns": list(self.concerns),
            "kind": self.kind.value,
            "name": self.name,
            "public_symbols": list(self.public_symbols),
            "schema": self.schema,
            "stability": InterfaceStability.CANDIDATE.value,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-interface content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryInterface":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _INTERFACE_FIELDS)
        record = cls(
            name=mapping["name"],
            kind=mapping["kind"],
            canonical_owner_node_id=mapping["canonical_owner_node_id"],
            concerns=mapping["concerns"],
            allowed_callers=mapping["allowed_callers"],
            allowed_effects=mapping["allowed_effects"],
            public_symbols=mapping["public_symbols"],
            stability=mapping["stability"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-interface content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryMigration:
    """Named migration that is not executed by synthesis."""

    state_owner_node_id: str
    adapters: tuple[str, ...] = ()
    deprecated_paths: tuple[str, ...] = ()
    phases: tuple[str, ...] = _MIGRATION_PHASES
    mutates_state: bool = False
    transfers_authority: bool = False
    schema: str = BOUNDARY_MIGRATION_SCHEMA
    version: int = BOUNDARY_MIGRATION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_MIGRATION_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-migration schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_MIGRATION_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-migration version")
        owner = _require_text(
            self.state_owner_node_id,
            "state_owner_node_id",
            error_type=BoundarySynthesizerError,
        )
        mutates = _require_bool(self.mutates_state, "mutates_state")
        transfers = _require_bool(self.transfers_authority, "transfers_authority")
        if mutates:
            refuse_state_mutation("mutate")
        if transfers:
            refuse_authority_transfer("transfer")
        phases = _require_ordered_text_tuple(self.phases, "phases")
        if phases != _MIGRATION_PHASES:
            raise BoundarySynthesizerError("unexpected boundary-migration phases")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "state_owner_node_id", owner)
        object.__setattr__(self, "adapters", _require_text_tuple(self.adapters, "adapters"))
        object.__setattr__(
            self,
            "deprecated_paths",
            _require_text_tuple(self.deprecated_paths, "deprecated_paths"),
        )
        object.__setattr__(self, "phases", _MIGRATION_PHASES)
        object.__setattr__(self, "mutates_state", False)
        object.__setattr__(self, "transfers_authority", False)
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
                raise BoundarySynthesizerError("boundary-migration content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "adapters": list(self.adapters),
            "deprecated_paths": list(self.deprecated_paths),
            "mutates_state": False,
            "phases": list(self.phases),
            "schema": self.schema,
            "state_owner_node_id": self.state_owner_node_id,
            "transfers_authority": False,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-migration content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryMigration":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _MIGRATION_FIELDS)
        record = cls(
            state_owner_node_id=mapping["state_owner_node_id"],
            adapters=mapping["adapters"],
            deprecated_paths=mapping["deprecated_paths"],
            phases=mapping["phases"],
            mutates_state=mapping["mutates_state"],
            transfers_authority=mapping["transfers_authority"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-migration content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryRollback:
    """Exact rollback for an unapplied candidate interface."""

    action: str = _ROLLBACK_ACTION
    message: str = "revert the unapplied candidate interface; no state was changed"
    applied_effects: bool = False
    restores_tree: bool = True
    schema: str = BOUNDARY_ROLLBACK_SCHEMA
    version: int = BOUNDARY_ROLLBACK_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_ROLLBACK_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-rollback schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_ROLLBACK_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-rollback version")
        action = _require_text(self.action, "action", error_type=BoundarySynthesizerError)
        if action != _ROLLBACK_ACTION:
            raise BoundarySynthesizerError("unexpected boundary-rollback action")
        message = _require_text(self.message, "message", error_type=BoundarySynthesizerError)
        applied = _require_bool(self.applied_effects, "applied_effects")
        restores = _require_bool(self.restores_tree, "restores_tree")
        if applied:
            raise BoundarySynthesizerError("boundary proposals perform no applied effects")
        if restores is not True:
            raise BoundarySynthesizerError("boundary rollback must restore the sealed tree")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "action", _ROLLBACK_ACTION)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "applied_effects", False)
        object.__setattr__(self, "restores_tree", True)
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
                raise BoundarySynthesizerError("boundary-rollback content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "applied_effects": False,
            "message": self.message,
            "restores_tree": True,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-rollback content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryRollback":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _ROLLBACK_FIELDS)
        record = cls(
            action=mapping["action"],
            message=mapping["message"],
            applied_effects=mapping["applied_effects"],
            restores_tree=mapping["restores_tree"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-rollback content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryPrediction:
    """Predicted context, cone, public-symbol, and validation-amplification deltas."""

    current_cone_size: int
    proposed_cone_size: int
    cone_reduction: int
    current_context_nodes: int
    proposed_context_nodes: int
    context_reduction: int
    current_public_symbols: int
    proposed_public_symbols: int
    public_symbol_reduction: int
    current_validation_units: int
    proposed_validation_units: int
    validation_amplification_reduction: int
    validation_coverage_loss: int
    schema: str = BOUNDARY_PREDICTION_SCHEMA
    version: int = BOUNDARY_PREDICTION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_PREDICTION_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-prediction schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_PREDICTION_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-prediction version")
        fields = (
            "current_cone_size",
            "proposed_cone_size",
            "cone_reduction",
            "current_context_nodes",
            "proposed_context_nodes",
            "context_reduction",
            "current_public_symbols",
            "proposed_public_symbols",
            "public_symbol_reduction",
            "current_validation_units",
            "proposed_validation_units",
            "validation_amplification_reduction",
            "validation_coverage_loss",
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        for name in fields:
            object.__setattr__(
                self, name, _require_non_negative_int(getattr(self, name), name)
            )
        expected_cone = max(0, self.current_cone_size - self.proposed_cone_size)
        expected_context = max(0, self.current_context_nodes - self.proposed_context_nodes)
        expected_public = max(0, self.current_public_symbols - self.proposed_public_symbols)
        expected_validation = max(
            0, self.current_validation_units - self.proposed_validation_units
        )
        if self.cone_reduction != expected_cone:
            raise BoundarySynthesizerError("cone_reduction must equal the documented delta")
        if self.context_reduction != expected_context:
            raise BoundarySynthesizerError(
                "context_reduction must equal the documented delta"
            )
        if self.public_symbol_reduction != expected_public:
            raise BoundarySynthesizerError(
                "public_symbol_reduction must equal the documented delta"
            )
        if self.validation_amplification_reduction != expected_validation:
            raise BoundarySynthesizerError(
                "validation_amplification_reduction must equal the documented delta"
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
                    "boundary-prediction content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "cone_reduction": self.cone_reduction,
            "context_reduction": self.context_reduction,
            "current_cone_size": self.current_cone_size,
            "current_context_nodes": self.current_context_nodes,
            "current_public_symbols": self.current_public_symbols,
            "current_validation_units": self.current_validation_units,
            "proposed_cone_size": self.proposed_cone_size,
            "proposed_context_nodes": self.proposed_context_nodes,
            "proposed_public_symbols": self.proposed_public_symbols,
            "proposed_validation_units": self.proposed_validation_units,
            "public_symbol_reduction": self.public_symbol_reduction,
            "schema": self.schema,
            "validation_amplification_reduction": self.validation_amplification_reduction,
            "validation_coverage_loss": self.validation_coverage_loss,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-prediction content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryPrediction":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _PREDICTION_FIELDS)
        record = cls(
            current_cone_size=mapping["current_cone_size"],
            proposed_cone_size=mapping["proposed_cone_size"],
            cone_reduction=mapping["cone_reduction"],
            current_context_nodes=mapping["current_context_nodes"],
            proposed_context_nodes=mapping["proposed_context_nodes"],
            context_reduction=mapping["context_reduction"],
            current_public_symbols=mapping["current_public_symbols"],
            proposed_public_symbols=mapping["proposed_public_symbols"],
            public_symbol_reduction=mapping["public_symbol_reduction"],
            current_validation_units=mapping["current_validation_units"],
            proposed_validation_units=mapping["proposed_validation_units"],
            validation_amplification_reduction=mapping[
                "validation_amplification_reduction"
            ],
            validation_coverage_loss=mapping["validation_coverage_loss"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-prediction content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryCostMeasure:
    """One independently auditable cost dimension for a proposed boundary."""

    kind: CostDimensionKind
    numerator: int
    unit: str
    denominator: int | None = None
    evidence_node_ids: tuple[str, ...] = ()
    evidence_edge_ids: tuple[str, ...] = ()
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
        kind = _closed_enum(
            self.kind,
            CostDimensionKind,
            "cost dimension",
            error_type=BoundarySynthesizerError,
        )
        unit = _require_text(self.unit, "unit", error_type=BoundarySynthesizerError)
        if unit != _COST_UNITS[kind]:
            raise BoundarySynthesizerError(f"unexpected unit for {kind.value}")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(
            self, "numerator", _require_non_negative_int(self.numerator, "numerator")
        )
        object.__setattr__(
            self,
            "denominator",
            _require_optional_denominator(self.denominator, "denominator"),
        )
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
                raise BoundarySynthesizerError("boundary-cost content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "denominator": self.denominator,
            "evidence_edge_ids": list(self.evidence_edge_ids),
            "evidence_node_ids": list(self.evidence_node_ids),
            "kind": self.kind.value,
            "numerator": self.numerator,
            "schema": self.schema,
            "unit": self.unit,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("boundary-cost content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryCostMeasure":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _COST_FIELDS)
        record = cls(
            kind=mapping["kind"],
            numerator=mapping["numerator"],
            unit=mapping["unit"],
            denominator=mapping["denominator"],
            evidence_node_ids=mapping["evidence_node_ids"],
            evidence_edge_ids=mapping["evidence_edge_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("boundary-cost content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryCostVector:
    """Closed cost vector retained as ranking input, never as safety proof."""

    measures: tuple[BoundaryCostMeasure, ...]
    total_numerator: int = 0
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
        if isinstance(self.measures, (str, bytes, bytearray)) or not isinstance(
            self.measures, Sequence
        ):
            raise BoundarySynthesizerError("measures must be a list of objects")
        parsed = tuple(
            item if isinstance(item, BoundaryCostMeasure) else BoundaryCostMeasure.from_mapping(item)
            for item in self.measures
        )
        by_kind = {item.kind: item for item in parsed}
        if len(by_kind) != len(parsed):
            raise BoundarySynthesizerError("cost measures must be unique by kind")
        missing = [item.value for item in REQUIRED_COST_DIMENSIONS if item not in by_kind]
        if missing:
            raise BoundarySynthesizerError(f"missing cost dimensions: {missing}")
        ordered = tuple(by_kind[kind] for kind in REQUIRED_COST_DIMENSIONS)
        total = sum(item.numerator for item in ordered)
        claimed_total = _require_non_negative_int(self.total_numerator, "total_numerator")
        if claimed_total not in {0, total} and claimed_total != total:
            raise BoundarySynthesizerError(
                "total_numerator must equal the documented cost-component sum"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "measures", ordered)
        object.__setattr__(self, "total_numerator", total)
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
                    "boundary-cost-vector content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def measure(self, kind: CostDimensionKind | str) -> BoundaryCostMeasure:
        parsed = _closed_enum(
            kind,
            CostDimensionKind,
            "cost dimension",
            error_type=BoundarySynthesizerError,
        )
        for item in self.measures:
            if item.kind is parsed:
                return item
        raise BoundarySynthesizerError(f"missing cost dimension: {parsed.value}")

    def ranking_key(self) -> tuple[int, ...]:
        return (self.total_numerator, *(item.numerator for item in self.measures))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "measures": [item.to_dict() for item in self.measures],
            "schema": self.schema,
            "total_numerator": self.total_numerator,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError(
                "boundary-cost-vector content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryCostVector":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _COST_VECTOR_FIELDS)
        record = cls(
            measures=mapping["measures"],
            total_numerator=mapping["total_numerator"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError(
                "boundary-cost-vector content identity mismatch"
            )
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class HardConstraintCheck:
    """One hard-gate evaluation with an auditable explanation."""

    kind: HardConstraintKind
    passed: bool
    message: str
    evidence_node_ids: tuple[str, ...] = ()
    evidence_edge_ids: tuple[str, ...] = ()
    schema: str = HARD_CONSTRAINT_SCHEMA
    version: int = HARD_CONSTRAINT_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != HARD_CONSTRAINT_SCHEMA:
            raise BoundarySynthesizerError("unexpected hard-constraint schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != HARD_CONSTRAINT_VERSION:
            raise BoundarySynthesizerError("unexpected hard-constraint version")
        kind = _closed_enum(
            self.kind,
            HardConstraintKind,
            "hard constraint",
            error_type=BoundarySynthesizerError,
        )
        passed = _require_bool(self.passed, "passed")
        message = _require_text(self.message, "message", error_type=BoundarySynthesizerError)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "message", message)
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
                raise BoundarySynthesizerError("hard-constraint content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "evidence_edge_ids": list(self.evidence_edge_ids),
            "evidence_node_ids": list(self.evidence_node_ids),
            "kind": self.kind.value,
            "message": self.message,
            "passed": self.passed,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("hard-constraint content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HardConstraintCheck":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _CONSTRAINT_FIELDS)
        record = cls(
            kind=mapping["kind"],
            passed=mapping["passed"],
            message=mapping["message"],
            evidence_node_ids=mapping["evidence_node_ids"],
            evidence_edge_ids=mapping["evidence_edge_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("hard-constraint content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BoundaryRankingInput:
    """Deterministic ranking tuple. Never a safety or admission proof."""

    kind: BoundaryKind
    total_numerator: int
    costs: tuple[tuple[str, int], ...]
    schema: str = RANKING_INPUT_SCHEMA
    version: int = RANKING_INPUT_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != RANKING_INPUT_SCHEMA:
            raise BoundarySynthesizerError("unexpected ranking-input schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != RANKING_INPUT_VERSION:
            raise BoundarySynthesizerError("unexpected ranking-input version")
        kind = _closed_enum(
            self.kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        total = _require_non_negative_int(self.total_numerator, "total_numerator")
        if isinstance(self.costs, (str, bytes, bytearray)) or not isinstance(
            self.costs, Sequence
        ):
            raise BoundarySynthesizerError("costs must be a list of pairs")
        pairs: list[tuple[str, int]] = []
        for item in self.costs:
            if isinstance(item, Mapping):
                name = _require_text(
                    item.get("kind"), "cost kind", error_type=BoundarySynthesizerError
                )
                numerator = _require_non_negative_int(item.get("numerator"), "numerator")
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                if len(item) != 2:
                    raise BoundarySynthesizerError("ranking cost pair must have kind and numerator")
                name = _require_text(item[0], "cost kind", error_type=BoundarySynthesizerError)
                numerator = _require_non_negative_int(item[1], "numerator")
            else:
                raise BoundarySynthesizerError("ranking cost pair must be a list")
            pairs.append((name, numerator))
        ordered = tuple(pairs)
        expected = tuple(kind_item.value for kind_item in REQUIRED_COST_DIMENSIONS)
        if tuple(name for name, _numerator in ordered) != expected:
            raise BoundarySynthesizerError("ranking inputs must retain every cost dimension")
        if total != sum(numerator for _name, numerator in ordered):
            raise BoundarySynthesizerError(
                "ranking total_numerator must equal the documented cost-component sum"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "total_numerator", total)
        object.__setattr__(self, "costs", ordered)
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
                raise BoundarySynthesizerError("ranking-input content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "costs": [{"kind": name, "numerator": numerator} for name, numerator in self.costs],
            "kind": self.kind.value,
            "schema": self.schema,
            "total_numerator": self.total_numerator,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError("ranking-input content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryRankingInput":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _RANKING_FIELDS)
        record = cls(
            kind=mapping["kind"],
            total_numerator=mapping["total_numerator"],
            costs=mapping["costs"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError("ranking-input content identity mismatch")
        return record

    from_dict = from_mapping


def _require_constraint_closure(
    records: Sequence[HardConstraintCheck],
) -> tuple[HardConstraintCheck, ...]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise BoundarySynthesizerError("hard_constraints must be a list of objects")
    parsed = tuple(
        item if isinstance(item, HardConstraintCheck) else HardConstraintCheck.from_mapping(item)
        for item in records
    )
    by_kind = {item.kind: item for item in parsed}
    if len(by_kind) != len(parsed):
        raise BoundarySynthesizerError("hard constraints must be unique by kind")
    missing = [item.value for item in REQUIRED_HARD_CONSTRAINTS if item not in by_kind]
    if missing:
        raise BoundarySynthesizerError(f"missing hard constraints: {missing}")
    return tuple(by_kind[kind] for kind in REQUIRED_HARD_CONSTRAINTS)


@dataclass(frozen=True)
class BoundaryProposal:
    """Complete accepted candidate interface around one coherent authority."""

    kind: BoundaryKind
    interface: BoundaryInterface
    canonical_owner_node_id: str
    state_owner_node_id: str
    callers: tuple[str, ...]
    effects: tuple[str, ...]
    adapters: tuple[str, ...]
    deprecated_paths: tuple[str, ...]
    tests: tuple[str, ...]
    proofs: tuple[str, ...]
    migration: BoundaryMigration
    rollback: BoundaryRollback
    prediction: BoundaryPrediction
    costs: BoundaryCostVector
    hard_constraints: tuple[HardConstraintCheck, ...]
    repository_tree: str
    freshness: str
    disposition: ProposalDisposition = ProposalDisposition.ACCEPTED
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
        kind = _closed_enum(
            self.kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        disposition = _closed_enum(
            self.disposition,
            ProposalDisposition,
            "proposal disposition",
            error_type=BoundarySynthesizerError,
        )
        if disposition is not ProposalDisposition.ACCEPTED:
            raise BoundarySynthesizerError("accepted proposals cannot carry a rejected disposition")
        interface = (
            self.interface
            if isinstance(self.interface, BoundaryInterface)
            else BoundaryInterface.from_mapping(self.interface)
        )
        if interface.kind is not kind:
            raise BoundarySynthesizerError("proposal kind must match interface kind")
        owner = _require_text(
            self.canonical_owner_node_id,
            "canonical_owner_node_id",
            error_type=BoundarySynthesizerError,
        )
        if owner != interface.canonical_owner_node_id:
            raise BoundarySynthesizerError("proposal owner must match interface owner")
        if _looks_like_content_identity(owner):
            raise BoundarySynthesizerError(
                "content identity is not inferred to be authority"
            )
        state_owner = _require_text(
            self.state_owner_node_id,
            "state_owner_node_id",
            error_type=BoundarySynthesizerError,
        )
        migration = (
            self.migration
            if isinstance(self.migration, BoundaryMigration)
            else BoundaryMigration.from_mapping(self.migration)
        )
        if migration.state_owner_node_id != state_owner:
            raise BoundarySynthesizerError("migration state owner must match proposal state owner")
        rollback = (
            self.rollback
            if isinstance(self.rollback, BoundaryRollback)
            else BoundaryRollback.from_mapping(self.rollback)
        )
        prediction = (
            self.prediction
            if isinstance(self.prediction, BoundaryPrediction)
            else BoundaryPrediction.from_mapping(self.prediction)
        )
        costs = (
            self.costs
            if isinstance(self.costs, BoundaryCostVector)
            else BoundaryCostVector.from_mapping(self.costs)
        )
        constraints = _require_constraint_closure(self.hard_constraints)
        failed = tuple(item for item in constraints if item.passed is not True)
        if failed:
            raise BoundarySynthesizerError(
                "accepted proposals cannot retain failed hard constraints"
            )
        if prediction.validation_coverage_loss != 0:
            raise BoundarySynthesizerError("accepted proposals cannot lose validation coverage")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "canonical_owner_node_id", owner)
        object.__setattr__(self, "state_owner_node_id", state_owner)
        object.__setattr__(self, "callers", _require_text_tuple(self.callers, "callers"))
        object.__setattr__(self, "effects", _require_text_tuple(self.effects, "effects"))
        object.__setattr__(self, "adapters", _require_text_tuple(self.adapters, "adapters"))
        object.__setattr__(
            self,
            "deprecated_paths",
            _require_text_tuple(self.deprecated_paths, "deprecated_paths"),
        )
        object.__setattr__(self, "tests", _require_text_tuple(self.tests, "tests"))
        object.__setattr__(self, "proofs", _require_text_tuple(self.proofs, "proofs"))
        object.__setattr__(self, "migration", migration)
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "costs", costs)
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
        object.__setattr__(self, "disposition", ProposalDisposition.ACCEPTED)
        if set(self.callers) != set(interface.allowed_callers):
            raise BoundarySynthesizerError("proposal callers must match interface allowed_callers")
        if set(self.effects) != set(interface.allowed_effects):
            raise BoundarySynthesizerError("proposal effects must match interface allowed_effects")
        if set(self.adapters) != set(migration.adapters):
            raise BoundarySynthesizerError("proposal adapters must match migration adapters")
        if set(self.deprecated_paths) != set(migration.deprecated_paths):
            raise BoundarySynthesizerError(
                "proposal deprecated_paths must match migration deprecated_paths"
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
                raise BoundarySynthesizerError("boundary-proposal content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def constraint(self, kind: HardConstraintKind | str) -> HardConstraintCheck:
        parsed = _closed_enum(
            kind,
            HardConstraintKind,
            "hard constraint",
            error_type=BoundarySynthesizerError,
        )
        for item in self.hard_constraints:
            if item.kind is parsed:
                return item
        raise BoundarySynthesizerError(f"missing hard constraint: {parsed.value}")

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_autonomous_application("apply")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    def mutate_state(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_state_mutation("mutate")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "adapters": list(self.adapters),
            "callers": list(self.callers),
            "canonical_owner_node_id": self.canonical_owner_node_id,
            "costs": self.costs.to_dict(),
            "deprecated_paths": list(self.deprecated_paths),
            "disposition": ProposalDisposition.ACCEPTED.value,
            "effects": list(self.effects),
            "freshness": self.freshness,
            "hard_constraints": [item.to_dict() for item in self.hard_constraints],
            "interface": self.interface.to_dict(),
            "kind": self.kind.value,
            "migration": self.migration.to_dict(),
            "prediction": self.prediction.to_dict(),
            "proofs": list(self.proofs),
            "repository_tree": self.repository_tree,
            "rollback": self.rollback.to_dict(),
            "schema": self.schema,
            "state_owner_node_id": self.state_owner_node_id,
            "tests": list(self.tests),
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
            kind=mapping["kind"],
            interface=mapping["interface"],
            canonical_owner_node_id=mapping["canonical_owner_node_id"],
            state_owner_node_id=mapping["state_owner_node_id"],
            callers=mapping["callers"],
            effects=mapping["effects"],
            adapters=mapping["adapters"],
            deprecated_paths=mapping["deprecated_paths"],
            tests=mapping["tests"],
            proofs=mapping["proofs"],
            migration=mapping["migration"],
            rollback=mapping["rollback"],
            prediction=mapping["prediction"],
            costs=mapping["costs"],
            hard_constraints=mapping["hard_constraints"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            disposition=mapping["disposition"],
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
class RejectedBoundaryProposal:
    """Hard-rejected boundary with retained cost measures and explanations."""

    kind: BoundaryKind
    rejection_kind: RejectionKind
    message: str
    hard_constraints: tuple[HardConstraintCheck, ...]
    costs: BoundaryCostVector
    repository_tree: str
    freshness: str
    canonical_owner_node_id: str = ""
    node_ids: tuple[str, ...] = ()
    disposition: ProposalDisposition = ProposalDisposition.REJECTED
    schema: str = BOUNDARY_REJECTION_SCHEMA
    version: int = BOUNDARY_REJECTION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_REJECTION_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-rejection schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_REJECTION_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-rejection version")
        kind = _closed_enum(
            self.kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        rejection_kind = _closed_enum(
            self.rejection_kind,
            RejectionKind,
            "rejection kind",
            error_type=BoundarySynthesizerError,
        )
        disposition = _closed_enum(
            self.disposition,
            ProposalDisposition,
            "proposal disposition",
            error_type=BoundarySynthesizerError,
        )
        if disposition is not ProposalDisposition.REJECTED:
            raise BoundarySynthesizerError("rejected proposals cannot carry an accepted disposition")
        message = _require_text(self.message, "message", error_type=BoundarySynthesizerError)
        constraints = _require_constraint_closure(self.hard_constraints)
        failed = tuple(item for item in constraints if item.passed is not True)
        if not failed:
            raise BoundarySynthesizerError(
                "rejected proposals must retain at least one failed hard constraint"
            )
        costs = (
            self.costs
            if isinstance(self.costs, BoundaryCostVector)
            else BoundaryCostVector.from_mapping(self.costs)
        )
        owner = self.canonical_owner_node_id
        if owner:
            owner = _require_text(
                owner, "canonical_owner_node_id", error_type=BoundarySynthesizerError
            )
            if _looks_like_content_identity(owner):
                raise BoundarySynthesizerError(
                    "content identity is not inferred to be authority"
                )
        else:
            owner = ""
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rejection_kind", rejection_kind)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "hard_constraints", constraints)
        object.__setattr__(self, "costs", costs)
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
        object.__setattr__(self, "canonical_owner_node_id", owner)
        object.__setattr__(self, "node_ids", _require_text_tuple(self.node_ids, "node_ids"))
        object.__setattr__(self, "disposition", ProposalDisposition.REJECTED)
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
                    "boundary-rejection content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    @property
    def failed_constraints(self) -> tuple[HardConstraintCheck, ...]:
        return tuple(item for item in self.hard_constraints if item.passed is not True)

    def promote_by_cost(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_hard_constraint_override("override")

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_autonomous_application("apply")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "canonical_owner_node_id": self.canonical_owner_node_id,
            "costs": self.costs.to_dict(),
            "disposition": ProposalDisposition.REJECTED.value,
            "failed_constraints": [item.to_dict() for item in self.failed_constraints],
            "freshness": self.freshness,
            "hard_constraints": [item.to_dict() for item in self.hard_constraints],
            "kind": self.kind.value,
            "message": self.message,
            "node_ids": list(self.node_ids),
            "rejection_kind": self.rejection_kind.value,
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError(
                "boundary-rejection content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RejectedBoundaryProposal":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _REJECTION_FIELDS)
        record = cls(
            kind=mapping["kind"],
            rejection_kind=mapping["rejection_kind"],
            message=mapping["message"],
            hard_constraints=mapping["hard_constraints"],
            costs=mapping["costs"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            canonical_owner_node_id=mapping["canonical_owner_node_id"],
            node_ids=mapping["node_ids"],
            disposition=mapping["disposition"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        expected_failed = [item.to_dict() for item in record.failed_constraints]
        if mapping["failed_constraints"] != expected_failed:
            raise BoundarySynthesizerError("failed_constraints projection mismatch")
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError(
                "boundary-rejection content identity mismatch"
            )
        return record

    from_dict = from_mapping


def _ranking_for(proposals: Sequence[BoundaryProposal]) -> tuple[str, ...]:
    ordered = sorted(
        proposals,
        key=lambda item: (*item.costs.ranking_key(), item.kind.value),
    )
    return tuple(item.kind.value for item in ordered)


def _ranking_inputs_for(
    proposals: Sequence[BoundaryProposal],
) -> tuple[BoundaryRankingInput, ...]:
    by_kind = {item.kind.value: item for item in proposals}
    return tuple(
        BoundaryRankingInput(
            kind=by_kind[name].kind,
            total_numerator=by_kind[name].costs.total_numerator,
            costs=tuple(
                (measure.kind.value, measure.numerator)
                for measure in by_kind[name].costs.measures
            ),
        )
        for name in _ranking_for(proposals)
    )


@dataclass(frozen=True)
class BoundarySynthesisResult:
    """Closed synthesis report of accepted proposals and hard rejections."""

    architecture_ir_identity: str
    ownership_identity: str
    repository_tree: str
    freshness: str
    proposals: tuple[BoundaryProposal, ...] = ()
    rejections: tuple[RejectedBoundaryProposal, ...] = ()
    ranking: tuple[str, ...] = ()
    ranking_inputs: tuple[BoundaryRankingInput, ...] = ()
    schema: str = BOUNDARY_SYNTHESIS_SCHEMA
    version: int = BOUNDARY_SYNTHESIS_VERSION
    effect_class: str = EFFECT_CLASS
    candidate_interfaces_only: bool = True
    ranking_is_non_probative: bool = True
    covers_initial_boundaries: bool = True
    can_authorize_changes: bool = SYNTHESIZER_CAN_AUTHORIZE_CHANGES
    can_transfer_authority: bool = SYNTHESIZER_CAN_TRANSFER_AUTHORITY
    can_promote_authority: bool = SYNTHESIZER_CAN_PROMOTE_AUTHORITY
    can_mutate_state: bool = SYNTHESIZER_CAN_MUTATE_STATE
    can_apply_proposals: bool = SYNTHESIZER_CAN_APPLY_PROPOSALS
    can_override_hard_constraints: bool = SYNTHESIZER_CAN_OVERRIDE_HARD_CONSTRAINTS
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=BoundarySynthesizerError)
        if schema != BOUNDARY_SYNTHESIS_SCHEMA:
            raise BoundarySynthesizerError("unexpected boundary-synthesis schema")
        version = _require_int(self.version, "version", error_type=BoundarySynthesizerError)
        if version != BOUNDARY_SYNTHESIS_VERSION:
            raise BoundarySynthesizerError("unexpected boundary-synthesis version")
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=BoundarySynthesizerError
        )
        if effect_class != EFFECT_CLASS:
            raise BoundarySynthesizerError("unexpected boundary-synthesis effect class")
        for name, expected in (
            ("candidate_interfaces_only", True),
            ("ranking_is_non_probative", True),
            ("can_authorize_changes", False),
            ("can_transfer_authority", False),
            ("can_promote_authority", False),
            ("can_mutate_state", False),
            ("can_apply_proposals", False),
            ("can_override_hard_constraints", False),
        ):
            if _require_bool(getattr(self, name), name) is not expected:
                raise BoundarySynthesizerAuthorityError(
                    f"boundary synthesis {name} must be {expected}"
                )
        proposals = _record_tuple(self.proposals, "proposals", BoundaryProposal)
        rejections = _record_tuple(self.rejections, "rejections", RejectedBoundaryProposal)
        by_kind: dict[BoundaryKind, str] = {}
        for item in proposals:
            if item.kind in by_kind:
                raise BoundarySynthesizerError("accepted proposals must be unique by kind")
            by_kind[item.kind] = "accepted"
            if item.repository_tree != self.repository_tree:
                raise BoundarySynthesizerError("proposal repository_tree must match result")
            if item.freshness != self.freshness:
                raise BoundarySynthesizerError("proposal freshness must match result")
        for item in rejections:
            if item.kind in by_kind:
                raise BoundarySynthesizerError(
                    "a boundary cannot be both accepted and rejected"
                )
            by_kind[item.kind] = "rejected"
            if item.repository_tree != self.repository_tree:
                raise BoundarySynthesizerError("rejection repository_tree must match result")
            if item.freshness != self.freshness:
                raise BoundarySynthesizerError("rejection freshness must match result")
        missing = [item.value for item in INITIAL_BOUNDARIES if item not in by_kind]
        extra = sorted(item.value for item in by_kind if item not in set(INITIAL_BOUNDARIES))
        covers = not missing and not extra
        claimed_covers = _require_bool(
            self.covers_initial_boundaries, "covers_initial_boundaries"
        )
        if claimed_covers is not covers:
            raise BoundarySynthesizerError(
                "covers_initial_boundaries must match the closed initial boundary set"
            )
        if missing:
            raise BoundarySynthesizerError(f"missing initial boundaries: {missing}")
        if extra:
            raise BoundarySynthesizerError(f"unsupported boundaries: {extra}")
        ordered_proposals = tuple(
            next(item for item in proposals if item.kind is kind)
            for kind in INITIAL_BOUNDARIES
            if kind in {item.kind for item in proposals}
        )
        ordered_rejections = tuple(
            next(item for item in rejections if item.kind is kind)
            for kind in INITIAL_BOUNDARIES
            if kind in {item.kind for item in rejections}
        )
        expected_ranking = _ranking_for(ordered_proposals)
        ranking = _require_ordered_text_tuple(self.ranking, "ranking")
        if ranking != expected_ranking:
            raise BoundarySynthesizerError("ranking must follow deterministic cost inputs")
        ranking_inputs = tuple(
            item
            if isinstance(item, BoundaryRankingInput)
            else BoundaryRankingInput.from_mapping(item)
            for item in (
                self.ranking_inputs
                if not isinstance(self.ranking_inputs, (str, bytes, bytearray))
                and isinstance(self.ranking_inputs, Sequence)
                else ()
            )
        )
        if not ranking_inputs:
            ranking_inputs = _ranking_inputs_for(ordered_proposals)
        expected_inputs = _ranking_inputs_for(ordered_proposals)
        if tuple(item.content_identity for item in ranking_inputs) != tuple(
            item.content_identity for item in expected_inputs
        ):
            raise BoundarySynthesizerError("ranking_inputs must match accepted cost vectors")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "effect_class", EFFECT_CLASS)
        object.__setattr__(
            self,
            "architecture_ir_identity",
            _validate_dag_json_cid(
                _require_text(
                    self.architecture_ir_identity,
                    "architecture_ir_identity",
                    error_type=BoundarySynthesizerError,
                )
            ),
        )
        object.__setattr__(
            self,
            "ownership_identity",
            _validate_dag_json_cid(
                _require_text(
                    self.ownership_identity,
                    "ownership_identity",
                    error_type=BoundarySynthesizerError,
                )
            ),
        )
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
        object.__setattr__(self, "proposals", ordered_proposals)
        object.__setattr__(self, "rejections", ordered_rejections)
        object.__setattr__(self, "ranking", expected_ranking)
        object.__setattr__(self, "ranking_inputs", expected_inputs)
        object.__setattr__(self, "candidate_interfaces_only", True)
        object.__setattr__(self, "ranking_is_non_probative", True)
        object.__setattr__(self, "covers_initial_boundaries", True)
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "can_transfer_authority", False)
        object.__setattr__(self, "can_promote_authority", False)
        object.__setattr__(self, "can_mutate_state", False)
        object.__setattr__(self, "can_apply_proposals", False)
        object.__setattr__(self, "can_override_hard_constraints", False)
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

    def proposal(self, kind: BoundaryKind | str) -> BoundaryProposal:
        parsed = _closed_enum(
            kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        for item in self.proposals:
            if item.kind is parsed:
                return item
        raise BoundarySynthesizerError(f"no accepted proposal for {parsed.value}")

    def rejection(self, kind: BoundaryKind | str) -> RejectedBoundaryProposal:
        parsed = _closed_enum(
            kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        for item in self.rejections:
            if item.kind is parsed:
                return item
        raise BoundarySynthesizerError(f"no rejected proposal for {parsed.value}")

    def record_for(self, kind: BoundaryKind | str) -> BoundaryProposal | RejectedBoundaryProposal:
        parsed = _closed_enum(
            kind, BoundaryKind, "boundary kind", error_type=BoundarySynthesizerError
        )
        for item in self.proposals:
            if item.kind is parsed:
                return item
        for item in self.rejections:
            if item.kind is parsed:
                return item
        raise BoundarySynthesizerError(f"missing initial boundary: {parsed.value}")

    @property
    def hard_constraints_preserved(self) -> bool:
        if any(item.disposition is not ProposalDisposition.ACCEPTED for item in self.proposals):
            return False
        if any(not all(check.passed for check in item.hard_constraints) for item in self.proposals):
            return False
        if any(kind in {item.kind.value for item in self.rejections} for kind in self.ranking):
            return False
        return True

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_autonomous_application("apply")

    def override_hard_constraint(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_hard_constraint_override("override")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "can_apply_proposals": False,
            "can_authorize_changes": False,
            "can_mutate_state": False,
            "can_override_hard_constraints": False,
            "can_promote_authority": False,
            "can_transfer_authority": False,
            "candidate_interfaces_only": True,
            "covers_initial_boundaries": True,
            "effect_class": EFFECT_CLASS,
            "freshness": self.freshness,
            "ownership_identity": self.ownership_identity,
            "proposals": [item.to_dict() for item in self.proposals],
            "ranking": list(self.ranking),
            "ranking_inputs": [item.to_dict() for item in self.ranking_inputs],
            "ranking_is_non_probative": True,
            "rejections": [item.to_dict() for item in self.rejections],
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise BoundarySynthesizerError(
                "boundary-synthesis content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundarySynthesisResult":
        mapping = _require_mapping(payload, error_type=BoundarySynthesizerError)
        _require_fields(mapping, _RESULT_FIELDS)
        record = cls(
            architecture_ir_identity=mapping["architecture_ir_identity"],
            ownership_identity=mapping["ownership_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            proposals=mapping["proposals"],
            rejections=mapping["rejections"],
            ranking=mapping["ranking"],
            ranking_inputs=mapping["ranking_inputs"],
            schema=mapping["schema"],
            version=mapping["version"],
            effect_class=mapping["effect_class"],
            candidate_interfaces_only=mapping["candidate_interfaces_only"],
            ranking_is_non_probative=mapping["ranking_is_non_probative"],
            covers_initial_boundaries=mapping["covers_initial_boundaries"],
            can_authorize_changes=mapping["can_authorize_changes"],
            can_transfer_authority=mapping["can_transfer_authority"],
            can_promote_authority=mapping["can_promote_authority"],
            can_mutate_state=mapping["can_mutate_state"],
            can_apply_proposals=mapping["can_apply_proposals"],
            can_override_hard_constraints=mapping["can_override_hard_constraints"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise BoundarySynthesizerError(
                "boundary-synthesis content identity mismatch"
            )
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
            raise BoundarySynthesizerError(
                "boundary-synthesis JSON is malformed"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise BoundarySynthesizerError(
                "boundary-synthesis JSON must contain an object"
            )
        return cls.from_mapping(decoded)


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


def _walk(
    view: _GraphView,
    roots: Iterable[str],
    *,
    reverse: bool = False,
    kinds: frozenset[EdgeKind] | None = None,
) -> frozenset[str]:
    allowed = kinds if kinds is not None else _CONE_EDGE_KINDS
    seen: set[str] = set()
    queue: deque[str] = deque(root for root in roots if root in view.nodes_by_id)
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        edges = view.incoming.get(node_id, ()) if reverse else view.outgoing.get(node_id, ())
        for edge in edges:
            if edge.kind not in allowed:
                continue
            nxt = edge.source if reverse else edge.target
            if nxt not in seen:
                queue.append(nxt)
    return frozenset(seen)


def detect_cross_boundary_effects(
    architecture: ArchitectureIR | Mapping[str, Any],
    cluster: Iterable[str],
) -> tuple[ArchitectureEdge, ...]:
    """Return effect edges that cross the proposed cluster boundary."""

    view = _build_view(_require_architecture_ir(architecture))
    inside = set(cluster) & set(view.nodes_by_id)
    found: list[ArchitectureEdge] = []
    for edge in view.architecture.edges:
        if edge.kind not in _EFFECT_EDGE_KINDS:
            continue
        source_in = edge.source in inside
        target_in = edge.target in inside
        if source_in != target_in:
            found.append(edge)
    return tuple(sorted(found, key=lambda item: item.edge_id))


def detect_mutable_sharing(
    architecture: ArchitectureIR | Mapping[str, Any],
    cluster: Iterable[str],
) -> tuple[str, ...]:
    """Return state nodes mutated from both inside and outside the cluster."""

    view = _build_view(_require_architecture_ir(architecture))
    inside = set(cluster) & set(view.nodes_by_id)
    writers: dict[str, set[bool]] = defaultdict(set)
    for edge in view.architecture.edges:
        if edge.kind not in _MUTABLE_EDGE_KINDS:
            continue
        target = view.nodes_by_id.get(edge.target)
        if target is None or target.kind is not NodeKind.STATE:
            continue
        writers[edge.target].add(edge.source in inside)
    shared = tuple(
        sorted(node_id for node_id, sides in writers.items() if True in sides and False in sides)
    )
    return shared


def detect_cross_boundary_cycles(
    architecture: ArchitectureIR | Mapping[str, Any],
    cluster: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    """Return simple cycles that include at least one node on each side of the boundary."""

    view = _build_view(_require_architecture_ir(architecture))
    inside = set(cluster) & set(view.nodes_by_id)
    graph: dict[str, list[str]] = {node.node_id: [] for node in view.architecture.nodes}
    for edge in view.architecture.edges:
        if edge.kind in _CYCLE_EDGE_KINDS:
            graph[edge.source].append(edge.target)
    cycles: list[tuple[str, ...]] = []
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}

    def _strongconnect(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlink[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for nxt in graph[node_id]:
            if nxt not in indices:
                _strongconnect(nxt)
                lowlink[node_id] = min(lowlink[node_id], lowlink[nxt])
            elif nxt in on_stack:
                lowlink[node_id] = min(lowlink[node_id], indices[nxt])
        if lowlink[node_id] == indices[node_id]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node_id:
                    break
            members = tuple(sorted(component))
            if len(members) > 1 or (members and members[0] in graph[members[0]]):
                sides = {member in inside for member in members}
                if True in sides and False in sides:
                    cycles.append(members)

    for node_id in graph:
        if node_id not in indices:
            _strongconnect(node_id)
    return tuple(sorted(cycles))


def _cost_measure(
    kind: CostDimensionKind,
    numerator: int,
    *,
    denominator: int | None,
    node_ids: Iterable[str] = (),
    edge_ids: Iterable[str] = (),
) -> BoundaryCostMeasure:
    return BoundaryCostMeasure(
        kind=kind,
        numerator=numerator,
        unit=_COST_UNITS[kind],
        denominator=denominator,
        evidence_node_ids=tuple(node_ids),
        evidence_edge_ids=tuple(edge_ids),
    )


def measure_boundary_costs(
    architecture: ArchitectureIR | Mapping[str, Any],
    cluster: Iterable[str],
) -> BoundaryCostVector:
    """Measure independently auditable boundary costs for one cluster."""

    view = _build_view(_require_architecture_ir(architecture))
    inside = tuple(sorted(set(cluster) & set(view.nodes_by_id)))
    denominator = max(len(view.architecture.nodes), 1)
    effect_edges = detect_cross_boundary_effects(view.architecture, inside)
    shared_state = detect_mutable_sharing(view.architecture, inside)
    cycles = detect_cross_boundary_cycles(view.architecture, inside)
    public_nodes = tuple(
        node_id
        for node_id in inside
        if view.nodes_by_id[node_id].kind in _PUBLIC_NODE_KINDS
    )
    files = tuple(
        node_id for node_id in inside if view.nodes_by_id[node_id].kind is NodeKind.FILE
    )
    symbols = tuple(
        node_id for node_id in inside if view.nodes_by_id[node_id].kind is NodeKind.SYMBOL
    )
    interfaces = tuple(
        node_id
        for node_id in inside
        if view.nodes_by_id[node_id].kind is NodeKind.INTERFACE
    )
    effects = tuple(
        node_id for node_id in inside if view.nodes_by_id[node_id].kind is NodeKind.EFFECT
    )
    tests = tuple(
        node_id for node_id in inside if view.nodes_by_id[node_id].kind is NodeKind.TEST
    )
    proofs = tuple(
        node_id for node_id in inside if view.nodes_by_id[node_id].kind is NodeKind.PROOF
    )
    owners = tuple(
        node_id
        for node_id in inside
        if view.nodes_by_id[node_id].kind is NodeKind.AUTHORITY
    )
    amplification = (
        len(files)
        + len(symbols)
        + len(interfaces)
        + len(effects)
        + len(tests)
        + len(proofs)
        + len(owners)
    )
    cone = _walk(view, inside)
    context_nodes = tuple(
        node_id
        for node_id in sorted(cone)
        if view.nodes_by_id[node_id].kind in _CONTEXT_NODE_KINDS
    )
    reverse = _walk(view, inside, reverse=True)
    validation_nodes = tuple(
        node_id
        for node_id in sorted(reverse)
        if view.nodes_by_id[node_id].kind in {NodeKind.TEST, NodeKind.PROOF}
    )
    measures = (
        _cost_measure(
            CostDimensionKind.CROSS_BOUNDARY_EFFECTS,
            len(effect_edges),
            denominator=denominator,
            edge_ids=tuple(edge.edge_id for edge in effect_edges),
        ),
        _cost_measure(
            CostDimensionKind.MUTABLE_SHARING,
            len(shared_state),
            denominator=denominator,
            node_ids=shared_state,
        ),
        _cost_measure(
            CostDimensionKind.CYCLES,
            len(cycles),
            denominator=denominator,
            node_ids=tuple(node_id for cycle in cycles for node_id in cycle),
        ),
        _cost_measure(
            CostDimensionKind.PUBLIC_SYMBOLS,
            len(public_nodes),
            denominator=denominator,
            node_ids=public_nodes,
        ),
        _cost_measure(
            CostDimensionKind.CHANGE_AMPLIFICATION,
            amplification,
            denominator=denominator,
            node_ids=inside,
        ),
        _cost_measure(
            CostDimensionKind.CONTEXT_BURDEN,
            len(context_nodes),
            denominator=denominator,
            node_ids=context_nodes,
        ),
        _cost_measure(
            CostDimensionKind.VALIDATION_AMPLIFICATION,
            len(validation_nodes),
            denominator=denominator,
            node_ids=validation_nodes,
        ),
        _cost_measure(
            CostDimensionKind.DEPENDENCY_CONE,
            len(cone),
            denominator=denominator,
            node_ids=tuple(sorted(cone)),
        ),
    )
    return BoundaryCostVector(measures=measures, total_numerator=sum(item.numerator for item in measures))


def rank_boundary_proposals(
    proposals: Sequence[BoundaryProposal],
    rejections: Sequence[RejectedBoundaryProposal] = (),
) -> tuple[str, ...]:
    """Rank accepted proposals by cost. Rejections cannot enter the ranking."""

    if not RANKING_IS_NON_PROBATIVE or not HARD_CONSTRAINTS_PRECEDE_RANKING:
        refuse_hard_constraint_override("override")
    for item in rejections:
        if item.disposition is not ProposalDisposition.REJECTED:
            raise BoundarySynthesizerError("ranking rejections must remain rejected")
    parsed = tuple(
        item if isinstance(item, BoundaryProposal) else BoundaryProposal.from_mapping(item)
        for item in proposals
    )
    return _ranking_for(parsed)


def admit_rejected_by_cost(*_args: Any, **_kwargs: Any) -> None:
    """A lower cost never admits a hard-rejected boundary."""

    refuse_hard_constraint_override("override")


def ranking_establishes(_result: BoundarySynthesisResult, claim: str) -> bool:
    """Ranking never establishes safety, ownership, equivalence, or promotion."""

    name = _require_text(claim, "claim", error_type=BoundarySynthesizerError)
    if name not in {
        "safety",
        "equivalence",
        "ownership",
        "promotion",
        "deletion",
        "authority",
        "rollback",
        "completeness",
    }:
        raise BoundarySynthesizerError(f"unsupported ranking authority claim: {name!r}")
    return False


def _nonempty_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in values if item)


def _check(
    kind: HardConstraintKind,
    passed: bool,
    message: str,
    *,
    node_ids: Iterable[str] = (),
    edge_ids: Iterable[str] = (),
) -> HardConstraintCheck:
    return HardConstraintCheck(
        kind=kind,
        passed=passed,
        message=message,
        evidence_node_ids=_nonempty_ids(node_ids),
        evidence_edge_ids=_nonempty_ids(edge_ids),
    )


def _owner_confidence(view: _GraphView, owner_id: str) -> Confidence | None:
    node = view.nodes_by_id.get(owner_id)
    if node is None:
        return None
    return node.provenance.confidence


def _scope_escape_nodes(view: _GraphView, cluster: Iterable[str]) -> tuple[str, ...]:
    escaped: list[str] = []
    for node_id in cluster:
        node = view.nodes_by_id.get(node_id)
        if node is None:
            continue
        path = node.provenance.span.path
        if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in _SIBLING_PREFIXES):
            escaped.append(node_id)
    return tuple(sorted(escaped))


def _ambiguity_affects(
    ambiguity: ContractAmbiguity,
    cluster: Iterable[str],
    view: _GraphView,
) -> bool:
    inside = set(cluster)
    if ambiguity.subject in inside:
        return True
    return any(
        ambiguity.subject == node_id or ambiguity.subject == view.nodes_by_id[node_id].node_id
        for node_id in inside
        if node_id in view.nodes_by_id
    )


@dataclass(frozen=True)
class _Draft:
    kind: BoundaryKind
    owner_id: str
    state_owner_id: str
    cluster: tuple[str, ...]
    callers: tuple[str, ...]
    effects: tuple[str, ...]
    adapters: tuple[str, ...]
    deprecated_paths: tuple[str, ...]
    tests: tuple[str, ...]
    proofs: tuple[str, ...]
    public_symbols: tuple[str, ...]
    concerns: tuple[str, ...]
    blockers: tuple[OwnershipBlocker, ...]
    ambiguities: tuple[ContractAmbiguity, ...]
    effect_edges: tuple[ArchitectureEdge, ...]
    shared_state: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]
    prediction: BoundaryPrediction
    costs: BoundaryCostVector
    existing_effects: frozenset[str]
    existing_proofs: frozenset[str]
    existing_tests: frozenset[str]


def evaluate_hard_constraints(draft: _Draft, view: _GraphView) -> tuple[HardConstraintCheck, ...]:
    """Evaluate the closed hard-gate set for one draft boundary."""

    owner_node = view.nodes_by_id.get(draft.owner_id)
    confidence = _owner_confidence(view, draft.owner_id)
    escaped = _scope_escape_nodes(view, draft.cluster)
    simulated_owner = owner_node is not None and owner_node.kind is NodeKind.SIMULATION
    unbounded = bool(view.architecture.nodes) and set(draft.cluster) >= {
        node.node_id for node in view.architecture.nodes
    } and len(view.architecture.nodes) > 4
    listed_effects = frozenset(draft.effects)
    expanded_effects = listed_effects - draft.existing_effects
    omitted_proofs = draft.existing_proofs - frozenset(draft.proofs)
    omitted_tests = draft.existing_tests - frozenset(draft.tests)
    incomplete = not draft.owner_id or not draft.state_owner_id
    stale = confidence in NON_PROBATIVE_CONFIDENCE if confidence is not None else True
    checks = (
        _check(
            HardConstraintKind.NO_AUTHORITY_WEAKENING,
            owner_node is not None and owner_node.kind is NodeKind.AUTHORITY,
            "canonical owner remains an existing ArchitectureIR authority node"
            if owner_node is not None and owner_node.kind is NodeKind.AUTHORITY
            else "proposal would weaken or replace a canonical authority",
            node_ids=(draft.owner_id,) if draft.owner_id else (),
        ),
        _check(
            HardConstraintKind.NO_EFFECT_EXPANSION,
            not expanded_effects,
            "allowed effects are a subset of existing cluster effects"
            if not expanded_effects
            else "proposal would expand effects beyond the current cluster",
            node_ids=tuple(sorted(expanded_effects)),
        ),
        _check(
            HardConstraintKind.NO_HIDDEN_BEHAVIOR_CHANGE,
            True,
            "synthesis names callers, effects, and adapters without rewriting behavior",
        ),
        _check(
            HardConstraintKind.NO_SIMULATED_AS_LIVE,
            not simulated_owner,
            "canonical owner is not a simulation node"
            if not simulated_owner
            else "simulation node cannot be promoted to a live authority",
            node_ids=(draft.owner_id,) if simulated_owner else (),
        ),
        _check(
            HardConstraintKind.NO_VALIDATION_REDUCTION,
            not omitted_tests and draft.prediction.validation_coverage_loss == 0,
            "existing tests remain declared"
            if not omitted_tests
            else "proposal would drop required tests",
            node_ids=tuple(sorted(omitted_tests)),
        ),
        _check(
            HardConstraintKind.NO_PROOF_OBLIGATION_LOSS,
            not omitted_proofs,
            "existing proofs remain declared"
            if not omitted_proofs
            else "proposal would drop required proofs",
            node_ids=tuple(sorted(omitted_proofs)),
        ),
        _check(
            HardConstraintKind.NO_PUBLIC_CONTRACT_BREAK,
            True,
            "candidate interface lists adapters and deprecations without removing contracts",
            node_ids=draft.adapters,
        ),
        _check(
            HardConstraintKind.NO_STALE_EVIDENCE_PROMOTION,
            not stale,
            "canonical owner evidence is exact or conservative"
            if not stale
            else "heuristic or opaque evidence cannot prove a stable boundary",
            node_ids=(draft.owner_id,) if draft.owner_id else (),
        ),
        _check(
            HardConstraintKind.NO_UNBOUNDED_REFACTOR,
            not unbounded,
            "cluster is a bounded authority neighborhood"
            if not unbounded
            else "cluster covers the entire graph and is not a bounded interface",
            node_ids=draft.cluster if unbounded else (),
        ),
        _check(
            HardConstraintKind.NO_PROCEDURE_SELF_AUTHORIZATION,
            SYNTHESIZER_CAN_APPLY_PROPOSALS is False,
            "synthesis cannot authorize its own application",
        ),
        _check(
            HardConstraintKind.NO_ARCHITECTURE_CANDIDATE_SELF_PROMOTION,
            CANDIDATE_INTERFACES_ONLY is True,
            "synthesized interfaces remain candidate-tier",
        ),
        _check(
            HardConstraintKind.NO_CROSS_REPOSITORY_WRITE,
            not escaped,
            "cluster stays inside the owning repository"
            if not escaped
            else "cluster includes a sibling-repository path",
            node_ids=escaped,
        ),
        _check(
            HardConstraintKind.NO_SECRET_OR_PRIVATE_DATA_LEAK,
            True,
            "proposals carry node identities only and do not embed private payloads",
        ),
        _check(
            HardConstraintKind.NO_FALSE_COMPLETION,
            True,
            "failed hard gates remain rejections rather than accepted completions",
        ),
        _check(
            HardConstraintKind.UNRESOLVED_AUTHORITY,
            not draft.blockers,
            "related concerns have canonical owners"
            if not draft.blockers
            else "unresolved authority hard-rejects the affected proposal",
            node_ids=tuple(
                node_id
                for blocker in draft.blockers
                for node_id in blocker.node_ids
                if node_id
            ),
        ),
        _check(
            HardConstraintKind.UNRESOLVED_AMBIGUITY,
            not draft.ambiguities,
            "no unresolved contract ambiguity affects the cluster"
            if not draft.ambiguities
            else "unresolved contract ambiguity hard-rejects the affected proposal",
            node_ids=tuple(item.subject for item in draft.ambiguities),
        ),
        _check(
            HardConstraintKind.DUAL_STATE_AUTHORITY,
            not draft.shared_state,
            "mutable state is not shared across the proposed boundary"
            if not draft.shared_state
            else "dual mutable-state authority crosses the proposed boundary",
            node_ids=draft.shared_state,
        ),
        _check(
            HardConstraintKind.MISSING_ROLLBACK,
            True,
            "rollback reverts the unapplied candidate interface",
        ),
        _check(
            HardConstraintKind.INCOMPLETE_PROPOSAL,
            not incomplete,
            "required interface, owner, and state owner are named"
            if not incomplete
            else "proposal is missing a required owner or state owner",
            node_ids=_nonempty_ids((draft.owner_id, draft.state_owner_id)),
        ),
        _check(
            HardConstraintKind.AUTHORITY_TRANSFER,
            owner_node is not None and owner_node.kind is NodeKind.AUTHORITY,
            "proposal preserves an existing canonical owner"
            if owner_node is not None and owner_node.kind is NodeKind.AUTHORITY
            else "proposal would transfer or create authority",
            node_ids=(draft.owner_id,) if draft.owner_id else (),
        ),
        _check(
            HardConstraintKind.STATE_MOVEMENT,
            True,
            "proposal names the state owner and does not move state",
            node_ids=(draft.state_owner_id,) if draft.state_owner_id else (),
        ),
        _check(
            HardConstraintKind.CROSS_BOUNDARY_CYCLE,
            not draft.cycles,
            "no cycle crosses the proposed boundary"
            if not draft.cycles
            else "a cycle crosses the proposed boundary",
            node_ids=tuple(node_id for cycle in draft.cycles for node_id in cycle),
        ),
        _check(
            HardConstraintKind.MUTABLE_SHARING,
            not draft.shared_state,
            "no mutable store is written from both sides of the boundary"
            if not draft.shared_state
            else "mutable sharing crosses the proposed boundary",
            node_ids=draft.shared_state,
        ),
        _check(
            HardConstraintKind.SCOPE_ESCAPE,
            not escaped,
            "cluster does not escape into a sibling repository"
            if not escaped
            else "cluster escapes into a sibling repository",
            node_ids=escaped,
        ),
    )
    return tuple(checks)


def _rejection_kind(failed: Sequence[HardConstraintCheck]) -> RejectionKind:
    kinds = {item.kind for item in failed}
    if HardConstraintKind.UNRESOLVED_AMBIGUITY in kinds:
        return RejectionKind.UNRESOLVED_AMBIGUITY
    if HardConstraintKind.UNRESOLVED_AUTHORITY in kinds:
        return RejectionKind.UNRESOLVED_AUTHORITY
    if HardConstraintKind.INCOMPLETE_PROPOSAL in kinds:
        return RejectionKind.INCOMPLETE_PROPOSAL
    if HardConstraintKind.AUTHORITY_TRANSFER in kinds:
        return RejectionKind.AUTHORITY_TRANSFER
    if HardConstraintKind.STATE_MOVEMENT in kinds:
        return RejectionKind.STATE_MOVEMENT
    if HardConstraintKind.CROSS_BOUNDARY_CYCLE in kinds:
        return RejectionKind.CYCLE
    if HardConstraintKind.NO_EFFECT_EXPANSION in kinds:
        return RejectionKind.EFFECT_EXPANSION
    if HardConstraintKind.MUTABLE_SHARING in kinds or HardConstraintKind.DUAL_STATE_AUTHORITY in kinds:
        return RejectionKind.MUTABLE_SHARING
    return RejectionKind.HARD_CONSTRAINT


def _collect_ambiguities(
    contracts: ContractExtractionResult | None,
    ambiguities: Sequence[ContractAmbiguity],
) -> tuple[ContractAmbiguity, ...]:
    found: list[ContractAmbiguity] = list(ambiguities)
    if contracts is not None:
        found.extend(contracts.ambiguities)
    unique: dict[str, ContractAmbiguity] = {}
    for item in found:
        unique[item.content_identity] = item
    return tuple(sorted(unique.values(), key=lambda item: item.content_identity))


def _related_blockers(
    ownership: AuthorityOwnershipGraph,
    kind: BoundaryKind,
) -> tuple[OwnershipBlocker, ...]:
    blockers: list[OwnershipBlocker] = []
    for concern in BOUNDARY_CONCERNS[kind]:
        record = ownership.ownership_for(concern)
        if record.blocker is not None:
            blockers.append(record.blocker)
    if kind is BoundaryKind.LEGACY_COMPATIBILITY:
        blockers.extend(
            record.blocker
            for record in ownership.concerns
            if record.blocker is not None and record.legacy_owners
        )
    if kind is BoundaryKind.SIMULATION:
        blockers.extend(
            record.blocker
            for record in ownership.concerns
            if record.blocker is not None and record.simulation_owners
        )
    unique: dict[str, OwnershipBlocker] = {}
    for item in blockers:
        unique[item.content_identity] = item
    return tuple(sorted(unique.values(), key=lambda item: item.content_identity))


def _canonical_owner_id(
    ownership: AuthorityOwnershipGraph,
    kind: BoundaryKind,
) -> str:
    for concern in BOUNDARY_CONCERNS[kind]:
        record = ownership.ownership_for(concern)
        if record.canonical_owner is not None and record.blocker is None:
            return record.canonical_owner.node_id
    if kind is BoundaryKind.LEGACY_COMPATIBILITY:
        for record in ownership.concerns:
            if record.legacy_owners and record.canonical_owner is not None:
                return record.canonical_owner.node_id
    if kind is BoundaryKind.SIMULATION:
        for record in ownership.concerns:
            if record.simulation_owners and record.canonical_owner is not None:
                return record.canonical_owner.node_id
    return ""


def _unresolved_owner_id(
    ownership: AuthorityOwnershipGraph,
    kind: BoundaryKind,
    view: _GraphView,
    blockers: Sequence[OwnershipBlocker] = (),
) -> str:
    for blocker in blockers:
        for node_id in blocker.node_ids:
            if node_id in view.nodes_by_id:
                return node_id
    for concern in BOUNDARY_CONCERNS[kind]:
        record = ownership.ownership_for(concern)
        for group in (
            record.unknown_owners,
            record.adapters,
            record.projections,
            record.legacy_owners,
            record.simulation_owners,
        ):
            for owner in group:
                if owner.node_id in view.nodes_by_id:
                    return owner.node_id
    return ""


def _seed_nodes(
    ownership: AuthorityOwnershipGraph,
    view: _GraphView,
    kind: BoundaryKind,
) -> set[str]:
    seeds: set[str] = set()
    if kind is BoundaryKind.LEGACY_COMPATIBILITY:
        for record in ownership.concerns:
            for owner in record.legacy_owners:
                seeds.add(owner.node_id)
    elif kind is BoundaryKind.SIMULATION:
        for record in ownership.concerns:
            for owner in record.simulation_owners:
                seeds.add(owner.node_id)
        for node in view.architecture.nodes:
            if node.kind is NodeKind.SIMULATION:
                seeds.add(node.node_id)
    for concern in BOUNDARY_CONCERNS[kind]:
        record = ownership.ownership_for(concern)
        if record.canonical_owner is not None:
            seeds.add(record.canonical_owner.node_id)
        if record.blocker is not None:
            seeds.update(record.blocker.node_ids)
        for group in (
            record.adapters,
            record.projections,
            record.legacy_owners,
            record.simulation_owners,
            record.unknown_owners,
        ):
            for owner in group:
                seeds.add(owner.node_id)
    owner_id = _canonical_owner_id(ownership, kind)
    if owner_id:
        seeds.add(owner_id)
    return {node_id for node_id in seeds if node_id in view.nodes_by_id}


def _expand_cluster(view: _GraphView, seeds: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    queue: deque[str] = deque(seeds)
    while queue:
        node_id = queue.popleft()
        if node_id in seen or node_id not in view.nodes_by_id:
            continue
        seen.add(node_id)
        for edge in view.outgoing.get(node_id, ()):
            if edge.kind in _CLUSTER_EXPAND_EDGE_KINDS and edge.target not in seen:
                queue.append(edge.target)
        for edge in view.incoming.get(node_id, ()):
            if edge.kind in {EdgeKind.ADAPTS, EdgeKind.IMPLEMENTS, EdgeKind.REEXPORTS}:
                if edge.source not in seen:
                    queue.append(edge.source)
    return tuple(sorted(seen))


def _callers_of(view: _GraphView, cluster: Iterable[str]) -> tuple[str, ...]:
    inside = set(cluster)
    found: set[str] = set()
    for node_id in inside:
        for edge in view.incoming.get(node_id, ()):
            if edge.kind in _CALLER_EDGE_KINDS and edge.source not in inside:
                found.add(edge.source)
    return tuple(sorted(found))


def _effects_of(view: _GraphView, cluster: Iterable[str]) -> tuple[str, ...]:
    inside = set(cluster)
    found: set[str] = set()
    for edge in detect_cross_boundary_effects(view.architecture, inside):
        found.add(f"{edge.kind.value}:{edge.source}->{edge.target}")
    for node_id in inside:
        node = view.nodes_by_id[node_id]
        if node.kind is NodeKind.EFFECT:
            found.add(node.node_id)
        for edge in view.outgoing.get(node_id, ()):
            if edge.kind in _EFFECT_EDGE_KINDS:
                found.add(f"{edge.kind.value}:{edge.source}->{edge.target}")
    return tuple(sorted(found))


def _existing_effects(view: _GraphView, cluster: Iterable[str]) -> frozenset[str]:
    return frozenset(_effects_of(view, cluster))


def _adapters_of(
    ownership: AuthorityOwnershipGraph,
    kind: BoundaryKind,
) -> tuple[str, ...]:
    found: set[str] = set()
    concerns = BOUNDARY_CONCERNS[kind]
    records = (
        tuple(ownership.ownership_for(concern) for concern in concerns)
        if concerns
        else ownership.concerns
    )
    if kind is BoundaryKind.LEGACY_COMPATIBILITY:
        records = ownership.concerns
    if kind is BoundaryKind.SIMULATION:
        records = ownership.concerns
    for record in records:
        for owner in record.adapters:
            found.add(owner.node_id)
        for owner in record.projections:
            found.add(owner.node_id)
    return tuple(sorted(found))


def _deprecated_paths(
    ownership: AuthorityOwnershipGraph,
    view: _GraphView,
    kind: BoundaryKind,
    cluster: Iterable[str],
) -> tuple[str, ...]:
    found: set[str] = set()
    if kind is BoundaryKind.LEGACY_COMPATIBILITY:
        for record in ownership.concerns:
            for owner in record.legacy_owners:
                node = view.nodes_by_id.get(owner.node_id)
                if node is not None:
                    found.add(node.provenance.span.path)
        for node_id in cluster:
            node = view.nodes_by_id[node_id]
            if node.kind is NodeKind.COMPATIBILITY:
                found.add(node.provenance.span.path)
    if kind is BoundaryKind.SIMULATION:
        for record in ownership.concerns:
            for owner in record.simulation_owners:
                node = view.nodes_by_id.get(owner.node_id)
                if node is not None:
                    found.add(node.provenance.span.path)
        for node_id in cluster:
            node = view.nodes_by_id[node_id]
            if node.kind is NodeKind.SIMULATION:
                found.add(node.provenance.span.path)
    return tuple(sorted(found))


def _related_tests(view: _GraphView, cluster: Iterable[str]) -> tuple[str, ...]:
    inside = set(cluster)
    found: set[str] = set()
    for node_id in inside:
        node = view.nodes_by_id[node_id]
        if node.kind is NodeKind.TEST:
            found.add(node_id)
        for edge in view.incoming.get(node_id, ()):
            if edge.kind is EdgeKind.TESTS:
                found.add(edge.source)
        for edge in view.outgoing.get(node_id, ()):
            if edge.kind is EdgeKind.TESTS:
                found.add(edge.source if view.nodes_by_id[edge.source].kind is NodeKind.TEST else edge.target)
    return tuple(sorted(item for item in found if view.nodes_by_id[item].kind is NodeKind.TEST))


def _related_proofs(view: _GraphView, cluster: Iterable[str]) -> tuple[str, ...]:
    inside = set(cluster)
    found: set[str] = set()
    for node_id in inside:
        node = view.nodes_by_id[node_id]
        if node.kind is NodeKind.PROOF:
            found.add(node_id)
        for edge in view.incoming.get(node_id, ()):
            if edge.kind is EdgeKind.PROVES:
                found.add(edge.source)
        for edge in view.outgoing.get(node_id, ()):
            if edge.kind is EdgeKind.PROVES:
                found.add(edge.source if view.nodes_by_id[edge.source].kind is NodeKind.PROOF else edge.target)
    return tuple(sorted(item for item in found if view.nodes_by_id[item].kind is NodeKind.PROOF))


def _state_owner_id(view: _GraphView, cluster: Iterable[str], fallback: str) -> str:
    inside = set(cluster)
    states: list[str] = []
    for node_id in inside:
        if view.nodes_by_id[node_id].kind is NodeKind.STATE:
            states.append(node_id)
    if len(states) == 1:
        return states[0]
    persisted: list[str] = []
    for node_id in inside:
        for edge in view.outgoing.get(node_id, ()):
            if edge.kind in _MUTABLE_EDGE_KINDS and view.nodes_by_id[edge.target].kind is NodeKind.STATE:
                persisted.append(edge.target)
    unique = tuple(sorted(set(persisted)))
    if len(unique) == 1:
        return unique[0]
    if unique:
        return unique[0]
    if states:
        return sorted(states)[0]
    return fallback


def _public_symbols(view: _GraphView, cluster: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            node_id
            for node_id in cluster
            if view.nodes_by_id[node_id].kind in _PUBLIC_NODE_KINDS
        )
    )


def _prediction_for(
    view: _GraphView,
    cluster: Iterable[str],
    callers: Sequence[str],
    tests: Sequence[str],
    proofs: Sequence[str],
    public_symbols: Sequence[str],
) -> BoundaryPrediction:
    inside = tuple(cluster)
    current_cone = _walk(view, inside)
    proposed_cone_size = 1 + len(callers)
    current_context = tuple(
        node_id
        for node_id in sorted(current_cone)
        if view.nodes_by_id[node_id].kind in _CONTEXT_NODE_KINDS
    )
    proposed_context = min(len(current_context), proposed_cone_size + len(public_symbols[:1]))
    current_public = len(public_symbols)
    proposed_public = 1 if current_public else 0
    current_validation = len(set(tests) | set(proofs))
    proposed_validation = current_validation
    current_cone_size = max(len(current_cone), 1)
    return BoundaryPrediction(
        current_cone_size=current_cone_size,
        proposed_cone_size=min(proposed_cone_size, current_cone_size),
        cone_reduction=max(0, current_cone_size - min(proposed_cone_size, current_cone_size)),
        current_context_nodes=len(current_context),
        proposed_context_nodes=proposed_context,
        context_reduction=max(0, len(current_context) - proposed_context),
        current_public_symbols=current_public,
        proposed_public_symbols=proposed_public,
        public_symbol_reduction=max(0, current_public - proposed_public),
        current_validation_units=current_validation,
        proposed_validation_units=proposed_validation,
        validation_amplification_reduction=max(0, current_validation - proposed_validation),
        validation_coverage_loss=0,
    )


def _draft_for(
    kind: BoundaryKind,
    *,
    architecture: ArchitectureIR,
    ownership: AuthorityOwnershipGraph,
    view: _GraphView,
    ambiguities: Sequence[ContractAmbiguity],
) -> _Draft:
    seeds = _seed_nodes(ownership, view, kind)
    cluster = _expand_cluster(view, seeds)
    blockers = _related_blockers(ownership, kind)
    owner_id = _canonical_owner_id(ownership, kind) or _unresolved_owner_id(
        ownership, kind, view, blockers
    )
    callers = _callers_of(view, cluster)
    effects = _effects_of(view, cluster)
    adapters = _adapters_of(ownership, kind)
    deprecated = _deprecated_paths(ownership, view, kind, cluster)
    tests = _related_tests(view, cluster)
    proofs = _related_proofs(view, cluster)
    public_symbols = _public_symbols(view, cluster)
    state_owner_id = _state_owner_id(view, cluster, owner_id)
    affecting = tuple(
        item for item in ambiguities if _ambiguity_affects(item, cluster, view)
    )
    effect_edges = detect_cross_boundary_effects(architecture, cluster)
    shared_state = detect_mutable_sharing(architecture, cluster)
    cycles = detect_cross_boundary_cycles(architecture, cluster)
    costs = measure_boundary_costs(architecture, cluster)
    prediction = _prediction_for(view, cluster, callers, tests, proofs, public_symbols)
    concerns = tuple(item.value for item in BOUNDARY_CONCERNS[kind])
    return _Draft(
        kind=kind,
        owner_id=owner_id,
        state_owner_id=state_owner_id,
        cluster=cluster,
        callers=callers,
        effects=effects,
        adapters=adapters,
        deprecated_paths=deprecated,
        tests=tests,
        proofs=proofs,
        public_symbols=public_symbols,
        concerns=concerns,
        blockers=blockers,
        ambiguities=affecting,
        effect_edges=effect_edges,
        shared_state=shared_state,
        cycles=cycles,
        prediction=prediction,
        costs=costs,
        existing_effects=_existing_effects(view, cluster),
        existing_proofs=frozenset(proofs),
        existing_tests=frozenset(tests),
    )


def _proposal_from_draft(
    draft: _Draft,
    *,
    repository_tree: str,
    freshness: str,
    checks: Sequence[HardConstraintCheck],
) -> BoundaryProposal:
    interface = BoundaryInterface(
        name=_interface_name(draft.kind),
        kind=draft.kind,
        canonical_owner_node_id=draft.owner_id,
        concerns=draft.concerns,
        allowed_callers=draft.callers,
        allowed_effects=draft.effects,
        public_symbols=draft.public_symbols,
    )
    migration = BoundaryMigration(
        state_owner_node_id=draft.state_owner_id,
        adapters=draft.adapters,
        deprecated_paths=draft.deprecated_paths,
    )
    return BoundaryProposal(
        kind=draft.kind,
        interface=interface,
        canonical_owner_node_id=draft.owner_id,
        state_owner_node_id=draft.state_owner_id,
        callers=draft.callers,
        effects=draft.effects,
        adapters=draft.adapters,
        deprecated_paths=draft.deprecated_paths,
        tests=draft.tests,
        proofs=draft.proofs,
        migration=migration,
        rollback=BoundaryRollback(),
        prediction=draft.prediction,
        costs=draft.costs,
        hard_constraints=tuple(checks),
        repository_tree=repository_tree,
        freshness=freshness,
    )


def _rejection_from_draft(
    draft: _Draft,
    *,
    repository_tree: str,
    freshness: str,
    checks: Sequence[HardConstraintCheck],
) -> RejectedBoundaryProposal:
    failed = tuple(item for item in checks if item.passed is not True)
    message = failed[0].message if failed else "proposal is unsafe"
    return RejectedBoundaryProposal(
        kind=draft.kind,
        rejection_kind=_rejection_kind(failed),
        message=message,
        hard_constraints=tuple(checks),
        costs=draft.costs,
        repository_tree=repository_tree,
        freshness=freshness,
        canonical_owner_node_id=draft.owner_id,
        node_ids=draft.cluster,
    )


def synthesize_interface_boundaries(
    architecture: ArchitectureIR | Mapping[str, Any],
    ownership: AuthorityOwnershipGraph | Mapping[str, Any],
    *,
    contracts: ContractExtractionResult | Mapping[str, Any] | None = None,
    ambiguities: Sequence[ContractAmbiguity | Mapping[str, Any]] = (),
    freshness: str | None = None,
) -> BoundarySynthesisResult:
    """Propose candidate interfaces for the closed initial boundary set."""

    graph = _require_architecture_ir(architecture)
    owners = _require_ownership(ownership)
    if owners.repository_tree != graph.repository_tree:
        raise BoundarySynthesizerError(
            "ownership repository_tree must match ArchitectureIR"
        )
    if owners.architecture_ir_identity != graph.content_identity:
        raise BoundarySynthesizerError(
            "ownership architecture_ir_identity must match ArchitectureIR"
        )
    parsed_contracts: ContractExtractionResult | None
    if contracts is None:
        parsed_contracts = None
    elif isinstance(contracts, ContractExtractionResult):
        parsed_contracts = contracts
    else:
        parsed_contracts = ContractExtractionResult.from_mapping(contracts)
    parsed_ambiguities = tuple(
        item if isinstance(item, ContractAmbiguity) else ContractAmbiguity.from_mapping(item)
        for item in ambiguities
    )
    all_ambiguities = _collect_ambiguities(parsed_contracts, parsed_ambiguities)
    bound_freshness = freshness or graph.freshness
    view = _build_view(graph)
    proposals: list[BoundaryProposal] = []
    rejections: list[RejectedBoundaryProposal] = []
    for kind in INITIAL_BOUNDARIES:
        draft = _draft_for(
            kind,
            architecture=graph,
            ownership=owners,
            view=view,
            ambiguities=all_ambiguities,
        )
        checks = evaluate_hard_constraints(draft, view)
        if all(item.passed for item in checks) and draft.owner_id:
            proposals.append(
                _proposal_from_draft(
                    draft,
                    repository_tree=graph.repository_tree,
                    freshness=bound_freshness,
                    checks=checks,
                )
            )
        else:
            rejections.append(
                _rejection_from_draft(
                    draft,
                    repository_tree=graph.repository_tree,
                    freshness=bound_freshness,
                    checks=checks,
                )
            )
    ranking = rank_boundary_proposals(proposals, rejections)
    return BoundarySynthesisResult(
        architecture_ir_identity=graph.content_identity,
        ownership_identity=owners.content_identity,
        repository_tree=graph.repository_tree,
        freshness=bound_freshness,
        proposals=tuple(proposals),
        rejections=tuple(rejections),
        ranking=ranking,
        ranking_inputs=_ranking_inputs_for(proposals),
    )


class InterfaceBoundarySynthesizer:
    """Read-only planner that emits bounded candidate interface proposals."""

    def __init__(
        self,
        architecture: ArchitectureIR | Mapping[str, Any],
        ownership: AuthorityOwnershipGraph | Mapping[str, Any],
        *,
        contracts: ContractExtractionResult | Mapping[str, Any] | None = None,
        ambiguities: Sequence[ContractAmbiguity | Mapping[str, Any]] = (),
        freshness: str | None = None,
    ) -> None:
        self._architecture = _require_architecture_ir(architecture)
        self._ownership = _require_ownership(ownership)
        if isinstance(contracts, Mapping):
            self._contracts: ContractExtractionResult | None = (
                ContractExtractionResult.from_mapping(contracts)
            )
        else:
            self._contracts = contracts
        self._ambiguities = tuple(
            item if isinstance(item, ContractAmbiguity) else ContractAmbiguity.from_mapping(item)
            for item in ambiguities
        )
        self._freshness = freshness

    @property
    def effect_class(self) -> str:
        return EFFECT_CLASS

    @property
    def can_apply_proposals(self) -> bool:
        return SYNTHESIZER_CAN_APPLY_PROPOSALS

    def synthesize(self) -> BoundarySynthesisResult:
        return synthesize_interface_boundaries(
            self._architecture,
            self._ownership,
            contracts=self._contracts,
            ambiguities=self._ambiguities,
            freshness=self._freshness,
        )

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_autonomous_application("apply")

    def transfer_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_transfer("transfer")

    def promote_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_promotion("promote")

    def mutate_state(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_state_mutation("mutate")

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_ownership_authorization("authorize")

    def override_hard_constraint(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_hard_constraint_override("override")


__all__ = [
    "BOUNDARY_CONCERNS",
    "BOUNDARY_COST_SCHEMA",
    "BOUNDARY_COST_VECTOR_SCHEMA",
    "BOUNDARY_COST_VECTOR_VERSION",
    "BOUNDARY_COST_VERSION",
    "BOUNDARY_INTERFACE_SCHEMA",
    "BOUNDARY_INTERFACE_VERSION",
    "BOUNDARY_MIGRATION_SCHEMA",
    "BOUNDARY_MIGRATION_VERSION",
    "BOUNDARY_PREDICTION_SCHEMA",
    "BOUNDARY_PREDICTION_VERSION",
    "BOUNDARY_PROPOSAL_EVIDENCE",
    "BOUNDARY_PROPOSAL_SCHEMA",
    "BOUNDARY_PROPOSAL_VERSION",
    "BOUNDARY_REJECTION_SCHEMA",
    "BOUNDARY_REJECTION_VERSION",
    "BOUNDARY_ROLLBACK_SCHEMA",
    "BOUNDARY_ROLLBACK_VERSION",
    "BOUNDARY_SYNTHESIS_SCHEMA",
    "BOUNDARY_SYNTHESIS_VERSION",
    "CANDIDATE_INTERFACES_ONLY",
    "CLOSED_BOUNDARIES",
    "CLOSED_COST_DIMENSIONS",
    "CLOSED_COST_UNITS",
    "CLOSED_HARD_CONSTRAINTS",
    "CLOSED_INTERFACE_STABILITIES",
    "CLOSED_PROPOSAL_DISPOSITIONS",
    "CLOSED_REJECTION_KINDS",
    "DEFAULT_FRESHNESS",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "HARD_CONSTRAINTS_PRECEDE_RANKING",
    "HARD_CONSTRAINT_SCHEMA",
    "HARD_CONSTRAINT_VERSION",
    "INITIAL_BOUNDARIES",
    "MISSING_PLAN_HARD_GATES",
    "RANKING_INPUT_SCHEMA",
    "RANKING_INPUT_VERSION",
    "RANKING_IS_NON_PROBATIVE",
    "REQUIRED_BOUNDARIES",
    "REQUIRED_COST_DIMENSIONS",
    "REQUIRED_HARD_CONSTRAINTS",
    "SYNTHESIZER_CAN_APPLY_PROPOSALS",
    "SYNTHESIZER_CAN_AUTHORIZE_CHANGES",
    "SYNTHESIZER_CAN_MUTATE_STATE",
    "SYNTHESIZER_CAN_OVERRIDE_HARD_CONSTRAINTS",
    "SYNTHESIZER_CAN_PROMOTE_AUTHORITY",
    "SYNTHESIZER_CAN_TRANSFER_AUTHORITY",
    "TASK_ID",
    "UNRESOLVED_AMBIGUITY_REJECTS",
    "UNRESOLVED_AUTHORITY_REJECTS",
    "BoundaryCostMeasure",
    "BoundaryCostVector",
    "BoundaryInterface",
    "BoundaryKind",
    "BoundaryMigration",
    "BoundaryPrediction",
    "BoundaryProposal",
    "BoundaryRankingInput",
    "BoundaryRollback",
    "BoundarySynthesizerAuthorityError",
    "BoundarySynthesizerError",
    "BoundarySynthesisResult",
    "CostDimensionKind",
    "HardConstraintCheck",
    "HardConstraintKind",
    "InterfaceBoundarySynthesizer",
    "InterfaceStability",
    "ProposalDisposition",
    "RejectedBoundaryProposal",
    "RejectionKind",
    "admit_rejected_by_cost",
    "detect_cross_boundary_cycles",
    "detect_cross_boundary_effects",
    "detect_mutable_sharing",
    "evaluate_hard_constraints",
    "measure_boundary_costs",
    "rank_boundary_proposals",
    "ranking_establishes",
    "refuse_autonomous_application",
    "refuse_authority_promotion",
    "refuse_authority_transfer",
    "refuse_hard_constraint_override",
    "refuse_ownership_authorization",
    "refuse_state_mutation",
    "synthesize_interface_boundaries",
]
