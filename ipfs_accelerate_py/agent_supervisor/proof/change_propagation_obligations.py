"""Fail-closed lowering of change-propagation evidence into LogicIR obligations.

RPR-035 / ``ChangePropagationObligations@1``

This module does not decide that a migration is safe.  It turns exact,
source-bound facts (contract deltas, impact closures, consumer migration
records, provenance graphs, and required-behavior contracts) into immutable,
bounded proof obligations.  Every automated migration choice is expressed as
finite premises, conclusions, reviewed assumptions, explicit unsupported
semantics, and counterexample targets.

No retrieved, vector, GraphRAG, or model statement becomes an axiom.  Solver
and premise candidates never grant authority; reconstruction under pinned
translator/toolchain/policy roots remains a later stage (RPR-036).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    DeltaDisposition,
    ImpactClosureReceipt,
    ImpactCompleteness,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
    ValueCandidate,
    ValueCandidateDisposition,
    ValueCandidateKind,
)
from ..analysis.value_provenance_graph import ValueProvenanceGraph
from ..integrations.change_propagation_capabilities import (
    ChangePropagationCapabilityReport,
    ChangePropagationCapabilityStatus,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
)


CHANGE_PROPAGATION_OBLIGATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-obligation@1"
)
CHANGE_PROPAGATION_IR_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-ir-claim@1"
)
VALUE_MAPPING_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/value-mapping-claim@1"
)
BEHAVIOR_REFINEMENT_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/behavior-refinement-claim@1"
)
CHANGE_PROPAGATION_OBLIGATIONS_INTERFACE: Final = "ChangePropagationObligations@1"
CHANGE_PROPAGATION_OBLIGATION_COMPILER_ID: Final = (
    "change-propagation-obligation-compiler@1"
)
MAX_OBLIGATIONS: Final = 128
MAX_CLAIMS: Final = 64
MAX_UNSUPPORTED: Final = 64
MAX_COUNTEREXAMPLE_TARGETS: Final = 64
MAX_PREMISE_IDS: Final = 256
_LOGIC_CAPABILITY_ID: Final = "datasets.logic_ir"


class ChangePropagationObligationError(ContractValidationError):
    """Evidence is insufficient or inconsistent for a sound lowering."""


class IncompleteImpactClosureError(ChangePropagationObligationError):
    """A partial impact closure attempted to claim a closed obligation set."""


class UnsupportedObligationError(ChangePropagationObligationError):
    """The admitted LogicIR capability cannot express an obligation."""


class UnauthorizedAssumptionError(ChangePropagationObligationError):
    """A retrieved or model statement attempted to become an axiom."""


class ObligationKind(str, Enum):
    """Closed set of separately lowerable change-propagation obligations."""

    CLOSURE_COVERAGE = "closure_coverage"
    CONSUMER_COMPATIBILITY = "consumer_compatibility"
    SOURCE_SCOPE_PATH_AVAILABILITY = "source_scope_path_availability"
    TYPE_SCHEMA_RANGE_NULLABILITY = "type_schema_range_nullability"
    INFORMATION_SUFFICIENCY = "information_sufficiency"
    CONVERSION_CONSTRUCTOR_TOTALITY = "conversion_constructor_totality"
    ERROR_COMPATIBILITY = "error_compatibility"
    EFFECT_COMPATIBILITY = "effect_compatibility"
    CAPABILITY_COMPATIBILITY = "capability_compatibility"
    AUTHORIZATION_COMPATIBILITY = "authorization_compatibility"
    TRUST_COMPATIBILITY = "trust_compatibility"
    RESOURCE_COMPATIBILITY = "resource_compatibility"
    OWNERSHIP_LIFETIME = "ownership_lifetime"
    MUTATION_CONCURRENCY = "mutation_concurrency"
    DEPENDENCY_CYCLE_ABSENCE = "dependency_cycle_absence"
    PARAMETER_THREADING = "parameter_threading"
    BEHAVIOR_INVARIANTS = "behavior_invariants"
    STATE_TRANSITIONS = "state_transitions"
    SERIALIZATION_MIGRATION = "serialization_migration"
    PLACEMENT = "placement"


class UnsupportedSemanticKind(str, Enum):
    """Closed vocabulary for analysis that cannot be approximated as axioms."""

    DYNAMIC = "dynamic"
    NATIVE = "native"
    REFLECTION = "reflection"
    FFI = "ffi"
    STRING_DISPATCH = "string_dispatch"
    PLUGIN_LOADING = "plugin_loading"
    GENERATED_SOURCE = "generated_source"
    MONKEY_PATCH = "monkey_patch"
    REMOTE_SERVICE = "remote_service"
    REGISTRY_UNKNOWN = "registry_unknown"
    FRONTIER_EDGE = "frontier_edge"
    FRONTIER_NODE = "frontier_node"


_RETRIEVED_ASSUMPTION_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "invented",
        "model",
        "llm",
        "retrieved",
        "vector",
        "graphrag",
        "similarity",
        "embedding",
        "hypothesis",
        "nomination",
        "candidate_only",
    }
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ChangePropagationObligationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ChangePropagationObligationError(f"{name} is required")
    if result and any(character.isspace() for character in result):
        raise ChangePropagationObligationError(f"{name} must be an opaque identifier")
    return result


def _ids(
    value: Sequence[str],
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_PREMISE_IDS,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ChangePropagationObligationError(f"{name} must be identifiers")
    if len(value) > limit:
        raise ChangePropagationObligationError(f"{name} exceeds its bounded size")
    result = tuple(sorted({_text(item, name) for item in value}))
    if required and not result:
        raise ChangePropagationObligationError(f"{name} must not be empty")
    return result


def _roots_match(left: PropagationAuthorityRoots, right: PropagationAuthorityRoots) -> bool:
    return left.to_dict() == right.to_dict()


@dataclass(frozen=True)
class AssumptionBinding:
    """A reviewed assumption identity, never a free-form solver axiom."""

    assumption_id: str
    kind: str
    evidence_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumption_id", _text(self.assumption_id, "assumption_id"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(self, "evidence_ref", _text(self.evidence_ref, "evidence_ref"))
        kind = self.kind.casefold().replace("-", "_")
        if "assumption" not in kind:
            raise UnauthorizedAssumptionError(
                "assumptions must be reviewed assumption evidence, never free-form axioms"
            )
        if any(marker in kind for marker in _RETRIEVED_ASSUMPTION_MARKERS):
            raise UnauthorizedAssumptionError(
                "retrieved, model, vector, or nomination statements cannot become axioms"
            )
        if any(
            marker in self.assumption_id.casefold() or marker in self.evidence_ref.casefold()
            for marker in ("llm:", "model:", "vector:", "retrieved:")
        ):
            raise UnauthorizedAssumptionError(
                "retrieved/model identifiers cannot be used as assumption evidence"
            )


@dataclass(frozen=True)
class UnsupportedSemantic:
    """An explicit unsupported analysis surface that stays non-authoritative."""

    kind: UnsupportedSemanticKind
    subject_id: str
    reason_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", UnsupportedSemanticKind(self.kind))
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "reason_ref", _text(self.reason_ref, "reason_ref"))

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "reason_ref": self.reason_ref,
        }


@dataclass(frozen=True)
class LogicCapabilityBinding:
    """Exact capability-report fact authorizing LogicIR lowering."""

    capability_id: str
    capability_revision: str
    reconstruction_compatible: bool = True
    supported_semantics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _text(self.capability_id, "capability_id"))
        object.__setattr__(
            self, "capability_revision", _text(self.capability_revision, "capability_revision")
        )
        if not self.reconstruction_compatible:
            raise UnsupportedObligationError(
                "LogicIR lowering requires independently reconstructable capability semantics"
            )
        semantics = _ids(self.supported_semantics, "supported_semantics")
        if "ir" not in semantics:
            raise UnsupportedObligationError(
                "capability report must explicitly admit LogicIR semantics"
            )
        object.__setattr__(self, "supported_semantics", semantics)

    @classmethod
    def from_report(cls, report: ChangePropagationCapabilityReport) -> "LogicCapabilityBinding":
        if not isinstance(report, ChangePropagationCapabilityReport):
            raise ChangePropagationObligationError(
                "capability_report must be ChangePropagationCapabilityReport"
            )
        try:
            capability = report.capability(_LOGIC_CAPABILITY_ID)
        except KeyError as exc:
            raise UnsupportedObligationError(
                f"capability report lacks {_LOGIC_CAPABILITY_ID}"
            ) from exc
        if capability.status is not ChangePropagationCapabilityStatus.AVAILABLE:
            raise UnsupportedObligationError(
                f"{_LOGIC_CAPABILITY_ID} is not available in this report"
            )
        if not capability.reconstruction_compatible:
            raise UnsupportedObligationError(
                f"{_LOGIC_CAPABILITY_ID} is not reconstruction compatible"
            )
        if "ir" not in capability.supported_semantics:
            raise UnsupportedObligationError(
                f"{_LOGIC_CAPABILITY_ID} does not declare LogicIR semantics"
            )
        revision = str(
            capability.details.get("capability_revision")
            or capability.interface_version
            or capability.schema_version
        )
        if not revision:
            raise UnsupportedObligationError(
                f"{_LOGIC_CAPABILITY_ID} has no pinned capability revision"
            )
        return cls(
            capability.capability_id,
            revision,
            capability.reconstruction_compatible,
            capability.supported_semantics,
        )


@dataclass(frozen=True)
class ObligationContext:
    """Exact non-semantic bindings shared by every emitted claim."""

    assumptions: tuple[AssumptionBinding, ...]
    capability: LogicCapabilityBinding
    unsupported_semantics: tuple[UnsupportedSemantic, ...] = ()
    allow_partial_frontier: bool = False
    counterexample_policy_ref: str = "counterexample:change-propagation@1"

    def __post_init__(self) -> None:
        if not isinstance(self.capability, LogicCapabilityBinding):
            raise ChangePropagationObligationError(
                "obligation context requires a LogicIR capability binding"
            )
        if not isinstance(self.assumptions, Sequence) or not self.assumptions:
            raise ChangePropagationObligationError(
                "obligation context requires reviewed assumption bindings"
            )
        if not all(isinstance(item, AssumptionBinding) for item in self.assumptions):
            raise ChangePropagationObligationError(
                "assumptions must be AssumptionBinding values"
            )
        if not isinstance(self.unsupported_semantics, Sequence):
            raise ChangePropagationObligationError(
                "unsupported_semantics must be a sequence"
            )
        if len(self.unsupported_semantics) > MAX_UNSUPPORTED:
            raise ChangePropagationObligationError(
                "unsupported_semantics exceeds its bounded size"
            )
        if not all(
            isinstance(item, UnsupportedSemantic) for item in self.unsupported_semantics
        ):
            raise ChangePropagationObligationError(
                "unsupported_semantics must be UnsupportedSemantic values"
            )
        if not isinstance(self.allow_partial_frontier, bool):
            raise ChangePropagationObligationError(
                "allow_partial_frontier must be boolean"
            )
        object.__setattr__(
            self,
            "assumptions",
            tuple(sorted(self.assumptions, key=lambda item: item.assumption_id)),
        )
        object.__setattr__(
            self,
            "unsupported_semantics",
            tuple(
                sorted(
                    self.unsupported_semantics,
                    key=lambda item: (item.kind.value, item.subject_id, item.reason_ref),
                )
            ),
        )
        object.__setattr__(
            self,
            "counterexample_policy_ref",
            _text(self.counterexample_policy_ref, "counterexample_policy_ref"),
        )

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return tuple(item.assumption_id for item in self.assumptions)


@dataclass(frozen=True)
class IRClaim(CanonicalContract):
    """Small immutable LogicIR claim with every source of authority named."""

    SCHEMA: ClassVar[str] = CHANGE_PROPAGATION_IR_CLAIM_SCHEMA

    predicate: str
    subject_id: str
    premise_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    repository_id: str
    tree_id: str
    graph_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str
    capability_id: str
    capability_revision: str
    counterexample_targets: tuple[str, ...] = ()
    unsupported_semantic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "predicate",
            "subject_id",
            "repository_id",
            "tree_id",
            "graph_id",
            "translator_id",
            "toolchain_id",
            "policy_id",
            "capability_id",
            "capability_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "premise_ids",
            "source_ids",
            "assumption_ids",
            "counterexample_targets",
            "unsupported_semantic_ids",
        ):
            required = name in {"premise_ids", "source_ids", "assumption_ids"}
            limit = (
                MAX_COUNTEREXAMPLE_TARGETS
                if name == "counterexample_targets"
                else MAX_PREMISE_IDS
            )
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=required, limit=limit),
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "subject_id": self.subject_id,
            "premise_ids": list(self.premise_ids),
            "source_ids": list(self.source_ids),
            "assumption_ids": list(self.assumption_ids),
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "graph_id": self.graph_id,
            "translator_id": self.translator_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "counterexample_targets": list(self.counterexample_targets),
            "unsupported_semantic_ids": list(self.unsupported_semantic_ids),
        }

    def to_logic_ir(self) -> dict[str, Any]:
        """Backend-neutral, fully bound lowering payload."""

        return self.to_dict()


@dataclass(frozen=True)
class ValueMappingClaim(CanonicalContract):
    """Finite claim that a nominated source satisfies one missing input facet set."""

    SCHEMA: ClassVar[str] = VALUE_MAPPING_CLAIM_SCHEMA

    claim_id_seed: str
    requirement_id: str
    candidate_id: str
    consumer_id: str
    parameter_name: str
    expression_ref: str
    type_ref: str
    premise_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    facet_predicates: tuple[str, ...]
    counterexample_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "claim_id_seed",
            "requirement_id",
            "candidate_id",
            "consumer_id",
            "parameter_name",
            "expression_ref",
            "type_ref",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "premise_ids",
            "source_ids",
            "assumption_ids",
            "facet_predicates",
            "counterexample_targets",
        ):
            required = name != "counterexample_targets"
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=required),
            )

    @property
    def claim_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "claim_id_seed": self.claim_id_seed,
            "requirement_id": self.requirement_id,
            "candidate_id": self.candidate_id,
            "consumer_id": self.consumer_id,
            "parameter_name": self.parameter_name,
            "expression_ref": self.expression_ref,
            "type_ref": self.type_ref,
            "premise_ids": list(self.premise_ids),
            "source_ids": list(self.source_ids),
            "assumption_ids": list(self.assumption_ids),
            "facet_predicates": list(self.facet_predicates),
            "counterexample_targets": list(self.counterexample_targets),
        }


@dataclass(frozen=True)
class BehaviorRefinementClaim(CanonicalContract):
    """Finite claim that required behavior refines admitted evidence precedence."""

    SCHEMA: ClassVar[str] = BEHAVIOR_REFINEMENT_CLAIM_SCHEMA

    claim_id_seed: str
    behavior_id: str
    subject_symbol_id: str
    evidence_precedence: str
    premise_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    structural_clause_ids: tuple[str, ...]
    placement_decision_ref: str = ""
    counterexample_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "claim_id_seed",
            "behavior_id",
            "subject_symbol_id",
            "evidence_precedence",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "placement_decision_ref",
            _text(self.placement_decision_ref, "placement_decision_ref", required=False),
        )
        for name in (
            "premise_ids",
            "source_ids",
            "assumption_ids",
            "structural_clause_ids",
            "counterexample_targets",
        ):
            required = name != "counterexample_targets"
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=required),
            )
        if self.evidence_precedence == BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS.value:
            raise UnauthorizedAssumptionError(
                "implementation hypotheses cannot become behavior-refinement axioms"
            )

    @property
    def claim_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "claim_id_seed": self.claim_id_seed,
            "behavior_id": self.behavior_id,
            "subject_symbol_id": self.subject_symbol_id,
            "evidence_precedence": self.evidence_precedence,
            "premise_ids": list(self.premise_ids),
            "source_ids": list(self.source_ids),
            "assumption_ids": list(self.assumption_ids),
            "structural_clause_ids": list(self.structural_clause_ids),
            "placement_decision_ref": self.placement_decision_ref,
            "counterexample_targets": list(self.counterexample_targets),
        }


@dataclass(frozen=True)
class ChangePropagationObligation(CanonicalContract):
    """One consumer-specific LogicIR claim plus its code-proof envelope."""

    SCHEMA: ClassVar[str] = CHANGE_PROPAGATION_OBLIGATION_SCHEMA

    kind: ObligationKind
    consumer_id: str
    delta_id: str
    claim: IRClaim
    code_obligation: CodeProofObligation
    source_ids: tuple[str, ...]
    value_mapping_claim_id: str = ""
    behavior_refinement_claim_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ObligationKind(self.kind))
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(self, "delta_id", _text(self.delta_id, "delta_id"))
        if not isinstance(self.claim, IRClaim):
            raise ChangePropagationObligationError("obligation requires a typed IRClaim")
        if not isinstance(self.code_obligation, CodeProofObligation):
            raise ChangePropagationObligationError(
                "obligation requires a typed CodeProofObligation"
            )
        object.__setattr__(self, "source_ids", _ids(self.source_ids, "source_ids"))
        object.__setattr__(
            self,
            "value_mapping_claim_id",
            _text(self.value_mapping_claim_id, "value_mapping_claim_id", required=False),
        )
        object.__setattr__(
            self,
            "behavior_refinement_claim_id",
            _text(
                self.behavior_refinement_claim_id,
                "behavior_refinement_claim_id",
                required=False,
            ),
        )
        if self.code_obligation.repository_tree_id != self.claim.tree_id:
            raise ChangePropagationObligationError(
                "code obligation tree must match the LogicIR claim"
            )
        if set(self.claim.source_ids).difference(self.source_ids):
            raise ChangePropagationObligationError(
                "claim source ids must be carried by source_ids"
            )

    @property
    def obligation_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "consumer_id": self.consumer_id,
            "delta_id": self.delta_id,
            "claim": self.claim.to_dict(),
            "code_obligation": self.code_obligation.to_dict(),
            "source_ids": list(self.source_ids),
            "value_mapping_claim_id": self.value_mapping_claim_id,
            "behavior_refinement_claim_id": self.behavior_refinement_claim_id,
        }


@dataclass(frozen=True)
class ChangePropagationObligationCompilation:
    """Complete deterministic collection compiled for one consumer and delta."""

    roots: PropagationAuthorityRoots
    delta_id: str
    consumer_id: str
    migration_obligation_id: str
    obligations: tuple[ChangePropagationObligation, ...]
    value_mapping_claims: tuple[ValueMappingClaim, ...] = ()
    behavior_refinement_claims: tuple[BehaviorRefinementClaim, ...] = ()
    unsupported_semantics: tuple[UnsupportedSemantic, ...] = ()
    compiler_id: str = CHANGE_PROPAGATION_OBLIGATION_COMPILER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ChangePropagationObligationError(
                "compilation requires PropagationAuthorityRoots"
            )
        object.__setattr__(self, "delta_id", _text(self.delta_id, "delta_id"))
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(
            self,
            "migration_obligation_id",
            _text(self.migration_obligation_id, "migration_obligation_id"),
        )
        object.__setattr__(
            self, "compiler_id", _text(self.compiler_id, "compiler_id")
        )
        if not self.obligations:
            raise ChangePropagationObligationError(
                "compilation requires a nonempty obligation set"
            )
        if len(self.obligations) > MAX_OBLIGATIONS:
            raise ChangePropagationObligationError(
                "compilation exceeds bounded obligation set"
            )
        if not all(
            isinstance(item, ChangePropagationObligation) for item in self.obligations
        ):
            raise ChangePropagationObligationError(
                "compilation obligations must be typed"
            )
        kinds = [item.kind for item in self.obligations]
        if len(kinds) != len(set(kinds)):
            raise ChangePropagationObligationError(
                "compilation contains duplicate obligation kinds"
            )
        if any(item.consumer_id != self.consumer_id for item in self.obligations):
            raise ChangePropagationObligationError(
                "obligations must bind one exact consumer"
            )
        if any(item.delta_id != self.delta_id for item in self.obligations):
            raise ChangePropagationObligationError(
                "obligations must bind one exact delta"
            )
        if len(self.value_mapping_claims) > MAX_CLAIMS:
            raise ChangePropagationObligationError(
                "value mapping claims exceed bound"
            )
        if len(self.behavior_refinement_claims) > MAX_CLAIMS:
            raise ChangePropagationObligationError(
                "behavior refinement claims exceed bound"
            )
        object.__setattr__(
            self,
            "obligations",
            tuple(sorted(self.obligations, key=lambda item: item.kind.value)),
        )
        object.__setattr__(
            self,
            "value_mapping_claims",
            tuple(
                sorted(
                    self.value_mapping_claims,
                    key=lambda item: (item.requirement_id, item.candidate_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "behavior_refinement_claims",
            tuple(
                sorted(
                    self.behavior_refinement_claims,
                    key=lambda item: item.behavior_id,
                )
            ),
        )
        object.__setattr__(
            self,
            "unsupported_semantics",
            tuple(
                sorted(
                    self.unsupported_semantics,
                    key=lambda item: (item.kind.value, item.subject_id),
                )
            ),
        )

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(item.obligation_id for item in self.obligations)

    @property
    def kinds(self) -> tuple[ObligationKind, ...]:
        return tuple(item.kind for item in self.obligations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": CHANGE_PROPAGATION_OBLIGATIONS_INTERFACE,
            "compiler_id": self.compiler_id,
            "roots": self.roots.to_dict(),
            "delta_id": self.delta_id,
            "consumer_id": self.consumer_id,
            "migration_obligation_id": self.migration_obligation_id,
            "obligation_ids": list(self.obligation_ids),
            "kinds": [item.value for item in self.kinds],
            "obligations": [item.to_dict() for item in self.obligations],
            "value_mapping_claims": [
                item.to_dict() for item in self.value_mapping_claims
            ],
            "behavior_refinement_claims": [
                item.to_dict() for item in self.behavior_refinement_claims
            ],
            "unsupported_semantics": [
                item.to_dict() for item in self.unsupported_semantics
            ],
        }


# Facet obligation kinds always required for value-mapping migrations.
_VALUE_MAPPING_FACETS: Final[tuple[ObligationKind, ...]] = (
    ObligationKind.SOURCE_SCOPE_PATH_AVAILABILITY,
    ObligationKind.TYPE_SCHEMA_RANGE_NULLABILITY,
    ObligationKind.INFORMATION_SUFFICIENCY,
    ObligationKind.CONVERSION_CONSTRUCTOR_TOTALITY,
    ObligationKind.ERROR_COMPATIBILITY,
    ObligationKind.EFFECT_COMPATIBILITY,
    ObligationKind.CAPABILITY_COMPATIBILITY,
    ObligationKind.AUTHORIZATION_COMPATIBILITY,
    ObligationKind.TRUST_COMPATIBILITY,
    ObligationKind.RESOURCE_COMPATIBILITY,
    ObligationKind.OWNERSHIP_LIFETIME,
    ObligationKind.MUTATION_CONCURRENCY,
    ObligationKind.DEPENDENCY_CYCLE_ABSENCE,
    ObligationKind.PARAMETER_THREADING,
)

_VALUE_MAPPING_FACET_PREDICATES: Final[tuple[str, ...]] = tuple(
    kind.value for kind in _VALUE_MAPPING_FACETS
)

_NOMINATION_ONLY_KINDS: Final[frozenset[ValueCandidateKind]] = frozenset(
    {
        ValueCandidateKind.VECTOR_NOMINATION,
        ValueCandidateKind.GRAPH_NOMINATION,
        ValueCandidateKind.HISTORY,
    }
)


class ChangePropagationObligationCompiler:
    """Compile mandatory LogicIR obligations without ranking or admitting repairs."""

    def compile(
        self,
        delta: ProgramContractDelta,
        closure: ImpactClosureReceipt,
        consumer_obligation: ConsumerMigrationObligation,
        context: ObligationContext,
        *,
        missing_inputs: Sequence[MissingInputRequirement] = (),
        value_candidates: Sequence[ValueCandidate] = (),
        value_provenance: ValueProvenanceGraph | None = None,
        behavior_contracts: Sequence[RequiredBehaviorContract] = (),
        graph_id: str = "",
    ) -> ChangePropagationObligationCompilation:
        roots = self._validate_bindings(
            delta,
            closure,
            consumer_obligation,
            context,
            missing_inputs,
            value_candidates,
            value_provenance,
            behavior_contracts,
        )
        unsupported = list(context.unsupported_semantics)
        unsupported.extend(self._closure_frontier_unsupported(closure, context))

        resolved_graph_id = self._resolve_graph_id(
            roots, value_provenance, graph_id, consumer_obligation
        )
        source_ids = self._collect_source_ids(
            delta,
            closure,
            consumer_obligation,
            missing_inputs,
            value_candidates,
            value_provenance,
            behavior_contracts,
        )
        premise_ids = self._collect_premise_ids(
            delta,
            closure,
            consumer_obligation,
            missing_inputs,
            value_candidates,
            value_provenance,
            behavior_contracts,
            context,
        )
        unsupported_ids = tuple(
            f"{item.kind.value}:{item.subject_id}" for item in unsupported
        )

        obligations: dict[ObligationKind, ChangePropagationObligation] = {}
        value_claims: list[ValueMappingClaim] = []
        behavior_claims: list[BehaviorRefinementClaim] = []

        # Always lower closure coverage and consumer compatibility separately.
        obligations[ObligationKind.CLOSURE_COVERAGE] = self._make_obligation(
            ObligationKind.CLOSURE_COVERAGE,
            roots,
            consumer_obligation,
            context,
            source_ids,
            premise_ids,
            resolved_graph_id,
            unsupported_ids,
            counterexample_targets=(
                f"counterexample:missing-consumer:{consumer_obligation.consumer_id}",
                f"counterexample:open-frontier:{closure.delta_id}",
            ),
        )
        obligations[ObligationKind.CONSUMER_COMPATIBILITY] = self._make_obligation(
            ObligationKind.CONSUMER_COMPATIBILITY,
            roots,
            consumer_obligation,
            context,
            source_ids,
            premise_ids,
            resolved_graph_id,
            unsupported_ids,
            counterexample_targets=(
                f"counterexample:consumer-incompatible:{consumer_obligation.consumer_id}",
            ),
        )

        needs_value_mapping = consumer_obligation.disposition in {
            ConsumerDisposition.MIGRATE,
            ConsumerDisposition.ADAPTER,
            ConsumerDisposition.UPSTREAM,
        } and (
            bool(consumer_obligation.missing_input_ids)
            or bool(missing_inputs)
            or bool(value_candidates)
        )

        if needs_value_mapping:
            if not missing_inputs:
                raise ChangePropagationObligationError(
                    "migrate/adapter/upstream obligations with missing inputs require "
                    "MissingInputRequirement records"
                )
            mapping_claim = self._build_value_mapping_claim(
                consumer_obligation,
                missing_inputs,
                value_candidates,
                premise_ids,
                source_ids,
                context,
            )
            if mapping_claim is not None:
                value_claims.append(mapping_claim)
            for kind in _VALUE_MAPPING_FACETS:
                obligations[kind] = self._make_obligation(
                    kind,
                    roots,
                    consumer_obligation,
                    context,
                    source_ids,
                    premise_ids,
                    resolved_graph_id,
                    unsupported_ids,
                    value_mapping_claim_id=(
                        mapping_claim.claim_id if mapping_claim is not None else ""
                    ),
                    counterexample_targets=(
                        f"counterexample:{kind.value}:{consumer_obligation.consumer_id}",
                    ),
                )

        if consumer_obligation.behavior_contract_ids or behavior_contracts:
            if not behavior_contracts:
                raise ChangePropagationObligationError(
                    "behavior_contract_ids require RequiredBehaviorContract records"
                )
            for behavior in behavior_contracts:
                claim = self._build_behavior_refinement_claim(
                    behavior, premise_ids, source_ids, context
                )
                behavior_claims.append(claim)
                for kind in (
                    ObligationKind.BEHAVIOR_INVARIANTS,
                    ObligationKind.STATE_TRANSITIONS,
                    ObligationKind.SERIALIZATION_MIGRATION,
                    ObligationKind.PLACEMENT,
                ):
                    obligations[kind] = self._make_obligation(
                        kind,
                        roots,
                        consumer_obligation,
                        context,
                        source_ids,
                        (*premise_ids, behavior.behavior_id, claim.claim_id),
                        resolved_graph_id,
                        unsupported_ids,
                        behavior_refinement_claim_id=claim.claim_id,
                        counterexample_targets=(
                            f"counterexample:{kind.value}:{behavior.behavior_id}",
                        ),
                    )

        # Compatible consumers still need explicit non-migration compatibility,
        # but must not invent mapping or placement claims.
        if consumer_obligation.disposition in {
            ConsumerDisposition.COMPATIBLE,
            ConsumerDisposition.EXCLUDED,
        }:
            forbidden = set(_VALUE_MAPPING_FACETS) | {
                ObligationKind.BEHAVIOR_INVARIANTS,
                ObligationKind.STATE_TRANSITIONS,
                ObligationKind.SERIALIZATION_MIGRATION,
                ObligationKind.PLACEMENT,
            }
            for kind in list(obligations):
                if kind in forbidden:
                    del obligations[kind]
            value_claims.clear()
            behavior_claims.clear()

        if consumer_obligation.disposition is ConsumerDisposition.FRONTIER:
            raise ChangePropagationObligationError(
                "frontier consumer obligations cannot be compiled as closed proof obligations"
            )

        if consumer_obligation.disposition is ConsumerDisposition.ABSTAIN:
            # Abstention is representable only as closure + consumer compatibility
            # with explicit unsupported surfaces; no automated mapping claims.
            keep = {
                ObligationKind.CLOSURE_COVERAGE,
                ObligationKind.CONSUMER_COMPATIBILITY,
            }
            for kind in list(obligations):
                if kind not in keep:
                    del obligations[kind]
            value_claims.clear()
            behavior_claims.clear()

        if not obligations:
            raise ChangePropagationObligationError(
                "no obligations could be lowered from the provided evidence"
            )

        return ChangePropagationObligationCompilation(
            roots=roots,
            delta_id=consumer_obligation.delta_id,
            consumer_id=consumer_obligation.consumer_id,
            migration_obligation_id=consumer_obligation.obligation_id,
            obligations=tuple(obligations.values()),
            value_mapping_claims=tuple(value_claims),
            behavior_refinement_claims=tuple(behavior_claims),
            unsupported_semantics=tuple(unsupported),
        )

    compile_consumer = compile

    def _validate_bindings(
        self,
        delta: ProgramContractDelta,
        closure: ImpactClosureReceipt,
        consumer_obligation: ConsumerMigrationObligation,
        context: ObligationContext,
        missing_inputs: Sequence[MissingInputRequirement],
        value_candidates: Sequence[ValueCandidate],
        value_provenance: ValueProvenanceGraph | None,
        behavior_contracts: Sequence[RequiredBehaviorContract],
    ) -> PropagationAuthorityRoots:
        if not isinstance(delta, ProgramContractDelta):
            raise ChangePropagationObligationError("delta must be ProgramContractDelta")
        if not isinstance(closure, ImpactClosureReceipt):
            raise ChangePropagationObligationError(
                "closure must be ImpactClosureReceipt"
            )
        if not isinstance(consumer_obligation, ConsumerMigrationObligation):
            raise ChangePropagationObligationError(
                "consumer_obligation must be ConsumerMigrationObligation"
            )
        if not isinstance(context, ObligationContext):
            raise ChangePropagationObligationError(
                "context must be ObligationContext"
            )

        roots = delta.roots
        for label, other in (
            ("closure", closure.roots),
            ("consumer_obligation", consumer_obligation.roots),
        ):
            if not _roots_match(roots, other):
                raise ChangePropagationObligationError(
                    f"{label} roots must bind the same authority snapshot as the delta"
                )

        # Consumer and closure share a symbolic or content-addressed delta id.
        # ProgramContractDelta is bound by matching roots and clause coverage; its
        # content_id is always a premise and need not equal the symbolic delta_id.
        if closure.delta_id != consumer_obligation.delta_id:
            raise ChangePropagationObligationError(
                "closure and consumer obligation must bind the same delta_id"
            )

        consumer_ids = {item.consumer_id for item in closure.consumers}
        if consumer_obligation.consumer_id not in consumer_ids:
            raise ChangePropagationObligationError(
                "consumer obligation is not present in the impact closure"
            )

        if closure.completeness is ImpactCompleteness.ABSTAINED:
            raise IncompleteImpactClosureError(
                "abstained impact closures cannot compile closed obligations"
            )
        if closure.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            if not context.allow_partial_frontier:
                raise IncompleteImpactClosureError(
                    "partial impact closures cannot be compiled as closed obligations "
                    "without allow_partial_frontier and explicit unsupported frontier facts"
                )
            if not (closure.frontier_node_ids or closure.frontier_edge_ids):
                raise IncompleteImpactClosureError(
                    "partial impact closures require an explicit frontier"
                )
        if closure.completeness is ImpactCompleteness.COMPLETE:
            if closure.frontier_node_ids or closure.frontier_edge_ids:
                raise ChangePropagationObligationError(
                    "complete impact closures cannot retain an open frontier"
                )

        clause_ids = {item.clause_id for item in delta.clauses}
        missing_clauses = set(consumer_obligation.clause_ids) - clause_ids
        if missing_clauses:
            raise ChangePropagationObligationError(
                f"consumer clause_ids missing from delta: {sorted(missing_clauses)}"
            )

        for item in missing_inputs:
            if not isinstance(item, MissingInputRequirement):
                raise ChangePropagationObligationError(
                    "missing_inputs must be MissingInputRequirement values"
                )
            if not _roots_match(roots, item.roots):
                raise ChangePropagationObligationError(
                    "missing input roots must match the delta roots"
                )
            if item.obligation_id != consumer_obligation.obligation_id:
                raise ChangePropagationObligationError(
                    "missing input must bind the consumer migration obligation_id"
                )
            if item.requirement_id not in consumer_obligation.missing_input_ids:
                raise ChangePropagationObligationError(
                    "missing input requirement_id must be listed on the consumer obligation"
                )

        for item in value_candidates:
            if not isinstance(item, ValueCandidate):
                raise ChangePropagationObligationError(
                    "value_candidates must be ValueCandidate values"
                )
            if not _roots_match(roots, item.roots):
                raise ChangePropagationObligationError(
                    "value candidate roots must match the delta roots"
                )
            if item.kind in _NOMINATION_ONLY_KINDS and item.semantic_authority:
                raise UnauthorizedAssumptionError(
                    "nominated value candidates cannot become axioms or semantic authority"
                )
            if item.disposition is ValueCandidateDisposition.NOMINATED and item.semantic_authority:
                raise UnauthorizedAssumptionError(
                    "nominated value candidates cannot claim semantic authority"
                )

        if value_provenance is not None:
            if not isinstance(value_provenance, ValueProvenanceGraph):
                raise ChangePropagationObligationError(
                    "value_provenance must be ValueProvenanceGraph"
                )
            # Provenance graphs bind ProgramGraphRoots; require tree identity
            # against the candidate tree so base/candidate drift fails closed.
            if value_provenance.roots.tree_id not in {
                roots.candidate_tree_id,
                roots.base_tree_id,
            }:
                raise ChangePropagationObligationError(
                    "value provenance tree_id must match base or candidate tree"
                )
            if (
                value_provenance.roots.toolchain_id
                and value_provenance.roots.toolchain_id != roots.toolchain_id
            ):
                raise ChangePropagationObligationError(
                    "value provenance toolchain_id must match propagation roots"
                )
            if value_provenance.unknown_frontier and not context.unsupported_semantics:
                # Unknown dynamic/native facts must be carried explicitly; the
                # compiler will also project frontier nodes when allow_partial is set.
                pass

        for item in behavior_contracts:
            if not isinstance(item, RequiredBehaviorContract):
                raise ChangePropagationObligationError(
                    "behavior_contracts must be RequiredBehaviorContract values"
                )
            if not _roots_match(roots, item.roots):
                raise ChangePropagationObligationError(
                    "behavior contract roots must match the delta roots"
                )
            if item.behavior_id not in consumer_obligation.behavior_contract_ids:
                raise ChangePropagationObligationError(
                    "behavior contract must be listed on the consumer obligation"
                )
            if (
                item.evidence_precedence
                is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            ):
                raise UnauthorizedAssumptionError(
                    "implementation hypotheses cannot compile as closed behavior obligations"
                )

        # Unsupported dynamic/native semantics on the delta must stay explicit.
        for clause in delta.clauses:
            if clause.disposition is DeltaDisposition.UNSUPPORTED:
                # Caller must have named the unsupported surface; do not invent it.
                if not any(
                    item.subject_id in {clause.clause_id, clause.subject_symbol_id}
                    for item in context.unsupported_semantics
                ):
                    raise UnsupportedObligationError(
                        "unsupported delta clauses require explicit UnsupportedSemantic bindings"
                    )

        return roots

    def _closure_frontier_unsupported(
        self,
        closure: ImpactClosureReceipt,
        context: ObligationContext,
    ) -> list[UnsupportedSemantic]:
        if closure.completeness is not ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            return []
        items: list[UnsupportedSemantic] = []
        for node_id in closure.frontier_node_ids:
            items.append(
                UnsupportedSemantic(
                    UnsupportedSemanticKind.FRONTIER_NODE,
                    node_id,
                    f"frontier-node:{node_id}",
                )
            )
        for edge_id in closure.frontier_edge_ids:
            items.append(
                UnsupportedSemantic(
                    UnsupportedSemanticKind.FRONTIER_EDGE,
                    edge_id,
                    f"frontier-edge:{edge_id}",
                )
            )
        # Deduplicate against already-declared context entries.
        existing = {
            (item.kind, item.subject_id) for item in context.unsupported_semantics
        }
        return [
            item
            for item in items
            if (item.kind, item.subject_id) not in existing
        ]

    def _resolve_graph_id(
        self,
        roots: PropagationAuthorityRoots,
        value_provenance: ValueProvenanceGraph | None,
        graph_id: str,
        consumer_obligation: ConsumerMigrationObligation,
    ) -> str:
        if value_provenance is not None:
            return value_provenance.graph_id
        if graph_id:
            return _text(graph_id, "graph_id")
        # Fall back to the shared program-graph root identity from authority roots.
        if roots.graph_id:
            return roots.graph_id
        return _text(consumer_obligation.node.node_id, "graph_id")

    def _collect_source_ids(
        self,
        delta: ProgramContractDelta,
        closure: ImpactClosureReceipt,
        consumer_obligation: ConsumerMigrationObligation,
        missing_inputs: Sequence[MissingInputRequirement],
        value_candidates: Sequence[ValueCandidate],
        value_provenance: ValueProvenanceGraph | None,
        behavior_contracts: Sequence[RequiredBehaviorContract],
    ) -> tuple[str, ...]:
        ids: set[str] = set(delta.evidence_refs)
        ids.update(delta.proof_refs)
        ids.update(closure.evidence_refs)
        ids.update(consumer_obligation.proof_refs)
        ids.update(consumer_obligation.invalidation_refs)
        ids.add(delta.content_id)
        ids.add(closure.content_id)
        ids.add(consumer_obligation.content_id)
        ids.add(consumer_obligation.node.artifact_id)
        for item in missing_inputs:
            ids.add(item.content_id)
            ids.update(item.proof_refs)
            ids.add(item.type_ref)
            ids.add(item.information_content_ref)
        for item in value_candidates:
            ids.add(item.content_id)
            ids.add(item.expression_ref)
            ids.add(item.source_node.artifact_id)
            ids.update(item.proof_refs)
        if value_provenance is not None:
            ids.add(value_provenance.graph_id)
            ids.add(value_provenance.roots_id)
        for item in behavior_contracts:
            ids.add(item.content_id)
            ids.update(item.proof_refs)
            ids.update(item.invariant_refs)
            ids.update(item.state_transition_refs)
        return _ids(tuple(ids), "source_ids")

    def _collect_premise_ids(
        self,
        delta: ProgramContractDelta,
        closure: ImpactClosureReceipt,
        consumer_obligation: ConsumerMigrationObligation,
        missing_inputs: Sequence[MissingInputRequirement],
        value_candidates: Sequence[ValueCandidate],
        value_provenance: ValueProvenanceGraph | None,
        behavior_contracts: Sequence[RequiredBehaviorContract],
        context: ObligationContext,
    ) -> tuple[str, ...]:
        ids: set[str] = {
            delta.content_id,
            closure.content_id,
            consumer_obligation.content_id,
            consumer_obligation.obligation_id,
            *consumer_obligation.clause_ids,
            *context.assumption_ids,
        }
        for clause in delta.clauses:
            ids.add(clause.clause_id)
            ids.add(clause.kind.value)
            ids.add(clause.disposition.value)
        for item in missing_inputs:
            ids.add(item.requirement_id)
            ids.add(item.parameter_name)
            ids.add(item.type_ref)
            ids.add(item.nullability)
            ids.add(item.information_content_ref)
            ids.update(item.construction_precondition_refs)
            ids.update(item.result_postcondition_refs)
            ids.update(item.allowed_error_refs)
            ids.update(item.effect_refs)
            ids.update(item.capability_refs)
            ids.update(item.authorization_refs)
            ids.update(item.resource_refs)
            ids.update(item.ownership_refs)
        for item in value_candidates:
            ids.add(item.candidate_id)
            ids.add(item.expression_ref)
            ids.add(item.type_ref)
            ids.add(item.kind.value)
            ids.add(item.disposition.value)
        if value_provenance is not None:
            ids.add(value_provenance.graph_id)
            for unknown in value_provenance.unknown_frontier[:MAX_UNSUPPORTED]:
                # Unknown facts are premises about incompleteness, not axioms of success.
                reason = unknown.reason.value if hasattr(unknown.reason, "value") else str(unknown.reason)
                ids.add(f"unknown:{unknown.fact_id}:{reason}")
        for item in behavior_contracts:
            ids.add(item.behavior_id)
            ids.add(item.kind.value)
            ids.add(item.evidence_precedence.value)
            ids.update(item.field_refs)
            ids.update(item.constructor_refs)
            ids.update(item.method_refs)
            ids.update(item.invariant_refs)
            ids.update(item.state_transition_refs)
        return _ids(tuple(ids), "premise_ids")

    def _build_value_mapping_claim(
        self,
        consumer_obligation: ConsumerMigrationObligation,
        missing_inputs: Sequence[MissingInputRequirement],
        value_candidates: Sequence[ValueCandidate],
        premise_ids: Sequence[str],
        source_ids: Sequence[str],
        context: ObligationContext,
    ) -> ValueMappingClaim | None:
        requirement = missing_inputs[0]
        candidate: ValueCandidate | None = None
        for item in value_candidates:
            if item.requirement_id == requirement.requirement_id:
                if item.kind in _NOMINATION_ONLY_KINDS:
                    # Nominations may appear as premises but cannot form a mapping claim
                    # that pretends to be an axiom; only non-nomination kinds map.
                    continue
                candidate = item
                break
        if candidate is None:
            # A missing-input requirement without a non-nomination candidate still
            # lowers facet obligations; the mapping claim is withheld until synthesis
            # (RPR-036) selects a unique proved source.
            return None
        if candidate.disposition is ValueCandidateDisposition.REFUTED:
            raise ChangePropagationObligationError(
                "refuted value candidates cannot compile value-mapping claims"
            )
        return ValueMappingClaim(
            claim_id_seed=(
                f"value-map:{requirement.requirement_id}:{candidate.candidate_id}"
            ),
            requirement_id=requirement.requirement_id,
            candidate_id=candidate.candidate_id,
            consumer_id=consumer_obligation.consumer_id,
            parameter_name=requirement.parameter_name,
            expression_ref=candidate.expression_ref,
            type_ref=candidate.type_ref or requirement.type_ref,
            premise_ids=tuple(premise_ids),
            source_ids=tuple(source_ids),
            assumption_ids=context.assumption_ids,
            facet_predicates=_VALUE_MAPPING_FACET_PREDICATES,
            counterexample_targets=(
                f"counterexample:value-mapping:{requirement.requirement_id}",
                f"counterexample:wrong-source:{candidate.candidate_id}",
            ),
        )

    def _build_behavior_refinement_claim(
        self,
        behavior: RequiredBehaviorContract,
        premise_ids: Sequence[str],
        source_ids: Sequence[str],
        context: ObligationContext,
    ) -> BehaviorRefinementClaim:
        structural = (
            *behavior.field_refs,
            *behavior.constructor_refs,
            *behavior.method_refs,
            *behavior.invariant_refs,
            *behavior.state_transition_refs,
        )
        return BehaviorRefinementClaim(
            claim_id_seed=f"behavior-refine:{behavior.behavior_id}",
            behavior_id=behavior.behavior_id,
            subject_symbol_id=behavior.subject_symbol_id,
            evidence_precedence=behavior.evidence_precedence.value,
            premise_ids=tuple(premise_ids),
            source_ids=tuple(source_ids),
            assumption_ids=context.assumption_ids,
            structural_clause_ids=structural,
            placement_decision_ref=behavior.placement_decision_ref,
            counterexample_targets=(
                f"counterexample:behavior:{behavior.behavior_id}",
                f"counterexample:placement:{behavior.behavior_id}",
            ),
        )

    def _make_obligation(
        self,
        kind: ObligationKind,
        roots: PropagationAuthorityRoots,
        consumer_obligation: ConsumerMigrationObligation,
        context: ObligationContext,
        source_ids: Sequence[str],
        premise_ids: Sequence[str],
        graph_id: str,
        unsupported_ids: Sequence[str],
        *,
        value_mapping_claim_id: str = "",
        behavior_refinement_claim_id: str = "",
        counterexample_targets: Sequence[str] = (),
    ) -> ChangePropagationObligation:
        all_sources = _ids(source_ids, "obligation sources")
        claim = IRClaim(
            predicate=kind.value,
            subject_id=consumer_obligation.consumer_id,
            premise_ids=_ids(premise_ids, "premise_ids"),
            source_ids=all_sources,
            assumption_ids=context.assumption_ids,
            repository_id=roots.repository_id,
            tree_id=roots.candidate_tree_id,
            graph_id=graph_id,
            translator_id=roots.translator_id,
            toolchain_id=roots.toolchain_id,
            policy_id=roots.policy_id,
            capability_id=context.capability.capability_id,
            capability_revision=context.capability.capability_revision,
            counterexample_targets=tuple(counterexample_targets),
            unsupported_semantic_ids=tuple(unsupported_ids),
        )
        scopes = tuple(
            sorted(
                {
                    consumer_obligation.node.artifact_id,
                    consumer_obligation.node.symbol_id,
                    consumer_obligation.node.path.replace("/", "."),
                }
            )
        )
        envelope = CodeProofObligation(
            repository_id=roots.repository_id,
            repository_tree_id=roots.candidate_tree_id,
            ast_scope_ids=scopes,
            statement=f"{kind.value}:{consumer_obligation.consumer_id}",
            premise_ids=claim.premise_ids,
            template_id=f"change-propagation/{kind.value}",
            template_version="1",
            template_semantic_hash=claim.content_id,
            invariant_class="change_propagation",
            task_id="RPR-035",
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            metadata={
                "claim_id": claim.content_id,
                "translator_id": roots.translator_id,
                "toolchain_id": roots.toolchain_id,
                "policy_id": roots.policy_id,
                "graph_id": graph_id,
                "capability_id": context.capability.capability_id,
                "capability_revision": context.capability.capability_revision,
                "supported_semantics": list(context.capability.supported_semantics),
                "counterexample_policy_ref": context.counterexample_policy_ref,
                "compiler_id": CHANGE_PROPAGATION_OBLIGATION_COMPILER_ID,
                "interface": CHANGE_PROPAGATION_OBLIGATIONS_INTERFACE,
            },
        )
        return ChangePropagationObligation(
            kind=kind,
            consumer_id=consumer_obligation.consumer_id,
            delta_id=consumer_obligation.delta_id
            if consumer_obligation.delta_id
            else claim.subject_id,
            claim=claim,
            code_obligation=envelope,
            source_ids=all_sources,
            value_mapping_claim_id=value_mapping_claim_id,
            behavior_refinement_claim_id=behavior_refinement_claim_id,
        )


__all__ = [
    "AssumptionBinding",
    "BEHAVIOR_REFINEMENT_CLAIM_SCHEMA",
    "BehaviorRefinementClaim",
    "CHANGE_PROPAGATION_IR_CLAIM_SCHEMA",
    "CHANGE_PROPAGATION_OBLIGATIONS_INTERFACE",
    "CHANGE_PROPAGATION_OBLIGATION_COMPILER_ID",
    "CHANGE_PROPAGATION_OBLIGATION_SCHEMA",
    "ChangePropagationObligation",
    "ChangePropagationObligationCompilation",
    "ChangePropagationObligationCompiler",
    "ChangePropagationObligationError",
    "IRClaim",
    "IncompleteImpactClosureError",
    "LogicCapabilityBinding",
    "MAX_CLAIMS",
    "MAX_OBLIGATIONS",
    "MAX_UNSUPPORTED",
    "ObligationContext",
    "ObligationKind",
    "UnauthorizedAssumptionError",
    "UnsupportedObligationError",
    "UnsupportedSemantic",
    "UnsupportedSemanticKind",
    "VALUE_MAPPING_CLAIM_SCHEMA",
    "ValueMappingClaim",
]
