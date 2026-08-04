"""Bridge admitted program-logic predictions into existing RPR synthesis (LPR-015).

Composition adapter only — never forks or weakens
:class:`RequiredBehaviorContract`, :class:`MissingInputSynthesisReceipt`,
source precedence, or proof obligations.

Responsibilities
----------------
* Map candidate-specific reconstructed substitution / equivalence / placement
  predictions into the exact :class:`CandidateProofBundle` evidence consumed
  by :class:`ContractRepairReranker` / :class:`ContractRepairProver`.
* Compose propagation consequences with the existing
  :class:`RequiredBehaviorSynthesizer`,
  :class:`ChangePropagationObligationCompiler` inputs, and
  :class:`MissingInputSynthesizer` value/behavior surfaces.
* Keep proof status orthogonal to source authority.
* Inherit the *weakest* effective :class:`BehaviorEvidencePrecedence` of a
  consequence's independent premises — never invent a new closed-enum rank.
* Never overwrite an explicit conflict, higher-precedence source, unsupported
  memory / lifetime / native / concurrency facet, or consumer-specific
  requirement.
* One admitted consequence maps to exact existing clause / value / placement
  references; stale or ambiguous proof remains a nomination.
* Outputs are canonical and accepted unchanged by existing repair-target and
  propagation-plan admission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.contract_repair_prover import (
    CONTRACT_REPAIR_PROVER_INTERFACE,
    CandidateProofBundle,
    CandidateProofResult,
    ContractRepairProofDisposition,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    content_identity,
)
from .change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    GraphNodeRef,
    GraphProvenance,
    MissingInputRequirement,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
    ValueCandidate,
    ValueCandidateDisposition,
    ValueCandidateKind,
)
from .program_logic_prediction_contracts import (
    LogicPredictionReceipt,
    PredictionDisposition,
    ProgramLogicAuthorityRoots,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
)
from .required_behavior_synthesis import (
    PRECEDENCE_RANK,
    BehaviorClauseFamily,
    BehaviorEvidenceAtom,
    RequiredBehaviorSynthesisReceipt,
    RequiredBehaviorSynthesizer,
    SynthesisDisposition,
    coerce_clause_family,
    coerce_precedence,
    is_authoritative,
    precedence_rank,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE: Final[str] = (
    "TacticianGuidedBehaviorSynthesizer@1"
)
CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE: Final[str] = (
    "ContractRepairPredictionBridge@1"
)
PREDICTION_EVIDENCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/prediction-evidence-binding@1"
)
TACTICIAN_BEHAVIOR_SYNTHESIS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-behavior-synthesis-receipt@1"
)
PRODUCER_ID: Final[str] = "tactician-guided-behavior-synthesis@1"
CONTRACT_VERSION: Final[int] = 1

MAX_BINDINGS: Final[int] = 256
MAX_PREDICTIONS: Final[int] = 256
MAX_PREMISE_PRECEDENCES: Final[int] = 512
MAX_REF_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 1_024
MAX_RECORD_BYTES: Final[int] = 262_144

# Facets that cannot be silently promoted from a prediction alone.
_PROTECTED_FACET_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "memory",
        "lifetime",
        "native",
        "concurrency",
        "ownership",
        "mutation",
        "disposal",
    }
)
_PROTECTED_CLAUSE_FAMILIES: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.OWNERSHIP,
        BehaviorClauseFamily.LIFETIME,
        BehaviorClauseFamily.MUTATION,
        BehaviorClauseFamily.CONCURRENCY,
        BehaviorClauseFamily.DISPOSAL,
    }
)

# Map independent source routes onto the *existing* BehaviorEvidencePrecedence
# lattice.  No new rank is introduced.
_ROUTE_TO_PRECEDENCE: Final[Mapping[SourceRouteKind, BehaviorEvidencePrecedence]] = {
    SourceRouteKind.REVIEWED_CONTRACT: BehaviorEvidencePrecedence.REVIEWED_IDL,
    SourceRouteKind.NORMATIVE_SPEC: BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    SourceRouteKind.REVIEWED_TEST: BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    SourceRouteKind.LOCAL_STATIC: BehaviorEvidencePrecedence.DATA_INVARIANT,
    SourceRouteKind.DATAFLOW: BehaviorEvidencePrecedence.DATA_INVARIANT,
    SourceRouteKind.HISTORY: BehaviorEvidencePrecedence.HISTORY,
    SourceRouteKind.GRAPH: BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP,
    SourceRouteKind.RUNTIME_WITNESS: BehaviorEvidencePrecedence.HISTORY,
    # Nominating routes never outrank independent sources.
    SourceRouteKind.VECTOR: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    SourceRouteKind.KNOWLEDGE_GRAPH: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    SourceRouteKind.TACTICIAN: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    SourceRouteKind.LLM: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    SourceRouteKind.SOLVER: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
}

_SOURCE_AUTHORITY_FLOOR: Final[
    Mapping[SourceAuthorityClass, BehaviorEvidencePrecedence]
] = {
    SourceAuthorityClass.AUTHORITATIVE: BehaviorEvidencePrecedence.REVIEWED_IDL,
    SourceAuthorityClass.CONFORMANCE: BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    SourceAuthorityClass.NOMINATING: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    SourceAuthorityClass.DIAGNOSTIC: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    SourceAuthorityClass.NONE: BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
}


# ---------------------------------------------------------------------------
# Errors / closed enumerations
# ---------------------------------------------------------------------------


class TacticianGuidedBehaviorSynthesisError(ValueError):
    """Malformed or unsafe bridge input."""


class TacticianGuidedBehaviorSynthesisAuthorityError(
    TacticianGuidedBehaviorSynthesisError
):
    """Root, identity, precedence, or proof-boundary failure."""


class ConsequenceKind(str, Enum):
    """Closed consequence surface projected from an admitted prediction."""

    CLAUSE = "clause"
    VALUE = "value"
    CONSTRUCTION = "construction"
    PLACEMENT = "placement"
    SUBSTITUTION = "substitution"
    EQUIVALENCE = "equivalence"
    BEHAVIOR = "behavior"


class BindingDisposition(str, Enum):
    """Whether a prediction consequence is admitted into RPR synthesis."""

    ADMITTED = "admitted"
    NOMINATION = "nomination"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"


class BridgeDisposition(str, Enum):
    """Outcome of one guided synthesis / projection attempt."""

    COMPOSED = "composed"
    PROOF_PROJECTED = "proof_projected"
    NOMINATION_ONLY = "nomination_only"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


class ProjectionKind(str, Enum):
    """Closed kinds of candidate-specific proof projection targets."""

    SUBSTITUTION = "substitution"
    EQUIVALENCE = "equivalence"
    PLACEMENT = "placement"
    VALUE = "value"
    BEHAVIOR = "behavior"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TacticianGuidedBehaviorSynthesisError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise TacticianGuidedBehaviorSynthesisError(f"{field_name} must not be empty")
    if len(result.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TacticianGuidedBehaviorSynthesisError(
            f"{field_name} exceeds its byte bound"
        )
    return result


def _identifier(value: Any, field_name: str) -> str:
    result = _text(value, field_name, required=True)
    if any(char.isspace() for char in result):
        raise TacticianGuidedBehaviorSynthesisError(
            f"{field_name} must be a compact identifier"
        )
    if len(result.encode("utf-8")) > MAX_REF_BYTES:
        raise TacticianGuidedBehaviorSynthesisError(
            f"{field_name} exceeds its byte bound"
        )
    return result


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TacticianGuidedBehaviorSynthesisError(f"{field_name} must be a boolean")
    return value


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    bound: int = MAX_BINDINGS,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TacticianGuidedBehaviorSynthesisError(
            f"{field_name} must be a sequence of identifiers"
        )
    else:
        raw = values
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = _identifier(item, field_name)
        if token not in seen:
            seen.add(token)
            result.append(token)
    ordered = tuple(sorted(result))
    if required and not ordered:
        raise TacticianGuidedBehaviorSynthesisError(f"{field_name} must not be empty")
    if len(ordered) > bound:
        raise TacticianGuidedBehaviorSynthesisError(
            f"{field_name} exceeds its item bound"
        )
    return ordered


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        try:
            return enum(value.strip())
        except ValueError as exc:
            raise TacticianGuidedBehaviorSynthesisError(
                f"{field_name} has unsupported value {value!r}"
            ) from exc
    raise TacticianGuidedBehaviorSynthesisError(
        f"{field_name} must be a {enum.__name__} or string"
    )


def _propagation_roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return PropagationAuthorityRoots.from_dict(value)
    raise TacticianGuidedBehaviorSynthesisError(
        "roots must be PropagationAuthorityRoots"
    )


def _prediction_receipt(value: Any) -> LogicPredictionReceipt:
    if isinstance(value, LogicPredictionReceipt):
        return value
    if isinstance(value, Mapping):
        return LogicPredictionReceipt.from_dict(value)
    raise TacticianGuidedBehaviorSynthesisError(
        "prediction must be LogicPredictionReceipt"
    )


def _proof_receipt(value: Any) -> ProofReceipt:
    if isinstance(value, ProofReceipt):
        return value
    if isinstance(value, Mapping):
        return ProofReceipt.from_dict(value)
    raise TacticianGuidedBehaviorSynthesisError("proof receipt must be ProofReceipt")


def weakest_precedence(
    precedences: Sequence[BehaviorEvidencePrecedence | str],
) -> BehaviorEvidencePrecedence:
    """Return the weakest (highest rank number) of independent premises.

    Uses the existing :data:`PRECEDENCE_RANK` lattice only — never invents a
    new closed-enum rank.
    """

    if not precedences:
        return BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    ranks = [coerce_precedence(item) for item in precedences]
    return max(ranks, key=precedence_rank)


def route_to_precedence(route: SourceRouteKind | str) -> BehaviorEvidencePrecedence:
    """Project a source route onto existing behavior-evidence precedence."""

    kind = _enum(route, SourceRouteKind, "source_route")
    return _ROUTE_TO_PRECEDENCE.get(
        kind, BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS  # type: ignore[arg-type]
    )


def source_authority_floor(
    authority: SourceAuthorityClass | str,
) -> BehaviorEvidencePrecedence:
    """Floor precedence implied by independent source authority alone.

    Proof status is deliberately *not* consulted.
    """

    auth = _enum(authority, SourceAuthorityClass, "source_authority")
    return _SOURCE_AUTHORITY_FLOOR.get(
        auth, BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS  # type: ignore[arg-type]
    )


def effective_consequence_precedence(
    *,
    source_authority: SourceAuthorityClass | str,
    premise_precedences: Sequence[BehaviorEvidencePrecedence | str] = (),
    evidence_routes: Sequence[SourceRouteKind | str] = (),
    proof_status: ProofStatus | str | None = None,
) -> BehaviorEvidencePrecedence:
    """Inherit the weakest independent premise precedence.

    ``proof_status`` is accepted for orthogonality documentation only and never
    strengthens precedence.
    """

    del proof_status  # orthogonal: never promotes rank
    candidates: list[BehaviorEvidencePrecedence] = [
        source_authority_floor(source_authority)
    ]
    for item in premise_precedences:
        candidates.append(coerce_precedence(item))
    for route in evidence_routes:
        candidates.append(route_to_precedence(route))
    return weakest_precedence(candidates)


def _protected_token_hit(tokens: Sequence[str]) -> bool:
    for token in tokens:
        lowered = token.strip().lower()
        if not lowered:
            continue
        for protected in _PROTECTED_FACET_TOKENS:
            if protected in lowered:
                return True
    return False


def _consequence_kind_for(receipt: LogicPredictionReceipt) -> ConsequenceKind:
    if receipt.derived_placement_ref:
        return ConsequenceKind.PLACEMENT
    if receipt.derived_value_ref:
        return ConsequenceKind.VALUE
    clause = receipt.derived_clause_ref
    if clause.startswith("substitution:") or ":substitution:" in clause:
        return ConsequenceKind.SUBSTITUTION
    if clause.startswith("equivalence:") or ":equivalence:" in clause:
        return ConsequenceKind.EQUIVALENCE
    if clause.startswith("construction:") or ":construction:" in clause:
        return ConsequenceKind.CONSTRUCTION
    if clause.startswith("behavior:") or ":behavior:" in clause:
        return ConsequenceKind.BEHAVIOR
    if clause:
        return ConsequenceKind.CLAUSE
    return ConsequenceKind.CLAUSE


def _projection_kind_for(kind: ConsequenceKind) -> ProjectionKind:
    mapping = {
        ConsequenceKind.SUBSTITUTION: ProjectionKind.SUBSTITUTION,
        ConsequenceKind.EQUIVALENCE: ProjectionKind.EQUIVALENCE,
        ConsequenceKind.PLACEMENT: ProjectionKind.PLACEMENT,
        ConsequenceKind.VALUE: ProjectionKind.VALUE,
        ConsequenceKind.BEHAVIOR: ProjectionKind.BEHAVIOR,
        ConsequenceKind.CONSTRUCTION: ProjectionKind.SUBSTITUTION,
        ConsequenceKind.CLAUSE: ProjectionKind.GENERIC,
    }
    return mapping[kind]


def _family_for_consequence(
    kind: ConsequenceKind,
    *,
    family: BehaviorClauseFamily | str | None = None,
) -> BehaviorClauseFamily:
    if family is not None:
        return coerce_clause_family(family)
    defaults = {
        ConsequenceKind.VALUE: BehaviorClauseFamily.DEFAULTS,
        ConsequenceKind.CONSTRUCTION: BehaviorClauseFamily.CONSTRUCTORS,
        ConsequenceKind.PLACEMENT: BehaviorClauseFamily.METHODS,
        ConsequenceKind.SUBSTITUTION: BehaviorClauseFamily.METHODS,
        ConsequenceKind.EQUIVALENCE: BehaviorClauseFamily.INVARIANTS,
        ConsequenceKind.BEHAVIOR: BehaviorClauseFamily.METHODS,
        ConsequenceKind.CLAUSE: BehaviorClauseFamily.INVARIANTS,
    }
    return defaults[kind]


def _is_reconstructed(receipt: LogicPredictionReceipt) -> bool:
    return (
        receipt.disposition is PredictionDisposition.PROVED
        and receipt.proof_status is ProofStatus.KERNEL_VERIFIED
        and bool(receipt.reconstruction_id)
        and bool(receipt.kernel_receipt_id)
        and receipt.source_authority
        in {SourceAuthorityClass.AUTHORITATIVE, SourceAuthorityClass.CONFORMANCE}
    )


def _is_nomination_quality(receipt: LogicPredictionReceipt) -> bool:
    if receipt.disposition in {
        PredictionDisposition.STALE,
        PredictionDisposition.INCONCLUSIVE,
        PredictionDisposition.ABSTAINED,
        PredictionDisposition.UNSUPPORTED,
        PredictionDisposition.ERROR,
    }:
        return True
    if receipt.proof_status in {
        ProofStatus.UNPROVED,
        ProofStatus.CANDIDATE,
        ProofStatus.SOLVER_CHECKED,
        ProofStatus.INCONCLUSIVE,
        ProofStatus.UNSUPPORTED,
        ProofStatus.STALE,
        ProofStatus.ERROR,
    }:
        return True
    if receipt.source_authority in {
        SourceAuthorityClass.NOMINATING,
        SourceAuthorityClass.DIAGNOSTIC,
        SourceAuthorityClass.NONE,
    }:
        return True
    return not _is_reconstructed(receipt)


def _clause_ref_for(receipt: LogicPredictionReceipt) -> str:
    return (
        receipt.derived_clause_ref
        or receipt.derived_value_ref
        or receipt.derived_placement_ref
        or f"consequence:{receipt.hypothesis_id}"
    )


def _value_ref_for(receipt: LogicPredictionReceipt) -> str:
    return (
        receipt.derived_value_ref
        or receipt.derived_clause_ref
        or receipt.derived_placement_ref
        or f"value:{receipt.hypothesis_id}"
    )


def _placement_ref_for(receipt: LogicPredictionReceipt) -> str:
    return receipt.derived_placement_ref


# ---------------------------------------------------------------------------
# PredictionEvidenceBinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionEvidenceBinding(CanonicalContract):
    """One prediction consequence mapped onto exact RPR clause/value/placement refs.

    ``effective_precedence`` is always drawn from the existing
    :class:`BehaviorEvidencePrecedence` lattice via weakest-premise inheritance.
    ``proof_status`` is recorded separately and never used to invent rank.
    """

    SCHEMA: ClassVar[str] = PREDICTION_EVIDENCE_BINDING_SCHEMA

    binding_id: str
    prediction_receipt_id: str
    hypothesis_id: str
    candidate_id: str
    consequence_kind: ConsequenceKind
    disposition: BindingDisposition
    source_authority: SourceAuthorityClass
    proof_status: ProofStatus
    effective_precedence: BehaviorEvidencePrecedence
    clause_ref: str = ""
    value_ref: str = ""
    placement_ref: str = ""
    family: BehaviorClauseFamily = BehaviorClauseFamily.INVARIANTS
    premise_ids: tuple[str, ...] = ()
    premise_precedences: tuple[str, ...] = ()
    evidence_routes: tuple[str, ...] = ()
    proof_ref: str = ""
    reconstruction_id: str = ""
    kernel_receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    unsupported_flags: tuple[str, ...] = ()
    residual_gap_ids: tuple[str, ...] = ()
    automation_eligible: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _identifier(self.binding_id, "binding_id")
        )
        object.__setattr__(
            self,
            "prediction_receipt_id",
            _identifier(self.prediction_receipt_id, "prediction_receipt_id"),
        )
        object.__setattr__(
            self, "hypothesis_id", _identifier(self.hypothesis_id, "hypothesis_id")
        )
        object.__setattr__(
            self, "candidate_id", _text(self.candidate_id, "candidate_id", required=False)
        )
        object.__setattr__(
            self,
            "consequence_kind",
            _enum(self.consequence_kind, ConsequenceKind, "consequence_kind"),
        )
        object.__setattr__(
            self, "disposition", _enum(self.disposition, BindingDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "source_authority",
            _enum(self.source_authority, SourceAuthorityClass, "source_authority"),
        )
        object.__setattr__(
            self, "proof_status", _enum(self.proof_status, ProofStatus, "proof_status")
        )
        object.__setattr__(
            self,
            "effective_precedence",
            coerce_precedence(self.effective_precedence),
        )
        object.__setattr__(
            self, "clause_ref", _text(self.clause_ref, "clause_ref", required=False)
        )
        object.__setattr__(
            self, "value_ref", _text(self.value_ref, "value_ref", required=False)
        )
        object.__setattr__(
            self,
            "placement_ref",
            _text(self.placement_ref, "placement_ref", required=False),
        )
        object.__setattr__(self, "family", coerce_clause_family(self.family))
        object.__setattr__(self, "premise_ids", _ids(self.premise_ids, "premise_ids"))
        object.__setattr__(
            self,
            "premise_precedences",
            _ids(self.premise_precedences, "premise_precedences", bound=MAX_PREMISE_PRECEDENCES),
        )
        object.__setattr__(
            self, "evidence_routes", _ids(self.evidence_routes, "evidence_routes")
        )
        object.__setattr__(
            self, "proof_ref", _text(self.proof_ref, "proof_ref", required=False)
        )
        object.__setattr__(
            self,
            "reconstruction_id",
            _text(self.reconstruction_id, "reconstruction_id", required=False),
        )
        object.__setattr__(
            self,
            "kernel_receipt_id",
            _text(self.kernel_receipt_id, "kernel_receipt_id", required=False),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self,
            "unsupported_flags",
            _ids(self.unsupported_flags, "unsupported_flags"),
        )
        object.__setattr__(
            self, "residual_gap_ids", _ids(self.residual_gap_ids, "residual_gap_ids")
        )
        object.__setattr__(
            self,
            "automation_eligible",
            _bool(self.automation_eligible, "automation_eligible"),
        )
        if self.disposition is BindingDisposition.ADMITTED:
            if not (self.clause_ref or self.value_ref or self.placement_ref):
                raise TacticianGuidedBehaviorSynthesisAuthorityError(
                    "admitted binding requires exact clause, value, or placement ref"
                )
            if self.effective_precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS:
                raise TacticianGuidedBehaviorSynthesisAuthorityError(
                    "admitted bindings cannot rest solely on implementation hypothesis"
                )

    @property
    def is_admitted(self) -> bool:
        return self.disposition is BindingDisposition.ADMITTED

    @property
    def is_nomination(self) -> bool:
        return self.disposition is BindingDisposition.NOMINATION

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "binding_id": self.binding_id,
            "prediction_receipt_id": self.prediction_receipt_id,
            "hypothesis_id": self.hypothesis_id,
            "candidate_id": self.candidate_id,
            "consequence_kind": self.consequence_kind.value,
            "disposition": self.disposition.value,
            "source_authority": self.source_authority.value,
            "proof_status": self.proof_status.value,
            "effective_precedence": self.effective_precedence.value,
            "clause_ref": self.clause_ref,
            "value_ref": self.value_ref,
            "placement_ref": self.placement_ref,
            "family": self.family.value,
            "premise_ids": list(self.premise_ids),
            "premise_precedences": list(self.premise_precedences),
            "evidence_routes": list(self.evidence_routes),
            "proof_ref": self.proof_ref,
            "reconstruction_id": self.reconstruction_id,
            "kernel_receipt_id": self.kernel_receipt_id,
            "reason_codes": list(self.reason_codes),
            "assumption_refs": list(self.assumption_refs),
            "unsupported_flags": list(self.unsupported_flags),
            "residual_gap_ids": list(self.residual_gap_ids),
            "automation_eligible": self.automation_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionEvidenceBinding":
        if not isinstance(payload, Mapping):
            raise TacticianGuidedBehaviorSynthesisError(
                "prediction evidence binding payload must be a mapping"
            )
        if payload.get("schema") not in (None, cls.SCHEMA):
            raise TacticianGuidedBehaviorSynthesisError(
                "prediction evidence binding has an unsupported schema"
            )
        fields = (
            "binding_id",
            "prediction_receipt_id",
            "hypothesis_id",
            "candidate_id",
            "consequence_kind",
            "disposition",
            "source_authority",
            "proof_status",
            "effective_precedence",
            "clause_ref",
            "value_ref",
            "placement_ref",
            "family",
            "premise_ids",
            "premise_precedences",
            "evidence_routes",
            "proof_ref",
            "reconstruction_id",
            "kernel_receipt_id",
            "reason_codes",
            "assumption_refs",
            "unsupported_flags",
            "residual_gap_ids",
            "automation_eligible",
        )
        values = {name: payload[name] for name in fields if name in payload}
        return cls(**values)


# ---------------------------------------------------------------------------
# ContractRepairPredictionBridge
# ---------------------------------------------------------------------------


def _default_code_obligation(
    *,
    repository_id: str,
    tree_id: str,
    statement: str,
    premise_ids: Sequence[str],
    scope_ids: Sequence[str],
    scope_hint: str = "prediction",
) -> CodeProofObligation:
    return CodeProofObligation(
        repository_id=repository_id,
        repository_tree_id=tree_id,
        ast_scope_ids=tuple(scope_ids) or (f"scope:{scope_hint}",),
        statement=statement,
        premise_ids=tuple(premise_ids) or ("premise:prediction",),
        template_id="contract-repair/prediction-bridge",
        template_version="1",
        template_semantic_hash=content_identity(
            {"template": "contract-repair/prediction-bridge", "statement": statement}
        ),
        invariant_class="contract_repair",
        task_id="LPR-015",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )


def _kernel_verified_receipt_from_prediction(
    receipt: LogicPredictionReceipt,
    *,
    repository_id: str,
    tree_id: str,
    obligation_id: str,
    translator_id: str,
    toolchain_id: str,
    policy_id: str,
    premise_ids: Sequence[str],
    scope_ids: Sequence[str],
    backend_id: str,
) -> ProofReceipt:
    """Project an independently reconstructed prediction into ProofReceipt.

    The receipt only claims KERNEL_VERIFIED when the prediction itself already
    carries reconstruction + kernel receipt identities under authoritative
    source class.  This never elevates solver-only claims.
    """

    if not _is_reconstructed(receipt):
        raise TacticianGuidedBehaviorSynthesisAuthorityError(
            "kernel-verified projection requires reconstructed authoritative prediction"
        )
    code = _default_code_obligation(
        repository_id=repository_id,
        tree_id=tree_id,
        statement=(
            receipt.derived_clause_ref
            or receipt.derived_value_ref
            or receipt.derived_placement_ref
            or receipt.hypothesis_id
        ),
        premise_ids=premise_ids,
        scope_ids=scope_ids,
        scope_hint=obligation_id,
    )
    # Prefer the caller-supplied obligation identity when it is a compact ref;
    # CodeProofObligation derives its own content id which remains the subject.
    resolved_obligation_id = obligation_id or code.obligation_id
    evidence = ProofEvidence(
        EvidenceKind.KERNEL_VERIFICATION,
        EvidenceAuthority.KERNEL,
        EvidenceVerdict.ACCEPTED,
        artifact_id=receipt.kernel_receipt_id,
        subject_id=resolved_obligation_id,
        verifier_id=receipt.reconstruction_id,
        independent=True,
        freshness=EvidenceFreshness.CURRENT,
    )
    return ProofReceipt(
        obligation_id=resolved_obligation_id,
        plan_id=content_identity(
            {
                "interface": CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE,
                "prediction": receipt.receipt_id,
            }
        ),
        attempt_id=content_identity(
            {
                "reconstruction": receipt.reconstruction_id,
                "kernel": receipt.kernel_receipt_id,
                "prediction": receipt.receipt_id,
            }
        ),
        repository_id=repository_id,
        repository_tree_id=tree_id,
        ast_scope_ids=code.ast_scope_ids,
        premise_ids=code.premise_ids,
        translator_id=translator_id or "translator:prediction-bridge",
        solver_id=backend_id,
        kernel_id=receipt.reconstruction_id,
        toolchain_id=toolchain_id or "toolchain:prediction-bridge",
        policy_id=policy_id or "policy:prediction-bridge",
        resource_budget=ResourceBudget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        freshness=EvidenceFreshness.CURRENT,
        metadata={
            "prediction_receipt_id": receipt.receipt_id,
            "hypothesis_id": receipt.hypothesis_id,
            "reconstruction_id": receipt.reconstruction_id,
            "kernel_receipt_id": receipt.kernel_receipt_id,
            "source_authority": receipt.source_authority.value,
            "proof_status": receipt.proof_status.value,
            "producer_id": PRODUCER_ID,
        },
    )


def _non_conclusive_receipt_from_prediction(
    receipt: LogicPredictionReceipt,
    *,
    repository_id: str,
    tree_id: str,
    obligation_id: str,
    translator_id: str,
    toolchain_id: str,
    policy_id: str,
    premise_ids: Sequence[str],
    scope_ids: Sequence[str],
    backend_id: str,
    reason: str,
    verdict: ProofVerdict = ProofVerdict.INCONCLUSIVE,
) -> ProofReceipt:
    code = _default_code_obligation(
        repository_id=repository_id,
        tree_id=tree_id,
        statement=(
            receipt.derived_clause_ref
            or receipt.derived_value_ref
            or receipt.derived_placement_ref
            or receipt.hypothesis_id
        ),
        premise_ids=premise_ids,
        scope_ids=scope_ids,
        scope_hint=obligation_id,
    )
    resolved_obligation_id = obligation_id or code.obligation_id
    return ProofReceipt(
        obligation_id=resolved_obligation_id,
        plan_id=content_identity(
            {
                "interface": CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE,
                "prediction": receipt.receipt_id,
            }
        ),
        attempt_id=content_identity(
            {"reason": reason, "prediction": receipt.receipt_id}
        ),
        repository_id=repository_id,
        repository_tree_id=tree_id,
        ast_scope_ids=code.ast_scope_ids,
        premise_ids=code.premise_ids,
        translator_id=translator_id or "translator:prediction-bridge",
        solver_id=backend_id,
        kernel_id="independent-reconstruction-required",
        toolchain_id=toolchain_id or "toolchain:prediction-bridge",
        policy_id=policy_id or "policy:prediction-bridge",
        resource_budget=ResourceBudget(),
        verdict=verdict,
        freshness=EvidenceFreshness.CURRENT,
        metadata={
            "reason_codes": [reason],
            "prediction_receipt_id": receipt.receipt_id,
            "hypothesis_id": receipt.hypothesis_id,
            "source_authority": receipt.source_authority.value,
            "proof_status": receipt.proof_status.value,
            "producer_id": PRODUCER_ID,
            "nomination_only": True,
        },
    )


class ContractRepairPredictionBridge:
    """Project reconstructed predictions into CandidateProofBundle evidence.

    The bridge never replaces :class:`ContractRepairProver`; it only maps
    already-reconstructed prediction consequences into the exact evidence
    shape consumed by :class:`ContractRepairReranker`.
    """

    INTERFACE: ClassVar[str] = CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE

    def __init__(
        self,
        *,
        backend_id: str = "prediction-bridge",
        backend_version: str = "1",
        producer_id: str = PRODUCER_ID,
    ) -> None:
        self.backend_id = _identifier(backend_id, "backend_id")
        self.backend_version = _text(backend_version, "backend_version")
        self.producer_id = _identifier(producer_id, "producer_id")

    def project_bundle(
        self,
        *,
        candidate_id: str,
        repository_id: str,
        tree_id: str,
        predictions: Sequence[LogicPredictionReceipt | Mapping[str, Any]],
        reconstruction_receipts: Mapping[str, ProofReceipt | Mapping[str, Any]] | None = None,
        obligation_ids: Mapping[str, str] | None = None,
        premise_ids_by_prediction: Mapping[str, Sequence[str]] | None = None,
        scope_ids: Sequence[str] = (),
        translator_id: str = "",
        toolchain_id: str = "",
        policy_id: str = "",
        reason_codes: Sequence[str] = (),
    ) -> CandidateProofBundle:
        """Map candidate-specific prediction results into a CandidateProofBundle."""

        candidate = _identifier(candidate_id, "candidate_id")
        repo = _identifier(repository_id, "repository_id")
        tree = _identifier(tree_id, "tree_id")
        if isinstance(predictions, (str, bytes, bytearray)) or not isinstance(
            predictions, Sequence
        ):
            raise TacticianGuidedBehaviorSynthesisError(
                "predictions must be a sequence"
            )
        if not predictions:
            raise TacticianGuidedBehaviorSynthesisError(
                "proof projection requires at least one prediction"
            )
        if len(predictions) > MAX_PREDICTIONS:
            raise TacticianGuidedBehaviorSynthesisError(
                "predictions exceeds its item bound"
            )

        recon_map: dict[str, ProofReceipt] = {}
        if reconstruction_receipts:
            for key, value in reconstruction_receipts.items():
                recon_map[_identifier(key, "reconstruction_receipts")] = _proof_receipt(
                    value
                )

        results: list[CandidateProofResult] = []
        for raw in predictions:
            prediction = _prediction_receipt(raw)
            if prediction.candidate_id and prediction.candidate_id != candidate:
                # Candidate-specific projection only.
                continue
            results.append(
                self._project_one(
                    prediction,
                    candidate_id=candidate,
                    repository_id=repo,
                    tree_id=tree,
                    reconstruction_receipts=recon_map,
                    obligation_ids=obligation_ids or {},
                    premise_ids_by_prediction=premise_ids_by_prediction or {},
                    scope_ids=scope_ids,
                    translator_id=translator_id,
                    toolchain_id=toolchain_id,
                    policy_id=policy_id,
                )
            )

        if not results:
            raise TacticianGuidedBehaviorSynthesisError(
                "no candidate-specific predictions available for projection"
            )

        return CandidateProofBundle(
            candidate,
            repo,
            tree,
            tuple(results),
            self.backend_id,
            self.backend_version,
            reason_codes=tuple(reason_codes) or ("prediction_bridge_projection",),
        )

    def _project_one(
        self,
        prediction: LogicPredictionReceipt,
        *,
        candidate_id: str,
        repository_id: str,
        tree_id: str,
        reconstruction_receipts: Mapping[str, ProofReceipt],
        obligation_ids: Mapping[str, str],
        premise_ids_by_prediction: Mapping[str, Sequence[str]],
        scope_ids: Sequence[str],
        translator_id: str,
        toolchain_id: str,
        policy_id: str,
    ) -> CandidateProofResult:
        kind = _consequence_kind_for(prediction)
        projection = _projection_kind_for(kind)
        obligation_id = (
            obligation_ids.get(prediction.receipt_id)
            or obligation_ids.get(prediction.hypothesis_id)
            or f"obligation:prediction:{projection.value}:{prediction.hypothesis_id}"
        )
        premise_ids = tuple(
            premise_ids_by_prediction.get(prediction.receipt_id)
            or premise_ids_by_prediction.get(prediction.hypothesis_id)
            or prediction.assumption_refs
            or ("premise:prediction",)
        )
        cache_key_id = content_identity(
            {
                "candidate_id": candidate_id,
                "prediction_receipt_id": prediction.receipt_id,
                "projection": projection.value,
                "tree_id": tree_id,
            }
        )

        # Prefer an externally supplied independent reconstruction receipt.
        supplied = (
            reconstruction_receipts.get(prediction.kernel_receipt_id)
            or reconstruction_receipts.get(prediction.reconstruction_id)
            or reconstruction_receipts.get(prediction.receipt_id)
        )

        if prediction.disposition is PredictionDisposition.VALIDATED_REFUTATION:
            # Refutation without an independently verified counterexample stays
            # non-conclusive nomination evidence for the repair path.
            receipt = supplied or _non_conclusive_receipt_from_prediction(
                prediction,
                repository_id=repository_id,
                tree_id=tree_id,
                obligation_id=obligation_id,
                translator_id=translator_id,
                toolchain_id=toolchain_id,
                policy_id=policy_id,
                premise_ids=premise_ids,
                scope_ids=scope_ids,
                backend_id=self.backend_id,
                reason="validated_refutation_without_counterexample_attachment",
                verdict=ProofVerdict.INCONCLUSIVE,
            )
            if (
                supplied is not None
                and supplied.authoritative_verdict is ProofVerdict.DISPROVED
            ):
                # Full REFUTED requires a FormalCounterexample attachment which
                # the bridge does not invent; keep non-conclusive without it.
                return CandidateProofResult(
                    obligation_id,
                    receipt,
                    ContractRepairProofDisposition.NON_CONCLUSIVE,
                    ("validated_refutation_nomination", "counterexample_not_attached"),
                    cache_key_id,
                )
            return CandidateProofResult(
                obligation_id,
                receipt,
                ContractRepairProofDisposition.NON_CONCLUSIVE,
                ("validated_refutation_nomination",),
                cache_key_id,
            )

        if _is_reconstructed(prediction):
            if supplied is not None:
                if not supplied.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED):
                    return CandidateProofResult(
                        obligation_id,
                        supplied,
                        ContractRepairProofDisposition.NON_CONCLUSIVE,
                        ("reconstruction_receipt_not_kernel_verified",),
                        cache_key_id,
                    )
                proof_receipt = supplied
            else:
                proof_receipt = _kernel_verified_receipt_from_prediction(
                    prediction,
                    repository_id=repository_id,
                    tree_id=tree_id,
                    obligation_id=obligation_id,
                    translator_id=translator_id,
                    toolchain_id=toolchain_id,
                    policy_id=policy_id,
                    premise_ids=premise_ids,
                    scope_ids=scope_ids,
                    backend_id=self.backend_id,
                )
            return CandidateProofResult(
                obligation_id,
                proof_receipt,
                ContractRepairProofDisposition.PROVED,
                (
                    "independent_reconstruction",
                    f"projection:{projection.value}",
                    "prediction_admitted",
                ),
                cache_key_id,
            )

        # Stale / ambiguous / solver-only remains a nomination.
        reason = "stale_or_ambiguous_prediction_nomination"
        if prediction.disposition is PredictionDisposition.STALE:
            reason = "stale_prediction_nomination"
        elif prediction.proof_status is ProofStatus.SOLVER_CHECKED:
            reason = "solver_only_prediction_nomination"
        elif prediction.disposition is PredictionDisposition.UNSUPPORTED:
            reason = "unsupported_prediction_nomination"
        receipt = supplied or _non_conclusive_receipt_from_prediction(
            prediction,
            repository_id=repository_id,
            tree_id=tree_id,
            obligation_id=obligation_id,
            translator_id=translator_id,
            toolchain_id=toolchain_id,
            policy_id=policy_id,
            premise_ids=premise_ids,
            scope_ids=scope_ids,
            backend_id=self.backend_id,
            reason=reason,
        )
        disposition = (
            ContractRepairProofDisposition.UNSUPPORTED
            if prediction.disposition is PredictionDisposition.UNSUPPORTED
            or prediction.proof_status is ProofStatus.UNSUPPORTED
            else ContractRepairProofDisposition.NON_CONCLUSIVE
        )
        return CandidateProofResult(
            obligation_id,
            receipt,
            disposition,
            (reason, f"projection:{projection.value}"),
            cache_key_id,
        )


# ---------------------------------------------------------------------------
# TacticianBehaviorSynthesisReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticianBehaviorSynthesisReceipt(CanonicalContract):
    """Canonical composition receipt for prediction → RPR behavior/value bridge."""

    SCHEMA: ClassVar[str] = TACTICIAN_BEHAVIOR_SYNTHESIS_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    receipt_id: str
    disposition: BridgeDisposition
    bindings: tuple[PredictionEvidenceBinding, ...]
    prediction_receipt_ids: tuple[str, ...] = ()
    behavior_receipt: RequiredBehaviorSynthesisReceipt | None = None
    behavior_contract: RequiredBehaviorContract | None = None
    proof_bundle: CandidateProofBundle | None = None
    value_candidates: tuple[ValueCandidate, ...] = ()
    residual_gap_ids: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    unsupported_flags: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    write_authority: bool = False
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _propagation_roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "disposition", _enum(self.disposition, BridgeDisposition, "disposition")
        )
        bindings = tuple(self.bindings or ())
        if not all(isinstance(item, PredictionEvidenceBinding) for item in bindings):
            raise TacticianGuidedBehaviorSynthesisError(
                "bindings must contain PredictionEvidenceBinding values"
            )
        if len(bindings) > MAX_BINDINGS:
            raise TacticianGuidedBehaviorSynthesisError(
                "bindings exceeds its item bound"
            )
        bindings = tuple(
            sorted(
                bindings,
                key=lambda item: (item.binding_id, item.prediction_receipt_id),
            )
        )
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self,
            "prediction_receipt_ids",
            _ids(self.prediction_receipt_ids, "prediction_receipt_ids"),
        )
        if self.behavior_receipt is not None and not isinstance(
            self.behavior_receipt, RequiredBehaviorSynthesisReceipt
        ):
            raise TacticianGuidedBehaviorSynthesisError(
                "behavior_receipt must be RequiredBehaviorSynthesisReceipt or None"
            )
        if self.behavior_contract is not None and not isinstance(
            self.behavior_contract, RequiredBehaviorContract
        ):
            raise TacticianGuidedBehaviorSynthesisError(
                "behavior_contract must be RequiredBehaviorContract or None"
            )
        if self.proof_bundle is not None and not isinstance(
            self.proof_bundle, CandidateProofBundle
        ):
            raise TacticianGuidedBehaviorSynthesisError(
                "proof_bundle must be CandidateProofBundle or None"
            )
        values = tuple(self.value_candidates or ())
        if not all(isinstance(item, ValueCandidate) for item in values):
            raise TacticianGuidedBehaviorSynthesisError(
                "value_candidates must contain ValueCandidate values"
            )
        object.__setattr__(
            self,
            "value_candidates",
            tuple(sorted(values, key=lambda item: item.candidate_id)),
        )
        object.__setattr__(
            self, "residual_gap_ids", _ids(self.residual_gap_ids, "residual_gap_ids")
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self, "unsupported_flags", _ids(self.unsupported_flags, "unsupported_flags")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        # Bridge never grants write or semantic authority.
        if self.write_authority is not False:
            raise TacticianGuidedBehaviorSynthesisAuthorityError(
                "tactician guided behavior synthesis cannot claim write authority"
            )
        if self.semantic_authority is not False:
            raise TacticianGuidedBehaviorSynthesisAuthorityError(
                "tactician guided behavior synthesis cannot claim semantic authority"
            )
        object.__setattr__(self, "write_authority", False)
        object.__setattr__(self, "semantic_authority", False)

    @property
    def admitted_bindings(self) -> tuple[PredictionEvidenceBinding, ...]:
        return tuple(item for item in self.bindings if item.is_admitted)

    @property
    def nomination_bindings(self) -> tuple[PredictionEvidenceBinding, ...]:
        return tuple(item for item in self.bindings if item.is_nomination)

    @property
    def behavior_contracts_for_obligations(self) -> tuple[RequiredBehaviorContract, ...]:
        """Contracts accepted unchanged by ChangePropagationObligationCompiler."""

        if self.behavior_contract is not None:
            return (self.behavior_contract,)
        if (
            self.behavior_receipt is not None
            and self.behavior_receipt.contract is not None
        ):
            return (self.behavior_receipt.contract,)
        return ()

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "disposition": self.disposition.value,
            "bindings": [item.to_dict() for item in self.bindings],
            "prediction_receipt_ids": list(self.prediction_receipt_ids),
            "behavior_receipt": (
                self.behavior_receipt.to_dict()
                if self.behavior_receipt is not None
                else None
            ),
            "behavior_contract": (
                self.behavior_contract.to_dict()
                if self.behavior_contract is not None
                else None
            ),
            "proof_bundle": (
                self.proof_bundle.to_dict() if self.proof_bundle is not None else None
            ),
            "value_candidates": [item.to_dict() for item in self.value_candidates],
            "residual_gap_ids": list(self.residual_gap_ids),
            "assumption_refs": list(self.assumption_refs),
            "unsupported_flags": list(self.unsupported_flags),
            "reason_codes": list(self.reason_codes),
            "producer_id": self.producer_id,
            "write_authority": False,
            "semantic_authority": False,
        }


# ---------------------------------------------------------------------------
# TacticianGuidedBehaviorSynthesizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PremisePrecedenceBinding:
    """Explicit independent premise → existing BehaviorEvidencePrecedence."""

    premise_id: str
    precedence: BehaviorEvidencePrecedence
    source_route: SourceRouteKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "premise_id", _identifier(self.premise_id, "premise_id")
        )
        object.__setattr__(
            self, "precedence", coerce_precedence(self.precedence)
        )
        if self.source_route is not None:
            object.__setattr__(
                self,
                "source_route",
                _enum(self.source_route, SourceRouteKind, "source_route"),
            )


class TacticianGuidedBehaviorSynthesizer:
    """Compose admitted predictions with existing RPR behavior/value synthesizers.

    Does not reimplement RequiredBehaviorSynthesizer or MissingInputSynthesizer;
    it produces the exact evidence and candidate surfaces those components
    already consume, while projecting CandidateProofBundle evidence for
    ContractRepairReranker.
    """

    INTERFACE: ClassVar[str] = TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE

    def __init__(
        self,
        roots: PropagationAuthorityRoots,
        *,
        proof_bridge: ContractRepairPredictionBridge | None = None,
        behavior_synthesizer: RequiredBehaviorSynthesizer | None = None,
        producer_id: str = PRODUCER_ID,
    ) -> None:
        self.roots = _propagation_roots(roots)
        self.proof_bridge = proof_bridge or ContractRepairPredictionBridge()
        self.behavior_synthesizer = behavior_synthesizer or RequiredBehaviorSynthesizer(
            self.roots
        )
        self.producer_id = _identifier(producer_id, "producer_id")

    def bind_prediction(
        self,
        prediction: LogicPredictionReceipt | Mapping[str, Any],
        *,
        premise_precedences: Sequence[
            PremisePrecedenceBinding | BehaviorEvidencePrecedence | str
        ] = (),
        evidence_routes: Sequence[SourceRouteKind | str] = (),
        family: BehaviorClauseFamily | str | None = None,
        existing_atoms: Sequence[BehaviorEvidenceAtom] = (),
        blocked_clause_refs: Sequence[str] = (),
        unsupported_facet_tokens: Sequence[str] = (),
        consumer_requirement_ids: Sequence[str] = (),
    ) -> PredictionEvidenceBinding:
        """Map one prediction into an exact clause/value/placement binding."""

        receipt = _prediction_receipt(prediction)
        kind = _consequence_kind_for(receipt)
        clause_ref = _clause_ref_for(receipt)
        value_ref = _value_ref_for(receipt)
        placement_ref = _placement_ref_for(receipt)
        fam = _family_for_consequence(kind, family=family)

        resolved_precedences = self._resolve_premise_precedences(
            premise_precedences, evidence_routes=evidence_routes
        )
        effective = effective_consequence_precedence(
            source_authority=receipt.source_authority,
            premise_precedences=resolved_precedences,
            evidence_routes=evidence_routes,
            proof_status=receipt.proof_status,
        )
        premise_ids = tuple(
            item.premise_id
            for item in premise_precedences
            if isinstance(item, PremisePrecedenceBinding)
        )
        route_values = tuple(
            sorted(
                {
                    (
                        item.value
                        if isinstance(item, SourceRouteKind)
                        else str(item).strip()
                    )
                    for item in evidence_routes
                }
            )
        )
        unsupported = tuple(
            sorted(
                {
                    *receipt.residual_gap_ids,
                    *[
                        token
                        for token in unsupported_facet_tokens
                        if isinstance(token, str) and token.strip()
                    ],
                }
            )
        )
        reasons: list[str] = []
        disposition = BindingDisposition.ADMITTED

        if receipt.disposition is PredictionDisposition.STALE:
            disposition = BindingDisposition.NOMINATION
            reasons.append("stale_prediction_remains_nomination")
        elif _is_nomination_quality(receipt):
            disposition = BindingDisposition.NOMINATION
            reasons.append("non_reconstructed_or_non_authoritative_prediction")
        if not (clause_ref or value_ref or placement_ref):
            disposition = BindingDisposition.BLOCKED
            reasons.append("missing_exact_consequence_ref")
        if clause_ref in set(blocked_clause_refs):
            disposition = BindingDisposition.BLOCKED
            reasons.append("explicit_conflict_or_blocked_clause")
        if fam in _PROTECTED_CLAUSE_FAMILIES and (
            _protected_token_hit(unsupported)
            or _protected_token_hit(receipt.residual_gap_ids)
            or any(
                flag for flag in unsupported if _protected_token_hit((flag,))
            )
        ):
            disposition = BindingDisposition.BLOCKED
            reasons.append("unsupported_protected_facet")
        if _protected_token_hit(unsupported_facet_tokens) and fam in _PROTECTED_CLAUSE_FAMILIES:
            disposition = BindingDisposition.BLOCKED
            reasons.append("unsupported_memory_lifetime_native_concurrency_facet")
        # Never overwrite a higher-precedence independent source for the same family.
        if disposition is BindingDisposition.ADMITTED and existing_atoms:
            for atom in existing_atoms:
                if atom.family is not fam:
                    continue
                if precedence_rank(atom.precedence) < precedence_rank(effective):
                    # Existing source is stronger; prediction cannot overwrite it.
                    if atom.value_ref != value_ref or atom.clause_ref != clause_ref:
                        disposition = BindingDisposition.SUPERSEDED
                        reasons.append("higher_precedence_source_preserved")
                        break
                if (
                    precedence_rank(atom.precedence) == precedence_rank(effective)
                    and atom.value_digest != content_identity(
                        {"value_ref": value_ref, "clause_ref": clause_ref}
                    )
                    and atom.value_ref != value_ref
                ):
                    disposition = BindingDisposition.BLOCKED
                    reasons.append("explicit_same_rank_conflict")
                    break
        # Consumer-specific requirement cannot be silently replaced by a generic prediction.
        if consumer_requirement_ids and disposition is BindingDisposition.ADMITTED:
            # Bindings remain admitted but record the consumer constraint for composition.
            reasons.append("consumer_requirement_preserved")

        if (
            disposition is BindingDisposition.ADMITTED
            and effective is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
        ):
            disposition = BindingDisposition.NOMINATION
            reasons.append("implementation_hypothesis_not_admitted")

        if not reasons:
            if disposition is BindingDisposition.ADMITTED:
                reasons.append("admitted_reconstructed_consequence")
            else:
                reasons.append("nomination_only")

        binding_id = content_identity(
            {
                "prediction": receipt.receipt_id,
                "hypothesis": receipt.hypothesis_id,
                "clause": clause_ref,
                "value": value_ref,
                "placement": placement_ref,
                "disposition": disposition.value,
            }
        )
        return PredictionEvidenceBinding(
            binding_id=f"binding:{binding_id[:40]}",
            prediction_receipt_id=receipt.receipt_id,
            hypothesis_id=receipt.hypothesis_id,
            candidate_id=receipt.candidate_id,
            consequence_kind=kind,
            disposition=disposition,
            source_authority=receipt.source_authority,
            proof_status=receipt.proof_status,
            effective_precedence=effective,
            clause_ref=clause_ref,
            value_ref=value_ref,
            placement_ref=placement_ref,
            family=fam,
            premise_ids=premise_ids,
            premise_precedences=tuple(
                sorted({coerce_precedence(p).value for p in resolved_precedences})
            ),
            evidence_routes=route_values,
            proof_ref=receipt.kernel_receipt_id or receipt.reconstruction_id,
            reconstruction_id=receipt.reconstruction_id,
            kernel_receipt_id=receipt.kernel_receipt_id,
            reason_codes=tuple(sorted(set(reasons))),
            assumption_refs=receipt.assumption_refs,
            unsupported_flags=tuple(sorted(set(unsupported))),
            residual_gap_ids=receipt.residual_gap_ids,
            automation_eligible=(
                disposition is BindingDisposition.ADMITTED
                and receipt.automation_eligible
            ),
        )

    def atoms_from_bindings(
        self,
        bindings: Sequence[PredictionEvidenceBinding],
        *,
        subject_symbol_id: str,
        include_nominations: bool = False,
    ) -> tuple[BehaviorEvidenceAtom, ...]:
        """Project admitted (and optionally nominated) bindings to evidence atoms."""

        subject = _identifier(subject_symbol_id, "subject_symbol_id")
        atoms: list[BehaviorEvidenceAtom] = []
        for binding in bindings:
            if binding.disposition is BindingDisposition.ADMITTED:
                authoritative = is_authoritative(binding.effective_precedence)
                atoms.append(
                    BehaviorEvidenceAtom(
                        roots=self.roots,
                        evidence_id=f"evidence:prediction:{binding.binding_id}",
                        precedence=binding.effective_precedence,
                        family=binding.family,
                        clause_ref=binding.clause_ref or binding.value_ref,
                        value_ref=binding.value_ref or binding.clause_ref,
                        subject_symbol_id=subject,
                        statement_ref=binding.prediction_receipt_id,
                        assumption=False,
                        unsupported=False,
                        authoritative=authoritative,
                        proof_ref=binding.proof_ref,
                        source_path="",
                    )
                )
            elif include_nominations and binding.disposition is BindingDisposition.NOMINATION:
                atoms.append(
                    BehaviorEvidenceAtom(
                        roots=self.roots,
                        evidence_id=f"evidence:nomination:{binding.binding_id}",
                        precedence=BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
                        family=binding.family,
                        clause_ref=binding.clause_ref or binding.value_ref or binding.binding_id,
                        value_ref=binding.value_ref or binding.clause_ref or binding.binding_id,
                        subject_symbol_id=subject,
                        statement_ref=binding.prediction_receipt_id,
                        assumption=False,
                        unsupported=bool(binding.unsupported_flags),
                        authoritative=False,
                        proof_ref="",
                        source_path="",
                    )
                )
        return tuple(atoms)

    def merge_evidence(
        self,
        existing: Sequence[BehaviorEvidenceAtom],
        predicted: Sequence[BehaviorEvidenceAtom],
    ) -> tuple[BehaviorEvidenceAtom, ...]:
        """Merge prediction atoms without overwriting stronger independent sources.

        Same-family higher-precedence existing atoms win.  Same-rank conflicting
        values are both retained so RequiredBehaviorSynthesizer can surface a
        conflict gap rather than silently picking a side.
        """

        by_family: dict[BehaviorClauseFamily, list[BehaviorEvidenceAtom]] = {}
        for atom in existing:
            if not isinstance(atom, BehaviorEvidenceAtom):
                raise TacticianGuidedBehaviorSynthesisError(
                    "existing evidence must be BehaviorEvidenceAtom"
                )
            if not self._roots_compatible(atom.roots):
                raise TacticianGuidedBehaviorSynthesisAuthorityError(
                    "existing evidence roots must match synthesizer roots"
                )
            by_family.setdefault(atom.family, []).append(atom)

        merged: list[BehaviorEvidenceAtom] = list(existing)
        for atom in predicted:
            if not isinstance(atom, BehaviorEvidenceAtom):
                raise TacticianGuidedBehaviorSynthesisError(
                    "predicted evidence must be BehaviorEvidenceAtom"
                )
            if not self._roots_compatible(atom.roots):
                raise TacticianGuidedBehaviorSynthesisAuthorityError(
                    "predicted evidence roots must match synthesizer roots"
                )
            incumbents = by_family.get(atom.family, [])
            if not incumbents:
                merged.append(atom)
                by_family.setdefault(atom.family, []).append(atom)
                continue
            best_existing = min(incumbents, key=lambda item: precedence_rank(item.precedence))
            if precedence_rank(best_existing.precedence) < precedence_rank(atom.precedence):
                # Stronger independent source preserved; drop weaker prediction.
                continue
            # Equal or stronger prediction: keep both so conflicts remain explicit.
            merged.append(atom)
            by_family.setdefault(atom.family, []).append(atom)
        # Deterministic order.
        return tuple(
            sorted(
                merged,
                key=lambda item: (item.family.value, item.evidence_id),
            )
        )

    def value_candidates_from_bindings(
        self,
        bindings: Sequence[PredictionEvidenceBinding],
        *,
        requirement: MissingInputRequirement,
        source_path: str = "prediction/bridge.py",
        symbol_id: str = "",
    ) -> tuple[ValueCandidate, ...]:
        """Project value/construction consequences for MissingInputSynthesizer."""

        if not isinstance(requirement, MissingInputRequirement):
            raise TacticianGuidedBehaviorSynthesisError(
                "requirement must be MissingInputRequirement"
            )
        if not self._roots_compatible(requirement.roots):
            raise TacticianGuidedBehaviorSynthesisAuthorityError(
                "requirement roots must match synthesizer roots"
            )
        symbol = _identifier(
            symbol_id or requirement.type_ref or requirement.parameter_name,
            "symbol_id",
        )
        path = _text(source_path, "source_path")
        candidates: list[ValueCandidate] = []
        for binding in bindings:
            if binding.consequence_kind not in {
                ConsequenceKind.VALUE,
                ConsequenceKind.CONSTRUCTION,
                ConsequenceKind.SUBSTITUTION,
            }:
                continue
            if not binding.value_ref and not binding.clause_ref:
                continue
            expression = binding.value_ref or binding.clause_ref
            if binding.disposition is BindingDisposition.ADMITTED:
                disposition = ValueCandidateDisposition.PROVED
                kind = ValueCandidateKind.CONSTRUCTION
                semantic = True
                proof_refs = tuple(
                    ref
                    for ref in (binding.proof_ref, binding.kernel_receipt_id, binding.reconstruction_id)
                    if ref
                )
                if not proof_refs:
                    # Fail closed: proved values require proof refs.
                    disposition = ValueCandidateDisposition.NOMINATED
                    kind = ValueCandidateKind.GRAPH_NOMINATION
                    semantic = False
                    proof_refs = ()
            elif binding.disposition is BindingDisposition.NOMINATION:
                disposition = ValueCandidateDisposition.NOMINATED
                kind = ValueCandidateKind.GRAPH_NOMINATION
                semantic = False
                proof_refs = ()
            else:
                continue
            node = GraphNodeRef(
                node_id=f"node:prediction:{binding.hypothesis_id}",
                kind="prediction_value",
                path=path,
                symbol_id=symbol,
                artifact_id=binding.prediction_receipt_id,
                provenance=(
                    GraphProvenance.TRUSTED
                    if disposition is ValueCandidateDisposition.PROVED
                    else GraphProvenance.NOMINATED
                ),
                extractor_id=self.producer_id
                if disposition is ValueCandidateDisposition.PROVED
                else "",
            )
            candidates.append(
                ValueCandidate(
                    roots=self.roots,
                    candidate_id=f"value-candidate:{binding.binding_id}",
                    requirement_id=requirement.requirement_id,
                    kind=kind,
                    disposition=disposition,
                    source_node=node,
                    expression_ref=expression,
                    type_ref=requirement.type_ref or "type:unknown",
                    semantic_authority=semantic,
                    proof_refs=proof_refs,
                    rejection_reasons=(),
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    def synthesize(
        self,
        predictions: Sequence[LogicPredictionReceipt | Mapping[str, Any]],
        *,
        requirement: MissingInputRequirement | None = None,
        kind: BehaviorKind | str = BehaviorKind.CLASS,
        subject_symbol_id: str = "",
        existing_evidence: Sequence[BehaviorEvidenceAtom | Mapping[str, Any]] = (),
        premise_precedences: Sequence[
            PremisePrecedenceBinding
            | BehaviorEvidencePrecedence
            | Mapping[str, Any]
            | str
        ] = (),
        evidence_routes: Sequence[SourceRouteKind | str] = (),
        unsupported_facet_tokens: Sequence[str] = (),
        blocked_clause_refs: Sequence[str] = (),
        project_proof_bundle: bool = False,
        candidate_id: str = "",
        repository_id: str = "",
        tree_id: str = "",
        reconstruction_receipts: Mapping[str, ProofReceipt | Mapping[str, Any]]
        | None = None,
        include_nominations_as_atoms: bool = False,
        synthesize_behavior: bool = True,
    ) -> TacticianBehaviorSynthesisReceipt:
        """Compose predictions with existing RPR synthesizers into one receipt."""

        if isinstance(predictions, (str, bytes, bytearray)) or not isinstance(
            predictions, Sequence
        ):
            raise TacticianGuidedBehaviorSynthesisError(
                "predictions must be a sequence"
            )
        if len(predictions) > MAX_PREDICTIONS:
            raise TacticianGuidedBehaviorSynthesisError(
                "predictions exceeds its item bound"
            )

        receipts = tuple(_prediction_receipt(item) for item in predictions)
        premise_bindings = tuple(
            self._coerce_premise_binding(item) for item in premise_precedences
        )
        existing_atoms = self._normalize_existing_atoms(existing_evidence)

        bindings: list[PredictionEvidenceBinding] = []
        for receipt in receipts:
            bindings.append(
                self.bind_prediction(
                    receipt,
                    premise_precedences=premise_bindings,
                    evidence_routes=evidence_routes,
                    existing_atoms=existing_atoms,
                    blocked_clause_refs=blocked_clause_refs,
                    unsupported_facet_tokens=unsupported_facet_tokens,
                    consumer_requirement_ids=(
                        (requirement.requirement_id,) if requirement is not None else ()
                    ),
                )
            )

        subject = _identifier(
            subject_symbol_id
            or (requirement.type_ref if requirement is not None else "")
            or (requirement.parameter_name if requirement is not None else "")
            or "symbol:predicted",
            "subject_symbol_id",
        )
        predicted_atoms = self.atoms_from_bindings(
            bindings,
            subject_symbol_id=subject,
            include_nominations=include_nominations_as_atoms,
        )
        merged = self.merge_evidence(existing_atoms, predicted_atoms)

        behavior_receipt: RequiredBehaviorSynthesisReceipt | None = None
        behavior_contract: RequiredBehaviorContract | None = None
        if synthesize_behavior and requirement is not None:
            behavior_receipt = self.behavior_synthesizer.synthesize(
                requirement,
                kind=kind,
                subject_symbol_id=subject,
                evidence=merged,
                include_requirement_atoms=True,
            )
            behavior_contract = behavior_receipt.contract

        value_candidates: tuple[ValueCandidate, ...] = ()
        if requirement is not None:
            value_candidates = self.value_candidates_from_bindings(
                bindings, requirement=requirement, symbol_id=subject
            )

        proof_bundle: CandidateProofBundle | None = None
        if project_proof_bundle:
            if not candidate_id:
                # Prefer candidate identity from predictions.
                for receipt in receipts:
                    if receipt.candidate_id:
                        candidate_id = receipt.candidate_id
                        break
            if not candidate_id:
                raise TacticianGuidedBehaviorSynthesisError(
                    "project_proof_bundle requires candidate_id"
                )
            repo = repository_id or self.roots.repository_id
            tree = tree_id or self.roots.candidate_tree_id or self.roots.base_tree_id
            proof_bundle = self.proof_bridge.project_bundle(
                candidate_id=candidate_id,
                repository_id=repo,
                tree_id=tree,
                predictions=receipts,
                reconstruction_receipts=reconstruction_receipts,
                translator_id=self.roots.translator_id,
                toolchain_id=self.roots.toolchain_id,
                policy_id=self.roots.policy_id,
            )

        admitted = [item for item in bindings if item.is_admitted]
        nominations = [item for item in bindings if item.is_nomination]
        blocked = [
            item
            for item in bindings
            if item.disposition
            in {BindingDisposition.BLOCKED, BindingDisposition.SUPERSEDED}
        ]

        reasons: list[str] = []
        if proof_bundle is not None and proof_bundle.candidate_authoritative:
            disposition = BridgeDisposition.PROOF_PROJECTED
            reasons.append("candidate_proof_bundle_projected")
        elif admitted and behavior_receipt is not None:
            disposition = BridgeDisposition.COMPOSED
            reasons.append("behavior_and_value_consequences_composed")
        elif admitted:
            disposition = BridgeDisposition.COMPOSED
            reasons.append("admitted_consequences_without_behavior_requirement")
        elif nominations and not admitted:
            disposition = BridgeDisposition.NOMINATION_ONLY
            reasons.append("stale_or_ambiguous_remain_nominations")
        elif blocked and not admitted and not nominations:
            disposition = BridgeDisposition.BLOCKED
            reasons.append("all_consequences_blocked")
        else:
            disposition = BridgeDisposition.ABSTAINED
            reasons.append("no_eligible_consequences")

        residual = tuple(
            sorted(
                {
                    gap
                    for item in bindings
                    for gap in item.residual_gap_ids
                }
            )
        )
        assumptions = tuple(
            sorted(
                {
                    assumption
                    for item in bindings
                    for assumption in item.assumption_refs
                }
            )
        )
        unsupported = tuple(
            sorted(
                {
                    flag
                    for item in bindings
                    for flag in item.unsupported_flags
                }
                | set(unsupported_facet_tokens)
            )
        )
        receipt_id = content_identity(
            {
                "roots": self.roots.content_id,
                "bindings": [item.binding_id for item in bindings],
                "disposition": disposition.value,
                "producer": self.producer_id,
            }
        )
        return TacticianBehaviorSynthesisReceipt(
            roots=self.roots,
            receipt_id=f"tactician-behavior:{receipt_id[:40]}",
            disposition=disposition,
            bindings=tuple(bindings),
            prediction_receipt_ids=tuple(
                sorted({item.receipt_id for item in receipts})
            ),
            behavior_receipt=behavior_receipt,
            behavior_contract=behavior_contract,
            proof_bundle=proof_bundle,
            value_candidates=value_candidates,
            residual_gap_ids=residual,
            assumption_refs=assumptions,
            unsupported_flags=unsupported,
            reason_codes=tuple(sorted(set(reasons))),
            producer_id=self.producer_id,
            write_authority=False,
            semantic_authority=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _roots_compatible(self, roots: PropagationAuthorityRoots) -> bool:
        return (
            roots.repository_id == self.roots.repository_id
            and roots.candidate_tree_id == self.roots.candidate_tree_id
            and roots.base_tree_id == self.roots.base_tree_id
            and roots.policy_id == self.roots.policy_id
        )

    def _resolve_premise_precedences(
        self,
        premise_precedences: Sequence[
            PremisePrecedenceBinding | BehaviorEvidencePrecedence | str
        ],
        *,
        evidence_routes: Sequence[SourceRouteKind | str],
    ) -> tuple[BehaviorEvidencePrecedence, ...]:
        resolved: list[BehaviorEvidencePrecedence] = []
        for item in premise_precedences:
            if isinstance(item, PremisePrecedenceBinding):
                resolved.append(item.precedence)
            else:
                resolved.append(coerce_precedence(item))
        for route in evidence_routes:
            resolved.append(route_to_precedence(route))
        return tuple(resolved)

    def _coerce_premise_binding(
        self,
        value: PremisePrecedenceBinding
        | BehaviorEvidencePrecedence
        | Mapping[str, Any]
        | str,
    ) -> PremisePrecedenceBinding:
        if isinstance(value, PremisePrecedenceBinding):
            return value
        if isinstance(value, BehaviorEvidencePrecedence) or isinstance(value, str):
            precedence = coerce_precedence(value)
            return PremisePrecedenceBinding(
                premise_id=f"premise:{precedence.value}",
                precedence=precedence,
            )
        if isinstance(value, Mapping):
            return PremisePrecedenceBinding(
                premise_id=str(value.get("premise_id") or value.get("id") or ""),
                precedence=coerce_precedence(
                    value.get("precedence") or value.get("rank") or "history"
                ),
                source_route=value.get("source_route") or value.get("route"),
            )
        raise TacticianGuidedBehaviorSynthesisError(
            "premise precedence must be PremisePrecedenceBinding, "
            "BehaviorEvidencePrecedence, string, or mapping"
        )

    def _normalize_existing_atoms(
        self, existing: Sequence[BehaviorEvidenceAtom | Mapping[str, Any]]
    ) -> tuple[BehaviorEvidenceAtom, ...]:
        atoms: list[BehaviorEvidenceAtom] = []
        for item in existing:
            if isinstance(item, BehaviorEvidenceAtom):
                atom = item
            elif isinstance(item, Mapping):
                atom = BehaviorEvidenceAtom.from_mapping(self.roots, item)
            else:
                raise TacticianGuidedBehaviorSynthesisError(
                    "existing evidence must be BehaviorEvidenceAtom or mapping"
                )
            if not self._roots_compatible(atom.roots):
                raise TacticianGuidedBehaviorSynthesisAuthorityError(
                    "existing evidence roots must match synthesizer roots"
                )
            atoms.append(atom)
        return tuple(atoms)


def create_tactician_guided_behavior_synthesizer(
    roots: PropagationAuthorityRoots,
    **kwargs: Any,
) -> TacticianGuidedBehaviorSynthesizer:
    """Factory for the LPR-015 composition adapter."""

    return TacticianGuidedBehaviorSynthesizer(roots, **kwargs)


def create_contract_repair_prediction_bridge(
    **kwargs: Any,
) -> ContractRepairPredictionBridge:
    """Factory for CandidateProofBundle projection from predictions."""

    return ContractRepairPredictionBridge(**kwargs)


__all__ = (
    "TACTICIAN_GUIDED_BEHAVIOR_SYNTHESIS_INTERFACE",
    "CONTRACT_REPAIR_PREDICTION_BRIDGE_INTERFACE",
    "PREDICTION_EVIDENCE_BINDING_SCHEMA",
    "TACTICIAN_BEHAVIOR_SYNTHESIS_RECEIPT_SCHEMA",
    "PRODUCER_ID",
    "ConsequenceKind",
    "BindingDisposition",
    "BridgeDisposition",
    "ProjectionKind",
    "TacticianGuidedBehaviorSynthesisError",
    "TacticianGuidedBehaviorSynthesisAuthorityError",
    "PremisePrecedenceBinding",
    "PredictionEvidenceBinding",
    "ContractRepairPredictionBridge",
    "TacticianBehaviorSynthesisReceipt",
    "TacticianGuidedBehaviorSynthesizer",
    "create_tactician_guided_behavior_synthesizer",
    "create_contract_repair_prediction_bridge",
    "weakest_precedence",
    "route_to_precedence",
    "source_authority_floor",
    "effective_consequence_precedence",
    # Re-export exact types consumed/produced for callers and tests.
    "CandidateProofBundle",
    "BehaviorEvidencePrecedence",
    "BehaviorEvidenceAtom",
    "RequiredBehaviorSynthesizer",
    "CONTRACT_REPAIR_PROVER_INTERFACE",
)
