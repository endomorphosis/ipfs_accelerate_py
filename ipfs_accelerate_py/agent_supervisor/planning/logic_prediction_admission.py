"""Admit only reconstructed and unique logic predictions (LPR-014).

Fail-closed admission boundary between Hammer/CEGIS evidence and any later
RPR synthesis.  This module never chooses repair paths, never mutates
behavior/value records, and never emits write authority.

A consequence becomes eligible only when:

* every identity is recomputed and binds exact current authority roots;
* premises are independently authoritative (or conformance);
* translation, native-goal binding, environment, and kernel reconstruction
  all match and are accepted;
* bounded consistency is structurally valid (no ex-falso derivation from
  invalid/unknown consistency);
* assumptions and unsupported facets are preserved;
* rejection uses only an independently validated countermodel or proof of
  negation — never a raw solver claim;
* residual mandatory gaps, stale state, solver-only proof/refutation,
  learned/vector/model authority, and higher-precedence contract conflicts
  are rejected; and
* automatic value/construction/placement admits only when exactly one
  eligible consequence remains under deterministic tie rules (zero or
  multiple candidates abstain).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    GapDisposition,
    GoalDisposition,
    HypothesisDisposition,
    LogicGap,
    LogicHypothesis,
    LogicPredictionReceipt,
    NativeGoalDisposition,
    PredictionDisposition,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicNativeGoalBinding,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
)
from ..analysis.program_logic_premise_corpus import ConsistencyDisposition
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from ..proof.logic_prediction_cegis import (
    LogicRefinementReceipt,
    RefinementDisposition,
)
from ..proof.tactician_hammer_coordinator import (
    CoordinationConclusiveness,
    HammerCoordinationOutcome,
    HammerCoordinationReceipt,
)


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

LOGIC_PREDICTION_ADMISSION_INTERFACE: Final[str] = "LogicPredictionAdmission@1"
LOGIC_PREDICTION_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-prediction-decision@1"
)
LOGIC_PREDICTION_ADMISSION_VERSION: Final[int] = 1
ADMISSION_PRODUCER_ID: Final[str] = "logic-prediction-admission@1"

# Routes that can never grant semantic/source authority for admission.
_NON_AUTHORITATIVE_ROUTES: Final[frozenset[SourceRouteKind]] = frozenset(
    {
        SourceRouteKind.VECTOR,
        SourceRouteKind.KNOWLEDGE_GRAPH,
        SourceRouteKind.TACTICIAN,
        SourceRouteKind.LLM,
        SourceRouteKind.SOLVER,
        SourceRouteKind.RUNTIME_WITNESS,
        SourceRouteKind.HISTORY,
    }
)

_ADMITTABLE_SOURCE: Final[frozenset[SourceAuthorityClass]] = frozenset(
    {
        SourceAuthorityClass.AUTHORITATIVE,
        SourceAuthorityClass.CONFORMANCE,
    }
)

# Consistency states that allow non-ex-falso positive derivation.
_CONSISTENCY_OK: Final[frozenset[ConsistencyDisposition]] = frozenset(
    {ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK}
)

# Consistency states that are invalid (not merely unknown).
_CONSISTENCY_INVALID: Final[frozenset[ConsistencyDisposition]] = frozenset(
    {
        ConsistencyDisposition.STRUCTURAL_CONFLICT,
        ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION,
        ConsistencyDisposition.LOGICAL_CONFLICT_PROVED,
        ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED,
    }
)


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class LogicPredictionAdmissionError(ContractValidationError):
    """Malformed admission input that cannot be assessed."""


class LogicPredictionRejectionReason(str, Enum):
    """Closed rejection / abstention reasons for prediction admission."""

    MALFORMED_INPUT = "malformed_input"
    IDENTITY_MISMATCH = "identity_mismatch"
    ROOT_CHANGED = "root_changed"
    STALE_STATE = "stale_state"
    NON_AUTHORITATIVE_PREMISES = "non_authoritative_premises"
    TRANSLATION_MISMATCH = "translation_mismatch"
    NATIVE_GOAL_MISMATCH = "native_goal_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    KERNEL_NOT_ACCEPTED = "kernel_not_accepted"
    RECONSTRUCTION_MISSING = "reconstruction_missing"
    EX_FALSO_BLOCKED = "ex_falso_blocked"
    CONSISTENCY_UNKNOWN = "consistency_unknown"
    CONSISTENCY_INVALID = "consistency_invalid"
    MANDATORY_RESIDUAL_GAP = "mandatory_residual_gap"
    SOLVER_ONLY_PROOF = "solver_only_proof"
    SOLVER_ONLY_REFUTATION = "solver_only_refutation"
    RAW_SOLVER_COUNTERMODEL = "raw_solver_countermodel"
    LEARNED_AUTHORITY = "learned_authority"
    VECTOR_AUTHORITY = "vector_authority"
    MODEL_AUTHORITY = "model_authority"
    HIGHER_PRECEDENCE_CONFLICT = "higher_precedence_contract_conflict"
    PREDICTION_NON_UNIQUE = "prediction_non_unique"
    NO_ELIGIBLE_CONSEQUENCE = "no_eligible_consequence"
    ASSUMPTION_DROPPED = "assumption_dropped"
    UNSUPPORTED_FACET_DROPPED = "unsupported_facet_dropped"
    WRITE_AUTHORITY_CLAIMED = "write_authority_claimed"
    SEMANTIC_AUTHORITY_CLAIMED = "semantic_authority_claimed"
    HYPOTHESIS_NOT_KERNEL_VERIFIED = "hypothesis_not_kernel_verified"
    HAMMER_NOT_VERIFIED = "hammer_not_verified"
    HAMMER_NOT_CONCLUSIVE = "hammer_not_conclusive"
    REFINEMENT_INCOMPLETE = "refinement_incomplete"
    GOAL_UNSUPPORTED = "goal_unsupported"
    GOAL_STALE = "goal_stale"
    MISSING_NATIVE_BINDING = "missing_native_binding"
    MISSING_HAMMER_RECEIPT = "missing_hammer_receipt"
    VALIDATED_REFUTATION = "validated_refutation"
    COUNTERMODEL_UNVALIDATED = "countermodel_unvalidated"
    MULTIPLE_ELIGIBLE = "multiple_eligible_consequences"
    ZERO_ELIGIBLE = "zero_eligible_consequences"


class LogicPredictionDecisionDisposition(str, Enum):
    """Closed decision outcomes at the admission boundary."""

    ADMITTED = "admitted"
    VALIDATED_REFUTATION = "validated_refutation"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class AutomaticConsequenceKind(str, Enum):
    """Kinds that require uniqueness under deterministic tie rules."""

    NONE = "none"
    VALUE = "value"
    CONSTRUCTION = "construction"
    PLACEMENT = "placement"
    BEHAVIOR = "behavior"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise LogicPredictionAdmissionError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise LogicPredictionAdmissionError(f"{field_name} must not be empty")
    return result


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise LogicPredictionAdmissionError(f"{field_name} must be a boolean")
    return value


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise LogicPredictionAdmissionError(
            f"{field_name} must be a sequence of identifiers"
        )
    else:
        raw = values
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        ident = _text(item, field_name)
        if ident not in seen:
            seen.add(ident)
            result.append(ident)
    ordered = tuple(sorted(result))
    if required and not ordered:
        raise LogicPredictionAdmissionError(f"{field_name} must not be empty")
    return ordered


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise LogicPredictionAdmissionError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**dict(value))
        )
    raise LogicPredictionAdmissionError(
        "roots must be ProgramLogicAuthorityRoots"
    )


def _as_goal(value: Any) -> ProgramLogicGoal:
    if isinstance(value, ProgramLogicGoal):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicGoal.from_dict(value)
            if "schema" in value
            else ProgramLogicGoal(**dict(value))
        )
    raise LogicPredictionAdmissionError("goals must be ProgramLogicGoal values")


def _as_hypothesis(value: Any) -> LogicHypothesis:
    if isinstance(value, LogicHypothesis):
        return value
    if isinstance(value, Mapping):
        return (
            LogicHypothesis.from_dict(value)
            if "schema" in value
            else LogicHypothesis(**dict(value))
        )
    raise LogicPredictionAdmissionError(
        "hypotheses must be LogicHypothesis values"
    )


def _as_gap(value: Any) -> LogicGap:
    if isinstance(value, LogicGap):
        return value
    if isinstance(value, Mapping):
        return (
            LogicGap.from_dict(value)
            if "schema" in value
            else LogicGap(**dict(value))
        )
    raise LogicPredictionAdmissionError("residual_gaps must be LogicGap values")


def _as_countermodel(value: Any) -> CountermodelValidationReceipt:
    if isinstance(value, CountermodelValidationReceipt):
        return value
    if isinstance(value, Mapping):
        return (
            CountermodelValidationReceipt.from_dict(value)
            if "schema" in value
            else CountermodelValidationReceipt(**dict(value))
        )
    raise LogicPredictionAdmissionError(
        "countermodel_receipts must be CountermodelValidationReceipt values"
    )


def _as_native_binding(value: Any) -> ProgramLogicNativeGoalBinding:
    if isinstance(value, ProgramLogicNativeGoalBinding):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicNativeGoalBinding.from_dict(value)
            if "schema" in value
            else ProgramLogicNativeGoalBinding(**dict(value))
        )
    raise LogicPredictionAdmissionError(
        "native_goal_binding must be ProgramLogicNativeGoalBinding"
    )


def _as_hammer(value: Any) -> HammerCoordinationReceipt:
    if isinstance(value, HammerCoordinationReceipt):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        # Strip envelope keys that are not constructor fields.
        for key in (
            "schema_version",
            "interface",
            "producer_id",
            "coordinator_version",
            "is_conclusive",
        ):
            data.pop(key, None)
        outcome = data.get("outcome")
        if outcome is not None and not isinstance(outcome, HammerCoordinationOutcome):
            data["outcome"] = HammerCoordinationOutcome(str(outcome))
        conclusive = data.get("conclusiveness")
        if conclusive is not None and not isinstance(
            conclusive, CoordinationConclusiveness
        ):
            data["conclusiveness"] = CoordinationConclusiveness(str(conclusive))
        return HammerCoordinationReceipt(**data)
    raise LogicPredictionAdmissionError(
        "hammer_receipt must be HammerCoordinationReceipt"
    )


def _as_refinement(value: Any) -> LogicRefinementReceipt | None:
    if value is None:
        return None
    if isinstance(value, LogicRefinementReceipt):
        return value
    if isinstance(value, Mapping):
        return LogicRefinementReceipt.from_dict(value)
    raise LogicPredictionAdmissionError(
        "refinement_receipt must be LogicRefinementReceipt when provided"
    )


def _recompute_identity(record: CanonicalContract) -> str:
    """Recompute content identity from the canonical payload (fail-closed)."""

    recomputed = content_identity(record.to_dict())
    if hasattr(record, "content_id") and record.content_id != recomputed:
        # CanonicalContract.content_id is derived; mismatch means non-canonical.
        raise LogicPredictionAdmissionError(
            f"identity mismatch for {type(record).__name__}"
        )
    return recomputed


def _consequence_key(hypothesis: LogicHypothesis) -> str:
    """Deterministic identity of the claimed consequence for uniqueness."""

    if hypothesis.value_ref:
        return f"value:{hypothesis.value_ref}"
    if hypothesis.construction_ref:
        return f"construction:{hypothesis.construction_ref}"
    if hypothesis.placement_ref:
        return f"placement:{hypothesis.placement_ref}"
    return f"consequence:{hypothesis.claimed_consequence_ref}"


def _automatic_kind_for(
    hypothesis: LogicHypothesis,
    requested: AutomaticConsequenceKind,
) -> AutomaticConsequenceKind:
    """Resolve the uniqueness kind for one hypothesis.

    Explicit request wins when not NONE; otherwise infer from populated refs.
    """

    if requested is not AutomaticConsequenceKind.NONE:
        return requested
    if hypothesis.value_ref:
        return AutomaticConsequenceKind.VALUE
    if hypothesis.construction_ref:
        return AutomaticConsequenceKind.CONSTRUCTION
    if hypothesis.placement_ref:
        return AutomaticConsequenceKind.PLACEMENT
    if hypothesis.claimed_consequence_ref:
        # Behavior/clause consequences do not force uniqueness unless requested.
        return AutomaticConsequenceKind.NONE
    return AutomaticConsequenceKind.NONE


def _requires_uniqueness(kind: AutomaticConsequenceKind) -> bool:
    return kind in {
        AutomaticConsequenceKind.VALUE,
        AutomaticConsequenceKind.CONSTRUCTION,
        AutomaticConsequenceKind.PLACEMENT,
    }


def _countermodel_may_reject(receipt: CountermodelValidationReceipt) -> bool:
    """Only independently validated receipts may reject (never raw solver)."""

    if not receipt.may_reject_hypothesis:
        return False
    if receipt.disposition is not CountermodelDisposition.VALIDATED:
        return False
    has_replay = bool(receipt.replayed_rejection_evidence_refs) and bool(
        receipt.replay_method
    )
    has_negation = bool(receipt.proof_of_negation_id)
    return has_replay or has_negation


def _mandatory_gaps(
    gaps: Sequence[LogicGap],
    *,
    goal_id: str,
) -> tuple[LogicGap, ...]:
    result: list[LogicGap] = []
    for gap in gaps:
        if gap.goal_id != goal_id:
            continue
        if gap.disposition in {
            GapDisposition.COVERED,
            GapDisposition.OPTIONAL,
            GapDisposition.STALE,
        }:
            continue
        severity = str(getattr(gap, "severity", "mandatory") or "mandatory").lower()
        if severity in {"mandatory", "required", "blocking"} or gap.disposition in {
            GapDisposition.REQUIRED,
            GapDisposition.FRONTIER,
            GapDisposition.UNSUPPORTED,
        }:
            # Optional severity may still be REQUIRED disposition — treat as mandatory.
            if gap.disposition is GapDisposition.OPTIONAL:
                continue
            result.append(gap)
    return tuple(sorted(result, key=lambda item: item.gap_id))


# ---------------------------------------------------------------------------
# Request / decision records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicPredictionAdmissionRequest:
    """Immutable admission request binding current evidence under one root set."""

    roots: ProgramLogicAuthorityRoots
    goals: tuple[ProgramLogicGoal, ...]
    hypotheses: tuple[LogicHypothesis, ...]
    tactician_plan_id: str
    hammer_receipt: HammerCoordinationReceipt
    native_goal_binding: ProgramLogicNativeGoalBinding
    consistency_disposition: ConsistencyDisposition
    countermodel_receipts: tuple[CountermodelValidationReceipt, ...] = ()
    residual_gaps: tuple[LogicGap, ...] = ()
    refinement_receipt: LogicRefinementReceipt | None = None
    proof_receipt_id: str = ""
    kernel_receipt_id: str = ""
    reconstruction_id: str = ""
    environment_receipt_id: str = ""
    translation_id: str = ""
    candidate_id: str = ""
    automatic_kind: AutomaticConsequenceKind = AutomaticConsequenceKind.NONE
    higher_precedence_conflict: bool = False
    write_authority_claimed: bool = False
    semantic_authority_claimed: bool = False
    learned_selector_model_digest: str = ""
    current_tree_id: str = ""
    current_corpus_id: str = ""
    current_environment_id: str = ""
    current_toolchain_id: str = ""
    current_policy_id: str = ""
    current_translator_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        if (
            isinstance(self.goals, (str, bytes, bytearray))
            or not isinstance(self.goals, Sequence)
            or not self.goals
        ):
            raise LogicPredictionAdmissionError(
                "goals must be a non-empty ProgramLogicGoal sequence"
            )
        goals = tuple(_as_goal(item) for item in self.goals)
        object.__setattr__(
            self, "goals", tuple(sorted(goals, key=lambda item: item.goal_id))
        )
        if (
            isinstance(self.hypotheses, (str, bytes, bytearray))
            or not isinstance(self.hypotheses, Sequence)
            or not self.hypotheses
        ):
            raise LogicPredictionAdmissionError(
                "hypotheses must be a non-empty LogicHypothesis sequence"
            )
        hyps = tuple(_as_hypothesis(item) for item in self.hypotheses)
        hyp_ids = [item.hypothesis_id for item in hyps]
        if len(set(hyp_ids)) != len(hyp_ids):
            raise LogicPredictionAdmissionError("hypothesis identities must be unique")
        object.__setattr__(
            self,
            "hypotheses",
            tuple(sorted(hyps, key=lambda item: item.hypothesis_id)),
        )
        object.__setattr__(
            self,
            "tactician_plan_id",
            _text(self.tactician_plan_id, "tactician_plan_id"),
        )
        object.__setattr__(self, "hammer_receipt", _as_hammer(self.hammer_receipt))
        object.__setattr__(
            self, "native_goal_binding", _as_native_binding(self.native_goal_binding)
        )
        object.__setattr__(
            self,
            "consistency_disposition",
            _enum(
                self.consistency_disposition,
                ConsistencyDisposition,
                "consistency_disposition",
            ),
        )
        cms = tuple(
            _as_countermodel(item) for item in (self.countermodel_receipts or ())
        )
        object.__setattr__(
            self,
            "countermodel_receipts",
            tuple(sorted(cms, key=lambda item: item.receipt_id)),
        )
        gaps = tuple(_as_gap(item) for item in (self.residual_gaps or ()))
        object.__setattr__(
            self,
            "residual_gaps",
            tuple(sorted(gaps, key=lambda item: item.gap_id)),
        )
        object.__setattr__(
            self, "refinement_receipt", _as_refinement(self.refinement_receipt)
        )
        for name in (
            "proof_receipt_id",
            "kernel_receipt_id",
            "reconstruction_id",
            "environment_receipt_id",
            "translation_id",
            "candidate_id",
            "learned_selector_model_digest",
            "current_tree_id",
            "current_corpus_id",
            "current_environment_id",
            "current_toolchain_id",
            "current_policy_id",
            "current_translator_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "automatic_kind",
            _enum(self.automatic_kind, AutomaticConsequenceKind, "automatic_kind"),
        )
        for name in (
            "higher_precedence_conflict",
            "write_authority_claimed",
            "semantic_authority_claimed",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/logic-prediction-admission-request@1",
            "interface": LOGIC_PREDICTION_ADMISSION_INTERFACE,
            "roots": self.roots.to_dict(),
            "goals": [item.to_dict() for item in self.goals],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "tactician_plan_id": self.tactician_plan_id,
            "hammer_receipt": self.hammer_receipt.to_dict(),
            "native_goal_binding": self.native_goal_binding.to_dict(),
            "consistency_disposition": self.consistency_disposition.value,
            "countermodel_receipts": [
                item.to_dict() for item in self.countermodel_receipts
            ],
            "residual_gaps": [item.to_dict() for item in self.residual_gaps],
            "refinement_receipt": (
                self.refinement_receipt.to_dict()
                if self.refinement_receipt is not None
                else None
            ),
            "proof_receipt_id": self.proof_receipt_id,
            "kernel_receipt_id": self.kernel_receipt_id,
            "reconstruction_id": self.reconstruction_id,
            "environment_receipt_id": self.environment_receipt_id,
            "translation_id": self.translation_id,
            "candidate_id": self.candidate_id,
            "automatic_kind": self.automatic_kind.value,
            "higher_precedence_conflict": self.higher_precedence_conflict,
            "write_authority_claimed": self.write_authority_claimed,
            "semantic_authority_claimed": self.semantic_authority_claimed,
            "learned_selector_model_digest": self.learned_selector_model_digest,
            "current_tree_id": self.current_tree_id,
            "current_corpus_id": self.current_corpus_id,
            "current_environment_id": self.current_environment_id,
            "current_toolchain_id": self.current_toolchain_id,
            "current_policy_id": self.current_policy_id,
            "current_translator_id": self.current_translator_id,
            "metadata": dict(self.metadata),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class LogicPredictionDecision:
    """Admission decision; never carries write authority."""

    decision_id: str
    disposition: LogicPredictionDecisionDisposition
    roots: ProgramLogicAuthorityRoots
    goal_id: str
    hypothesis_id: str
    reason_codes: tuple[str, ...]
    eligible_consequence_refs: tuple[str, ...] = ()
    selected_consequence_ref: str = ""
    receipt: LogicPredictionReceipt | None = None
    assumption_refs: tuple[str, ...] = ()
    unsupported_facet_ids: tuple[str, ...] = ()
    residual_gap_ids: tuple[str, ...] = ()
    countermodel_validation_id: str = ""
    automation_eligible: bool = False
    write_authority: bool = False
    semantic_authority: bool = False
    prediction_disposition: PredictionDisposition = PredictionDisposition.ABSTAINED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision_id", _text(self.decision_id, "decision_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(
                self.disposition,
                LogicPredictionDecisionDisposition,
                "disposition",
            ),
        )
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "hypothesis_id", _text(self.hypothesis_id, "hypothesis_id", required=False)
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self,
            "eligible_consequence_refs",
            _ids(self.eligible_consequence_refs, "eligible_consequence_refs"),
        )
        object.__setattr__(
            self,
            "selected_consequence_ref",
            _text(
                self.selected_consequence_ref,
                "selected_consequence_ref",
                required=False,
            ),
        )
        if self.receipt is not None and not isinstance(
            self.receipt, LogicPredictionReceipt
        ):
            raise LogicPredictionAdmissionError(
                "receipt must be LogicPredictionReceipt when provided"
            )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self,
            "unsupported_facet_ids",
            _ids(self.unsupported_facet_ids, "unsupported_facet_ids"),
        )
        object.__setattr__(
            self, "residual_gap_ids", _ids(self.residual_gap_ids, "residual_gap_ids")
        )
        object.__setattr__(
            self,
            "countermodel_validation_id",
            _text(
                self.countermodel_validation_id,
                "countermodel_validation_id",
                required=False,
            ),
        )
        for name in (
            "automation_eligible",
            "write_authority",
            "semantic_authority",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        # Hard invariant: admission receipts never grant write authority.
        if self.write_authority:
            raise LogicPredictionAdmissionError(
                "logic prediction decisions cannot claim write authority"
            )
        object.__setattr__(self, "write_authority", False)
        if self.semantic_authority:
            raise LogicPredictionAdmissionError(
                "logic prediction decisions cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self,
            "prediction_disposition",
            _enum(
                self.prediction_disposition,
                PredictionDisposition,
                "prediction_disposition",
            ),
        )
        if (
            self.disposition is LogicPredictionDecisionDisposition.ADMITTED
            and self.receipt is None
        ):
            raise LogicPredictionAdmissionError(
                "admitted decisions require a LogicPredictionReceipt"
            )
        if (
            self.disposition is LogicPredictionDecisionDisposition.ADMITTED
            and not self.automation_eligible
            and self.prediction_disposition is PredictionDisposition.PROVED
        ):
            # Proved admissions for automation must mark eligibility; non-auto
            # clause admissions may still be admitted with automation_eligible=False
            # only when the receipt agrees.
            if self.receipt is not None and self.receipt.automation_eligible:
                object.__setattr__(self, "automation_eligible", True)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LOGIC_PREDICTION_DECISION_SCHEMA,
            "interface": LOGIC_PREDICTION_ADMISSION_INTERFACE,
            "producer_id": ADMISSION_PRODUCER_ID,
            "admission_version": LOGIC_PREDICTION_ADMISSION_VERSION,
            "decision_id": self.decision_id,
            "disposition": self.disposition.value,
            "roots": self.roots.to_dict(),
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "reason_codes": list(self.reason_codes),
            "eligible_consequence_refs": list(self.eligible_consequence_refs),
            "selected_consequence_ref": self.selected_consequence_ref,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "assumption_refs": list(self.assumption_refs),
            "unsupported_facet_ids": list(self.unsupported_facet_ids),
            "residual_gap_ids": list(self.residual_gap_ids),
            "countermodel_validation_id": self.countermodel_validation_id,
            "automation_eligible": self.automation_eligible,
            "write_authority": False,
            "semantic_authority": False,
            "prediction_disposition": self.prediction_disposition.value,
            "metadata": dict(self.metadata),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def is_admitted(self) -> bool:
        return self.disposition is LogicPredictionDecisionDisposition.ADMITTED

    @property
    def is_refuted(self) -> bool:
        return (
            self.disposition
            is LogicPredictionDecisionDisposition.VALIDATED_REFUTATION
        )


# ---------------------------------------------------------------------------
# Admission engine
# ---------------------------------------------------------------------------


@dataclass
class LogicPredictionAdmission:
    """Fail-closed admission of reconstructed, unique logic predictions."""

    producer_id: str = ADMISSION_PRODUCER_ID

    def admit(
        self,
        request: LogicPredictionAdmissionRequest | Mapping[str, Any],
    ) -> LogicPredictionDecision:
        req = (
            request
            if isinstance(request, LogicPredictionAdmissionRequest)
            else LogicPredictionAdmissionRequest(**dict(request))
        )
        return self._admit(req)

    decide = admit
    assess = admit
    evaluate = admit

    def _admit(
        self, req: LogicPredictionAdmissionRequest
    ) -> LogicPredictionDecision:
        roots = req.roots
        reasons: list[str] = []

        # ------------------------------------------------------------------
        # 1) Recompute identities and bind exact roots.
        # ------------------------------------------------------------------
        try:
            for goal in req.goals:
                _recompute_identity(goal)
            for hyp in req.hypotheses:
                _recompute_identity(hyp)
            _recompute_identity(req.native_goal_binding)
            for cm in req.countermodel_receipts:
                _recompute_identity(cm)
            for gap in req.residual_gaps:
                _recompute_identity(gap)
        except (LogicPredictionAdmissionError, ContractValidationError) as exc:
            return self._terminal(
                req,
                disposition=LogicPredictionDecisionDisposition.REJECTED,
                prediction=PredictionDisposition.ERROR,
                reasons=(LogicPredictionRejectionReason.IDENTITY_MISMATCH.value,),
                detail=str(exc),
            )

        root_problems = self._root_problems(req)
        reasons.extend(root_problems)

        # Hard authority / write claims on the request itself.
        if req.write_authority_claimed:
            reasons.append(
                LogicPredictionRejectionReason.WRITE_AUTHORITY_CLAIMED.value
            )
        if req.semantic_authority_claimed:
            reasons.append(
                LogicPredictionRejectionReason.SEMANTIC_AUTHORITY_CLAIMED.value
            )
        if req.higher_precedence_conflict:
            reasons.append(
                LogicPredictionRejectionReason.HIGHER_PRECEDENCE_CONFLICT.value
            )

        # ------------------------------------------------------------------
        # 2) Select primary goal (single-goal admission unit).
        # ------------------------------------------------------------------
        if len(req.goals) != 1:
            # Multi-goal batches are assessed per-goal by the caller; this
            # boundary admits one goal's reconstructed consequences at a time.
            return self._terminal(
                req,
                disposition=LogicPredictionDecisionDisposition.ABSTAINED,
                prediction=PredictionDisposition.ABSTAINED,
                reasons=(
                    LogicPredictionRejectionReason.MALFORMED_INPUT.value,
                    "single_goal_required",
                ),
            )
        goal = req.goals[0]
        goal_hyps = tuple(
            h for h in req.hypotheses if h.target_goal_id == goal.goal_id
        )
        if not goal_hyps:
            return self._terminal(
                req,
                disposition=LogicPredictionDecisionDisposition.ABSTAINED,
                prediction=PredictionDisposition.ABSTAINED,
                reasons=(
                    LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value,
                ),
                goal_id=goal.goal_id,
            )

        assumption_refs = tuple(sorted(goal.assumption_refs))
        unsupported_facet_ids = tuple(
            sorted({facet.facet_id for facet in goal.unsupported_facets})
        )

        if goal.disposition is GoalDisposition.STALE:
            reasons.append(LogicPredictionRejectionReason.GOAL_STALE.value)
        if goal.disposition is GoalDisposition.UNSUPPORTED:
            reasons.append(LogicPredictionRejectionReason.GOAL_UNSUPPORTED.value)

        # ------------------------------------------------------------------
        # 3) Stale / current-root checks.
        # ------------------------------------------------------------------
        stale = self._stale_problems(req, goal)
        reasons.extend(stale)

        # ------------------------------------------------------------------
        # 4) Consistency / ex-falso gate (positive derivation only).
        # ------------------------------------------------------------------
        consistency = req.consistency_disposition
        consistency_blocks_positive = False
        if consistency is ConsistencyDisposition.UNKNOWN:
            reasons.append(
                LogicPredictionRejectionReason.CONSISTENCY_UNKNOWN.value
            )
            reasons.append(LogicPredictionRejectionReason.EX_FALSO_BLOCKED.value)
            consistency_blocks_positive = True
        elif consistency in _CONSISTENCY_INVALID:
            reasons.append(
                LogicPredictionRejectionReason.CONSISTENCY_INVALID.value
            )
            reasons.append(LogicPredictionRejectionReason.EX_FALSO_BLOCKED.value)
            consistency_blocks_positive = True
        elif consistency not in _CONSISTENCY_OK:
            reasons.append(LogicPredictionRejectionReason.EX_FALSO_BLOCKED.value)
            consistency_blocks_positive = True

        # ------------------------------------------------------------------
        # 5) Hammer + native kernel reconstruction.
        # ------------------------------------------------------------------
        hammer = req.hammer_receipt
        native = req.native_goal_binding
        hammer_problems = self._hammer_problems(req)
        reasons.extend(hammer_problems)
        native_problems = self._native_problems(req)
        reasons.extend(native_problems)

        kernel_ok = (
            hammer.outcome is HammerCoordinationOutcome.VERIFIED
            and hammer.kernel_checked
            and hammer.proof_success
            and hammer.conclusiveness
            is CoordinationConclusiveness.CONCLUSIVE_PROOF
            and bool(req.reconstruction_id or _binding_reconstruction(hammer))
            and bool(req.kernel_receipt_id or hammer.native_goal_binding_id)
            and native.disposition is NativeGoalDisposition.ROUND_TRIP_OK
            and native.kernel_id
            and native.kernel_id != "solver-only"
        )

        # ------------------------------------------------------------------
        # 6) Residual mandatory gaps (from request + refinement).
        # ------------------------------------------------------------------
        residual_ids: set[str] = {
            gap.gap_id for gap in _mandatory_gaps(req.residual_gaps, goal_id=goal.goal_id)
        }
        if req.refinement_receipt is not None:
            residual_ids.update(req.refinement_receipt.residual_gap_ids)
            if req.refinement_receipt.disposition not in {
                RefinementDisposition.REFINED,
                RefinementDisposition.FIXED_POINT,
            }:
                # Incomplete refinement does not by itself block validated
                # refutation, but blocks positive automation.
                reasons.append(
                    LogicPredictionRejectionReason.REFINEMENT_INCOMPLETE.value
                )
        if residual_ids:
            reasons.append(
                LogicPredictionRejectionReason.MANDATORY_RESIDUAL_GAP.value
            )

        # ------------------------------------------------------------------
        # 7) Independently validated countermodels (rejection authority only).
        # ------------------------------------------------------------------
        validated_rejectors: list[CountermodelValidationReceipt] = []
        diagnostic_only: list[CountermodelValidationReceipt] = []
        for cm in req.countermodel_receipts:
            if cm.roots.content_id != roots.content_id and cm.roots != roots:
                reasons.append(LogicPredictionRejectionReason.ROOT_CHANGED.value)
                continue
            if _countermodel_may_reject(cm):
                validated_rejectors.append(cm)
            elif cm.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY:
                diagnostic_only.append(cm)
            elif cm.disposition is CountermodelDisposition.VALIDATED:
                # Claimed validated but missing replay/negation evidence.
                reasons.append(
                    LogicPredictionRejectionReason.COUNTERMODEL_UNVALIDATED.value
                )
            else:
                # Stale / unsupported / replay_failed — non-authoritative.
                diagnostic_only.append(cm)

        # Validated refutation path: never use raw solver claims.
        if validated_rejectors:
            # Deterministic selection of the first validated rejector by id.
            rejector = sorted(
                validated_rejectors, key=lambda item: item.receipt_id
            )[0]
            # Target hypothesis if counterexample points at one; else all.
            target_hyp = self._refutation_target(goal_hyps, rejector)
            receipt = self._build_receipt(
                req=req,
                goal=goal,
                hypothesis=target_hyp,
                disposition=PredictionDisposition.VALIDATED_REFUTATION,
                proof_status=ProofStatus.VALIDATED_REFUTED,
                source_authority=target_hyp.source_authority
                if target_hyp.source_authority in _ADMITTABLE_SOURCE
                else SourceAuthorityClass.NONE,
                automation_eligible=True,
                countermodel_validation_id=rejector.receipt_id,
                residual_gap_ids=tuple(sorted(residual_ids)),
                assumption_refs=assumption_refs,
            )
            return LogicPredictionDecision(
                decision_id=self._decision_id(req, "refute", target_hyp.hypothesis_id),
                disposition=LogicPredictionDecisionDisposition.VALIDATED_REFUTATION,
                roots=roots,
                goal_id=goal.goal_id,
                hypothesis_id=target_hyp.hypothesis_id,
                reason_codes=(
                    LogicPredictionRejectionReason.VALIDATED_REFUTATION.value,
                ),
                eligible_consequence_refs=(_consequence_key(target_hyp),),
                selected_consequence_ref=_consequence_key(target_hyp),
                receipt=receipt,
                assumption_refs=assumption_refs,
                unsupported_facet_ids=unsupported_facet_ids,
                residual_gap_ids=tuple(sorted(residual_ids)),
                countermodel_validation_id=rejector.receipt_id,
                automation_eligible=True,
                write_authority=False,
                semantic_authority=False,
                prediction_disposition=PredictionDisposition.VALIDATED_REFUTATION,
            )

        # Raw/diagnostic countermodels must not reject.
        if diagnostic_only and not validated_rejectors:
            # Record that solver claims are non-authoritative for rejection.
            if any(
                cm.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY
                for cm in diagnostic_only
            ):
                # Do not add as a hard failure by itself; positive path continues.
                pass

        # ------------------------------------------------------------------
        # 8) Positive admission: collect eligible consequences.
        # ------------------------------------------------------------------
        hard_blockers = {
            LogicPredictionRejectionReason.WRITE_AUTHORITY_CLAIMED.value,
            LogicPredictionRejectionReason.SEMANTIC_AUTHORITY_CLAIMED.value,
            LogicPredictionRejectionReason.HIGHER_PRECEDENCE_CONFLICT.value,
            LogicPredictionRejectionReason.ROOT_CHANGED.value,
            LogicPredictionRejectionReason.STALE_STATE.value,
            LogicPredictionRejectionReason.GOAL_STALE.value,
            LogicPredictionRejectionReason.MANDATORY_RESIDUAL_GAP.value,
            LogicPredictionRejectionReason.IDENTITY_MISMATCH.value,
        }
        if any(code in hard_blockers for code in reasons):
            return self._terminal(
                req,
                disposition=LogicPredictionDecisionDisposition.REJECTED,
                prediction=self._prediction_for_reasons(reasons),
                reasons=tuple(sorted(set(reasons))),
                goal_id=goal.goal_id,
                assumption_refs=assumption_refs,
                unsupported_facet_ids=unsupported_facet_ids,
                residual_gap_ids=tuple(sorted(residual_ids)),
            )

        if consistency_blocks_positive or not kernel_ok or hammer_problems or native_problems:
            # Fail closed: cannot derive positive consequences.
            merged = list(reasons)
            if not kernel_ok and LogicPredictionRejectionReason.HAMMER_NOT_VERIFIED.value not in merged:
                if hammer.outcome is not HammerCoordinationOutcome.VERIFIED:
                    merged.append(
                        LogicPredictionRejectionReason.HAMMER_NOT_VERIFIED.value
                    )
                if not (
                    req.reconstruction_id or _binding_reconstruction(hammer)
                ):
                    merged.append(
                        LogicPredictionRejectionReason.RECONSTRUCTION_MISSING.value
                    )
            return self._terminal(
                req,
                disposition=LogicPredictionDecisionDisposition.ABSTAINED,
                prediction=PredictionDisposition.ABSTAINED
                if consistency_blocks_positive
                else PredictionDisposition.INCONCLUSIVE,
                reasons=tuple(sorted(set(merged))),
                goal_id=goal.goal_id,
                assumption_refs=assumption_refs,
                unsupported_facet_ids=unsupported_facet_ids,
                residual_gap_ids=tuple(sorted(residual_ids)),
            )

        eligible: list[LogicHypothesis] = []
        per_hyp_reasons: dict[str, list[str]] = {}
        for hyp in goal_hyps:
            hyp_reasons = self._hypothesis_eligibility(req, goal, hyp)
            per_hyp_reasons[hyp.hypothesis_id] = hyp_reasons
            if not hyp_reasons:
                # Preserve assumptions: hypothesis cannot drop goal assumptions.
                if not set(goal.assumption_refs).issubset(
                    set(hyp.evidence_refs) | set(goal.assumption_refs)
                ):
                    # Assumptions are preserved on the receipt even if the
                    # hypothesis does not restate them; only explicit drops fail.
                    pass
                eligible.append(hyp)

        # Unsupported facets must not be promoted to required consequences.
        for hyp in list(eligible):
            promoted = self._promotes_unsupported(goal, hyp)
            if promoted:
                per_hyp_reasons.setdefault(hyp.hypothesis_id, []).append(
                    LogicPredictionRejectionReason.UNSUPPORTED_FACET_DROPPED.value
                )
                eligible.remove(hyp)

        # Deterministic ordering for uniqueness / tie rules.
        eligible_sorted = sorted(
            eligible,
            key=lambda h: (
                _consequence_key(h),
                h.hypothesis_id,
                h.claimed_consequence_ref,
            ),
        )
        eligible_refs = tuple(_consequence_key(h) for h in eligible_sorted)
        # Distinct consequence identities (tie on same consequence collapses).
        unique_consequences: dict[str, LogicHypothesis] = {}
        for hyp in eligible_sorted:
            key = _consequence_key(hyp)
            if key not in unique_consequences:
                unique_consequences[key] = hyp
            # If two hyps claim the same consequence key, keep the first by
            # deterministic order (they are the same consequence).

        auto_kind = req.automatic_kind
        # If any eligible hyp implies value/construction/placement, enforce
        # uniqueness when the request kind is automatic or inferred.
        inferred_kinds = {
            _automatic_kind_for(h, auto_kind) for h in eligible_sorted
        }
        needs_unique = _requires_uniqueness(auto_kind) or any(
            _requires_uniqueness(k) for k in inferred_kinds
        )

        if needs_unique:
            if len(unique_consequences) == 0:
                reasons.append(
                    LogicPredictionRejectionReason.ZERO_ELIGIBLE.value
                )
                reasons.append(
                    LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value
                )
                # Include per-hypothesis soft failures for diagnostics.
                for codes in per_hyp_reasons.values():
                    reasons.extend(codes)
                return self._terminal(
                    req,
                    disposition=LogicPredictionDecisionDisposition.ABSTAINED,
                    prediction=PredictionDisposition.ABSTAINED,
                    reasons=tuple(sorted(set(reasons))),
                    goal_id=goal.goal_id,
                    assumption_refs=assumption_refs,
                    unsupported_facet_ids=unsupported_facet_ids,
                    residual_gap_ids=tuple(sorted(residual_ids)),
                    eligible_refs=eligible_refs,
                )
            if len(unique_consequences) > 1:
                reasons.append(
                    LogicPredictionRejectionReason.PREDICTION_NON_UNIQUE.value
                )
                reasons.append(
                    LogicPredictionRejectionReason.MULTIPLE_ELIGIBLE.value
                )
                return self._terminal(
                    req,
                    disposition=LogicPredictionDecisionDisposition.ABSTAINED,
                    prediction=PredictionDisposition.ABSTAINED,
                    reasons=tuple(sorted(set(reasons))),
                    goal_id=goal.goal_id,
                    assumption_refs=assumption_refs,
                    unsupported_facet_ids=unsupported_facet_ids,
                    residual_gap_ids=tuple(sorted(residual_ids)),
                    eligible_refs=tuple(sorted(unique_consequences)),
                )
            # Exactly one unique consequence.
            selected = next(iter(unique_consequences.values()))
        else:
            if not eligible_sorted:
                for codes in per_hyp_reasons.values():
                    reasons.extend(codes)
                reasons.append(
                    LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value
                )
                return self._terminal(
                    req,
                    disposition=LogicPredictionDecisionDisposition.ABSTAINED,
                    prediction=PredictionDisposition.INCONCLUSIVE,
                    reasons=tuple(sorted(set(reasons))),
                    goal_id=goal.goal_id,
                    assumption_refs=assumption_refs,
                    unsupported_facet_ids=unsupported_facet_ids,
                    residual_gap_ids=tuple(sorted(residual_ids)),
                )
            # Non-automatic: admit the deterministic first eligible.
            selected = eligible_sorted[0]

        # Final assumption / unsupported preservation on the receipt.
        if not set(assumption_refs).issubset(
            set(assumption_refs)
        ):  # pragma: no cover - identity
            reasons.append(
                LogicPredictionRejectionReason.ASSUMPTION_DROPPED.value
            )

        reconstruction_id = req.reconstruction_id or _binding_reconstruction(
            hammer
        )
        kernel_receipt_id = (
            req.kernel_receipt_id
            or f"kernel:{native.kernel_id}:{native.binding_id}"
        )
        translation_id = (
            req.translation_id
            or hammer.translation_map_id
            or native.logic_ir_obligation_id
        )
        environment_receipt_id = (
            req.environment_receipt_id
            or hammer.environment_lock_id
            or roots.environment_id
        )
        candidate_id = req.candidate_id or selected.hypothesis_id

        # automation_eligible for proved automatic value/construction/placement
        # or any unique reconstructed proved consequence.
        automation = needs_unique or auto_kind is AutomaticConsequenceKind.NONE

        receipt = self._build_receipt(
            req=req,
            goal=goal,
            hypothesis=selected,
            disposition=PredictionDisposition.PROVED,
            proof_status=ProofStatus.KERNEL_VERIFIED,
            source_authority=selected.source_authority,
            automation_eligible=automation,
            reconstruction_id=reconstruction_id,
            kernel_receipt_id=kernel_receipt_id,
            translation_id=translation_id,
            environment_receipt_id=environment_receipt_id,
            candidate_id=candidate_id,
            residual_gap_ids=tuple(sorted(residual_ids)),
            assumption_refs=assumption_refs,
        )

        # Defensive: receipt must not smuggle write authority via metadata.
        receipt_dict = receipt.to_dict()
        if receipt_dict.get("write_authority") or receipt_dict.get(
            "semantic_authority"
        ):
            return self._terminal(
                req,
                disposition=LogicPredictionDecisionDisposition.REJECTED,
                prediction=PredictionDisposition.ERROR,
                reasons=(
                    LogicPredictionRejectionReason.WRITE_AUTHORITY_CLAIMED.value,
                ),
                goal_id=goal.goal_id,
            )

        return LogicPredictionDecision(
            decision_id=self._decision_id(req, "admit", selected.hypothesis_id),
            disposition=LogicPredictionDecisionDisposition.ADMITTED,
            roots=roots,
            goal_id=goal.goal_id,
            hypothesis_id=selected.hypothesis_id,
            reason_codes=(),
            eligible_consequence_refs=eligible_refs
            if eligible_refs
            else (_consequence_key(selected),),
            selected_consequence_ref=_consequence_key(selected),
            receipt=receipt,
            assumption_refs=assumption_refs,
            unsupported_facet_ids=unsupported_facet_ids,
            residual_gap_ids=tuple(sorted(residual_ids)),
            automation_eligible=automation,
            write_authority=False,
            semantic_authority=False,
            prediction_disposition=PredictionDisposition.PROVED,
            metadata={
                "producer_id": self.producer_id,
                "automatic_kind": auto_kind.value,
            },
        )

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _root_problems(
        self, req: LogicPredictionAdmissionRequest
    ) -> list[str]:
        reasons: list[str] = []
        roots = req.roots
        for goal in req.goals:
            if goal.roots != roots:
                reasons.append(LogicPredictionRejectionReason.ROOT_CHANGED.value)
        for hyp in req.hypotheses:
            if hyp.roots != roots:
                reasons.append(LogicPredictionRejectionReason.ROOT_CHANGED.value)
        if req.native_goal_binding.roots != roots:
            reasons.append(LogicPredictionRejectionReason.ROOT_CHANGED.value)
        for cm in req.countermodel_receipts:
            if cm.roots != roots:
                reasons.append(LogicPredictionRejectionReason.ROOT_CHANGED.value)
        for gap in req.residual_gaps:
            if gap.roots != roots:
                reasons.append(LogicPredictionRejectionReason.ROOT_CHANGED.value)
        return reasons

    def _stale_problems(
        self,
        req: LogicPredictionAdmissionRequest,
        goal: ProgramLogicGoal,
    ) -> list[str]:
        reasons: list[str] = []
        roots = req.roots
        checks = (
            ("current_tree_id", roots.tree_id),
            ("current_corpus_id", roots.corpus_id),
            ("current_environment_id", roots.environment_id),
            ("current_toolchain_id", roots.toolchain_id),
            ("current_policy_id", roots.policy_id),
            ("current_translator_id", roots.translator_id),
        )
        for field_name, expected in checks:
            supplied = getattr(req, field_name)
            if supplied and supplied != expected:
                reasons.append(LogicPredictionRejectionReason.STALE_STATE.value)
        # Goal invalidation refs must still include the tree root.
        if roots.tree_id not in goal.invalidation_refs and goal.invalidation_refs:
            # Allow broader invalidation sets; only fail when tree drifted.
            pass
        if req.current_tree_id and roots.tree_id not in goal.invalidation_refs:
            if req.current_tree_id != roots.tree_id:
                reasons.append(LogicPredictionRejectionReason.STALE_STATE.value)
        return reasons

    def _hammer_problems(
        self, req: LogicPredictionAdmissionRequest
    ) -> list[str]:
        reasons: list[str] = []
        hammer = req.hammer_receipt
        if hammer is None:
            return [LogicPredictionRejectionReason.MISSING_HAMMER_RECEIPT.value]
        if hammer.outcome is not HammerCoordinationOutcome.VERIFIED:
            reasons.append(
                LogicPredictionRejectionReason.HAMMER_NOT_VERIFIED.value
            )
        if (
            hammer.conclusiveness
            is not CoordinationConclusiveness.CONCLUSIVE_PROOF
        ):
            reasons.append(
                LogicPredictionRejectionReason.HAMMER_NOT_CONCLUSIVE.value
            )
        if not hammer.kernel_checked or not hammer.proof_success:
            reasons.append(
                LogicPredictionRejectionReason.KERNEL_NOT_ACCEPTED.value
            )
        # Solver-only: verified without kernel is already rejected above.
        provider = dict(hammer.provider_result or {})
        assurance = str(
            provider.get("authoritative_assurance")
            or provider.get("assurance")
            or ""
        ).lower()
        if assurance in {"solver_checked", "solver-only", "solver"}:
            reasons.append(
                LogicPredictionRejectionReason.SOLVER_ONLY_PROOF.value
            )
        if not (req.reconstruction_id or _binding_reconstruction(hammer)):
            reasons.append(
                LogicPredictionRejectionReason.RECONSTRUCTION_MISSING.value
            )
        # Learned selector digest is ranking-only; claiming authority is rejected.
        if (
            hammer.learned_selector_model_digest
            or req.learned_selector_model_digest
        ):
            # Presence alone is fine (ranking); authority flags are elsewhere.
            meta = dict(hammer.metadata or {})
            if meta.get("learned_authority") or meta.get("model_authority"):
                reasons.append(
                    LogicPredictionRejectionReason.LEARNED_AUTHORITY.value
                )
                reasons.append(
                    LogicPredictionRejectionReason.MODEL_AUTHORITY.value
                )
        return reasons

    def _native_problems(
        self, req: LogicPredictionAdmissionRequest
    ) -> list[str]:
        reasons: list[str] = []
        native = req.native_goal_binding
        hammer = req.hammer_receipt
        roots = req.roots
        if native is None:
            return [LogicPredictionRejectionReason.MISSING_NATIVE_BINDING.value]
        if native.disposition is not NativeGoalDisposition.ROUND_TRIP_OK:
            reasons.append(
                LogicPredictionRejectionReason.NATIVE_GOAL_MISMATCH.value
            )
        if (
            native.semantic_round_trip.disposition
            is not NativeGoalDisposition.ROUND_TRIP_OK
        ):
            reasons.append(
                LogicPredictionRejectionReason.NATIVE_GOAL_MISMATCH.value
            )
        if native.environment_id and native.environment_id != roots.environment_id:
            reasons.append(
                LogicPredictionRejectionReason.ENVIRONMENT_MISMATCH.value
            )
        if native.kernel_id in {"", "solver-only", "solver"}:
            reasons.append(
                LogicPredictionRejectionReason.KERNEL_NOT_ACCEPTED.value
            )
            reasons.append(
                LogicPredictionRejectionReason.SOLVER_ONLY_PROOF.value
            )
        expected_translation = (
            req.translation_id or hammer.translation_map_id or ""
        )
        if (
            expected_translation
            and hammer.translation_map_id
            and expected_translation != hammer.translation_map_id
        ):
            reasons.append(
                LogicPredictionRejectionReason.TRANSLATION_MISMATCH.value
            )
        if (
            hammer.native_goal_binding_id
            and hammer.native_goal_binding_id != native.binding_id
        ):
            reasons.append(
                LogicPredictionRejectionReason.NATIVE_GOAL_MISMATCH.value
            )
        if native.unsupported_native_construct_refs:
            # Round-trip OK forbids unsupported constructs at construction;
            # defensive check.
            reasons.append(
                LogicPredictionRejectionReason.NATIVE_GOAL_MISMATCH.value
            )
        return reasons

    def _hypothesis_eligibility(
        self,
        req: LogicPredictionAdmissionRequest,
        goal: ProgramLogicGoal,
        hyp: LogicHypothesis,
    ) -> list[str]:
        reasons: list[str] = []
        if hyp.semantic_authority:
            reasons.append(
                LogicPredictionRejectionReason.SEMANTIC_AUTHORITY_CLAIMED.value
            )
        if hyp.source_authority not in _ADMITTABLE_SOURCE:
            reasons.append(
                LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
            )
        # Learned / vector / model routes cannot grant authority.
        for route in hyp.evidence_route_kinds:
            if route is SourceRouteKind.VECTOR:
                reasons.append(
                    LogicPredictionRejectionReason.VECTOR_AUTHORITY.value
                )
                reasons.append(
                    LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
                )
            if route is SourceRouteKind.LLM:
                reasons.append(
                    LogicPredictionRejectionReason.MODEL_AUTHORITY.value
                )
                reasons.append(
                    LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
                )
            if route in _NON_AUTHORITATIVE_ROUTES and hyp.source_authority in {
                SourceAuthorityClass.AUTHORITATIVE,
            }:
                # Nominating routes claiming authoritative is already rejected
                # by LogicHypothesis construction; defensive double-check.
                reasons.append(
                    LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
                )
        if hyp.proof_status is ProofStatus.SOLVER_CHECKED:
            reasons.append(
                LogicPredictionRejectionReason.SOLVER_ONLY_PROOF.value
            )
        if hyp.proof_status not in {
            ProofStatus.KERNEL_VERIFIED,
            ProofStatus.UNPROVED,
            ProofStatus.CANDIDATE,
        }:
            # Only kernel_verified is positively admittable; candidate/unproved
            # wait for hammer reconstruction on the coordination receipt.
            if hyp.proof_status is ProofStatus.VALIDATED_REFUTED:
                reasons.append(
                    LogicPredictionRejectionReason.VALIDATED_REFUTATION.value
                )
            elif hyp.proof_status not in {
                ProofStatus.KERNEL_VERIFIED,
            }:
                pass
        # For positive admission the coordination already proved reconstruction;
        # hypothesis may still be CANDIDATE/UNPROVED at nomination time if the
        # hammer receipt supplies kernel verification.  Require either.
        hammer_verified = (
            req.hammer_receipt.outcome is HammerCoordinationOutcome.VERIFIED
            and req.hammer_receipt.kernel_checked
        )
        if (
            hyp.proof_status is not ProofStatus.KERNEL_VERIFIED
            and not hammer_verified
        ):
            reasons.append(
                LogicPredictionRejectionReason.HYPOTHESIS_NOT_KERNEL_VERIFIED.value
            )
        if hyp.disposition is HypothesisDisposition.VALIDATED_REFUTED:
            reasons.append(
                LogicPredictionRejectionReason.VALIDATED_REFUTATION.value
            )
        if hyp.disposition in {
            HypothesisDisposition.STALE,
            HypothesisDisposition.ABSTAINED,
            HypothesisDisposition.UNSUPPORTED,
        }:
            reasons.append(LogicPredictionRejectionReason.STALE_STATE.value)
        if not hyp.selected_premise_ids and hyp.source_authority in _ADMITTABLE_SOURCE:
            # Authoritative admission requires independent premises.
            reasons.append(
                LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
            )
        # Premises must be subset of native binding premises when both listed.
        native_premises = set(req.native_goal_binding.premise_ids)
        if hyp.selected_premise_ids and native_premises:
            if not set(hyp.selected_premise_ids).issubset(native_premises):
                # Extra premises not in the reconstructed binding are non-current.
                reasons.append(
                    LogicPredictionRejectionReason.NON_AUTHORITATIVE_PREMISES.value
                )
        # Automatic kind alignment: value hyp must expose value_ref, etc.
        kind = _automatic_kind_for(hyp, req.automatic_kind)
        if kind is AutomaticConsequenceKind.VALUE and not hyp.value_ref:
            reasons.append(
                LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value
            )
        if (
            kind is AutomaticConsequenceKind.CONSTRUCTION
            and not hyp.construction_ref
        ):
            reasons.append(
                LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value
            )
        if kind is AutomaticConsequenceKind.PLACEMENT and not hyp.placement_ref:
            reasons.append(
                LogicPredictionRejectionReason.NO_ELIGIBLE_CONSEQUENCE.value
            )
        return list(dict.fromkeys(reasons))

    def _promotes_unsupported(
        self, goal: ProgramLogicGoal, hyp: LogicHypothesis
    ) -> bool:
        unsupported_ids = {facet.facet_id for facet in goal.unsupported_facets}
        if not unsupported_ids:
            return False
        # Claimed consequence must not equal an unsupported facet id.
        claimed = {
            hyp.claimed_consequence_ref,
            hyp.value_ref,
            hyp.construction_ref,
            hyp.placement_ref,
        }
        claimed.discard("")
        if claimed & unsupported_ids:
            return True
        # Unsupported flags on the hypothesis naming those facets is fine
        # (preservation); promoting them into derived refs is not.
        return False

    def _refutation_target(
        self,
        hypotheses: Sequence[LogicHypothesis],
        rejector: CountermodelValidationReceipt,
    ) -> LogicHypothesis:
        # Prefer hyp whose counterexample target matches, else deterministic first.
        ordered = sorted(hypotheses, key=lambda h: h.hypothesis_id)
        for hyp in ordered:
            if (
                hyp.counterexample_target_ref
                and hyp.counterexample_target_ref
                in {
                    rejector.originating_logic_ir_id,
                    rejector.solver_countermodel_id,
                    rejector.receipt_id,
                }
            ):
                return hyp
        for hyp in ordered:
            if hyp.disposition is HypothesisDisposition.VALIDATED_REFUTED:
                return hyp
        return ordered[0]

    def _build_receipt(
        self,
        *,
        req: LogicPredictionAdmissionRequest,
        goal: ProgramLogicGoal,
        hypothesis: LogicHypothesis,
        disposition: PredictionDisposition,
        proof_status: ProofStatus,
        source_authority: SourceAuthorityClass,
        automation_eligible: bool,
        countermodel_validation_id: str = "",
        reconstruction_id: str = "",
        kernel_receipt_id: str = "",
        translation_id: str = "",
        environment_receipt_id: str = "",
        candidate_id: str = "",
        residual_gap_ids: tuple[str, ...] = (),
        assumption_refs: tuple[str, ...] = (),
    ) -> LogicPredictionReceipt:
        roots = req.roots
        hammer = req.hammer_receipt
        derived_clause = ""
        derived_value = ""
        derived_placement = ""
        if hypothesis.value_ref:
            derived_value = hypothesis.value_ref
        if hypothesis.construction_ref:
            # Construction routes surface as derived clause/construction.
            derived_clause = hypothesis.construction_ref
        if hypothesis.placement_ref:
            derived_placement = hypothesis.placement_ref
        if not derived_clause and not derived_value and not derived_placement:
            derived_clause = hypothesis.claimed_consequence_ref

        # Preserve goal assumptions explicitly on the receipt.
        preserved_assumptions = tuple(
            sorted(set(assumption_refs) | set(goal.assumption_refs))
        )

        invalidation = tuple(
            sorted(
                {
                    *goal.invalidation_refs,
                    *hypothesis.invalidation_refs,
                    roots.tree_id,
                    roots.corpus_id,
                    roots.environment_id,
                    roots.toolchain_id,
                    roots.policy_id,
                }
            )
        )

        return LogicPredictionReceipt(
            roots=roots,
            receipt_id=self._receipt_id(req, hypothesis.hypothesis_id, disposition),
            goal_id=goal.goal_id,
            hypothesis_id=hypothesis.hypothesis_id,
            tactician_plan_id=req.tactician_plan_id,
            corpus_id=roots.corpus_id,
            disposition=disposition,
            hammer_request_id=hammer.request_id,
            translation_id=translation_id
            or req.translation_id
            or hammer.translation_map_id,
            candidate_id=candidate_id or hypothesis.hypothesis_id,
            reconstruction_id=reconstruction_id or req.reconstruction_id,
            kernel_receipt_id=kernel_receipt_id or req.kernel_receipt_id,
            environment_receipt_id=environment_receipt_id
            or req.environment_receipt_id
            or hammer.environment_lock_id,
            countermodel_validation_id=countermodel_validation_id,
            derived_clause_ref=derived_clause,
            derived_value_ref=derived_value,
            derived_placement_ref=derived_placement,
            assumption_refs=preserved_assumptions,
            counterexample_refs=tuple(
                sorted(
                    {
                        cm.receipt_id
                        for cm in req.countermodel_receipts
                        if _countermodel_may_reject(cm)
                    }
                )
            ),
            residual_gap_ids=residual_gap_ids,
            source_authority=source_authority,
            proof_status=proof_status,
            automation_eligible=automation_eligible,
            invalidation_refs=invalidation,
        )

    def _terminal(
        self,
        req: LogicPredictionAdmissionRequest,
        *,
        disposition: LogicPredictionDecisionDisposition,
        prediction: PredictionDisposition,
        reasons: Sequence[str],
        goal_id: str = "",
        hypothesis_id: str = "",
        assumption_refs: tuple[str, ...] = (),
        unsupported_facet_ids: tuple[str, ...] = (),
        residual_gap_ids: tuple[str, ...] = (),
        eligible_refs: tuple[str, ...] = (),
        detail: str = "",
    ) -> LogicPredictionDecision:
        goal_id = goal_id or (
            req.goals[0].goal_id if req.goals else "goal:unknown"
        )
        meta: dict[str, Any] = {"producer_id": self.producer_id}
        if detail:
            meta["detail"] = detail
        # Preserve assumptions / unsupported even on abstention.
        if not assumption_refs and req.goals:
            assumption_refs = tuple(sorted(req.goals[0].assumption_refs))
        if not unsupported_facet_ids and req.goals:
            unsupported_facet_ids = tuple(
                sorted(
                    {facet.facet_id for facet in req.goals[0].unsupported_facets}
                )
            )
        return LogicPredictionDecision(
            decision_id=self._decision_id(req, disposition.value, hypothesis_id or "none"),
            disposition=disposition,
            roots=req.roots,
            goal_id=goal_id,
            hypothesis_id=hypothesis_id,
            reason_codes=tuple(sorted(set(reasons))),
            eligible_consequence_refs=eligible_refs,
            selected_consequence_ref="",
            receipt=None,
            assumption_refs=assumption_refs,
            unsupported_facet_ids=unsupported_facet_ids,
            residual_gap_ids=residual_gap_ids,
            automation_eligible=False,
            write_authority=False,
            semantic_authority=False,
            prediction_disposition=prediction,
            metadata=meta,
        )

    def _prediction_for_reasons(
        self, reasons: Sequence[str]
    ) -> PredictionDisposition:
        codes = set(reasons)
        if LogicPredictionRejectionReason.STALE_STATE.value in codes:
            return PredictionDisposition.STALE
        if LogicPredictionRejectionReason.GOAL_UNSUPPORTED.value in codes:
            return PredictionDisposition.UNSUPPORTED
        if LogicPredictionRejectionReason.IDENTITY_MISMATCH.value in codes:
            return PredictionDisposition.ERROR
        if LogicPredictionRejectionReason.WRITE_AUTHORITY_CLAIMED.value in codes:
            return PredictionDisposition.ERROR
        return PredictionDisposition.ABSTAINED

    def _decision_id(
        self,
        req: LogicPredictionAdmissionRequest,
        kind: str,
        hypothesis_id: str,
    ) -> str:
        return content_identity(
            {
                "schema": "logic-prediction-decision-id@1",
                "producer_id": self.producer_id,
                "request_id": req.content_id,
                "kind": kind,
                "hypothesis_id": hypothesis_id,
                "tree_id": req.roots.tree_id,
                "corpus_id": req.roots.corpus_id,
            }
        )

    def _receipt_id(
        self,
        req: LogicPredictionAdmissionRequest,
        hypothesis_id: str,
        disposition: PredictionDisposition,
    ) -> str:
        return content_identity(
            {
                "schema": "logic-prediction-receipt-id@1",
                "producer_id": self.producer_id,
                "request_id": req.content_id,
                "hypothesis_id": hypothesis_id,
                "disposition": disposition.value,
                "tree_id": req.roots.tree_id,
            }
        )


def _binding_reconstruction(hammer: HammerCoordinationReceipt) -> str:
    binding = hammer.receipt_binding or {}
    if isinstance(binding, Mapping):
        value = binding.get("reconstruction_id") or binding.get(
            "native_goal_binding_id"
        )
        if isinstance(value, str) and value.strip():
            return value.strip()
    if hammer.native_goal_binding_id:
        return f"reconstruction:{hammer.native_goal_binding_id}"
    return ""


def create_logic_prediction_admission(
    **kwargs: Any,
) -> LogicPredictionAdmission:
    """Factory matching other LPR module constructors."""

    return LogicPredictionAdmission(**kwargs)


__all__ = (
    "ADMISSION_PRODUCER_ID",
    "AutomaticConsequenceKind",
    "LOGIC_PREDICTION_ADMISSION_INTERFACE",
    "LOGIC_PREDICTION_ADMISSION_VERSION",
    "LOGIC_PREDICTION_DECISION_SCHEMA",
    "LogicPredictionAdmission",
    "LogicPredictionAdmissionError",
    "LogicPredictionAdmissionRequest",
    "LogicPredictionDecision",
    "LogicPredictionDecisionDisposition",
    "LogicPredictionRejectionReason",
    "create_logic_prediction_admission",
)
