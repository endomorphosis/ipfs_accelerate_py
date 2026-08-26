"""Fail-closed expert promotion and exact authorized rollback.

Eligibility is deliberately separate from publication: a positive evaluation
does not mutate a route.  Only this state owner can CAS its in-process head,
using an authorization verified by a control-plane caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, Final, Mapping

from .contracts import (
    ResidualIntelligenceError,
    RiskClass,
    TrainingAvailability,
    bounded_int,
    optional_text,
    required_text,
    text_tuple,
)
from .rights import TrainingCorpusAdmission

PROMOTION_EVIDENCE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-promotion-evidence@1"
PROMOTION_AUTHORIZATION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-promotion-authorization@1"
PROMOTION_HEAD_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-promotion-head@1"
PROMOTION_DECISION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-promotion-decision@1"
ROLLBACK_RECEIPT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-rollback-receipt@1"
MAX_PPM: Final = 1_000_000
MAX_COUNT: Final = 100_000_000
MAX_COST_UNITS: Final = 10**15

HARD_GATES: Final[tuple[str, ...]] = (
    "rights", "lineage", "leakage", "privacy", "safety", "quality",
    "efficiency", "autonomy", "amortization",
)
EFFICIENCY_BOUNDS: Final[dict[str, int]] = {
    "remote_call_reduction": 45,
    "input_token_reduction": 35,
    "output_token_reduction": 60,
    "strong_model_call_reduction": 50,
    "median_latency_reduction": 30,
}
AUTONOMY_BOUNDS: Final[dict[str, int]] = {
    "classification_ranking_coverage": 70,
    "typed_procedure_hole_coverage": 40,
    "human_intervention_reduction": 25,
}
AMORTIZATION_FIELDS: Final[frozenset[str]] = frozenset({
    "training_evaluation_cost", "per_use_saving", "expected_break_even_uses",
    "observed_uses", "observed_savings",
})


class PromotionAction(str, Enum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"


def _exact_bool_map(value: Any, name: str, keys: tuple[str, ...]) -> Mapping[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ResidualIntelligenceError(f"{name} must bind exactly: {', '.join(keys)}")
    result = {key: value[key] for key in keys}
    if any(type(item) is not bool for item in result.values()):
        raise ResidualIntelligenceError(f"{name} values must be boolean")
    return MappingProxyType(result)


def _exact_int_map(
    value: Any, name: str, bounds: Mapping[str, int], maximum: int
) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(bounds):
        raise ResidualIntelligenceError(f"{name} must bind exactly: {', '.join(bounds)}")
    return MappingProxyType({
        key: bounded_int(value[key], f"{name}.{key}", minimum=0, maximum=maximum)
        for key in bounds
    })


def _exact_gate_evidence(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(HARD_GATES):
        raise ResidualIntelligenceError(f"gate_evidence must bind exactly: {', '.join(HARD_GATES)}")
    result = {
        key: required_text(value[key], f"gate_evidence.{key}") for key in HARD_GATES
    }
    if len(set(result.values())) != len(result):
        raise ResidualIntelligenceError("each hard gate requires independent evidence")
    return MappingProxyType(result)


def _validated_amortization(value: Any) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != AMORTIZATION_FIELDS:
        raise ResidualIntelligenceError("amortization must bind complete cost, saving, and break-even denominators")
    result = {
        key: bounded_int(value[key], f"amortization.{key}", minimum=0, maximum=MAX_COST_UNITS)
        for key in AMORTIZATION_FIELDS
    }
    cost, saving = result["training_evaluation_cost"], result["per_use_saving"]
    if cost and not saving:
        raise ResidualIntelligenceError("amortization requires a positive per-use saving")
    expected = 0 if not cost else (cost + saving - 1) // saving
    if result["expected_break_even_uses"] != expected:
        raise ResidualIntelligenceError("expected_break_even_uses has an incorrect denominator")
    if result["observed_savings"] != result["observed_uses"] * saving:
        raise ResidualIntelligenceError("observed_savings must equal observed uses times per-use saving")
    if result["observed_uses"] < expected or result["observed_savings"] < cost:
        raise ResidualIntelligenceError("amortization has not reached declared break-even")
    return MappingProxyType(result)


@dataclass(frozen=True)
class PromotionEvidence:
    """Current evidence for one expert, including concrete admitted-data ties."""

    gates: Mapping[str, bool]
    gate_evidence: Mapping[str, str]
    precision_ppm: int
    accepted_count: int
    true_accepted_count: int
    critical_false_accepts: int
    efficiency: Mapping[str, int]
    autonomy: Mapping[str, int]
    amortization: Mapping[str, int]
    risk: RiskClass
    cas_identity: str
    expert_identity: str
    admission: TrainingCorpusAdmission
    admission_id: str
    split_root: str
    leakage_audit_id: str
    schema: str = PROMOTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROMOTION_EVIDENCE_SCHEMA:
            raise ResidualIntelligenceError("unsupported promotion evidence schema")
        object.__setattr__(self, "gates", _exact_bool_map(self.gates, "gates", HARD_GATES))
        object.__setattr__(self, "gate_evidence", _exact_gate_evidence(self.gate_evidence))
        accepted = bounded_int(self.accepted_count, "accepted_count", minimum=1, maximum=MAX_COUNT)
        true_accepted = bounded_int(self.true_accepted_count, "true_accepted_count", minimum=0, maximum=accepted)
        critical = bounded_int(self.critical_false_accepts, "critical_false_accepts", minimum=0, maximum=accepted)
        precision = bounded_int(self.precision_ppm, "precision_ppm", minimum=0, maximum=MAX_PPM)
        if critical > accepted - true_accepted:
            raise ResidualIntelligenceError("critical false accepts cannot exceed false accepts")
        if precision != (true_accepted * MAX_PPM) // accepted:
            raise ResidualIntelligenceError("precision_ppm does not match exact accept denominators")
        object.__setattr__(self, "accepted_count", accepted)
        object.__setattr__(self, "true_accepted_count", true_accepted)
        object.__setattr__(self, "critical_false_accepts", critical)
        object.__setattr__(self, "precision_ppm", precision)
        object.__setattr__(self, "efficiency", _exact_int_map(self.efficiency, "efficiency", EFFICIENCY_BOUNDS, 100))
        object.__setattr__(self, "autonomy", _exact_int_map(self.autonomy, "autonomy", AUTONOMY_BOUNDS, 100))
        object.__setattr__(self, "amortization", _validated_amortization(self.amortization))
        object.__setattr__(self, "risk", RiskClass(self.risk))
        object.__setattr__(self, "cas_identity", required_text(self.cas_identity, "cas_identity"))
        object.__setattr__(self, "expert_identity", required_text(self.expert_identity, "expert_identity"))
        if not isinstance(self.admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("promotion evidence requires a typed corpus admission")
        if self.admission.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError("promotion evidence requires an admitted corpus")
        for field in ("admission_id", "split_root", "leakage_audit_id"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.admission_id != self.admission.admission_id:
            raise ResidualIntelligenceError("promotion evidence admission_id mismatch")
        if self.split_root != self.admission.split_root:
            raise ResidualIntelligenceError("promotion evidence split_root mismatch")
        if self.leakage_audit_id != self.admission.leakage_audit.audit_id:
            raise ResidualIntelligenceError("promotion evidence leakage audit identity mismatch")
        if not self.admission.leakage_audit.passed:
            raise ResidualIntelligenceError("promotion evidence requires a passing leakage audit")


@dataclass(frozen=True)
class PromotionAuthorization:
    """A control-plane authorization already verified by a trusted owner."""

    authority_identity: str
    action: PromotionAction
    subject_identity: str
    expected_current_identity: str
    expected_generation: int
    cas_identity: str
    schema: str = PROMOTION_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROMOTION_AUTHORIZATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported promotion authorization schema")
        object.__setattr__(self, "authority_identity", required_text(self.authority_identity, "authority_identity"))
        object.__setattr__(self, "action", PromotionAction(self.action))
        object.__setattr__(self, "subject_identity", required_text(self.subject_identity, "subject_identity"))
        object.__setattr__(self, "expected_current_identity", optional_text(self.expected_current_identity, "expected_current_identity"))
        object.__setattr__(self, "expected_generation", bounded_int(self.expected_generation, "expected_generation", minimum=0, maximum=MAX_COUNT))
        object.__setattr__(self, "cas_identity", required_text(self.cas_identity, "cas_identity"))
        if self.authority_identity == self.subject_identity:
            raise ResidualIntelligenceError("a candidate cannot authorize its own promotion")


@dataclass(frozen=True)
class PromotionHead:
    current_identity: str = ""
    generation: int = 0
    rollback_identity: str = ""
    schema: str = PROMOTION_HEAD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROMOTION_HEAD_SCHEMA:
            raise ResidualIntelligenceError("unsupported promotion head schema")
        object.__setattr__(self, "current_identity", optional_text(self.current_identity, "current_identity"))
        object.__setattr__(self, "rollback_identity", optional_text(self.rollback_identity, "rollback_identity"))
        object.__setattr__(self, "generation", bounded_int(self.generation, "generation", minimum=0, maximum=MAX_COUNT))
        if self.rollback_identity and not self.current_identity:
            raise ResidualIntelligenceError("an empty promotion head cannot have a rollback target")
        if self.current_identity and self.current_identity == self.rollback_identity:
            raise ResidualIntelligenceError("rollback target must differ from current identity")


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason_codes: tuple[str, ...]
    cas_identity: str
    current_identity: str = ""
    previous_identity: str = ""
    generation: int = 0
    authorization_identity: str = ""
    schema: str = PROMOTION_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.promoted) is not bool or self.schema != PROMOTION_DECISION_SCHEMA:
            raise ResidualIntelligenceError("invalid promotion decision")
        object.__setattr__(self, "reason_codes", text_tuple(self.reason_codes, "reason_codes"))
        object.__setattr__(self, "cas_identity", required_text(self.cas_identity, "cas_identity"))
        for field in ("current_identity", "previous_identity", "authorization_identity"):
            object.__setattr__(self, field, optional_text(getattr(self, field), field))
        object.__setattr__(self, "generation", bounded_int(self.generation, "generation", minimum=0, maximum=MAX_COUNT))


@dataclass(frozen=True)
class ExpertRollbackReceipt:
    from_identity: str
    to_identity: str
    cas_identity: str
    rolled_back: bool = False
    reason_codes: tuple[str, ...] = ()
    generation: int = 0
    authorization_identity: str = ""
    schema: str = ROLLBACK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field in ("from_identity", "to_identity", "cas_identity"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.from_identity == self.to_identity or type(self.rolled_back) is not bool:
            raise ResidualIntelligenceError("invalid rollback receipt")
        object.__setattr__(self, "reason_codes", text_tuple(self.reason_codes, "reason_codes"))
        object.__setattr__(self, "generation", bounded_int(self.generation, "generation", minimum=0, maximum=MAX_COUNT))
        object.__setattr__(self, "authorization_identity", optional_text(self.authorization_identity, "authorization_identity"))
        if self.schema != ROLLBACK_RECEIPT_SCHEMA:
            raise ResidualIntelligenceError("unsupported rollback receipt schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "from_identity": self.from_identity,
            "to_identity": self.to_identity, "cas_identity": self.cas_identity,
            "promoted": False, "rolled_back": self.rolled_back,
            "reason_codes": list(self.reason_codes), "generation": self.generation,
            "authorization_identity": self.authorization_identity,
        }


class ExpertPromotionGate:
    """Lock-protected owner for a promotion route head."""

    def __init__(self, *, initial_identity: str = "", trusted_authorities: tuple[str, ...] = ()) -> None:
        self._head = PromotionHead(current_identity=initial_identity)
        self._trusted_authorities = frozenset(text_tuple(trusted_authorities, "trusted_authorities"))
        self._lock = RLock()

    def current_head(self) -> PromotionHead:
        with self._lock:
            return self._head

    def decide(self, evidence: PromotionEvidence) -> PromotionDecision:
        """Evaluate every gate without publishing a route."""
        if not isinstance(evidence, PromotionEvidence):
            raise ResidualIntelligenceError("promotion requires typed PromotionEvidence")
        reasons = [f"{name}_gate_failed" for name in HARD_GATES if not evidence.gates[name]]
        if evidence.precision_ppm < 990_000:
            reasons.append("precision_below_99")
        if evidence.critical_false_accepts:
            reasons.append("critical_false_accept")
        reasons.extend(
            f"efficiency_{key}" for key, bound in EFFICIENCY_BOUNDS.items()
            if evidence.efficiency[key] < bound
        )
        reasons.extend(
            f"autonomy_{key}" for key, bound in AUTONOMY_BOUNDS.items()
            if evidence.autonomy[key] < bound
        )
        if evidence.risk in {RiskClass.R4, RiskClass.R5}:
            reasons.append("r4_r5_proposal_only")
        return PromotionDecision(False, tuple(reasons), evidence.cas_identity)

    def _authorization_reason(
        self, authorization: PromotionAuthorization | None, action: PromotionAction,
        subject: str, head: PromotionHead, cas_identity: str,
    ) -> str:
        if not isinstance(authorization, PromotionAuthorization):
            return "authorization_missing"
        if authorization.action is not action:
            return "authorization_action_mismatch"
        if authorization.authority_identity not in self._trusted_authorities:
            return "authorization_untrusted"
        if authorization.subject_identity != subject:
            return "authorization_subject_mismatch"
        if authorization.expected_current_identity != head.current_identity:
            return "cas_identity_mismatch"
        if authorization.expected_generation != head.generation:
            return "cas_generation_mismatch"
        if authorization.cas_identity != cas_identity:
            return "cas_authorization_mismatch"
        return ""

    def promote(self, evidence: PromotionEvidence, *, authorization: PromotionAuthorization | None) -> PromotionDecision:
        """Publish an eligible R0--R3 expert through authorized exact CAS."""
        eligibility = self.decide(evidence)
        if eligibility.reason_codes:
            return eligibility
        with self._lock:
            head = self._head
            if not head.current_identity:
                return PromotionDecision(
                    False,
                    ("promotion_missing_current_route",),
                    evidence.cas_identity,
                    generation=head.generation,
                )
            reason = self._authorization_reason(authorization, PromotionAction.PROMOTE, evidence.expert_identity, head, evidence.cas_identity)
            if reason:
                return PromotionDecision(False, (reason,), evidence.cas_identity, head.current_identity, generation=head.generation)
            if evidence.expert_identity == head.current_identity:
                return PromotionDecision(False, ("candidate_already_current",), evidence.cas_identity, head.current_identity, generation=head.generation, authorization_identity=authorization.authority_identity)
            self._head = PromotionHead(evidence.expert_identity, head.generation + 1, head.current_identity)
            return PromotionDecision(True, (), evidence.cas_identity, self._head.current_identity, head.current_identity, self._head.generation, authorization.authority_identity)

    def rollback(
        self, *, from_identity: str, to_identity: str, cas_identity: str,
        authorization: PromotionAuthorization | None = None, fence_drained: bool = False,
    ) -> ExpertRollbackReceipt:
        """Restore only the recorded prior route after its fence has drained."""
        from_identity, to_identity, cas_identity = (
            required_text(from_identity, "from_identity"), required_text(to_identity, "to_identity"),
            required_text(cas_identity, "cas_identity"),
        )
        with self._lock:
            head = self._head
            if from_identity != head.current_identity:
                reason = "rollback_from_identity_mismatch"
            elif not head.rollback_identity or to_identity != head.rollback_identity:
                reason = "rollback_target_not_exact_prior"
            elif type(fence_drained) is not bool or not fence_drained:
                reason = "fenced_work_not_drained"
            else:
                reason = self._authorization_reason(authorization, PromotionAction.ROLLBACK, from_identity, head, cas_identity)
            if reason:
                return ExpertRollbackReceipt(from_identity, to_identity, cas_identity, reason_codes=(reason,), generation=head.generation)
            self._head = PromotionHead(to_identity, head.generation + 1, from_identity)
            return ExpertRollbackReceipt(from_identity, to_identity, cas_identity, True, generation=self._head.generation, authorization_identity=authorization.authority_identity)


__all__ = (
    "AMORTIZATION_FIELDS", "AUTONOMY_BOUNDS", "EFFICIENCY_BOUNDS", "HARD_GATES",
    "PROMOTION_AUTHORIZATION_SCHEMA", "PROMOTION_EVIDENCE_SCHEMA", "ExpertPromotionGate",
    "ExpertRollbackReceipt", "PromotionAction", "PromotionAuthorization", "PromotionDecision",
    "PromotionEvidence", "PromotionHead",
)
