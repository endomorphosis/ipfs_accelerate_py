"""Selective prediction and closed residual abstention dispositions.

Every learned residual output is one of six closed dispositions.  Accept
requires current group-keyed evidence and that group's own threshold.  R4/R5
remain proposal-tier.  A model cannot lower or otherwise rewrite its
threshold.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .calibration import (
    MAX_CALIBRATION_GROUPS,
    MAX_SCORE_PPM,
    CalibrationEvidence,
    CalibrationGroup,
    CalibrationThresholdBinding,
    ThresholdChangeOrigin,
    apply_threshold_cas,
    reject_global_threshold_fields,
    rollback_threshold_binding,
)
from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    RiskClass,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)

ABSTENTION_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-abstention-decision@1"
)
SELECTIVE_PREDICTION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-selective-prediction-policy@1"
)
CLOSED_DISPOSITIONS: Final[tuple[ExpertDisposition, ...]] = (
    ExpertDisposition.ACCEPT,
    ExpertDisposition.ABSTAIN,
    ExpertDisposition.REJECT_INPUT,
    ExpertDisposition.OUT_OF_DISTRIBUTION,
    ExpertDisposition.CAPABILITY_UNAVAILABLE,
    ExpertDisposition.VALIDATION_REQUIRED,
)
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})
REASON_REJECT_INPUT: Final = "reject_input"
REASON_CAPABILITY_UNAVAILABLE: Final = "capability_unavailable"
REASON_MISSING_CALIBRATION_GROUP: Final = "missing_calibration_group"
REASON_OOD: Final = "out_of_distribution"
REASON_CURRENT_EVIDENCE: Final = "current_evidence_required"
REASON_CRITICAL_BOUNDARY: Final = "critical_boundary_abstention"
REASON_CRITICAL_FALSE_ACCEPT: Final = "critical_false_accept"
REASON_NO_GROUP_THRESHOLD: Final = "no_group_threshold"
REASON_BELOW_GROUP_THRESHOLD: Final = "below_group_threshold"
REASON_VALIDATION_REQUIRED: Final = "VALIDATION_REQUIRED"
REASON_R4_R5_PROPOSAL: Final = "r4_r5_proposal_tier"
REASON_ACCEPT: Final = "group_threshold_met"


def _score_ppm(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_SCORE_PPM)


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class SelectivePredictionRequest:
    """One bounded expert score presented to a grouped selective-prediction policy."""

    group: CalibrationGroup
    score_ppm: int
    input_valid: bool = True
    capability_available: bool = True
    out_of_distribution: bool = False
    validation_satisfied: bool = True
    critical_boundary: bool = False
    model_proposed_threshold_ppm: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, CalibrationGroup):
            raise ResidualIntelligenceError(
                "selective prediction requires a typed calibration group"
            )
        object.__setattr__(self, "score_ppm", _score_ppm(self.score_ppm, "score_ppm"))
        for field in (
            "input_valid",
            "capability_available",
            "out_of_distribution",
            "validation_satisfied",
            "critical_boundary",
        ):
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        if self.model_proposed_threshold_ppm is not None:
            raise ResidualIntelligenceError(
                "self-threshold-rejection: a model cannot change its threshold"
            )


@dataclass(frozen=True)
class AbstentionDecision:
    """Closed residual disposition for one group-keyed prediction."""

    disposition: ExpertDisposition
    group_key: str
    risk_class: RiskClass
    score_ppm: int
    group_threshold_bound: bool
    group_threshold_ppm: int
    reason_codes: tuple[str, ...]
    evidence_id: str = ""
    candidate_only: bool = True
    schema: str = ABSTENTION_DECISION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "decision_id",
            "disposition",
            "group_key",
            "risk_class",
            "score_ppm",
            "group_threshold_bound",
            "group_threshold_ppm",
            "reason_codes",
            "evidence_id",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != ABSTENTION_DECISION_SCHEMA:
            raise ResidualIntelligenceError("unsupported abstention decision schema")
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        if self.disposition not in CLOSED_DISPOSITIONS:
            raise ResidualIntelligenceError("disposition is outside the closed six-value set")
        object.__setattr__(self, "group_key", required_text(self.group_key, "group_key"))
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        object.__setattr__(self, "score_ppm", _score_ppm(self.score_ppm, "score_ppm"))
        object.__setattr__(
            self,
            "group_threshold_bound",
            _require_bool(self.group_threshold_bound, "group_threshold_bound"),
        )
        object.__setattr__(
            self,
            "group_threshold_ppm",
            _score_ppm(self.group_threshold_ppm, "group_threshold_ppm"),
        )
        if not self.group_threshold_bound and self.group_threshold_ppm != 0:
            raise ResidualIntelligenceError(
                "unbound group threshold must not carry a numeric fallback"
            )
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", allow_empty=False, max_items=16),
        )
        object.__setattr__(
            self,
            "evidence_id",
            "" if self.evidence_id in (None, "") else required_text(self.evidence_id, "evidence_id"),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.disposition is ExpertDisposition.ACCEPT:
            if self.risk_class in PROPOSAL_RISKS:
                raise ResidualIntelligenceError(
                    "R4/R5 outputs remain proposal-tier and cannot ACCEPT"
                )
            if not self.group_threshold_bound:
                raise ResidualIntelligenceError("ACCEPT requires the group's own threshold")
            if self.score_ppm < self.group_threshold_ppm:
                raise ResidualIntelligenceError("ACCEPT requires score at or above the group threshold")
            if not self.evidence_id:
                raise ResidualIntelligenceError("ACCEPT requires current evidence")

    @property
    def decision_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def abstained(self) -> bool:
        return self.disposition not in {
            ExpertDisposition.ACCEPT,
            ExpertDisposition.VALIDATION_REQUIRED,
        }

    @property
    def proposal_tier(self) -> bool:
        return self.risk_class in PROPOSAL_RISKS or (
            self.disposition is ExpertDisposition.VALIDATION_REQUIRED
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "disposition": self.disposition.value,
            "group_key": self.group_key,
            "risk_class": self.risk_class.value,
            "score_ppm": self.score_ppm,
            "group_threshold_bound": self.group_threshold_bound,
            "group_threshold_ppm": self.group_threshold_ppm,
            "reason_codes": list(self.reason_codes),
            "evidence_id": self.evidence_id,
            "candidate_only": True,
        }
        if include_id:
            result["decision_id"] = self.decision_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AbstentionDecision:
        reject_global_threshold_fields(payload, noun="abstention decision")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"decision_id", "evidence_id"},
            noun="abstention decision",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            disposition=ExpertDisposition(str(payload.get("disposition") or "")),
            group_key=str(payload.get("group_key") or ""),
            risk_class=RiskClass(str(payload.get("risk_class") or "")),
            score_ppm=payload.get("score_ppm"),
            group_threshold_bound=payload.get("group_threshold_bound"),
            group_threshold_ppm=payload.get("group_threshold_ppm"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence_id=str(payload.get("evidence_id") or ""),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("decision_id") or "")
        if claimed and claimed != result.decision_id:
            raise ResidualIntelligenceError("abstention decision identity mismatch")
        return result


@dataclass(frozen=True)
class SelectivePredictionPolicy:
    """Group-keyed selective prediction; never a global accept threshold."""

    current_admission_id: str
    current_split_root: str
    current_holdout_root: str
    current_evaluation_identity: str
    evidence: tuple[CalibrationEvidence, ...]
    bindings: tuple[CalibrationThresholdBinding, ...] = ()
    ood_signals_binding: bool = False
    schema: str = SELECTIVE_PREDICTION_POLICY_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "policy_id",
            "current_admission_id",
            "current_split_root",
            "current_holdout_root",
            "current_evaluation_identity",
            "evidence",
            "bindings",
            "ood_signals_binding",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != SELECTIVE_PREDICTION_POLICY_SCHEMA:
            raise ResidualIntelligenceError("unsupported selective prediction policy schema")
        for field in (
            "current_admission_id",
            "current_split_root",
            "current_holdout_root",
            "current_evaluation_identity",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        object.__setattr__(
            self,
            "ood_signals_binding",
            _require_bool(self.ood_signals_binding, "ood_signals_binding"),
        )
        evidence = tuple(self.evidence)
        if len(evidence) > MAX_CALIBRATION_GROUPS:
            raise ResidualIntelligenceError("selective prediction exceeds calibration group bound")
        if any(not isinstance(item, CalibrationEvidence) for item in evidence):
            raise ResidualIntelligenceError("policy evidence must be typed CalibrationEvidence")
        evidence_keys = [item.group.group_key for item in evidence]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ResidualIntelligenceError("calibration evidence is not isolated by group-key")
        object.__setattr__(self, "evidence", evidence)
        bindings = tuple(self.bindings)
        if len(bindings) > MAX_CALIBRATION_GROUPS:
            raise ResidualIntelligenceError("selective prediction exceeds calibration group bound")
        if any(not isinstance(item, CalibrationThresholdBinding) for item in bindings):
            raise ResidualIntelligenceError(
                "policy bindings must be typed CalibrationThresholdBinding"
            )
        binding_keys = [item.group_key for item in bindings]
        if len(set(binding_keys)) != len(binding_keys):
            raise ResidualIntelligenceError("threshold bindings are not isolated by group-key")
        evidence_by_key = {item.group.group_key: item for item in evidence}
        for binding in bindings:
            if binding.group_key not in evidence_by_key:
                raise ResidualIntelligenceError(
                    "threshold binding has no calibration evidence for its group-key"
                )
            if binding.evidence_id != evidence_by_key[binding.group_key].evidence_id:
                raise ResidualIntelligenceError(
                    "threshold binding evidence_id does not match group evidence"
                )
        object.__setattr__(self, "bindings", bindings)

    @property
    def policy_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def evidence_by_group_key(self) -> dict[str, CalibrationEvidence]:
        return {item.group.group_key: item for item in self.evidence}

    def binding_by_group_key(self) -> dict[str, CalibrationThresholdBinding]:
        return {item.group_key: item for item in self.bindings}

    def current_evidence_for(self, group: CalibrationGroup) -> CalibrationEvidence | None:
        record = self.evidence_by_group_key().get(group.group_key)
        if record is None:
            return None
        if not record.is_current(
            admission_id=self.current_admission_id,
            split_root=self.current_split_root,
            evaluation_identity=self.current_evaluation_identity,
            holdout_root=self.current_holdout_root,
        ):
            return None
        return record

    def decide(self, request: SelectivePredictionRequest) -> AbstentionDecision:
        if not isinstance(request, SelectivePredictionRequest):
            raise ResidualIntelligenceError("decide requires a typed SelectivePredictionRequest")
        group = request.group
        evidence = self.evidence_by_group_key().get(group.group_key)
        binding = self.binding_by_group_key().get(group.group_key)
        bound = binding is not None
        threshold = binding.accept_threshold_ppm if binding is not None else 0
        evidence_id = evidence.evidence_id if evidence is not None else ""
        current = self.current_evidence_for(group)

        def emit(disposition: ExpertDisposition, *reasons: str) -> AbstentionDecision:
            return AbstentionDecision(
                disposition=disposition,
                group_key=group.group_key,
                risk_class=group.risk,
                score_ppm=request.score_ppm,
                group_threshold_bound=bound,
                group_threshold_ppm=threshold,
                reason_codes=reasons,
                evidence_id=evidence_id,
                candidate_only=True,
            )

        if not request.input_valid:
            return emit(ExpertDisposition.REJECT_INPUT, REASON_REJECT_INPUT)
        if not request.capability_available:
            return emit(ExpertDisposition.CAPABILITY_UNAVAILABLE, REASON_CAPABILITY_UNAVAILABLE)
        if evidence is None:
            return emit(ExpertDisposition.OUT_OF_DISTRIBUTION, REASON_MISSING_CALIBRATION_GROUP)
        if self.ood_signals_binding and request.out_of_distribution:
            return emit(ExpertDisposition.OUT_OF_DISTRIBUTION, REASON_OOD)
        if current is None:
            return emit(ExpertDisposition.ABSTAIN, REASON_CURRENT_EVIDENCE)
        if request.critical_boundary:
            return emit(ExpertDisposition.ABSTAIN, REASON_CRITICAL_BOUNDARY)
        if not current.zero_critical_false_accepts:
            return emit(ExpertDisposition.ABSTAIN, REASON_CRITICAL_FALSE_ACCEPT)
        if binding is None:
            return emit(ExpertDisposition.ABSTAIN, REASON_NO_GROUP_THRESHOLD)
        if request.score_ppm < binding.accept_threshold_ppm:
            return emit(ExpertDisposition.ABSTAIN, REASON_BELOW_GROUP_THRESHOLD)
        if group.risk in PROPOSAL_RISKS:
            return emit(
                ExpertDisposition.VALIDATION_REQUIRED,
                REASON_VALIDATION_REQUIRED,
                REASON_R4_R5_PROPOSAL,
            )
        if not request.validation_satisfied:
            return emit(ExpertDisposition.VALIDATION_REQUIRED, REASON_VALIDATION_REQUIRED)
        return emit(ExpertDisposition.ACCEPT, REASON_ACCEPT)

    def apply_threshold_cas(
        self,
        *,
        group: CalibrationGroup,
        proposed_threshold_ppm: int,
        origin: ThresholdChangeOrigin | str,
        cas_identity: str,
        expected_binding_id: str = "",
    ) -> SelectivePredictionPolicy:
        evidence = self.current_evidence_for(group)
        if evidence is None:
            raise ResidualIntelligenceError("threshold change requires current evidence")
        replacement = apply_threshold_cas(
            self.binding_by_group_key().get(group.group_key),
            group=group,
            evidence=evidence,
            proposed_threshold_ppm=proposed_threshold_ppm,
            origin=origin,
            cas_identity=cas_identity,
            expected_binding_id=expected_binding_id,
            admission_id=self.current_admission_id,
            split_root=self.current_split_root,
            evaluation_identity=self.current_evaluation_identity,
            holdout_root=self.current_holdout_root,
        )
        remaining = tuple(
            item for item in self.bindings if item.group_key != group.group_key
        )
        return SelectivePredictionPolicy(
            schema=self.schema,
            current_admission_id=self.current_admission_id,
            current_split_root=self.current_split_root,
            current_holdout_root=self.current_holdout_root,
            current_evaluation_identity=self.current_evaluation_identity,
            evidence=self.evidence,
            bindings=remaining + (replacement,),
            ood_signals_binding=self.ood_signals_binding,
        )

    def rollback_threshold(
        self,
        *,
        group: CalibrationGroup,
        cas_identity: str,
    ) -> SelectivePredictionPolicy:
        binding = self.binding_by_group_key().get(group.group_key)
        if binding is None:
            raise ResidualIntelligenceError("rollback requires an admitted group threshold")
        replacement = rollback_threshold_binding(binding, cas_identity=cas_identity)
        remaining = tuple(
            item for item in self.bindings if item.group_key != group.group_key
        )
        return SelectivePredictionPolicy(
            schema=self.schema,
            current_admission_id=self.current_admission_id,
            current_split_root=self.current_split_root,
            current_holdout_root=self.current_holdout_root,
            current_evaluation_identity=self.current_evaluation_identity,
            evidence=self.evidence,
            bindings=remaining + (replacement,),
            ood_signals_binding=self.ood_signals_binding,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "current_admission_id": self.current_admission_id,
            "current_split_root": self.current_split_root,
            "current_holdout_root": self.current_holdout_root,
            "current_evaluation_identity": self.current_evaluation_identity,
            "evidence": [item.to_dict() for item in self.evidence],
            "bindings": [item.to_dict() for item in self.bindings],
            "ood_signals_binding": self.ood_signals_binding,
        }
        if include_id:
            result["policy_id"] = self.policy_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SelectivePredictionPolicy:
        reject_global_threshold_fields(payload, noun="selective prediction policy")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"policy_id", "bindings", "ood_signals_binding"},
            noun="selective prediction policy",
        )
        evidence_payload = payload.get("evidence")
        if isinstance(evidence_payload, (str, bytes, bytearray)) or not isinstance(
            evidence_payload, Sequence
        ):
            raise ResidualIntelligenceError("selective prediction evidence must be a sequence")
        bindings_payload = payload.get("bindings") or ()
        if isinstance(bindings_payload, (str, bytes, bytearray)) or not isinstance(
            bindings_payload, Sequence
        ):
            raise ResidualIntelligenceError("selective prediction bindings must be a sequence")
        result = cls(
            schema=str(payload.get("schema") or ""),
            current_admission_id=str(payload.get("current_admission_id") or ""),
            current_split_root=str(payload.get("current_split_root") or ""),
            current_holdout_root=str(payload.get("current_holdout_root") or ""),
            current_evaluation_identity=str(payload.get("current_evaluation_identity") or ""),
            evidence=tuple(CalibrationEvidence.from_dict(item) for item in evidence_payload),
            bindings=tuple(
                CalibrationThresholdBinding.from_dict(item) for item in bindings_payload
            ),
            ood_signals_binding=payload.get("ood_signals_binding", False),
        )
        claimed = str(payload.get("policy_id") or "")
        if claimed and claimed != result.policy_id:
            raise ResidualIntelligenceError("selective prediction policy identity mismatch")
        return result


def selectively_predict(
    policy: SelectivePredictionPolicy,
    request: SelectivePredictionRequest,
) -> AbstentionDecision:
    """Apply the group-keyed policy; never a global threshold."""

    if not isinstance(policy, SelectivePredictionPolicy):
        raise ResidualIntelligenceError("selectively_predict requires a typed policy")
    return policy.decide(request)


__all__ = (
    "ABSTENTION_DECISION_SCHEMA",
    "CLOSED_DISPOSITIONS",
    "PROPOSAL_RISKS",
    "SELECTIVE_PREDICTION_POLICY_SCHEMA",
    "AbstentionDecision",
    "SelectivePredictionPolicy",
    "SelectivePredictionRequest",
    "selectively_predict",
)
