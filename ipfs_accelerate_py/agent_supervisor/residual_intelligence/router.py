"""Hard-constraint residual expert router.

``ResidualExpertRouter`` walks the admitted cascade, records every eligible
candidate and every hard rejection, and selects the earliest surviving stage.
Remote routes require both privacy permission and an inference-policy grant.
Simulation never becomes a live route.  Human review remains the safe fallback.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

from .cascade import (
    CASCADE_ORDER,
    CASCADE_POLICY_VERSION,
    CascadeCandidate,
    CascadeHardRejection,
    CascadeStage,
    REASON_FAMILY,
    REASON_HUMAN_FALLBACK,
    REASON_RISK,
    REASON_SAFE_FALLBACK,
    ResidualCascade,
    ResidualCascadeContext,
    ResidualCascadeWalk,
    effective_privacy_class,
    parse_cascade_stage,
)
from .contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)
from .residual_ir import MAX_TOKEN_BUDGET, ResidualTaskInput
from .rights import TrainingCorpusAdmission
from .task_families import (
    ABSTAIN_OUTPUT_CLASS,
    LOCAL_ONLY_PRIVACY,
    REASON_OUTPUT_CLASS,
    REASON_TASK_FAMILY_MISMATCH,
    ResidualTaskFamilySpec,
    family_spec_for,
    risk_rank,
)

ROUTE_REQUEST_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-route-request@1"
ROUTE_DECISION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-route-decision@1"
ROUTER_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-router@1"
ROUTER_POLICY_VERSION: Final = "vrif-013-router@1"
DEFAULT_COST_BUDGET_MICROUNITS: Final = 1_000_000_000
DEFAULT_LOCAL_HARDWARE: Final[tuple[str, ...]] = ("cpu-small-hermetic",)
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})
MAX_REASON_CODES: Final = 32


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _optional_identity(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return required_text(value, name)


def _stage_int_map(value: Any, name: str) -> dict[str, int]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        stage = parse_cascade_stage(str(key)).value
        if stage in result:
            raise ResidualIntelligenceError(f"{name} contains duplicate cascade stages")
        result[stage] = bounded_int(item, f"{name}.{stage}", minimum=0, maximum=1_000_000_000_000)
    return result


def _inspect_task_input(
    task_input: ResidualTaskInput,
    family_spec: ResidualTaskFamilySpec,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        if task_input.task_family is not family_spec.task_family:
            raise ResidualIntelligenceError(REASON_TASK_FAMILY_MISMATCH)
        family_spec.reject_unsupported_risk(task_input.risk_class)
        family_spec.validate_compact_features(task_input.compact_features)
        illegal = [item for item in task_input.allowed_outputs if item not in family_spec.output_classes]
        if illegal:
            raise ResidualIntelligenceError(REASON_OUTPUT_CLASS)
        if family_spec.always_abstain and tuple(task_input.allowed_outputs) != (ABSTAIN_OUTPUT_CLASS,):
            raise ResidualIntelligenceError("always-abstain families only allow ABSTAIN")
    except ResidualIntelligenceError as exc:
        message = str(exc)
        if REASON_TASK_FAMILY_MISMATCH in message:
            reasons.append(REASON_FAMILY)
        elif "risk" in message.casefold():
            reasons.append(REASON_RISK)
        else:
            reasons.append(REASON_FAMILY)
            reasons.append(message.split(":", 1)[0].strip() or REASON_FAMILY)
    if task_input.risk_class not in family_spec.allowed_risk_classes:
        reasons.append(REASON_RISK)
    elif risk_rank(task_input.risk_class) > risk_rank(family_spec.risk_ceiling):
        reasons.append(REASON_RISK)
    unique = tuple(dict.fromkeys(reasons))
    return (not unique, unique)


@dataclass(frozen=True)
class ResidualRouteRequest:
    """Typed routing capsule: IR plus hard constraint and budget facts."""

    task_input: ResidualTaskInput
    privacy_class: PrivacyClass
    local_execution_available: bool = True
    provider_authorized: bool = False
    provider_healthy: bool = False
    hardware_available: tuple[str, ...] = DEFAULT_LOCAL_HARDWARE
    available_capabilities: tuple[str, ...] = ()
    cache_hit: bool = False
    cache_identity: str = ""
    procedure_available: bool = False
    procedure_preconditions_satisfied: bool = False
    procedure_root: str = ""
    deterministic_rule_available: bool = False
    rule_identity: str = ""
    local_linear_available: bool = False
    local_ranker_available: bool = False
    local_structured_available: bool = False
    local_general_available: bool = False
    remote_standard_available: bool = False
    remote_strong_available: bool = False
    human_review_available: bool = True
    validation_available: bool = True
    simulated: bool = False
    capability_inferred_from_importability: bool = False
    inference_policy_permits_remote: bool = False
    ood_conservative_abstain: bool = False
    cost_budget_microunits: int = DEFAULT_COST_BUDGET_MICROUNITS
    expected_decision_value_microunits: int = 0
    evidence_references: tuple[str, ...] = ()
    calibration_group_key: str = ""
    admission: TrainingCorpusAdmission | None = None
    stage_cost_microunits: Mapping[str, int] = field(default_factory=dict)
    stage_input_tokens: Mapping[str, int] = field(default_factory=dict)
    stage_output_tokens: Mapping[str, int] = field(default_factory=dict)
    stage_quality_ppm: Mapping[str, int] = field(default_factory=dict)
    schema: str = ROUTE_REQUEST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "request_id",
            "task_input",
            "privacy_class",
            "local_execution_available",
            "provider_authorized",
            "provider_healthy",
            "hardware_available",
            "available_capabilities",
            "cache_hit",
            "cache_identity",
            "procedure_available",
            "procedure_preconditions_satisfied",
            "procedure_root",
            "deterministic_rule_available",
            "rule_identity",
            "local_linear_available",
            "local_ranker_available",
            "local_structured_available",
            "local_general_available",
            "remote_standard_available",
            "remote_strong_available",
            "human_review_available",
            "validation_available",
            "simulated",
            "capability_inferred_from_importability",
            "inference_policy_permits_remote",
            "ood_conservative_abstain",
            "cost_budget_microunits",
            "expected_decision_value_microunits",
            "evidence_references",
            "calibration_group_key",
            "admission",
            "stage_cost_microunits",
            "stage_input_tokens",
            "stage_output_tokens",
            "stage_quality_ppm",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != ROUTE_REQUEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual route request schema")
        if not isinstance(self.task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        for field_name in (
            "local_execution_available",
            "provider_authorized",
            "provider_healthy",
            "cache_hit",
            "procedure_available",
            "procedure_preconditions_satisfied",
            "deterministic_rule_available",
            "local_linear_available",
            "local_ranker_available",
            "local_structured_available",
            "local_general_available",
            "remote_standard_available",
            "remote_strong_available",
            "human_review_available",
            "validation_available",
            "simulated",
            "capability_inferred_from_importability",
            "inference_policy_permits_remote",
            "ood_conservative_abstain",
        ):
            object.__setattr__(self, field_name, _require_bool(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "hardware_available",
            text_tuple(self.hardware_available, "hardware_available"),
        )
        object.__setattr__(
            self,
            "available_capabilities",
            text_tuple(self.available_capabilities, "available_capabilities"),
        )
        object.__setattr__(self, "cache_identity", _optional_identity(self.cache_identity, "cache_identity"))
        object.__setattr__(self, "procedure_root", _optional_identity(self.procedure_root, "procedure_root"))
        object.__setattr__(self, "rule_identity", _optional_identity(self.rule_identity, "rule_identity"))
        object.__setattr__(
            self,
            "cost_budget_microunits",
            bounded_int(
                self.cost_budget_microunits,
                "cost_budget_microunits",
                minimum=0,
                maximum=1_000_000_000_000,
            ),
        )
        object.__setattr__(
            self,
            "expected_decision_value_microunits",
            bounded_int(
                self.expected_decision_value_microunits,
                "expected_decision_value_microunits",
                minimum=0,
                maximum=1_000_000_000_000,
            ),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references"),
        )
        object.__setattr__(
            self,
            "calibration_group_key",
            _optional_identity(self.calibration_group_key, "calibration_group_key"),
        )
        if self.admission is not None and not isinstance(self.admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
        object.__setattr__(
            self, "stage_cost_microunits", _stage_int_map(self.stage_cost_microunits, "stage_cost_microunits")
        )
        object.__setattr__(
            self, "stage_input_tokens", _stage_int_map(self.stage_input_tokens, "stage_input_tokens")
        )
        object.__setattr__(
            self, "stage_output_tokens", _stage_int_map(self.stage_output_tokens, "stage_output_tokens")
        )
        object.__setattr__(
            self, "stage_quality_ppm", _stage_int_map(self.stage_quality_ppm, "stage_quality_ppm")
        )

    @property
    def family_spec(self) -> ResidualTaskFamilySpec:
        return family_spec_for(self.task_input.task_family)

    @property
    def effective_privacy_class(self) -> PrivacyClass:
        privacy = effective_privacy_class(self.family_spec.privacy_class, self.privacy_class)
        if self.admission is not None:
            admitted = PrivacyClass(self.admission.privacy_classification)
            privacy = effective_privacy_class(privacy, admitted)
        return privacy

    @property
    def request_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_cascade_context(self) -> ResidualCascadeContext:
        family_spec = self.family_spec
        input_valid, input_reasons = _inspect_task_input(self.task_input, family_spec)
        inference_remote = self.inference_policy_permits_remote
        if self.effective_privacy_class in LOCAL_ONLY_PRIVACY:
            inference_remote = False
        return ResidualCascadeContext(
            family_spec=family_spec,
            risk_class=self.task_input.risk_class,
            privacy_class=self.effective_privacy_class,
            input_valid=input_valid,
            input_reason_codes=input_reasons,
            local_execution_available=self.local_execution_available,
            provider_authorized=self.provider_authorized,
            provider_healthy=self.provider_healthy,
            hardware_available=self.hardware_available,
            available_capabilities=self.available_capabilities,
            cache_hit=self.cache_hit,
            cache_identity=self.cache_identity,
            procedure_available=self.procedure_available,
            procedure_preconditions_satisfied=self.procedure_preconditions_satisfied,
            procedure_root=self.procedure_root,
            deterministic_rule_available=self.deterministic_rule_available,
            rule_identity=self.rule_identity,
            local_linear_available=self.local_linear_available,
            local_ranker_available=self.local_ranker_available,
            local_structured_available=self.local_structured_available,
            local_general_available=self.local_general_available,
            remote_standard_available=self.remote_standard_available,
            remote_strong_available=self.remote_strong_available,
            human_review_available=self.human_review_available,
            validation_available=self.validation_available,
            simulated=self.simulated,
            capability_inferred_from_importability=self.capability_inferred_from_importability,
            inference_policy_permits_remote=inference_remote,
            ood_conservative_abstain=self.ood_conservative_abstain,
            token_budget=self.task_input.token_budget,
            cost_budget_microunits=self.cost_budget_microunits,
            expected_decision_value_microunits=self.expected_decision_value_microunits,
            evidence_references=self.evidence_references,
            stage_cost_microunits=self.stage_cost_microunits,
            stage_input_tokens=self.stage_input_tokens,
            stage_output_tokens=self.stage_output_tokens,
            stage_quality_ppm=self.stage_quality_ppm,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_input": self.task_input.to_dict(),
            "privacy_class": self.privacy_class.value,
            "local_execution_available": self.local_execution_available,
            "provider_authorized": self.provider_authorized,
            "provider_healthy": self.provider_healthy,
            "hardware_available": list(self.hardware_available),
            "available_capabilities": list(self.available_capabilities),
            "cache_hit": self.cache_hit,
            "cache_identity": self.cache_identity,
            "procedure_available": self.procedure_available,
            "procedure_preconditions_satisfied": self.procedure_preconditions_satisfied,
            "procedure_root": self.procedure_root,
            "deterministic_rule_available": self.deterministic_rule_available,
            "rule_identity": self.rule_identity,
            "local_linear_available": self.local_linear_available,
            "local_ranker_available": self.local_ranker_available,
            "local_structured_available": self.local_structured_available,
            "local_general_available": self.local_general_available,
            "remote_standard_available": self.remote_standard_available,
            "remote_strong_available": self.remote_strong_available,
            "human_review_available": self.human_review_available,
            "validation_available": self.validation_available,
            "simulated": self.simulated,
            "capability_inferred_from_importability": self.capability_inferred_from_importability,
            "inference_policy_permits_remote": self.inference_policy_permits_remote,
            "ood_conservative_abstain": self.ood_conservative_abstain,
            "cost_budget_microunits": self.cost_budget_microunits,
            "expected_decision_value_microunits": self.expected_decision_value_microunits,
            "evidence_references": list(self.evidence_references),
            "calibration_group_key": self.calibration_group_key,
            "admission": None if self.admission is None else self.admission.to_dict(),
            "stage_cost_microunits": dict(self.stage_cost_microunits),
            "stage_input_tokens": dict(self.stage_input_tokens),
            "stage_output_tokens": dict(self.stage_output_tokens),
            "stage_quality_ppm": dict(self.stage_quality_ppm),
        }
        if include_id:
            result["request_id"] = self.request_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualRouteRequest:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("residual route request must be an object")
        unknown = sorted(str(key) for key in payload if key not in cls._FIELDS)
        if unknown:
            raise UnknownFieldError(
                f"residual route request contains unknown fields: {', '.join(unknown)}"
            )
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required={"schema", "task_input", "privacy_class"},
            noun="residual route request",
        )
        admission_payload = payload.get("admission")
        admission = (
            None
            if admission_payload in (None, {})
            else TrainingCorpusAdmission.from_dict(admission_payload)
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_input=ResidualTaskInput.from_dict(payload.get("task_input") or {}),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            local_execution_available=payload.get("local_execution_available", True),
            provider_authorized=payload.get("provider_authorized", False),
            provider_healthy=payload.get("provider_healthy", False),
            hardware_available=tuple(
                payload["hardware_available"]
                if "hardware_available" in payload
                else DEFAULT_LOCAL_HARDWARE
            ),
            available_capabilities=tuple(payload.get("available_capabilities") or ()),
            cache_hit=payload.get("cache_hit", False),
            cache_identity=str(payload.get("cache_identity") or ""),
            procedure_available=payload.get("procedure_available", False),
            procedure_preconditions_satisfied=payload.get(
                "procedure_preconditions_satisfied", False
            ),
            procedure_root=str(payload.get("procedure_root") or ""),
            deterministic_rule_available=payload.get("deterministic_rule_available", False),
            rule_identity=str(payload.get("rule_identity") or ""),
            local_linear_available=payload.get("local_linear_available", False),
            local_ranker_available=payload.get("local_ranker_available", False),
            local_structured_available=payload.get("local_structured_available", False),
            local_general_available=payload.get("local_general_available", False),
            remote_standard_available=payload.get("remote_standard_available", False),
            remote_strong_available=payload.get("remote_strong_available", False),
            human_review_available=payload.get("human_review_available", True),
            validation_available=payload.get("validation_available", True),
            simulated=payload.get("simulated", False),
            capability_inferred_from_importability=payload.get(
                "capability_inferred_from_importability", False
            ),
            inference_policy_permits_remote=payload.get("inference_policy_permits_remote", False),
            ood_conservative_abstain=payload.get("ood_conservative_abstain", False),
            cost_budget_microunits=payload.get(
                "cost_budget_microunits", DEFAULT_COST_BUDGET_MICROUNITS
            ),
            expected_decision_value_microunits=payload.get(
                "expected_decision_value_microunits", 0
            ),
            evidence_references=tuple(payload.get("evidence_references") or ()),
            calibration_group_key=str(payload.get("calibration_group_key") or ""),
            admission=admission,
            stage_cost_microunits=payload.get("stage_cost_microunits") or {},
            stage_input_tokens=payload.get("stage_input_tokens") or {},
            stage_output_tokens=payload.get("stage_output_tokens") or {},
            stage_quality_ppm=payload.get("stage_quality_ppm") or {},
        )
        claimed = str(payload.get("request_id") or "")
        if claimed and claimed != result.request_id:
            raise ResidualIntelligenceError("residual route request identity mismatch")
        return result


def _human_disposition(walk: ResidualCascadeWalk) -> ExpertDisposition:
    producing = [item for item in walk.candidates if item.stage is not CascadeStage.HUMAN_REVIEW]
    if producing:
        return ExpertDisposition.ABSTAIN
    rejection_reasons = {code for item in walk.hard_rejections for code in item.reason_codes}
    if "ood_conservative_abstain" in rejection_reasons:
        return ExpertDisposition.OUT_OF_DISTRIBUTION
    capability_codes = {
        "capability_unavailable",
        "hardware_unavailable",
        "provider_unhealthy",
        "local_execution_unavailable",
        "capability_inferred_from_importability",
    }
    availability_only = {
        "cache_miss",
        "procedure_unavailable",
        "procedure_precondition_failure",
        "deterministic_rule_unavailable",
        "stage_unavailable",
        "candidate_evidence_required",
        "expected_decision_value_insufficient",
        "token_or_cost_budget_exceeded",
        "unsupported_family_class",
        "always_abstain_family",
        "privacy_route_denied",
        "private_to_unauthorized_provider",
        "inference_policy_denies_remote",
        "provider_unauthorized",
        "simulation_forbidden",
        "validation_unavailable",
        "family_out_of_bound",
        "risk_ceiling_exceeded",
    }
    # Incidental hardware/provider codes on unoffered remote stages must not
    # promote a safe human fallback into CAPABILITY_UNAVAILABLE.
    offered_reasons = {
        code
        for item in walk.hard_rejections
        for code in item.reason_codes
        if not (set(item.reason_codes) & availability_only)
    }
    if offered_reasons & capability_codes and not (offered_reasons - capability_codes):
        return ExpertDisposition.CAPABILITY_UNAVAILABLE
    return ExpertDisposition.ABSTAIN


@dataclass(frozen=True)
class ResidualRouteDecision:
    """Selected cascade stage plus the complete candidate and rejection receipt."""

    request_id: str
    family: ResidualTaskFamily
    risk_class: RiskClass
    privacy_class: PrivacyClass
    selected_stage: CascadeStage
    selected_expert_id: str
    disposition: ExpertDisposition
    candidates: tuple[CascadeCandidate, ...]
    hard_rejections: tuple[CascadeHardRejection, ...]
    expected_cost_microunits: int
    expected_input_tokens: int
    expected_output_tokens: int
    expected_quality_ppm: int
    abstention_behavior: str
    validator_identity: str
    validation_required: bool
    fallback_stage: CascadeStage
    reason_codes: tuple[str, ...]
    evidence_references: tuple[str, ...]
    calibration_group_key: str = ""
    cascade_policy_version: str = CASCADE_POLICY_VERSION
    walk_id: str = ""
    candidate_only: bool = True
    schema: str = ROUTE_DECISION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "decision_id",
            "request_id",
            "family",
            "risk_class",
            "privacy_class",
            "selected_stage",
            "selected_expert_id",
            "disposition",
            "candidates",
            "hard_rejections",
            "expected_cost_microunits",
            "expected_input_tokens",
            "expected_output_tokens",
            "expected_quality_ppm",
            "abstention_behavior",
            "validator_identity",
            "validation_required",
            "fallback_stage",
            "reason_codes",
            "evidence_references",
            "calibration_group_key",
            "cascade_policy_version",
            "walk_id",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != ROUTE_DECISION_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual route decision schema")
        object.__setattr__(self, "request_id", required_text(self.request_id, "request_id"))
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        object.__setattr__(self, "selected_stage", CascadeStage(self.selected_stage))
        object.__setattr__(
            self, "selected_expert_id", required_text(self.selected_expert_id, "selected_expert_id")
        )
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        candidates = tuple(self.candidates)
        rejections = tuple(self.hard_rejections)
        if any(not isinstance(item, CascadeCandidate) for item in candidates):
            raise ResidualIntelligenceError("decision candidates must be typed CascadeCandidate")
        if any(not isinstance(item, CascadeHardRejection) for item in rejections):
            raise ResidualIntelligenceError(
                "decision hard_rejections must be typed CascadeHardRejection"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "hard_rejections", rejections)
        object.__setattr__(
            self,
            "expected_cost_microunits",
            bounded_int(
                self.expected_cost_microunits,
                "expected_cost_microunits",
                minimum=0,
                maximum=1_000_000_000_000,
            ),
        )
        object.__setattr__(
            self,
            "expected_input_tokens",
            bounded_int(
                self.expected_input_tokens,
                "expected_input_tokens",
                minimum=0,
                maximum=MAX_TOKEN_BUDGET,
            ),
        )
        object.__setattr__(
            self,
            "expected_output_tokens",
            bounded_int(
                self.expected_output_tokens,
                "expected_output_tokens",
                minimum=0,
                maximum=MAX_TOKEN_BUDGET,
            ),
        )
        object.__setattr__(
            self,
            "expected_quality_ppm",
            bounded_int(
                self.expected_quality_ppm,
                "expected_quality_ppm",
                minimum=0,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self,
            "abstention_behavior",
            required_text(self.abstention_behavior, "abstention_behavior"),
        )
        object.__setattr__(
            self,
            "validator_identity",
            required_text(self.validator_identity, "validator_identity"),
        )
        if not self.validator_identity.startswith("validator:"):
            raise ResidualIntelligenceError("validator_identity must remain an independent validator")
        object.__setattr__(
            self,
            "validation_required",
            _require_bool(self.validation_required, "validation_required"),
        )
        if not self.validation_required:
            raise ResidualIntelligenceError("required validation must be preserved")
        object.__setattr__(self, "fallback_stage", CascadeStage(self.fallback_stage))
        if self.fallback_stage is not CascadeStage.HUMAN_REVIEW:
            raise ResidualIntelligenceError("safe fallback must remain human_review")
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", allow_empty=False, max_items=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", allow_empty=False),
        )
        object.__setattr__(
            self,
            "calibration_group_key",
            _optional_identity(self.calibration_group_key, "calibration_group_key"),
        )
        object.__setattr__(
            self,
            "cascade_policy_version",
            required_text(self.cascade_policy_version, "cascade_policy_version"),
        )
        object.__setattr__(self, "walk_id", _optional_identity(self.walk_id, "walk_id"))
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("route decisions must remain candidate_only=true")
        if self.disposition is ExpertDisposition.ACCEPT and self.risk_class in PROPOSAL_RISKS:
            raise ResidualIntelligenceError("R4/R5 routes cannot ACCEPT")
        if self.selected_stage not in {item.stage for item in self.candidates}:
            raise ResidualIntelligenceError("selected stage must be present in candidates")
        if CascadeStage.HUMAN_REVIEW not in {item.stage for item in self.candidates}:
            raise ResidualIntelligenceError("human fallback must remain reachable")
        observed = tuple(item.stage for item in self.candidates) + tuple(
            item.stage for item in self.hard_rejections
        )
        if len(observed) != len(CASCADE_ORDER) or set(observed) != set(CASCADE_ORDER):
            raise ResidualIntelligenceError("route decision must record every cascade stage")
        candidate_stages = tuple(item.stage for item in self.candidates)
        rejection_stages = tuple(item.stage for item in self.hard_rejections)
        expected_candidates = tuple(
            stage for stage in CASCADE_ORDER if stage in set(candidate_stages)
        )
        expected_rejections = tuple(
            stage for stage in CASCADE_ORDER if stage in set(rejection_stages)
        )
        if candidate_stages != expected_candidates:
            raise ResidualIntelligenceError(
                "decision candidates must follow the admitted production order"
            )
        if rejection_stages != expected_rejections:
            raise ResidualIntelligenceError(
                "decision hard rejections must follow the admitted production order"
            )
        if self.selected_stage is not candidate_stages[0]:
            raise ResidualIntelligenceError(
                "selected stage must be the earliest surviving candidate"
            )

    @property
    def decision_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def selected_candidate(self) -> CascadeCandidate:
        for item in self.candidates:
            if item.stage is self.selected_stage:
                return item
        raise ResidualIntelligenceError("selected stage has no candidate receipt")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "request_id": self.request_id,
            "family": self.family.value,
            "risk_class": self.risk_class.value,
            "privacy_class": self.privacy_class.value,
            "selected_stage": self.selected_stage.value,
            "selected_expert_id": self.selected_expert_id,
            "disposition": self.disposition.value,
            "candidates": [item.to_dict() for item in self.candidates],
            "hard_rejections": [item.to_dict() for item in self.hard_rejections],
            "expected_cost_microunits": self.expected_cost_microunits,
            "expected_input_tokens": self.expected_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "expected_quality_ppm": self.expected_quality_ppm,
            "abstention_behavior": self.abstention_behavior,
            "validator_identity": self.validator_identity,
            "validation_required": True,
            "fallback_stage": CascadeStage.HUMAN_REVIEW.value,
            "reason_codes": list(self.reason_codes),
            "evidence_references": list(self.evidence_references),
            "calibration_group_key": self.calibration_group_key,
            "cascade_policy_version": self.cascade_policy_version,
            "walk_id": self.walk_id,
            "candidate_only": True,
        }
        if include_id:
            result["decision_id"] = self.decision_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualRouteDecision:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("residual route decision must be an object")
        unknown = sorted(str(key) for key in payload if key not in cls._FIELDS)
        if unknown:
            raise UnknownFieldError(
                f"residual route decision contains unknown fields: {', '.join(unknown)}"
            )
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"decision_id", "calibration_group_key", "walk_id"},
            noun="residual route decision",
        )
        raw_candidates = payload.get("candidates")
        raw_rejections = payload.get("hard_rejections")
        if isinstance(raw_candidates, (str, bytes, bytearray)) or not isinstance(
            raw_candidates, Sequence
        ):
            raise ResidualIntelligenceError("route decision candidates must be a sequence")
        if isinstance(raw_rejections, (str, bytes, bytearray)) or not isinstance(
            raw_rejections, Sequence
        ):
            raise ResidualIntelligenceError("route decision hard_rejections must be a sequence")
        result = cls(
            schema=str(payload.get("schema") or ""),
            request_id=str(payload.get("request_id") or ""),
            family=ResidualTaskFamily(str(payload.get("family") or "")),
            risk_class=RiskClass(str(payload.get("risk_class") or "")),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            selected_stage=CascadeStage(str(payload.get("selected_stage") or "")),
            selected_expert_id=str(payload.get("selected_expert_id") or ""),
            disposition=ExpertDisposition(str(payload.get("disposition") or "")),
            candidates=tuple(CascadeCandidate.from_dict(item) for item in raw_candidates),
            hard_rejections=tuple(
                CascadeHardRejection.from_dict(item) for item in raw_rejections
            ),
            expected_cost_microunits=payload.get("expected_cost_microunits"),
            expected_input_tokens=payload.get("expected_input_tokens"),
            expected_output_tokens=payload.get("expected_output_tokens"),
            expected_quality_ppm=payload.get("expected_quality_ppm"),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            validator_identity=str(payload.get("validator_identity") or ""),
            validation_required=payload.get("validation_required"),
            fallback_stage=CascadeStage(str(payload.get("fallback_stage") or "")),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence_references=tuple(payload.get("evidence_references") or ()),
            calibration_group_key=str(payload.get("calibration_group_key") or ""),
            cascade_policy_version=str(payload.get("cascade_policy_version") or ""),
            walk_id=str(payload.get("walk_id") or ""),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("decision_id") or "")
        if claimed and claimed != result.decision_id:
            raise ResidualIntelligenceError("residual route decision identity mismatch")
        return result


def _build_decision(
    request: ResidualRouteRequest,
    walk: ResidualCascadeWalk,
    context: ResidualCascadeContext,
) -> ResidualRouteDecision:
    selected = walk.selected_candidate()
    if not context.input_valid:
        disposition = ExpertDisposition.REJECT_INPUT
        reasons = tuple(dict.fromkeys((*context.input_reason_codes, *selected.reason_codes)))
    elif walk.selected_stage is CascadeStage.HUMAN_REVIEW:
        reasons = tuple(
            dict.fromkeys((REASON_HUMAN_FALLBACK, REASON_SAFE_FALLBACK, *selected.reason_codes))
        )
        disposition = _human_disposition(walk)
    elif request.task_input.risk_class in PROPOSAL_RISKS:
        disposition = ExpertDisposition.VALIDATION_REQUIRED
        reasons = tuple(
            dict.fromkeys((*selected.reason_codes, "VALIDATION_REQUIRED", "r4_r5_proposal_tier"))
        )
    else:
        disposition = ExpertDisposition.ACCEPT
        reasons = selected.reason_codes
    family_spec = request.family_spec
    return ResidualRouteDecision(
        request_id=request.request_id,
        family=request.task_input.task_family,
        risk_class=request.task_input.risk_class,
        privacy_class=context.effective_privacy,
        selected_stage=walk.selected_stage,
        selected_expert_id=selected.expert_id,
        disposition=disposition,
        candidates=walk.candidates,
        hard_rejections=walk.hard_rejections,
        expected_cost_microunits=selected.expected_cost_microunits,
        expected_input_tokens=selected.expected_input_tokens,
        expected_output_tokens=selected.expected_output_tokens,
        expected_quality_ppm=selected.expected_quality_ppm,
        abstention_behavior=family_spec.abstention_behavior,
        validator_identity=family_spec.validator_identity,
        validation_required=True,
        fallback_stage=CascadeStage.HUMAN_REVIEW,
        reason_codes=reasons,
        evidence_references=selected.evidence_references,
        calibration_group_key=request.calibration_group_key,
        cascade_policy_version=walk.policy_version,
        walk_id=walk.walk_id,
        candidate_only=True,
    )


@dataclass(frozen=True)
class ResidualExpertRouter:
    """Exclusive cascade owner.  Canonical provider/procedure systems stay outside."""

    cascade: ResidualCascade = field(default_factory=ResidualCascade)
    policy_version: str = ROUTER_POLICY_VERSION
    schema: str = ROUTER_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "router_id", "policy_version", "cascade"}
    )

    def __post_init__(self) -> None:
        if self.schema != ROUTER_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual expert router schema")
        if not isinstance(self.cascade, ResidualCascade):
            raise ResidualIntelligenceError("router cascade must be ResidualCascade")
        object.__setattr__(
            self, "policy_version", required_text(self.policy_version, "policy_version")
        )
        if self.policy_version != ROUTER_POLICY_VERSION:
            raise ResidualIntelligenceError("router policy_version is not the admitted table")

    @property
    def router_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def route(self, request: ResidualRouteRequest) -> ResidualRouteDecision:
        if not isinstance(request, ResidualRouteRequest):
            raise ResidualIntelligenceError("route requires ResidualRouteRequest")
        context = request.to_cascade_context()
        walk = self.cascade.evaluate(context)
        return _build_decision(request, walk, context)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "policy_version": self.policy_version,
            "cascade": self.cascade.to_dict(),
        }
        if include_id:
            result["router_id"] = self.router_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualExpertRouter:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"router_id"},
            noun="residual expert router",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            policy_version=str(payload.get("policy_version") or ""),
            cascade=ResidualCascade.from_dict(payload.get("cascade") or {}),
        )
        claimed = str(payload.get("router_id") or "")
        if claimed and claimed != result.router_id:
            raise ResidualIntelligenceError("residual expert router identity mismatch")
        return result


DEFAULT_ROUTER: Final[ResidualExpertRouter] = ResidualExpertRouter()


def route(request: ResidualRouteRequest) -> ResidualRouteDecision:
    """Route one residual task through the admitted cascade."""

    return DEFAULT_ROUTER.route(request)


__all__ = (
    "DEFAULT_ROUTER",
    "ROUTER_POLICY_VERSION",
    "ResidualExpertRouter",
    "ResidualRouteDecision",
    "ResidualRouteRequest",
    "route",
)
