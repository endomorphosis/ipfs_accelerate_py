"""Fixed residual expert cascade and hard routing constraints.

The production order is exact cache, verified procedure, deterministic rule,
local linear/ranker/structured/general specialists, remote standard/strong
models, then human review.  Hard family, risk, capability, privacy,
validation, hardware, provider-health, simulation, budget, and evidence
gates reject a stage without skipping later recording.  Learned and remote
outputs remain candidates; the cascade never simulates a live answer.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
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
from .expert_specs import (
    EXPERT_CLASS_HARDWARE,
    ExpertClass,
    expert_spec_for,
)
from .residual_ir import MAX_SCORE_PPM, MAX_TOKEN_BUDGET
from .task_families import (
    LOCAL_ONLY_PRIVACY,
    PRIVACY_ROUTE_LOCAL_ONLY,
    ResidualTaskFamilySpec,
    risk_rank,
)

CASCADE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-cascade@1"
CASCADE_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-cascade-candidate@1"
)
CASCADE_REJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-cascade-hard-rejection@1"
)
CASCADE_WALK_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-cascade-walk@1"
CASCADE_POLICY_VERSION: Final = "vrif-013-cascade@1"
MAX_ROUTE_COST_MICROUNITS: Final = 1_000_000_000_000
MAX_REASON_CODES: Final = 32
HUMAN_REVIEW_HARDWARE: Final = "human-reviewer"
LOCAL_GENERAL_HARDWARE: Final = "cpu-gpu-optional-bounded"
REMOTE_STANDARD_HARDWARE: Final = "provider-standard"
REMOTE_STRONG_HARDWARE: Final = "provider-strong"

REASON_EXACT_CACHE: Final = "exact_cache"
REASON_VERIFIED_PROCEDURE: Final = "verified_procedure"
REASON_DETERMINISTIC_RULE: Final = "deterministic_rule"
REASON_LOCAL_LINEAR: Final = "local_linear_expert"
REASON_LOCAL_RANKER: Final = "local_ranker"
REASON_LOCAL_STRUCTURED: Final = "local_structured_specialist"
REASON_LOCAL_GENERAL: Final = "local_general_model"
REASON_REMOTE_STANDARD: Final = "remote_standard_model"
REASON_REMOTE_STRONG: Final = "remote_strong_model"
REASON_HUMAN_FALLBACK: Final = "human_review_required"
REASON_SAFE_FALLBACK: Final = "safe_fallback"
REASON_FAMILY: Final = "family_out_of_bound"
REASON_RISK: Final = "risk_ceiling_exceeded"
REASON_UNSUPPORTED_CLASS: Final = "unsupported_family_class"
REASON_CAPABILITY: Final = "capability_unavailable"
REASON_IMPORTABILITY: Final = "capability_inferred_from_importability"
REASON_PRIVACY: Final = "privacy_route_denied"
REASON_PRIVATE_REMOTE: Final = "private_to_unauthorized_provider"
REASON_HARDWARE: Final = "hardware_unavailable"
REASON_PROVIDER_HEALTH: Final = "provider_unhealthy"
REASON_PROVIDER_AUTH: Final = "provider_unauthorized"
REASON_SIMULATION: Final = "simulation_forbidden"
REASON_VALIDATION: Final = "validation_unavailable"
REASON_BUDGET: Final = "token_or_cost_budget_exceeded"
REASON_VALUE: Final = "expected_decision_value_insufficient"
REASON_CACHE_MISS: Final = "cache_miss"
REASON_PROCEDURE_UNAVAILABLE: Final = "procedure_unavailable"
REASON_PROCEDURE_PRECONDITION: Final = "procedure_precondition_failure"
REASON_RULE_UNAVAILABLE: Final = "deterministic_rule_unavailable"
REASON_STAGE_UNAVAILABLE: Final = "stage_unavailable"
REASON_ALWAYS_ABSTAIN: Final = "always_abstain_family"
REASON_OOD: Final = "ood_conservative_abstain"
REASON_MISSING_EVIDENCE: Final = "candidate_evidence_required"
REASON_INFERENCE_POLICY: Final = "inference_policy_denies_remote"
REASON_LOCAL_EXECUTION: Final = "local_execution_unavailable"

CLOSED_CONSTRAINTS: Final[frozenset[str]] = frozenset(
    {
        "family",
        "risk",
        "capability",
        "privacy",
        "validation",
        "hardware",
        "provider",
        "simulation",
        "budget",
        "value",
        "evidence",
        "availability",
        "ood",
        "input",
    }
)

PRIVACY_STRICTNESS: Final[Mapping[PrivacyClass, int]] = {
    PrivacyClass.PUBLIC: 0,
    PrivacyClass.INTERNAL: 1,
    PrivacyClass.REPOSITORY_PRIVATE: 2,
    PrivacyClass.TENANT_PRIVATE: 3,
    PrivacyClass.MATTER_CONFIDENTIAL: 4,
    PrivacyClass.PERSONAL_DATA: 5,
    PrivacyClass.HEALTH_DATA: 6,
    PrivacyClass.LEGAL_PRIVILEGED: 7,
    PrivacyClass.CREDENTIAL: 8,
    PrivacyClass.PROOF_WITNESS: 8,
}


class CascadeStage(str, Enum):
    """Closed residual cascade stages in production order."""

    EXACT_CACHE = "exact_cache"
    VERIFIED_PROCEDURE = "verified_procedure"
    DETERMINISTIC_RULE = "deterministic_rule"
    LOCAL_LINEAR_EXPERT = "local_linear_expert"
    LOCAL_RANKER = "local_ranker"
    LOCAL_STRUCTURED_SPECIALIST = "local_structured_specialist"
    LOCAL_GENERAL_MODEL = "local_general_model"
    REMOTE_STANDARD_MODEL = "remote_standard_model"
    REMOTE_STRONG_MODEL = "remote_strong_model"
    HUMAN_REVIEW = "human_review"


CASCADE_ORDER: Final[tuple[CascadeStage, ...]] = (
    CascadeStage.EXACT_CACHE,
    CascadeStage.VERIFIED_PROCEDURE,
    CascadeStage.DETERMINISTIC_RULE,
    CascadeStage.LOCAL_LINEAR_EXPERT,
    CascadeStage.LOCAL_RANKER,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST,
    CascadeStage.LOCAL_GENERAL_MODEL,
    CascadeStage.REMOTE_STANDARD_MODEL,
    CascadeStage.REMOTE_STRONG_MODEL,
    CascadeStage.HUMAN_REVIEW,
)

DETERMINISTIC_STAGES: Final[frozenset[CascadeStage]] = frozenset(
    {
        CascadeStage.EXACT_CACHE,
        CascadeStage.VERIFIED_PROCEDURE,
        CascadeStage.DETERMINISTIC_RULE,
    }
)
LOCAL_STAGES: Final[frozenset[CascadeStage]] = frozenset(
    {
        CascadeStage.EXACT_CACHE,
        CascadeStage.VERIFIED_PROCEDURE,
        CascadeStage.DETERMINISTIC_RULE,
        CascadeStage.LOCAL_LINEAR_EXPERT,
        CascadeStage.LOCAL_RANKER,
        CascadeStage.LOCAL_STRUCTURED_SPECIALIST,
        CascadeStage.LOCAL_GENERAL_MODEL,
    }
)
REMOTE_STAGES: Final[frozenset[CascadeStage]] = frozenset(
    {
        CascadeStage.REMOTE_STANDARD_MODEL,
        CascadeStage.REMOTE_STRONG_MODEL,
    }
)
LEARNED_STAGES: Final[frozenset[CascadeStage]] = frozenset(
    {
        CascadeStage.LOCAL_LINEAR_EXPERT,
        CascadeStage.LOCAL_RANKER,
        CascadeStage.LOCAL_STRUCTURED_SPECIALIST,
        CascadeStage.LOCAL_GENERAL_MODEL,
        CascadeStage.REMOTE_STANDARD_MODEL,
        CascadeStage.REMOTE_STRONG_MODEL,
    }
)
STAGE_EXPERT_CLASS: Final[Mapping[CascadeStage, ExpertClass]] = {
    CascadeStage.EXACT_CACHE: ExpertClass.A,
    CascadeStage.VERIFIED_PROCEDURE: ExpertClass.B,
    CascadeStage.DETERMINISTIC_RULE: ExpertClass.B,
    CascadeStage.LOCAL_LINEAR_EXPERT: ExpertClass.C,
    CascadeStage.LOCAL_RANKER: ExpertClass.D,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: ExpertClass.E,
}
STAGE_HARDWARE: Final[Mapping[CascadeStage, tuple[str, ...]]] = {
    CascadeStage.EXACT_CACHE: EXPERT_CLASS_HARDWARE[ExpertClass.A],
    CascadeStage.VERIFIED_PROCEDURE: EXPERT_CLASS_HARDWARE[ExpertClass.B],
    CascadeStage.DETERMINISTIC_RULE: EXPERT_CLASS_HARDWARE[ExpertClass.B],
    CascadeStage.LOCAL_LINEAR_EXPERT: EXPERT_CLASS_HARDWARE[ExpertClass.C],
    CascadeStage.LOCAL_RANKER: EXPERT_CLASS_HARDWARE[ExpertClass.D],
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: EXPERT_CLASS_HARDWARE[ExpertClass.E],
    CascadeStage.LOCAL_GENERAL_MODEL: (LOCAL_GENERAL_HARDWARE,),
    CascadeStage.REMOTE_STANDARD_MODEL: (REMOTE_STANDARD_HARDWARE,),
    CascadeStage.REMOTE_STRONG_MODEL: (REMOTE_STRONG_HARDWARE,),
    CascadeStage.HUMAN_REVIEW: (HUMAN_REVIEW_HARDWARE,),
}
STAGE_EXPECTED_COST_MICROUNITS: Final[Mapping[CascadeStage, int]] = {
    CascadeStage.EXACT_CACHE: 0,
    CascadeStage.VERIFIED_PROCEDURE: 1,
    CascadeStage.DETERMINISTIC_RULE: 2,
    CascadeStage.LOCAL_LINEAR_EXPERT: 100,
    CascadeStage.LOCAL_RANKER: 500,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: 2_000,
    CascadeStage.LOCAL_GENERAL_MODEL: 8_000,
    CascadeStage.REMOTE_STANDARD_MODEL: 25_000,
    CascadeStage.REMOTE_STRONG_MODEL: 80_000,
    CascadeStage.HUMAN_REVIEW: 100_000,
}
STAGE_EXPECTED_INPUT_TOKENS: Final[Mapping[CascadeStage, int]] = {
    CascadeStage.EXACT_CACHE: 0,
    CascadeStage.VERIFIED_PROCEDURE: 0,
    CascadeStage.DETERMINISTIC_RULE: 0,
    CascadeStage.LOCAL_LINEAR_EXPERT: 64,
    CascadeStage.LOCAL_RANKER: 128,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: 256,
    CascadeStage.LOCAL_GENERAL_MODEL: 512,
    CascadeStage.REMOTE_STANDARD_MODEL: 1_024,
    CascadeStage.REMOTE_STRONG_MODEL: 2_048,
    CascadeStage.HUMAN_REVIEW: 0,
}
STAGE_EXPECTED_OUTPUT_TOKENS: Final[Mapping[CascadeStage, int]] = {
    CascadeStage.EXACT_CACHE: 0,
    CascadeStage.VERIFIED_PROCEDURE: 0,
    CascadeStage.DETERMINISTIC_RULE: 0,
    CascadeStage.LOCAL_LINEAR_EXPERT: 32,
    CascadeStage.LOCAL_RANKER: 64,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: 128,
    CascadeStage.LOCAL_GENERAL_MODEL: 256,
    CascadeStage.REMOTE_STANDARD_MODEL: 512,
    CascadeStage.REMOTE_STRONG_MODEL: 1_024,
    CascadeStage.HUMAN_REVIEW: 0,
}
STAGE_EXPECTED_QUALITY_PPM: Final[Mapping[CascadeStage, int]] = {
    CascadeStage.EXACT_CACHE: MAX_SCORE_PPM,
    CascadeStage.VERIFIED_PROCEDURE: MAX_SCORE_PPM,
    CascadeStage.DETERMINISTIC_RULE: 950_000,
    CascadeStage.LOCAL_LINEAR_EXPERT: 800_000,
    CascadeStage.LOCAL_RANKER: 820_000,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: 850_000,
    CascadeStage.LOCAL_GENERAL_MODEL: 880_000,
    CascadeStage.REMOTE_STANDARD_MODEL: 900_000,
    CascadeStage.REMOTE_STRONG_MODEL: 930_000,
    CascadeStage.HUMAN_REVIEW: MAX_SCORE_PPM,
}
STAGE_SELECTION_REASON: Final[Mapping[CascadeStage, str]] = {
    CascadeStage.EXACT_CACHE: REASON_EXACT_CACHE,
    CascadeStage.VERIFIED_PROCEDURE: REASON_VERIFIED_PROCEDURE,
    CascadeStage.DETERMINISTIC_RULE: REASON_DETERMINISTIC_RULE,
    CascadeStage.LOCAL_LINEAR_EXPERT: REASON_LOCAL_LINEAR,
    CascadeStage.LOCAL_RANKER: REASON_LOCAL_RANKER,
    CascadeStage.LOCAL_STRUCTURED_SPECIALIST: REASON_LOCAL_STRUCTURED,
    CascadeStage.LOCAL_GENERAL_MODEL: REASON_LOCAL_GENERAL,
    CascadeStage.REMOTE_STANDARD_MODEL: REASON_REMOTE_STANDARD,
    CascadeStage.REMOTE_STRONG_MODEL: REASON_REMOTE_STRONG,
    CascadeStage.HUMAN_REVIEW: REASON_HUMAN_FALLBACK,
}


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _ppm(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_SCORE_PPM)


def _cost(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_ROUTE_COST_MICROUNITS)


def _tokens(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_TOKEN_BUDGET)


def parse_cascade_stage(value: CascadeStage | str) -> CascadeStage:
    if isinstance(value, CascadeStage):
        return value
    text = required_text(value, "cascade stage", max_bytes=64)
    try:
        return CascadeStage(text)
    except ValueError as exc:
        raise ResidualIntelligenceError(f"unknown cascade stage {text}") from exc


def stage_is_local(stage: CascadeStage) -> bool:
    return parse_cascade_stage(stage) in LOCAL_STAGES


def stage_is_remote(stage: CascadeStage) -> bool:
    return parse_cascade_stage(stage) in REMOTE_STAGES


def stage_is_deterministic(stage: CascadeStage) -> bool:
    return parse_cascade_stage(stage) in DETERMINISTIC_STAGES


def stage_is_learned(stage: CascadeStage) -> bool:
    return parse_cascade_stage(stage) in LEARNED_STAGES


def effective_privacy_class(
    family_privacy: PrivacyClass | str,
    request_privacy: PrivacyClass | str,
) -> PrivacyClass:
    family = PrivacyClass(family_privacy)
    requested = PrivacyClass(request_privacy)
    if PRIVACY_STRICTNESS[requested] >= PRIVACY_STRICTNESS[family]:
        return requested
    return family


def expert_id_for_stage(stage: CascadeStage, family: ResidualTaskFamily) -> str:
    parsed = parse_cascade_stage(stage)
    family = ResidualTaskFamily(family)
    expert_class = STAGE_EXPERT_CLASS.get(parsed)
    if expert_class is not None:
        try:
            return expert_spec_for(family, expert_class).expert_id
        except ResidualIntelligenceError:
            pass
    slug = family.value.lower().replace("_", "-")
    return f"cascade:{parsed.value}:{slug}"


def _int_override(
    overrides: Mapping[str, int],
    stage: CascadeStage,
    default: int,
    *,
    name: str,
    parser,
) -> int:
    if stage.value not in overrides:
        return default
    return parser(overrides[stage.value], f"{name}.{stage.value}")


@dataclass(frozen=True)
class CascadeCandidate:
    """One hard-constraint-eligible cascade stage, not a selected authority."""

    stage: CascadeStage
    expert_id: str
    expected_cost_microunits: int
    expected_input_tokens: int
    expected_output_tokens: int
    expected_quality_ppm: int
    local_execution: bool
    evidence_references: tuple[str, ...]
    reason_codes: tuple[str, ...]
    candidate_only: bool = True
    schema: str = CASCADE_CANDIDATE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "candidate_id",
            "stage",
            "expert_id",
            "expected_cost_microunits",
            "expected_input_tokens",
            "expected_output_tokens",
            "expected_quality_ppm",
            "local_execution",
            "evidence_references",
            "reason_codes",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CASCADE_CANDIDATE_SCHEMA:
            raise ResidualIntelligenceError("unsupported cascade candidate schema")
        object.__setattr__(self, "stage", parse_cascade_stage(self.stage))
        object.__setattr__(self, "expert_id", required_text(self.expert_id, "expert_id"))
        object.__setattr__(
            self,
            "expected_cost_microunits",
            _cost(self.expected_cost_microunits, "expected_cost_microunits"),
        )
        object.__setattr__(
            self,
            "expected_input_tokens",
            _tokens(self.expected_input_tokens, "expected_input_tokens"),
        )
        object.__setattr__(
            self,
            "expected_output_tokens",
            _tokens(self.expected_output_tokens, "expected_output_tokens"),
        )
        object.__setattr__(
            self,
            "expected_quality_ppm",
            _ppm(self.expected_quality_ppm, "expected_quality_ppm"),
        )
        object.__setattr__(
            self,
            "local_execution",
            _require_bool(self.local_execution, "local_execution"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", allow_empty=False),
        )
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", allow_empty=False, max_items=MAX_REASON_CODES),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("cascade candidates must remain candidate_only=true")

    @property
    def candidate_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "stage": self.stage.value,
            "expert_id": self.expert_id,
            "expected_cost_microunits": self.expected_cost_microunits,
            "expected_input_tokens": self.expected_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "expected_quality_ppm": self.expected_quality_ppm,
            "local_execution": self.local_execution,
            "evidence_references": list(self.evidence_references),
            "reason_codes": list(self.reason_codes),
            "candidate_only": True,
        }
        if include_id:
            result["candidate_id"] = self.candidate_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CascadeCandidate:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"candidate_id"},
            noun="cascade candidate",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            stage=parse_cascade_stage(str(payload.get("stage") or "")),
            expert_id=str(payload.get("expert_id") or ""),
            expected_cost_microunits=payload.get("expected_cost_microunits"),
            expected_input_tokens=payload.get("expected_input_tokens"),
            expected_output_tokens=payload.get("expected_output_tokens"),
            expected_quality_ppm=payload.get("expected_quality_ppm"),
            local_execution=payload.get("local_execution"),
            evidence_references=tuple(payload.get("evidence_references") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("candidate_id") or "")
        if claimed and claimed != result.candidate_id:
            raise ResidualIntelligenceError("cascade candidate identity mismatch")
        return result


@dataclass(frozen=True)
class CascadeHardRejection:
    """One recorded hard constraint failure for a cascade stage."""

    stage: CascadeStage
    constraint: str
    reason_codes: tuple[str, ...]
    schema: str = CASCADE_REJECTION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "rejection_id", "stage", "constraint", "reason_codes"}
    )

    def __post_init__(self) -> None:
        if self.schema != CASCADE_REJECTION_SCHEMA:
            raise ResidualIntelligenceError("unsupported cascade hard-rejection schema")
        object.__setattr__(self, "stage", parse_cascade_stage(self.stage))
        object.__setattr__(self, "constraint", required_text(self.constraint, "constraint", max_bytes=32))
        if self.constraint not in CLOSED_CONSTRAINTS:
            raise ResidualIntelligenceError("hard-rejection constraint is outside the closed set")
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", allow_empty=False, max_items=MAX_REASON_CODES),
        )

    @property
    def rejection_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "stage": self.stage.value,
            "constraint": self.constraint,
            "reason_codes": list(self.reason_codes),
        }
        if include_id:
            result["rejection_id"] = self.rejection_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CascadeHardRejection:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"rejection_id"},
            noun="cascade hard rejection",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            stage=parse_cascade_stage(str(payload.get("stage") or "")),
            constraint=str(payload.get("constraint") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = str(payload.get("rejection_id") or "")
        if claimed and claimed != result.rejection_id:
            raise ResidualIntelligenceError("cascade hard-rejection identity mismatch")
        return result


@dataclass(frozen=True)
class ResidualCascadeContext:
    """Runtime facts consumed by one cascade walk.  Never a live model probe."""

    family_spec: ResidualTaskFamilySpec
    risk_class: RiskClass
    privacy_class: PrivacyClass
    input_valid: bool
    input_reason_codes: tuple[str, ...]
    local_execution_available: bool
    provider_authorized: bool
    provider_healthy: bool
    hardware_available: tuple[str, ...]
    available_capabilities: tuple[str, ...]
    cache_hit: bool
    cache_identity: str
    procedure_available: bool
    procedure_preconditions_satisfied: bool
    procedure_root: str
    deterministic_rule_available: bool
    rule_identity: str
    local_linear_available: bool
    local_ranker_available: bool
    local_structured_available: bool
    local_general_available: bool
    remote_standard_available: bool
    remote_strong_available: bool
    human_review_available: bool
    validation_available: bool
    simulated: bool
    capability_inferred_from_importability: bool
    inference_policy_permits_remote: bool
    ood_conservative_abstain: bool
    token_budget: int
    cost_budget_microunits: int
    expected_decision_value_microunits: int
    evidence_references: tuple[str, ...]
    stage_cost_microunits: Mapping[str, int]
    stage_input_tokens: Mapping[str, int]
    stage_output_tokens: Mapping[str, int]
    stage_quality_ppm: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.family_spec, ResidualTaskFamilySpec):
            raise ResidualIntelligenceError("cascade context requires ResidualTaskFamilySpec")
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        for field in (
            "input_valid",
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
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        object.__setattr__(
            self,
            "input_reason_codes",
            text_tuple(self.input_reason_codes, "input_reason_codes"),
        )
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
        object.__setattr__(
            self,
            "cache_identity",
            ""
            if self.cache_identity in (None, "")
            else required_text(self.cache_identity, "cache_identity"),
        )
        object.__setattr__(
            self,
            "procedure_root",
            ""
            if self.procedure_root in (None, "")
            else required_text(self.procedure_root, "procedure_root"),
        )
        object.__setattr__(
            self,
            "rule_identity",
            ""
            if self.rule_identity in (None, "")
            else required_text(self.rule_identity, "rule_identity"),
        )
        object.__setattr__(self, "token_budget", _tokens(self.token_budget, "token_budget"))
        object.__setattr__(
            self,
            "cost_budget_microunits",
            _cost(self.cost_budget_microunits, "cost_budget_microunits"),
        )
        object.__setattr__(
            self,
            "expected_decision_value_microunits",
            _cost(self.expected_decision_value_microunits, "expected_decision_value_microunits"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references"),
        )
        object.__setattr__(self, "stage_cost_microunits", _stage_int_map(self.stage_cost_microunits, "stage_cost_microunits", _cost))
        object.__setattr__(self, "stage_input_tokens", _stage_int_map(self.stage_input_tokens, "stage_input_tokens", _tokens))
        object.__setattr__(self, "stage_output_tokens", _stage_int_map(self.stage_output_tokens, "stage_output_tokens", _tokens))
        object.__setattr__(self, "stage_quality_ppm", _stage_int_map(self.stage_quality_ppm, "stage_quality_ppm", _ppm))

    @property
    def family(self) -> ResidualTaskFamily:
        return self.family_spec.task_family

    @property
    def effective_privacy(self) -> PrivacyClass:
        return effective_privacy_class(self.family_spec.privacy_class, self.privacy_class)


def _stage_int_map(value: Any, name: str, parser) -> dict[str, int]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        stage = parse_cascade_stage(str(key))
        result[stage.value] = parser(item, f"{name}.{stage.value}")
    return result


def _primary_constraint(reason_codes: Sequence[str]) -> str:
    mapping: tuple[tuple[str, str], ...] = (
        (REASON_SIMULATION, "simulation"),
        (REASON_FAMILY, "family"),
        (REASON_RISK, "risk"),
        (REASON_ALWAYS_ABSTAIN, "family"),
        (REASON_UNSUPPORTED_CLASS, "family"),
        (REASON_CACHE_MISS, "availability"),
        (REASON_PROCEDURE_UNAVAILABLE, "availability"),
        (REASON_PROCEDURE_PRECONDITION, "availability"),
        (REASON_RULE_UNAVAILABLE, "availability"),
        (REASON_STAGE_UNAVAILABLE, "availability"),
        (REASON_LOCAL_EXECUTION, "capability"),
        (REASON_MISSING_EVIDENCE, "evidence"),
        (REASON_PRIVACY, "privacy"),
        (REASON_PRIVATE_REMOTE, "privacy"),
        (REASON_INFERENCE_POLICY, "privacy"),
        (REASON_HARDWARE, "hardware"),
        (REASON_PROVIDER_HEALTH, "provider"),
        (REASON_PROVIDER_AUTH, "provider"),
        (REASON_IMPORTABILITY, "capability"),
        (REASON_CAPABILITY, "capability"),
        (REASON_VALIDATION, "validation"),
        (REASON_OOD, "ood"),
        (REASON_BUDGET, "budget"),
        (REASON_VALUE, "value"),
    )
    codes = set(reason_codes)
    for reason, constraint in mapping:
        if reason in codes:
            return constraint
    return "availability"


@dataclass(frozen=True)
class ResidualCascadeWalk:
    """Complete cascade receipt: every stage is a candidate or a hard rejection."""

    policy_version: str
    family: ResidualTaskFamily
    risk_class: RiskClass
    candidates: tuple[CascadeCandidate, ...]
    hard_rejections: tuple[CascadeHardRejection, ...]
    selected_stage: CascadeStage
    fallback_stage: CascadeStage = CascadeStage.HUMAN_REVIEW
    candidate_only: bool = True
    schema: str = CASCADE_WALK_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "walk_id",
            "policy_version",
            "family",
            "risk_class",
            "candidates",
            "hard_rejections",
            "selected_stage",
            "fallback_stage",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CASCADE_WALK_SCHEMA:
            raise ResidualIntelligenceError("unsupported cascade walk schema")
        object.__setattr__(
            self, "policy_version", required_text(self.policy_version, "policy_version")
        )
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        candidates = tuple(self.candidates)
        rejections = tuple(self.hard_rejections)
        if any(not isinstance(item, CascadeCandidate) for item in candidates):
            raise ResidualIntelligenceError("walk candidates must be typed CascadeCandidate")
        if any(not isinstance(item, CascadeHardRejection) for item in rejections):
            raise ResidualIntelligenceError(
                "walk hard_rejections must be typed CascadeHardRejection"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "hard_rejections", rejections)
        object.__setattr__(self, "selected_stage", parse_cascade_stage(self.selected_stage))
        object.__setattr__(self, "fallback_stage", parse_cascade_stage(self.fallback_stage))
        if self.fallback_stage is not CascadeStage.HUMAN_REVIEW:
            raise ResidualIntelligenceError("cascade fallback must remain human_review")
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("cascade walks must remain candidate_only=true")
        observed = tuple(item.stage for item in candidates) + tuple(item.stage for item in rejections)
        if len(observed) != len(CASCADE_ORDER) or set(observed) != set(CASCADE_ORDER):
            raise ResidualIntelligenceError("cascade walk must record every stage exactly once")
        if self.selected_stage not in {item.stage for item in candidates}:
            raise ResidualIntelligenceError("selected stage must be a recorded candidate")
        if CascadeStage.HUMAN_REVIEW not in {item.stage for item in candidates}:
            raise ResidualIntelligenceError("human fallback must remain reachable")
        candidate_stages = tuple(item.stage for item in candidates)
        rejection_stages = tuple(item.stage for item in rejections)
        expected_candidates = tuple(
            stage for stage in CASCADE_ORDER if stage in set(candidate_stages)
        )
        expected_rejections = tuple(
            stage for stage in CASCADE_ORDER if stage in set(rejection_stages)
        )
        if candidate_stages != expected_candidates:
            raise ResidualIntelligenceError(
                "cascade candidates must follow the admitted production order"
            )
        if rejection_stages != expected_rejections:
            raise ResidualIntelligenceError(
                "cascade hard rejections must follow the admitted production order"
            )
        if self.selected_stage is not candidate_stages[0]:
            raise ResidualIntelligenceError(
                "selected stage must be the earliest surviving candidate"
            )

    @property
    def walk_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def candidate_for(self, stage: CascadeStage) -> CascadeCandidate | None:
        for item in self.candidates:
            if item.stage is stage:
                return item
        return None

    def selected_candidate(self) -> CascadeCandidate:
        match = self.candidate_for(self.selected_stage)
        if match is None:
            raise ResidualIntelligenceError("selected stage has no candidate receipt")
        return match

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "policy_version": self.policy_version,
            "family": self.family.value,
            "risk_class": self.risk_class.value,
            "candidates": [item.to_dict() for item in self.candidates],
            "hard_rejections": [item.to_dict() for item in self.hard_rejections],
            "selected_stage": self.selected_stage.value,
            "fallback_stage": CascadeStage.HUMAN_REVIEW.value,
            "candidate_only": True,
        }
        if include_id:
            result["walk_id"] = self.walk_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualCascadeWalk:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("cascade walk must be an object")
        unknown = sorted(str(key) for key in payload if key not in cls._FIELDS)
        if unknown:
            raise UnknownFieldError(f"cascade walk contains unknown fields: {', '.join(unknown)}")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"walk_id"},
            noun="cascade walk",
        )
        raw_candidates = payload.get("candidates")
        raw_rejections = payload.get("hard_rejections")
        if isinstance(raw_candidates, (str, bytes, bytearray)) or not isinstance(
            raw_candidates, Sequence
        ):
            raise ResidualIntelligenceError("cascade walk candidates must be a sequence")
        if isinstance(raw_rejections, (str, bytes, bytearray)) or not isinstance(
            raw_rejections, Sequence
        ):
            raise ResidualIntelligenceError("cascade walk hard_rejections must be a sequence")
        result = cls(
            schema=str(payload.get("schema") or ""),
            policy_version=str(payload.get("policy_version") or ""),
            family=ResidualTaskFamily(str(payload.get("family") or "")),
            risk_class=RiskClass(str(payload.get("risk_class") or "")),
            candidates=tuple(CascadeCandidate.from_dict(item) for item in raw_candidates),
            hard_rejections=tuple(
                CascadeHardRejection.from_dict(item) for item in raw_rejections
            ),
            selected_stage=parse_cascade_stage(str(payload.get("selected_stage") or "")),
            fallback_stage=parse_cascade_stage(str(payload.get("fallback_stage") or "")),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("walk_id") or "")
        if claimed and claimed != result.walk_id:
            raise ResidualIntelligenceError("cascade walk identity mismatch")
        return result


def _stage_available(context: ResidualCascadeContext, stage: CascadeStage) -> tuple[bool, tuple[str, ...]]:
    if stage is CascadeStage.EXACT_CACHE:
        if not context.cache_hit:
            return False, (REASON_CACHE_MISS,)
        return True, ()
    if stage is CascadeStage.VERIFIED_PROCEDURE:
        if not context.procedure_available:
            return False, (REASON_PROCEDURE_UNAVAILABLE,)
        if not context.procedure_preconditions_satisfied:
            return False, (REASON_PROCEDURE_PRECONDITION,)
        return True, ()
    if stage is CascadeStage.DETERMINISTIC_RULE:
        if not context.deterministic_rule_available:
            return False, (REASON_RULE_UNAVAILABLE,)
        return True, ()
    flags = {
        CascadeStage.LOCAL_LINEAR_EXPERT: context.local_linear_available,
        CascadeStage.LOCAL_RANKER: context.local_ranker_available,
        CascadeStage.LOCAL_STRUCTURED_SPECIALIST: context.local_structured_available,
        CascadeStage.LOCAL_GENERAL_MODEL: context.local_general_available,
        CascadeStage.REMOTE_STANDARD_MODEL: context.remote_standard_available,
        CascadeStage.REMOTE_STRONG_MODEL: context.remote_strong_available,
    }
    if not flags.get(stage, False):
        return False, (REASON_STAGE_UNAVAILABLE,)
    return True, ()


def _stage_evidence(context: ResidualCascadeContext, stage: CascadeStage) -> tuple[str, ...]:
    extra: list[str] = []
    if stage is CascadeStage.EXACT_CACHE and context.cache_identity:
        extra.append(context.cache_identity)
    if stage is CascadeStage.VERIFIED_PROCEDURE and context.procedure_root:
        extra.append(context.procedure_root)
    if stage is CascadeStage.DETERMINISTIC_RULE and context.rule_identity:
        extra.append(context.rule_identity)
    extra.extend(context.evidence_references)
    extra.append(f"stage:{stage.value}")
    extra.append(expert_id_for_stage(stage, context.family))
    return tuple(dict.fromkeys(extra))


def _missing_evidence(context: ResidualCascadeContext, stage: CascadeStage) -> bool:
    if stage is CascadeStage.EXACT_CACHE:
        return not context.cache_identity
    if stage is CascadeStage.VERIFIED_PROCEDURE:
        return not context.procedure_root
    if stage is CascadeStage.DETERMINISTIC_RULE:
        return not context.rule_identity
    return False


def evaluate_stage_constraints(
    context: ResidualCascadeContext,
    stage: CascadeStage,
) -> tuple[str, ...]:
    """Return every hard-rejection reason for one stage, in stable order."""

    parsed = parse_cascade_stage(stage)
    if parsed is CascadeStage.HUMAN_REVIEW:
        return ()
    reasons: list[str] = []
    family_spec = context.family_spec
    if context.simulated:
        reasons.append(REASON_SIMULATION)
    if not context.input_valid:
        reasons.extend(context.input_reason_codes or (REASON_FAMILY,))
    if context.risk_class not in family_spec.allowed_risk_classes:
        reasons.append(REASON_RISK)
    elif risk_rank(context.risk_class) > risk_rank(family_spec.risk_ceiling):
        reasons.append(REASON_RISK)
    if family_spec.always_abstain:
        reasons.append(REASON_ALWAYS_ABSTAIN)
    expert_class = STAGE_EXPERT_CLASS.get(parsed)
    if expert_class is not None and not family_spec.allows_expert_class(expert_class.value):
        reasons.append(REASON_UNSUPPORTED_CLASS)
    available, availability_reasons = _stage_available(context, parsed)
    if not available:
        reasons.extend(availability_reasons)
    if parsed in LOCAL_STAGES and not context.local_execution_available:
        reasons.append(REASON_LOCAL_EXECUTION)
    if available and _missing_evidence(context, parsed):
        reasons.append(REASON_MISSING_EVIDENCE)
    local_execution = parsed in LOCAL_STAGES
    privacy = context.effective_privacy
    if parsed in REMOTE_STAGES:
        if privacy in LOCAL_ONLY_PRIVACY or family_spec.privacy_route_policy == PRIVACY_ROUTE_LOCAL_ONLY:
            reasons.append(REASON_PRIVACY)
        if not family_spec.privacy_route_permits(
            provider_authorized=context.provider_authorized,
            local_execution=False,
        ):
            reasons.append(REASON_PRIVATE_REMOTE)
        if not context.inference_policy_permits_remote:
            reasons.append(REASON_INFERENCE_POLICY)
        if not context.provider_authorized:
            reasons.append(REASON_PROVIDER_AUTH)
        if not context.provider_healthy:
            reasons.append(REASON_PROVIDER_HEALTH)
    elif not family_spec.privacy_route_permits(
        provider_authorized=context.provider_authorized,
        local_execution=local_execution,
    ):
        reasons.append(REASON_PRIVACY)
    required_hardware = STAGE_HARDWARE[parsed]
    if any(item not in context.hardware_available for item in required_hardware):
        reasons.append(REASON_HARDWARE)
    if context.capability_inferred_from_importability and parsed in LEARNED_STAGES:
        reasons.append(REASON_IMPORTABILITY)
    if parsed in LEARNED_STAGES and context.available_capabilities:
        required = [
            item
            for item in family_spec.capabilities
            if item not in {"no_network", "provider_free_capability_contract"}
        ]
        missing = [item for item in required if item not in context.available_capabilities]
        if missing:
            reasons.append(REASON_CAPABILITY)
    if family_spec.independent_validator_required and not context.validation_available:
        reasons.append(REASON_VALIDATION)
    if context.ood_conservative_abstain and parsed in LEARNED_STAGES:
        reasons.append(REASON_OOD)
    cost = _int_override(
        context.stage_cost_microunits,
        parsed,
        STAGE_EXPECTED_COST_MICROUNITS[parsed],
        name="stage_cost_microunits",
        parser=_cost,
    )
    input_tokens = _int_override(
        context.stage_input_tokens,
        parsed,
        STAGE_EXPECTED_INPUT_TOKENS[parsed],
        name="stage_input_tokens",
        parser=_tokens,
    )
    output_tokens = _int_override(
        context.stage_output_tokens,
        parsed,
        STAGE_EXPECTED_OUTPUT_TOKENS[parsed],
        name="stage_output_tokens",
        parser=_tokens,
    )
    if cost > context.cost_budget_microunits or (input_tokens + output_tokens) > context.token_budget:
        reasons.append(REASON_BUDGET)
    if parsed in LEARNED_STAGES and cost > context.expected_decision_value_microunits:
        reasons.append(REASON_VALUE)
    return tuple(dict.fromkeys(reasons))


def _make_candidate(context: ResidualCascadeContext, stage: CascadeStage) -> CascadeCandidate:
    parsed = parse_cascade_stage(stage)
    reasons = (STAGE_SELECTION_REASON[parsed],)
    if parsed is CascadeStage.HUMAN_REVIEW:
        reasons = (REASON_HUMAN_FALLBACK, REASON_SAFE_FALLBACK)
    return CascadeCandidate(
        stage=parsed,
        expert_id=expert_id_for_stage(parsed, context.family),
        expected_cost_microunits=_int_override(
            context.stage_cost_microunits,
            parsed,
            STAGE_EXPECTED_COST_MICROUNITS[parsed],
            name="stage_cost_microunits",
            parser=_cost,
        ),
        expected_input_tokens=_int_override(
            context.stage_input_tokens,
            parsed,
            STAGE_EXPECTED_INPUT_TOKENS[parsed],
            name="stage_input_tokens",
            parser=_tokens,
        ),
        expected_output_tokens=_int_override(
            context.stage_output_tokens,
            parsed,
            STAGE_EXPECTED_OUTPUT_TOKENS[parsed],
            name="stage_output_tokens",
            parser=_tokens,
        ),
        expected_quality_ppm=_int_override(
            context.stage_quality_ppm,
            parsed,
            STAGE_EXPECTED_QUALITY_PPM[parsed],
            name="stage_quality_ppm",
            parser=_ppm,
        ),
        local_execution=parsed in LOCAL_STAGES or parsed is CascadeStage.HUMAN_REVIEW,
        evidence_references=_stage_evidence(context, parsed),
        reason_codes=reasons,
        candidate_only=True,
    )


@dataclass(frozen=True)
class ResidualCascade:
    """Versioned exact-order cascade.  Provider routers remain canonical owners."""

    policy_version: str = CASCADE_POLICY_VERSION
    schema: str = CASCADE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"schema", "cascade_id", "policy_version", "stages"})

    def __post_init__(self) -> None:
        if self.schema != CASCADE_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual cascade schema")
        object.__setattr__(
            self, "policy_version", required_text(self.policy_version, "policy_version")
        )
        if self.policy_version != CASCADE_POLICY_VERSION:
            raise ResidualIntelligenceError("cascade policy_version is not the admitted table")

    @property
    def stages(self) -> tuple[CascadeStage, ...]:
        return CASCADE_ORDER

    @property
    def cascade_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def evaluate(self, context: ResidualCascadeContext) -> ResidualCascadeWalk:
        if not isinstance(context, ResidualCascadeContext):
            raise ResidualIntelligenceError("evaluate requires ResidualCascadeContext")
        candidates: list[CascadeCandidate] = []
        rejections: list[CascadeHardRejection] = []
        for stage in CASCADE_ORDER:
            if stage is CascadeStage.HUMAN_REVIEW:
                candidates.append(_make_candidate(context, stage))
                continue
            reasons = evaluate_stage_constraints(context, stage)
            if reasons:
                rejections.append(
                    CascadeHardRejection(
                        stage=stage,
                        constraint=_primary_constraint(reasons),
                        reason_codes=reasons,
                    )
                )
            else:
                candidates.append(_make_candidate(context, stage))
        if not candidates:
            candidates.append(_make_candidate(context, CascadeStage.HUMAN_REVIEW))
        selected = candidates[0].stage
        return ResidualCascadeWalk(
            policy_version=self.policy_version,
            family=context.family,
            risk_class=context.risk_class,
            candidates=tuple(candidates),
            hard_rejections=tuple(rejections),
            selected_stage=selected,
            fallback_stage=CascadeStage.HUMAN_REVIEW,
            candidate_only=True,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "policy_version": self.policy_version,
            "stages": [item.value for item in CASCADE_ORDER],
        }
        if include_id:
            result["cascade_id"] = self.cascade_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualCascade:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"cascade_id"},
            noun="residual cascade",
        )
        stages = tuple(parse_cascade_stage(item) for item in (payload.get("stages") or ()))
        if stages != CASCADE_ORDER:
            raise ResidualIntelligenceError("cascade stages must match the admitted production order")
        result = cls(
            schema=str(payload.get("schema") or ""),
            policy_version=str(payload.get("policy_version") or ""),
        )
        claimed = str(payload.get("cascade_id") or "")
        if claimed and claimed != result.cascade_id:
            raise ResidualIntelligenceError("residual cascade identity mismatch")
        return result


__all__ = (
    "CASCADE_ORDER",
    "CASCADE_POLICY_VERSION",
    "CASCADE_SCHEMA",
    "CascadeCandidate",
    "CascadeHardRejection",
    "CascadeStage",
    "DETERMINISTIC_STAGES",
    "LEARNED_STAGES",
    "LOCAL_STAGES",
    "REASON_EXACT_CACHE",
    "REASON_FAMILY",
    "REASON_HUMAN_FALLBACK",
    "REASON_INFERENCE_POLICY",
    "REASON_PRIVACY",
    "REASON_PROVIDER_HEALTH",
    "REASON_RISK",
    "REASON_SAFE_FALLBACK",
    "REASON_SIMULATION",
    "REASON_VALIDATION",
    "REMOTE_STAGES",
    "ResidualCascade",
    "ResidualCascadeContext",
    "ResidualCascadeWalk",
    "effective_privacy_class",
    "evaluate_stage_constraints",
    "expert_id_for_stage",
    "parse_cascade_stage",
    "stage_is_deterministic",
    "stage_is_learned",
    "stage_is_local",
    "stage_is_remote",
)
