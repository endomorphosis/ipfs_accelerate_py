"""Task-family expert specifications for residual specialists.

Each expert binds one exact family boundary, a class in A through E, closed
input/output schemas, the family grammar, token and output limits, a risk
ceiling, privacy route policy, capabilities, a required validator, typed
errors, and abstention behavior.  Larger forms than the smallest admitted
class require a routing-changing quality delta.  Prose is not the default
output.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)
from .inventory import ResidualFamilyBoundary
from .residual_ir import ResidualTaskInput, ResidualTaskOutput
from .rights import TrainingCorpusAdmission
from .structured_decoding import grammar_for
from .task_families import (
    CANDIDATE_ONLY_AUTHORITY,
    EFFECT_CLASS,
    EXPERT_CLASS_VALUES,
    FAMILY_SPECS,
    MAX_FAMILY_INPUT_TOKENS,
    MAX_FAMILY_OUTPUT_TOKENS,
    PROPOSAL_RISKS,
    REASON_CAPABILITY_UNAVAILABLE,
    REASON_DATASET_UNADMITTED,
    REASON_FAMILY_MISMATCH,
    REASON_NOVEL_UNBOUNDED,
    REASON_PROSE_DEFAULT,
    REASON_RISK_CEILING,
    REASON_TOKEN_LIMIT,
    REASON_UNKNOWN_INPUT_FIELD,
    REASON_VALIDATOR_REQUIRED,
    ClosedSchema,
    ResidualTaskFamilySpec,
    assert_exact_family_boundary,
    assert_family_risk_admitted,
    family_spec_for,
    reject_similarity_grouping,
    reject_spec_examples,
)

EXPERT_SPEC_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-spec@1"
MODEL_SIZE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-model-size-policy@1"
)
MAX_SCORE_PPM: Final = 1_000_000
REASON_LARGER_FORM: Final = "larger_form_needs_routing_changing_quality_delta"
REASON_FORM_CEILING: Final = "requested_form_exceeds_family_maximum"
REASON_BEYOND_E: Final = "form_beyond_class_e_needs_quality_delta"

HARDWARE_BY_CLASS: Final[Mapping[str, str]] = {
    "A": "cpu-small-hermetic",
    "B": "cpu-small-hermetic",
    "C": "cpu-small-hermetic",
    "D": "cpu-medium-statistical",
    "E": "cpu-medium-local-proof",
}


class ExpertClass(str, Enum):
    """Closed specialist forms, smallest first.

    A: exact lookup / cache
    B: declarative rule / verified procedure
    C: linear / logistic model
    D: small ranker / encoder
    E: constrained structured decoder

    Parameter-efficient adapters, quantized local general models, and remote
    routes sit beyond class E and require a routing-changing quality delta.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


SMALLEST_FORM_ORDER: Final[tuple[ExpertClass, ...]] = (
    ExpertClass.A,
    ExpertClass.B,
    ExpertClass.C,
    ExpertClass.D,
    ExpertClass.E,
)
EXPERT_CLASS_FORMS: Final[Mapping[ExpertClass, str]] = {
    ExpertClass.A: "exact_lookup",
    ExpertClass.B: "declarative_rule",
    ExpertClass.C: "linear_logistic",
    ExpertClass.D: "ranker_encoder",
    ExpertClass.E: "structured_decoder",
}
BEYOND_E_FORMS: Final[tuple[str, ...]] = (
    "parameter_efficient_adapter",
    "quantized_local_general",
    "remote_standard",
    "remote_strong",
)


def expert_class_rank(value: ExpertClass | str) -> int:
    return SMALLEST_FORM_ORDER.index(ExpertClass(value))


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return required_text(value, name)


@dataclass(frozen=True)
class ModelSizePolicy:
    """Smallest-form-first policy; larger models need a quality delta."""

    smallest_form: ExpertClass
    maximum_form: ExpertClass
    larger_form_requires_quality_delta: bool = True
    routing_changing_quality_delta_ppm: int = 1
    quality_delta_evidence_cid: str = ""
    adapter_permitted: bool = False
    quantized_local_general_permitted: bool = False
    remote_permitted: bool = False
    schema: str = MODEL_SIZE_POLICY_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "policy_id",
            "smallest_form",
            "maximum_form",
            "larger_form_requires_quality_delta",
            "routing_changing_quality_delta_ppm",
            "quality_delta_evidence_cid",
            "adapter_permitted",
            "quantized_local_general_permitted",
            "remote_permitted",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != MODEL_SIZE_POLICY_SCHEMA:
            raise ResidualIntelligenceError("unsupported model size policy schema")
        object.__setattr__(self, "smallest_form", ExpertClass(self.smallest_form))
        object.__setattr__(self, "maximum_form", ExpertClass(self.maximum_form))
        if expert_class_rank(self.smallest_form) > expert_class_rank(self.maximum_form):
            raise ResidualIntelligenceError("smallest form exceeds maximum form")
        object.__setattr__(
            self,
            "larger_form_requires_quality_delta",
            _require_bool(
                self.larger_form_requires_quality_delta,
                "larger_form_requires_quality_delta",
            ),
        )
        object.__setattr__(
            self,
            "routing_changing_quality_delta_ppm",
            bounded_int(
                self.routing_changing_quality_delta_ppm,
                "routing_changing_quality_delta_ppm",
                minimum=0,
                maximum=MAX_SCORE_PPM,
            ),
        )
        object.__setattr__(
            self,
            "quality_delta_evidence_cid",
            _optional_cid(self.quality_delta_evidence_cid, "quality_delta_evidence_cid"),
        )
        for field in (
            "adapter_permitted",
            "quantized_local_general_permitted",
            "remote_permitted",
        ):
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        if self.beyond_e_permitted and not self.quality_delta_admits_larger:
            raise ResidualIntelligenceError(REASON_BEYOND_E)

    @property
    def policy_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def beyond_e_permitted(self) -> bool:
        return (
            self.adapter_permitted
            or self.quantized_local_general_permitted
            or self.remote_permitted
        )

    @property
    def quality_delta_admits_larger(self) -> bool:
        return bool(self.quality_delta_evidence_cid) and self.routing_changing_quality_delta_ppm > 0

    def admits(
        self,
        requested: ExpertClass | str,
        *,
        quality_delta_ppm: int = 0,
        evidence_cid: str = "",
        beyond_e_form: str = "",
    ) -> bool:
        form = ExpertClass(requested)
        if expert_class_rank(form) > expert_class_rank(self.maximum_form):
            return False
        if beyond_e_form:
            if beyond_e_form not in BEYOND_E_FORMS:
                return False
            if beyond_e_form == "parameter_efficient_adapter" and not self.adapter_permitted:
                return False
            if (
                beyond_e_form == "quantized_local_general"
                and not self.quantized_local_general_permitted
            ):
                return False
            if beyond_e_form in {"remote_standard", "remote_strong"} and not self.remote_permitted:
                return False
            return self._delta_satisfied(quality_delta_ppm, evidence_cid)
        if expert_class_rank(form) <= expert_class_rank(self.smallest_form):
            return True
        if not self.larger_form_requires_quality_delta:
            return True
        return self._delta_satisfied(quality_delta_ppm, evidence_cid)

    def _delta_satisfied(self, quality_delta_ppm: int, evidence_cid: str) -> bool:
        if not self.larger_form_requires_quality_delta:
            return True
        if not evidence_cid or evidence_cid != self.quality_delta_evidence_cid:
            return False
        if type(quality_delta_ppm) is not int:
            return False
        return quality_delta_ppm >= self.routing_changing_quality_delta_ppm > 0

    def admit(
        self,
        requested: ExpertClass | str,
        *,
        quality_delta_ppm: int = 0,
        evidence_cid: str = "",
        beyond_e_form: str = "",
    ) -> ExpertClass:
        form = ExpertClass(requested)
        if expert_class_rank(form) > expert_class_rank(self.maximum_form):
            raise ResidualIntelligenceError(REASON_FORM_CEILING)
        if not self.admits(
            form,
            quality_delta_ppm=quality_delta_ppm,
            evidence_cid=evidence_cid,
            beyond_e_form=beyond_e_form,
        ):
            raise ResidualIntelligenceError(REASON_LARGER_FORM)
        return form

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "smallest_form": self.smallest_form.value,
            "maximum_form": self.maximum_form.value,
            "larger_form_requires_quality_delta": self.larger_form_requires_quality_delta,
            "routing_changing_quality_delta_ppm": self.routing_changing_quality_delta_ppm,
            "quality_delta_evidence_cid": self.quality_delta_evidence_cid,
            "adapter_permitted": self.adapter_permitted,
            "quantized_local_general_permitted": self.quantized_local_general_permitted,
            "remote_permitted": self.remote_permitted,
        }
        if include_id:
            result["policy_id"] = self.policy_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelSizePolicy:
        reject_spec_examples(payload, noun="model size policy")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"policy_id"},
            noun="model size policy",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            smallest_form=ExpertClass(str(payload.get("smallest_form") or "")),
            maximum_form=ExpertClass(str(payload.get("maximum_form") or "")),
            larger_form_requires_quality_delta=payload.get("larger_form_requires_quality_delta"),
            routing_changing_quality_delta_ppm=payload.get("routing_changing_quality_delta_ppm"),
            quality_delta_evidence_cid=str(payload.get("quality_delta_evidence_cid") or ""),
            adapter_permitted=payload.get("adapter_permitted"),
            quantized_local_general_permitted=payload.get("quantized_local_general_permitted"),
            remote_permitted=payload.get("remote_permitted"),
        )
        claimed = str(payload.get("policy_id") or "")
        if claimed and claimed != result.policy_id:
            raise ResidualIntelligenceError("model size policy identity mismatch")
        return result

    @classmethod
    def for_family(
        cls,
        family_spec: ResidualTaskFamilySpec,
        *,
        quality_delta_evidence_cid: str = "",
        routing_changing_quality_delta_ppm: int = 1,
        adapter_permitted: bool = False,
        quantized_local_general_permitted: bool = False,
        remote_permitted: bool = False,
    ) -> ModelSizePolicy:
        classes = tuple(ExpertClass(item) for item in family_spec.allowed_expert_classes)
        return cls(
            smallest_form=classes[0],
            maximum_form=classes[-1],
            larger_form_requires_quality_delta=True,
            routing_changing_quality_delta_ppm=routing_changing_quality_delta_ppm,
            quality_delta_evidence_cid=quality_delta_evidence_cid,
            adapter_permitted=adapter_permitted,
            quantized_local_general_permitted=quantized_local_general_permitted,
            remote_permitted=remote_permitted,
        )


@dataclass(frozen=True)
class ResidualExpertSpec:
    """One family-bounded expert at an exact class in A through E."""

    expert_id: str
    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    model_size_policy: ModelSizePolicy
    family_spec_id: str
    family_boundary_id: str
    input_schema: ClosedSchema
    output_schema: ClosedSchema
    grammar_id: str
    risk_ceiling: RiskClass
    privacy_class: PrivacyClass
    privacy_route_policy: str
    authority_class: str
    validation_contract: str
    error_behavior: str
    abstention_behavior: str
    capabilities: tuple[str, ...]
    hardware_class: str
    runtime_requirements: tuple[str, ...]
    input_token_limit: int
    output_token_limit: int
    maximum_output_bytes: int
    validator_required: bool = True
    emit_prose_by_default: bool = False
    prose_token_budget: int = 0
    candidate_only: bool = True
    evaluation_corpus_admission_id: str = ""
    training_corpus_admission_id: str = ""
    schema: str = EXPERT_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "spec_id",
            "expert_id",
            "task_family",
            "expert_class",
            "implementation_form",
            "model_size_policy",
            "family_spec_id",
            "family_boundary_id",
            "input_schema",
            "output_schema",
            "grammar_id",
            "risk_ceiling",
            "privacy_class",
            "privacy_route_policy",
            "authority_class",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "capabilities",
            "hardware_class",
            "runtime_requirements",
            "input_token_limit",
            "output_token_limit",
            "maximum_output_bytes",
            "validator_required",
            "emit_prose_by_default",
            "prose_token_budget",
            "candidate_only",
            "evaluation_corpus_admission_id",
            "training_corpus_admission_id",
            "effect_class",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != EXPERT_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual expert spec schema")
        object.__setattr__(
            self, "expert_id", required_text(self.expert_id, "expert_id", max_bytes=256)
        )
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", ExpertClass(self.expert_class))
        if not isinstance(self.model_size_policy, ModelSizePolicy):
            raise ResidualIntelligenceError("expert spec requires a typed ModelSizePolicy")
        family = family_spec_for(self.task_family)
        object.__setattr__(
            self, "family_spec_id", required_text(self.family_spec_id, "family_spec_id")
        )
        if self.family_spec_id != family.family_spec_id:
            raise ResidualIntelligenceError("expert spec family_spec_id mismatch")
        object.__setattr__(
            self,
            "family_boundary_id",
            required_text(self.family_boundary_id, "family_boundary_id"),
        )
        if self.family_boundary_id != family.boundary_id:
            raise ResidualIntelligenceError("expert spec family boundary identity mismatch")
        if not isinstance(self.input_schema, ClosedSchema) or not isinstance(
            self.output_schema, ClosedSchema
        ):
            raise ResidualIntelligenceError("expert spec requires typed closed schemas")
        if self.input_schema.to_dict() != family.input_schema.to_dict():
            raise ResidualIntelligenceError("expert input schema must match the family schema")
        if self.output_schema.to_dict() != family.output_schema.to_dict():
            raise ResidualIntelligenceError("expert output schema must match the family schema")
        object.__setattr__(self, "grammar_id", required_text(self.grammar_id, "grammar_id"))
        if self.grammar_id != grammar_for(self.task_family).grammar_id:
            raise ResidualIntelligenceError("expert grammar_id must match the family grammar")
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        if self.risk_ceiling is not family.risk_ceiling:
            raise ResidualIntelligenceError("expert risk ceiling must match the family ceiling")
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        if self.privacy_class is not family.privacy_class:
            raise ResidualIntelligenceError("expert privacy class must match the family")
        object.__setattr__(
            self,
            "privacy_route_policy",
            required_text(self.privacy_route_policy, "privacy_route_policy", max_bytes=64),
        )
        if self.privacy_route_policy != family.privacy_route_policy:
            raise ResidualIntelligenceError("expert privacy route must match the family")
        object.__setattr__(
            self,
            "authority_class",
            required_text(self.authority_class, "authority_class", max_bytes=64),
        )
        if self.authority_class.casefold() != CANDIDATE_ONLY_AUTHORITY:
            raise ResidualIntelligenceError("expert authority_class must be candidate_only")
        for field in ("validation_contract", "error_behavior", "abstention_behavior"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.validation_contract != family.validation_contract:
            raise ResidualIntelligenceError("expert validator must match the family contract")
        if self.error_behavior != family.error_behavior:
            raise ResidualIntelligenceError("expert error behavior must match the family")
        if self.abstention_behavior != family.abstention_behavior:
            raise ResidualIntelligenceError("expert abstention behavior must match the family")
        object.__setattr__(self, "capabilities", text_tuple(self.capabilities, "capabilities"))
        if set(self.capabilities) != set(family.capabilities):
            raise ResidualIntelligenceError("expert capabilities must match the family")
        object.__setattr__(
            self,
            "hardware_class",
            required_text(self.hardware_class, "hardware_class", max_bytes=64),
        )
        object.__setattr__(
            self,
            "runtime_requirements",
            text_tuple(self.runtime_requirements, "runtime_requirements", allow_empty=False),
        )
        object.__setattr__(
            self,
            "input_token_limit",
            bounded_int(
                self.input_token_limit,
                "input_token_limit",
                minimum=1,
                maximum=MAX_FAMILY_INPUT_TOKENS,
            ),
        )
        object.__setattr__(
            self,
            "output_token_limit",
            bounded_int(
                self.output_token_limit,
                "output_token_limit",
                minimum=1,
                maximum=MAX_FAMILY_OUTPUT_TOKENS,
            ),
        )
        object.__setattr__(
            self,
            "maximum_output_bytes",
            bounded_int(
                self.maximum_output_bytes,
                "maximum_output_bytes",
                minimum=128,
                maximum=family.maximum_output_bytes,
            ),
        )
        if self.input_token_limit != family.input_token_limit:
            raise ResidualIntelligenceError("expert input token limit must match the family")
        if self.output_token_limit != family.output_token_limit:
            raise ResidualIntelligenceError("expert output token limit must match the family")
        if self.maximum_output_bytes != family.maximum_output_bytes:
            raise ResidualIntelligenceError("expert output byte limit must match the family grammar")
        object.__setattr__(
            self, "validator_required", _require_bool(self.validator_required, "validator_required")
        )
        if self.validator_required is not True:
            raise ResidualIntelligenceError("every expert spec requires an independent validator")
        object.__setattr__(
            self,
            "emit_prose_by_default",
            _require_bool(self.emit_prose_by_default, "emit_prose_by_default"),
        )
        if self.emit_prose_by_default is not False:
            raise ResidualIntelligenceError(REASON_PROSE_DEFAULT)
        object.__setattr__(
            self,
            "prose_token_budget",
            bounded_int(self.prose_token_budget, "prose_token_budget", minimum=0, maximum=0),
        )
        object.__setattr__(
            self, "candidate_only", _require_bool(self.candidate_only, "candidate_only")
        )
        if self.candidate_only is not True:
            raise ResidualIntelligenceError("expert specs must remain candidate_only=true")
        object.__setattr__(
            self,
            "evaluation_corpus_admission_id",
            _optional_cid(
                self.evaluation_corpus_admission_id, "evaluation_corpus_admission_id"
            ),
        )
        object.__setattr__(
            self,
            "training_corpus_admission_id",
            _optional_cid(self.training_corpus_admission_id, "training_corpus_admission_id"),
        )
        if self.expert_class.value not in family.allowed_expert_classes:
            raise ResidualIntelligenceError("expert class is outside the family form ceiling")
        if expert_class_rank(self.expert_class) > expert_class_rank(
            self.model_size_policy.maximum_form
        ):
            raise ResidualIntelligenceError(REASON_FORM_CEILING)
        if self.model_size_policy.smallest_form.value != family.allowed_expert_classes[0]:
            raise ResidualIntelligenceError("model size policy must start at the family's smallest form")
        if self.model_size_policy.maximum_form.value != family.allowed_expert_classes[-1]:
            raise ResidualIntelligenceError("model size policy must end at the family's maximum form")

    @property
    def spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def implementation_form(self) -> str:
        return EXPERT_CLASS_FORMS[self.expert_class]

    @property
    def effect_class(self) -> str:
        return EFFECT_CLASS

    @property
    def family_spec(self) -> ResidualTaskFamilySpec:
        return family_spec_for(self.task_family)

    @property
    def family_boundary(self) -> ResidualFamilyBoundary:
        return self.family_spec.as_boundary()

    @property
    def proposal_tier(self) -> bool:
        return self.risk_ceiling in PROPOSAL_RISKS

    @property
    def always_abstain(self) -> bool:
        return self.family_spec.always_abstain

    def admit_form(
        self,
        requested: ExpertClass | str | None = None,
        *,
        quality_delta_ppm: int = 0,
        evidence_cid: str = "",
        beyond_e_form: str = "",
    ) -> ExpertClass:
        form = self.expert_class if requested is None else ExpertClass(requested)
        return self.model_size_policy.admit(
            form,
            quality_delta_ppm=quality_delta_ppm,
            evidence_cid=evidence_cid,
            beyond_e_form=beyond_e_form,
        )

    def bind_corpus_admission(self, admission: TrainingCorpusAdmission) -> None:
        referenced = {
            self.evaluation_corpus_admission_id,
            self.training_corpus_admission_id,
        } - {""}
        if not referenced:
            raise ResidualIntelligenceError("expert spec carries no dataset reference")
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError(
                "dataset reference must be a typed TrainingCorpusAdmission"
            )
        if admission.admission_id not in referenced:
            raise ResidualIntelligenceError("dataset reference does not match admission identity")
        if admission.admission_decision.value != "admitted":
            raise ResidualIntelligenceError(REASON_DATASET_UNADMITTED)
        if self.training_corpus_admission_id and not admission.can_train:
            raise ResidualIntelligenceError(REASON_DATASET_UNADMITTED)

    def evaluate_input(
        self,
        task_input: ResidualTaskInput,
        *,
        capability_available: bool = True,
        quality_delta_ppm: int = 0,
        evidence_cid: str = "",
    ) -> tuple[ExpertDisposition, tuple[str, ...]]:
        reasons = list(self.family_spec.validate_task_input(task_input))
        if not capability_available:
            reasons.append(REASON_CAPABILITY_UNAVAILABLE)
        if not self.model_size_policy.admits(
            self.expert_class,
            quality_delta_ppm=quality_delta_ppm,
            evidence_cid=evidence_cid,
        ):
            reasons.append(REASON_LARGER_FORM)
        unique = tuple(dict.fromkeys(reasons))
        if REASON_CAPABILITY_UNAVAILABLE in unique:
            return ExpertDisposition.CAPABILITY_UNAVAILABLE, unique
        reject_reasons = {
            REASON_FAMILY_MISMATCH,
            REASON_RISK_CEILING,
            REASON_UNKNOWN_INPUT_FIELD,
            REASON_TOKEN_LIMIT,
            REASON_LARGER_FORM,
        }
        if any(item in reject_reasons for item in unique):
            return ExpertDisposition.REJECT_INPUT, unique
        if REASON_NOVEL_UNBOUNDED in unique or self.always_abstain:
            return ExpertDisposition.ABSTAIN, unique or (REASON_NOVEL_UNBOUNDED,)
        return ExpertDisposition.VALIDATION_REQUIRED, (REASON_VALIDATOR_REQUIRED,)

    def validate_output(self, task_output: ResidualTaskOutput) -> None:
        self.family_spec.validate_task_output(task_output)
        if task_output.candidate_only is not True:
            raise ResidualIntelligenceError("expert outputs must remain candidate_only=true")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "expert_id": self.expert_id,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "implementation_form": self.implementation_form,
            "model_size_policy": self.model_size_policy.to_dict(),
            "family_spec_id": self.family_spec_id,
            "family_boundary_id": self.family_boundary_id,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "grammar_id": self.grammar_id,
            "risk_ceiling": self.risk_ceiling.value,
            "privacy_class": self.privacy_class.value,
            "privacy_route_policy": self.privacy_route_policy,
            "authority_class": self.authority_class,
            "validation_contract": self.validation_contract,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "capabilities": list(self.capabilities),
            "hardware_class": self.hardware_class,
            "runtime_requirements": list(self.runtime_requirements),
            "input_token_limit": self.input_token_limit,
            "output_token_limit": self.output_token_limit,
            "maximum_output_bytes": self.maximum_output_bytes,
            "validator_required": True,
            "emit_prose_by_default": False,
            "prose_token_budget": 0,
            "candidate_only": True,
            "evaluation_corpus_admission_id": self.evaluation_corpus_admission_id,
            "training_corpus_admission_id": self.training_corpus_admission_id,
            "effect_class": self.effect_class,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualExpertSpec:
        reject_spec_examples(payload, noun="expert spec")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"spec_id", "implementation_form", "effect_class"},
            noun="expert spec",
        )
        policy_payload = payload.get("model_size_policy")
        input_schema_payload = payload.get("input_schema")
        output_schema_payload = payload.get("output_schema")
        if not isinstance(policy_payload, Mapping):
            raise ResidualIntelligenceError("model_size_policy must be an object")
        if not isinstance(input_schema_payload, Mapping) or not isinstance(
            output_schema_payload, Mapping
        ):
            raise ResidualIntelligenceError("expert spec schemas must be objects")
        result = cls(
            schema=str(payload.get("schema") or ""),
            expert_id=str(payload.get("expert_id") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=ExpertClass(str(payload.get("expert_class") or "")),
            model_size_policy=ModelSizePolicy.from_dict(policy_payload),
            family_spec_id=str(payload.get("family_spec_id") or ""),
            family_boundary_id=str(payload.get("family_boundary_id") or ""),
            input_schema=ClosedSchema.from_dict(input_schema_payload),
            output_schema=ClosedSchema.from_dict(output_schema_payload),
            grammar_id=str(payload.get("grammar_id") or ""),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            privacy_route_policy=str(payload.get("privacy_route_policy") or ""),
            authority_class=str(payload.get("authority_class") or ""),
            validation_contract=str(payload.get("validation_contract") or ""),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            capabilities=tuple(payload.get("capabilities") or ()),
            hardware_class=str(payload.get("hardware_class") or ""),
            runtime_requirements=tuple(payload.get("runtime_requirements") or ()),
            input_token_limit=payload.get("input_token_limit"),
            output_token_limit=payload.get("output_token_limit"),
            maximum_output_bytes=payload.get("maximum_output_bytes"),
            validator_required=payload.get("validator_required"),
            emit_prose_by_default=payload.get("emit_prose_by_default"),
            prose_token_budget=payload.get("prose_token_budget"),
            candidate_only=payload.get("candidate_only"),
            evaluation_corpus_admission_id=str(
                payload.get("evaluation_corpus_admission_id") or ""
            ),
            training_corpus_admission_id=str(payload.get("training_corpus_admission_id") or ""),
        )
        claimed = str(payload.get("spec_id") or "")
        if claimed and claimed != result.spec_id:
            raise ResidualIntelligenceError("expert spec identity mismatch")
        return result

    @classmethod
    def from_family(
        cls,
        family_spec: ResidualTaskFamilySpec,
        *,
        expert_class: ExpertClass | str = ExpertClass.A,
        model_size_policy: ModelSizePolicy | None = None,
        quality_delta_evidence_cid: str = "",
        routing_changing_quality_delta_ppm: int = 1,
    ) -> ResidualExpertSpec:
        if not isinstance(family_spec, ResidualTaskFamilySpec):
            raise ResidualIntelligenceError("from_family requires ResidualTaskFamilySpec")
        form = ExpertClass(expert_class)
        policy = model_size_policy or ModelSizePolicy.for_family(
            family_spec,
            quality_delta_evidence_cid=quality_delta_evidence_cid,
            routing_changing_quality_delta_ppm=routing_changing_quality_delta_ppm,
        )
        hardware = HARDWARE_BY_CLASS[form.value]
        return cls(
            expert_id=f"residual-expert:{family_spec.task_family.value}:{form.value}@1",
            task_family=family_spec.task_family,
            expert_class=form,
            model_size_policy=policy,
            family_spec_id=family_spec.family_spec_id,
            family_boundary_id=family_spec.boundary_id,
            input_schema=family_spec.input_schema,
            output_schema=family_spec.output_schema,
            grammar_id=grammar_for(family_spec.task_family).grammar_id,
            risk_ceiling=family_spec.risk_ceiling,
            privacy_class=family_spec.privacy_class,
            privacy_route_policy=family_spec.privacy_route_policy,
            authority_class=family_spec.authority_class,
            validation_contract=family_spec.validation_contract,
            error_behavior=family_spec.error_behavior,
            abstention_behavior=family_spec.abstention_behavior,
            capabilities=family_spec.capabilities,
            hardware_class=hardware,
            runtime_requirements=("provider-free", hardware),
            input_token_limit=family_spec.input_token_limit,
            output_token_limit=family_spec.output_token_limit,
            maximum_output_bytes=family_spec.maximum_output_bytes,
        )


def _build_expert_specs() -> dict[tuple[ResidualTaskFamily, ExpertClass], ResidualExpertSpec]:
    registry: dict[tuple[ResidualTaskFamily, ExpertClass], ResidualExpertSpec] = {}
    for family, family_spec in FAMILY_SPECS.items():
        policy = ModelSizePolicy.for_family(family_spec)
        for class_name in family_spec.allowed_expert_classes:
            form = ExpertClass(class_name)
            registry[(family, form)] = ResidualExpertSpec.from_family(
                family_spec, expert_class=form, model_size_policy=policy
            )
    return registry


EXPERT_SPECS: Final[Mapping[tuple[ResidualTaskFamily, ExpertClass], ResidualExpertSpec]] = (
    _build_expert_specs()
)


def expert_spec_for(
    task_family: ResidualTaskFamily | str,
    expert_class: ExpertClass | str = ExpertClass.A,
) -> ResidualExpertSpec:
    key = (ResidualTaskFamily(task_family), ExpertClass(expert_class))
    spec = EXPERT_SPECS.get(key)
    if spec is None:
        raise ResidualIntelligenceError("no expert spec for requested family and class")
    return spec


def smallest_expert_for(task_family: ResidualTaskFamily | str) -> ResidualExpertSpec:
    family = family_spec_for(task_family)
    return expert_spec_for(family.task_family, family.allowed_expert_classes[0])


def experts_of_class(expert_class: ExpertClass | str) -> tuple[ResidualExpertSpec, ...]:
    form = ExpertClass(expert_class)
    return tuple(spec for (_family, cls), spec in EXPERT_SPECS.items() if cls is form)


def default_expert_specs() -> tuple[ResidualExpertSpec, ...]:
    return tuple(smallest_expert_for(family) for family in ResidualTaskFamily)


def assert_validator_required(spec: ResidualExpertSpec | ResidualTaskFamilySpec) -> None:
    if spec.validator_required is not True or not spec.validation_contract:
        raise ResidualIntelligenceError(REASON_VALIDATOR_REQUIRED)


def assert_no_prose_default(spec: ResidualExpertSpec | ResidualTaskFamilySpec) -> None:
    if spec.emit_prose_by_default is not False or spec.prose_token_budget != 0:
        raise ResidualIntelligenceError(REASON_PROSE_DEFAULT)


__all__ = (
    "BEYOND_E_FORMS",
    "EXPERT_CLASS_FORMS",
    "EXPERT_CLASS_VALUES",
    "EXPERT_SPECS",
    "EXPERT_SPEC_SCHEMA",
    "MODEL_SIZE_POLICY_SCHEMA",
    "REASON_BEYOND_E",
    "REASON_FORM_CEILING",
    "REASON_LARGER_FORM",
    "SMALLEST_FORM_ORDER",
    "ExpertClass",
    "ModelSizePolicy",
    "ResidualExpertSpec",
    "ResidualTaskFamilySpec",
    "assert_exact_family_boundary",
    "assert_family_risk_admitted",
    "assert_no_prose_default",
    "assert_validator_required",
    "default_expert_specs",
    "expert_class_rank",
    "expert_spec_for",
    "experts_of_class",
    "family_spec_for",
    "reject_similarity_grouping",
    "smallest_expert_for",
)
