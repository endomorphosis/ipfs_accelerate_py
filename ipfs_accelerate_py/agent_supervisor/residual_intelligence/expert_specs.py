"""Closed expert specifications bound to residual task-family contracts.

Every expert has a class A-E, closed input/output schema, grammar,
enumerations, output limits, abstention representation, validation policy,
family boundary, risk ceiling, hardware/runtime requirements, and privacy
route policy.  A larger form is eligible only with current held-out evidence
of a routing-changing quality gain for the exact family and risk group.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    UnknownFieldError,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)
from .inventory import ResidualFamilyBoundary
from .residual_ir import ResidualTaskInput
from .rights import TrainingCorpusAdmission
from .structured_decoding import ExpertGrammar, grammar_for
from .task_families import (
    ABSTAIN_OUTPUT_CLASS,
    CANDIDATE_ONLY_AUTHORITY,
    CLOSED_EXPERT_CLASS_LETTERS,
    ERROR_INVALID_OUTPUT,
    FAMILY_SPEC_REGISTRY_VERSION,
    REASON_PROSE_DEFAULT,
    REASON_UNSUPPORTED_FAMILY_RISK,
    REASON_VALIDATOR_REQUIRED,
    ResidualTaskFamilySpec,
    family_spec_for,
)

EXPERT_SPEC_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-spec@1"
EXPERT_SPEC_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-expert-spec-registry@1"
)
MODEL_SIZE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-model-size-policy@1"
)
EXPERT_SPEC_REGISTRY_VERSION: Final = "vrif-010-expert-specs@1"
REASON_UNSUPPORTED_EXPERT_CLASS: Final = "unsupported_family_class"
REASON_LARGER_FORM: Final = "larger form needs a routing-changing quality delta"
REASON_CURRENT_EVIDENCE: Final = "current held-out evidence required"
REASON_TRAINING_UNAVAILABLE: Final = "training_unavailable"
REASON_EXAMPLES_FORBIDDEN: Final = "specifications_carry_no_examples"
REASON_GLOBAL_SIZE: Final = "global bigger-is-better policy is forbidden"
REASON_SKIP_SMALLER: Final = "smallest-form-order forbids skipping a smaller class without evidence"
MIN_ROUTING_CHANGING_DELTA_PPM: Final = 10_000
MAX_QUALITY_DELTA_PPM: Final = 1_000_000
_FORBIDDEN_EXAMPLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "examples",
        "training_examples",
        "evaluation_examples",
        "example_payloads",
        "private_bodies",
        "raw_bodies",
    }
)


class ExpertClass(str, Enum):
    """Local specialist form classes in smallest-to-largest order."""

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
EXPERT_CLASS_FORMS: Final[Mapping[ExpertClass, tuple[str, ...]]] = {
    ExpertClass.A: ("exact_lookup",),
    ExpertClass.B: ("verified_procedure", "declarative_rule", "deterministic_ranking"),
    ExpertClass.C: ("linear_logistic",),
    ExpertClass.D: ("small_ranker",),
    ExpertClass.E: ("constrained_structured_decoder",),
}
EXPERT_CLASS_HARDWARE: Final[Mapping[ExpertClass, tuple[str, ...]]] = {
    ExpertClass.A: ("cpu-small-hermetic",),
    ExpertClass.B: ("cpu-small-hermetic",),
    ExpertClass.C: ("cpu-small-batch",),
    ExpertClass.D: ("cpu-medium-batch",),
    ExpertClass.E: ("cpu-gpu-optional-bounded",),
}
EXPERT_CLASS_RUNTIME: Final[Mapping[ExpertClass, tuple[str, ...]]] = {
    ExpertClass.A: ("deterministic", "provider_free"),
    ExpertClass.B: ("deterministic", "provider_free"),
    ExpertClass.C: ("integer_linear", "provider_free"),
    ExpertClass.D: ("local_encoder", "provider_free"),
    ExpertClass.E: ("constrained_decoder", "provider_free"),
}


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def expert_class_rank(value: ExpertClass | str) -> int:
    letter = ExpertClass(value)
    return SMALLEST_FORM_ORDER.index(letter)


def parse_expert_class(value: ExpertClass | str) -> ExpertClass:
    if isinstance(value, ExpertClass):
        return value
    text = required_text(value, "expert_class", max_bytes=1)
    if text not in CLOSED_EXPERT_CLASS_LETTERS:
        raise ResidualIntelligenceError("expert_class must be one of A-E")
    return ExpertClass(text)


def enumerations_from_grammar(grammar: ExpertGrammar) -> dict[str, tuple[str, ...]]:
    enumerations: dict[str, tuple[str, ...]] = {"output_classes": grammar.output_classes}
    for name, contract in grammar.field_contracts.items():
        if contract.allowed_values:
            enumerations[name] = contract.allowed_values
    return enumerations


@dataclass(frozen=True)
class ModelSizePolicy:
    """Smallest-form order; larger classes need a routing-changing quality delta."""

    form_order: tuple[ExpertClass, ...] = SMALLEST_FORM_ORDER
    minimum_routing_changing_delta_ppm: int = MIN_ROUTING_CHANGING_DELTA_PPM
    require_current_held_out_evidence: bool = True
    allow_global_bigger_is_better: bool = False
    allow_skip_smaller_form: bool = False
    schema: str = MODEL_SIZE_POLICY_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "policy_id",
            "form_order",
            "minimum_routing_changing_delta_ppm",
            "require_current_held_out_evidence",
            "allow_global_bigger_is_better",
            "allow_skip_smaller_form",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != MODEL_SIZE_POLICY_SCHEMA:
            raise ResidualIntelligenceError("unsupported model size policy schema")
        order = tuple(parse_expert_class(item) for item in self.form_order)
        if order != SMALLEST_FORM_ORDER:
            raise ResidualIntelligenceError("model size policy form_order must be A-E")
        object.__setattr__(self, "form_order", order)
        object.__setattr__(
            self,
            "minimum_routing_changing_delta_ppm",
            bounded_int(
                self.minimum_routing_changing_delta_ppm,
                "minimum_routing_changing_delta_ppm",
                minimum=1,
                maximum=MAX_QUALITY_DELTA_PPM,
            ),
        )
        object.__setattr__(
            self,
            "require_current_held_out_evidence",
            _require_bool(
                self.require_current_held_out_evidence,
                "require_current_held_out_evidence",
            ),
        )
        object.__setattr__(
            self,
            "allow_global_bigger_is_better",
            _require_bool(self.allow_global_bigger_is_better, "allow_global_bigger_is_better"),
        )
        if self.allow_global_bigger_is_better:
            raise ResidualIntelligenceError(REASON_GLOBAL_SIZE)
        object.__setattr__(
            self,
            "allow_skip_smaller_form",
            _require_bool(self.allow_skip_smaller_form, "allow_skip_smaller_form"),
        )
        if self.allow_skip_smaller_form:
            raise ResidualIntelligenceError(REASON_SKIP_SMALLER)

    @property
    def policy_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def admit_requested_class(
        self,
        family_spec: ResidualTaskFamilySpec,
        requested: ExpertClass | str,
        *,
        risk: RiskClass | str,
        quality_delta_ppm: int = 0,
        routing_changing: bool = False,
        evidence_current: bool = False,
        compared_class: ExpertClass | str | None = None,
        admission: TrainingCorpusAdmission | None = None,
    ) -> ExpertClass:
        if not isinstance(family_spec, ResidualTaskFamilySpec):
            raise ResidualIntelligenceError("family_spec must be ResidualTaskFamilySpec")
        family_spec.reject_unsupported_risk(risk)
        wanted = parse_expert_class(requested)
        if not family_spec.allows_expert_class(wanted.value):
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_EXPERT_CLASS)
        smallest = parse_expert_class(family_spec.smallest_expert_class)
        if expert_class_rank(wanted) < expert_class_rank(smallest):
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_EXPERT_CLASS)
        if wanted is smallest:
            return wanted
        baseline = parse_expert_class(compared_class) if compared_class is not None else smallest
        if expert_class_rank(wanted) - expert_class_rank(baseline) > 1:
            raise ResidualIntelligenceError(REASON_SKIP_SMALLER)
        if self.require_current_held_out_evidence and not evidence_current:
            raise ResidualIntelligenceError(REASON_CURRENT_EVIDENCE)
        if admission is not None:
            if not isinstance(admission, TrainingCorpusAdmission):
                raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
            if admission.admission_decision is not TrainingAvailability.ADMITTED:
                raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
        delta = bounded_int(
            quality_delta_ppm,
            "quality_delta_ppm",
            minimum=0,
            maximum=MAX_QUALITY_DELTA_PPM,
        )
        if not _require_bool(routing_changing, "routing_changing"):
            raise ResidualIntelligenceError(REASON_LARGER_FORM)
        if delta < self.minimum_routing_changing_delta_ppm:
            raise ResidualIntelligenceError(REASON_LARGER_FORM)
        return wanted

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "form_order": [item.value for item in self.form_order],
            "minimum_routing_changing_delta_ppm": self.minimum_routing_changing_delta_ppm,
            "require_current_held_out_evidence": True,
            "allow_global_bigger_is_better": False,
            "allow_skip_smaller_form": False,
        }
        if include_id:
            result["policy_id"] = self.policy_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelSizePolicy:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"policy_id"},
            noun="model size policy",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            form_order=tuple(payload.get("form_order") or ()),
            minimum_routing_changing_delta_ppm=payload.get("minimum_routing_changing_delta_ppm"),
            require_current_held_out_evidence=payload.get("require_current_held_out_evidence"),
            allow_global_bigger_is_better=payload.get("allow_global_bigger_is_better"),
            allow_skip_smaller_form=payload.get("allow_skip_smaller_form"),
        )
        claimed = str(payload.get("policy_id") or "")
        if claimed and claimed != result.policy_id:
            raise ResidualIntelligenceError("model size policy identity mismatch")
        return result


DEFAULT_MODEL_SIZE_POLICY: Final[ModelSizePolicy] = ModelSizePolicy()


@dataclass(frozen=True)
class ResidualExpertSpec:
    """One family-bounded expert contract at a single class A-E."""

    expert_id: str
    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    family_spec_id: str
    family_boundary_id: str
    input_schema: str
    output_schema: str
    grammar_id: str
    enumerations: Mapping[str, tuple[str, ...]]
    forms: tuple[str, ...]
    maximum_input_tokens: int
    maximum_output_tokens: int
    maximum_output_bytes: int
    abstention_output_class: str
    validation_policy: str
    validator_identity: str
    independent_validator_required: bool
    risk_ceiling: RiskClass
    allowed_risk_classes: tuple[RiskClass, ...]
    privacy_class: str
    privacy_route_policy: str
    hardware_requirements: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    capabilities: tuple[str, ...]
    error_behavior: str
    abstention_behavior: str
    emit_prose_by_default: bool
    always_abstain: bool
    candidate_only: bool = True
    authority_class: str = CANDIDATE_ONLY_AUTHORITY
    evaluation_corpus_admission_id: str = ""
    schema: str = EXPERT_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "spec_id",
            "expert_id",
            "task_family",
            "expert_class",
            "family_spec_id",
            "family_boundary_id",
            "input_schema",
            "output_schema",
            "grammar_id",
            "enumerations",
            "forms",
            "maximum_input_tokens",
            "maximum_output_tokens",
            "maximum_output_bytes",
            "abstention_output_class",
            "validation_policy",
            "validator_identity",
            "independent_validator_required",
            "risk_ceiling",
            "allowed_risk_classes",
            "privacy_class",
            "privacy_route_policy",
            "hardware_requirements",
            "runtime_requirements",
            "capabilities",
            "error_behavior",
            "abstention_behavior",
            "emit_prose_by_default",
            "always_abstain",
            "candidate_only",
            "authority_class",
            "evaluation_corpus_admission_id",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != EXPERT_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual expert spec schema")
        object.__setattr__(self, "expert_id", required_text(self.expert_id, "expert_id"))
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", parse_expert_class(self.expert_class))
        for field in (
            "family_spec_id",
            "family_boundary_id",
            "input_schema",
            "output_schema",
            "grammar_id",
            "validation_policy",
            "validator_identity",
            "privacy_class",
            "privacy_route_policy",
            "error_behavior",
            "abstention_behavior",
            "authority_class",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.authority_class != CANDIDATE_ONLY_AUTHORITY:
            raise ResidualIntelligenceError("expert authority_class must be candidate_only")
        enumerations = _enumerations_payload(self.enumerations)
        object.__setattr__(self, "enumerations", enumerations)
        object.__setattr__(self, "forms", text_tuple(self.forms, "forms", allow_empty=False, max_items=8))
        expected_forms = EXPERT_CLASS_FORMS[self.expert_class]
        if self.forms != expected_forms:
            raise ResidualIntelligenceError("expert forms must match the closed class A-E mapping")
        object.__setattr__(
            self,
            "maximum_input_tokens",
            bounded_int(self.maximum_input_tokens, "maximum_input_tokens", minimum=32, maximum=8_192),
        )
        object.__setattr__(
            self,
            "maximum_output_tokens",
            bounded_int(self.maximum_output_tokens, "maximum_output_tokens", minimum=8, maximum=2_048),
        )
        object.__setattr__(
            self,
            "maximum_output_bytes",
            bounded_int(self.maximum_output_bytes, "maximum_output_bytes", minimum=128, maximum=32_768),
        )
        object.__setattr__(
            self,
            "abstention_output_class",
            required_text(self.abstention_output_class, "abstention_output_class"),
        )
        if self.abstention_output_class != ABSTAIN_OUTPUT_CLASS:
            raise ResidualIntelligenceError("abstention representation must be ABSTAIN")
        object.__setattr__(
            self,
            "independent_validator_required",
            _require_bool(self.independent_validator_required, "independent_validator_required"),
        )
        if not self.independent_validator_required:
            raise ResidualIntelligenceError(REASON_VALIDATOR_REQUIRED)
        if not self.validator_identity.startswith("validator:"):
            raise ResidualIntelligenceError(REASON_VALIDATOR_REQUIRED)
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        allowed = tuple(RiskClass(item) for item in self.allowed_risk_classes)
        if not allowed:
            raise ResidualIntelligenceError("allowed_risk_classes must not be empty")
        if self.risk_ceiling not in allowed:
            raise ResidualIntelligenceError("risk_ceiling must be an allowed risk class")
        object.__setattr__(self, "allowed_risk_classes", allowed)
        object.__setattr__(
            self,
            "hardware_requirements",
            text_tuple(self.hardware_requirements, "hardware_requirements", allow_empty=False),
        )
        object.__setattr__(
            self,
            "runtime_requirements",
            text_tuple(self.runtime_requirements, "runtime_requirements", allow_empty=False),
        )
        if "provider_free" not in self.runtime_requirements:
            raise ResidualIntelligenceError("expert runtime must remain provider_free at spec time")
        object.__setattr__(
            self, "capabilities", text_tuple(self.capabilities, "capabilities", allow_empty=False)
        )
        if self.error_behavior != ERROR_INVALID_OUTPUT:
            raise ResidualIntelligenceError("expert error_behavior must be invalid_output")
        object.__setattr__(
            self,
            "emit_prose_by_default",
            _require_bool(self.emit_prose_by_default, "emit_prose_by_default"),
        )
        if self.emit_prose_by_default:
            raise ResidualIntelligenceError(REASON_PROSE_DEFAULT)
        object.__setattr__(self, "always_abstain", _require_bool(self.always_abstain, "always_abstain"))
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("expert specifications must remain candidate_only=true")
        admission_id = self.evaluation_corpus_admission_id
        if admission_id in (None, ""):
            object.__setattr__(self, "evaluation_corpus_admission_id", "")
        else:
            object.__setattr__(
                self,
                "evaluation_corpus_admission_id",
                required_text(admission_id, "evaluation_corpus_admission_id"),
            )

    @property
    def spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def family_spec(self) -> ResidualTaskFamilySpec:
        spec = family_spec_for(self.task_family)
        if spec.spec_id != self.family_spec_id:
            raise ResidualIntelligenceError("expert family spec identity mismatch")
        return spec

    def family_boundary(self) -> ResidualFamilyBoundary:
        boundary = self.family_spec().to_family_boundary()
        if not isinstance(boundary, ResidualFamilyBoundary):
            raise ResidualIntelligenceError("family boundary must be typed")
        if boundary.boundary_id != self.family_boundary_id:
            raise ResidualIntelligenceError("expert family boundary identity mismatch")
        return boundary

    def grammar(self) -> ExpertGrammar:
        grammar = grammar_for(self.task_family)
        if grammar.grammar_id != self.grammar_id:
            raise ResidualIntelligenceError("expert grammar identity mismatch")
        return grammar

    def allows_risk(self, risk: RiskClass | str) -> bool:
        try:
            parsed = RiskClass(risk)
        except ValueError:
            return False
        return parsed in self.allowed_risk_classes

    def reject_unsupported_risk(self, risk: RiskClass | str) -> RiskClass:
        parsed = RiskClass(risk)
        if parsed not in self.allowed_risk_classes:
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_FAMILY_RISK)
        return parsed

    def validate_task_input(self, task_input: ResidualTaskInput) -> None:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
        self.family_spec().validate_task_input(task_input)
        self.reject_unsupported_risk(task_input.risk_class)

    def bind_evaluation_admission(self, admission: TrainingCorpusAdmission) -> None:
        if not self.evaluation_corpus_admission_id:
            raise ResidualIntelligenceError("expert spec carries no evaluation dataset reference")
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
        if admission.admission_id != self.evaluation_corpus_admission_id:
            raise ResidualIntelligenceError("evaluation corpus admission identity mismatch")
        if admission.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "expert_id": self.expert_id,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "family_spec_id": self.family_spec_id,
            "family_boundary_id": self.family_boundary_id,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "grammar_id": self.grammar_id,
            "enumerations": {
                key: list(value) for key, value in sorted(self.enumerations.items())
            },
            "forms": list(self.forms),
            "maximum_input_tokens": self.maximum_input_tokens,
            "maximum_output_tokens": self.maximum_output_tokens,
            "maximum_output_bytes": self.maximum_output_bytes,
            "abstention_output_class": self.abstention_output_class,
            "validation_policy": self.validation_policy,
            "validator_identity": self.validator_identity,
            "independent_validator_required": True,
            "risk_ceiling": self.risk_ceiling.value,
            "allowed_risk_classes": [item.value for item in self.allowed_risk_classes],
            "privacy_class": self.privacy_class,
            "privacy_route_policy": self.privacy_route_policy,
            "hardware_requirements": list(self.hardware_requirements),
            "runtime_requirements": list(self.runtime_requirements),
            "capabilities": list(self.capabilities),
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "emit_prose_by_default": False,
            "always_abstain": self.always_abstain,
            "candidate_only": True,
            "authority_class": CANDIDATE_ONLY_AUTHORITY,
            "evaluation_corpus_admission_id": self.evaluation_corpus_admission_id,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualExpertSpec:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("expert spec must be an object")
        examples = sorted(str(key) for key in payload if key in _FORBIDDEN_EXAMPLE_FIELDS)
        if examples:
            raise UnknownFieldError(f"{REASON_EXAMPLES_FORBIDDEN}: {', '.join(examples)}")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"spec_id", "evaluation_corpus_admission_id"},
            noun="residual expert spec",
        )
        raw_enums = payload.get("enumerations") or {}
        if not isinstance(raw_enums, Mapping):
            raise ResidualIntelligenceError("enumerations must be an object")
        enumerations = {
            required_text(key, "enumeration name", max_bytes=256): tuple(value)
            for key, value in raw_enums.items()
        }
        result = cls(
            schema=str(payload.get("schema") or ""),
            expert_id=str(payload.get("expert_id") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=parse_expert_class(str(payload.get("expert_class") or "")),
            family_spec_id=str(payload.get("family_spec_id") or ""),
            family_boundary_id=str(payload.get("family_boundary_id") or ""),
            input_schema=str(payload.get("input_schema") or ""),
            output_schema=str(payload.get("output_schema") or ""),
            grammar_id=str(payload.get("grammar_id") or ""),
            enumerations=enumerations,
            forms=tuple(payload.get("forms") or ()),
            maximum_input_tokens=payload.get("maximum_input_tokens"),
            maximum_output_tokens=payload.get("maximum_output_tokens"),
            maximum_output_bytes=payload.get("maximum_output_bytes"),
            abstention_output_class=str(payload.get("abstention_output_class") or ""),
            validation_policy=str(payload.get("validation_policy") or ""),
            validator_identity=str(payload.get("validator_identity") or ""),
            independent_validator_required=payload.get("independent_validator_required"),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            allowed_risk_classes=tuple(
                RiskClass(str(item)) for item in (payload.get("allowed_risk_classes") or ())
            ),
            privacy_class=str(payload.get("privacy_class") or ""),
            privacy_route_policy=str(payload.get("privacy_route_policy") or ""),
            hardware_requirements=tuple(payload.get("hardware_requirements") or ()),
            runtime_requirements=tuple(payload.get("runtime_requirements") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            emit_prose_by_default=payload.get("emit_prose_by_default"),
            always_abstain=payload.get("always_abstain"),
            candidate_only=payload.get("candidate_only"),
            authority_class=str(payload.get("authority_class") or ""),
            evaluation_corpus_admission_id=str(payload.get("evaluation_corpus_admission_id") or ""),
        )
        claimed = str(payload.get("spec_id") or "")
        if claimed and claimed != result.spec_id:
            raise ResidualIntelligenceError("residual expert spec identity mismatch")
        return result

    @classmethod
    def from_family(
        cls,
        family_spec: ResidualTaskFamilySpec,
        expert_class: ExpertClass | str,
    ) -> ResidualExpertSpec:
        if not isinstance(family_spec, ResidualTaskFamilySpec):
            raise ResidualIntelligenceError("family_spec must be ResidualTaskFamilySpec")
        wanted = parse_expert_class(expert_class)
        if not family_spec.allows_expert_class(wanted.value):
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_EXPERT_CLASS)
        grammar = grammar_for(family_spec.task_family)
        boundary = family_spec.to_family_boundary()
        slug = family_spec.task_family.value.lower().replace("_", "-")
        return cls(
            expert_id=f"expert:{slug}:class-{wanted.value}:{FAMILY_SPEC_REGISTRY_VERSION}",
            task_family=family_spec.task_family,
            expert_class=wanted,
            family_spec_id=family_spec.spec_id,
            family_boundary_id=boundary.boundary_id,
            input_schema=family_spec.input_schema,
            output_schema=family_spec.output_schema,
            grammar_id=grammar.grammar_id,
            enumerations=enumerations_from_grammar(grammar),
            forms=EXPERT_CLASS_FORMS[wanted],
            maximum_input_tokens=family_spec.maximum_input_tokens,
            maximum_output_tokens=family_spec.maximum_output_tokens,
            maximum_output_bytes=family_spec.maximum_output_bytes,
            abstention_output_class=grammar.abstention_output_class,
            validation_policy=family_spec.validation_contract,
            validator_identity=family_spec.validator_identity,
            independent_validator_required=True,
            risk_ceiling=family_spec.risk_ceiling,
            allowed_risk_classes=family_spec.allowed_risk_classes,
            privacy_class=family_spec.privacy_class.value,
            privacy_route_policy=family_spec.privacy_route_policy,
            hardware_requirements=EXPERT_CLASS_HARDWARE[wanted],
            runtime_requirements=EXPERT_CLASS_RUNTIME[wanted],
            capabilities=family_spec.capabilities,
            error_behavior=family_spec.error_behavior,
            abstention_behavior=family_spec.abstention_behavior,
            emit_prose_by_default=False,
            always_abstain=family_spec.always_abstain,
            candidate_only=True,
            authority_class=CANDIDATE_ONLY_AUTHORITY,
            evaluation_corpus_admission_id=family_spec.evaluation_corpus_admission_id,
        )


def _enumerations_payload(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError("enumerations must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, items in value.items():
        name = required_text(key, "enumeration name", max_bytes=256)
        result[name] = text_tuple(items, f"enumerations.{name}", allow_empty=False, max_items=256)
    if "output_classes" not in result:
        raise ResidualIntelligenceError("enumerations must declare output_classes")
    if ABSTAIN_OUTPUT_CLASS not in result["output_classes"]:
        raise ResidualIntelligenceError("enumerations.output_classes must include ABSTAIN")
    return result


def _build_expert_specs() -> dict[tuple[ResidualTaskFamily, ExpertClass], ResidualExpertSpec]:
    registry: dict[tuple[ResidualTaskFamily, ExpertClass], ResidualExpertSpec] = {}
    for family in ResidualTaskFamily:
        family_spec = family_spec_for(family)
        for letter in family_spec.eligible_expert_classes:
            expert_class = parse_expert_class(letter)
            registry[(family, expert_class)] = ResidualExpertSpec.from_family(
                family_spec, expert_class
            )
    return registry


EXPERT_SPECS: Final[Mapping[tuple[ResidualTaskFamily, ExpertClass], ResidualExpertSpec]] = (
    _build_expert_specs()
)


def expert_spec_for(
    task_family: ResidualTaskFamily | str,
    expert_class: ExpertClass | str | None = None,
) -> ResidualExpertSpec:
    family = ResidualTaskFamily(task_family)
    family_spec = family_spec_for(family)
    wanted = (
        parse_expert_class(family_spec.smallest_expert_class)
        if expert_class is None
        else parse_expert_class(expert_class)
    )
    try:
        return EXPERT_SPECS[(family, wanted)]
    except KeyError as exc:
        raise ResidualIntelligenceError(REASON_UNSUPPORTED_EXPERT_CLASS) from exc


def expert_specs_for_family(
    task_family: ResidualTaskFamily | str,
) -> tuple[ResidualExpertSpec, ...]:
    family = ResidualTaskFamily(task_family)
    family_spec = family_spec_for(family)
    return tuple(
        EXPERT_SPECS[(family, parse_expert_class(letter))]
        for letter in family_spec.eligible_expert_classes
    )


def all_expert_specs() -> tuple[ResidualExpertSpec, ...]:
    specs: list[ResidualExpertSpec] = []
    for family in ResidualTaskFamily:
        specs.extend(expert_specs_for_family(family))
    return tuple(specs)


def admit_expert_class(
    task_family: ResidualTaskFamily | str,
    requested: ExpertClass | str,
    *,
    risk: RiskClass | str,
    quality_delta_ppm: int = 0,
    routing_changing: bool = False,
    evidence_current: bool = False,
    compared_class: ExpertClass | str | None = None,
    admission: TrainingCorpusAdmission | None = None,
    policy: ModelSizePolicy | None = None,
) -> ResidualExpertSpec:
    family_spec = family_spec_for(task_family)
    size_policy = policy or DEFAULT_MODEL_SIZE_POLICY
    admitted = size_policy.admit_requested_class(
        family_spec,
        requested,
        risk=risk,
        quality_delta_ppm=quality_delta_ppm,
        routing_changing=routing_changing,
        evidence_current=evidence_current,
        compared_class=compared_class,
        admission=admission,
    )
    return expert_spec_for(family_spec.task_family, admitted)


def expert_spec_registry_payload() -> dict[str, Any]:
    specs = all_expert_specs()
    return {
        "schema": EXPERT_SPEC_REGISTRY_SCHEMA,
        "registry_version": EXPERT_SPEC_REGISTRY_VERSION,
        "family_registry_version": FAMILY_SPEC_REGISTRY_VERSION,
        "expert_count": len(specs),
        "form_order": [item.value for item in SMALLEST_FORM_ORDER],
        "experts": [item.to_dict() for item in specs],
    }


__all__ = (
    "DEFAULT_MODEL_SIZE_POLICY",
    "EXPERT_CLASS_FORMS",
    "EXPERT_SPECS",
    "EXPERT_SPEC_REGISTRY_SCHEMA",
    "EXPERT_SPEC_REGISTRY_VERSION",
    "EXPERT_SPEC_SCHEMA",
    "ExpertClass",
    "MIN_ROUTING_CHANGING_DELTA_PPM",
    "MODEL_SIZE_POLICY_SCHEMA",
    "ModelSizePolicy",
    "REASON_LARGER_FORM",
    "REASON_UNSUPPORTED_EXPERT_CLASS",
    "ResidualExpertSpec",
    "SMALLEST_FORM_ORDER",
    "admit_expert_class",
    "all_expert_specs",
    "enumerations_from_grammar",
    "expert_class_rank",
    "expert_spec_for",
    "expert_spec_registry_payload",
    "expert_specs_for_family",
    "parse_expert_class",
)
