"""Task-family expert specifications for residual intelligence.

Each expert binds one closed family boundary, class A-E, schemas, grammar,
token/output limits, risk ceiling, privacy route, capabilities, validator,
error codes, and abstention behavior. Larger forms require a routing-changing
held-out quality delta. Specifications carry no examples.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    bounded_int,
    canonical_id,
    optional_text,
    required_text,
    strict_fields,
    text_tuple,
)
from .residual_ir import ResidualTaskInput, ResidualTaskOutput
from .rights import TrainingCorpusAdmission
from .structured_decoding import grammar_for
from .task_families import (
    DEFAULT_FAMILY_SPECS,
    DEFAULT_SIZE_FOR_CLASS,
    EXPERT_CLASS_FORMS,
    MODEL_SIZE_ORDER,
    SMALLEST_FORM_ORDER,
    ClosedValueSchema,
    ExpertClass,
    ModelSizePolicy,
    ResidualTaskFamilySpec,
    assert_family_risk_allowed,
    expert_class_rank,
    family_spec_for,
    model_size_rank,
    risk_rank,
)

EXPERT_SPEC_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-spec@1"
MAX_QUALITY_DELTA_PPM: Final = 1_000_000
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def resolve_dataset_reference(
    reference: str,
    admission: TrainingCorpusAdmission | None,
    *,
    noun: str,
) -> None:
    """Fail closed unless a non-empty dataset reference is an admitted corpus."""

    if not reference:
        return
    if admission is None or not admission.can_train:
        raise ResidualIntelligenceError(
            f"{noun} must resolve to an admitted TrainingCorpusAdmission"
        )
    allowed = {
        admission.admission_id,
        admission.corpus_root,
        admission.split_root,
        *admission.holdout_roots,
    }
    if reference not in allowed:
        raise ResidualIntelligenceError(
            f"{noun} does not resolve to the admitted TrainingCorpusAdmission"
        )


def assert_larger_form_permitted(
    *,
    family_spec: ResidualTaskFamilySpec,
    requested_class: ExpertClass,
    requested_size: ModelSizePolicy,
    routing_changing_quality_delta_ppm: int,
    held_out_evidence_current: bool,
) -> None:
    """A larger class or size is eligible only with a current routing-changing delta."""

    preferred = family_spec.preferred_expert_class
    preferred_size = DEFAULT_SIZE_FOR_CLASS[preferred]
    family_spec.assert_expert_class_allowed(requested_class)
    family_spec.assert_model_size_allowed(requested_size)
    larger_class = expert_class_rank(requested_class) > expert_class_rank(preferred)
    larger_size = model_size_rank(requested_size) > model_size_rank(preferred_size)
    if not larger_class and not larger_size:
        return
    delta = bounded_int(
        routing_changing_quality_delta_ppm,
        "routing_changing_quality_delta_ppm",
        minimum=0,
        maximum=MAX_QUALITY_DELTA_PPM,
    )
    if type(held_out_evidence_current) is not bool:
        raise ResidualIntelligenceError("held_out_evidence_current must be boolean")
    if not held_out_evidence_current or delta <= 0:
        raise ResidualIntelligenceError(
            "larger form needs a routing-changing quality delta from current held-out evidence"
        )


@dataclass(frozen=True)
class ResidualExpertSpec:
    """One family-bounded expert contract; never an authority or completion."""

    expert_id: str
    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    model_size_policy: ModelSizePolicy
    family_spec_id: str
    input_schema: ClosedValueSchema
    output_schema: ClosedValueSchema
    grammar_id: str
    allowed_outputs: tuple[str, ...]
    max_input_tokens: int
    max_output_tokens: int
    max_output_bytes: int
    risk_ceiling: RiskClass
    privacy_class: PrivacyClass
    privacy_route_policy: str
    hardware_class: str
    runtime_requirements: tuple[str, ...]
    capabilities: tuple[str, ...]
    validation_policy: str
    validator_required: bool
    error_behavior: str
    abstention_behavior: str
    error_codes: tuple[str, ...]
    abstention_codes: tuple[str, ...]
    candidate_only: bool = True
    emits_prose_by_default: bool = False
    evaluation_dataset_reference: str = ""
    training_dataset_reference: str = ""
    schema: str = EXPERT_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "expert_spec_id",
            "expert_id",
            "task_family",
            "expert_class",
            "form",
            "model_size_policy",
            "family_spec_id",
            "input_schema",
            "output_schema",
            "grammar_id",
            "allowed_outputs",
            "max_input_tokens",
            "max_output_tokens",
            "max_output_bytes",
            "risk_ceiling",
            "privacy_class",
            "privacy_route_policy",
            "hardware_class",
            "runtime_requirements",
            "capabilities",
            "validation_policy",
            "validator_required",
            "error_behavior",
            "abstention_behavior",
            "error_codes",
            "abstention_codes",
            "candidate_only",
            "emits_prose_by_default",
            "evaluation_dataset_reference",
            "training_dataset_reference",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != EXPERT_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual expert spec schema")
        object.__setattr__(self, "expert_id", required_text(self.expert_id, "expert_id"))
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", ExpertClass(self.expert_class))
        object.__setattr__(self, "model_size_policy", ModelSizePolicy(self.model_size_policy))
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        family = family_spec_for(self.task_family)
        object.__setattr__(
            self, "family_spec_id", required_text(self.family_spec_id, "family_spec_id")
        )
        if self.family_spec_id != family.family_spec_id:
            raise ResidualIntelligenceError("expert family_spec_id does not match the family registry")
        family.assert_expert_class_allowed(self.expert_class)
        family.assert_model_size_allowed(self.model_size_policy)
        family.assert_risk_allowed(self.risk_ceiling)
        if risk_rank(self.risk_ceiling) > risk_rank(family.risk_ceiling):
            raise ResidualIntelligenceError("expert risk ceiling exceeds the family risk ceiling")
        if not isinstance(self.input_schema, ClosedValueSchema):
            raise ResidualIntelligenceError("input_schema must be ClosedValueSchema")
        if not isinstance(self.output_schema, ClosedValueSchema):
            raise ResidualIntelligenceError("output_schema must be ClosedValueSchema")
        if self.input_schema != family.input_schema or self.output_schema != family.output_schema:
            raise ResidualIntelligenceError("expert schemas must equal the family closed schemas")
        grammar = grammar_for(self.task_family)
        object.__setattr__(self, "grammar_id", required_text(self.grammar_id, "grammar_id"))
        if self.grammar_id != family.grammar_id or self.grammar_id != grammar.grammar_id:
            raise ResidualIntelligenceError("expert grammar_id must equal the family grammar")
        object.__setattr__(
            self,
            "allowed_outputs",
            text_tuple(self.allowed_outputs, "allowed_outputs", allow_empty=False),
        )
        if tuple(self.allowed_outputs) != tuple(family.allowed_outputs):
            raise ResidualIntelligenceError("expert allowed_outputs must equal the family grammar")
        for field in (
            "privacy_route_policy",
            "hardware_class",
            "validation_policy",
            "error_behavior",
            "abstention_behavior",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.privacy_route_policy != family.privacy_route_policy:
            raise ResidualIntelligenceError("expert privacy route must equal the family policy")
        if self.validation_policy != family.validation_contract:
            raise ResidualIntelligenceError("expert validation_policy must equal the family validator")
        object.__setattr__(
            self, "validator_required", _require_bool(self.validator_required, "validator_required")
        )
        if not self.validator_required:
            raise ResidualIntelligenceError("every expert requires an independent validator")
        object.__setattr__(
            self, "candidate_only", _require_bool(self.candidate_only, "candidate_only")
        )
        if self.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        object.__setattr__(
            self,
            "emits_prose_by_default",
            _require_bool(self.emits_prose_by_default, "emits_prose_by_default"),
        )
        if self.emits_prose_by_default:
            raise ResidualIntelligenceError("typed structured output is the default; prose is not")
        object.__setattr__(
            self,
            "max_input_tokens",
            bounded_int(self.max_input_tokens, "max_input_tokens", minimum=1, maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            bounded_int(self.max_output_tokens, "max_output_tokens", minimum=1, maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "max_output_bytes",
            bounded_int(self.max_output_bytes, "max_output_bytes", minimum=128, maximum=32_768),
        )
        if self.max_output_bytes != family.max_output_bytes:
            raise ResidualIntelligenceError("expert max_output_bytes must match the family grammar")
        if self.max_input_tokens > family.max_input_tokens:
            raise ResidualIntelligenceError("expert input token limit exceeds the family bound")
        if self.max_output_tokens > family.max_output_tokens:
            raise ResidualIntelligenceError("expert output token limit exceeds the family bound")
        object.__setattr__(
            self,
            "runtime_requirements",
            text_tuple(self.runtime_requirements, "runtime_requirements", allow_empty=False),
        )
        object.__setattr__(
            self, "capabilities", text_tuple(self.capabilities, "capabilities", allow_empty=False)
        )
        object.__setattr__(
            self, "error_codes", text_tuple(self.error_codes, "error_codes", allow_empty=False)
        )
        object.__setattr__(
            self,
            "abstention_codes",
            text_tuple(self.abstention_codes, "abstention_codes", allow_empty=False),
        )
        if tuple(self.error_codes) != tuple(family.error_codes):
            raise ResidualIntelligenceError("expert error codes must equal the family contract")
        if tuple(self.abstention_codes) != tuple(family.abstention_codes):
            raise ResidualIntelligenceError("expert abstention codes must equal the family contract")
        object.__setattr__(
            self,
            "evaluation_dataset_reference",
            optional_text(self.evaluation_dataset_reference, "evaluation_dataset_reference"),
        )
        object.__setattr__(
            self,
            "training_dataset_reference",
            optional_text(self.training_dataset_reference, "training_dataset_reference"),
        )

    @property
    def form(self) -> str:
        return EXPERT_CLASS_FORMS[self.expert_class]

    @property
    def expert_spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def family_spec(self) -> ResidualTaskFamilySpec:
        return family_spec_for(self.task_family)

    @property
    def proposal_tier(self) -> bool:
        return self.risk_ceiling in PROPOSAL_RISKS

    def resolve_dataset_references(self, admission: TrainingCorpusAdmission | None) -> None:
        resolve_dataset_reference(
            self.evaluation_dataset_reference,
            admission,
            noun="evaluation_dataset_reference",
        )
        resolve_dataset_reference(
            self.training_dataset_reference,
            admission,
            noun="training_dataset_reference",
        )

    def validate_input(self, task_input: ResidualTaskInput) -> None:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
        if task_input.task_family is not self.task_family:
            raise ResidualIntelligenceError("task input family differs from expert family boundary")
        self.family_spec.assert_risk_allowed(task_input.risk_class)
        if task_input.validation_policy != self.validation_policy:
            raise ResidualIntelligenceError("task input validation_policy must equal the expert validator")
        if task_input.token_budget > self.max_input_tokens + self.max_output_tokens:
            raise ResidualIntelligenceError("task input token_budget exceeds expert limits")
        extra_outputs = sorted(set(task_input.allowed_outputs) - set(self.allowed_outputs))
        if extra_outputs:
            raise ResidualIntelligenceError(
                "task input allowed_outputs are outside the closed expert schema: "
                + ", ".join(extra_outputs)
            )
        compact = task_input.compact_features
        unknown = sorted(set(compact) - set(self.family_spec.compact_feature_keys))
        if unknown:
            raise ResidualIntelligenceError(
                "compact_features contains unknown fields: " + ", ".join(unknown)
            )
        self.input_schema.validate_mapping(
            {key: value for key, value in task_input.to_dict(include_id=False).items() if key != "schema"},
            noun="residual task input",
        )

    def validate_output(self, task_output: ResidualTaskOutput) -> None:
        if not isinstance(task_output, ResidualTaskOutput):
            raise ResidualIntelligenceError("task_output must be ResidualTaskOutput")
        if task_output.output_class not in self.allowed_outputs:
            raise ResidualIntelligenceError("residual output class is outside the closed expert schema")
        if task_output.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if task_output.output_class == grammar_for(self.task_family).abstention_output_class:
            if not task_output.abstained:
                raise ResidualIntelligenceError("abstention class requires abstained=true")
            return
        if self.proposal_tier and "VALIDATION_REQUIRED" not in task_output.reason_codes:
            if not task_output.abstained:
                raise ResidualIntelligenceError(
                    "R4/R5 learned outputs must abstain or remain explicitly validation-required"
                )
        self.output_schema.validate_mapping(task_output.structured_payload, noun="structured_payload")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "expert_id": self.expert_id,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "form": self.form,
            "model_size_policy": self.model_size_policy.value,
            "family_spec_id": self.family_spec_id,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "grammar_id": self.grammar_id,
            "allowed_outputs": list(self.allowed_outputs),
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_output_bytes": self.max_output_bytes,
            "risk_ceiling": self.risk_ceiling.value,
            "privacy_class": self.privacy_class.value,
            "privacy_route_policy": self.privacy_route_policy,
            "hardware_class": self.hardware_class,
            "runtime_requirements": list(self.runtime_requirements),
            "capabilities": list(self.capabilities),
            "validation_policy": self.validation_policy,
            "validator_required": True,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "error_codes": list(self.error_codes),
            "abstention_codes": list(self.abstention_codes),
            "candidate_only": True,
            "emits_prose_by_default": False,
            "evaluation_dataset_reference": self.evaluation_dataset_reference,
            "training_dataset_reference": self.training_dataset_reference,
        }
        if include_id:
            result["expert_spec_id"] = self.expert_spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualExpertSpec:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"expert_spec_id", "form"},
            noun="residual expert spec",
        )
        input_schema = payload.get("input_schema")
        output_schema = payload.get("output_schema")
        result = cls(
            schema=str(payload.get("schema") or ""),
            expert_id=str(payload.get("expert_id") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=ExpertClass(str(payload.get("expert_class") or "")),
            model_size_policy=ModelSizePolicy(str(payload.get("model_size_policy") or "")),
            family_spec_id=str(payload.get("family_spec_id") or ""),
            input_schema=(
                input_schema
                if isinstance(input_schema, ClosedValueSchema)
                else ClosedValueSchema.from_dict(input_schema or {})
            ),
            output_schema=(
                output_schema
                if isinstance(output_schema, ClosedValueSchema)
                else ClosedValueSchema.from_dict(output_schema or {})
            ),
            grammar_id=str(payload.get("grammar_id") or ""),
            allowed_outputs=tuple(payload.get("allowed_outputs") or ()),
            max_input_tokens=payload.get("max_input_tokens"),
            max_output_tokens=payload.get("max_output_tokens"),
            max_output_bytes=payload.get("max_output_bytes"),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            privacy_route_policy=str(payload.get("privacy_route_policy") or ""),
            hardware_class=str(payload.get("hardware_class") or ""),
            runtime_requirements=tuple(payload.get("runtime_requirements") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            validation_policy=str(payload.get("validation_policy") or ""),
            validator_required=payload.get("validator_required"),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            error_codes=tuple(payload.get("error_codes") or ()),
            abstention_codes=tuple(payload.get("abstention_codes") or ()),
            candidate_only=payload.get("candidate_only"),
            emits_prose_by_default=payload.get("emits_prose_by_default"),
            evaluation_dataset_reference=str(payload.get("evaluation_dataset_reference") or ""),
            training_dataset_reference=str(payload.get("training_dataset_reference") or ""),
        )
        claimed_form = str(payload.get("form") or "")
        if claimed_form and claimed_form != result.form:
            raise ResidualIntelligenceError("expert form does not match expert class")
        claimed = str(payload.get("expert_spec_id") or "")
        if claimed and claimed != result.expert_spec_id:
            raise ResidualIntelligenceError("residual expert spec identity mismatch")
        return result


def _expert_from_family(family: ResidualTaskFamilySpec) -> ResidualExpertSpec:
    return ResidualExpertSpec(
        expert_id=f"expert:{family.task_family.value}:{family.preferred_expert_class.value}",
        task_family=family.task_family,
        expert_class=family.preferred_expert_class,
        model_size_policy=DEFAULT_SIZE_FOR_CLASS[family.preferred_expert_class],
        family_spec_id=family.family_spec_id,
        input_schema=family.input_schema,
        output_schema=family.output_schema,
        grammar_id=family.grammar_id,
        allowed_outputs=family.allowed_outputs,
        max_input_tokens=family.max_input_tokens,
        max_output_tokens=family.max_output_tokens,
        max_output_bytes=family.max_output_bytes,
        risk_ceiling=family.risk_ceiling,
        privacy_class=family.privacy_class,
        privacy_route_policy=family.privacy_route_policy,
        hardware_class=family.hardware_class,
        runtime_requirements=family.runtime_requirements,
        capabilities=family.capabilities,
        validation_policy=family.validation_contract,
        validator_required=True,
        error_behavior=family.error_behavior,
        abstention_behavior=family.abstention_behavior,
        error_codes=family.error_codes,
        abstention_codes=family.abstention_codes,
        candidate_only=True,
        emits_prose_by_default=False,
    )


DEFAULT_EXPERT_SPECS: Final[Mapping[ResidualTaskFamily, ResidualExpertSpec]] = {
    family: _expert_from_family(spec) for family, spec in DEFAULT_FAMILY_SPECS.items()
}


def expert_spec_for(
    task_family: ResidualTaskFamily | str,
    expert_class: ExpertClass | str | None = None,
    *,
    model_size_policy: ModelSizePolicy | str | None = None,
    routing_changing_quality_delta_ppm: int = 0,
    held_out_evidence_current: bool = False,
    evaluation_admission: TrainingCorpusAdmission | None = None,
    evaluation_dataset_reference: str = "",
    training_dataset_reference: str = "",
) -> ResidualExpertSpec:
    """Return the family expert, or a larger form when a quality delta is proven."""

    family = family_spec_for(task_family)
    requested_class = (
        family.preferred_expert_class if expert_class is None else ExpertClass(expert_class)
    )
    requested_size = (
        DEFAULT_SIZE_FOR_CLASS[requested_class]
        if model_size_policy is None
        else ModelSizePolicy(model_size_policy)
    )
    assert_larger_form_permitted(
        family_spec=family,
        requested_class=requested_class,
        requested_size=requested_size,
        routing_changing_quality_delta_ppm=routing_changing_quality_delta_ppm,
        held_out_evidence_current=held_out_evidence_current,
    )
    spec = ResidualExpertSpec(
        expert_id=f"expert:{family.task_family.value}:{requested_class.value}",
        task_family=family.task_family,
        expert_class=requested_class,
        model_size_policy=requested_size,
        family_spec_id=family.family_spec_id,
        input_schema=family.input_schema,
        output_schema=family.output_schema,
        grammar_id=family.grammar_id,
        allowed_outputs=family.allowed_outputs,
        max_input_tokens=family.max_input_tokens,
        max_output_tokens=family.max_output_tokens,
        max_output_bytes=family.max_output_bytes,
        risk_ceiling=family.risk_ceiling,
        privacy_class=family.privacy_class,
        privacy_route_policy=family.privacy_route_policy,
        hardware_class=family.hardware_class,
        runtime_requirements=family.runtime_requirements,
        capabilities=family.capabilities,
        validation_policy=family.validation_contract,
        validator_required=True,
        error_behavior=family.error_behavior,
        abstention_behavior=family.abstention_behavior,
        error_codes=family.error_codes,
        abstention_codes=family.abstention_codes,
        candidate_only=True,
        emits_prose_by_default=False,
        evaluation_dataset_reference=evaluation_dataset_reference,
        training_dataset_reference=training_dataset_reference,
    )
    spec.resolve_dataset_references(evaluation_admission)
    return spec


if set(DEFAULT_EXPERT_SPECS) != set(ResidualTaskFamily):
    raise ResidualIntelligenceError("expert spec registry is not a bijection over the taxonomy")
if {item.expert_class for item in DEFAULT_EXPERT_SPECS.values()} != set(SMALLEST_FORM_ORDER):
    raise ResidualIntelligenceError("default experts must cover classes A through E")


__all__ = (
    "DEFAULT_EXPERT_SPECS",
    "EXPERT_CLASS_FORMS",
    "EXPERT_SPEC_SCHEMA",
    "MODEL_SIZE_ORDER",
    "SMALLEST_FORM_ORDER",
    "ClosedValueSchema",
    "ExpertClass",
    "ModelSizePolicy",
    "ResidualExpertSpec",
    "ResidualTaskFamilySpec",
    "assert_family_risk_allowed",
    "assert_larger_form_permitted",
    "expert_spec_for",
    "family_spec_for",
    "resolve_dataset_reference",
)
