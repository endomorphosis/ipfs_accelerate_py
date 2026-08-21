"""Family-bounded residual expert specifications.

Every expert shares its family's exact semantic boundary, closed schemas,
grammar, risk ceiling, required validator, and abstention contract.  A larger
form is eligible only when a routing-changing quality delta is supplied for
that family.  Specifications carry no examples and remain candidate-only.
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
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)
from .inventory import ResidualFamilyBoundary
from .residual_ir import MAX_TOKEN_BUDGET, ResidualTaskInput
from .structured_decoding import MAX_STRUCTURED_OUTPUT_BYTES
from .task_families import (
    AUTHORITY_CANDIDATE_ONLY,
    DEFAULT_TASK_FAMILY_SPECS,
    ClosedSchema,
    ExpertClass,
    ResidualTaskFamilySpec,
    family_spec_for,
    privacy_rank,
    risk_rank,
)

EXPERT_SPEC_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-spec@1"
EXPERT_SPEC_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-expert-spec-registry@1"
)


class ModelSizePolicy(str, Enum):
    """Smallest-to-largest residual expert form.  Declaration order is binding."""

    EXACT_LOOKUP = "exact_lookup"
    DECLARATIVE_RULE = "declarative_rule"
    LINEAR_LOGISTIC = "linear_logistic"
    SMALL_RANKER_ENCODER = "small_ranker_encoder"
    CONSTRAINED_STRUCTURED_DECODER = "constrained_structured_decoder"
    PARAMETER_EFFICIENT_ADAPTER = "parameter_efficient_adapter"
    QUANTIZED_LOCAL_GENERAL = "quantized_local_general"
    REMOTE_STANDARD = "remote_standard"
    REMOTE_STRONG = "remote_strong"
    HUMAN_REVIEW = "human_review"


SMALLEST_FORM_ORDER: Final[tuple[ModelSizePolicy, ...]] = tuple(ModelSizePolicy)
LEARNED_FORMS: Final[frozenset[ModelSizePolicy]] = frozenset(
    {
        ModelSizePolicy.LINEAR_LOGISTIC,
        ModelSizePolicy.SMALL_RANKER_ENCODER,
        ModelSizePolicy.CONSTRAINED_STRUCTURED_DECODER,
        ModelSizePolicy.PARAMETER_EFFICIENT_ADAPTER,
        ModelSizePolicy.QUANTIZED_LOCAL_GENERAL,
        ModelSizePolicy.REMOTE_STANDARD,
        ModelSizePolicy.REMOTE_STRONG,
    }
)
REMOTE_FORMS: Final[frozenset[ModelSizePolicy]] = frozenset(
    {ModelSizePolicy.REMOTE_STANDARD, ModelSizePolicy.REMOTE_STRONG}
)
SAFE_FALLBACK_FORMS: Final[frozenset[ModelSizePolicy]] = frozenset(
    {ModelSizePolicy.HUMAN_REVIEW}
)


def _require_bool(value: Any, name: str, *, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    if expected is not None and value is not expected:
        raise ResidualIntelligenceError(f"{name} must be {expected}")
    return value


def form_rank(value: ModelSizePolicy | str) -> int:
    return SMALLEST_FORM_ORDER.index(ModelSizePolicy(value))


def form_requires_quality_delta(
    smallest_form: ModelSizePolicy | str,
    requested_form: ModelSizePolicy | str,
) -> bool:
    smallest = ModelSizePolicy(smallest_form)
    requested = ModelSizePolicy(requested_form)
    if requested in SAFE_FALLBACK_FORMS:
        return False
    return form_rank(requested) > form_rank(smallest)


def allowed_forms_for(
    family_spec: ResidualTaskFamilySpec,
    *,
    smallest_form: ModelSizePolicy,
) -> tuple[ModelSizePolicy, ...]:
    if family_spec.expert_class is ExpertClass.E:
        return (ModelSizePolicy.HUMAN_REVIEW,)
    start = form_rank(smallest_form)
    forms = []
    for item in SMALLEST_FORM_ORDER:
        if form_rank(item) < start:
            continue
        if item in REMOTE_FORMS and not family_spec.remote_route_permitted:
            continue
        forms.append(item)
    return tuple(forms)


def _require_form(value: Any, name: str) -> ModelSizePolicy:
    if isinstance(value, ModelSizePolicy):
        return value
    token = required_text(value, name, max_bytes=64)
    try:
        return ModelSizePolicy(token)
    except ValueError as exc:
        raise ResidualIntelligenceError(f"{name} is not a closed model-size policy") from exc


@dataclass(frozen=True)
class ResidualExpertSpec:
    """One candidate expert bound to exactly one residual family specification."""

    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    family_boundary_id: str
    form: ModelSizePolicy
    smallest_form: ModelSizePolicy
    allowed_forms: tuple[ModelSizePolicy, ...]
    input_schema: ClosedSchema
    output_schema: ClosedSchema
    grammar_id: str
    token_budget: int
    maximum_output_bytes: int
    maximum_output_tokens: int
    risk_ceiling: RiskClass
    privacy_class: PrivacyClass
    capabilities: tuple[str, ...]
    validation_contract: str
    error_behavior: str
    abstention_behavior: str
    routing_changing_quality_delta: bool = False
    remote_route_permitted: bool = False
    always_abstain: bool = False
    validator_required: bool = True
    prose_default: bool = False
    candidate_only: bool = True
    authority_class: str = AUTHORITY_CANDIDATE_ONLY
    schema: str = EXPERT_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "spec_id",
            "task_family",
            "expert_class",
            "family_boundary_id",
            "form",
            "smallest_form",
            "allowed_forms",
            "input_schema",
            "output_schema",
            "grammar_id",
            "token_budget",
            "maximum_output_bytes",
            "maximum_output_tokens",
            "risk_ceiling",
            "privacy_class",
            "capabilities",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "routing_changing_quality_delta",
            "remote_route_permitted",
            "always_abstain",
            "validator_required",
            "prose_default",
            "candidate_only",
            "authority_class",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != EXPERT_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual expert spec schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", ExpertClass(self.expert_class))
        family = family_spec_for(self.task_family)
        object.__setattr__(
            self,
            "family_boundary_id",
            required_text(self.family_boundary_id, "family_boundary_id"),
        )
        if self.family_boundary_id != family.family_boundary_id:
            raise ResidualIntelligenceError(
                "expert family boundary identity mismatch; prompt similarity is not a boundary"
            )
        if self.expert_class is not family.expert_class:
            raise ResidualIntelligenceError("expert class must equal the family expert class")
        object.__setattr__(self, "form", _require_form(self.form, "form"))
        object.__setattr__(
            self, "smallest_form", _require_form(self.smallest_form, "smallest_form")
        )
        if isinstance(self.allowed_forms, (str, bytes, bytearray)) or not isinstance(
            self.allowed_forms, Sequence
        ):
            raise ResidualIntelligenceError("allowed_forms must be a sequence")
        forms = tuple(ModelSizePolicy(item) for item in self.allowed_forms)
        if not forms:
            raise ResidualIntelligenceError("allowed_forms must not be empty")
        if len(set(forms)) != len(forms):
            raise ResidualIntelligenceError("allowed_forms contains duplicate values")
        if self.form not in forms or self.smallest_form not in forms:
            raise ResidualIntelligenceError("form is outside the expert size policy")
        ranked = [form_rank(item) for item in forms if item not in SAFE_FALLBACK_FORMS]
        if ranked and ranked != sorted(ranked):
            raise ResidualIntelligenceError("allowed_forms must follow smallest-form-order")
        object.__setattr__(self, "allowed_forms", forms)
        if not isinstance(self.input_schema, ClosedSchema) or not isinstance(
            self.output_schema, ClosedSchema
        ):
            raise ResidualIntelligenceError("expert input and output schemas must be ClosedSchema")
        if self.input_schema.schema_id != family.input_schema.schema_id:
            raise ResidualIntelligenceError("expert input schema must equal the family closed schema")
        if self.output_schema.schema_id != family.output_schema.schema_id:
            raise ResidualIntelligenceError("expert output schema must equal the family closed schema")
        object.__setattr__(self, "grammar_id", required_text(self.grammar_id, "grammar_id"))
        if self.grammar_id != family.grammar_id:
            raise ResidualIntelligenceError("expert grammar_id must equal the family grammar")
        object.__setattr__(
            self,
            "token_budget",
            bounded_int(self.token_budget, "token_budget", minimum=1, maximum=MAX_TOKEN_BUDGET),
        )
        if self.token_budget > family.token_budget:
            raise ResidualIntelligenceError("expert token budget exceeds the family limit")
        object.__setattr__(
            self,
            "maximum_output_bytes",
            bounded_int(
                self.maximum_output_bytes,
                "maximum_output_bytes",
                minimum=128,
                maximum=MAX_STRUCTURED_OUTPUT_BYTES,
            ),
        )
        if self.maximum_output_bytes > family.maximum_output_bytes:
            raise ResidualIntelligenceError("expert output bound exceeds the family grammar")
        object.__setattr__(
            self,
            "maximum_output_tokens",
            bounded_int(
                self.maximum_output_tokens,
                "maximum_output_tokens",
                minimum=1,
                maximum=MAX_TOKEN_BUDGET,
            ),
        )
        if self.maximum_output_tokens > family.maximum_output_tokens:
            raise ResidualIntelligenceError("expert output-token bound exceeds the family limit")
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        if risk_rank(self.risk_ceiling) > risk_rank(family.risk_ceiling):
            raise ResidualIntelligenceError("expert risk ceiling exceeds the family risk ceiling")
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        if privacy_rank(self.privacy_class) < privacy_rank(family.privacy_class):
            raise ResidualIntelligenceError("expert privacy class cannot weaken the family privacy")
        object.__setattr__(
            self,
            "capabilities",
            text_tuple(self.capabilities, "capabilities", allow_empty=False, max_items=32),
        )
        for field in (
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "authority_class",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.validation_contract != family.validation_contract:
            raise ResidualIntelligenceError("expert validator must equal the family validator")
        if self.error_behavior != family.error_behavior:
            raise ResidualIntelligenceError("expert error behavior must equal the family contract")
        if self.abstention_behavior != family.abstention_behavior:
            raise ResidualIntelligenceError("expert abstention behavior must equal the family contract")
        object.__setattr__(
            self,
            "routing_changing_quality_delta",
            _require_bool(
                self.routing_changing_quality_delta, "routing_changing_quality_delta"
            ),
        )
        object.__setattr__(
            self,
            "remote_route_permitted",
            _require_bool(self.remote_route_permitted, "remote_route_permitted"),
        )
        if self.remote_route_permitted and not family.remote_route_permitted:
            raise ResidualIntelligenceError("expert cannot enable a remote route the family forbids")
        if self.form in REMOTE_FORMS and not self.remote_route_permitted:
            raise ResidualIntelligenceError("remote form is outside the family privacy route policy")
        object.__setattr__(
            self, "always_abstain", _require_bool(self.always_abstain, "always_abstain")
        )
        if self.always_abstain is not family.always_abstain:
            raise ResidualIntelligenceError("expert abstain flag must equal the family contract")
        object.__setattr__(
            self,
            "validator_required",
            _require_bool(self.validator_required, "validator_required", expected=True),
        )
        object.__setattr__(
            self,
            "prose_default",
            _require_bool(self.prose_default, "prose_default", expected=False),
        )
        object.__setattr__(
            self,
            "candidate_only",
            _require_bool(self.candidate_only, "candidate_only", expected=True),
        )
        if self.authority_class.casefold() != AUTHORITY_CANDIDATE_ONLY:
            raise ResidualIntelligenceError(
                "residual expert authority_class must be candidate_only"
            )
        if family.expert_class is ExpertClass.E:
            if self.form is not ModelSizePolicy.HUMAN_REVIEW or self.smallest_form is not (
                ModelSizePolicy.HUMAN_REVIEW
            ):
                raise ResidualIntelligenceError("class E families cannot route a learned form")
            if not self.always_abstain:
                raise ResidualIntelligenceError("class E experts must always abstain")
        if form_requires_quality_delta(self.smallest_form, self.form) and not (
            self.routing_changing_quality_delta
        ):
            raise ResidualIntelligenceError(
                "larger form needs a routing-changing quality delta"
            )

    @property
    def spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def family_spec(self) -> ResidualTaskFamilySpec:
        return family_spec_for(self.task_family)

    def boundary(self) -> ResidualFamilyBoundary:
        return self.family_spec.boundary()

    def admit_risk(self, risk: RiskClass | str) -> RiskClass:
        ranked = self.family_spec.admit_risk(risk)
        if risk_rank(ranked) > risk_rank(self.risk_ceiling):
            raise ResidualIntelligenceError(
                f"risk {ranked.value} exceeds expert ceiling {self.risk_ceiling.value}"
            )
        return ranked

    def admit_form(
        self,
        form: ModelSizePolicy | str,
        *,
        routing_changing_quality_delta: bool = False,
    ) -> ModelSizePolicy:
        requested = ModelSizePolicy(form)
        if requested not in self.allowed_forms:
            raise ResidualIntelligenceError("form is outside the expert size policy")
        if requested in REMOTE_FORMS and not self.remote_route_permitted:
            raise ResidualIntelligenceError("remote form is outside the family privacy route policy")
        if self.expert_class is ExpertClass.E and requested is not ModelSizePolicy.HUMAN_REVIEW:
            raise ResidualIntelligenceError("class E families cannot route a learned form")
        if form_requires_quality_delta(self.smallest_form, requested) and not (
            routing_changing_quality_delta
        ):
            raise ResidualIntelligenceError(
                "larger form needs a routing-changing quality delta"
            )
        return requested

    def with_form(
        self,
        form: ModelSizePolicy | str,
        *,
        routing_changing_quality_delta: bool = False,
    ) -> ResidualExpertSpec:
        admitted = self.admit_form(
            form, routing_changing_quality_delta=routing_changing_quality_delta
        )
        payload = self.to_dict(include_id=False)
        payload["form"] = admitted.value
        payload["routing_changing_quality_delta"] = bool(
            form_requires_quality_delta(self.smallest_form, admitted)
            and routing_changing_quality_delta
        )
        return ResidualExpertSpec.from_dict(payload)

    def admit_input(self, task_input: ResidualTaskInput) -> ResidualTaskInput:
        admitted = self.family_spec.admit_input(task_input)
        self.admit_risk(admitted.risk_class)
        if admitted.token_budget > self.token_budget:
            raise ResidualIntelligenceError("task input token budget exceeds the expert limit")
        return admitted

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "family_boundary_id": self.family_boundary_id,
            "form": self.form.value,
            "smallest_form": self.smallest_form.value,
            "allowed_forms": [item.value for item in self.allowed_forms],
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "grammar_id": self.grammar_id,
            "token_budget": self.token_budget,
            "maximum_output_bytes": self.maximum_output_bytes,
            "maximum_output_tokens": self.maximum_output_tokens,
            "risk_ceiling": self.risk_ceiling.value,
            "privacy_class": self.privacy_class.value,
            "capabilities": list(self.capabilities),
            "validation_contract": self.validation_contract,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "routing_changing_quality_delta": self.routing_changing_quality_delta,
            "remote_route_permitted": self.remote_route_permitted,
            "always_abstain": self.always_abstain,
            "validator_required": True,
            "prose_default": False,
            "candidate_only": True,
            "authority_class": AUTHORITY_CANDIDATE_ONLY,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualExpertSpec:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"spec_id"},
            noun="residual expert spec",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=ExpertClass(str(payload.get("expert_class") or "")),
            family_boundary_id=str(payload.get("family_boundary_id") or ""),
            form=payload.get("form") or "",
            smallest_form=payload.get("smallest_form") or "",
            allowed_forms=tuple(payload.get("allowed_forms") or ()),
            input_schema=ClosedSchema.from_dict(payload.get("input_schema") or {}),
            output_schema=ClosedSchema.from_dict(payload.get("output_schema") or {}),
            grammar_id=str(payload.get("grammar_id") or ""),
            token_budget=payload.get("token_budget"),
            maximum_output_bytes=payload.get("maximum_output_bytes"),
            maximum_output_tokens=payload.get("maximum_output_tokens"),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            capabilities=tuple(payload.get("capabilities") or ()),
            validation_contract=str(payload.get("validation_contract") or ""),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            routing_changing_quality_delta=payload.get("routing_changing_quality_delta", False),
            remote_route_permitted=payload.get("remote_route_permitted", False),
            always_abstain=payload.get("always_abstain", False),
            validator_required=payload.get("validator_required", True),
            prose_default=payload.get("prose_default", False),
            candidate_only=payload.get("candidate_only", True),
            authority_class=str(payload.get("authority_class") or AUTHORITY_CANDIDATE_ONLY),
        )
        claimed = str(payload.get("spec_id") or "")
        if claimed and claimed != result.spec_id:
            raise ResidualIntelligenceError("residual expert spec identity mismatch")
        return result

    @classmethod
    def from_family(
        cls,
        family: ResidualTaskFamilySpec | ResidualTaskFamily | str,
        *,
        form: ModelSizePolicy | str | None = None,
        routing_changing_quality_delta: bool = False,
    ) -> ResidualExpertSpec:
        spec = family if isinstance(family, ResidualTaskFamilySpec) else family_spec_for(family)
        smallest = (
            ModelSizePolicy.HUMAN_REVIEW
            if spec.expert_class is ExpertClass.E
            else ModelSizePolicy.EXACT_LOOKUP
        )
        requested = smallest if form is None else ModelSizePolicy(form)
        allowed = allowed_forms_for(spec, smallest_form=smallest)
        return cls(
            task_family=spec.task_family,
            expert_class=spec.expert_class,
            family_boundary_id=spec.family_boundary_id,
            form=requested,
            smallest_form=smallest,
            allowed_forms=allowed,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            grammar_id=spec.grammar_id,
            token_budget=spec.token_budget,
            maximum_output_bytes=spec.maximum_output_bytes,
            maximum_output_tokens=spec.maximum_output_tokens,
            risk_ceiling=spec.risk_ceiling,
            privacy_class=spec.privacy_class,
            capabilities=spec.capabilities,
            validation_contract=spec.validation_contract,
            error_behavior=spec.error_behavior,
            abstention_behavior=spec.abstention_behavior,
            routing_changing_quality_delta=routing_changing_quality_delta,
            remote_route_permitted=spec.remote_route_permitted,
            always_abstain=spec.always_abstain,
        )


def _build_expert_specs() -> dict[ResidualTaskFamily, ResidualExpertSpec]:
    specs = tuple(
        ResidualExpertSpec.from_family(family_spec)
        for family_spec in DEFAULT_TASK_FAMILY_SPECS.values()
    )
    families = [item.task_family for item in specs]
    if set(families) != set(ResidualTaskFamily) or len(families) != len(set(families)):
        raise ResidualIntelligenceError("expert specs must cover the closed 24-family taxonomy")
    if {item.expert_class for item in specs} != set(ExpertClass):
        raise ResidualIntelligenceError("expert specs must populate expert classes A through E")
    return {item.task_family: item for item in specs}


DEFAULT_EXPERT_SPECS: Final[Mapping[ResidualTaskFamily, ResidualExpertSpec]] = _build_expert_specs()


def expert_spec_for(task_family: ResidualTaskFamily | str) -> ResidualExpertSpec:
    family = ResidualTaskFamily(task_family)
    try:
        return DEFAULT_EXPERT_SPECS[family]
    except KeyError as exc:
        raise ResidualIntelligenceError(f"missing expert spec for {family.value}") from exc


__all__ = (
    "DEFAULT_EXPERT_SPECS",
    "EXPERT_SPEC_REGISTRY_SCHEMA",
    "EXPERT_SPEC_SCHEMA",
    "LEARNED_FORMS",
    "REMOTE_FORMS",
    "SAFE_FALLBACK_FORMS",
    "SMALLEST_FORM_ORDER",
    "ExpertClass",
    "ModelSizePolicy",
    "ResidualExpertSpec",
    "ResidualTaskFamilySpec",
    "allowed_forms_for",
    "expert_spec_for",
    "family_spec_for",
    "form_rank",
    "form_requires_quality_delta",
)
