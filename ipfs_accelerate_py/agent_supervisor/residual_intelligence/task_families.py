"""Closed residual task-family specifications.

Families share input semantics, output semantics, risk, authority, validation,
error, and abstention behavior. Prompt or embedding similarity is not a family
boundary. Specifications carry no examples.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping
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
    optional_text,
    required_text,
    strict_fields,
    text_tuple,
)
from .inventory import ResidualFamilyBoundary
from .structured_decoding import grammar_for

TASK_FAMILY_SPEC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-task-family-spec@1"
)
CLOSED_VALUE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-closed-schema@1"
MAX_TOKEN_LIMIT: Final = 1_000_000
AUTHORITY_CLASS: Final = "candidate_only"

CLOSED_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        "typed_candidate",
        "local_cpu",
        "local_gpu",
        "batch",
        "grammar_constrained",
        "exact_lookup",
        "declarative_rule",
        "linear_logistic",
        "ranker_encoder",
        "structured_decode",
        "procedure_compiler",
        "prover",
        "isolated_worktree",
        "remote_provider",
        "abstain",
    }
)
CLOSED_PRIVACY_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "local_only",
        "authorized_internal",
        "authorized_public_provider",
        "never_remote",
        "never_shared_expert",
    }
)
CLOSED_ERROR_CODES: Final[tuple[str, ...]] = (
    "invalid_output",
    "reject_input",
    "unsupported_family_risk",
    "schema_violation",
    "grammar_violation",
    "validator_required",
)
CLOSED_ABSTENTION_CODES: Final[tuple[str, ...]] = (
    "out_of_distribution",
    "capability_unavailable",
    "critical_boundary",
    "unknown_repository",
    "unknown_authority",
    "incomplete_context",
    "novel_unbounded",
)
RISK_ORDER: Final[tuple[RiskClass, ...]] = (
    RiskClass.R0,
    RiskClass.R1,
    RiskClass.R2,
    RiskClass.R3,
    RiskClass.R4,
    RiskClass.R5,
)


class ExpertClass(str, Enum):
    """Closed five-class expert form taxonomy in smallest-first order."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class ModelSizePolicy(str, Enum):
    """Closed model-size ceiling; a larger size needs a routing-changing delta."""

    NONE = "none"
    LINEAR = "linear"
    SMALL_RANKER = "small_ranker"
    STRUCTURED_SPECIALIST = "structured_specialist"
    PARAMETER_EFFICIENT_ADAPTER = "parameter_efficient_adapter"
    QUANTIZED_LOCAL_GENERAL = "quantized_local_general"
    REMOTE_STANDARD = "remote_standard"
    REMOTE_STRONG = "remote_strong"


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
    ExpertClass.E: "constrained_structured_decoder",
}
MODEL_SIZE_ORDER: Final[tuple[ModelSizePolicy, ...]] = (
    ModelSizePolicy.NONE,
    ModelSizePolicy.LINEAR,
    ModelSizePolicy.SMALL_RANKER,
    ModelSizePolicy.STRUCTURED_SPECIALIST,
    ModelSizePolicy.PARAMETER_EFFICIENT_ADAPTER,
    ModelSizePolicy.QUANTIZED_LOCAL_GENERAL,
    ModelSizePolicy.REMOTE_STANDARD,
    ModelSizePolicy.REMOTE_STRONG,
)
DEFAULT_SIZE_FOR_CLASS: Final[Mapping[ExpertClass, ModelSizePolicy]] = {
    ExpertClass.A: ModelSizePolicy.NONE,
    ExpertClass.B: ModelSizePolicy.NONE,
    ExpertClass.C: ModelSizePolicy.LINEAR,
    ExpertClass.D: ModelSizePolicy.SMALL_RANKER,
    ExpertClass.E: ModelSizePolicy.STRUCTURED_SPECIALIST,
}


def expert_class_rank(value: ExpertClass | str) -> int:
    return SMALLEST_FORM_ORDER.index(ExpertClass(value))


def model_size_rank(value: ModelSizePolicy | str) -> int:
    return MODEL_SIZE_ORDER.index(ModelSizePolicy(value))


def risk_rank(value: RiskClass | str) -> int:
    return RISK_ORDER.index(RiskClass(value))


def risks_between(floor: RiskClass, ceiling: RiskClass) -> tuple[RiskClass, ...]:
    low = risk_rank(floor)
    high = risk_rank(ceiling)
    if low > high:
        raise ResidualIntelligenceError("risk floor exceeds risk ceiling")
    return RISK_ORDER[low : high + 1]


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _closed_tokens(
    values: Any,
    name: str,
    *,
    allowed: frozenset[str],
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = text_tuple(values, name, allow_empty=allow_empty, max_items=64)
    unknown = sorted(item for item in result if item not in allowed)
    if unknown:
        raise ResidualIntelligenceError(f"{name} contains unknown values: {', '.join(unknown)}")
    return result


@dataclass(frozen=True)
class ClosedValueSchema:
    """Closed field set; unknown keys and prose defaults fail closed."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    enumerations: Mapping[str, tuple[str, ...]]
    maximum_bytes: int
    allow_prose: bool = False
    schema: str = CLOSED_VALUE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "required_fields",
            "optional_fields",
            "enumerations",
            "maximum_bytes",
            "allow_prose",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CLOSED_VALUE_SCHEMA:
            raise ResidualIntelligenceError("unsupported closed value schema")
        required = text_tuple(self.required_fields, "required_fields")
        optional = text_tuple(self.optional_fields, "optional_fields")
        overlap = sorted(set(required) & set(optional))
        if overlap:
            raise ResidualIntelligenceError(
                "closed schema fields cannot be both required and optional: "
                + ", ".join(overlap)
            )
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "optional_fields", optional)
        if not isinstance(self.enumerations, Mapping):
            raise ResidualIntelligenceError("enumerations must be an object")
        enums: dict[str, tuple[str, ...]] = {}
        allowed = set(required) | set(optional)
        for key, values in self.enumerations.items():
            name = required_text(key, "enumeration field", max_bytes=256)
            if name not in allowed:
                raise ResidualIntelligenceError(f"enumeration {name} is not a declared field")
            enums[name] = text_tuple(values, f"enumerations.{name}", allow_empty=False)
        object.__setattr__(self, "enumerations", enums)
        object.__setattr__(
            self,
            "maximum_bytes",
            bounded_int(self.maximum_bytes, "maximum_bytes", minimum=128, maximum=32_768),
        )
        object.__setattr__(self, "allow_prose", _require_bool(self.allow_prose, "allow_prose"))
        if self.allow_prose:
            raise ResidualIntelligenceError("prose is not a permitted default schema mode")

    @property
    def allowed_fields(self) -> frozenset[str]:
        return frozenset(self.required_fields) | frozenset(self.optional_fields)

    def validate_mapping(self, payload: Mapping[str, Any], *, noun: str) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError(f"{noun} must be an object")
        strict_fields(
            payload,
            allowed=self.allowed_fields,
            required=self.required_fields,
            noun=noun,
        )
        result: dict[str, Any] = dict(payload)
        for field, choices in self.enumerations.items():
            if field in result and result[field] not in choices:
                raise ResidualIntelligenceError(f"{noun}.{field} is outside its closed enumeration")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "enumerations": {
                key: list(values) for key, values in sorted(self.enumerations.items())
            },
            "maximum_bytes": self.maximum_bytes,
            "allow_prose": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClosedValueSchema:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS,
            noun="closed value schema",
        )
        enumerations = payload.get("enumerations") or {}
        if not isinstance(enumerations, Mapping):
            raise ResidualIntelligenceError("enumerations must be an object")
        return cls(
            schema=str(payload.get("schema") or ""),
            required_fields=tuple(payload.get("required_fields") or ()),
            optional_fields=tuple(payload.get("optional_fields") or ()),
            enumerations={
                str(key): tuple(values) for key, values in enumerations.items()
            },
            maximum_bytes=payload.get("maximum_bytes"),
            allow_prose=payload.get("allow_prose"),
        )


@dataclass(frozen=True)
class ResidualTaskFamilySpec:
    """Exact semantic family boundary plus closed I/O, risk, and validation."""

    task_family: ResidualTaskFamily
    input_semantics: str
    output_semantics: str
    risk_floor: RiskClass
    risk_ceiling: RiskClass
    authority_class: str
    validation_contract: str
    validator_required: bool
    error_behavior: str
    abstention_behavior: str
    error_codes: tuple[str, ...]
    abstention_codes: tuple[str, ...]
    input_schema: ClosedValueSchema
    output_schema: ClosedValueSchema
    compact_feature_keys: tuple[str, ...]
    allowed_outputs: tuple[str, ...]
    grammar_id: str
    max_input_tokens: int
    max_output_tokens: int
    max_output_bytes: int
    privacy_class: PrivacyClass
    privacy_route_policy: str
    preferred_expert_class: ExpertClass
    allowed_expert_classes: tuple[ExpertClass, ...]
    model_size_ceiling: ModelSizePolicy
    hardware_class: str
    runtime_requirements: tuple[str, ...]
    capabilities: tuple[str, ...]
    emits_prose_by_default: bool = False
    evaluation_dataset_reference: str = ""
    training_dataset_reference: str = ""
    schema: str = TASK_FAMILY_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "family_spec_id",
            "task_family",
            "input_semantics",
            "output_semantics",
            "risk_floor",
            "risk_ceiling",
            "allowed_risks",
            "authority_class",
            "validation_contract",
            "validator_required",
            "error_behavior",
            "abstention_behavior",
            "error_codes",
            "abstention_codes",
            "input_schema",
            "output_schema",
            "compact_feature_keys",
            "allowed_outputs",
            "grammar_id",
            "max_input_tokens",
            "max_output_tokens",
            "max_output_bytes",
            "privacy_class",
            "privacy_route_policy",
            "preferred_expert_class",
            "allowed_expert_classes",
            "model_size_ceiling",
            "hardware_class",
            "runtime_requirements",
            "capabilities",
            "emits_prose_by_default",
            "evaluation_dataset_reference",
            "training_dataset_reference",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != TASK_FAMILY_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual task family spec schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "risk_floor", RiskClass(self.risk_floor))
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        object.__setattr__(
            self, "preferred_expert_class", ExpertClass(self.preferred_expert_class)
        )
        object.__setattr__(self, "model_size_ceiling", ModelSizePolicy(self.model_size_ceiling))
        for field in (
            "input_semantics",
            "output_semantics",
            "authority_class",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "grammar_id",
            "hardware_class",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.authority_class != AUTHORITY_CLASS:
            raise ResidualIntelligenceError(
                "residual expert family authority_class must be candidate_only"
            )
        object.__setattr__(
            self, "validator_required", _require_bool(self.validator_required, "validator_required")
        )
        if not self.validator_required:
            raise ResidualIntelligenceError("every family requires an independent validator")
        object.__setattr__(
            self,
            "emits_prose_by_default",
            _require_bool(self.emits_prose_by_default, "emits_prose_by_default"),
        )
        if self.emits_prose_by_default:
            raise ResidualIntelligenceError("typed structured output is the default; prose is not")
        if not isinstance(self.input_schema, ClosedValueSchema):
            raise ResidualIntelligenceError("input_schema must be ClosedValueSchema")
        if not isinstance(self.output_schema, ClosedValueSchema):
            raise ResidualIntelligenceError("output_schema must be ClosedValueSchema")
        object.__setattr__(
            self,
            "error_codes",
            _closed_tokens(self.error_codes, "error_codes", allowed=frozenset(CLOSED_ERROR_CODES)),
        )
        object.__setattr__(
            self,
            "abstention_codes",
            _closed_tokens(
                self.abstention_codes,
                "abstention_codes",
                allowed=frozenset(CLOSED_ABSTENTION_CODES),
            ),
        )
        object.__setattr__(
            self,
            "compact_feature_keys",
            text_tuple(self.compact_feature_keys, "compact_feature_keys"),
        )
        object.__setattr__(
            self,
            "allowed_outputs",
            text_tuple(self.allowed_outputs, "allowed_outputs", allow_empty=False),
        )
        grammar = grammar_for(self.task_family)
        if self.grammar_id != grammar.grammar_id:
            raise ResidualIntelligenceError("family grammar_id does not match the closed grammar")
        if set(self.allowed_outputs) != set(grammar.output_classes):
            raise ResidualIntelligenceError("allowed_outputs must equal the closed grammar classes")
        if grammar.abstention_output_class not in self.allowed_outputs:
            raise ResidualIntelligenceError("family allowed_outputs must include abstention")
        object.__setattr__(
            self,
            "max_input_tokens",
            bounded_int(self.max_input_tokens, "max_input_tokens", minimum=1, maximum=MAX_TOKEN_LIMIT),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            bounded_int(
                self.max_output_tokens, "max_output_tokens", minimum=1, maximum=MAX_TOKEN_LIMIT
            ),
        )
        object.__setattr__(
            self,
            "max_output_bytes",
            bounded_int(self.max_output_bytes, "max_output_bytes", minimum=128, maximum=32_768),
        )
        if self.max_output_bytes != grammar.maximum_output_bytes:
            raise ResidualIntelligenceError("max_output_bytes must match the family grammar bound")
        if self.output_schema.maximum_bytes != self.max_output_bytes:
            raise ResidualIntelligenceError("output schema byte bound must match max_output_bytes")
        if self.privacy_route_policy not in CLOSED_PRIVACY_ROUTES:
            raise ResidualIntelligenceError("privacy_route_policy is outside the closed route set")
        classes = tuple(ExpertClass(item) for item in self.allowed_expert_classes)
        if not classes:
            raise ResidualIntelligenceError("allowed_expert_classes must not be empty")
        if len(set(classes)) != len(classes):
            raise ResidualIntelligenceError("allowed_expert_classes contains duplicates")
        ordered = tuple(sorted(classes, key=expert_class_rank))
        if classes != ordered:
            raise ResidualIntelligenceError("allowed_expert_classes must be in smallest-form order")
        if self.preferred_expert_class not in classes:
            raise ResidualIntelligenceError("preferred expert class is not allowed for this family")
        object.__setattr__(self, "allowed_expert_classes", classes)
        preferred_size = DEFAULT_SIZE_FOR_CLASS[self.preferred_expert_class]
        if model_size_rank(preferred_size) > model_size_rank(self.model_size_ceiling):
            raise ResidualIntelligenceError("preferred form exceeds the family model-size ceiling")
        object.__setattr__(
            self,
            "runtime_requirements",
            text_tuple(self.runtime_requirements, "runtime_requirements", allow_empty=False),
        )
        object.__setattr__(
            self,
            "capabilities",
            _closed_tokens(
                self.capabilities,
                "capabilities",
                allowed=CLOSED_CAPABILITIES,
                allow_empty=False,
            ),
        )
        if "typed_candidate" not in self.capabilities or "abstain" not in self.capabilities:
            raise ResidualIntelligenceError("family capabilities must include typed_candidate and abstain")
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
        if self.task_family is ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING:
            if self.allowed_outputs != ("ABSTAIN",):
                raise ResidualIntelligenceError(
                    "NOVEL_UNBOUNDED_REASONING may only emit the abstention class"
                )
            if self.allowed_expert_classes != (ExpertClass.A,):
                raise ResidualIntelligenceError(
                    "NOVEL_UNBOUNDED_REASONING is limited to exact-lookup abstention"
                )

    @property
    def allowed_risks(self) -> tuple[RiskClass, ...]:
        return risks_between(self.risk_floor, self.risk_ceiling)

    @property
    def family_spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def max_token_budget(self) -> int:
        return self.max_input_tokens + self.max_output_tokens

    def to_boundary(self) -> ResidualFamilyBoundary:
        return ResidualFamilyBoundary(
            task_family=self.task_family,
            input_semantics=self.input_semantics,
            output_semantics=self.output_semantics,
            risk_class=self.risk_ceiling,
            authority_class=self.authority_class,
            validation_contract=self.validation_contract,
            error_behavior=self.error_behavior,
            abstention_behavior=self.abstention_behavior,
        )

    def assert_risk_allowed(self, risk: RiskClass | str) -> RiskClass:
        classified = RiskClass(risk)
        if classified not in self.allowed_risks:
            raise ResidualIntelligenceError(
                f"unsupported family-risk pair {self.task_family.value}/{classified.value} "
                f"exceeds risk ceiling {self.risk_ceiling.value}"
                if risk_rank(classified) > risk_rank(self.risk_ceiling)
                else (
                    f"unsupported family-risk pair {self.task_family.value}/{classified.value}"
                )
            )
        return classified

    def assert_expert_class_allowed(self, expert_class: ExpertClass | str) -> ExpertClass:
        classified = ExpertClass(expert_class)
        if classified not in self.allowed_expert_classes:
            raise ResidualIntelligenceError(
                f"expert class {classified.value} is outside family {self.task_family.value}"
            )
        return classified

    def assert_model_size_allowed(self, policy: ModelSizePolicy | str) -> ModelSizePolicy:
        classified = ModelSizePolicy(policy)
        if model_size_rank(classified) > model_size_rank(self.model_size_ceiling):
            raise ResidualIntelligenceError(
                f"model size {classified.value} exceeds family ceiling {self.model_size_ceiling.value}"
            )
        if classified in {ModelSizePolicy.REMOTE_STANDARD, ModelSizePolicy.REMOTE_STRONG}:
            if self.privacy_route_policy in {"local_only", "never_remote", "never_shared_expert"}:
                raise ResidualIntelligenceError(
                    "privacy route policy forbids a remote model-size policy"
                )
        return classified

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "risk_floor": self.risk_floor.value,
            "risk_ceiling": self.risk_ceiling.value,
            "allowed_risks": [item.value for item in self.allowed_risks],
            "authority_class": self.authority_class,
            "validation_contract": self.validation_contract,
            "validator_required": True,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "error_codes": list(self.error_codes),
            "abstention_codes": list(self.abstention_codes),
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "compact_feature_keys": list(self.compact_feature_keys),
            "allowed_outputs": list(self.allowed_outputs),
            "grammar_id": self.grammar_id,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_output_bytes": self.max_output_bytes,
            "privacy_class": self.privacy_class.value,
            "privacy_route_policy": self.privacy_route_policy,
            "preferred_expert_class": self.preferred_expert_class.value,
            "allowed_expert_classes": [item.value for item in self.allowed_expert_classes],
            "model_size_ceiling": self.model_size_ceiling.value,
            "hardware_class": self.hardware_class,
            "runtime_requirements": list(self.runtime_requirements),
            "capabilities": list(self.capabilities),
            "emits_prose_by_default": False,
            "evaluation_dataset_reference": self.evaluation_dataset_reference,
            "training_dataset_reference": self.training_dataset_reference,
        }
        if include_id:
            result["family_spec_id"] = self.family_spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualTaskFamilySpec:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"family_spec_id", "allowed_risks"},
            noun="residual task family spec",
        )
        input_schema = payload.get("input_schema")
        output_schema = payload.get("output_schema")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            input_semantics=str(payload.get("input_semantics") or ""),
            output_semantics=str(payload.get("output_semantics") or ""),
            risk_floor=RiskClass(str(payload.get("risk_floor") or "")),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            authority_class=str(payload.get("authority_class") or ""),
            validation_contract=str(payload.get("validation_contract") or ""),
            validator_required=payload.get("validator_required"),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            error_codes=tuple(payload.get("error_codes") or ()),
            abstention_codes=tuple(payload.get("abstention_codes") or ()),
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
            compact_feature_keys=tuple(payload.get("compact_feature_keys") or ()),
            allowed_outputs=tuple(payload.get("allowed_outputs") or ()),
            grammar_id=str(payload.get("grammar_id") or ""),
            max_input_tokens=payload.get("max_input_tokens"),
            max_output_tokens=payload.get("max_output_tokens"),
            max_output_bytes=payload.get("max_output_bytes"),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            privacy_route_policy=str(payload.get("privacy_route_policy") or ""),
            preferred_expert_class=ExpertClass(str(payload.get("preferred_expert_class") or "")),
            allowed_expert_classes=tuple(payload.get("allowed_expert_classes") or ()),
            model_size_ceiling=ModelSizePolicy(str(payload.get("model_size_ceiling") or "")),
            hardware_class=str(payload.get("hardware_class") or ""),
            runtime_requirements=tuple(payload.get("runtime_requirements") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            emits_prose_by_default=payload.get("emits_prose_by_default"),
            evaluation_dataset_reference=str(payload.get("evaluation_dataset_reference") or ""),
            training_dataset_reference=str(payload.get("training_dataset_reference") or ""),
        )
        claimed_risks = payload.get("allowed_risks")
        if claimed_risks is not None:
            observed = tuple(RiskClass(str(item)) for item in claimed_risks)
            if observed != result.allowed_risks:
                raise ResidualIntelligenceError("allowed_risks does not match floor/ceiling")
        claimed = str(payload.get("family_spec_id") or "")
        if claimed and claimed != result.family_spec_id:
            raise ResidualIntelligenceError("residual task family spec identity mismatch")
        return result


def _input_schema() -> ClosedValueSchema:
    return ClosedValueSchema(
        required_fields=(
            "task_family",
            "question_id",
            "repository_state_cid",
            "objective_cid",
            "task_cid",
            "policy_cid",
            "context_capsule_cid",
            "compact_features",
            "allowed_outputs",
            "risk_class",
            "validation_policy",
            "token_budget",
        ),
        optional_fields=(),
        enumerations={"task_family": tuple(item.value for item in ResidualTaskFamily)},
        maximum_bytes=16_384,
        allow_prose=False,
    )


def _output_schema_from_grammar(family: ResidualTaskFamily) -> ClosedValueSchema:
    grammar = grammar_for(family)
    enumerations = {
        name: contract.allowed_values
        for name, contract in grammar.field_contracts.items()
        if contract.allowed_values
    }
    required = grammar.required_payload_fields
    optional = tuple(name for name in grammar.payload_fields if name not in required)
    return ClosedValueSchema(
        required_fields=required,
        optional_fields=optional,
        enumerations=enumerations,
        maximum_bytes=grammar.maximum_output_bytes,
        allow_prose=False,
    )


def _family(
    family: ResidualTaskFamily,
    *,
    input_semantics: str,
    output_semantics: str,
    risk_floor: RiskClass,
    risk_ceiling: RiskClass,
    privacy_class: PrivacyClass,
    privacy_route_policy: str,
    preferred_expert_class: ExpertClass,
    model_size_ceiling: ModelSizePolicy,
    hardware_class: str,
    runtime_requirements: tuple[str, ...],
    capabilities: tuple[str, ...],
    validation_contract: str,
    error_behavior: str,
    abstention_behavior: str,
    compact_feature_keys: tuple[str, ...],
    max_input_tokens: int,
    max_output_tokens: int,
    allowed_expert_classes: tuple[ExpertClass, ...] = SMALLEST_FORM_ORDER,
    abstention_codes: tuple[str, ...] = (
        "out_of_distribution",
        "capability_unavailable",
        "incomplete_context",
    ),
) -> ResidualTaskFamilySpec:
    grammar = grammar_for(family)
    return ResidualTaskFamilySpec(
        task_family=family,
        input_semantics=input_semantics,
        output_semantics=output_semantics,
        risk_floor=risk_floor,
        risk_ceiling=risk_ceiling,
        authority_class=AUTHORITY_CLASS,
        validation_contract=validation_contract,
        validator_required=True,
        error_behavior=error_behavior,
        abstention_behavior=abstention_behavior,
        error_codes=CLOSED_ERROR_CODES,
        abstention_codes=abstention_codes,
        input_schema=_input_schema(),
        output_schema=_output_schema_from_grammar(family),
        compact_feature_keys=compact_feature_keys,
        allowed_outputs=grammar.output_classes,
        grammar_id=grammar.grammar_id,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_output_bytes=grammar.maximum_output_bytes,
        privacy_class=privacy_class,
        privacy_route_policy=privacy_route_policy,
        preferred_expert_class=preferred_expert_class,
        allowed_expert_classes=allowed_expert_classes,
        model_size_ceiling=model_size_ceiling,
        hardware_class=hardware_class,
        runtime_requirements=runtime_requirements,
        capabilities=capabilities,
        emits_prose_by_default=False,
    )


_LOCAL_TYPED: Final[tuple[str, ...]] = (
    "typed_candidate",
    "local_cpu",
    "abstain",
    "grammar_constrained",
)
_CLASSIFY: Final[tuple[str, ...]] = _LOCAL_TYPED + ("exact_lookup", "declarative_rule")
_LINEAR: Final[tuple[str, ...]] = _CLASSIFY + ("linear_logistic", "batch")
_RANK: Final[tuple[str, ...]] = _LINEAR + ("ranker_encoder",)
_STRUCTURED: Final[tuple[str, ...]] = _RANK + ("structured_decode",)


DEFAULT_FAMILY_SPECS: Final[Mapping[ResidualTaskFamily, ResidualTaskFamilySpec]] = {
    ResidualTaskFamily.TASK_CLASSIFICATION: _family(
        ResidualTaskFamily.TASK_CLASSIFICATION,
        input_semantics="bounded residual question features and declared taxonomy membership",
        output_semantics="one closed task-family label candidate",
        risk_floor=RiskClass.R0,
        risk_ceiling=RiskClass.R2,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="authorized_internal",
        preferred_expert_class=ExpertClass.A,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_CLASSIFY,
        validation_contract="validator:task-classification@1",
        error_behavior="unknown taxonomy values and extra fields fail as invalid_output",
        abstention_behavior="unknown question types abstain rather than invent a family",
        compact_feature_keys=("question_type", "reference_ids"),
        max_input_tokens=512,
        max_output_tokens=64,
    ),
    ResidualTaskFamily.RISK_CLASSIFICATION: _family(
        ResidualTaskFamily.RISK_CLASSIFICATION,
        input_semantics="declared effects, authority needs, and bounded context sufficiency flags",
        output_semantics="one closed risk-class label candidate",
        risk_floor=RiskClass.R1,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="authorized_internal",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_CLASSIFY,
        validation_contract="validator:risk-classification@1",
        error_behavior="under-classification or unknown labels fail closed",
        abstention_behavior="unknown effects or authority abstain to the higher conservative class",
        compact_feature_keys=("effect_classes", "authority_class", "context_complete"),
        max_input_tokens=512,
        max_output_tokens=64,
    ),
    ResidualTaskFamily.EFFECT_CLASSIFICATION: _family(
        ResidualTaskFamily.EFFECT_CLASSIFICATION,
        input_semantics="bounded operation features and declared effect catalog references",
        output_semantics="one or more closed effect-class tokens",
        risk_floor=RiskClass.R1,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="authorized_internal",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_CLASSIFY,
        validation_contract="validator:effect-classification@1",
        error_behavior="unknown effect tokens and extra fields fail as invalid_output",
        abstention_behavior="unseen operations abstain instead of inventing effects",
        compact_feature_keys=("operation_id", "path_ids", "effect_hints"),
        max_input_tokens=512,
        max_output_tokens=128,
    ),
    ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION: _family(
        ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION,
        input_semantics="declared operation, effect, and policy identities requiring an authority class",
        output_semantics="one closed authority-requirement label candidate",
        risk_floor=RiskClass.R3,
        risk_ceiling=RiskClass.R4,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir", "authority_catalog"),
        capabilities=_CLASSIFY,
        validation_contract="validator:authority-requirement@1",
        error_behavior="authority-shaped or completion-shaped labels fail closed",
        abstention_behavior="unknown authority or policy identities abstain",
        compact_feature_keys=("operation_id", "effect_classes", "policy_cid"),
        max_input_tokens=512,
        max_output_tokens=64,
        abstention_codes=(
            "out_of_distribution",
            "unknown_authority",
            "capability_unavailable",
        ),
    ),
    ResidualTaskFamily.CONTEXT_SUFFICIENCY: _family(
        ResidualTaskFamily.CONTEXT_SUFFICIENCY,
        input_semantics="compact context-capsule identity and missing-reference candidates",
        output_semantics="boolean sufficiency with bounded missing-reference identifiers",
        risk_floor=RiskClass.R1,
        risk_ceiling=RiskClass.R2,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir", "context_governor"),
        capabilities=_CLASSIFY,
        validation_contract="validator:context-sufficiency@1",
        error_behavior="untyped sufficiency values fail as invalid_output",
        abstention_behavior="incomplete capsules abstain rather than asserting sufficiency",
        compact_feature_keys=("context_capsule_cid", "required_reference_ids"),
        max_input_tokens=1024,
        max_output_tokens=128,
        abstention_codes=(
            "incomplete_context",
            "out_of_distribution",
            "capability_unavailable",
        ),
    ),
    ResidualTaskFamily.EVIDENCE_RANKING: _family(
        ResidualTaskFamily.EVIDENCE_RANKING,
        input_semantics="bounded evidence reference identifiers with compact ranking features",
        output_semantics="descending scored ranking over declared evidence identifiers",
        risk_floor=RiskClass.R1,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.D,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-medium",
        runtime_requirements=("typed_ir",),
        capabilities=_RANK,
        validation_contract="validator:evidence-ranking@1",
        error_behavior="misaligned scores or unknown identifiers fail as invalid_output",
        abstention_behavior="empty or unseen evidence sets abstain",
        compact_feature_keys=("candidate_ids", "feature_matrix_cid"),
        max_input_tokens=1024,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.PROCEDURE_MATCHING: _family(
        ResidualTaskFamily.PROCEDURE_MATCHING,
        input_semantics="exact procedure catalog identity plus declared precondition references",
        output_semantics="one procedure identifier and closed match class",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.A,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir", "procedure_compiler"),
        capabilities=_CLASSIFY + ("procedure_compiler",),
        validation_contract="validator:procedure-matching@1",
        error_behavior="precondition mismatch fails closed without substituting a procedure",
        abstention_behavior="no exact catalog match abstains",
        compact_feature_keys=("procedure_catalog_cid", "precondition_reference_ids"),
        max_input_tokens=1024,
        max_output_tokens=128,
    ),
    ResidualTaskFamily.PLAN_BRANCH_RANKING: _family(
        ResidualTaskFamily.PLAN_BRANCH_RANKING,
        input_semantics="bounded plan-branch identifiers with compact obligation features",
        output_semantics="descending scored ranking over declared plan branches",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.D,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-medium",
        runtime_requirements=("typed_ir", "planner"),
        capabilities=_RANK,
        validation_contract="validator:plan-branch-ranking@1",
        error_behavior="unknown branch identifiers fail as invalid_output",
        abstention_behavior="incomplete plan context abstains",
        compact_feature_keys=("branch_ids", "obligation_reference_ids"),
        max_input_tokens=1024,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.TEST_SELECTION: _family(
        ResidualTaskFamily.TEST_SELECTION,
        input_semantics="changed-symbol and coverage references for selecting existing tests",
        output_semantics="closed test-identifier list without deletions or weakening",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir", "test_index"),
        capabilities=_RANK,
        validation_contract="validator:test-selection@1",
        error_behavior="test deletion or unknown test identifiers fail closed",
        abstention_behavior="missing coverage map abstains rather than dropping tests",
        compact_feature_keys=("changed_symbol_ids", "coverage_reference_ids"),
        max_input_tokens=1024,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.PROOF_SELECTION: _family(
        ResidualTaskFamily.PROOF_SELECTION,
        input_semantics="exact current prover obligation and candidate proof identities",
        output_semantics="ranked proof identifiers that remain nominations",
        risk_floor=RiskClass.R4,
        risk_ceiling=RiskClass.R5,
        privacy_class=PrivacyClass.PROOF_WITNESS,
        privacy_route_policy="never_remote",
        preferred_expert_class=ExpertClass.D,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-medium-prover",
        runtime_requirements=("typed_ir", "prover"),
        capabilities=_RANK + ("prover",),
        validation_contract="validator:proof-selection@1",
        error_behavior="omitted prover check or stale obligation fails closed",
        abstention_behavior="missing prover capability or unknown obligation abstains",
        compact_feature_keys=("obligation_id", "proof_candidate_ids"),
        max_input_tokens=2048,
        max_output_tokens=256,
        abstention_codes=(
            "capability_unavailable",
            "out_of_distribution",
            "incomplete_context",
        ),
    ),
    ResidualTaskFamily.FAILURE_ATTRIBUTION: _family(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        input_semantics="validated failure signature plus bounded dependency references",
        output_semantics="one failure class and one bounded action candidate",
        risk_floor=RiskClass.R1,
        risk_ceiling=RiskClass.R2,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.C,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_LINEAR,
        validation_contract="validator:failure-attribution@1",
        error_behavior="invalid output or failed validation escalates",
        abstention_behavior="unknown signatures abstain",
        compact_feature_keys=("exit_code", "failure_signature"),
        max_input_tokens=512,
        max_output_tokens=64,
    ),
    ResidualTaskFamily.RETRY_OR_ESCALATE: _family(
        ResidualTaskFamily.RETRY_OR_ESCALATE,
        input_semantics="bounded failure class, attempt count, and capability flags",
        output_semantics="one closed retry, escalate, or stop decision",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="authorized_internal",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_CLASSIFY,
        validation_contract="validator:retry-or-escalate@1",
        error_behavior="unknown decision tokens fail as invalid_output",
        abstention_behavior="exhausted attempts without a safe action abstain to human review",
        compact_feature_keys=("failure_class", "attempt_count", "capability_available"),
        max_input_tokens=512,
        max_output_tokens=64,
    ),
    ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION: _family(
        ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION,
        input_semantics="exact content identities and declared dependency references",
        output_semantics="boolean reuse candidate with dependency identifiers",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.A,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir", "content_store"),
        capabilities=_CLASSIFY,
        validation_contract="validator:cache-reuse@1",
        error_behavior="identity mismatch cannot be recovered as reuse",
        abstention_behavior="stale or unknown identities abstain",
        compact_feature_keys=("cache_entry_cid", "dependency_reference_ids"),
        max_input_tokens=512,
        max_output_tokens=64,
    ),
    ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION: _family(
        ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION,
        input_semantics="conflicted path and symbol identities with bounded merge features",
        output_semantics="one conflict class plus symbol identifiers",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_CLASSIFY,
        validation_contract="validator:merge-conflict@1",
        error_behavior="binary or out-of-path conflict payloads fail closed",
        abstention_behavior="unrecognized conflict shapes abstain",
        compact_feature_keys=("path_ids", "symbol_ids"),
        max_input_tokens=1024,
        max_output_tokens=128,
    ),
    ResidualTaskFamily.PATCH_TEMPLATE_SELECTION: _family(
        ResidualTaskFamily.PATCH_TEMPLATE_SELECTION,
        input_semantics="failure class, symbol identities, and admitted template catalog references",
        output_semantics="one template identifier candidate",
        risk_floor=RiskClass.R3,
        risk_ceiling=RiskClass.R4,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.C,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_LINEAR,
        validation_contract="validator:patch-template-selection@1",
        error_behavior="unknown templates or authority-shaped payloads fail closed",
        abstention_behavior="no catalog match abstains",
        compact_feature_keys=("failure_class", "symbol_ids", "template_catalog_cid"),
        max_input_tokens=1024,
        max_output_tokens=128,
    ),
    ResidualTaskFamily.PROCEDURE_HOLE_FILLING: _family(
        ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
        input_semantics="declared typed hole, compiler preconditions, and operator catalog",
        output_semantics="one ProcedureHoleResolution candidate under compiler bounds",
        risk_floor=RiskClass.R3,
        risk_ceiling=RiskClass.R4,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.E,
        model_size_ceiling=ModelSizePolicy.STRUCTURED_SPECIALIST,
        hardware_class="cpu-gpu-optional",
        runtime_requirements=("typed_ir", "procedure_compiler", "structured_decode"),
        capabilities=_STRUCTURED + ("procedure_compiler",),
        validation_contract="validator:procedure-hole@1",
        error_behavior="undeclared holes, extra fields, or shell payloads fail as invalid_output",
        abstention_behavior="unsatisfied preconditions or missing compiler abstain",
        compact_feature_keys=("hole_id", "precondition_reference_ids", "operator_catalog_cid"),
        max_input_tokens=2048,
        max_output_tokens=512,
    ),
    ResidualTaskFamily.PATCH_SKETCH_GENERATION: _family(
        ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        input_semantics="exact allowed paths, symbols, and predetermined validation identities",
        output_semantics="bounded PatchSketchIR candidate with no test deletion or authority change",
        risk_floor=RiskClass.R4,
        risk_ceiling=RiskClass.R5,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.E,
        model_size_ceiling=ModelSizePolicy.STRUCTURED_SPECIALIST,
        hardware_class="cpu-gpu-optional",
        runtime_requirements=("typed_ir", "isolated_worktree", "structured_decode"),
        capabilities=_STRUCTURED + ("isolated_worktree",),
        validation_contract="validator:patch-sketch@1",
        error_behavior="arbitrary paths, binaries, test deletion, or authority fields fail closed",
        abstention_behavior="out-of-scope files or missing validation identities abstain",
        compact_feature_keys=("allowed_paths", "symbol_ids", "validation_ids"),
        max_input_tokens=4096,
        max_output_tokens=1024,
        abstention_codes=(
            "out_of_distribution",
            "critical_boundary",
            "incomplete_context",
            "capability_unavailable",
        ),
    ),
    ResidualTaskFamily.LEMMA_SUGGESTION: _family(
        ResidualTaskFamily.LEMMA_SUGGESTION,
        input_semantics="exact obligation identity and admissible premise identifiers",
        output_semantics="lemma identifier nominations for the current prover obligation",
        risk_floor=RiskClass.R4,
        risk_ceiling=RiskClass.R5,
        privacy_class=PrivacyClass.PROOF_WITNESS,
        privacy_route_policy="never_remote",
        preferred_expert_class=ExpertClass.D,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-medium-prover",
        runtime_requirements=("typed_ir", "prover"),
        capabilities=_RANK + ("prover",),
        validation_contract="validator:lemma-suggestion@1",
        error_behavior="a suggestion labeled as a proof fails closed",
        abstention_behavior="unknown obligation or missing prover abstains",
        compact_feature_keys=("obligation_id", "premise_ids"),
        max_input_tokens=2048,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.TACTIC_SUGGESTION: _family(
        ResidualTaskFamily.TACTIC_SUGGESTION,
        input_semantics="exact obligation identity and admissible tactic catalog references",
        output_semantics="tactic identifier nominations checked by the actual prover",
        risk_floor=RiskClass.R4,
        risk_ceiling=RiskClass.R5,
        privacy_class=PrivacyClass.PROOF_WITNESS,
        privacy_route_policy="never_remote",
        preferred_expert_class=ExpertClass.D,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-medium-prover",
        runtime_requirements=("typed_ir", "prover"),
        capabilities=_RANK + ("prover",),
        validation_contract="validator:tactic-suggestion@1",
        error_behavior="proof omission or stale obligation fails closed",
        abstention_behavior="unavailable prover or unknown tactic catalog abstains",
        compact_feature_keys=("obligation_id", "tactic_catalog_cid", "premise_ids"),
        max_input_tokens=2048,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION: _family(
        ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION,
        input_semantics="counterexample identity, violated invariant identifiers, and failure class features",
        output_semantics="closed failure class plus counterexample references",
        risk_floor=RiskClass.R3,
        risk_ceiling=RiskClass.R4,
        privacy_class=PrivacyClass.PROOF_WITNESS,
        privacy_route_policy="never_remote",
        preferred_expert_class=ExpertClass.C,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-medium-prover",
        runtime_requirements=("typed_ir", "prover"),
        capabilities=_LINEAR + ("prover",),
        validation_contract="validator:counterexample-explanation@1",
        error_behavior="recoverable private witness bodies fail closed",
        abstention_behavior="incomplete counterexample context abstains",
        compact_feature_keys=("counterexample_id", "violated_invariant_ids"),
        max_input_tokens=2048,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE: _family(
        ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE,
        input_semantics="parent goal identity and closed candidate goal-kind catalog",
        output_semantics="candidate goal kinds that remain proposals",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="authorized_internal",
        preferred_expert_class=ExpertClass.D,
        model_size_ceiling=ModelSizePolicy.SMALL_RANKER,
        hardware_class="cpu-medium",
        runtime_requirements=("typed_ir",),
        capabilities=_RANK,
        validation_contract="validator:goal-refinement@1",
        error_behavior="completion-shaped goal payloads fail closed",
        abstention_behavior="unknown parent goals abstain",
        compact_feature_keys=("parent_goal_id", "goal_kind_catalog_cid"),
        max_input_tokens=1024,
        max_output_tokens=256,
    ),
    ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION: _family(
        ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION,
        input_semantics="documentation claim identity and bounded evidence references",
        output_semantics="one claim class with rewrite-required flag",
        risk_floor=RiskClass.R1,
        risk_ceiling=RiskClass.R2,
        privacy_class=PrivacyClass.INTERNAL,
        privacy_route_policy="authorized_internal",
        preferred_expert_class=ExpertClass.C,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_LINEAR,
        validation_contract="validator:documentation-claim@1",
        error_behavior="unsupported claim classes fail as invalid_output",
        abstention_behavior="claims without evidence references abstain",
        compact_feature_keys=("claim_id", "evidence_reference_ids"),
        max_input_tokens=1024,
        max_output_tokens=128,
    ),
    ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION: _family(
        ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION,
        input_semantics="risk, capability, and validation flags for a residual decision",
        output_semantics="boolean escalate candidate with a closed reason code",
        risk_floor=RiskClass.R2,
        risk_ceiling=RiskClass.R3,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="local_only",
        preferred_expert_class=ExpertClass.B,
        model_size_ceiling=ModelSizePolicy.LINEAR,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=_CLASSIFY,
        validation_contract="validator:human-escalation@1",
        error_behavior="authority or completion claims in escalation payloads fail closed",
        abstention_behavior="unknown reason codes abstain",
        compact_feature_keys=("risk_class", "capability_available", "validation_satisfied"),
        max_input_tokens=512,
        max_output_tokens=64,
    ),
    ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING: _family(
        ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
        input_semantics="any residual question that has no closed family, schema, or validator",
        output_semantics="explicit abstention only; no typed non-abstaining payload",
        risk_floor=RiskClass.R5,
        risk_ceiling=RiskClass.R5,
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        privacy_route_policy="never_remote",
        preferred_expert_class=ExpertClass.A,
        model_size_ceiling=ModelSizePolicy.NONE,
        hardware_class="cpu-small-hermetic",
        runtime_requirements=("typed_ir",),
        capabilities=("typed_candidate", "local_cpu", "abstain", "exact_lookup"),
        validation_contract="validator:novel-unbounded-abstention@1",
        error_behavior="any non-abstaining or prose output fails as invalid_output",
        abstention_behavior="the family always abstains and escalates to human review",
        compact_feature_keys=("reason_code",),
        max_input_tokens=256,
        max_output_tokens=32,
        allowed_expert_classes=(ExpertClass.A,),
        abstention_codes=("novel_unbounded", "critical_boundary", "out_of_distribution"),
    ),
}


def family_spec_for(task_family: ResidualTaskFamily | str) -> ResidualTaskFamilySpec:
    return DEFAULT_FAMILY_SPECS[ResidualTaskFamily(task_family)]


def assert_family_risk_allowed(
    task_family: ResidualTaskFamily | str, risk: RiskClass | str
) -> ResidualTaskFamilySpec:
    spec = family_spec_for(task_family)
    spec.assert_risk_allowed(risk)
    return spec


def all_family_boundaries() -> tuple[ResidualFamilyBoundary, ...]:
    return tuple(spec.to_boundary() for spec in DEFAULT_FAMILY_SPECS.values())


if set(DEFAULT_FAMILY_SPECS) != set(ResidualTaskFamily):
    raise ResidualIntelligenceError("family spec registry is not a bijection over the taxonomy")
if any(
    spec.task_family is not family for family, spec in DEFAULT_FAMILY_SPECS.items()
):
    raise ResidualIntelligenceError("family spec registry keys are misaligned")


__all__ = (
    "AUTHORITY_CLASS",
    "CLOSED_ABSTENTION_CODES",
    "CLOSED_CAPABILITIES",
    "CLOSED_ERROR_CODES",
    "CLOSED_PRIVACY_ROUTES",
    "CLOSED_VALUE_SCHEMA",
    "DEFAULT_FAMILY_SPECS",
    "DEFAULT_SIZE_FOR_CLASS",
    "EXPERT_CLASS_FORMS",
    "MODEL_SIZE_ORDER",
    "RISK_ORDER",
    "SMALLEST_FORM_ORDER",
    "TASK_FAMILY_SPEC_SCHEMA",
    "ClosedValueSchema",
    "ExpertClass",
    "ModelSizePolicy",
    "ResidualTaskFamilySpec",
    "all_family_boundaries",
    "assert_family_risk_allowed",
    "expert_class_rank",
    "family_spec_for",
    "model_size_rank",
    "risk_rank",
    "risks_between",
)
