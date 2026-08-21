"""Exact residual task-family boundaries and closed capability contracts.

Families share input semantics, output semantics, risk, authority, validation,
error, and abstention behavior.  Prompt or embedding similarity is not a
grouping key.  Specifications carry no examples; dataset references must name
an admitted TrainingCorpusAdmission.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from .inventory import ResidualFamilyBoundary
from .residual_ir import (
    MAX_TOKEN_BUDGET,
    RESIDUAL_TASK_INPUT_SCHEMA,
    RESIDUAL_TASK_OUTPUT_SCHEMA,
    ResidualTaskInput,
    ResidualTaskOutput,
)
from .rights import TrainingCorpusAdmission
from .structured_decoding import grammar_for

TASK_FAMILY_SPEC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-task-family-spec@1"
)
CLOSED_SCHEMA_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-closed-schema@1"
CANDIDATE_ONLY_AUTHORITY: Final = "candidate_only"
EFFECT_CLASS: Final = "provider_free_capability_contract"
MAX_FAMILY_INPUT_TOKENS: Final = 30_000
MAX_FAMILY_OUTPUT_TOKENS: Final = 9_000
EXPERT_CLASS_VALUES: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E")
RISK_ORDER: Final[tuple[RiskClass, ...]] = (
    RiskClass.R0,
    RiskClass.R1,
    RiskClass.R2,
    RiskClass.R3,
    RiskClass.R4,
    RiskClass.R5,
)
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})
PRIVACY_ROUTE_POLICIES: Final[frozenset[str]] = frozenset(
    {"local_only", "admitted_remote", "human_review_only"}
)
REMOTE_FORBIDDEN_PRIVACY: Final[frozenset[PrivacyClass]] = frozenset(
    {
        PrivacyClass.TENANT_PRIVATE,
        PrivacyClass.MATTER_CONFIDENTIAL,
        PrivacyClass.CREDENTIAL,
        PrivacyClass.PERSONAL_DATA,
        PrivacyClass.HEALTH_DATA,
        PrivacyClass.LEGAL_PRIVILEGED,
        PrivacyClass.PROOF_WITNESS,
    }
)
ALLOWED_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        "provider-free",
        "cpu-small-hermetic",
        "cpu-standard",
        "cpu-medium-statistical",
        "cpu-medium-local-proof",
        "cpu-gpu-optional-bounded",
        "exact-lookup",
        "declarative-rule",
        "linear-logistic",
        "ranker-encoder",
        "structured-decoder",
    }
)
CONTROL_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "input_valid",
    "critical_boundary",
    "procedure_root",
    "procedure_answer_available",
    "procedure_preconditions_satisfied",
    "capability_available",
    "schema",
    "operation",
    "repository",
    "effects",
    "authority_class",
    "language",
    "framework",
    "context_tier",
    "reference_ids",
)
SIMILARITY_GROUPING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prompt_similarity",
        "embedding_similarity",
        "cosine_similarity",
        "nearest_neighbor",
        "few_shot",
        "example_similarity",
    }
)
FORBIDDEN_SPEC_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "examples",
        "example",
        "few_shot",
        "demonstrations",
        "prompt_similarity",
        "embedding_similarity",
        "chain_of_thought",
        "private_chain_of_thought",
        "hidden_test_body",
        "prose",
        "raw_body",
    }
)

REASON_FAMILY_MISMATCH: Final = "task_family_mismatch"
REASON_RISK_CEILING: Final = "unsupported_family_risk_pair"
REASON_UNKNOWN_INPUT_FIELD: Final = "unknown_input_field"
REASON_TOKEN_LIMIT: Final = "token_limit_exceeded"
REASON_NOVEL_UNBOUNDED: Final = "novel_unbounded_reasoning"
REASON_VALIDATOR_REQUIRED: Final = "validator_required"
REASON_CAPABILITY_UNAVAILABLE: Final = "capability_unavailable"
REASON_PROMPT_SIMILARITY: Final = "prompt_similarity_cannot_override_family_boundary"
REASON_DATASET_UNADMITTED: Final = "dataset_reference_requires_admitted_corpus"
REASON_PROSE_DEFAULT: Final = "prose_is_not_the_default_output"


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def risk_rank(value: RiskClass | str) -> int:
    return RISK_ORDER.index(RiskClass(value))


def risk_at_most(value: RiskClass | str, ceiling: RiskClass | str) -> bool:
    return risk_rank(value) <= risk_rank(ceiling)


def risk_at_least(value: RiskClass | str, floor: RiskClass | str) -> bool:
    return risk_rank(value) >= risk_rank(floor)


def _closed_class_prefix(values: Sequence[str]) -> tuple[str, ...]:
    classes = text_tuple(values, "allowed_expert_classes", allow_empty=False, max_items=5)
    if any(item not in EXPERT_CLASS_VALUES for item in classes):
        raise ResidualIntelligenceError("allowed_expert_classes must be class A through E")
    expected = EXPERT_CLASS_VALUES[: len(classes)]
    if classes != expected:
        raise ResidualIntelligenceError(
            "allowed_expert_classes must follow smallest-form order A through E"
        )
    return classes


def _capability_tuple(values: Sequence[str]) -> tuple[str, ...]:
    capabilities = text_tuple(values, "capabilities", allow_empty=False, max_items=32)
    unknown = sorted(item for item in capabilities if item not in ALLOWED_CAPABILITIES)
    if unknown:
        raise ResidualIntelligenceError(
            "capabilities contain unsupported values: " + ", ".join(unknown)
        )
    if "provider-free" not in capabilities:
        raise ResidualIntelligenceError("family capabilities must include provider-free")
    return capabilities


def _privacy_route(value: Any, privacy: PrivacyClass) -> str:
    policy = required_text(value, "privacy_route_policy", max_bytes=64)
    if policy not in PRIVACY_ROUTE_POLICIES:
        raise ResidualIntelligenceError("unsupported privacy route policy")
    if privacy in REMOTE_FORBIDDEN_PRIVACY and policy == "admitted_remote":
        raise ResidualIntelligenceError("privacy class forbids an admitted_remote route")
    return policy


def reject_similarity_grouping(grouping_key: str) -> None:
    """Exact family equality is required; prompt similarity cannot group experts."""

    key = required_text(grouping_key, "grouping_key", max_bytes=128).casefold().replace("-", "_")
    if key in SIMILARITY_GROUPING_KEYS or "similarity" in key:
        raise ResidualIntelligenceError(REASON_PROMPT_SIMILARITY)


def reject_spec_examples(payload: Mapping[str, Any], *, noun: str) -> None:
    """Specifications carry no examples or recoverable private bodies."""

    if not isinstance(payload, Mapping):
        raise ResidualIntelligenceError(f"{noun} must be an object")
    forbidden = sorted(
        str(key)
        for key in payload
        if str(key).strip().casefold().replace("-", "_") in FORBIDDEN_SPEC_PAYLOAD_KEYS
    )
    if forbidden:
        raise UnknownFieldError(
            f"{noun} carries example or similarity fields: {', '.join(forbidden)}"
        )


@dataclass(frozen=True)
class ClosedSchema:
    """Named closed field set; unknown keys are never recovered as prose."""

    schema_id: str
    fields: tuple[str, ...]
    required_fields: tuple[str, ...] = ()
    schema: str = CLOSED_SCHEMA_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "schema_id", "fields", "required_fields"}
    )

    def __post_init__(self) -> None:
        if self.schema != CLOSED_SCHEMA_SCHEMA:
            raise ResidualIntelligenceError("unsupported closed schema contract")
        object.__setattr__(
            self, "schema_id", required_text(self.schema_id, "schema_id", max_bytes=256)
        )
        object.__setattr__(
            self, "fields", text_tuple(self.fields, "fields", allow_empty=False, max_items=256)
        )
        object.__setattr__(
            self,
            "required_fields",
            text_tuple(self.required_fields, "required_fields", max_items=256),
        )
        if not set(self.required_fields).issubset(self.fields):
            raise ResidualIntelligenceError("required schema fields are not declared")

    def reject_unknown(self, payload: Mapping[str, Any], *, noun: str) -> None:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError(f"{noun} must be an object")
        unknown = sorted(str(key) for key in payload if str(key) not in self.fields)
        if unknown:
            raise UnknownFieldError(f"{noun} contains unknown fields: {', '.join(unknown)}")
        missing = sorted(field for field in self.required_fields if field not in payload)
        if missing:
            raise ResidualIntelligenceError(
                f"{noun} is missing required fields: {', '.join(missing)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_id": self.schema_id,
            "fields": list(self.fields),
            "required_fields": list(self.required_fields),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClosedSchema:
        reject_spec_examples(payload, noun="closed schema")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS,
            noun="closed schema",
        )
        return cls(
            schema=str(payload.get("schema") or ""),
            schema_id=str(payload.get("schema_id") or ""),
            fields=tuple(payload.get("fields") or ()),
            required_fields=tuple(payload.get("required_fields") or ()),
        )


@dataclass(frozen=True)
class ResidualTaskFamilySpec:
    """Shared semantic boundary, schemas, limits, and gates for one family."""

    task_family: ResidualTaskFamily
    input_semantics: str
    output_semantics: str
    input_schema: ClosedSchema
    output_schema: ClosedSchema
    risk_floor: RiskClass
    risk_ceiling: RiskClass
    privacy_class: PrivacyClass
    privacy_route_policy: str
    authority_class: str
    validation_contract: str
    error_behavior: str
    abstention_behavior: str
    capabilities: tuple[str, ...]
    allowed_expert_classes: tuple[str, ...]
    input_token_limit: int
    output_token_limit: int
    maximum_output_bytes: int
    validator_required: bool = True
    emit_prose_by_default: bool = False
    prose_token_budget: int = 0
    evaluation_corpus_admission_id: str = ""
    training_corpus_admission_id: str = ""
    schema: str = TASK_FAMILY_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "family_spec_id",
            "task_family",
            "input_semantics",
            "output_semantics",
            "input_schema",
            "output_schema",
            "risk_floor",
            "risk_ceiling",
            "privacy_class",
            "privacy_route_policy",
            "authority_class",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "capabilities",
            "allowed_expert_classes",
            "input_token_limit",
            "output_token_limit",
            "maximum_output_bytes",
            "validator_required",
            "emit_prose_by_default",
            "prose_token_budget",
            "evaluation_corpus_admission_id",
            "training_corpus_admission_id",
            "boundary_id",
            "effect_class",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != TASK_FAMILY_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual task family spec schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        for field in (
            "input_semantics",
            "output_semantics",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if not isinstance(self.input_schema, ClosedSchema) or not isinstance(
            self.output_schema, ClosedSchema
        ):
            raise ResidualIntelligenceError("family specs require typed closed schemas")
        object.__setattr__(self, "risk_floor", RiskClass(self.risk_floor))
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        if risk_rank(self.risk_floor) > risk_rank(self.risk_ceiling):
            raise ResidualIntelligenceError("family risk floor exceeds risk ceiling")
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        object.__setattr__(
            self,
            "privacy_route_policy",
            _privacy_route(self.privacy_route_policy, self.privacy_class),
        )
        object.__setattr__(
            self,
            "authority_class",
            required_text(self.authority_class, "authority_class", max_bytes=64),
        )
        if self.authority_class.casefold() != CANDIDATE_ONLY_AUTHORITY:
            raise ResidualIntelligenceError(
                "residual family authority_class must be candidate_only"
            )
        object.__setattr__(self, "capabilities", _capability_tuple(self.capabilities))
        object.__setattr__(
            self, "allowed_expert_classes", _closed_class_prefix(self.allowed_expert_classes)
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
        grammar = grammar_for(self.task_family)
        object.__setattr__(
            self,
            "maximum_output_bytes",
            bounded_int(
                self.maximum_output_bytes,
                "maximum_output_bytes",
                minimum=128,
                maximum=MAX_TOKEN_BUDGET,
            ),
        )
        if self.maximum_output_bytes != grammar.maximum_output_bytes:
            raise ResidualIntelligenceError(
                "family maximum_output_bytes must match the family grammar"
            )
        if set(self.output_schema.fields) != set(grammar.payload_fields):
            raise ResidualIntelligenceError(
                "family output schema fields must match the family grammar"
            )
        if not set(self.output_schema.required_fields).issubset(grammar.required_payload_fields):
            raise ResidualIntelligenceError(
                "family output required fields must be grammar-required"
            )
        object.__setattr__(
            self, "validator_required", _require_bool(self.validator_required, "validator_required")
        )
        if self.validator_required is not True:
            raise ResidualIntelligenceError("every family spec requires an independent validator")
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
            self,
            "evaluation_corpus_admission_id",
            ""
            if self.evaluation_corpus_admission_id in (None, "")
            else required_text(
                self.evaluation_corpus_admission_id, "evaluation_corpus_admission_id"
            ),
        )
        object.__setattr__(
            self,
            "training_corpus_admission_id",
            ""
            if self.training_corpus_admission_id in (None, "")
            else required_text(self.training_corpus_admission_id, "training_corpus_admission_id"),
        )
        if self.task_family is ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING:
            if self.allowed_expert_classes != ("A",):
                raise ResidualIntelligenceError(
                    "NOVEL_UNBOUNDED_REASONING may only declare class A abstention"
                )
            if self.privacy_route_policy != "human_review_only":
                raise ResidualIntelligenceError(
                    "NOVEL_UNBOUNDED_REASONING requires human_review_only routing"
                )

    @property
    def family_spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def effect_class(self) -> str:
        return EFFECT_CLASS

    @property
    def proposal_tier(self) -> bool:
        return self.risk_ceiling in PROPOSAL_RISKS or self.risk_floor in PROPOSAL_RISKS

    @property
    def always_abstain(self) -> bool:
        return self.task_family is ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING

    def as_boundary(self) -> ResidualFamilyBoundary:
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

    @property
    def boundary_id(self) -> str:
        return self.as_boundary().boundary_id

    def grammar(self) -> Any:
        return grammar_for(self.task_family)

    def admits_risk(self, risk: RiskClass | str) -> bool:
        value = RiskClass(risk)
        return risk_at_least(value, self.risk_floor) and risk_at_most(value, self.risk_ceiling)

    def bind_corpus_admission(self, admission: TrainingCorpusAdmission) -> None:
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError(
                "dataset reference must be a typed TrainingCorpusAdmission"
            )
        referenced = {
            self.evaluation_corpus_admission_id,
            self.training_corpus_admission_id,
        } - {""}
        if not referenced:
            raise ResidualIntelligenceError("family spec carries no dataset reference")
        if admission.admission_id not in referenced:
            raise ResidualIntelligenceError("dataset reference does not match admission identity")
        if admission.admission_decision.value != "admitted":
            raise ResidualIntelligenceError(REASON_DATASET_UNADMITTED)
        if self.training_corpus_admission_id and not admission.can_train:
            raise ResidualIntelligenceError(REASON_DATASET_UNADMITTED)

    def validate_task_input(self, task_input: ResidualTaskInput) -> tuple[str, ...]:
        """Return reason codes for hard family/schema/risk/limit rejections."""

        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
        reasons: list[str] = []
        if task_input.task_family is not self.task_family:
            reasons.append(REASON_FAMILY_MISMATCH)
        if not self.admits_risk(task_input.risk_class):
            reasons.append(REASON_RISK_CEILING)
        try:
            self.input_schema.reject_unknown(task_input.compact_features, noun="compact_features")
        except ResidualIntelligenceError:
            reasons.append(REASON_UNKNOWN_INPUT_FIELD)
        if task_input.token_budget > self.input_token_limit:
            reasons.append(REASON_TOKEN_LIMIT)
        if self.always_abstain:
            reasons.append(REASON_NOVEL_UNBOUNDED)
        return tuple(dict.fromkeys(reasons))

    def validate_task_output(self, task_output: ResidualTaskOutput) -> None:
        if not isinstance(task_output, ResidualTaskOutput):
            raise ResidualIntelligenceError("task_output must be ResidualTaskOutput")
        grammar = self.grammar()
        if task_output.candidate_only is not True:
            raise ResidualIntelligenceError("family outputs must remain candidate_only=true")
        if self.always_abstain and not task_output.abstained:
            raise ResidualIntelligenceError(REASON_NOVEL_UNBOUNDED)
        if task_output.output_class not in grammar.output_classes:
            raise ResidualIntelligenceError("output_class is outside the family grammar")
        if any(
            key in task_output.structured_payload for key in ("prose", "explanation", "narrative")
        ):
            raise ResidualIntelligenceError(REASON_PROSE_DEFAULT)
        if not task_output.abstained:
            self.output_schema.reject_unknown(
                task_output.structured_payload, noun="structured_payload"
            )
            if not task_output.evidence_references:
                raise ResidualIntelligenceError(REASON_VALIDATOR_REQUIRED)
        if self.proposal_tier and not (
            task_output.abstained or "VALIDATION_REQUIRED" in task_output.reason_codes
        ):
            raise ResidualIntelligenceError(
                "R4/R5 family outputs must abstain or remain validation-required"
            )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "risk_floor": self.risk_floor.value,
            "risk_ceiling": self.risk_ceiling.value,
            "privacy_class": self.privacy_class.value,
            "privacy_route_policy": self.privacy_route_policy,
            "authority_class": self.authority_class,
            "validation_contract": self.validation_contract,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "capabilities": list(self.capabilities),
            "allowed_expert_classes": list(self.allowed_expert_classes),
            "input_token_limit": self.input_token_limit,
            "output_token_limit": self.output_token_limit,
            "maximum_output_bytes": self.maximum_output_bytes,
            "validator_required": True,
            "emit_prose_by_default": False,
            "prose_token_budget": 0,
            "evaluation_corpus_admission_id": self.evaluation_corpus_admission_id,
            "training_corpus_admission_id": self.training_corpus_admission_id,
            "boundary_id": self.boundary_id,
            "effect_class": self.effect_class,
        }
        if include_id:
            result["family_spec_id"] = self.family_spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualTaskFamilySpec:
        reject_spec_examples(payload, noun="task family spec")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"family_spec_id", "boundary_id", "effect_class"},
            noun="task family spec",
        )
        input_schema_payload = payload.get("input_schema")
        output_schema_payload = payload.get("output_schema")
        if not isinstance(input_schema_payload, Mapping) or not isinstance(
            output_schema_payload, Mapping
        ):
            raise ResidualIntelligenceError("family spec schemas must be objects")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            input_semantics=str(payload.get("input_semantics") or ""),
            output_semantics=str(payload.get("output_semantics") or ""),
            input_schema=ClosedSchema.from_dict(input_schema_payload),
            output_schema=ClosedSchema.from_dict(output_schema_payload),
            risk_floor=RiskClass(str(payload.get("risk_floor") or "")),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            privacy_route_policy=str(payload.get("privacy_route_policy") or ""),
            authority_class=str(payload.get("authority_class") or ""),
            validation_contract=str(payload.get("validation_contract") or ""),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            capabilities=tuple(payload.get("capabilities") or ()),
            allowed_expert_classes=tuple(payload.get("allowed_expert_classes") or ()),
            input_token_limit=payload.get("input_token_limit"),
            output_token_limit=payload.get("output_token_limit"),
            maximum_output_bytes=payload.get("maximum_output_bytes"),
            validator_required=payload.get("validator_required"),
            emit_prose_by_default=payload.get("emit_prose_by_default"),
            prose_token_budget=payload.get("prose_token_budget"),
            evaluation_corpus_admission_id=str(
                payload.get("evaluation_corpus_admission_id") or ""
            ),
            training_corpus_admission_id=str(payload.get("training_corpus_admission_id") or ""),
        )
        claimed = str(payload.get("family_spec_id") or "")
        if claimed and claimed != result.family_spec_id:
            raise ResidualIntelligenceError("task family spec identity mismatch")
        claimed_boundary = str(payload.get("boundary_id") or "")
        if claimed_boundary and claimed_boundary != result.boundary_id:
            raise ResidualIntelligenceError("task family spec boundary identity mismatch")
        return result


def _fields(*extra: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*CONTROL_INPUT_FIELDS, *extra)))


def _closed_input(family: ResidualTaskFamily, fields: tuple[str, ...]) -> ClosedSchema:
    return ClosedSchema(
        schema_id=f"ipfs_accelerate_py/agent-supervisor/residual-family-input/{family.value}@1",
        fields=fields,
    )


def _closed_output(family: ResidualTaskFamily) -> ClosedSchema:
    grammar = grammar_for(family)
    fields = grammar.payload_fields or ("reason_code",)
    return ClosedSchema(
        schema_id=f"ipfs_accelerate_py/agent-supervisor/residual-family-output/{family.value}@1",
        fields=fields,
        required_fields=grammar.required_payload_fields,
    )


def _validator(family: ResidualTaskFamily) -> str:
    return family.value.lower().replace("_", "-") + "-validator@1"


def _spec(
    family: ResidualTaskFamily,
    *,
    input_semantics: str,
    output_semantics: str,
    risk_ceiling: RiskClass,
    error_behavior: str,
    abstention_behavior: str,
    extra_input_fields: tuple[str, ...] = (),
    risk_floor: RiskClass | None = None,
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL,
    privacy_route_policy: str = "local_only",
    capabilities: tuple[str, ...] = ("provider-free", "cpu-small-hermetic", "exact-lookup"),
    allowed_expert_classes: tuple[str, ...] = ("A", "B", "C"),
    input_token_limit: int = 4_096,
    output_token_limit: int = 256,
) -> ResidualTaskFamilySpec:
    grammar = grammar_for(family)
    return ResidualTaskFamilySpec(
        task_family=family,
        input_semantics=input_semantics,
        output_semantics=output_semantics,
        input_schema=_closed_input(family, _fields(*extra_input_fields)),
        output_schema=_closed_output(family),
        risk_floor=risk_ceiling if risk_floor is None else risk_floor,
        risk_ceiling=risk_ceiling,
        privacy_class=privacy_class,
        privacy_route_policy=privacy_route_policy,
        authority_class=CANDIDATE_ONLY_AUTHORITY,
        validation_contract=_validator(family),
        error_behavior=error_behavior,
        abstention_behavior=abstention_behavior,
        capabilities=capabilities,
        allowed_expert_classes=allowed_expert_classes,
        input_token_limit=input_token_limit,
        output_token_limit=output_token_limit,
        maximum_output_bytes=grammar.maximum_output_bytes,
    )


def _build_family_specs() -> dict[ResidualTaskFamily, ResidualTaskFamilySpec]:
    classification_caps = ("provider-free", "cpu-small-hermetic", "exact-lookup", "linear-logistic")
    ranking_caps = (
        "provider-free",
        "cpu-medium-statistical",
        "exact-lookup",
        "ranker-encoder",
    )
    structured_caps = (
        "provider-free",
        "cpu-medium-local-proof",
        "exact-lookup",
        "structured-decoder",
    )
    specs = (
        _spec(
            ResidualTaskFamily.TASK_CLASSIFICATION,
            input_semantics="bounded objective, task, and compact operation features",
            output_semantics="one closed task-family label and supporting reference ids",
            risk_ceiling=RiskClass.R1,
            risk_floor=RiskClass.R0,
            extra_input_fields=("label_candidates", "objective_kind"),
            error_behavior="unknown operation class is invalid_output and does not route",
            abstention_behavior="ambiguous or multi-family signatures abstain",
            capabilities=classification_caps,
            allowed_expert_classes=("A", "B", "C"),
        ),
        _spec(
            ResidualTaskFamily.RISK_CLASSIFICATION,
            input_semantics="bounded effect, authority, and validation-demand features",
            output_semantics="one closed risk label in R0 through R5",
            risk_ceiling=RiskClass.R1,
            risk_floor=RiskClass.R0,
            extra_input_fields=("effect_classes", "authority_demand"),
            error_behavior="out-of-enum risk labels are invalid_output",
            abstention_behavior="missing effect or authority features abstain",
            capabilities=classification_caps,
        ),
        _spec(
            ResidualTaskFamily.EFFECT_CLASSIFICATION,
            input_semantics="declared operator, path, and mutation-surface features",
            output_semantics="one or more closed effect classes with references",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R0,
            extra_input_fields=("operator_ids", "path_ids"),
            error_behavior="undeclared effect classes are invalid_output",
            abstention_behavior="unseen operators or empty surfaces abstain",
            capabilities=classification_caps,
        ),
        _spec(
            ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION,
            input_semantics="requested mutation class plus current authority surface",
            output_semantics="one closed authority-requirement label",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R0,
            extra_input_fields=("requested_effect", "current_authority"),
            error_behavior="authority-shaped candidate fields are rejected",
            abstention_behavior="unknown authority surfaces abstain rather than grant",
            capabilities=classification_caps,
        ),
        _spec(
            ResidualTaskFamily.CONTEXT_SUFFICIENCY,
            input_semantics="compiled context capsule identities and missing-reference hints",
            output_semantics="boolean sufficiency plus missing reference ids",
            risk_ceiling=RiskClass.R1,
            risk_floor=RiskClass.R0,
            extra_input_fields=("missing_reference_ids", "context_capsule_bytes"),
            error_behavior="oversized or secret-bearing context features are rejected",
            abstention_behavior="incomplete capsules abstain as insufficient",
            capabilities=classification_caps,
        ),
        _spec(
            ResidualTaskFamily.EVIDENCE_RANKING,
            input_semantics="bounded evidence identities with comparable ranking signals",
            output_semantics="descending evidence ranking with aligned scores",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R0,
            extra_input_fields=("ranking_candidates", "ranking_signals"),
            error_behavior="misaligned or non-descending scores are invalid_output",
            abstention_behavior="empty or incomparable evidence sets abstain",
            capabilities=ranking_caps,
            allowed_expert_classes=("A", "B", "C", "D"),
            input_token_limit=8_192,
            output_token_limit=512,
        ),
        _spec(
            ResidualTaskFamily.PROCEDURE_MATCHING,
            input_semantics="procedure identity, preconditions, and hole features",
            output_semantics="one procedure id and a closed match class",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R1,
            extra_input_fields=("procedure_id", "precondition_reference_ids", "match_class"),
            error_behavior="unsatisfied preconditions cannot emit a match",
            abstention_behavior="unbound or stale procedure roots abstain",
            capabilities=classification_caps + ("declarative-rule",),
            allowed_expert_classes=("A", "B", "C"),
        ),
        _spec(
            ResidualTaskFamily.PLAN_BRANCH_RANKING,
            input_semantics="plan-branch identities with bounded heuristic signals",
            output_semantics="descending branch ranking with aligned scores",
            risk_ceiling=RiskClass.R3,
            risk_floor=RiskClass.R1,
            extra_input_fields=("ranking_candidates", "ranking_signals", "obligation_ids"),
            error_behavior="unknown branches or inverted scores are invalid_output",
            abstention_behavior="missing obligations or empty branch sets abstain",
            capabilities=ranking_caps,
            allowed_expert_classes=("A", "B", "C", "D"),
            input_token_limit=8_192,
            output_token_limit=512,
        ),
        _spec(
            ResidualTaskFamily.TEST_SELECTION,
            input_semantics="changed symbols, coverage identities, and test catalog ids",
            output_semantics="closed selected test ids with coverage references",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R0,
            extra_input_fields=("test_ids", "coverage_reference_ids", "changed_symbol_ids"),
            error_behavior="test deletion or weakening selections are invalid_output",
            abstention_behavior="unknown catalogs or empty coverage abstain",
            capabilities=classification_caps + ("declarative-rule",),
            allowed_expert_classes=("A", "B", "C", "D"),
        ),
        _spec(
            ResidualTaskFamily.PROOF_SELECTION,
            input_semantics="current prover obligation, environment, and proof identities",
            output_semantics="closed selected proof ids bound to the obligation",
            risk_ceiling=RiskClass.R3,
            risk_floor=RiskClass.R1,
            extra_input_fields=("proof_ids", "obligation_reference_ids", "prover_environment"),
            error_behavior="proof omission or environment mismatch is invalid_output",
            abstention_behavior="missing prover capability or obligation abstains",
            capabilities=ranking_caps + ("cpu-medium-local-proof",),
            allowed_expert_classes=("A", "B", "C", "D"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        ),
        _spec(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            input_semantics="validated failure signature plus bounded dependency references",
            output_semantics="one failure class and one bounded action candidate",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R0,
            extra_input_fields=("exit_code", "failure_signature", "recommended_action"),
            error_behavior="invalid output or failed validation escalates",
            abstention_behavior="unknown signatures abstain",
            capabilities=classification_caps + ("declarative-rule",),
        ),
        _spec(
            ResidualTaskFamily.RETRY_OR_ESCALATE,
            input_semantics="attempt counts, failure class, and remaining budget features",
            output_semantics="closed retry, escalate, or stop decision",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R0,
            extra_input_fields=("attempt_count", "remaining_budget", "failure_class"),
            error_behavior="open-ended retry loops are invalid_output",
            abstention_behavior="exhausted or unknown budgets abstain to escalate",
            capabilities=classification_caps + ("declarative-rule",),
        ),
        _spec(
            ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION,
            input_semantics="cache identity, dependency pins, and staleness features",
            output_semantics="boolean reuse decision with dependency references",
            risk_ceiling=RiskClass.R1,
            risk_floor=RiskClass.R0,
            extra_input_fields=("cache_identity", "dependency_reference_ids", "stale"),
            error_behavior="stale or secret-bearing cache hits are rejected",
            abstention_behavior="unknown dependency pins abstain from reuse",
            capabilities=classification_caps,
        ),
        _spec(
            ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION,
            input_semantics="conflicting symbol identities and bounded merge hunks",
            output_semantics="one conflict class with involved symbol ids",
            risk_ceiling=RiskClass.R2,
            risk_floor=RiskClass.R1,
            extra_input_fields=("symbol_ids", "conflict_class", "hunk_ids"),
            error_behavior="authority or trusted-key conflicts cannot auto-resolve",
            abstention_behavior="overlapping semantic edits abstain to human merge",
            capabilities=classification_caps + ("declarative-rule",),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        ),
        _spec(
            ResidualTaskFamily.PATCH_TEMPLATE_SELECTION,
            input_semantics="failure class, symbol ids, and admitted template catalog",
            output_semantics="one template id with targeted symbol references",
            risk_ceiling=RiskClass.R3,
            risk_floor=RiskClass.R1,
            extra_input_fields=("template_id", "symbol_ids", "failure_class"),
            error_behavior="templates that delete tests or weaken validation are rejected",
            abstention_behavior="catalog misses or multi-template ties abstain",
            capabilities=classification_caps + ("declarative-rule",),
            allowed_expert_classes=("A", "B", "C", "D"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        ),
        _spec(
            ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
            input_semantics="typed hole id, operator catalog, and compiler preconditions",
            output_semantics="bounded ProcedureHoleResolution candidate",
            risk_ceiling=RiskClass.R4,
            risk_floor=RiskClass.R3,
            extra_input_fields=("hole_id", "operator_id", "argument_reference_ids"),
            error_behavior="undeclared holes or failed preconditions are invalid_output",
            abstention_behavior="missing compiler capability or unbound holes abstain",
            capabilities=structured_caps,
            allowed_expert_classes=("A", "B", "C", "D", "E"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            input_token_limit=8_192,
            output_token_limit=1_024,
        ),
        _spec(
            ResidualTaskFamily.PATCH_SKETCH_GENERATION,
            input_semantics="exact paths, symbols, line bounds, and validation identities",
            output_semantics="bounded PatchSketchIR candidate with no free-form shell",
            risk_ceiling=RiskClass.R4,
            risk_floor=RiskClass.R3,
            extra_input_fields=("files", "symbol_ids", "operations", "maximum_changed_lines"),
            error_behavior="arbitrary paths, test deletion, or authority edits are invalid_output",
            abstention_behavior="unbounded or unvalidated sketches abstain",
            capabilities=structured_caps + ("cpu-gpu-optional-bounded",),
            allowed_expert_classes=("A", "B", "C", "D", "E"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            input_token_limit=16_384,
            output_token_limit=2_048,
        ),
        _spec(
            ResidualTaskFamily.LEMMA_SUGGESTION,
            input_semantics="current obligation, premises, and lemma catalog identities",
            output_semantics="ranked lemma ids that remain prover-checked candidates",
            risk_ceiling=RiskClass.R4,
            risk_floor=RiskClass.R2,
            extra_input_fields=("lemma_ids", "premise_ids", "obligation_id"),
            error_behavior="invented obligations or proof-acceptance fields are rejected",
            abstention_behavior="empty catalogs or prover unavailability abstain",
            capabilities=structured_caps + ("ranker-encoder",),
            allowed_expert_classes=("A", "B", "C", "D", "E"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            input_token_limit=8_192,
            output_token_limit=512,
        ),
        _spec(
            ResidualTaskFamily.TACTIC_SUGGESTION,
            input_semantics="goal, premises, and allowed tactic identities",
            output_semantics="ranked tactic ids that never accept the proof",
            risk_ceiling=RiskClass.R4,
            risk_floor=RiskClass.R2,
            extra_input_fields=("tactic_ids", "premise_ids", "obligation_id"),
            error_behavior="tactic output cannot carry proof_accepted or completion",
            abstention_behavior="unknown goals or missing prover environment abstain",
            capabilities=structured_caps + ("ranker-encoder",),
            allowed_expert_classes=("A", "B", "C", "D", "E"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            input_token_limit=8_192,
            output_token_limit=512,
        ),
        _spec(
            ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION,
            input_semantics="failed invariant identities and compact counterexample refs",
            output_semantics="failure class plus bounded counterexample references",
            risk_ceiling=RiskClass.R3,
            risk_floor=RiskClass.R1,
            extra_input_fields=(
                "failure_class",
                "violated_invariant_ids",
                "counterexample_reference_ids",
            ),
            error_behavior="prose-only explanations without references are invalid_output",
            abstention_behavior="missing counterexample identities abstain",
            capabilities=classification_caps + ("structured-decoder",),
            allowed_expert_classes=("A", "B", "C", "D", "E"),
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        ),
        _spec(
            ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE,
            input_semantics="parent goal identity and closed candidate goal kinds",
            output_semantics="candidate goal kinds that never mark the parent complete",
            risk_ceiling=RiskClass.R3,
            risk_floor=RiskClass.R1,
            extra_input_fields=("parent_goal_id", "candidate_goal_kinds", "acceptance_reference_ids"),
            error_behavior="completion-shaped refinements are rejected",
            abstention_behavior="unknown parent goals or empty kind catalogs abstain",
            capabilities=classification_caps + ("structured-decoder",),
            allowed_expert_classes=("A", "B", "C", "D", "E"),
        ),
        _spec(
            ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION,
            input_semantics="documentation claim identity and supporting evidence refs",
            output_semantics="closed claim class plus rewrite-required flag",
            risk_ceiling=RiskClass.R1,
            risk_floor=RiskClass.R0,
            extra_input_fields=("claim_class", "evidence_reference_ids", "rewrite_required"),
            error_behavior="unsupported documentation authority claims are rejected",
            abstention_behavior="claims without evidence references abstain",
            capabilities=classification_caps,
        ),
        _spec(
            ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION,
            input_semantics="risk, disagreement, and review-budget features",
            output_semantics="boolean escalate decision with a closed reason code",
            risk_ceiling=RiskClass.R3,
            risk_floor=RiskClass.R1,
            extra_input_fields=("escalate", "reason_code", "disagreement"),
            error_behavior="model self-approval cannot clear escalation",
            abstention_behavior="missing review capability abstains as escalate",
            capabilities=classification_caps + ("declarative-rule",),
            privacy_route_policy="human_review_only",
        ),
        _spec(
            ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
            input_semantics="any request outside the closed 23-family residual taxonomy",
            output_semantics="mandatory abstention with a typed reason code",
            risk_ceiling=RiskClass.R5,
            risk_floor=RiskClass.R5,
            extra_input_fields=("reason_code",),
            error_behavior="any non-abstaining or prose payload is invalid_output",
            abstention_behavior="always abstain and escalate to human review",
            capabilities=("provider-free", "cpu-small-hermetic", "exact-lookup"),
            allowed_expert_classes=("A",),
            privacy_route_policy="human_review_only",
            input_token_limit=1_024,
            output_token_limit=128,
        ),
    )
    mapping = {item.task_family: item for item in specs}
    if set(mapping) != set(ResidualTaskFamily):
        missing = sorted(item.value for item in ResidualTaskFamily if item not in mapping)
        raise ResidualIntelligenceError(
            "family spec registry is missing taxonomies: " + ", ".join(missing)
        )
    return mapping


FAMILY_SPECS: Final[Mapping[ResidualTaskFamily, ResidualTaskFamilySpec]] = _build_family_specs()


def family_spec_for(task_family: ResidualTaskFamily | str) -> ResidualTaskFamilySpec:
    return FAMILY_SPECS[ResidualTaskFamily(task_family)]


def family_boundary_for(task_family: ResidualTaskFamily | str) -> ResidualFamilyBoundary:
    return family_spec_for(task_family).as_boundary()


def assert_exact_family_boundary(
    left: ResidualTaskFamilySpec | ResidualFamilyBoundary,
    right: ResidualTaskFamilySpec | ResidualFamilyBoundary,
) -> None:
    left_boundary = left.as_boundary() if isinstance(left, ResidualTaskFamilySpec) else left
    right_boundary = right.as_boundary() if isinstance(right, ResidualTaskFamilySpec) else right
    if not isinstance(left_boundary, ResidualFamilyBoundary) or not isinstance(
        right_boundary, ResidualFamilyBoundary
    ):
        raise ResidualIntelligenceError("exact family boundary requires typed boundary records")
    if left_boundary.boundary_id != right_boundary.boundary_id:
        raise ResidualIntelligenceError(
            "experts cannot share a family without an exact semantic boundary"
        )


def assert_family_risk_admitted(
    task_family: ResidualTaskFamily | str, risk: RiskClass | str
) -> None:
    spec = family_spec_for(task_family)
    if not spec.admits_risk(risk):
        raise ResidualIntelligenceError(REASON_RISK_CEILING)


__all__ = (
    "ALLOWED_CAPABILITIES",
    "CANDIDATE_ONLY_AUTHORITY",
    "CLOSED_SCHEMA_SCHEMA",
    "CONTROL_INPUT_FIELDS",
    "EFFECT_CLASS",
    "EXPERT_CLASS_VALUES",
    "FAMILY_SPECS",
    "MAX_FAMILY_INPUT_TOKENS",
    "MAX_FAMILY_OUTPUT_TOKENS",
    "PROPOSAL_RISKS",
    "REASON_CAPABILITY_UNAVAILABLE",
    "REASON_DATASET_UNADMITTED",
    "REASON_FAMILY_MISMATCH",
    "REASON_NOVEL_UNBOUNDED",
    "REASON_PROMPT_SIMILARITY",
    "REASON_PROSE_DEFAULT",
    "REASON_RISK_CEILING",
    "REASON_TOKEN_LIMIT",
    "REASON_UNKNOWN_INPUT_FIELD",
    "REASON_VALIDATOR_REQUIRED",
    "RESIDUAL_TASK_INPUT_SCHEMA",
    "RESIDUAL_TASK_OUTPUT_SCHEMA",
    "RISK_ORDER",
    "TASK_FAMILY_SPEC_SCHEMA",
    "ClosedSchema",
    "ResidualTaskFamilySpec",
    "assert_exact_family_boundary",
    "assert_family_risk_admitted",
    "family_boundary_for",
    "family_spec_for",
    "reject_similarity_grouping",
    "reject_spec_examples",
    "risk_at_least",
    "risk_at_most",
    "risk_rank",
)
