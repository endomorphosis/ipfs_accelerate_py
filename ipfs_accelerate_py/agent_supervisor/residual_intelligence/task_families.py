"""Closed semantic boundaries for every residual task family.

A family is defined by shared input semantics, output semantics, risk,
authority, validation, error, and abstention behavior.  Prompt or embedding
similarity cannot merge families.  Specifications carry no examples.
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
from .residual_ir import ResidualTaskInput
from .rights import TrainingCorpusAdmission
from .structured_decoding import grammar_for

TASK_FAMILY_SPEC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-task-family-spec@1"
)
FAMILY_SPEC_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-task-family-spec-registry@1"
)
FAMILY_SPEC_REGISTRY_VERSION: Final = "vrif-010-task-family-specs@1"
CANDIDATE_ONLY_AUTHORITY: Final = "candidate_only"
ABSTAIN_OUTPUT_CLASS: Final = "ABSTAIN"
ERROR_INVALID_OUTPUT: Final = "invalid_output"
REASON_TASK_FAMILY_MISMATCH: Final = "task_family_mismatch"
REASON_UNSUPPORTED_FAMILY_RISK: Final = "unsupported_family_risk"
REASON_RISK_CEILING: Final = "risk_ceiling_exceeded"
REASON_UNKNOWN_COMPACT_FEATURE: Final = "unknown_compact_feature"
REASON_MISSING_COMPACT_FEATURE: Final = "missing_compact_feature"
REASON_TOKEN_LIMIT: Final = "family_token_limit_exceeded"
REASON_OUTPUT_CLASS: Final = "output_class_outside_family_grammar"
REASON_VALIDATOR_REQUIRED: Final = "validator_required"
REASON_PROSE_DEFAULT: Final = "prose_default_forbidden"
REASON_TRAINING_UNAVAILABLE: Final = "training_unavailable"
REASON_EXAMPLES_FORBIDDEN: Final = "specifications_carry_no_examples"
MAX_FAMILY_INPUT_TOKENS: Final = 8_192
MAX_FAMILY_OUTPUT_TOKENS: Final = 2_048
CLOSED_EXPERT_CLASS_LETTERS: Final[frozenset[str]] = frozenset({"A", "B", "C", "D", "E"})
RISK_ORDER: Final[tuple[RiskClass, ...]] = (
    RiskClass.R0,
    RiskClass.R1,
    RiskClass.R2,
    RiskClass.R3,
    RiskClass.R4,
    RiskClass.R5,
)
PRIVACY_ROUTE_LOCAL_ONLY: Final = "local_only"
PRIVACY_ROUTE_AUTHORIZED_PROVIDER: Final = "authorized_provider"
PRIVACY_ROUTE_PUBLIC_OR_LOCAL: Final = "public_or_local"
CLOSED_PRIVACY_ROUTES: Final[frozenset[str]] = frozenset(
    {
        PRIVACY_ROUTE_LOCAL_ONLY,
        PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
        PRIVACY_ROUTE_PUBLIC_OR_LOCAL,
    }
)
LOCAL_ONLY_PRIVACY: Final[frozenset[PrivacyClass]] = frozenset(
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
SHARED_OPTIONAL_FEATURES: Final[tuple[str, ...]] = (
    "operation",
    "schema_name",
    "repository_family",
    "language_framework",
    "context_complete",
)
SEMANTIC_KIND_CLASSIFICATION: Final = "classification"
SEMANTIC_KIND_RANKING: Final = "ranking"
SEMANTIC_KIND_MATCHING: Final = "matching"
SEMANTIC_KIND_STRUCTURED: Final = "structured"
SEMANTIC_KIND_UNBOUNDED: Final = "unbounded"
CLOSED_SEMANTIC_KINDS: Final[frozenset[str]] = frozenset(
    {
        SEMANTIC_KIND_CLASSIFICATION,
        SEMANTIC_KIND_RANKING,
        SEMANTIC_KIND_MATCHING,
        SEMANTIC_KIND_STRUCTURED,
        SEMANTIC_KIND_UNBOUNDED,
    }
)
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


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def risk_rank(value: RiskClass | str) -> int:
    risk = RiskClass(value)
    return RISK_ORDER.index(risk)


def _risk_span(floor: RiskClass, ceiling: RiskClass) -> tuple[RiskClass, ...]:
    start = risk_rank(floor)
    end = risk_rank(ceiling)
    if start > end:
        raise ResidualIntelligenceError("family risk floor exceeds risk ceiling")
    return RISK_ORDER[start : end + 1]


def _schema_name(family: ResidualTaskFamily, kind: str) -> str:
    slug = family.value.lower().replace("_", "-")
    return f"ipfs_accelerate_py/agent-supervisor/residual-family-{slug}-{kind}@1"


def _validator_identity(family: ResidualTaskFamily) -> str:
    slug = family.value.lower().replace("_", "-")
    return f"validator:{slug}@1"


def _expert_class_letters(values: Any, name: str) -> tuple[str, ...]:
    letters = text_tuple(values, name, allow_empty=False, max_items=5)
    unknown = sorted(set(letters) - CLOSED_EXPERT_CLASS_LETTERS)
    if unknown:
        raise ResidualIntelligenceError(f"{name} contains unknown expert classes: {', '.join(unknown)}")
    if list(letters) != sorted(letters):
        raise ResidualIntelligenceError(f"{name} must be listed in smallest-form order")
    return letters


@dataclass(frozen=True)
class ResidualTaskFamilySpec:
    """Exact shared semantic boundary, schemas, limits, and gates for one family."""

    task_family: ResidualTaskFamily
    semantic_kind: str
    input_semantics: str
    output_semantics: str
    authority_class: str
    validation_contract: str
    validator_identity: str
    independent_validator_required: bool
    error_behavior: str
    abstention_behavior: str
    risk_floor: RiskClass
    risk_ceiling: RiskClass
    allowed_risk_classes: tuple[RiskClass, ...]
    input_schema: str
    output_schema: str
    output_classes: tuple[str, ...]
    allowed_compact_feature_keys: tuple[str, ...]
    required_compact_feature_keys: tuple[str, ...]
    maximum_input_tokens: int
    maximum_output_tokens: int
    maximum_output_bytes: int
    privacy_class: PrivacyClass
    privacy_route_policy: str
    capabilities: tuple[str, ...]
    smallest_expert_class: str
    eligible_expert_classes: tuple[str, ...]
    emit_prose_by_default: bool
    always_abstain: bool
    candidate_only: bool = True
    evaluation_corpus_admission_id: str = ""
    schema: str = TASK_FAMILY_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "spec_id",
            "task_family",
            "semantic_kind",
            "input_semantics",
            "output_semantics",
            "authority_class",
            "validation_contract",
            "validator_identity",
            "independent_validator_required",
            "error_behavior",
            "abstention_behavior",
            "risk_floor",
            "risk_ceiling",
            "allowed_risk_classes",
            "input_schema",
            "output_schema",
            "output_classes",
            "allowed_compact_feature_keys",
            "required_compact_feature_keys",
            "maximum_input_tokens",
            "maximum_output_tokens",
            "maximum_output_bytes",
            "privacy_class",
            "privacy_route_policy",
            "capabilities",
            "smallest_expert_class",
            "eligible_expert_classes",
            "emit_prose_by_default",
            "always_abstain",
            "candidate_only",
            "evaluation_corpus_admission_id",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != TASK_FAMILY_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual task family spec schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(
            self, "semantic_kind", required_text(self.semantic_kind, "semantic_kind", max_bytes=64)
        )
        if self.semantic_kind not in CLOSED_SEMANTIC_KINDS:
            raise ResidualIntelligenceError("semantic_kind is outside the closed taxonomy")
        for field in (
            "input_semantics",
            "output_semantics",
            "authority_class",
            "validation_contract",
            "validator_identity",
            "error_behavior",
            "abstention_behavior",
            "input_schema",
            "output_schema",
            "privacy_route_policy",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.authority_class != CANDIDATE_ONLY_AUTHORITY:
            raise ResidualIntelligenceError(
                "residual family authority_class must be candidate_only"
            )
        object.__setattr__(
            self,
            "independent_validator_required",
            _require_bool(self.independent_validator_required, "independent_validator_required"),
        )
        if not self.independent_validator_required:
            raise ResidualIntelligenceError(REASON_VALIDATOR_REQUIRED)
        if not self.validator_identity.startswith("validator:"):
            raise ResidualIntelligenceError(REASON_VALIDATOR_REQUIRED)
        object.__setattr__(self, "risk_floor", RiskClass(self.risk_floor))
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        allowed = tuple(RiskClass(item) for item in self.allowed_risk_classes)
        expected = _risk_span(self.risk_floor, self.risk_ceiling)
        if allowed != expected:
            raise ResidualIntelligenceError("allowed_risk_classes must equal the closed floor-ceiling span")
        object.__setattr__(self, "allowed_risk_classes", allowed)
        object.__setattr__(
            self,
            "output_classes",
            text_tuple(self.output_classes, "output_classes", allow_empty=False, max_items=16),
        )
        if ABSTAIN_OUTPUT_CLASS not in self.output_classes:
            raise ResidualIntelligenceError("family output classes must include ABSTAIN")
        object.__setattr__(
            self,
            "allowed_compact_feature_keys",
            text_tuple(
                self.allowed_compact_feature_keys,
                "allowed_compact_feature_keys",
                allow_empty=False,
                max_items=64,
            ),
        )
        object.__setattr__(
            self,
            "required_compact_feature_keys",
            text_tuple(
                self.required_compact_feature_keys,
                "required_compact_feature_keys",
                max_items=32,
            ),
        )
        if not set(self.required_compact_feature_keys).issubset(self.allowed_compact_feature_keys):
            raise ResidualIntelligenceError("required compact features are not in the closed input schema")
        object.__setattr__(
            self,
            "maximum_input_tokens",
            bounded_int(
                self.maximum_input_tokens,
                "maximum_input_tokens",
                minimum=32,
                maximum=MAX_FAMILY_INPUT_TOKENS,
            ),
        )
        object.__setattr__(
            self,
            "maximum_output_tokens",
            bounded_int(
                self.maximum_output_tokens,
                "maximum_output_tokens",
                minimum=8,
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
                maximum=32_768,
            ),
        )
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        if self.privacy_route_policy not in CLOSED_PRIVACY_ROUTES:
            raise ResidualIntelligenceError("privacy_route_policy is outside the closed set")
        if self.privacy_class in LOCAL_ONLY_PRIVACY:
            if self.privacy_route_policy != PRIVACY_ROUTE_LOCAL_ONLY:
                raise ResidualIntelligenceError(
                    "high-sensitivity privacy classes require a local_only privacy route"
                )
        object.__setattr__(
            self,
            "capabilities",
            text_tuple(self.capabilities, "capabilities", allow_empty=False, max_items=16),
        )
        eligible = _expert_class_letters(self.eligible_expert_classes, "eligible_expert_classes")
        smallest = required_text(self.smallest_expert_class, "smallest_expert_class", max_bytes=1)
        if smallest not in CLOSED_EXPERT_CLASS_LETTERS:
            raise ResidualIntelligenceError("smallest_expert_class must be one of A-E")
        if smallest != eligible[0]:
            raise ResidualIntelligenceError("smallest_expert_class must be the first eligible class")
        object.__setattr__(self, "eligible_expert_classes", eligible)
        object.__setattr__(self, "smallest_expert_class", smallest)
        object.__setattr__(
            self,
            "emit_prose_by_default",
            _require_bool(self.emit_prose_by_default, "emit_prose_by_default"),
        )
        if self.emit_prose_by_default:
            raise ResidualIntelligenceError(REASON_PROSE_DEFAULT)
        object.__setattr__(self, "always_abstain", _require_bool(self.always_abstain, "always_abstain"))
        if self.always_abstain and self.output_classes != (ABSTAIN_OUTPUT_CLASS,):
            raise ResidualIntelligenceError("always-abstain families may only emit ABSTAIN")
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("family specifications must remain candidate_only=true")
        admission_id = self.evaluation_corpus_admission_id
        if admission_id in (None, ""):
            object.__setattr__(self, "evaluation_corpus_admission_id", "")
        else:
            object.__setattr__(
                self,
                "evaluation_corpus_admission_id",
                required_text(admission_id, "evaluation_corpus_admission_id"),
            )
        if self.error_behavior != ERROR_INVALID_OUTPUT:
            raise ResidualIntelligenceError("family error_behavior must be invalid_output")

    @property
    def spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_family_boundary(self) -> ResidualFamilyBoundary:
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

    def allows_risk(self, risk: RiskClass | str) -> bool:
        try:
            return RiskClass(risk) in self.allowed_risk_classes
        except ValueError:
            return False

    def reject_unsupported_risk(self, risk: RiskClass | str) -> RiskClass:
        parsed = RiskClass(risk)
        if parsed not in self.allowed_risk_classes:
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_FAMILY_RISK)
        if risk_rank(parsed) > risk_rank(self.risk_ceiling):
            raise ResidualIntelligenceError(REASON_RISK_CEILING)
        return parsed

    def allows_expert_class(self, letter: str) -> bool:
        return required_text(letter, "expert class", max_bytes=1) in self.eligible_expert_classes

    def privacy_route_permits(self, *, provider_authorized: bool, local_execution: bool) -> bool:
        if type(provider_authorized) is not bool or type(local_execution) is not bool:
            raise ResidualIntelligenceError("privacy route flags must be boolean")
        if self.privacy_route_policy == PRIVACY_ROUTE_LOCAL_ONLY:
            return local_execution
        if not local_execution and self.privacy_class in LOCAL_ONLY_PRIVACY:
            return False
        return local_execution or provider_authorized

    def validate_compact_features(self, features: Mapping[str, Any]) -> None:
        if not isinstance(features, Mapping):
            raise ResidualIntelligenceError("compact_features must be an object")
        unknown = sorted(str(key) for key in features if key not in self.allowed_compact_feature_keys)
        if unknown:
            raise ResidualIntelligenceError(
                f"{REASON_UNKNOWN_COMPACT_FEATURE}: {', '.join(unknown)}"
            )
        missing = [key for key in self.required_compact_feature_keys if key not in features]
        if missing:
            raise ResidualIntelligenceError(
                f"{REASON_MISSING_COMPACT_FEATURE}: {', '.join(missing)}"
            )

    def validate_task_input(self, task_input: ResidualTaskInput) -> None:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
        if task_input.task_family is not self.task_family:
            raise ResidualIntelligenceError(REASON_TASK_FAMILY_MISMATCH)
        self.reject_unsupported_risk(task_input.risk_class)
        self.validate_compact_features(task_input.compact_features)
        if task_input.token_budget > self.maximum_input_tokens:
            raise ResidualIntelligenceError(REASON_TOKEN_LIMIT)
        illegal = [item for item in task_input.allowed_outputs if item not in self.output_classes]
        if illegal:
            raise ResidualIntelligenceError(REASON_OUTPUT_CLASS)
        if self.always_abstain and tuple(task_input.allowed_outputs) != (ABSTAIN_OUTPUT_CLASS,):
            raise ResidualIntelligenceError("always-abstain families only allow ABSTAIN")

    def bind_evaluation_admission(self, admission: TrainingCorpusAdmission) -> None:
        """Dataset references are declarative and must resolve to an admitted corpus."""

        if not self.evaluation_corpus_admission_id:
            raise ResidualIntelligenceError("family spec carries no evaluation dataset reference")
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
        if admission.admission_id != self.evaluation_corpus_admission_id:
            raise ResidualIntelligenceError("evaluation corpus admission identity mismatch")
        if admission.admission_decision.value != "admitted":
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "semantic_kind": self.semantic_kind,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "authority_class": self.authority_class,
            "validation_contract": self.validation_contract,
            "validator_identity": self.validator_identity,
            "independent_validator_required": True,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "risk_floor": self.risk_floor.value,
            "risk_ceiling": self.risk_ceiling.value,
            "allowed_risk_classes": [item.value for item in self.allowed_risk_classes],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "output_classes": list(self.output_classes),
            "allowed_compact_feature_keys": list(self.allowed_compact_feature_keys),
            "required_compact_feature_keys": list(self.required_compact_feature_keys),
            "maximum_input_tokens": self.maximum_input_tokens,
            "maximum_output_tokens": self.maximum_output_tokens,
            "maximum_output_bytes": self.maximum_output_bytes,
            "privacy_class": self.privacy_class.value,
            "privacy_route_policy": self.privacy_route_policy,
            "capabilities": list(self.capabilities),
            "smallest_expert_class": self.smallest_expert_class,
            "eligible_expert_classes": list(self.eligible_expert_classes),
            "emit_prose_by_default": False,
            "always_abstain": self.always_abstain,
            "candidate_only": True,
            "evaluation_corpus_admission_id": self.evaluation_corpus_admission_id,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualTaskFamilySpec:
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("family spec must be an object")
        examples = sorted(str(key) for key in payload if key in _FORBIDDEN_EXAMPLE_FIELDS)
        if examples:
            raise UnknownFieldError(f"{REASON_EXAMPLES_FORBIDDEN}: {', '.join(examples)}")
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"spec_id", "evaluation_corpus_admission_id"},
            noun="residual task family spec",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            semantic_kind=str(payload.get("semantic_kind") or ""),
            input_semantics=str(payload.get("input_semantics") or ""),
            output_semantics=str(payload.get("output_semantics") or ""),
            authority_class=str(payload.get("authority_class") or ""),
            validation_contract=str(payload.get("validation_contract") or ""),
            validator_identity=str(payload.get("validator_identity") or ""),
            independent_validator_required=payload.get("independent_validator_required"),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            risk_floor=RiskClass(str(payload.get("risk_floor") or "")),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            allowed_risk_classes=tuple(
                RiskClass(str(item)) for item in (payload.get("allowed_risk_classes") or ())
            ),
            input_schema=str(payload.get("input_schema") or ""),
            output_schema=str(payload.get("output_schema") or ""),
            output_classes=tuple(payload.get("output_classes") or ()),
            allowed_compact_feature_keys=tuple(payload.get("allowed_compact_feature_keys") or ()),
            required_compact_feature_keys=tuple(payload.get("required_compact_feature_keys") or ()),
            maximum_input_tokens=payload.get("maximum_input_tokens"),
            maximum_output_tokens=payload.get("maximum_output_tokens"),
            maximum_output_bytes=payload.get("maximum_output_bytes"),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            privacy_route_policy=str(payload.get("privacy_route_policy") or ""),
            capabilities=tuple(payload.get("capabilities") or ()),
            smallest_expert_class=str(payload.get("smallest_expert_class") or ""),
            eligible_expert_classes=tuple(payload.get("eligible_expert_classes") or ()),
            emit_prose_by_default=payload.get("emit_prose_by_default"),
            always_abstain=payload.get("always_abstain"),
            candidate_only=payload.get("candidate_only"),
            evaluation_corpus_admission_id=str(payload.get("evaluation_corpus_admission_id") or ""),
        )
        claimed = str(payload.get("spec_id") or "")
        if claimed and claimed != result.spec_id:
            raise ResidualIntelligenceError("residual task family spec identity mismatch")
        return result


def _features(*required: str, extra: Sequence[str] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required_keys = tuple(required)
    allowed = tuple(dict.fromkeys((*required_keys, *extra, *SHARED_OPTIONAL_FEATURES)))
    return allowed, required_keys


def _family_spec(
    family: ResidualTaskFamily,
    *,
    semantic_kind: str,
    input_semantics: str,
    output_semantics: str,
    abstention_behavior: str,
    risk_floor: RiskClass,
    risk_ceiling: RiskClass,
    privacy_class: PrivacyClass,
    privacy_route_policy: str,
    capabilities: tuple[str, ...],
    eligible_expert_classes: tuple[str, ...],
    feature_keys: tuple[str, ...],
    required_feature_keys: tuple[str, ...],
    maximum_input_tokens: int,
    maximum_output_tokens: int,
    always_abstain: bool = False,
) -> ResidualTaskFamilySpec:
    grammar = grammar_for(family)
    return ResidualTaskFamilySpec(
        task_family=family,
        semantic_kind=semantic_kind,
        input_semantics=input_semantics,
        output_semantics=output_semantics,
        authority_class=CANDIDATE_ONLY_AUTHORITY,
        validation_contract=f"{_validator_identity(family)}#independent",
        validator_identity=_validator_identity(family),
        independent_validator_required=True,
        error_behavior=ERROR_INVALID_OUTPUT,
        abstention_behavior=abstention_behavior,
        risk_floor=risk_floor,
        risk_ceiling=risk_ceiling,
        allowed_risk_classes=_risk_span(risk_floor, risk_ceiling),
        input_schema=_schema_name(family, "input"),
        output_schema=_schema_name(family, "output"),
        output_classes=grammar.output_classes,
        allowed_compact_feature_keys=feature_keys,
        required_compact_feature_keys=required_feature_keys,
        maximum_input_tokens=maximum_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
        maximum_output_bytes=grammar.maximum_output_bytes,
        privacy_class=privacy_class,
        privacy_route_policy=privacy_route_policy,
        capabilities=capabilities,
        smallest_expert_class=eligible_expert_classes[0],
        eligible_expert_classes=eligible_expert_classes,
        emit_prose_by_default=False,
        always_abstain=always_abstain,
        candidate_only=True,
    )


_CLASSIFICATION: Final[tuple[str, ...]] = ("A", "B", "C")
_RANKING: Final[tuple[str, ...]] = ("A", "B", "C", "D")
_MATCHING: Final[tuple[str, ...]] = ("A", "B", "C")
_STRUCTURED: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E")
_UNBOUNDED: Final[tuple[str, ...]] = ("A",)
_CAP_CLASSIFY: Final[tuple[str, ...]] = (
    "provider_free_capability_contract",
    "cpu-small-hermetic",
    "independent_validator",
    "no_network",
)
_CAP_RANK: Final[tuple[str, ...]] = (
    "provider_free_capability_contract",
    "cpu-medium-batch",
    "independent_validator",
    "no_network",
)
_CAP_MATCH: Final[tuple[str, ...]] = (
    "provider_free_capability_contract",
    "cpu-small-hermetic",
    "independent_validator",
    "no_network",
)
_CAP_STRUCTURED: Final[tuple[str, ...]] = (
    "provider_free_capability_contract",
    "cpu-gpu-optional-bounded",
    "independent_validator",
    "no_network",
    "grammar_constrained_decode",
)
_CAP_UNBOUNDED: Final[tuple[str, ...]] = (
    "provider_free_capability_contract",
    "cpu-small-hermetic",
    "independent_validator",
    "no_network",
    "human_review_fallback",
)


def _build_family_specs() -> dict[ResidualTaskFamily, ResidualTaskFamilySpec]:
    failure_features = _features(
        "exit_code",
        "failure_signature",
        extra=(
            "procedure_root",
            "procedure_answer_available",
            "procedure_preconditions_satisfied",
            "ranking_candidates",
            "ranking_signals",
        ),
    )
    ranking_features = _features("ranking_candidates", extra=("ranking_signals", "obligation_id"))
    matching_features = _features(
        "procedure_root",
        extra=("procedure_answer_available", "procedure_preconditions_satisfied", "template_id"),
    )
    structured_features = _features(
        "hole_id",
        extra=(
            "procedure_root",
            "procedure_preconditions_satisfied",
            "symbol_ids",
            "allowed_paths",
            "obligation_id",
        ),
    )
    classify_features = _features("label_candidates")
    recipes = (
        _family_spec(
            ResidualTaskFamily.TASK_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="one bounded supervisor question plus closed compact operation features",
            output_semantics="exactly one residual task-family label or ABSTAIN",
            abstention_behavior="unknown operations and mixed-family prompts abstain",
            risk_floor=RiskClass.R0,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.INTERNAL,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=classify_features[0],
            required_feature_keys=classify_features[1],
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.RISK_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="bounded effect, authority, and validation features for one task capsule",
            output_semantics="exactly one closed risk label R0-R5 or ABSTAIN",
            abstention_behavior="missing effect or authority features abstain rather than guess R0",
            risk_floor=RiskClass.R0,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.INTERNAL,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=classify_features[0],
            required_feature_keys=classify_features[1],
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.EFFECT_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="declared mutation surfaces, paths, and capability tokens for one action",
            output_semantics="one closed non-empty effect-class set or ABSTAIN",
            abstention_behavior="unseen effect tokens abstain; never invent an effect class",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("effect_candidates")[0],
            required_feature_keys=_features("effect_candidates")[1],
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="requested mutation, current policy identity, and declared authority tokens",
            output_semantics="one required-authority label that remains candidate_only or ABSTAIN",
            abstention_behavior="unknown authority tokens and confirmation-shaped requests abstain",
            risk_floor=RiskClass.R2,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("authority_candidates")[0],
            required_feature_keys=_features("authority_candidates")[1],
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.CONTEXT_SUFFICIENCY,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="context-capsule identity plus required evidence-field occupancy flags",
            output_semantics="boolean sufficiency with missing-reference identifiers or ABSTAIN",
            abstention_behavior="incomplete occupancy maps and unknown context tiers abstain",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("required_context_fields", extra=("occupied_context_fields",))[0],
            required_feature_keys=_features("required_context_fields")[1],
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.EVIDENCE_RANKING,
            semantic_kind=SEMANTIC_KIND_RANKING,
            input_semantics="closed candidate evidence identities with compact numeric ranking signals",
            output_semantics="permutation of those identities with descending scores_ppm or ABSTAIN",
            abstention_behavior="empty candidate sets, score ties without a rule, and missing groups abstain",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_RANK,
            eligible_expert_classes=_RANKING,
            feature_keys=ranking_features[0],
            required_feature_keys=ranking_features[1],
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.PROCEDURE_MATCHING,
            semantic_kind=SEMANTIC_KIND_MATCHING,
            input_semantics="current procedure-root identity, hole signature, and precondition flags",
            output_semantics="one procedure_id and match_class under compiler preconditions or ABSTAIN",
            abstention_behavior="unsatisfied preconditions and unbound procedure roots abstain",
            risk_floor=RiskClass.R2,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_MATCH,
            eligible_expert_classes=_MATCHING,
            feature_keys=matching_features[0],
            required_feature_keys=("procedure_root",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.PLAN_BRANCH_RANKING,
            semantic_kind=SEMANTIC_KIND_RANKING,
            input_semantics="closed plan-branch identities bound to one objective and obligation set",
            output_semantics="ranked branch identities with descending scores_ppm or ABSTAIN",
            abstention_behavior="unbound objectives and missing branch signals abstain",
            risk_floor=RiskClass.R2,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_RANK,
            eligible_expert_classes=_RANKING,
            feature_keys=ranking_features[0],
            required_feature_keys=ranking_features[1],
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.TEST_SELECTION,
            semantic_kind=SEMANTIC_KIND_RANKING,
            input_semantics="changed-symbol identities plus a closed selectable test-identity list",
            output_semantics="non-empty test_ids that do not delete or weaken tests, or ABSTAIN",
            abstention_behavior="empty selectable sets and hidden-test identifiers abstain",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_RANK,
            eligible_expert_classes=_RANKING,
            feature_keys=_features("selectable_test_ids", extra=("changed_symbol_ids",))[0],
            required_feature_keys=_features("selectable_test_ids")[1],
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.PROOF_SELECTION,
            semantic_kind=SEMANTIC_KIND_RANKING,
            input_semantics="exact current prover obligation identity and closed candidate proof ids",
            output_semantics="ranked proof_ids that remain nominations for the actual prover, or ABSTAIN",
            abstention_behavior="stale obligations, missing prover capability, and empty sets abstain",
            risk_floor=RiskClass.R3,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.PROOF_WITNESS,
            privacy_route_policy=PRIVACY_ROUTE_LOCAL_ONLY,
            capabilities=_CAP_RANK,
            eligible_expert_classes=_RANKING,
            feature_keys=_features("obligation_id", extra=("ranking_candidates", "ranking_signals"))[0],
            required_feature_keys=("obligation_id",),
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="validated failure signature, exit code, and bounded dependency references",
            output_semantics="one failure_class and one recommended_action candidate, or ABSTAIN",
            abstention_behavior="unknown signatures and critical-boundary mismatches abstain",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=failure_features[0],
            required_feature_keys=failure_features[1],
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.RETRY_OR_ESCALATE,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="prior attempt disposition, remaining budget, and failure-class identity",
            output_semantics="exactly one of retry, escalate, or stop, or ABSTAIN",
            abstention_behavior="exhausted budgets without an escalate path and unknown failures abstain",
            risk_floor=RiskClass.R2,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.INTERNAL,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("attempt_disposition", extra=("remaining_budget", "failure_class"))[0],
            required_feature_keys=("attempt_disposition",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="cache identity, current repository-state cid, and dependency references",
            output_semantics="boolean reuse decision with dependency references, or ABSTAIN",
            abstention_behavior="stale state cids and missing dependency edges abstain rather than reuse",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("cache_identity", extra=("dependency_reference_ids",))[0],
            required_feature_keys=("cache_identity",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="conflicting symbol identities and hunk metadata for one merge attempt",
            output_semantics="one conflict_class with involved symbol_ids, or ABSTAIN",
            abstention_behavior="binary conflicts, unknown symbols, and authority-file hunks abstain",
            risk_floor=RiskClass.R2,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("symbol_ids", extra=("hunk_count",))[0],
            required_feature_keys=("symbol_ids",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.PATCH_TEMPLATE_SELECTION,
            semantic_kind=SEMANTIC_KIND_MATCHING,
            input_semantics="failure class, symbol identities, and a closed selectable template list",
            output_semantics="one template_id bound to those symbols, or ABSTAIN",
            abstention_behavior="templates that would delete tests or change authority files abstain",
            risk_floor=RiskClass.R3,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_MATCH,
            eligible_expert_classes=_MATCHING,
            feature_keys=_features("template_candidates", extra=("symbol_ids", "failure_class"))[0],
            required_feature_keys=("template_candidates",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
            semantic_kind=SEMANTIC_KIND_STRUCTURED,
            input_semantics="declared typed hole identity, operator vocabulary, and compiler preconditions",
            output_semantics="ProcedureHoleResolution operator and argument references, or ABSTAIN",
            abstention_behavior="missing compiler capability or failed preconditions abstain",
            risk_floor=RiskClass.R3,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_STRUCTURED,
            eligible_expert_classes=_STRUCTURED,
            feature_keys=structured_features[0],
            required_feature_keys=("hole_id",),
            maximum_input_tokens=4096,
            maximum_output_tokens=512,
        ),
        _family_spec(
            ResidualTaskFamily.PATCH_SKETCH_GENERATION,
            semantic_kind=SEMANTIC_KIND_STRUCTURED,
            input_semantics="allowed relative paths, symbol bounds, and predetermined validation identities",
            output_semantics="PatchSketchIR paths, symbols, operations, and line bound, or ABSTAIN",
            abstention_behavior="out-of-scope paths, test deletion, and validation weakening abstain",
            risk_floor=RiskClass.R4,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_STRUCTURED,
            eligible_expert_classes=_STRUCTURED,
            feature_keys=_features(
                "allowed_paths",
                extra=("symbol_ids", "maximum_changed_lines", "validation_ids"),
            )[0],
            required_feature_keys=("allowed_paths",),
            maximum_input_tokens=4096,
            maximum_output_tokens=1024,
        ),
        _family_spec(
            ResidualTaskFamily.LEMMA_SUGGESTION,
            semantic_kind=SEMANTIC_KIND_STRUCTURED,
            input_semantics="current obligation identity and closed premise identities for one prover goal",
            output_semantics="lemma_ids nominating lemmas for the actual prover, or ABSTAIN",
            abstention_behavior="stale obligations and missing prover environment abstain",
            risk_floor=RiskClass.R4,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.PROOF_WITNESS,
            privacy_route_policy=PRIVACY_ROUTE_LOCAL_ONLY,
            capabilities=_CAP_STRUCTURED,
            eligible_expert_classes=_STRUCTURED,
            feature_keys=_features("obligation_id", extra=("premise_ids",))[0],
            required_feature_keys=("obligation_id",),
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.TACTIC_SUGGESTION,
            semantic_kind=SEMANTIC_KIND_STRUCTURED,
            input_semantics="current obligation identity, allowed tactic vocabulary, and premise identities",
            output_semantics="tactic_ids ranked as nominations, never as proof, or ABSTAIN",
            abstention_behavior="unknown tactics, omitted obligations, and prover unavailability abstain",
            risk_floor=RiskClass.R4,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.PROOF_WITNESS,
            privacy_route_policy=PRIVACY_ROUTE_LOCAL_ONLY,
            capabilities=_CAP_STRUCTURED,
            eligible_expert_classes=_STRUCTURED,
            feature_keys=_features("obligation_id", extra=("premise_ids", "allowed_tactic_ids"))[0],
            required_feature_keys=("obligation_id",),
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION,
            semantic_kind=SEMANTIC_KIND_STRUCTURED,
            input_semantics="counterexample identity, failed invariant ids, and current obligation binding",
            output_semantics="failure_class plus counterexample references, or ABSTAIN",
            abstention_behavior="missing counterexample identities and recoverable witnesses abstain",
            risk_floor=RiskClass.R3,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.PROOF_WITNESS,
            privacy_route_policy=PRIVACY_ROUTE_LOCAL_ONLY,
            capabilities=_CAP_STRUCTURED,
            eligible_expert_classes=_STRUCTURED,
            feature_keys=_features(
                "counterexample_reference_ids",
                extra=("violated_invariant_ids", "obligation_id"),
            )[0],
            required_feature_keys=("counterexample_reference_ids",),
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE,
            semantic_kind=SEMANTIC_KIND_STRUCTURED,
            input_semantics="parent goal identity and closed candidate goal-kind vocabulary",
            output_semantics="parent_goal_id with candidate_goal_kinds, or ABSTAIN",
            abstention_behavior="unknown parent goals and completion-shaped refinements abstain",
            risk_floor=RiskClass.R3,
            risk_ceiling=RiskClass.R4,
            privacy_class=PrivacyClass.INTERNAL,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_STRUCTURED,
            eligible_expert_classes=_STRUCTURED,
            feature_keys=_features("parent_goal_id", extra=("candidate_goal_kinds",))[0],
            required_feature_keys=("parent_goal_id",),
            maximum_input_tokens=2048,
            maximum_output_tokens=256,
        ),
        _family_spec(
            ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="documentation claim identity plus bounded evidence-reference occupancy",
            output_semantics="claim_class, rewrite_required flag, and evidence ids, or ABSTAIN",
            abstention_behavior="claims without evidence references and unverifiable prose abstain",
            risk_floor=RiskClass.R1,
            risk_ceiling=RiskClass.R3,
            privacy_class=PrivacyClass.INTERNAL,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("claim_identity", extra=("evidence_reference_ids",))[0],
            required_feature_keys=("claim_identity",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION,
            semantic_kind=SEMANTIC_KIND_CLASSIFICATION,
            input_semantics="prior abstention, risk, and disagreement features for one residual decision",
            output_semantics="boolean escalate with a closed reason_code, or ABSTAIN",
            abstention_behavior="missing disagreement evidence at high risk conservatively escalates via ABSTAIN",
            risk_floor=RiskClass.R2,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.INTERNAL,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_CLASSIFY,
            eligible_expert_classes=_CLASSIFICATION,
            feature_keys=_features("prior_disposition", extra=("disagreement", "risk_class"))[0],
            required_feature_keys=("prior_disposition",),
            maximum_input_tokens=1024,
            maximum_output_tokens=128,
        ),
        _family_spec(
            ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
            semantic_kind=SEMANTIC_KIND_UNBOUNDED,
            input_semantics="any residual request that has left the closed 23-family boundary",
            output_semantics="ABSTAIN only; never a free-form answer, proof, or completion",
            abstention_behavior="always abstain and escalate; no local or remote specialist may accept",
            risk_floor=RiskClass.R5,
            risk_ceiling=RiskClass.R5,
            privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
            privacy_route_policy=PRIVACY_ROUTE_AUTHORIZED_PROVIDER,
            capabilities=_CAP_UNBOUNDED,
            eligible_expert_classes=_UNBOUNDED,
            feature_keys=_features("unbounded_reason")[0],
            required_feature_keys=("unbounded_reason",),
            maximum_input_tokens=256,
            maximum_output_tokens=32,
            always_abstain=True,
        ),
    )
    mapping = {item.task_family: item for item in recipes}
    if set(mapping) != set(ResidualTaskFamily):
        missing = sorted(item.value for item in set(ResidualTaskFamily) - set(mapping))
        raise ResidualIntelligenceError(f"family spec registry is missing {', '.join(missing)}")
    semantics = [(item.input_semantics, item.output_semantics) for item in recipes]
    if len(set(semantics)) != len(semantics):
        raise ResidualIntelligenceError("two families share input and output semantics")
    return mapping


FAMILY_SPECS: Final[Mapping[ResidualTaskFamily, ResidualTaskFamilySpec]] = _build_family_specs()


def family_spec_for(task_family: ResidualTaskFamily | str) -> ResidualTaskFamilySpec:
    return FAMILY_SPECS[ResidualTaskFamily(task_family)]


def all_family_specs() -> tuple[ResidualTaskFamilySpec, ...]:
    return tuple(FAMILY_SPECS[item] for item in ResidualTaskFamily)


def reject_unsupported_family_risk(
    task_family: ResidualTaskFamily | str,
    risk: RiskClass | str,
) -> ResidualTaskFamilySpec:
    spec = family_spec_for(task_family)
    spec.reject_unsupported_risk(risk)
    return spec


def family_spec_registry_payload() -> dict[str, Any]:
    specs = all_family_specs()
    return {
        "schema": FAMILY_SPEC_REGISTRY_SCHEMA,
        "registry_version": FAMILY_SPEC_REGISTRY_VERSION,
        "family_count": len(specs),
        "families": [item.to_dict() for item in specs],
    }


__all__ = (
    "ABSTAIN_OUTPUT_CLASS",
    "CANDIDATE_ONLY_AUTHORITY",
    "CLOSED_EXPERT_CLASS_LETTERS",
    "ERROR_INVALID_OUTPUT",
    "FAMILY_SPECS",
    "FAMILY_SPEC_REGISTRY_SCHEMA",
    "FAMILY_SPEC_REGISTRY_VERSION",
    "PRIVACY_ROUTE_AUTHORIZED_PROVIDER",
    "PRIVACY_ROUTE_LOCAL_ONLY",
    "PRIVACY_ROUTE_PUBLIC_OR_LOCAL",
    "REASON_PROSE_DEFAULT",
    "REASON_RISK_CEILING",
    "REASON_TASK_FAMILY_MISMATCH",
    "REASON_UNSUPPORTED_FAMILY_RISK",
    "REASON_VALIDATOR_REQUIRED",
    "ResidualTaskFamilySpec",
    "TASK_FAMILY_SPEC_SCHEMA",
    "all_family_specs",
    "family_spec_for",
    "family_spec_registry_payload",
    "reject_unsupported_family_risk",
    "risk_rank",
)
