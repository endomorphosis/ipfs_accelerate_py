"""Closed residual task-family specifications and semantic boundaries.

Each of the 24 taxonomy families has one exact semantic boundary.  Prompt or
embedding similarity cannot merge families.  Specifications carry no examples
and never grant training, promotion, or completion authority.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import json
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
    bounded_json_mapping,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)
from .inventory import ResidualFamilyBoundary
from .residual_ir import MAX_TOKEN_BUDGET, ResidualTaskInput
from .structured_decoding import (
    MAX_STRUCTURED_OUTPUT_BYTES,
    ExpertGrammar,
    grammar_for,
)

CLOSED_SCHEMA_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-closed-schema@1"
TASK_FAMILY_SPEC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-task-family-spec@1"
)
AUTHORITY_CANDIDATE_ONLY: Final = "candidate_only"
ERROR_INVALID_OUTPUT: Final = (
    "invalid_output; no prose recovery; failed validation escalates"
)
ERROR_CLASS_E_ABSTAIN: Final = (
    "invalid_output; class E never emits a learned candidate"
)
ABSTENTION_SELECTIVE: Final = (
    "abstain on reject_input, OOD, missing calibration, below-threshold, "
    "capability_unavailable, and critical-boundary"
)
ABSTENTION_ALWAYS: Final = "unconditional abstention; never ACCEPT"
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})
RISK_ORDER: Final[tuple[RiskClass, ...]] = (
    RiskClass.R0,
    RiskClass.R1,
    RiskClass.R2,
    RiskClass.R3,
    RiskClass.R4,
    RiskClass.R5,
)
PRIVACY_STRICTNESS: Final[tuple[PrivacyClass, ...]] = (
    PrivacyClass.PUBLIC,
    PrivacyClass.INTERNAL,
    PrivacyClass.REPOSITORY_PRIVATE,
    PrivacyClass.TENANT_PRIVATE,
    PrivacyClass.MATTER_CONFIDENTIAL,
    PrivacyClass.PERSONAL_DATA,
    PrivacyClass.HEALTH_DATA,
    PrivacyClass.LEGAL_PRIVILEGED,
    PrivacyClass.PROOF_WITNESS,
    PrivacyClass.CREDENTIAL,
)
_PROSE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "prose",
        "explanation",
        "rationale",
        "commentary",
        "markdown",
        "natural_language",
        "freeform",
        "narrative",
    }
)
_REMOTE_PRIVACY: Final[frozenset[PrivacyClass]] = frozenset(
    {PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}
)


class ExpertClass(str, Enum):
    """Closed specialist class for one residual family.

    A: classification.  B: ranking.  C: selection.  D: structured generation.
    E: novel unbounded reasoning, which must abstain.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


def _require_bool(value: Any, name: str, *, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    if expected is not None and value is not expected:
        raise ResidualIntelligenceError(f"{name} must be {expected}")
    return value


def _kebab(family: ResidualTaskFamily) -> str:
    return family.value.lower().replace("_", "-")


def risks_through(
    ceiling: RiskClass, *, floor: RiskClass = RiskClass.R0
) -> tuple[RiskClass, ...]:
    start = RISK_ORDER.index(RiskClass(floor))
    end = RISK_ORDER.index(RiskClass(ceiling))
    if start > end:
        raise ResidualIntelligenceError("risk floor exceeds risk ceiling")
    return RISK_ORDER[start : end + 1]


def risk_rank(value: RiskClass | str) -> int:
    return RISK_ORDER.index(RiskClass(value))


def privacy_rank(value: PrivacyClass | str) -> int:
    return PRIVACY_STRICTNESS.index(PrivacyClass(value))


def _enumerations(value: Any) -> dict[str, tuple[str, ...]]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError("enumerations must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, items in value.items():
        name = required_text(key, "enumeration field", max_bytes=256)
        if name in _PROSE_FIELD_NAMES:
            raise ResidualIntelligenceError(f"closed schema cannot enumerate prose field {name}")
        result[name] = text_tuple(items, f"enumerations.{name}", allow_empty=False, max_items=256)
    return result


@dataclass(frozen=True)
class ClosedSchema:
    """One closed object schema.  Unknown fields and prose defaults are forbidden."""

    name: str
    fields: tuple[str, ...]
    required_fields: tuple[str, ...] = ()
    enumerations: Mapping[str, tuple[str, ...]] = None  # type: ignore[assignment]
    maximum_bytes: int = 4096
    allow_unknown: bool = False
    allow_prose: bool = False
    schema: str = CLOSED_SCHEMA_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "schema_id",
            "name",
            "fields",
            "required_fields",
            "enumerations",
            "maximum_bytes",
            "allow_unknown",
            "allow_prose",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CLOSED_SCHEMA_SCHEMA:
            raise ResidualIntelligenceError("unsupported closed schema")
        object.__setattr__(self, "name", required_text(self.name, "name", max_bytes=256))
        object.__setattr__(self, "fields", text_tuple(self.fields, "fields", max_items=64))
        object.__setattr__(
            self,
            "required_fields",
            text_tuple(self.required_fields, "required_fields", max_items=64),
        )
        if not set(self.required_fields).issubset(self.fields):
            raise ResidualIntelligenceError("required schema fields are not declared")
        prose = sorted(name for name in self.fields if name.casefold() in _PROSE_FIELD_NAMES)
        if prose:
            raise ResidualIntelligenceError(
                "closed schema cannot declare prose fields: " + ", ".join(prose)
            )
        enumerations = _enumerations(self.enumerations)
        unknown_enum = sorted(set(enumerations) - set(self.fields))
        if unknown_enum:
            raise ResidualIntelligenceError(
                "enumerations name undeclared fields: " + ", ".join(unknown_enum)
            )
        object.__setattr__(self, "enumerations", enumerations)
        object.__setattr__(
            self,
            "maximum_bytes",
            bounded_int(
                self.maximum_bytes,
                "maximum_bytes",
                minimum=1,
                maximum=MAX_STRUCTURED_OUTPUT_BYTES,
            ),
        )
        object.__setattr__(
            self, "allow_unknown", _require_bool(self.allow_unknown, "allow_unknown", expected=False)
        )
        object.__setattr__(
            self, "allow_prose", _require_bool(self.allow_prose, "allow_prose", expected=False)
        )

    @property
    def schema_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def validate(self, payload: Any, *, noun: str = "payload") -> dict[str, Any]:
        normalized = bounded_json_mapping(payload, noun)
        unknown = sorted(set(normalized) - set(self.fields))
        if unknown:
            raise ResidualIntelligenceError(f"{noun} contains unknown fields: {', '.join(unknown)}")
        missing = sorted(set(self.required_fields) - set(normalized))
        if missing:
            raise ResidualIntelligenceError(
                f"{noun} is missing required fields: {', '.join(missing)}"
            )
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self.maximum_bytes:
            raise ResidualIntelligenceError(f"{noun} exceeds {self.maximum_bytes} bytes")
        for key, allowed in self.enumerations.items():
            if key not in normalized:
                continue
            value = normalized[key]
            if isinstance(value, str):
                if value not in allowed:
                    raise ResidualIntelligenceError(f"{noun}.{key} is outside its closed enumeration")
                continue
            if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                invalid = [item for item in value if item not in allowed]
                if invalid:
                    raise ResidualIntelligenceError(f"{noun}.{key} is outside its closed enumeration")
                continue
            raise ResidualIntelligenceError(f"{noun}.{key} must be a token or token list")
        return normalized

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "name": self.name,
            "fields": list(self.fields),
            "required_fields": list(self.required_fields),
            "enumerations": {
                key: list(values) for key, values in sorted(self.enumerations.items())
            },
            "maximum_bytes": self.maximum_bytes,
            "allow_unknown": False,
            "allow_prose": False,
        }
        if include_id:
            result["schema_id"] = self.schema_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClosedSchema:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"schema_id"},
            noun="closed schema",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            name=str(payload.get("name") or ""),
            fields=tuple(payload.get("fields") or ()),
            required_fields=tuple(payload.get("required_fields") or ()),
            enumerations=payload.get("enumerations") or {},
            maximum_bytes=payload.get("maximum_bytes"),
            allow_unknown=payload.get("allow_unknown"),
            allow_prose=payload.get("allow_prose"),
        )
        claimed = str(payload.get("schema_id") or "")
        if claimed and claimed != result.schema_id:
            raise ResidualIntelligenceError("closed schema identity mismatch")
        return result


def _output_schema_for(grammar: ExpertGrammar) -> ClosedSchema:
    enumerations = {
        name: contract.allowed_values
        for name, contract in grammar.field_contracts.items()
        if contract.allowed_values
    }
    return ClosedSchema(
        name=f"{_kebab(grammar.task_family)}-output@1",
        fields=grammar.payload_fields,
        required_fields=grammar.required_payload_fields,
        enumerations=enumerations,
        maximum_bytes=grammar.maximum_output_bytes,
    )


def _input_schema_for(
    family: ResidualTaskFamily,
    fields: tuple[str, ...],
    *,
    maximum_bytes: int,
) -> ClosedSchema:
    return ClosedSchema(
        name=f"{_kebab(family)}-input@1",
        fields=fields,
        required_fields=(),
        enumerations={},
        maximum_bytes=maximum_bytes,
    )


@dataclass(frozen=True)
class ResidualTaskFamilySpec:
    """Exact shared semantic contract for one residual task family."""

    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    input_semantics: str
    output_semantics: str
    input_schema: ClosedSchema
    output_schema: ClosedSchema
    grammar_id: str
    output_classes: tuple[str, ...]
    token_budget: int
    maximum_output_bytes: int
    maximum_output_tokens: int
    risk_ceiling: RiskClass
    allowed_risk_classes: tuple[RiskClass, ...]
    privacy_class: PrivacyClass
    capabilities: tuple[str, ...]
    validation_contract: str
    error_behavior: str
    abstention_behavior: str
    remote_route_permitted: bool
    always_abstain: bool = False
    validator_required: bool = True
    prose_default: bool = False
    candidate_only: bool = True
    authority_class: str = AUTHORITY_CANDIDATE_ONLY
    schema: str = TASK_FAMILY_SPEC_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "spec_id",
            "task_family",
            "expert_class",
            "input_semantics",
            "output_semantics",
            "input_schema",
            "output_schema",
            "grammar_id",
            "output_classes",
            "token_budget",
            "maximum_output_bytes",
            "maximum_output_tokens",
            "risk_ceiling",
            "allowed_risk_classes",
            "privacy_class",
            "capabilities",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "remote_route_permitted",
            "always_abstain",
            "validator_required",
            "prose_default",
            "candidate_only",
            "authority_class",
            "family_boundary_id",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != TASK_FAMILY_SPEC_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual task family spec schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", ExpertClass(self.expert_class))
        for field in (
            "input_semantics",
            "output_semantics",
            "grammar_id",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
            "authority_class",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if not isinstance(self.input_schema, ClosedSchema) or not isinstance(
            self.output_schema, ClosedSchema
        ):
            raise ResidualIntelligenceError("family input and output schemas must be ClosedSchema")
        object.__setattr__(
            self,
            "output_classes",
            text_tuple(self.output_classes, "output_classes", allow_empty=False, max_items=16),
        )
        object.__setattr__(
            self,
            "token_budget",
            bounded_int(self.token_budget, "token_budget", minimum=1, maximum=MAX_TOKEN_BUDGET),
        )
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
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        if isinstance(self.allowed_risk_classes, (str, bytes, bytearray)) or not isinstance(
            self.allowed_risk_classes, Sequence
        ):
            raise ResidualIntelligenceError("allowed_risk_classes must be a sequence")
        risks = tuple(RiskClass(item) for item in self.allowed_risk_classes)
        if not risks:
            raise ResidualIntelligenceError("allowed_risk_classes must not be empty")
        if len(set(risks)) != len(risks):
            raise ResidualIntelligenceError("allowed_risk_classes contains duplicate values")
        if self.risk_ceiling not in risks:
            raise ResidualIntelligenceError("risk ceiling is outside allowed_risk_classes")
        if any(risk_rank(item) > risk_rank(self.risk_ceiling) for item in risks):
            raise ResidualIntelligenceError("allowed risk exceeds the family risk ceiling")
        object.__setattr__(self, "allowed_risk_classes", risks)
        object.__setattr__(self, "privacy_class", PrivacyClass(self.privacy_class))
        object.__setattr__(
            self,
            "capabilities",
            text_tuple(self.capabilities, "capabilities", allow_empty=False, max_items=32),
        )
        object.__setattr__(
            self,
            "remote_route_permitted",
            _require_bool(self.remote_route_permitted, "remote_route_permitted"),
        )
        if self.remote_route_permitted and self.privacy_class not in _REMOTE_PRIVACY:
            raise ResidualIntelligenceError(
                "private families cannot permit an unauthorized remote route"
            )
        object.__setattr__(
            self, "always_abstain", _require_bool(self.always_abstain, "always_abstain")
        )
        object.__setattr__(
            self,
            "validator_required",
            _require_bool(self.validator_required, "validator_required", expected=True),
        )
        object.__setattr__(
            self, "prose_default", _require_bool(self.prose_default, "prose_default", expected=False)
        )
        object.__setattr__(
            self,
            "candidate_only",
            _require_bool(self.candidate_only, "candidate_only", expected=True),
        )
        if self.authority_class.casefold() != AUTHORITY_CANDIDATE_ONLY:
            raise ResidualIntelligenceError(
                "residual expert family authority_class must be candidate_only"
            )
        grammar = grammar_for(self.task_family)
        if self.grammar_id != grammar.grammar_id:
            raise ResidualIntelligenceError("family spec grammar_id does not match the closed grammar")
        if set(self.output_classes) != set(grammar.output_classes):
            raise ResidualIntelligenceError("family output classes must equal the closed grammar")
        if self.maximum_output_bytes != grammar.maximum_output_bytes:
            raise ResidualIntelligenceError("family output bound must equal the closed grammar")
        if self.output_schema.schema_id != _output_schema_for(grammar).schema_id:
            raise ResidualIntelligenceError("family output schema must equal the closed grammar")
        if self.expert_class is ExpertClass.E:
            if not self.always_abstain:
                raise ResidualIntelligenceError("class E families must always abstain")
            if self.risk_ceiling is not RiskClass.R5:
                raise ResidualIntelligenceError("class E risk ceiling must be R5")
        elif self.always_abstain:
            raise ResidualIntelligenceError("non-E families cannot unconditionally abstain")

    @property
    def spec_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def boundary(self) -> ResidualFamilyBoundary:
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
    def family_boundary_id(self) -> str:
        return self.boundary().boundary_id

    def admit_risk(self, risk: RiskClass | str) -> RiskClass:
        ranked = RiskClass(risk)
        if ranked not in self.allowed_risk_classes:
            raise ResidualIntelligenceError(
                f"unsupported family-risk pair {self.task_family.value}/{ranked.value}"
            )
        if risk_rank(ranked) > risk_rank(self.risk_ceiling):
            raise ResidualIntelligenceError(
                f"risk {ranked.value} exceeds {self.task_family.value} ceiling {self.risk_ceiling.value}"
            )
        return ranked

    def proposal_tier(self, risk: RiskClass | str) -> bool:
        return self.admit_risk(risk) in PROPOSAL_RISKS or self.always_abstain

    def admit_input(self, task_input: ResidualTaskInput) -> ResidualTaskInput:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("admit_input requires ResidualTaskInput")
        if task_input.task_family is not self.task_family:
            raise ResidualIntelligenceError("task input family does not match the family spec")
        self.admit_risk(task_input.risk_class)
        if task_input.token_budget > self.token_budget:
            raise ResidualIntelligenceError("task input token budget exceeds the family limit")
        if task_input.validation_policy != self.validation_contract:
            raise ResidualIntelligenceError("task input validation_policy is not the required validator")
        unknown_outputs = sorted(set(task_input.allowed_outputs) - set(self.output_classes))
        if unknown_outputs:
            raise ResidualIntelligenceError(
                "task input allowed_outputs are outside the closed grammar: "
                + ", ".join(unknown_outputs)
            )
        self.input_schema.validate(task_input.compact_features, noun="compact_features")
        return task_input

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "grammar_id": self.grammar_id,
            "output_classes": list(self.output_classes),
            "token_budget": self.token_budget,
            "maximum_output_bytes": self.maximum_output_bytes,
            "maximum_output_tokens": self.maximum_output_tokens,
            "risk_ceiling": self.risk_ceiling.value,
            "allowed_risk_classes": [item.value for item in self.allowed_risk_classes],
            "privacy_class": self.privacy_class.value,
            "capabilities": list(self.capabilities),
            "validation_contract": self.validation_contract,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
            "remote_route_permitted": self.remote_route_permitted,
            "always_abstain": self.always_abstain,
            "validator_required": True,
            "prose_default": False,
            "candidate_only": True,
            "authority_class": AUTHORITY_CANDIDATE_ONLY,
            "family_boundary_id": self.family_boundary_id,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualTaskFamilySpec:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"spec_id", "family_boundary_id"},
            noun="residual task family spec",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=ExpertClass(str(payload.get("expert_class") or "")),
            input_semantics=str(payload.get("input_semantics") or ""),
            output_semantics=str(payload.get("output_semantics") or ""),
            input_schema=ClosedSchema.from_dict(payload.get("input_schema") or {}),
            output_schema=ClosedSchema.from_dict(payload.get("output_schema") or {}),
            grammar_id=str(payload.get("grammar_id") or ""),
            output_classes=tuple(payload.get("output_classes") or ()),
            token_budget=payload.get("token_budget"),
            maximum_output_bytes=payload.get("maximum_output_bytes"),
            maximum_output_tokens=payload.get("maximum_output_tokens"),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            allowed_risk_classes=tuple(payload.get("allowed_risk_classes") or ()),
            privacy_class=PrivacyClass(str(payload.get("privacy_class") or "")),
            capabilities=tuple(payload.get("capabilities") or ()),
            validation_contract=str(payload.get("validation_contract") or ""),
            error_behavior=str(payload.get("error_behavior") or ""),
            abstention_behavior=str(payload.get("abstention_behavior") or ""),
            remote_route_permitted=payload.get("remote_route_permitted"),
            always_abstain=payload.get("always_abstain", False),
            validator_required=payload.get("validator_required", True),
            prose_default=payload.get("prose_default", False),
            candidate_only=payload.get("candidate_only", True),
            authority_class=str(payload.get("authority_class") or AUTHORITY_CANDIDATE_ONLY),
        )
        claimed = str(payload.get("spec_id") or "")
        if claimed and claimed != result.spec_id:
            raise ResidualIntelligenceError("residual task family spec identity mismatch")
        claimed_boundary = str(payload.get("family_boundary_id") or "")
        if claimed_boundary and claimed_boundary != result.family_boundary_id:
            raise ResidualIntelligenceError("family boundary identity mismatch")
        return result


@dataclass(frozen=True)
class _FamilyBlueprint:
    family: ResidualTaskFamily
    expert_class: ExpertClass
    input_semantics: str
    output_semantics: str
    input_fields: tuple[str, ...]
    risk_ceiling: RiskClass
    allowed_risks: tuple[RiskClass, ...]
    privacy: PrivacyClass
    token_budget: int
    capabilities: tuple[str, ...]
    remote: bool = False
    always_abstain: bool = False


def _spec_from_blueprint(item: _FamilyBlueprint) -> ResidualTaskFamilySpec:
    grammar = grammar_for(item.family)
    abstain = item.always_abstain or item.expert_class is ExpertClass.E
    return ResidualTaskFamilySpec(
        task_family=item.family,
        expert_class=item.expert_class,
        input_semantics=item.input_semantics,
        output_semantics=item.output_semantics,
        input_schema=_input_schema_for(
            item.family, item.input_fields, maximum_bytes=min(item.token_budget * 8, 4096)
        ),
        output_schema=_output_schema_for(grammar),
        grammar_id=grammar.grammar_id,
        output_classes=grammar.output_classes,
        token_budget=item.token_budget,
        maximum_output_bytes=grammar.maximum_output_bytes,
        maximum_output_tokens=max(1, grammar.maximum_output_bytes // 4),
        risk_ceiling=item.risk_ceiling,
        allowed_risk_classes=item.allowed_risks,
        privacy_class=item.privacy,
        capabilities=item.capabilities,
        validation_contract=f"{_kebab(item.family)}-validator@1",
        error_behavior=ERROR_CLASS_E_ABSTAIN if abstain else ERROR_INVALID_OUTPUT,
        abstention_behavior=ABSTENTION_ALWAYS if abstain else ABSTENTION_SELECTIVE,
        remote_route_permitted=item.remote,
        always_abstain=abstain,
    )


_BLUEPRINTS: Final[tuple[_FamilyBlueprint, ...]] = (
    _FamilyBlueprint(
        ResidualTaskFamily.TASK_CLASSIFICATION,
        ExpertClass.A,
        "bounded task descriptors and candidate family identifiers",
        "one closed task-family label plus optional evidence references",
        ("candidate_label_ids", "evidence_reference_ids"),
        RiskClass.R2,
        risks_through(RiskClass.R2),
        PrivacyClass.INTERNAL,
        512,
        ("cpu-small-hermetic", "provider-free", "classification"),
        remote=True,
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.RISK_CLASSIFICATION,
        ExpertClass.A,
        "bounded effect and authority references for a candidate action",
        "one closed risk class in R0-R5 plus optional evidence references",
        ("candidate_risk_ids", "evidence_reference_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3),
        PrivacyClass.INTERNAL,
        512,
        ("cpu-small-hermetic", "provider-free", "classification"),
        remote=True,
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.EFFECT_CLASSIFICATION,
        ExpertClass.A,
        "bounded operation descriptors and declared effect vocabularies",
        "one or more closed effect classes plus optional evidence references",
        ("candidate_effect_ids", "evidence_reference_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        768,
        ("cpu-small-hermetic", "provider-free", "classification", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION,
        ExpertClass.A,
        "bounded operation, state, and authority-surface references",
        "one closed authority-requirement label plus optional evidence references",
        ("candidate_authority_ids", "evidence_reference_ids"),
        RiskClass.R4,
        risks_through(RiskClass.R4, floor=RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        768,
        ("cpu-small-hermetic", "provider-free", "classification", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.CONTEXT_SUFFICIENCY,
        ExpertClass.A,
        "present context identities versus required reference identities",
        "boolean sufficiency, missing references, and one reason code",
        ("present_reference_ids", "required_reference_ids"),
        RiskClass.R2,
        risks_through(RiskClass.R2),
        PrivacyClass.INTERNAL,
        512,
        ("cpu-small-hermetic", "provider-free", "classification"),
        remote=True,
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.EVIDENCE_RANKING,
        ExpertClass.B,
        "bounded candidate evidence identities for one question",
        "aligned descending evidence ranking with scores in ppm",
        ("candidate_reference_ids",),
        RiskClass.R2,
        risks_through(RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-batch", "provider-free", "ranking", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.PROCEDURE_MATCHING,
        ExpertClass.C,
        "bounded procedure identities and precondition references",
        "one procedure identity, match class, and precondition references",
        ("procedure_ids", "precondition_reference_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-hermetic", "provider-free", "selection", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.PLAN_BRANCH_RANKING,
        ExpertClass.B,
        "bounded plan-branch identities under one parent goal",
        "aligned descending branch ranking with scores in ppm",
        ("candidate_reference_ids",),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-batch", "provider-free", "ranking", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.TEST_SELECTION,
        ExpertClass.C,
        "bounded candidate test identities and coverage references",
        "one non-empty closed test-id list plus optional coverage references",
        ("candidate_test_ids", "coverage_reference_ids"),
        RiskClass.R2,
        risks_through(RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-hermetic", "provider-free", "selection", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.PROOF_SELECTION,
        ExpertClass.C,
        "bounded proof identities and current prover-obligation references",
        "one non-empty closed proof-id list plus obligation references",
        ("candidate_proof_ids", "obligation_reference_ids"),
        RiskClass.R4,
        risks_through(RiskClass.R4, floor=RiskClass.R3),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-hermetic", "provider-free", "selection", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        ExpertClass.A,
        "validated failure signature plus bounded dependency references",
        "one failure class and one bounded action candidate",
        ("failure_signature_ids", "dependency_reference_ids"),
        RiskClass.R2,
        risks_through(RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        768,
        ("cpu-small-hermetic", "provider-free", "classification", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.RETRY_OR_ESCALATE,
        ExpertClass.A,
        "bounded attempt identities and prior failure-class references",
        "one closed retry, escalate, or stop decision plus a reason code",
        ("attempt_reference_ids", "failure_class_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.INTERNAL,
        512,
        ("cpu-small-hermetic", "provider-free", "classification"),
        remote=True,
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION,
        ExpertClass.A,
        "cache identity plus exact dependency references",
        "boolean reuse decision, dependency references, and a reason code",
        ("cache_identity_ids", "dependency_reference_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        512,
        ("cpu-small-hermetic", "provider-free", "classification", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION,
        ExpertClass.A,
        "bounded conflict symbols and merge-hunk references",
        "one conflict class plus symbol and evidence references",
        ("symbol_ids", "conflict_reference_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        768,
        ("cpu-small-hermetic", "provider-free", "classification", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.PATCH_TEMPLATE_SELECTION,
        ExpertClass.C,
        "bounded template identities and target symbol references",
        "one template identity plus symbol and evidence references",
        ("template_ids", "symbol_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-hermetic", "provider-free", "selection", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
        ExpertClass.D,
        "declared typed hole identities and compiler precondition references",
        "one hole identity, operator, and bounded argument/precondition references",
        ("hole_ids", "precondition_reference_ids"),
        RiskClass.R4,
        risks_through(RiskClass.R4, floor=RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        2048,
        ("cpu-small-hermetic", "provider-free", "structured-decoder", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        ExpertClass.D,
        "repository-relative paths, symbols, and allowed patch operations",
        "bounded PatchSketch files, symbols, operations, and line ceiling",
        ("file_path_ids", "symbol_ids", "operation_ids"),
        RiskClass.R4,
        risks_through(RiskClass.R4, floor=RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        4096,
        ("cpu-small-hermetic", "provider-free", "structured-decoder", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.LEMMA_SUGGESTION,
        ExpertClass.C,
        "current prover obligation and admissible premise identities",
        "ranked lemma identities bound to one obligation and premises",
        ("obligation_ids", "premise_ids"),
        RiskClass.R4,
        risks_through(RiskClass.R4, floor=RiskClass.R3),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-hermetic", "provider-free", "selection", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.TACTIC_SUGGESTION,
        ExpertClass.C,
        "current prover obligation and admissible premise identities",
        "ranked tactic identities bound to one obligation and premises",
        ("obligation_ids", "premise_ids"),
        RiskClass.R4,
        risks_through(RiskClass.R4, floor=RiskClass.R3),
        PrivacyClass.REPOSITORY_PRIVATE,
        1024,
        ("cpu-small-hermetic", "provider-free", "selection", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION,
        ExpertClass.D,
        "bounded counterexample identities and violated invariant references",
        "one failure class plus invariant and counterexample references",
        ("counterexample_reference_ids", "invariant_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        2048,
        ("cpu-small-hermetic", "provider-free", "structured-decoder", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE,
        ExpertClass.D,
        "parent goal identity and acceptance-reference identities",
        "parent goal, closed candidate goal kinds, and acceptance references",
        ("parent_goal_ids", "acceptance_reference_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R2),
        PrivacyClass.REPOSITORY_PRIVATE,
        2048,
        ("cpu-small-hermetic", "provider-free", "structured-decoder", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION,
        ExpertClass.A,
        "bounded documentation-claim identities and supporting evidence references",
        "one claim class, evidence references, and rewrite-required flag",
        ("claim_ids", "evidence_reference_ids"),
        RiskClass.R2,
        risks_through(RiskClass.R2),
        PrivacyClass.INTERNAL,
        768,
        ("cpu-small-hermetic", "provider-free", "classification"),
        remote=True,
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION,
        ExpertClass.A,
        "bounded evidence identities and escalation reason codes",
        "boolean escalate decision, reason code, and evidence references",
        ("evidence_reference_ids", "reason_code_ids"),
        RiskClass.R3,
        risks_through(RiskClass.R3, floor=RiskClass.R1),
        PrivacyClass.REPOSITORY_PRIVATE,
        512,
        ("cpu-small-hermetic", "provider-free", "classification", "local-only"),
    ),
    _FamilyBlueprint(
        ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
        ExpertClass.E,
        "no admitted compact feature may request novel unbounded generation",
        "unconditional ABSTAIN with a closed reason code and empty payload",
        (),
        RiskClass.R5,
        (RiskClass.R5,),
        PrivacyClass.REPOSITORY_PRIVATE,
        256,
        ("human-review", "provider-free", "always-abstain", "local-only"),
        always_abstain=True,
    ),
)


def _build_family_specs() -> dict[ResidualTaskFamily, ResidualTaskFamilySpec]:
    specs = tuple(_spec_from_blueprint(item) for item in _BLUEPRINTS)
    families = [item.task_family for item in specs]
    if set(families) != set(ResidualTaskFamily) or len(families) != len(set(families)):
        raise ResidualIntelligenceError("task family specs must cover the closed 24-family taxonomy")
    classes = {item.expert_class for item in specs}
    if classes != set(ExpertClass):
        raise ResidualIntelligenceError("task family specs must populate expert classes A through E")
    return {item.task_family: item for item in specs}


DEFAULT_TASK_FAMILY_SPECS: Final[Mapping[ResidualTaskFamily, ResidualTaskFamilySpec]] = (
    _build_family_specs()
)


def family_spec_for(task_family: ResidualTaskFamily | str) -> ResidualTaskFamilySpec:
    family = ResidualTaskFamily(task_family)
    try:
        return DEFAULT_TASK_FAMILY_SPECS[family]
    except KeyError as exc:
        raise ResidualIntelligenceError(f"missing task family spec for {family.value}") from exc


__all__ = (
    "ABSTENTION_ALWAYS",
    "ABSTENTION_SELECTIVE",
    "AUTHORITY_CANDIDATE_ONLY",
    "CLOSED_SCHEMA_SCHEMA",
    "DEFAULT_TASK_FAMILY_SPECS",
    "ERROR_CLASS_E_ABSTAIN",
    "ERROR_INVALID_OUTPUT",
    "PRIVACY_STRICTNESS",
    "PROPOSAL_RISKS",
    "RISK_ORDER",
    "TASK_FAMILY_SPEC_SCHEMA",
    "ClosedSchema",
    "ExpertClass",
    "ResidualTaskFamilySpec",
    "family_spec_for",
    "privacy_rank",
    "risk_rank",
    "risks_through",
)
