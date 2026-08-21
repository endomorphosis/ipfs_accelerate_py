"""Compact, candidate-only Residual Intelligence IR envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    bounded_int,
    bounded_json_mapping,
    canonical_id,
    reject_candidate_authority,
    reject_secret_material,
    required_text,
    strict_fields,
    text_tuple,
)

RESIDUAL_TASK_INPUT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-task-input@1"
RESIDUAL_TASK_OUTPUT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-task-output@1"
RESIDUAL_INTELLIGENCE_IR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-ir@1"
)
MAX_TOKEN_BUDGET: Final = 1_000_000
MAX_SCORE_PPM: Final = 1_000_000


@dataclass(frozen=True)
class ResidualTaskInput:
    """Bounded semantic capsule presented to one residual expert."""

    task_family: ResidualTaskFamily
    question_id: str
    repository_state_cid: str
    objective_cid: str
    task_cid: str
    policy_cid: str
    context_capsule_cid: str
    compact_features: Mapping[str, Any]
    allowed_outputs: tuple[str, ...]
    risk_class: RiskClass
    validation_policy: str
    token_budget: int
    schema: str = RESIDUAL_TASK_INPUT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "input_id",
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
        }
    )

    def __post_init__(self) -> None:
        if self.schema != RESIDUAL_TASK_INPUT_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual task input schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        for field in (
            "question_id",
            "repository_state_cid",
            "objective_cid",
            "task_cid",
            "policy_cid",
            "context_capsule_cid",
            "validation_policy",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        features = bounded_json_mapping(self.compact_features, "compact_features")
        reject_secret_material(features, noun="compact_features")
        object.__setattr__(self, "compact_features", features)
        object.__setattr__(
            self,
            "allowed_outputs",
            text_tuple(
                self.allowed_outputs,
                "allowed_outputs",
                allow_empty=False,
                max_items=256,
            ),
        )
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        object.__setattr__(
            self,
            "token_budget",
            bounded_int(
                self.token_budget,
                "token_budget",
                minimum=1,
                maximum=MAX_TOKEN_BUDGET,
            ),
        )

    @property
    def input_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "question_id": self.question_id,
            "repository_state_cid": self.repository_state_cid,
            "objective_cid": self.objective_cid,
            "task_cid": self.task_cid,
            "policy_cid": self.policy_cid,
            "context_capsule_cid": self.context_capsule_cid,
            "compact_features": dict(self.compact_features),
            "allowed_outputs": list(self.allowed_outputs),
            "risk_class": self.risk_class.value,
            "validation_policy": self.validation_policy,
            "token_budget": self.token_budget,
        }
        if include_id:
            result["input_id"] = self.input_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualTaskInput:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"input_id"},
            noun="residual task input",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            question_id=str(payload.get("question_id") or ""),
            repository_state_cid=str(payload.get("repository_state_cid") or ""),
            objective_cid=str(payload.get("objective_cid") or ""),
            task_cid=str(payload.get("task_cid") or ""),
            policy_cid=str(payload.get("policy_cid") or ""),
            context_capsule_cid=str(payload.get("context_capsule_cid") or ""),
            compact_features=payload.get("compact_features") or {},
            allowed_outputs=tuple(payload.get("allowed_outputs") or ()),
            risk_class=RiskClass(str(payload.get("risk_class") or "")),
            validation_policy=str(payload.get("validation_policy") or ""),
            token_budget=payload.get("token_budget"),
        )
        claimed = str(payload.get("input_id") or "")
        if claimed and claimed != result.input_id:
            raise ResidualIntelligenceError("residual task input identity mismatch")
        return result


@dataclass(frozen=True)
class ResidualTaskOutput:
    """Strict structured candidate returned by any learned expert.

    ``candidate_only`` cannot be set false.  Independent validation and the
    canonical authority engine remain outside this record.
    """

    output_class: str
    structured_payload: Mapping[str, Any]
    confidence_or_score: int
    calibration_group: str
    abstained: bool
    reason_codes: tuple[str, ...]
    evidence_references: tuple[str, ...]
    candidate_only: bool = True
    schema: str = RESIDUAL_TASK_OUTPUT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "output_id",
            "output_class",
            "structured_payload",
            "confidence_or_score",
            "calibration_group",
            "abstained",
            "reason_codes",
            "evidence_references",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != RESIDUAL_TASK_OUTPUT_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual task output schema")
        object.__setattr__(self, "output_class", required_text(self.output_class, "output_class"))
        payload = bounded_json_mapping(self.structured_payload, "structured_payload")
        reject_secret_material(payload, noun="structured_payload")
        reject_candidate_authority(payload)
        object.__setattr__(self, "structured_payload", payload)
        object.__setattr__(
            self,
            "confidence_or_score",
            bounded_int(
                self.confidence_or_score,
                "confidence_or_score",
                minimum=0,
                maximum=MAX_SCORE_PPM,
            ),
        )
        object.__setattr__(
            self,
            "calibration_group",
            required_text(self.calibration_group, "calibration_group"),
        )
        if type(self.abstained) is not bool:
            raise ResidualIntelligenceError("abstained must be boolean")
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", max_items=64),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=256),
        )
        if self.abstained and not self.reason_codes:
            raise ResidualIntelligenceError("an abstention requires at least one reason code")
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")

    @property
    def output_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "output_class": self.output_class,
            "structured_payload": dict(self.structured_payload),
            "confidence_or_score": self.confidence_or_score,
            "calibration_group": self.calibration_group,
            "abstained": self.abstained,
            "reason_codes": list(self.reason_codes),
            "evidence_references": list(self.evidence_references),
            "candidate_only": True,
        }
        if include_id:
            result["output_id"] = self.output_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualTaskOutput:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"output_id"},
            noun="residual task output",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            output_class=str(payload.get("output_class") or ""),
            structured_payload=payload.get("structured_payload") or {},
            confidence_or_score=payload.get("confidence_or_score"),
            calibration_group=str(payload.get("calibration_group") or ""),
            abstained=payload.get("abstained"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence_references=tuple(payload.get("evidence_references") or ()),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("output_id") or "")
        if claimed and claimed != result.output_id:
            raise ResidualIntelligenceError("residual task output identity mismatch")
        return result

    def validate_for(self, task_input: ResidualTaskInput) -> None:
        if self.output_class not in task_input.allowed_outputs:
            raise ResidualIntelligenceError(
                "residual output class is outside the input allowed_outputs"
            )
        if task_input.risk_class in {RiskClass.R4, RiskClass.R5} and not (
            self.abstained or "VALIDATION_REQUIRED" in self.reason_codes
        ):
            raise ResidualIntelligenceError(
                "R4/R5 learned outputs must abstain or remain explicitly validation-required"
            )


@dataclass(frozen=True)
class ResidualIntelligenceIR:
    """Input/output pair with exact semantic identities."""

    task_input: ResidualTaskInput
    task_output: ResidualTaskOutput
    schema: str = RESIDUAL_INTELLIGENCE_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDUAL_INTELLIGENCE_IR_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual intelligence IR schema")
        if not isinstance(self.task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
        if not isinstance(self.task_output, ResidualTaskOutput):
            raise ResidualIntelligenceError("task_output must be ResidualTaskOutput")
        self.task_output.validate_for(self.task_input)

    @property
    def ir_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_input": self.task_input.to_dict(),
            "task_output": self.task_output.to_dict(),
        }
        if include_id:
            result["ir_id"] = self.ir_id
        return result


__all__ = (
    "MAX_SCORE_PPM",
    "MAX_TOKEN_BUDGET",
    "RESIDUAL_INTELLIGENCE_IR_SCHEMA",
    "RESIDUAL_TASK_INPUT_SCHEMA",
    "RESIDUAL_TASK_OUTPUT_SCHEMA",
    "ResidualIntelligenceIR",
    "ResidualTaskInput",
    "ResidualTaskOutput",
)
